"""Week-4 hyper-parameter tuning for the click classifier, logged to MLflow.

Tunes the DEPLOYED model (`fistvsrest_frozen_mnv3_large`, AD-21) whose record
flags "more epochs / unfreeze_blocks / augmentation" as untried. Two arms, sized
to a CPU budget (see configs/tune_fistvsrest.yaml for the rationale):

  Arm A (`--stage a`)  TPE search over head/optimizer hyper-parameters, trained
                       on CACHED frozen-backbone features -- identical math to
                       the frozen end-to-end loop, ~1000x cheaper, so a real
                       60-trial Bayesian search is affordable. Includes a
                       seed-repeat re-check of the top-K (val is only 308 imgs).
  Arm B (`--stage b`)  the structural levers that invalidate the cache
                       (unfrozen blocks, augmentation): a small end-to-end grid.
  Final (`--stage final`)  take the best VAL config across both arms, evaluate
                       the test split ONCE (AD-10), save a self-describing
                       checkpoint, log + register the model in MLflow.

Every trial is one MLflow run (nested under a per-arm parent run). Selection is
on val only; nothing in stage a/b ever reads the test split.

    python -m src.tune --stage a     --config configs/tune_fistvsrest.yaml
    python -m src.tune --stage b
    python -m src.tune --stage final
    python -m src.tune --stage all               # a -> b -> final
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.data.config import DataConfig
from src.data.dataset import build_dataloaders
from src.models.build import build_model, param_summary, pooled_features
from src.train import eval_loader, set_seed

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# data versioning
# --------------------------------------------------------------------------- #
def dataset_fingerprint(manifest: Path) -> str:
    """Short content hash of the split manifest = the data version.

    The manifest is (uuid, label, user_id, split) for every image, so this hash
    changes if and only if the data or the split changes. Logged as an MLflow
    param on every run, which is what makes a run's numbers traceable to an
    exact dataset state (M4A1 §3, data versioning).
    """
    return hashlib.sha256(Path(manifest).read_bytes()).hexdigest()[:12]


def feature_cache_key(cfg: dict, data_cfg: DataConfig, fingerprint: str) -> str:
    """Cache identity: data version + everything that changes the features."""
    parts = [
        fingerprint,
        cfg["model"]["backbone"],
        str(cfg["model"]["pretrained"]),
        data_cfg.input.crop_mode,
        str(data_cfg.input.bbox_pad),
        ",".join(data_cfg.classes),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# feature cache (Arm A's fast path)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _forward_features(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    feats, labels = [], []
    n, t0 = 0, time.time()
    for x, y in loader:
        feats.append(pooled_features(model, x.to(device)).cpu())
        labels.append(y)
        n += len(y)
        print(f"    cached {n} imgs ({n / max(time.time() - t0, 1e-9):.0f}/s)", end="\r")
    print()
    return torch.cat(feats), torch.cat(labels)


def load_or_build_features(cfg, data_cfg, device, cache_dir: Path, key: str) -> dict:
    """{split: (features, labels)} for the frozen backbone, cached to disk.

    Valid only for the frozen + no-augmentation regime (deterministic features).
    Arm B never uses this.
    """
    cache = cache_dir / f"feats_{key}.pt"
    if cache.exists():
        print(f"feature cache hit -> {cache.name}")
        blob = torch.load(cache, map_location="cpu")
        return {k: (v["x"], v["y"]) for k, v in blob.items()}

    print(f"feature cache miss -> building {cache.name} (one frozen pass)")
    model = build_model(
        cfg["model"]["backbone"], cfg["model"]["num_classes"],
        freeze_backbone=True, pretrained=cfg["model"]["pretrained"], unfreeze_blocks=0,
    ).to(device)
    loaders, _, _ = build_dataloaders(
        data_cfg, backbone=cfg["model"]["backbone"],
        write_split_manifest=False, augment_train=False,
    )
    feats = {}
    for split in ("train", "val", "test"):
        print(f"  {split}:")
        feats[split] = _forward_features(model, loaders[split], device)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: {"x": v[0], "y": v[1]} for k, v in feats.items()}, cache)
    return feats


# --------------------------------------------------------------------------- #
# optimizers
# --------------------------------------------------------------------------- #
def make_optimizer(name: str, groups, lr: float, wd: float):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(groups, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"unknown optimizer: {name}")


# --------------------------------------------------------------------------- #
# trial result
# --------------------------------------------------------------------------- #
@dataclass
class TrialResult:
    name: str
    arm: str
    params: dict
    val_acc: float
    best_epoch: int
    train_acc: float
    val_loss: float
    seconds: float
    checkpoint: str | None = None
    history: list = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "run": self.name, "arm": self.arm, "val_acc": round(self.val_acc, 4),
            "best_epoch": self.best_epoch, "train_acc": round(self.train_acc, 4),
            "val_loss": round(self.val_loss, 4), "seconds": round(self.seconds, 1),
            **self.params,
        }


# --------------------------------------------------------------------------- #
# Arm A -- head training on cached features
# --------------------------------------------------------------------------- #
def train_head_cached(head_proto, feats, hp: dict, device, seed: int,
                      max_epochs: int, patience: int,
                      head_init: str = "pytorch_linear") -> TrialResult:
    """Train a fresh copy of the classifier head on cached features.

    Identical math to training the frozen full model end-to-end (the backbone is
    a fixed function), so an Arm-A val number is directly comparable to the
    deployed baseline. Best-by-val epoch is kept, with early stopping.

    `head_proto` is the head exactly as timm built it. `head_init` chooses whether
    to keep that init (`timm_default` -- reproduces the deployed run) or reset it
    to PyTorch's `Linear` default (`pytorch_linear`); see
    src/models/build.py::init_classifier for why that matters here.
    """
    set_seed(seed)
    head = copy.deepcopy(head_proto).to(device)
    if head_init == "pytorch_linear":
        for m in head.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
    elif head_init != "timm_default":
        raise ValueError(f"unknown head_init {head_init!r}")

    opt = make_optimizer(hp["optimizer"], head.parameters(), hp["lr"], hp["weight_decay"])
    lossf = nn.CrossEntropyLoss(label_smoothing=hp.get("label_smoothing", 0.0))

    Xtr, ytr = feats["train"]
    Xva, yva = feats["val"]
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=int(hp["batch_size"]), shuffle=True)

    best = {"val_acc": -1.0, "epoch": 0, "train_acc": 0.0, "val_loss": float("nan")}
    best_state, history, t0 = None, [], time.time()
    for epoch in range(1, max_epochs + 1):
        head.train()
        for xb, yb in loader:
            opt.zero_grad()
            lossf(head(xb.to(device)), yb.to(device)).backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            ltr = head(Xtr.to(device)).cpu()
            lva = head(Xva.to(device)).cpu()
        tr_acc = float((ltr.argmax(1) == ytr).float().mean())
        va_acc = float((lva.argmax(1) == yva).float().mean())
        va_loss = float(nn.functional.cross_entropy(lva, yva))
        history.append({"epoch": epoch, "train_acc": tr_acc, "val_acc": va_acc,
                        "val_loss": va_loss})
        if va_acc > best["val_acc"]:
            best = {"val_acc": va_acc, "epoch": epoch, "train_acc": tr_acc, "val_loss": va_loss}
            best_state = copy.deepcopy(head.state_dict())
        elif epoch - best["epoch"] >= patience:
            break

    head.load_state_dict(best_state)
    return TrialResult(
        name="", arm="A", params={**hp, "head_init": head_init}, val_acc=best["val_acc"],
        best_epoch=best["epoch"], train_acc=best["train_acc"], val_loss=best["val_loss"],
        seconds=time.time() - t0, history=history,
    ), head


# --------------------------------------------------------------------------- #
# Arm B -- end-to-end structural runs
# --------------------------------------------------------------------------- #
def train_end_to_end(cfg, data_cfg, run_spec: dict, hp: dict, device,
                     epochs: int, backbone_lr_mult: float, seed: int,
                     head_init: str = "pytorch_linear") -> tuple[TrialResult, nn.Module]:
    """Standard fwd/bwd loop with `unfreeze_blocks` backbone stages trainable and
    optional augmentation. Unfrozen backbone params get lr * backbone_lr_mult
    (the usual fine-tuning convention -- NOT swept, see the config's note).

    Uses the same fixed head init as Arm A: with unfrozen blocks, timm's
    +/-27-logit init would push large gradients straight into the pretrained
    features on step 1, which is the opposite of what fine-tuning wants."""
    set_seed(seed)
    mcfg = cfg["model"]
    model = build_model(
        mcfg["backbone"], mcfg["num_classes"], freeze_backbone=True,
        pretrained=mcfg["pretrained"], unfreeze_blocks=int(run_spec["unfreeze_blocks"]),
        head_init=head_init,
    ).to(device)
    ps = param_summary(model)
    print(f"  trainable {ps['trainable']:,}/{ps['total']:,} ({ps['trainable_pct']:.2f}%)")

    head_ids = {id(p) for p in model.get_classifier().parameters()}
    head_params = [p for p in model.get_classifier().parameters() if p.requires_grad]
    bb_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": head_params, "lr": hp["lr"]}]
    if bb_params:
        groups.append({"params": bb_params, "lr": hp["lr"] * backbone_lr_mult})

    opt = make_optimizer(hp["optimizer"], groups, hp["lr"], hp["weight_decay"])
    lossf = nn.CrossEntropyLoss(label_smoothing=hp.get("label_smoothing", 0.0))

    loaders, _, _ = build_dataloaders(
        data_cfg, backbone=mcfg["backbone"], write_split_manifest=False,
        augment_train=bool(run_spec["augment"]),
    )

    best = {"val_acc": -1.0, "epoch": 0, "val_loss": float("nan")}
    best_state, history, t0 = None, [], time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        seen = 0
        for x, y in loaders["train"]:
            opt.zero_grad()
            lossf(model(x.to(device)), y.to(device)).backward()
            opt.step()
            seen += len(y)
            print(f"    epoch {epoch}: {seen}/{len(loaders['train'].dataset)}", end="\r")
        va_loss, va_acc, _, _ = eval_loader(model, loaders["val"], device)
        history.append({"epoch": epoch, "train_acc": float("nan"), "val_acc": va_acc,
                        "val_loss": va_loss})
        print(f"    epoch {epoch:3d}  val_loss {va_loss:.4f}  val_acc {va_acc:.4f}"
              f"  [{time.time() - t0:.0f}s]")
        if va_acc > best["val_acc"]:
            best = {"val_acc": va_acc, "epoch": epoch, "val_loss": va_loss}
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return TrialResult(
        name=run_spec["name"], arm="B",
        params={**hp, "unfreeze_blocks": run_spec["unfreeze_blocks"],
                "augment_train": bool(run_spec["augment"]), "epochs": epochs,
                "backbone_lr_mult": backbone_lr_mult, "head_init": head_init,
                **{f"aug_{k}": v for k, v in (run_spec.get("augment_overrides") or {}).items()}},
        val_acc=best["val_acc"], best_epoch=best["epoch"], train_acc=float("nan"),
        val_loss=best["val_loss"], seconds=time.time() - t0, history=history,
    ), model


# --------------------------------------------------------------------------- #
# checkpoint I/O (same format as src/train.py -> loadable by src/rt/model_loader)
# --------------------------------------------------------------------------- #
def save_checkpoint(path: Path, model, data_cfg, size, mean, std,
                    val_acc: float, test_acc: float | None, meta: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "backbone": meta["backbone"],
            "classes": data_cfg.classes,
            "input": {"size": size, "mean": mean, "std": std,
                      "crop_mode": data_cfg.input.crop_mode},
            "val_acc": val_acc,
            "test_acc": test_acc,
            "tuning": meta,
        },
        path,
    )
    return path


# --------------------------------------------------------------------------- #
# MLflow helpers
# --------------------------------------------------------------------------- #
def mlflow_init(cfg) -> None:
    mlflow.set_tracking_uri((REPO / "mlruns").as_uri())
    mlflow.set_experiment(cfg["mlflow"]["experiment"])


def log_trial(res: TrialResult, common: dict, tags: dict) -> None:
    with mlflow.start_run(run_name=res.name, nested=True):
        mlflow.set_tags({"arm": res.arm, **tags})
        mlflow.log_params({**common, **res.params})
        mlflow.log_metrics({
            "val_accuracy": res.val_acc, "val_loss": res.val_loss,
            "train_accuracy_at_best": res.train_acc, "best_epoch": res.best_epoch,
            "seconds": res.seconds,
        })
        for h in res.history:  # the learning curve, per epoch
            mlflow.log_metrics({"epoch_val_accuracy": h["val_acc"],
                                "epoch_val_loss": h["val_loss"]}, step=h["epoch"])


def state_path(out_dir: Path, name: str) -> Path:
    return out_dir / "tuning" / name


def write_state(out_dir: Path, name: str, payload) -> Path:
    p = state_path(out_dir, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {p}")
    return p


def read_state(out_dir: Path, name: str):
    p = state_path(out_dir, name)
    if not p.exists():
        raise FileNotFoundError(f"missing {p} -- run the earlier --stage first")
    return json.loads(p.read_text())


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def stage_a(cfg, data_cfg, device, common) -> list[dict]:
    """TPE search on cached features + seed-repeat of the top-K."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    scfg = cfg["search"]
    out_dir = REPO / cfg["out_dir"]
    feats = load_or_build_features(cfg, data_cfg, device, out_dir / "cache", common["feature_cache_key"])
    print("cached feature shapes:", {k: tuple(v[0].shape) for k, v in feats.items()})

    # The head prototype must be the head timm ACTUALLY hands us in the real
    # pipeline, drawn from the same RNG state src/train.py would have (seed, then
    # create_model) -- that is what makes the `timm_init` baseline an exact
    # reproduction rather than just a same-distribution redraw.
    set_seed(scfg["seed"])
    proto = build_model(cfg["model"]["backbone"], cfg["model"]["num_classes"],
                        freeze_backbone=True,
                        pretrained=cfg["model"]["pretrained"]).get_classifier()

    space = scfg["space"]

    def suggest(trial) -> dict:
        hp = {}
        for name, spec in space.items():
            t = spec["type"]
            if t == "loguniform":
                hp[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
            elif t == "uniform":
                hp[name] = trial.suggest_float(name, spec["low"], spec["high"])
            elif t == "categorical":
                hp[name] = trial.suggest_categorical(name, spec["choices"])
            else:
                raise ValueError(f"unknown space type {t} for {name}")
        return hp

    # --- pre-tuning references, run in THIS harness ------------------------- #
    # (1) the deployed recipe verbatim, timm's head init -> must reproduce the
    #     committed 0.9221, which is what licenses every comparison below;
    # (2) the same recipe with the head init fixed -> isolates how much of the
    #     headroom is the init alone, BEFORE any hyper-parameter search.
    bl = cfg["baseline"]
    baselines = {}
    for tag, init in (("timm_init", "timm_default"), ("fixed_init", "pytorch_linear")):
        res, _ = train_head_cached(proto, feats, bl["params"], device, scfg["seed"],
                                   bl["epochs"], patience=bl["epochs"],  # no early stop
                                   head_init=init)
        res.name = f"{bl['name']}_{tag}"
        res.arm = "baseline"
        baselines[tag] = res
        print(f"baseline {tag:<10} val {res.val_acc:.4f} @ep{res.best_epoch:3d} "
              f"(init={init})")

    base_res = baselines["timm_init"]
    drift = abs(base_res.val_acc - bl["expect_val_acc"])
    print(f"drift check: deployed run recorded {bl['expect_val_acc']:.4f}, "
          f"harness reproduces {base_res.val_acc:.4f} (delta {drift:+.4f})")
    if drift > bl["tolerance"]:
        print("  *** HARNESS DRIFT WARNING: this harness does not reproduce the "
              "committed baseline. Any 'post-tuning' gain below is NOT trustworthy "
              "until this is explained. ***")

    results: list[TrialResult] = []
    sampler = (optuna.samplers.TPESampler(seed=scfg["seed"]) if scfg["sampler"] == "tpe"
               else optuna.samplers.RandomSampler(seed=scfg["seed"]))
    study = optuna.create_study(direction="maximize", sampler=sampler)

    with mlflow.start_run(run_name="armA_head_search"):
        for tag, res in baselines.items():
            log_trial(res, common, {"stage": "baseline_reference", "head_init_ablation": tag,
                                    "note": "deployed recipe, re-run in the tuner harness"})
        mlflow.set_tags({"arm": "A", "stage": "tuning",
                         "search": f"{scfg['sampler']} x{scfg['trials']}",
                         "regime": "frozen backbone, head-only (cached features)"})
        mlflow.log_params({**common, "sampler": scfg["sampler"], "trials": scfg["trials"],
                           "max_epochs": scfg["max_epochs"], "patience": scfg["patience"]})

        def objective(trial):
            hp = suggest(trial)
            res, _ = train_head_cached(proto, feats, hp, device, scfg["seed"],
                                       scfg["max_epochs"], scfg["patience"],
                                       head_init=scfg["head_init"])
            res.name = f"A{trial.number:03d}"
            print(f"  {res.name}  val {res.val_acc:.4f} @ep{res.best_epoch:3d}  "
                  f"({res.seconds:.0f}s)  {hp}")
            log_trial(res, common, {"stage": "tuning", "trial": str(trial.number)})
            results.append(res)
            return res.val_acc

        study.optimize(objective, n_trials=scfg["trials"])
        best = max(results, key=lambda r: r.val_acc)
        mlflow.log_metrics({"best_val_accuracy": best.val_acc,
                            "trials_completed": len(results)})
        print(f"\nArm A best: {best.name} val {best.val_acc:.4f}  {best.params}")

        # --- seed repeat: is the top-K ranking real, or val noise? ------------
        rcfg = cfg["seed_repeat"]
        top = sorted(results, key=lambda r: -r.val_acc)[: rcfg["top_k"]]
        repeats = []
        for rank, cand in enumerate(top):
            accs = []
            for seed in rcfg["seeds"]:
                r, _ = train_head_cached(proto, feats, cand.params, device, seed,
                                         scfg["max_epochs"], scfg["patience"],
                                         head_init=scfg["head_init"])
                r.name = f"{cand.name}_seed{seed}"
                r.arm = "A-repeat"
                log_trial(r, common, {"stage": "seed_repeat", "of": cand.name,
                                      "seed_repeat": str(seed)})
                accs.append(r.val_acc)
                print(f"  repeat {r.name}: val {r.val_acc:.4f}")
            rec = {"of": cand.name, "rank": rank + 1, "params": cand.params,
                   "seeds": rcfg["seeds"], "val_accs": [round(a, 4) for a in accs],
                   "mean": float(np.mean(accs)), "std": float(np.std(accs))}
            repeats.append(rec)
            print(f"  -> {cand.name}: mean {rec['mean']:.4f} +/- {rec['std']:.4f}")
        mlflow.log_dict({"repeats": repeats}, "seed_repeat.json")

    # Selection uses the seed-MEAN of the top-K (a single-seed win inside the
    # noise band is not a win). Ranking + raw trials are both persisted.
    winner = max(repeats, key=lambda r: r["mean"])
    write_state(REPO / cfg["out_dir"], "arm_a.json", {
        "baseline": {
            "expect_val_acc": bl["expect_val_acc"], "drift": round(drift, 4),
            "timm_init": {**baselines["timm_init"].as_row(),
                          "history": baselines["timm_init"].history},
            "fixed_init": {**baselines["fixed_init"].as_row(),
                           "history": baselines["fixed_init"].history},
        },
        "trials": [r.as_row() for r in results],
        "seed_repeat": repeats,
        "best": {"name": winner["of"], "params": winner["params"],
                 "val_acc_mean": winner["mean"], "val_acc_std": winner["std"],
                 "val_acc_single_seed": max(r.val_acc for r in results
                                            if r.name == winner["of"])},
    })
    return [r.as_row() for r in results]


def stage_b(cfg, data_cfg, device, common) -> list[dict]:
    """End-to-end structural grid, seeded with Arm A's winning optimizer HPs."""
    out_dir = REPO / cfg["out_dir"]
    arm_a = read_state(out_dir, "arm_a.json")
    hp = dict(arm_a["best"]["params"])
    print(f"Arm B uses Arm A's winning optimizer HPs: {hp}")

    bcfg = cfg["structural"]
    rows, records = [], []
    with mlflow.start_run(run_name="armB_structural_grid"):
        mlflow.set_tags({"arm": "B", "stage": "tuning",
                         "regime": "end-to-end (unfreeze / augmentation)"})
        mlflow.log_params({**common, "epochs": bcfg["epochs"],
                           "backbone_lr_mult": bcfg["backbone_lr_mult"],
                           "grid": ",".join(r["name"] for r in bcfg["runs"])})
        for spec in bcfg["runs"]:
            print(f"\n--- Arm B: {spec['name']} "
                  f"(unfreeze={spec['unfreeze_blocks']}, aug={spec['augment']}) ---")
            # A per-run copy of the data config so augment_overrides can't leak
            # between runs.
            dcfg = copy.deepcopy(data_cfg)
            for k, v in (spec.get("augment_overrides") or {}).items():
                setattr(dcfg.augment, k, v)
            res, model = train_end_to_end(cfg, dcfg, spec, hp, device, bcfg["epochs"],
                                          bcfg["backbone_lr_mult"], cfg["search"]["seed"],
                                          head_init=cfg["search"]["head_init"])
            size, mean, std = _resolve_input(cfg)
            ckpt = save_checkpoint(
                out_dir / f"struct_{spec['name']}" / "best.pt", model, dcfg, size, mean, std,
                res.val_acc, None,
                {"backbone": cfg["model"]["backbone"], "arm": "B", "run": spec["name"],
                 "params": res.params, "data_version": common["data_version"]},
            )
            res.checkpoint = str(ckpt.relative_to(REPO)).replace("\\", "/")
            log_trial(res, common, {"stage": "tuning", "checkpoint": res.checkpoint})
            print(f"  {spec['name']}: val {res.val_acc:.4f} @ep{res.best_epoch} "
                  f"({res.seconds/60:.1f} min) -> {res.checkpoint}")
            rows.append(res.as_row())
            records.append({**res.as_row(), "checkpoint": res.checkpoint})
        mlflow.log_metrics({"best_val_accuracy": max(r["val_acc"] for r in rows)})

    write_state(out_dir, "arm_b.json", {"runs": records})
    return rows


def _resolve_input(cfg) -> tuple[int, tuple, tuple]:
    from src.data.transforms import resolve_backbone_config
    return resolve_backbone_config(cfg["model"]["backbone"])


def stage_final(cfg, data_cfg, device, common) -> dict:
    """Pick the best VAL config across both arms, evaluate test ONCE, register."""
    out_dir = REPO / cfg["out_dir"]
    arm_a = read_state(out_dir, "arm_a.json")
    candidates = [{"arm": "A", "name": arm_a["best"]["name"],
                   "val_acc": arm_a["best"]["val_acc_mean"], "params": arm_a["best"]["params"]}]
    try:
        arm_b = read_state(out_dir, "arm_b.json")
        candidates += [{"arm": "B", "name": r["run"], "val_acc": r["val_acc"],
                        "params": {k: v for k, v in r.items()
                                   if k not in {"run", "arm", "val_acc", "best_epoch",
                                                "train_acc", "val_loss", "seconds",
                                                "checkpoint"}},
                        "checkpoint": r["checkpoint"]} for r in arm_b["runs"]]
    except FileNotFoundError:
        print("no arm_b.json -- finalizing from Arm A only")

    for c in candidates:
        print(f"  candidate {c['arm']}/{c['name']}: val {c['val_acc']:.4f}")
    winner = max(candidates, key=lambda c: c["val_acc"])
    print(f"\nWINNER (by val): {winner['arm']}/{winner['name']} val {winner['val_acc']:.4f}")

    size, mean, std = _resolve_input(cfg)
    mcfg = cfg["model"]

    if winner["arm"] == "A":
        # Cheap to reproduce exactly: retrain the head on cached features.
        feats = load_or_build_features(cfg, data_cfg, device, out_dir / "cache",
                                       common["feature_cache_key"])
        set_seed(cfg["search"]["seed"])  # same RNG state as stage_a's proto
        proto = build_model(mcfg["backbone"], mcfg["num_classes"], True,
                            mcfg["pretrained"]).get_classifier()
        head_init = winner["params"].get("head_init", cfg["search"]["head_init"])
        res, head = train_head_cached(proto, feats, winner["params"], device,
                                      cfg["search"]["seed"], cfg["search"]["max_epochs"],
                                      cfg["search"]["patience"], head_init=head_init)
        model = build_model(mcfg["backbone"], mcfg["num_classes"], True,
                            mcfg["pretrained"], unfreeze_blocks=0).to(device)
        model.get_classifier().load_state_dict(head.state_dict())
        model.eval()
        # Test-set forward through the SAME cached features (identical math).
        with torch.no_grad():
            logits = head(feats["test"][0].to(device)).cpu()
        y_true = feats["test"][1].numpy()
        y_pred = logits.argmax(1).numpy()
        probs = torch.softmax(logits, dim=1).numpy()
        val_acc, best_epoch = res.val_acc, res.best_epoch
    else:
        from src.rt.model_loader import load_checkpoint
        lm = load_checkpoint(REPO / winner["checkpoint"], device=device)
        model = lm.model
        loaders, _, _ = build_dataloaders(data_cfg, backbone=mcfg["backbone"],
                                          write_split_manifest=False, augment_train=False)
        _, _, y_true, y_pred = eval_loader(model, loaders["test"], device)
        probs = None
        val_acc, best_epoch = winner["val_acc"], -1

    test_acc = float((y_true == y_pred).mean())
    print(f"\nFINAL  val {val_acc:.4f}  |  TEST {test_acc:.4f}  (reported once, AD-10)")

    ckpt = save_checkpoint(
        out_dir / "best.pt", model, data_cfg, size, mean, std, val_acc, test_acc,
        {"backbone": mcfg["backbone"], "arm": winner["arm"], "run": winner["name"],
         "params": winner["params"], "data_version": common["data_version"],
         "selected_by": "val accuracy (test read once, after selection)"},
    )
    (out_dir / "config.snapshot.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    np.savez(out_dir / "test_predictions.npz", y_true=y_true, y_pred=y_pred,
             probs=(probs if probs is not None else np.zeros((0, 0))))
    print(f"saved -> {ckpt}")

    # --- register in MLflow -------------------------------------------------- #
    from src.serve.pyfunc_model import log_and_register
    with mlflow.start_run(run_name=f"final_{winner['arm']}_{winner['name']}") as run:
        mlflow.set_tags({"arm": winner["arm"], "stage": "final",
                         "role": "registered click classifier (Week-4 tuned)"})
        mlflow.log_params({**common, **{k: str(v) for k, v in winner["params"].items()},
                           "selected_from": f"{winner['arm']}/{winner['name']}",
                           "best_epoch": best_epoch})
        mlflow.log_metrics({"val_accuracy": val_acc, "test_accuracy": test_acc,
                            "random_baseline": 1.0 / len(data_cfg.classes)})
        version = log_and_register(
            checkpoint=ckpt, classes=data_cfg.classes,
            registered_model_name=cfg["mlflow"]["registered_model"],
        )
        run_id = run.info.run_id
    print(f"registered {cfg['mlflow']['registered_model']} version {version} (run {run_id})")

    summary = {"winner": winner, "val_acc": val_acc, "test_acc": test_acc,
               "checkpoint": str(ckpt.relative_to(REPO)).replace("\\", "/"),
               "registered_model": cfg["mlflow"]["registered_model"],
               "model_version": version, "run_id": run_id,
               "data_version": common["data_version"],
               "candidates": [{k: v for k, v in c.items() if k != "params"} for c in candidates]}
    write_state(out_dir, "final.json", summary)
    return summary


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/tune_fistvsrest.yaml")
    ap.add_argument("--stage", choices=["a", "b", "final", "all"], default="all")
    ap.add_argument("--trials", type=int, default=0, help="override search.trials")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / args.config).read_text())
    if args.trials:
        cfg["search"]["trials"] = args.trials
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = DataConfig.from_yaml(REPO / cfg["data_config"])
    data_cfg.loader.num_workers = 0   # CPU box: in-process decode beats IPC (see configs)

    # The manifest is the data version; write it if it isn't there yet.
    manifest = REPO / data_cfg.manifest
    if not manifest.exists():
        print(f"writing split manifest -> {manifest}")
        from src.data.dataset import build_datasets
        build_datasets(data_cfg, backbone=cfg["model"]["backbone"],
                       write_split_manifest=True, augment_train=False)

    fingerprint = dataset_fingerprint(manifest)
    common = {
        "backbone": cfg["model"]["backbone"], "num_classes": cfg["model"]["num_classes"],
        "pretrained": cfg["model"]["pretrained"], "classes": ",".join(data_cfg.classes),
        "crop_mode": data_cfg.input.crop_mode, "bbox_pad": data_cfg.input.bbox_pad,
        "input_size": data_cfg.input.size,
        "split_scheme": "70/15/15 by user_id (grouped, stratified)",
        "split_seed": data_cfg.split.seed, "balance": data_cfg.balance,
        "data_version": fingerprint, "device": device,
        "feature_cache_key": feature_cache_key(cfg, data_cfg, fingerprint),
    }
    print(f"device={device}  data_version={fingerprint}  classes={data_cfg.classes}")

    mlflow_init(cfg)
    t0 = time.time()
    if args.stage in ("a", "all"):
        stage_a(cfg, data_cfg, device, common)
    if args.stage in ("b", "all"):
        stage_b(cfg, data_cfg, device, common)
    if args.stage in ("final", "all"):
        stage_final(cfg, data_cfg, device, common)
    print(f"\nstage(s) '{args.stage}' done in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()

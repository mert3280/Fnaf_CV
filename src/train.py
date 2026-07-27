"""Config-driven training for the gesture classifier.

Two paths, chosen by `train.precompute_features`:
  * linear-probe (frozen backbone + no aug): cache the backbone's pooled
    features once, then train the head on cached tensors -- identical math to
    the frozen-backbone loop, dramatically faster (esp. on CPU).
  * standard end-to-end loop: forward -> CrossEntropy -> backward -> step; the
    path used once layers are unfrozen or augmentation is on (Phase 3).

Saves the best-by-val-accuracy checkpoint (the FULL model: backbone + trained
head) plus its resolved config, and writes an honest metrics report
(val + held-out test, per-class P/R/F1, confusion matrix) to DOCS/results.md.

    python -m src.train --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from src.data.config import DataConfig
from src.data.dataset import build_dataloaders
from src.models.build import build_model, get_data_config, param_summary, pooled_features


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_optimizer(name: str, params, lr: float, wd: float):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"unknown optimizer: {name}")


@torch.no_grad()
def cache_features(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """One forward pass through the frozen backbone -> (features, labels)."""
    model.eval()
    feats, labels = [], []
    n, t0 = 0, time.time()
    for x, y in loader:
        feats.append(pooled_features(model, x.to(device)).cpu())
        labels.append(y)
        n += len(y)
        print(f"    cached {n} imgs ({n / (time.time() - t0):.0f}/s)", end="\r")
    print()
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def evaluate_logits(logits: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    loss = nn.functional.cross_entropy(logits, y).item()
    acc = (logits.argmax(1) == y).float().mean().item()
    return loss, acc


# --------------------------------------------------------------------------- #
# training paths
# --------------------------------------------------------------------------- #
def train_linear_probe(model, feats, tcfg, device):
    """Train only the classifier head on cached features. Returns best head
    state (by val acc) and the per-epoch history."""
    head = model.get_classifier().to(device)
    opt = make_optimizer(tcfg["optimizer"], head.parameters(), tcfg["lr"], tcfg["weight_decay"])
    Xtr, ytr = feats["train"]
    Xva, yva = feats["val"]
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=tcfg["batch_size"], shuffle=True)

    best_acc, best_state, history = -1.0, None, []
    for epoch in range(1, tcfg["epochs"] + 1):
        head.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = nn.functional.cross_entropy(head(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            tr_loss, tr_acc = evaluate_logits(head(Xtr.to(device)).cpu(), ytr)
            va_loss, va_acc = evaluate_logits(head(Xva.to(device)).cpu(), yva)
        history.append((epoch, tr_loss, tr_acc, va_loss, va_acc))
        if va_acc > best_acc:
            best_acc, best_state = va_acc, copy.deepcopy(head.state_dict())
        print(f"  epoch {epoch:3d}  train_loss {tr_loss:.4f} acc {tr_acc:.4f}"
              f"  |  val_loss {va_loss:.4f} acc {va_acc:.4f}")
    head.load_state_dict(best_state)  # leave the model at its best head
    return best_acc, history


def train_standard(model, loaders, tcfg, device):
    """Standard end-to-end loop (used when unfrozen / augmented)."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(tcfg["optimizer"], params, tcfg["lr"], tcfg["weight_decay"])
    best_acc, best_state, history = -1.0, None, []
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        for x, y in loaders["train"]:
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
        va_loss, va_acc, _, _ = eval_loader(model, loaders["val"], device)
        history.append((epoch, float("nan"), float("nan"), va_loss, va_acc))
        if va_acc > best_acc:
            best_acc, best_state = va_acc, copy.deepcopy(model.state_dict())
        print(f"  epoch {epoch:3d}  val_loss {va_loss:.4f} acc {va_acc:.4f}")
    model.load_state_dict(best_state)
    return best_acc, history


@torch.no_grad()
def eval_loader(model, loader, device):
    """Full-model evaluation over a DataLoader -> (loss, acc, y_true, y_pred)."""
    model.eval()
    losses, ys, ps = [], [], []
    for x, y in loader:
        logits = model(x.to(device)).cpu()
        losses.append(nn.functional.cross_entropy(logits, y, reduction="sum").item())
        ys.append(y)
        ps.append(logits.argmax(1))
    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(ps).numpy()
    loss = sum(losses) / len(y_true)
    acc = float((y_true == y_pred).mean())
    return loss, acc, y_true, y_pred


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def write_report(path, cfg_model, val_acc, test_acc, y_true, y_pred, classes, extra):
    rep = classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    header = "| true\\pred | " + " | ".join(classes) + " |"
    sep = "|" + "---|" * (len(classes) + 1)
    rows = [f"| **{classes[i]}** | " + " | ".join(str(int(v)) for v in cm[i]) + " |"
            for i in range(len(classes))]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        f"# Results\n\n"
        f"_Auto-generated by `src/train.py`. {extra}_\n\n"
        f"## Baseline — {cfg_model['backbone']} (frozen backbone, head-only)\n\n"
        f"- **Backbone:** `{cfg_model['backbone']}` (pretrained, frozen)\n"
        f"- **Best val accuracy:** {val_acc:.4f}\n"
        f"- **Held-out test accuracy:** {test_acc:.4f}  (random = {1/len(classes):.4f})\n\n"
        f"### Per-class report (test)\n\n```\n{rep}\n```\n\n"
        f"### Confusion matrix (test, rows = true)\n\n{header}\n{sep}\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote report -> {path}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    mcfg, tcfg = cfg["model"], cfg["train"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(torch.get_num_threads())
    set_seed(tcfg["seed"])
    print(f"device={device}  backbone={mcfg['backbone']}  frozen={mcfg['freeze_backbone']}"
          f"  head_init={mcfg.get('head_init', 'timm_default')}")

    data_cfg = DataConfig.from_yaml(cfg["data_config"])
    data_cfg.loader.batch_size = tcfg["batch_size"]
    data_cfg.loader.num_workers = tcfg["num_workers"]

    unfreeze_blocks = mcfg.get("unfreeze_blocks", 0)  # AD-08 Stage B (default 0 = frozen)
    # head_init defaults to timm's own init, so omitting it reproduces every
    # previously committed run exactly. `pytorch_linear` is the Week-4 finding
    # (see src/models/build.py::init_classifier) -- opt in per config.
    head_init = mcfg.get("head_init", "timm_default")
    model = build_model(
        mcfg["backbone"], mcfg["num_classes"], mcfg["freeze_backbone"], mcfg["pretrained"],
        unfreeze_blocks=unfreeze_blocks, head_init=head_init,
    ).to(device)
    ps = param_summary(model)
    print(f"params: trainable {ps['trainable']:,} / total {ps['total']:,} "
          f"({ps['trainable_pct']:.2f}% trainable)")

    loaders, splits, (size, mean, std) = build_dataloaders(
        data_cfg, backbone=mcfg["backbone"], write_split_manifest=True,
        augment_train=tcfg["augment_train"],
    )
    print("split sizes:", {k: len(v.dataset) for k, v in loaders.items()})

    # Feature caching is only valid when the backbone is fully frozen and
    # deterministic -- any unfrozen blocks or augmentation force the standard loop.
    use_cache = (
        tcfg.get("precompute_features")
        and mcfg["freeze_backbone"]
        and unfreeze_blocks == 0
        and not tcfg["augment_train"]
    )
    if use_cache:
        print("caching frozen-backbone features (one pass)...")
        feats = {k: cache_features(model, loaders[k], device) for k in ("train", "val", "test")}
        val_acc, _ = train_linear_probe(model, feats, tcfg, device)
        # test eval from cached features through the trained head
        head = model.get_classifier().eval()
        with torch.no_grad():
            logits = head(feats["test"][0].to(device)).cpu()
        y_true = feats["test"][1].numpy()
        y_pred = logits.argmax(1).numpy()
        test_acc = float((y_true == y_pred).mean())
    else:
        val_acc, _ = train_standard(model, loaders, tcfg, device)
        _, test_acc, y_true, y_pred = eval_loader(model, loaders["test"], device)

    print(f"\nBEST val acc {val_acc:.4f}  |  TEST acc {test_acc:.4f}")

    # save checkpoint (full model) + resolved config
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "backbone": mcfg["backbone"],
            "classes": data_cfg.classes,
            "input": {"size": size, "mean": mean, "std": std, "crop_mode": data_cfg.input.crop_mode},
            "val_acc": val_acc,
            "test_acc": test_acc,
        },
        out / "best.pt",
    )
    (out / "config.snapshot.json").write_text(json.dumps(cfg, indent=2))
    print(f"saved checkpoint -> {out/'best.pt'}")

    write_report(
        "DOCS/results.md", mcfg, val_acc, test_acc, y_true, y_pred, data_cfg.classes,
        extra=f"device={device}, input={size}px, {'linear-probe on cached features' if use_cache else 'end-to-end loop'}",
    )


if __name__ == "__main__":
    main()

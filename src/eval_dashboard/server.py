"""The evaluation dashboard web app.

Serves a single-page obstacle course and owns the hand-tracking loop. Launch it,
open the page, and drive the whole thing with your hand:

    python -m src.eval_dashboard.server --checkpoint models/palmfist_frozen_mnv3_large/best.pt

The tracker moves the REAL cursor and fires REAL clicks into the browser, so the
three obstacles (precision, rapid-click, tracer) measure the actual controller.
Hitting *Finish* in the page calls `/api/finish`, which stops the tracker
(kill-switch) so the 5-question survey can be answered with the normal mouse.
`/api/results` then writes the run to a doc file (see `results.py`).

Flags mirror `src.control.play` so a tuned run carries over. Two extra modes for
setup/testing:
  --dry-run     : run the full pipeline but send NO real input (needs a camera+model)
  --no-tracker  : don't start the tracker at all; test the UI with the normal mouse

ESC is the global kill-switch while the tracker runs (CLAUDE.md).
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from src.control.strategies import DEFAULT_STRATEGY, FSM_PRESETS
from src.eval_dashboard.model_registry import list_models
from src.eval_dashboard.results import save_results, summarize_run
from src.eval_dashboard.tracker import HandTracker, TrackerConfig

STATIC_DIR = Path(__file__).parent / "static"

app = Flask(__name__, static_folder=None)
_tracker: HandTracker | None = None      # None in --no-tracker mode
_results_dir = "eval_results"
_run_meta: dict = {}                      # model/strategy stamped onto each saved run
_strategy_applicable = False             # do the 3.2 / 3.2.1 presets fit this checkpoint?
_current_strategy: str | None = None     # which preset the next run will use
_available_models: list = []             # checkpoints on disk + their DOCS metadata
_current_checkpoint: str | None = None   # checkpoint the next run will start with
_device = "cpu"


def _derive_meta(checkpoint: str, device: str, tracker: bool) -> dict:
    """Label a run with the model + strategy it exercised, for the Overview tab."""
    if not tracker:
        return {"model": "mouse (no tracker)", "strategy": "— UI test", "device": device}
    name = Path(checkpoint).parent.name or Path(checkpoint).stem
    low = name.lower()
    if "fistvsrest" in low:
        strategy = "3.2 · fist-vs-rest"
    elif "palmfist" in low:
        strategy = "3.0 · palm/fist"
    else:
        strategy = "3 · cursor & click"
    return {"model": name, "checkpoint": str(checkpoint),
            "strategy": strategy, "device": device}


def _strategy_fits(checkpoint: str, tracker: bool) -> bool:
    """The 3.2 / 3.2.1 presets are the fist-vs-rest FSM tunings (AD-21), so only
    offer the picker when a fist-vs-rest checkpoint is loaded."""
    if not tracker:
        return False
    return "fistvsrest" in (Path(checkpoint).parent.name or Path(checkpoint).stem).lower()


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/static/<path:name>")
def static_files(name: str):
    return send_from_directory(STATIC_DIR, name)


@app.route("/favicon.ico")
def favicon():
    # Browsers auto-request this; without a route it 404s in the console. We have
    # no icon file, so answer 204 No Content to keep the console clean.
    return ("", 204)


@app.route("/api/status")
def api_status():
    if _tracker is None:
        return jsonify(running=False, input_enabled=False, dry_run=False,
                       tracker=False, error=None)
    s = _tracker.status()
    return jsonify(tracker=True, **vars(s))


@app.route("/api/strategies")
def api_strategies():
    """Which click-FSM presets the page may offer, and the current pick. Only
    meaningful for a fist-vs-rest checkpoint (else `applicable` is False)."""
    return jsonify(
        applicable=_strategy_applicable,
        current=_current_strategy,
        options=[{"id": p.id, "label": p.label, "blurb": p.blurb}
                 for p in FSM_PRESETS.values()],
    )


@app.route("/api/models")
def api_models():
    """Checkpoints on disk (+ their DOCS/models/README.md metadata), and which
    one the next run will start with -- feeds the Start-tab model picker and
    the Overview tab's model-metrics panel."""
    return jsonify(models=_available_models, current=_current_checkpoint)


@app.route("/api/start", methods=["POST"])
def api_start():
    """Arm the hand tracker from the page (opens the camera + loads the model).
    The body may carry `{"checkpoint": "models/.../best.pt"}` to switch which
    checkpoint this run loads, and `{"strategy": "3.2.1"}` to pick the FSM
    preset -- both are applied to the tracker config *before* the loop starts."""
    global _current_strategy, _current_checkpoint, _run_meta, _strategy_applicable
    if _tracker is None:
        return jsonify(ok=False, error="server started with --no-tracker"), 400
    if _tracker.status().running:
        return jsonify(ok=False, error="tracker already running"), 400
    data = request.get_json(force=True, silent=True) or {}

    cp = data.get("checkpoint")
    if cp and cp != _current_checkpoint and any(m["path"] == cp for m in _available_models):
        _current_checkpoint = cp
        _tracker.cfg.checkpoint = cp              # cfg is read when the thread starts below
        _run_meta = _derive_meta(cp, _device, tracker=True)
        _strategy_applicable = _strategy_fits(cp, tracker=True)
        _current_strategy = DEFAULT_STRATEGY if _strategy_applicable else None

    sid = data.get("strategy")
    if _strategy_applicable and sid in FSM_PRESETS:
        preset = FSM_PRESETS[sid]
        _tracker.cfg.fsm_conf = preset.fsm_conf
        _tracker.cfg.fsm_k = preset.fsm_k
        _current_strategy = sid
        _run_meta["strategy"] = preset.label       # stamp what this run actually ran
    _tracker.start()
    return jsonify(ok=True, strategy=_current_strategy, checkpoint=_current_checkpoint)


@app.route("/api/finish", methods=["POST"])
def api_finish():
    """Stop the tracker (kill-switch) so the survey uses the normal mouse."""
    if _tracker is not None:
        _tracker.stop()
    return jsonify(ok=True)


@app.route("/api/history")
def api_history():
    """Summaries of every past run in the results dir, newest first (Overview tab)."""
    d = Path(_results_dir)
    runs = []
    if d.exists():
        for p in sorted(d.glob("*.json"), reverse=True):
            s = summarize_run(p)
            if s is not None:
                runs.append(s)
    return jsonify(runs=runs)


@app.route("/api/results", methods=["POST"])
def api_results():
    data = request.get_json(force=True, silent=True) or {}
    if _run_meta and not data.get("meta"):   # stamp which model/strategy this run used
        data["meta"] = _run_meta
    try:
        path = save_results(data, out_dir=_results_dir)
    except Exception as e:  # never lose the run to a write error without saying so
        return jsonify(ok=False, error=f"{type(e).__name__}: {e}"), 500
    return jsonify(ok=True, path=str(path), filename=Path(path).name)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="models/palmfist_frozen_mnv3_large/best.pt",
                    help="binary click model (2 classes incl. 'fist')")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--results-dir", default="eval_results",
                    help="where run docs (.docx/.txt) and .json are written")
    ap.add_argument("--no-browser", action="store_true", help="don't auto-open the page")
    ap.add_argument("--dry-run", action="store_true",
                    help="run the pipeline but send NO real OS input")
    ap.add_argument("--no-tracker", action="store_true",
                    help="serve the UI only; drive it with the normal mouse (UI testing)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-mirror", action="store_true")
    # detector / mapper / FSM -- same flags & defaults as src.control.play
    ap.add_argument("--pad", type=float, default=0.15)
    ap.add_argument("--hand-model", default=None)
    ap.add_argument("--detect-confidence", type=float, default=0.3)
    ap.add_argument("--presence-confidence", type=float, default=0.3)
    ap.add_argument("--tracking-confidence", type=float, default=0.3)
    ap.add_argument("--anchor", choices=("palm", "box"), default="palm")
    ap.add_argument("--box-mode", choices=("adaptive", "fixed"), default="adaptive")
    ap.add_argument("--box-gain", type=float, default=4.0)
    ap.add_argument("--box-w", type=float, default=0.60)
    ap.add_argument("--box-h", type=float, default=0.55)
    ap.add_argument("--cursor-alpha", type=float, default=0.35)
    ap.add_argument("--deadzone", type=float, default=0.005)
    ap.add_argument("--strategy", choices=list(FSM_PRESETS), default=None,
                    help="initial click-FSM preset for a fist-vs-rest checkpoint "
                         "(the page can switch it before starting); default 3.2")
    ap.add_argument("--fsm-k", type=int, default=3)
    ap.add_argument("--fsm-conf", type=float, default=0.70)
    ap.add_argument("--fsm-grace", type=int, default=10)
    ap.add_argument("--cooldown", type=float, default=0.30)
    # runtime / latency (Strategy 3.2.3) -- same flags & defaults as src.control.play
    ap.add_argument("--backend", choices=("auto", "onnx", "torch"), default="auto",
                    help="stage-2 inference backend: onnx is 3-4x faster on this CPU and "
                         "parity-checked at load; torch reverts (default: auto)")
    ap.add_argument("--torch-threads", type=int, default=4,
                    help="torch intra-op threads (measured: 4 beats the 16-thread default)")
    ap.add_argument("--no-threaded-capture", action="store_true",
                    help="grab frames synchronously (pre-3.2.3 behaviour: same FPS, but "
                         "~67 ms staler frames)")
    ap.add_argument("--click-hold", type=float, default=0.06,
                    help="seconds to hold the mouse button down per click; 0 = the old "
                         "single-SendInput pulse a DirectX game can miss")
    return ap.parse_args()


def main() -> None:
    global _tracker, _results_dir, _run_meta, _strategy_applicable, _current_strategy
    global _available_models, _current_checkpoint, _device
    args = parse_args()
    _results_dir = args.results_dir
    _device = args.device
    _current_checkpoint = args.checkpoint
    _available_models = list_models()
    _run_meta = _derive_meta(args.checkpoint, args.device, tracker=not args.no_tracker)
    _strategy_applicable = _strategy_fits(args.checkpoint, tracker=not args.no_tracker)
    _current_strategy = (args.strategy or DEFAULT_STRATEGY) if _strategy_applicable else None

    if not args.no_tracker:
        cfg = TrackerConfig(
            checkpoint=args.checkpoint, camera=args.camera, device=args.device,
            dry_run=args.dry_run, mirror=not args.no_mirror,
            pad=args.pad, hand_model=args.hand_model,
            detect_confidence=args.detect_confidence,
            presence_confidence=args.presence_confidence,
            tracking_confidence=args.tracking_confidence,
            anchor=args.anchor, box_mode=args.box_mode, box_gain=args.box_gain,
            box_w=args.box_w, box_h=args.box_h, cursor_alpha=args.cursor_alpha,
            deadzone=args.deadzone, fsm_k=args.fsm_k, fsm_conf=args.fsm_conf,
            fsm_grace=args.fsm_grace, cooldown=args.cooldown,
            backend=args.backend, torch_threads=args.torch_threads,
            threaded_capture=not args.no_threaded_capture, click_hold=args.click_hold,
        )
        if _current_strategy:   # default preset for a fist-vs-rest checkpoint
            preset = FSM_PRESETS[_current_strategy]
            cfg.fsm_conf, cfg.fsm_k = preset.fsm_conf, preset.fsm_k
            _run_meta["strategy"] = preset.label
        _tracker = HandTracker(cfg)
        mode = "DRY-RUN (no real input)" if args.dry_run else "LIVE (real cursor + clicks)"
        strat = f"  strategy {_current_strategy}" if _current_strategy else ""
        print(f"[tracker] ready -- {mode}{strat}. Start it from the page. ESC = kill-switch.")
    else:
        print("[tracker] --no-tracker: UI only, use the normal mouse.")

    url = f"http://{args.host}:{args.port}/"
    print(f"[dashboard] serving {url}  (results -> {args.results_dir}/)")
    print("[dashboard] tip: press F11 in the browser for fullscreen so the "
          "cursor can reach every button.")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        # threaded so /api/status polls don't block; no reloader (it would spawn
        # a second tracker + camera).
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        if _tracker is not None:
            _tracker.stop()


if __name__ == "__main__":
    main()

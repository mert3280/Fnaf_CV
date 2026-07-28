"""Measure the live loop's frame budget -- the evidence behind Strategy 3.2.3.

Runs the *real* `play.py` stages (capture -> MediaPipe -> preprocess -> classify
-> HUD) against the real camera and reports per-stage milliseconds and the FPS
that falls out of them, for each runtime configuration. No game, no input
simulation, no clicks -- this only reads.

    python scripts/bench_pipeline.py                       # all configs, 8 s each
    python scripts/bench_pipeline.py --seconds 15 --preview

Keep a hand in frame for a representative run: with no hand, MediaPipe re-runs
full palm detection every frame (its slow path) and the classifier never fires,
so the two stages that dominate are both mismeasured. The report says which
regime it saw.

Two things this exists to keep honest:
* the per-stage table in the 3.2.3 strategy doc, and
* the claim that threaded capture is a **latency** fix, not a throughput one --
  `--stale` measures the camera backlog a synchronous read serves from.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402

from src.rt.backends import make_runner  # noqa: E402
from src.rt.capture import open_camera  # noqa: E402
from src.rt.detector import DEFAULT_MODEL, HandDetector  # noqa: E402
from src.rt.model_loader import load_checkpoint  # noqa: E402
from src.rt.preprocess import Preprocessor  # noqa: E402

CKPT = "models/week4_tuned_fistvsrest/best.pt"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=8.0, help="per configuration")
    ap.add_argument("--preview", action="store_true", help="include the HUD cost (imshow)")
    ap.add_argument("--stale", action="store_true",
                    help="also measure the synchronous-read camera backlog (input lag)")
    ap.add_argument("--torch-threads", type=int, default=4,
                    help="torch intra-op threads for the torch rows (0 = leave the "
                         "library default, which is what the loop used pre-3.2.3)")
    ap.add_argument("--one", default=None, metavar="BACKEND,CAPTURE",
                    help="internal: measure a single config in this process and print JSON")
    return ap.parse_args()


def ms(x: float | None) -> str:
    """Medians, because a mean here is hostage to Windows scheduler spikes."""
    return f"{x:6.1f}" if x else "    --"


def power_state() -> str:
    """AC vs. battery + current CPU clock -- stamped onto every report.

    On this laptop, running on battery drops the CPU to its 1.4 GHz base clock
    and costs ~2.1x on EVERY stage: a bigger term than anything Strategy 3.2.3
    changed. A benchmark number without this label is not comparable to another.
    """
    if sys.platform != "win32":
        return "power state unknown (non-Windows)"
    try:
        import ctypes

        class SPS(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte),
                        ("BatteryLifePercent", ctypes.c_byte), ("SystemStatusFlag", ctypes.c_byte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

        st = SPS()
        ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st))
        ac = {0: "ON BATTERY", 1: "on AC"}.get(st.ACLineStatus, "unknown power")
        mhz = subprocess.run(["wmic", "cpu", "get", "CurrentClockSpeed"],
                             capture_output=True, text=True).stdout
        clock = next((w for w in mhz.split() if w.isdigit()), "?")
        warn = "  <-- plug in: costs ~2x" if st.ACLineStatus == 0 else ""
        return f"{ac}, CPU {clock} MHz{warn}"
    except Exception:
        return "power state unavailable"


def bench(args, lm, pre, detector, backend: str, threaded: bool) -> dict:
    run, chosen = make_runner(lm, backend, "cpu", args.torch_threads or None)
    cap = open_camera(args.camera, threaded=threaded)
    t_read, t_det, t_pre, t_cls, t_hud, t_frame = [], [], [], [], [], []
    hands = 0
    try:
        cap.read()  # warm up the camera + first-detection path
        t_end = time.perf_counter() + args.seconds
        while time.perf_counter() < t_end:
            f0 = time.perf_counter()
            t = time.perf_counter()
            ok, frame = cap.read()
            t_read.append(time.perf_counter() - t)
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            t = time.perf_counter()
            hb = detector.detect(frame)
            t_det.append(time.perf_counter() - t)
            crop = detector.crop(frame, hb) if hb is not None else None
            if hb is not None:
                hands += 1
            if crop is None or crop.size == 0:  # keep stage 2 in the budget either way
                h, w = frame.shape[:2]
                s = min(h, w) // 3
                crop = frame[h // 2 - s:h // 2 + s, w // 2 - s:w // 2 + s]

            t = time.perf_counter()
            x = pre(crop)
            t_pre.append(time.perf_counter() - t)
            t = time.perf_counter()
            run(x)
            t_cls.append(time.perf_counter() - t)

            if args.preview:
                t = time.perf_counter()
                cv2.putText(frame, "bench", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 230, 0), 2, cv2.LINE_AA)
                cv2.imshow("bench", frame)
                cv2.waitKey(1)
                t_hud.append(time.perf_counter() - t)
            t_frame.append(time.perf_counter() - f0)
        dropped = cap.dropped
    finally:
        cap.release()
        if args.preview:
            cv2.destroyAllWindows()

    total = statistics.median(t_frame)
    return dict(backend=chosen, capture="threaded" if threaded else "sync",
                read=t_read, det=t_det, pre=t_pre, cls=t_cls, hud=t_hud,
                total=total, fps=len(t_frame) / sum(t_frame), n=len(t_frame),
                hand_frac=hands / max(1, len(t_frame)), dropped=dropped)


def measure_stale(camera: int) -> None:
    """How old is the frame a synchronous read hands back when the loop is
    slower than the camera? Run a deliberately slow loop, then stop working and
    count the frames that come back instantly -- that backlog IS the input lag."""
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
    for _ in range(5):
        cap.read()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 4.0:
        cap.read()
        time.sleep(0.057)  # ~17 FPS loop against a 30 FPS camera
    backlog = 0
    while backlog < 200:
        t = time.perf_counter()
        cap.read()
        if (time.perf_counter() - t) * 1000 > 15:  # blocked => queue drained
            break
        backlog += 1
    cap.release()
    print(f"\nsynchronous-read backlog after a slow loop: {backlog} frames "
          f"(~{backlog * 33.3:.0f} ms of input lag; threaded capture removes it)")


def run_one(args) -> None:
    """Measure a single config and print it as JSON (the subprocess entry point)."""
    backend, capture = args.one.split(",")
    lm = load_checkpoint(args.checkpoint, device="cpu")
    pre = Preprocessor(lm.size, lm.mean, lm.std)
    detector = HandDetector(DEFAULT_MODEL)
    try:
        r = bench(args, lm, pre, detector, backend, capture == "threaded")
    finally:
        detector.close()
    for k in ("read", "det", "pre", "cls", "hud"):  # medians only, ms
        r[k] = statistics.median(r[k]) * 1000 if r[k] else None
    print("BENCH_JSON " + json.dumps(r))


def main() -> None:
    args = parse_args()
    if args.one:
        return run_one(args)

    print(load_checkpoint(args.checkpoint, device="cpu").describe())
    # Each config runs in its own process: an ONNX session and a torch runner
    # each hold a thread pool, and measuring several in one process
    # oversubscribes the CPU -- which made an earlier version of this script
    # report every config after the first as progressively slower.
    rows = []
    for backend, capture in (("torch", "sync"), ("torch", "threaded"),
                             ("onnx", "sync"), ("onnx", "threaded")):
        cmd = [sys.executable, __file__, "--one", f"{backend},{capture}",
               "--checkpoint", args.checkpoint, "--camera", str(args.camera),
               "--seconds", str(args.seconds),
               "--torch-threads", str(args.torch_threads)]
        if args.preview:
            cmd.append("--preview")
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
        line = next((ln for ln in out.stdout.splitlines() if ln.startswith("BENCH_JSON ")), None)
        if line is None:
            print(f"[warn] {backend}/{capture} failed:\n{out.stdout[-500:]}{out.stderr[-500:]}")
            continue
        rows.append(json.loads(line[len("BENCH_JSON "):]))

    print(f"\nper-frame medians, {args.seconds:.0f}s each, "
          f"HUD {'on' if args.preview else 'off'}  |  {power_state()}\n")
    print(f"{'backend':7} {'capture':8} {'read':>6} {'detect':>6} {'prep':>6} "
          f"{'model':>6} {'hud':>6} {'total':>6} {'FPS':>6}  hand")
    for r in rows:
        print(f"{r['backend']:7} {r['capture']:8} {ms(r['read'])} {ms(r['det'])} "
              f"{ms(r['pre'])} {ms(r['cls'])} {ms(r['hud'])} "
              f"{r['total'] * 1000:6.1f} {r['fps']:6.1f}  {r['hand_frac'] * 100:3.0f}%")
    if rows and rows[0]["hand_frac"] < 0.5:
        print("\n[warn] a hand was in frame for <50% of frames -- MediaPipe was mostly "
              "on its slow full-detection path; re-run with a hand up for the real number.")
    if args.stale:
        measure_stale(args.camera)


if __name__ == "__main__":
    main()

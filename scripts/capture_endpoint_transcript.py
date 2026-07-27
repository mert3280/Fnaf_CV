"""Start the inference endpoint, exercise it with REAL images, and write the
request/response transcript the Week-4 report quotes (M4A1 section 4).

Nothing here is hand-written into the report: it boots `src/serve/app.py` in a
subprocess, waits for /health, issues the calls, measures latency over repeated
requests, then shuts the server down and writes the transcript verbatim.

    python scripts/capture_endpoint_transcript.py                    # champion
    python scripts/capture_endpoint_transcript.py --checkpoint models/.../best.pt
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import requests  # noqa: E402

OUT = REPO / "DOCS" / "class-related" / "week4" / "endpoint-transcript.md"


def crop_b64(gesture: str, which: int = 0) -> str:
    """A real HaGRID image of `gesture`, cropped exactly as training crops (AD-04)."""
    from PIL import Image

    from src.data.config import DataConfig
    from src.data.dataset import padded_bbox_pixels
    from src.data.hagrid_annotations import build_index

    cfg = DataConfig.from_yaml(REPO / "configs" / "data_fistvsrest.yaml")
    images, ann = cfg.sources[0]
    idx = build_index(REPO / images, REPO / ann, [gesture], {gesture: 0})
    row = idx.iloc[which]
    img = Image.open(row["path"]).convert("RGB")
    w, h = img.size
    img = img.crop(padded_bbox_pixels(row["bbox"], w, h, cfg.input.bbox_pad))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def wait_for(url: str, timeout: float = 180.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.ok:
                return r.json()
        except requests.RequestException as exc:
            last = exc
        time.sleep(1.0)
    raise SystemExit(f"endpoint never became healthy at {url}: {last}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="serve a checkpoint instead of the registry champion")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--latency-samples", type=int, default=30)
    ap.add_argument("--server-log", default="pipeline_runs/endpoint_server.log",
                    help="where the served app's own stdout/stderr goes")
    args = ap.parse_args()

    cmd = [sys.executable, "-u", "-m", "src.serve.app", "--port", str(args.port)]
    if args.checkpoint:
        cmd += ["--checkpoint", args.checkpoint]
    print(f"$ {' '.join(cmd[2:])}")
    # The server's output goes to a FILE, never to an unread PIPE: Flask logs a line
    # per request (plus a traceback for the malformed-input case), and once an
    # undrained pipe buffer fills, the child blocks on write and the endpoint
    # deadlocks mid-transcript.
    log = Path(args.server_log)
    log.parent.mkdir(parents=True, exist_ok=True)
    server_log = open(log, "w", encoding="utf-8")
    server = subprocess.Popen(cmd, cwd=REPO, stdout=server_log,
                              stderr=subprocess.STDOUT, text=True)
    print(f"  (server log -> {log})")
    base = f"http://127.0.0.1:{args.port}"
    sections: list[tuple[str, str, str]] = []   # (title, request, response)
    try:
        health = wait_for(f"{base}/health")
        print("health:", health)
        sections.append(("Liveness / which model is loaded",
                         f"curl -s {base}/health",
                         json.dumps(health, indent=2)))

        info = requests.get(f"{base}/model", timeout=10).json()
        sections.append(("Served model metadata",
                         f"curl -s {base}/model",
                         json.dumps(info, indent=2)))

        # --- single prediction on a real fist crop ------------------------- #
        fist = crop_b64("fist")
        r = requests.post(f"{base}/predict", json={"image_b64": fist}, timeout=60)
        single = r.json()
        print("predict(fist):", single["predictions"][0])
        req = ('curl -s -X POST %s/predict -H "Content-Type: application/json" \\\n'
               '  -d \'{"image_b64": "<base64 of a 224px hand crop, %d chars>"}\''
               % (base, len(fist)))
        sections.append(("Single prediction — a real `fist` crop (a click)", req,
                         json.dumps(single, indent=2)))

        # --- batch: 2 fists + 2 palms -------------------------------------- #
        batch = [crop_b64("fist", 0), crop_b64("fist", 1),
                 crop_b64("palm", 0), crop_b64("palm", 1)]
        rb = requests.post(f"{base}/predict", json={"images": batch}, timeout=120).json()
        sections.append(("Batch of 4 — two `fist`, two `palm`",
                         f'curl -s -X POST {base}/predict -H "Content-Type: application/json" \\\n'
                         f'  -d \'{{"images": ["<fist>", "<fist>", "<palm>", "<palm>"]}}\'',
                         json.dumps(rb, indent=2)))

        # --- an error case: not an image ----------------------------------- #
        bad = requests.post(f"{base}/predict",
                            json={"image_b64": base64.b64encode(b"not-an-image").decode()},
                            timeout=30)
        sections.append((f"Malformed input (HTTP {bad.status_code}) — the endpoint "
                         f"rejects rather than guessing",
                         f'curl -s -X POST {base}/predict -H "Content-Type: application/json" \\\n'
                         f'  -d \'{{"image_b64": "bm90LWFuLWltYWdl"}}\'',
                         json.dumps(bad.json(), indent=2)))

        # --- latency ------------------------------------------------------- #
        lat = []
        for _ in range(args.latency_samples):
            t0 = time.perf_counter()
            requests.post(f"{base}/predict", json={"image_b64": fist}, timeout=60)
            lat.append((time.perf_counter() - t0) * 1000)
        lat.sort()
        stats = {
            "samples": len(lat),
            "p50_ms": round(statistics.median(lat), 1),
            "p95_ms": round(lat[int(0.95 * (len(lat) - 1))], 1),
            "min_ms": round(lat[0], 1),
            "max_ms": round(lat[-1], 1),
            # server-side timings exclude the client/loopback overhead
            "server_single_ms": single["latency_ms"],
            "server_batch4_per_image_ms": round(rb["latency_ms"] / len(batch), 1),
        }
        print("latency:", stats)
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
        server_log.close()

    body = [
        "# Inference endpoint — captured transcript",
        "",
        "_Generated by `scripts/capture_endpoint_transcript.py` "
        f"({time.strftime('%Y-%m-%d %H:%M')}). Every request and response below is "
        "verbatim: the script boots `src/serve/app.py`, calls it over HTTP with real "
        "HaGRID hand crops, then shuts it down._",
        "",
        "```bash",
        f"python -m src.serve.app{' --checkpoint ' + args.checkpoint if args.checkpoint else ''}"
        f"  # serves {'that checkpoint' if args.checkpoint else 'models:/fnaf-click-classifier@champion'}",
        "```",
        "",
    ]
    for title, req, resp in sections:
        body += [f"## {title}", "", "```bash", req, "```", "", "```json", resp, "```", ""]
    body += [
        "## Latency (loopback HTTP, single-image requests, CPU)",
        "",
        "| metric | ms |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in stats.items()],
        "",
        "The number that matters for the product is not this one: the live controller "
        "calls the model **in-process** (see the report §4), because a webcam frame "
        "budget at 30 fps is ~33 ms and the model alone spends most of that.",
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(body), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

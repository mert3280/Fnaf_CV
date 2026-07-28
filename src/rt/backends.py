"""Stage-2 inference backends -- fix #2 of Strategy 3.2.3.

The classifier is the largest *compute* term in the live loop. Measured on this
box (`mobilenetv3_large_100`, 224 px, same input, same CPU):

    PyTorch, 16 threads (as shipped)   29.4 ms
    PyTorch, 4 threads                 23.1 ms
    ONNX Runtime, 4 threads             5.3 ms   <- 4.3x

The `model.onnx` next to each checkpoint is **already** exported and parity-
checked by the training DAG (AD-11 / `src/pipeline/tasks.py::export_onnx`); it
simply wasn't what the live loop loaded. This module makes it selectable.

**The crop/normalize contract (A.3) is untouched**: both backends consume the
*same* tensor from `Preprocessor`, so the ONNX path changes the arithmetic's
speed, not its inputs. And because a silently-wrong backend would be the worst
possible bug here, `make_runner` re-runs the torch-vs-ONNX parity check **at
load time** on this machine's runtime, and refuses to use ONNX if it fails --
the loop falls back to torch and says so, rather than serving a different model
than the one that was evaluated.

`--backend torch` reverts; `--backend auto` (default) prefers ONNX when it's
available and passes parity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from src.rt.model_loader import LoadedModel

# Max acceptable per-probability disagreement between backends at load time.
# The DAG's export-time check on logits lands ~1e-6; 1e-3 on probabilities is
# loose enough to absorb a different machine's kernels, tight enough that no
# label or FSM confidence gate could flip because of it.
PARITY_TOL = 1e-3


def _torch_runner(lm: LoadedModel, device: str, threads: int | None) -> Callable:
    if threads:
        # Measured: 16 threads is *slower* than 4 on this CPU -- a 224px
        # mobilenet is too small to amortize the sync overhead across cores.
        torch.set_num_threads(threads)
    model = lm.model

    def run(tensor: torch.Tensor) -> np.ndarray:
        with torch.inference_mode():
            logits = model(tensor.to(device))
            return torch.softmax(logits, dim=1)[0].cpu().numpy()

    return run


def _onnx_runner(onnx_path: Path, threads: int | None) -> Callable:
    import onnxruntime as ort

    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    def run(tensor: torch.Tensor) -> np.ndarray:
        logits = sess.run(None, {name: tensor.numpy()})[0]
        e = np.exp(logits[0] - logits[0].max())  # softmax, matching the torch path
        return (e / e.sum()).astype(np.float32)

    return run


def _parity_ok(a: Callable, b: Callable, size: int) -> tuple[bool, float]:
    """Do the two backends agree on this machine? Compares *probabilities*,
    which is what the FSM's confidence gate actually reads."""
    x = torch.rand(1, 3, size, size)
    diff = float(np.abs(np.asarray(a(x)) - np.asarray(b(x))).max())
    return diff <= PARITY_TOL, diff


def make_runner(
    lm: LoadedModel,
    backend: str = "auto",
    device: str = "cpu",
    threads: int | None = 4,
) -> tuple[Callable[[torch.Tensor], np.ndarray], str]:
    """Build the per-frame `tensor -> probs` callable. Returns `(run, chosen)`.

    `backend`: "torch" | "onnx" | "auto" (ONNX when usable, else torch).
    "onnx" is a *request*, not a guarantee -- a missing file, a missing
    `onnxruntime`, or a failed parity check falls back to torch with a warning,
    because running an unverified backend against the real game is worse than
    running a slow one.
    """
    torch_run = _torch_runner(lm, device, threads)
    if backend == "torch":
        return torch_run, "torch"

    onnx_path = lm.path.parent / "model.onnx"
    why = None
    if device != "cpu":
        why = f"--device {device} requested (the ONNX session here is CPU-only)"
    elif not onnx_path.exists():
        why = f"no {onnx_path} (export it: python -m src.pipeline.run --dag training)"
    if why is None:
        try:
            onnx_run = _onnx_runner(onnx_path, threads)
        except ImportError:
            why = "onnxruntime not installed (pip install onnxruntime)"
        except Exception as e:  # corrupt/incompatible export -- never fatal
            why = f"could not load {onnx_path.name}: {e}"
    if why is None:
        ok, diff = _parity_ok(torch_run, onnx_run, lm.size)
        if ok:
            print(f"[backend] onnx  ({onnx_path.name}, parity max|torch-onnx| = {diff:.2e})")
            return onnx_run, "onnx"
        why = f"parity check FAILED: max|torch-onnx| = {diff:.2e} > {PARITY_TOL:.0e}"

    level = "warn" if backend == "onnx" else "info"
    print(f"[{level}] backend onnx unavailable -- {why}; using torch")
    return torch_run, "torch"

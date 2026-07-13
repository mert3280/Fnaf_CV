"""Real-time inference package (Phases 4-5).

Turns a trained checkpoint into a live webcam classifier. Everything a model
needs to run -- backbone name, class list, input size/normalization, crop mode
-- is read back out of the checkpoint written by `src/train.py`, so switching
models is just pointing `--checkpoint` at a different `best.pt`.
"""

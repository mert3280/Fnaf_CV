# Project Proposal — Gesture-Controlled *Five Nights at Freddy's*

**Author:** Ted Roper
**Course:** ML Projects, Summer 2026
**Status:** Solo project
**Date:** 2026-06-30

---

## What?

Most video games assume a player who can comfortably use a keyboard and mouse. That excludes people with certain motor limitations, and it ignores a more playful question: *what if your hands were the controller — no peripheral at all?* This project builds a system that lets a person play the original **Five Nights at Freddy's (FNAF)** — a tense survival-horror game about watching security cameras and slamming doors to keep animatronics out — using only **hand gestures captured by a standard webcam**. A camera watches the player's hand, a computer-vision model recognizes which gesture they are making (a fist, a thumbs-up, an open palm, and so on), and the system translates that gesture into the mouse clicks the game expects — closing a door, flicking a light, raising the camera monitor. Under the hood, the recognition engine is a **deep-learning image classifier built by transfer learning**: we take a model already trained to understand hands and re-train its decision layer on the **HaGRID** gesture dataset, then wrap it in a real-time loop. The business-facing output is a **playable, hands-free control layer**: point a webcam at yourself and survive Night 1 of FNAF without touching the mouse.

## Why?

I chose this project because it sits at the exact intersection of two things I find genuinely fun: **computer vision** and **games**. Gesture control is one of those technologies that feels like magic when it works and is deeply unimpressive when it lags or misfires — which makes it a great forcing function for building something that is not just "accurate on a test set" but *actually responsive in the real world*. FNAF is a deliberately chosen target because it is **tense and unforgiving**: a half-second of input latency or one misclassified gesture can get you killed in-game, so the project has a built-in, brutally honest evaluation metric — *can you actually survive the night?* That tight feedback loop is far more motivating to me than a static accuracy number, and it pushes the work past "train a classifier" into the messier, more interesting territory of latency, debouncing, and human-in-the-loop reliability.

## My Takeaway

The specific capability I want to prove to myself is **end-to-end transfer learning** — not just calling `.fit()` on a pretrained model, but genuinely understanding the mechanics: which layers to **freeze**, when to **swap and re-initialize the classification head**, how to **stage fine-tuning** (frozen backbone first, then progressive unfreezing), and how to read training curves to diagnose under- vs. over-fitting. That is the gap I most want to close; I have used pretrained models as black boxes before, and I want to come out of this able to *justify every layer decision*. The second skill is **bridging an offline model to a live system**: taking a model that scores well on a held-out test set and making it survive contact with a noisy webcam feed, variable lighting, and the real-time constraints of driving another application. If I can do both, I will have proven I can carry a model from dataset to a thing a human actually uses. I am also very interested in learing how to interface my ML projects into different applications other than a dashboard or a notebook, and this project is a good way to explore that.

## Tech Stack

Annotated; **(familiar)** = I have used it before, **(new)** = new to me, with a get-up-to-speed note.

| Layer | Technology | Familiarity & ramp-up plan |
|---|---|---|
| **Language** | Python 3.11 | **(familiar)** — primary language. |
| **Dataset** | HaGRID (HAnd Gesture Recognition Image Dataset), 18 static classes | **(new)** — I already have the annotation files locally (`DATA/`). Ramp-up: read the WACV 2024 paper and the dataset README to understand the bbox + 21-landmark JSON schema and the train/test split-by-user methodology. |
| **Model framework** | PyTorch | **(familiar)** — core training/inference. |
| **Pretrained models** | `timm` (PyTorch Image Models) + Hugging Face Hub | **(new)** — following the Christian Mills timm/HF transfer-learning tutorial. Ramp-up: work through that tutorial end-to-end on a toy run before applying it to HaGRID. |
| **Hand detection / cropping** | MediaPipe Hands (used as a preprocessing cropper, optional) | **(new)** — read the official solution guide; only needed if full-frame classification proves unreliable. |
| **Data / numerics** | NumPy, Pandas, Pillow, OpenCV | **(familiar)** — image I/O, annotation parsing, webcam capture. |
| **Experiment tracking** | TensorBoard (or Weights & Biases) | **(familiar/new)** — TensorBoard known; will evaluate W&B for run comparison. |
| **Real-time capture** | OpenCV `VideoCapture` | **(familiar)** — webcam frame loop. |
| **Input simulation** | `pydirectinput` (preferred on Windows for games) with `pyautogui` fallback | **(new)** — `pyautogui`'s SendInput doesn't always register in DirectX games; ramp-up by testing click registration against FNAF early in Phase 4 to de-risk. |
| **Model export** | TorchScript / ONNX + ONNX Runtime | **(new)** — for low-latency inference; ramp-up via PyTorch export docs, with a fallback to plain PyTorch eval if export stalls. |
| **Target game** | *Five Nights at Freddy's* (Steam) | **(familiar as a player)** — mouse-driven; I own a copy of it. |
| **Environment** | Windows 11, conda/venv, local **NVIDIA RTX 4060 Laptop GPU (8 GB VRAM, CUDA)** | **(familiar)** — all training and inference run locally; **no cloud/Colab.** 8 GB VRAM is sufficient for a small backbone on the 8-class subset (mixed precision if needed). |
| **Version control / docs** | Git + GitHub, Markdown | **(familiar)** — branch → PR → merge to `main` with `DOCS/`. |

## Proposed Schedule

A 5-week plan. Weeks 1–2 establish setup and **data validation**; meaningful technical milestones land from Week 2 onward. Detail lives in [schedule.md](schedule.md) and the per-phase files in [phases/](phases/).

| Week | Goal | "Behind schedule" signal |
|---|---|---|
| **1** | Environment + repo + **data validation**: parse HaGRID annotations, acquire matching images (subsample/512px), build a verified PyTorch `Dataset`/`DataLoader`, run EDA (class balance, sample crops), finalize the gesture → FNAF action map. | End of week and I cannot load a batch of (image, label) pairs and visualize them. |
| **2** | **Baseline model**: timm pretrained backbone with a fresh classification head, backbone **frozen**, trained on a subset. First real train/val curves and a confusion matrix. | No training loop runs to completion; no baseline accuracy recorded. |
| **3** | **Full training + fine-tuning**: progressive unfreezing, augmentation, hyperparameter tuning; evaluate on the held-out test set; export the best model. | Test accuracy stuck below a usable threshold (target ≥ 90% on the chosen subset) with no diagnosed cause. |
| **4** | **Real-time + game control**: webcam inference loop, temporal smoothing/debouncing, gesture → simulated-input layer; verified clicks landing in FNAF. | Webcam → gesture works but no input reliably registers in the actual game. |
| **5** | **Integration, robustness, demo**: calibration, lighting robustness, full Night-1 playthrough, documentation, and a recorded demo. | Cannot complete a full hands-free Night 1; demo not recorded. |

## Claude & AI Usage Plan

Full plan in [Claude.md](Claude.md); living weekly log in [AI-usage.md](AI-usage.md), and the agent-context file at the repo root [`CLAUDE.md`](../CLAUDE.md). In short: AI assistance is **scoped to scaffolding and acceleration, not to replacing the learning objectives.** I will use Claude for boilerplate (DataLoader/transform scaffolding, argparse CLIs, plotting), for **debugging** (interpreting stack traces, CUDA/dtype mismatches, input-simulation quirks), and for **documentation**. The core learning targets — deciding the freezing strategy, reading training curves, and tuning the real-time control loop — I will **drive manually**, using Claude as a sounding board rather than an autopilot. Every non-trivial AI interaction (task, the prompt/context that worked, and any output I had to correct) is logged weekly in `AI-usage.md`.

## Scope Justification

*Not applicable — this is a solo project.* The scope is deliberately bounded to a single player, a single game (FNAF 1), **static gestures only**, and a curated subset of HaGRID classes, which keeps it achievable solo within five weeks while still exercising the full pipeline from dataset to a live, human-operated system.

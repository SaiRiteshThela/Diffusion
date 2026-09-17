# Twelve-hour A40 campaign

The campaign is complete. The selected ADM64 checkpoint at step 66,500 achieved
FID **6.382** with 10,000 samples and 100 Euler steps.
See the [results report](../reports/campaign_20260917.md).
W&B links remain in ignored local runtime files and are excluded from public reports.

The campaign screens six configurations: two unconditional ADM-style U-Net widths (23.0M and 51.8M parameters), each at peak learning rates 1e-4, 2e-4, and 3e-4. It keeps the existing independent Gaussian pairing, uniform time sampling, straight-line path, velocity MSE, and CelebA crop178/resize64 preprocessing. It does not resume or overwrite Experiment 6.

Training uses BF16 forwards with FP32 velocity MSE, channels-last convolution, AdamW, gradient clipping, cosine learning rate, and warm-started EMA. Each of the six configurations receives a 25-minute pilot, with existing training time credited. The best learning rate for each width receives a further 45 minutes. The controller then continues the stronger finalist in resumable segments for the remaining training budget, roughly six hours after screening and evaluation overhead. Candidate comparisons use 2,048-sample Euler32 FID. The final 60 minutes are reserved for 10,000-sample Euler100 evaluation; the original deadline remains fixed. Screening FID must only be compared to results with the same sample count and solver budget. The final report distinguishes the previously recorded Experiment 5 baseline from newly computed scores.

The fixed deadline for this run is **2026-09-17 15:15:44 UTC**. The training loop stops between completed optimizer updates, and a separate controller enforces process deadlines. No new GPU work starts after the deadline. A stuck worker is terminated before the global deadline. Existing periodic checkpoints remain available if a final save fails.

## Monitor

Open `experiments/campaign_12h_live.ipynb` in the pod's Jupyter server and run its monitor cells. Training is independent of the notebook. Reconnect and rerun the monitor whenever needed.

```bash
tmux attach -t a40-12h
tail -f .training/a40-12h/controller.log
```

Detach tmux with Ctrl-b, then d. The RunPod instance must remain running; disconnecting the client is fine.

Runtime state is in `outputs/campaign_20260917/status.json`. Each candidate's `run.json` contains its W&B URL. Its `latest.pt.pointer.json` identifies the immutable file containing full optimizer/EMA/scheduler/RNG state; `best_val.pt.pointer.json` identifies the fixed-validation best. The last three versions per stream are retained. Inference snapshots have unique `*_ema.step*.pt` filenames. Immutable files avoid stale replaced-file reads observed on GeeseFS; `training.trainer.resolve_checkpoint_path` resolves a logical checkpoint name. Final checkpoint selection and sample paths are written to `selected_model.json`.

## Launch / recover

The current session was launched with a file lock to prevent duplicate controllers:

```bash
flock -n .training/a40-12h/worker.lock .venv/bin/python -u experiments/campaign_12h.py --publish-results
```

Run this inside tmux only if no campaign controller is already running. The controller reads saved phase state and keeps the original deadline. `--publish-results` commits and pushes only the final small Markdown report; weights, datasets, W&B files, and credentials are excluded. The source code and monitoring notebook are committed separately before leaving the run unattended.

Shared real FID statistics use the existing 162,770-image cache. They were adopted after provenance validation (same dataset/cache, preprocessing, split, resolution, and feature protocol), rather than independently recomputed. Generated-image cache manifests include checkpoint identity, config, EMA selection, seed, precision, batch size, and solver budget.

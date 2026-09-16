# Diffusion

An unconditional MNIST flow-matching experiment built with PyTorch and a custom
U-Net.

## Structure

- `models/` — U-Net, attention, residual blocks, and continuous-time embeddings
- `datasets/` — cached MNIST sampler with train/validation/test splits
- `training/` — Gaussian probability path and conditional flow-matching trainer
- `sampling/` — ODE interfaces and Euler integration
- `configs/` — model configurations
- `tests/` — unit and integration tests
- `experiments/` — numbered experiment notebooks

## Setup

```bash
uv sync --dev
```

## RunPod

Use the pod disk for the active checkout and virtual environment, and the
GeeseFS global volume for persistent data:

```bash
bash /workspace-global/bootstrap-diffusion.sh
cd /workspace/Diffusion
```

The bootstrap script restores GitHub access, clones or updates the repository,
installs `uv`, and creates the environment on the fast pod disk. Training data,
checkpoints, samples, and W&B files are written beneath
`/workspace-global/Diffusion-data`. Override that location with
`DIFFUSION_DATA_ROOT`.

## Tests

```bash
uv run pytest
```

## Training

Sign in to Weights & Biases once with `uv run wandb login`, then open
`experiments/experiment_1.ipynb` with the `Python (diffusion)` kernel and run
both cells. The
trainer periodically records validation loss, writes checkpoints under
`checkpoints/`, saves trajectories under `samples/`, and logs losses and
trajectory grids to the `mnist-flow-matching` W&B project.

`experiments/experiment_2.ipynb` runs a 30-trial W&B Bayesian sweep over the
U-Net architecture and optimizer settings, minimizing the running-best
validation loss. It then retrains the winning configuration for 5,000 steps for
a fair comparison with Experiment 1. Sweep checkpoints persist under
`/workspace-global/Diffusion-data/checkpoints/experiment_2`.

Generate 25 samples from the Experiment 2 winner with 1,000 Euler steps:

```bash
uv run python -m sampling.generate_experiment_2
```

`experiments/experiment_3.ipynb` keeps the Experiment 2 winning architecture
and applies the training profile from
[`Michedev/flow-matching-mnist`](https://github.com/Michedev/flow-matching-mnist):
Adam, learning rate `1e-4`, batch size 32, and ten epoch-equivalents. Its target
is to improve on the verified Experiment 2 validation loss of `0.124551`.

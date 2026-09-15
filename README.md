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
- `dev.ipynb` — model setup and an interactive MNIST training run

## Setup

```bash
uv sync --dev
```

## Tests

```bash
uv run pytest
```

## Training

Open `dev.ipynb` with the `Python (diffusion)` kernel and run both cells. The
trainer periodically records validation loss, writes checkpoints under
`checkpoints/`, and saves trajectories under `samples/`.

"""Save a viewing grid from a checkpoint. Prefers EMA weights, then raw."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from models.config import load_config
from models.flow import FlowModel
from sampling.fid_celeba import load_eval_weights
from sampling.ode import EulerSimulator, FlowODE


def generate_final_grid(
    model: torch.nn.Module,
    checkpoint: Path,
    output_dir: Path,
    *,
    num_samples: int = 25,
    ode_steps: int = 100,
    seed: int = 42,
    image_size: int = 64,
    in_channels: int = 3,
    device: torch.device | None = None,
    display_size: int = 256,
) -> Path:
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if ode_steps < 1:
        raise ValueError("ode_steps must be positive")
    if device is None:
        device = next(model.parameters()).device

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    used_ema = state.get("ema_model") is not None
    load_eval_weights(model, state, use_ema=True)
    step = int(state.get("step", -1))
    print(
        f"generating from {checkpoint} "
        f"(step {step}, {'EMA' if used_ema else 'raw'} weights)"
    )

    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        num_samples,
        in_channels,
        image_size,
        image_size,
        generator=generator,
        device=device,
    )
    ts = torch.linspace(0, 1, ode_steps + 1, device=device).expand(num_samples, -1)
    samples = EulerSimulator(FlowODE(model)).simulate(noise, ts).clamp(-1, 1).cpu()

    output_dir = Path(output_dir)
    individual_dir = output_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)
    torch.save(samples, output_dir / "samples.pt")
    display_samples = F.interpolate(
        samples,
        size=(display_size, display_size),
        mode="bilinear",
        align_corners=False,
    )
    grid_path = output_dir / "grid_5x5.png"
    nrow = int(num_samples**0.5)
    save_image(
        display_samples,
        grid_path,
        nrow=max(nrow, 1),
        normalize=True,
        value_range=(-1, 1),
        padding=4,
        pad_value=1,
    )
    for index, sample in enumerate(display_samples):
        save_image(
            sample,
            individual_dir / f"sample_{index + 1:02d}.png",
            normalize=True,
            value_range=(-1, 1),
        )
    print("grid", grid_path)
    return grid_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment_6.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("samples/experiment_6/final"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    checkpoint = args.checkpoint
    if checkpoint is None:
        ckpts = sorted(Path("checkpoints/experiment_6").glob("*_best.pt"))
        if not ckpts:
            raise FileNotFoundError("no experiment_6 *_best.pt checkpoint found")
        checkpoint = ckpts[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlowModel.from_config(args.config).to(device)
    generate_final_grid(
        model,
        checkpoint,
        args.output_dir,
        num_samples=cfg["sampling"]["num_samples"],
        ode_steps=cfg["sampling"]["ode_steps"],
        seed=cfg["data"]["seed"],
        image_size=cfg["data"]["image_size"],
        in_channels=cfg["unet"]["in_channels"],
        device=device,
    )


if __name__ == "__main__":
    main()

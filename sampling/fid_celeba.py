"""CelebA-64 FID for an existing flow-matching checkpoint.

Protocol matches common paper reporting: 50k generated samples, pytorch-fid
Inception-v3 pool3 features, official CelebA train images at 64x64.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.utils import save_image
from tqdm import tqdm

from models.config import load_config
from models.flow import FlowModel
from sampling.ode import EulerSimulator, FlowODE

DEFAULT_CHECKPOINT = Path("checkpoints/experiment_5/ybap4pux_best.pt")
DEFAULT_CONFIG = Path("configs/experiment_5.yaml")


def to_uint8(images: Tensor) -> Tensor:
    """Map [-1, 1] floats to uint8 pixels, matching a save-then-reload PNG."""
    return ((images.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def load_eval_weights(model: nn.Module, state: dict[str, Any], use_ema: bool = True) -> nn.Module:
    weights = state.get("ema_model") if use_ema else None
    if weights is None:
        weights = state["model"]
    model.load_state_dict(weights)
    model.eval()
    model.requires_grad_(False)
    return model


class _AutocastFlow(nn.Module):
    """BF16 forward, FP32 velocity so Euler does not accumulate in BF16."""

    def __init__(self, flow: nn.Module, enabled: bool):
        super().__init__()
        self.flow = flow
        self.enabled = enabled

    def forward(self, xt: Tensor, t: Tensor) -> Tensor:
        if not self.enabled:
            return self.flow(xt, t)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return self.flow(xt, t).float()


@torch.no_grad()
def generate_uint8(
    model: nn.Module,
    num_samples: int,
    image_size: int,
    channels: int,
    ode_steps: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> Tensor:
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if ode_steps < 1:
        raise ValueError("ode_steps must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    flow = _AutocastFlow(model, enabled=device.type == "cuda")
    simulator = EulerSimulator(FlowODE(flow))
    generator = torch.Generator(device=device).manual_seed(seed)
    images = torch.empty(num_samples, channels, image_size, image_size, dtype=torch.uint8)
    done = 0
    progress = tqdm(total=num_samples, desc="generate")
    while done < num_samples:
        current = min(batch_size, num_samples - done)
        noise = torch.randn(
            current,
            channels,
            image_size,
            image_size,
            generator=generator,
            device=device,
        )
        ts = torch.linspace(0, 1, ode_steps + 1, device=device).expand(current, -1)
        samples = simulator.simulate(noise, ts, use_tqdm=False).clamp(-1, 1)
        images[done : done + current] = to_uint8(samples.cpu())
        done += current
        progress.update(current)
    progress.close()
    return images


def activation_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("need at least two feature rows to estimate FID stats")
    mean = np.mean(features, axis=0)
    covariance = np.cov(features, rowvar=False)
    return mean, covariance


def frechet_distance(
    mean_real: np.ndarray,
    cov_real: np.ndarray,
    mean_fake: np.ndarray,
    cov_fake: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """pytorch-fid Frechet distance, compatible with SciPy 1.18+ (no sqrtm disp)."""
    from scipy import linalg

    if mean_real.shape != mean_fake.shape:
        raise ValueError("mean vectors have different lengths")
    if cov_real.shape != cov_fake.shape:
        raise ValueError("covariance matrices have different shapes")

    diff = mean_real - mean_fake
    covmean = linalg.sqrtm(cov_real.dot(cov_fake))
    if not np.isfinite(covmean).all():
        offset = np.eye(cov_real.shape[0]) * eps
        covmean = linalg.sqrtm((cov_real + offset).dot(cov_fake + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real
    return float(
        diff.dot(diff) + np.trace(cov_real) + np.trace(cov_fake) - 2 * np.trace(covmean)
    )


@torch.no_grad()
def inception_features(
    images: Tensor,
    inception: nn.Module,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must be uint8 NCHW RGB")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    inception.eval()
    chunks: list[Tensor] = []
    for start in tqdm(range(0, images.shape[0], batch_size), desc="inception"):
        batch = images[start : start + batch_size].to(device, non_blocking=True)
        predicted = inception(batch.float().div_(255.0))[0]
        if predicted.shape[2] != 1 or predicted.shape[3] != 1:
            predicted = torch.nn.functional.adaptive_avg_pool2d(predicted, (1, 1))
        chunks.append(predicted.flatten(1).cpu())
    return torch.cat(chunks, dim=0).numpy()


def celeba_cache_path(root: str | Path, split: str, image_size: int) -> Path:
    return Path(root) / "celeba" / f"cache_{split}_{image_size}.pt"


def resolve_data_root() -> Path:
    local = Path("data/raw")
    if local.exists():
        return local
    return Path("/workspace-global/Diffusion-data/datasets")


def choose_batch_size(
    model: nn.Module,
    channels: int,
    image_size: int,
    device: torch.device,
    requested: int,
) -> int:
    if requested > 0:
        return requested
    if device.type != "cuda":
        return 32
    flow = _AutocastFlow(model, enabled=True)
    simulator = EulerSimulator(FlowODE(flow))
    for batch_size in (256, 192, 128, 96, 64, 32, 16, 8):
        try:
            noise = torch.randn(batch_size, channels, image_size, image_size, device=device)
            ts = torch.linspace(0, 1, 3, device=device).expand(batch_size, -1)
            simulator.simulate(noise, ts, use_tqdm=False)
            torch.cuda.synchronize()
            del noise
            torch.cuda.empty_cache()
            return batch_size
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    return 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CelebA-64 FID from an existing checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("samples/experiment_5/fid"))
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50_000,
        help="50k is the paper protocol; 10k is enough for a debug FID",
    )
    parser.add_argument(
        "--ode-steps",
        type=int,
        default=100,
        help="Tong et al. TMLR 2024 Table 5 reports Euler FID at 100 and 1000 NFE",
    )
    parser.add_argument("--batch-size", type=int, default=0, help="0 chooses the largest batch that fits")
    parser.add_argument("--inception-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real-split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--regenerate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples < 2:
        raise ValueError("num-samples must be at least 2")
    if args.ode_steps < 1:
        raise ValueError("ode-steps must be positive")
    if args.inception_batch_size < 1:
        raise ValueError("inception-batch-size must be positive")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    from pytorch_fid.inception import InceptionV3

    cfg = load_config(args.config)
    image_size = int(cfg["data"]["image_size"])
    channels = int(cfg["unet"]["in_channels"])
    data_root = args.data_root or resolve_data_root()
    cache_path = celeba_cache_path(data_root, args.real_split, image_size)
    if not cache_path.exists():
        raise FileNotFoundError(f"missing CelebA cache {cache_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CelebA-64 FID should run on GPU")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_path = (
        args.output_dir / f"generated_uint8_n{args.num_samples}_steps{args.ode_steps}.pt"
    )
    real_stats_path = args.output_dir / f"celeba_{args.real_split}_{image_size}_stats.npz"
    result_path = args.output_dir / "fid.json"

    model = FlowModel.from_config(args.config).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    load_eval_weights(model, state, use_ema=not args.no_ema)
    batch_size = choose_batch_size(model, channels, image_size, device, args.batch_size)
    print(
        "checkpoint",
        args.checkpoint,
        "step",
        state.get("step"),
        "ema",
        not args.no_ema and state.get("ema_model") is not None,
        "batch",
        batch_size,
        "ode_steps",
        args.ode_steps,
    )

    if generated_path.exists() and not args.regenerate:
        generated = torch.load(generated_path, map_location="cpu", weights_only=False)
        if generated.shape[0] < args.num_samples:
            generated = generate_uint8(
                model,
                args.num_samples,
                image_size,
                channels,
                args.ode_steps,
                batch_size,
                device,
                args.seed,
            )
            torch.save(generated, generated_path)
        elif generated.shape[0] > args.num_samples:
            generated = generated[: args.num_samples]
    else:
        generated = generate_uint8(
            model,
            args.num_samples,
            image_size,
            channels,
            args.ode_steps,
            batch_size,
            device,
            args.seed,
        )
        torch.save(generated, generated_path)

    preview = generated[:64].float().div(255.0)
    save_image(preview, args.output_dir / "preview_8x8.png", nrow=8)

    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    if real_stats_path.exists() and not args.regenerate:
        real_npz = np.load(real_stats_path)
        real_mean, real_cov = real_npz["mu"], real_npz["sigma"]
        n_real = int(real_npz["n"])
    else:
        real_images = torch.load(cache_path, map_location="cpu", weights_only=False)["images"]
        real_features = inception_features(
            real_images, inception, args.inception_batch_size, device
        )
        real_mean, real_cov = activation_statistics(real_features)
        n_real = int(real_features.shape[0])
        np.savez(real_stats_path, mu=real_mean, sigma=real_cov, n=n_real)
        del real_images, real_features

    generated_features = inception_features(
        generated, inception, args.inception_batch_size, device
    )
    generated_mean, generated_cov = activation_statistics(generated_features)
    np.savez(
        args.output_dir / "generated_stats.npz",
        mu=generated_mean,
        sigma=generated_cov,
        n=generated_features.shape[0],
    )
    fid = frechet_distance(real_mean, real_cov, generated_mean, generated_cov)

    result = {
        "fid": fid,
        "num_generated": int(generated.shape[0]),
        "num_real": n_real,
        "real_split": args.real_split,
        "image_size": image_size,
        "ode_steps": args.ode_steps,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": int(state.get("step", -1)),
        "use_ema": not args.no_ema and state.get("ema_model") is not None,
        "inception": "pytorch-fid Inception-v3 pool3 (2048)",
        "real_cache": str(cache_path.resolve()),
        "generated_path": str(generated_path.resolve()),
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"FID {fid:.3f}")


if __name__ == "__main__":
    main()

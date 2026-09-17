"""CelebA-64 FID for an existing flow-matching checkpoint.

Protocol matches common paper reporting: 50k generated samples, pytorch-fid
Inception-v3 pool3 features, official CelebA train images at 64x64.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing as mp
import time
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
INCEPTION_PROTOCOL = "pytorch-fid Inception-v3 pool3 (2048)"
CACHE_VERSION = 2


def check_deadline(deadline_unix: float | None) -> None:
    if deadline_unix is not None and time.time() >= deadline_unix:
        raise TimeoutError("FID evaluation deadline reached")


def file_identity(path: str | Path, *, hash_contents: bool = False) -> dict[str, Any]:
    """Detect replacement/overwrite without hashing multi-gigabyte checkpoints."""
    path = Path(path).resolve()
    stat = path.stat()
    result = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if hash_contents:
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def cache_matches(path: Path, identity: dict[str, Any]) -> bool:
    metadata_path = manifest_path(path)
    if not path.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
        return metadata.get("identity") == identity and metadata.get("artifact") == file_identity(path)
    except (OSError, ValueError, TypeError):
        return False


def write_manifest(path: Path, identity: dict[str, Any], **provenance: Any) -> None:
    metadata_path = manifest_path(path)
    payload = {"identity": identity, "artifact": file_identity(path), **provenance}
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(metadata_path)


def generated_cache_identity(
    checkpoint: Path, config: Path, *, num_samples: int, ode_steps: int,
    seed: int, use_ema: bool, image_size: int, channels: int,
    batch_size: int, precision: str,
) -> dict[str, Any]:
    import torchvision
    return {
        "version": CACHE_VERSION,
        "checkpoint": file_identity(checkpoint),
        "config": file_identity(config, hash_contents=True),
        "num_samples": num_samples, "ode_steps": ode_steps,
        "solver": "euler", "seed": seed, "use_ema": use_ema,
        "image_size": image_size, "channels": channels,
        "batch_size": batch_size, "precision": precision,
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "pixel_conversion": "clamp[-1,1]; round((x+1)*127.5); uint8",
    }


def real_cache_identity(cache: Path, split: str, image_size: int) -> dict[str, Any]:
    return {
        "version": CACHE_VERSION, "cache": file_identity(cache),
        "split": split, "image_size": image_size,
        "preprocessing": "CelebA aligned RGB; center-crop178; PIL bilinear resize; uint8",
        "inception": INCEPTION_PROTOCOL,
        "feature_input": "uint8/255; pytorch-fid default resize299 and normalize_input",
    }


def validate_real_stats(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with np.load(path) as data:
        mean, covariance, n_real = data["mu"], data["sigma"], int(data["n"])
    if mean.shape != (2048,) or covariance.shape != (2048, 2048) or n_real < 2:
        raise ValueError(f"invalid real FID statistics in {path}")
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError(f"nonfinite real FID statistics in {path}")
    return mean, covariance, n_real


def adopt_legacy_real_stats(
    stats_path: Path, cache_path: Path, report_path: Path, *, split: str, image_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Explicit provenance-based adoption; does not recompute Inception features.

    This migration is for the known original CelebA178 preprocessing run. Unknown
    caches without provenance must be recomputed instead of silently adopted.
    """
    report = json.loads(report_path.read_text())
    expected = {"real_split": split, "image_size": image_size, "inception": INCEPTION_PROTOCOL}
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("legacy FID report uses a different real-data protocol")
    if report_path.resolve().parent != stats_path.resolve().parent:
        raise ValueError("legacy report must accompany the original real statistics")
    if report.get("real_cache") and Path(report["real_cache"]).resolve() != cache_path.resolve():
        raise ValueError("legacy report names a different real cache")
    if cache_path.stat().st_mtime_ns > stats_path.stat().st_mtime_ns:
        raise ValueError("real cache changed after the legacy statistics were generated")
    mean, covariance, n_real = validate_real_stats(stats_path)
    state = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)
    images = state["images"]
    if images.dtype != torch.uint8 or images.shape != (n_real, 3, image_size, image_size):
        raise ValueError("legacy real-statistics count or cached image format does not match")
    if report.get("num_real") != n_real:
        raise ValueError("legacy FID report and real-statistics counts disagree")
    del images, state
    write_manifest(
        stats_path, real_cache_identity(cache_path, split, image_size),
        provenance="adopted from known legacy run; features not numerically recomputed",
        legacy_report=file_identity(report_path, hash_contents=True),
    )
    return mean, covariance, n_real


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

    def __init__(self, flow: nn.Module, enabled: bool, deadline_unix: float | None = None):
        super().__init__()
        self.flow = flow
        self.enabled = enabled
        self.deadline_unix = deadline_unix

    def forward(self, xt: Tensor, t: Tensor) -> Tensor:
        check_deadline(self.deadline_unix)
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
    deadline_unix: float | None = None,
) -> Tensor:
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if ode_steps < 1:
        raise ValueError("ode_steps must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    flow = _AutocastFlow(model, enabled=device.type == "cuda", deadline_unix=deadline_unix)
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
    deadline_unix: float | None = None,
) -> np.ndarray:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must be uint8 NCHW RGB")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    inception.eval()
    chunks: list[Tensor] = []
    for start in tqdm(range(0, images.shape[0], batch_size), desc="inception"):
        check_deadline(deadline_unix)
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
    deadline_unix: float | None = None,
) -> int:
    if requested > 0:
        return requested
    if device.type != "cuda":
        return 32
    flow = _AutocastFlow(model, enabled=True, deadline_unix=deadline_unix)
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


def _frechet_worker(queue, arrays) -> None:
    try:
        # Avoid expensive BLAS oversubscription on large RunPod CPU hosts.
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            value = frechet_distance(*arrays)
        else:
            with threadpool_limits(limits=4):
                value = frechet_distance(*arrays)
        queue.put((True, value))
    except Exception as error:
        queue.put((False, f"{type(error).__name__}: {error}"))


def frechet_with_deadline(
    mean_real: np.ndarray, cov_real: np.ndarray,
    mean_fake: np.ndarray, cov_fake: np.ndarray,
    deadline_unix: float | None = None,
) -> float:
    if deadline_unix is None:
        return frechet_distance(mean_real, cov_real, mean_fake, cov_fake)
    check_deadline(deadline_unix)
    context = mp.get_context("spawn")
    queue = context.Queue()
    worker = context.Process(
        target=_frechet_worker, args=(queue, (mean_real, cov_real, mean_fake, cov_fake)),
    )
    worker.start()
    try:
        worker.join(timeout=max(0, deadline_unix - time.time()))
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
            raise TimeoutError("FID covariance calculation exceeded evaluation deadline")
        if worker.exitcode != 0:
            raise RuntimeError(f"FID covariance worker exited with code {worker.exitcode}")
        success, value = queue.get(timeout=5)
        if not success:
            raise RuntimeError(value)
        return float(value)
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
        queue.close()


def evaluate_checkpoint(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    num_samples: int = 2048,
    ode_steps: int = 32,
    real_stats_path: str | Path | None = None,
    legacy_real_stats_report: str | Path | None = None,
    data_root: str | Path | None = None,
    real_split: str = "train",
    seed: int = 42,
    use_ema: bool = True,
    batch_size: int = 0,
    inception_batch_size: int = 128,
    deadline_unix: float | None = None,
    regenerate: bool = False,
    regenerate_real: bool = False,
) -> dict[str, Any]:
    """Screen with 2,048 images; use num_samples=10_000 for matched final FID.

    All cached artifacts require matching manifests. ``legacy_real_stats_report``
    explicitly adopts known original statistics after checking their provenance.
    A deadline interrupts at each ODE forward / Inception batch and bounds the
    separate CPU covariance worker. A currently executing GPU kernel must finish.
    """
    if num_samples < 2 or ode_steps < 1 or inception_batch_size < 1:
        raise ValueError("require num_samples>=2, ode_steps>=1, inception_batch_size>=1")
    if real_split not in {"train", "val", "test"}:
        raise ValueError("real_split must be train, val or test")
    check_deadline(deadline_unix)
    config_path, checkpoint_path, output_dir = map(Path, (config_path, checkpoint_path, output_dir))
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    from pytorch_fid.inception import InceptionV3

    cfg = load_config(config_path)
    image_size = int(cfg["data"]["image_size"])
    channels = int(cfg.get("unet", cfg.get("adm_unet", {})).get("in_channels", 3))
    cache_path = celeba_cache_path(data_root or resolve_data_root(), real_split, image_size)
    if not cache_path.exists():
        raise FileNotFoundError(f"missing CelebA cache {cache_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CelebA-64 FID should run on GPU")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_path = output_dir / f"generated_uint8_n{num_samples}_steps{ode_steps}.pt"
    real_stats_path = Path(real_stats_path) if real_stats_path else output_dir / f"celeba_{real_split}_{image_size}_stats.npz"
    real_identity = real_cache_identity(cache_path, real_split, image_size)
    real_stats = None
    if not regenerate_real and cache_matches(real_stats_path, real_identity):
        real_stats = validate_real_stats(real_stats_path)
    elif not regenerate_real and real_stats_path.exists() and legacy_real_stats_report is not None and not manifest_path(real_stats_path).exists():
        real_stats = adopt_legacy_real_stats(
            real_stats_path, cache_path, Path(legacy_real_stats_report),
            split=real_split, image_size=image_size,
        )
    elif real_stats_path.exists() and not regenerate_real:
        raise ValueError(
            f"real statistics have missing/mismatched provenance: {real_stats_path}; "
            "supply legacy_real_stats_report for the known original cache or regenerate_real=True"
        )

    # Keep optimizer tensors off GPU. mmap avoids materializing the complete
    # checkpoint in host RAM; release its tensor references before generation.
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    checkpoint_step = int(state.get("step", -1))
    actual_ema = use_ema and state.get("ema_model") is not None
    model = FlowModel.from_config(config_path)
    load_eval_weights(model, state, use_ema=use_ema)
    del state
    gc.collect()
    check_deadline(deadline_unix)
    model = model.to(device, memory_format=torch.channels_last)
    try:
        batch_size = choose_batch_size(model, channels, image_size, device, batch_size, deadline_unix)
        identity = generated_cache_identity(
            checkpoint_path, config_path, num_samples=num_samples, ode_steps=ode_steps,
            seed=seed, use_ema=actual_ema, image_size=image_size, channels=channels,
            batch_size=batch_size, precision="bf16-forward/fp32-Euler/channels-last",
        )
        print(f"checkpoint {checkpoint_path} step {checkpoint_step} ema {actual_ema} batch {batch_size} Euler {ode_steps}", flush=True)
        if not regenerate and cache_matches(generated_path, identity):
            generated = torch.load(generated_path, map_location="cpu", weights_only=True, mmap=True)
            if generated.dtype != torch.uint8 or generated.shape != (num_samples, channels, image_size, image_size):
                raise ValueError("generated cache does not match its declared shape or dtype")
        else:
            generated = generate_uint8(
                model, num_samples, image_size, channels, ode_steps, batch_size,
                device, seed, deadline_unix,
            )
            check_deadline(deadline_unix)
            torch.save(generated, generated_path)
            write_manifest(generated_path, identity)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    check_deadline(deadline_unix)
    save_image(generated[:64].float().div(255.0), output_dir / "preview_8x8.png", nrow=8)
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    try:
        if real_stats is None:
            real_images = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)["images"]
            real_features = inception_features(real_images, inception, inception_batch_size, device, deadline_unix)
            real_mean, real_cov = activation_statistics(real_features)
            n_real = int(real_features.shape[0])
            check_deadline(deadline_unix)
            real_stats_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(real_stats_path, mu=real_mean, sigma=real_cov, n=n_real)
            write_manifest(real_stats_path, real_identity, provenance="computed from identified real cache")
            del real_images, real_features
        else:
            real_mean, real_cov, n_real = real_stats
        generated_features = inception_features(generated, inception, inception_batch_size, device, deadline_unix)
    finally:
        del inception
        gc.collect()
        torch.cuda.empty_cache()
    check_deadline(deadline_unix)
    generated_mean, generated_cov = activation_statistics(generated_features)
    np.savez(output_dir / "generated_stats.npz", mu=generated_mean, sigma=generated_cov, n=num_samples)
    fid = frechet_with_deadline(real_mean, real_cov, generated_mean, generated_cov, deadline_unix)
    result = {
        "fid": fid, "num_generated": num_samples, "num_real": n_real,
        "real_split": real_split, "image_size": image_size, "ode_steps": ode_steps,
        "seed": seed, "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_step": checkpoint_step, "use_ema": actual_ema,
        "inception": INCEPTION_PROTOCOL, "real_cache": str(cache_path.resolve()),
        "real_stats": str(real_stats_path.resolve()), "generated_path": str(generated_path.resolve()),
        "generated_identity": identity, "real_identity": real_identity,
        "evaluation_kind": "screening" if num_samples < 10_000 else "final",
    }
    (output_dir / "fid.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CelebA-64 FID with validated artifact caches")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("samples/experiment_5/fid"))
    parser.add_argument("--num-samples", type=int, default=50_000, help="2048 screening; 10000 matched debug FID; 50000 paper protocol")
    parser.add_argument("--ode-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=0, help="0 chooses largest batch that fits")
    parser.add_argument("--inception-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real-split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--real-stats", type=Path, help="reuse real statistics with matching provenance")
    parser.add_argument("--legacy-real-stats-report", type=Path, help="explicit provenance-based adoption of known original real stats")
    parser.add_argument("--deadline-unix", type=float, help="absolute UTC Unix timestamp")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--regenerate", action="store_true", help="regenerate model samples")
    parser.add_argument("--regenerate-real", action="store_true", help="recompute real statistics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_checkpoint(
        args.config, args.checkpoint, args.output_dir,
        num_samples=args.num_samples, ode_steps=args.ode_steps,
        real_stats_path=args.real_stats, legacy_real_stats_report=args.legacy_real_stats_report,
        data_root=args.data_root, real_split=args.real_split, seed=args.seed,
        use_ema=not args.no_ema, batch_size=args.batch_size,
        inception_batch_size=args.inception_batch_size, deadline_unix=args.deadline_unix,
        regenerate=args.regenerate, regenerate_real=args.regenerate_real,
    )


if __name__ == "__main__":
    main()

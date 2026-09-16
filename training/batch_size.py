from copy import deepcopy

import torch

from training.trainer import FlowTrainer


def largest_fitting_batch(fits, low: int, high: int) -> int:
    """Return the largest integer in [low, high] for which fits(n) is True."""
    if low < 1:
        raise ValueError("low must be positive")
    if high < low:
        raise ValueError("high must be >= low")
    best: int | None = None
    lo, hi = low, high
    while lo <= hi:
        mid = (lo + hi) // 2
        if fits(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        raise RuntimeError(f"no batch size in [{low}, {high}] fits")
    return best


def align_batch_size(batch_size: int, multiple: int = 8) -> int:
    if multiple < 1:
        raise ValueError("multiple must be positive")
    aligned = batch_size - (batch_size % multiple)
    if aligned < 1:
        raise RuntimeError(f"batch size {batch_size} is smaller than alignment {multiple}")
    return aligned


def memory_within_fraction(peak_bytes: int, total_bytes: int, max_fraction: float) -> bool:
    if not 0 < max_fraction <= 1:
        raise ValueError("max_fraction must be in (0, 1]")
    if total_bytes < 1:
        raise ValueError("total_bytes must be positive")
    return peak_bytes <= int(max_fraction * total_bytes)


def probe_train_batch_size(
    trainer: FlowTrainer,
    *,
    device: torch.device,
    min_batch: int = 16,
    max_batch: int = 384,
    steps: int = 3,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    max_grad_norm: float | None = 1.0,
    ema_decay: float = 0.999,
    max_memory_fraction: float = 0.88,
) -> int:
    """Largest train batch that completes AdamW+EMA steps without crowding the GPU."""
    if device.type != "cuda":
        return min_batch
    if min_batch < 1 or max_batch < min_batch:
        raise ValueError("invalid batch search range")

    model = trainer.model
    model.to(device)
    total_bytes = torch.cuda.get_device_properties(device).total_memory

    def fits(batch_size: int) -> bool:
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        ema = deepcopy(model).eval()
        ema.requires_grad_(False)
        try:
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                loss = trainer.get_train_loss(batch_size=batch_size)
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
                with torch.no_grad():
                    for ema_param, param in zip(ema.parameters(), model.parameters()):
                        ema_param.lerp_(param, 1 - ema_decay)
            if trainer.val_path is not None:
                trainer.model.eval()
                with torch.no_grad():
                    trainer.get_val_loss(batch_size=batch_size)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated(device)
            ok = memory_within_fraction(peak, total_bytes, max_memory_fraction)
            print(
                f"batch {batch_size}: peak {peak / 1024**3:.2f} GiB / "
                f"{total_bytes / 1024**3:.2f} GiB ({'ok' if ok else 'too close'})"
            )
            return ok
        except torch.cuda.OutOfMemoryError:
            print(f"batch {batch_size}: OOM")
            return False
        finally:
            del opt, ema
            model.zero_grad(set_to_none=True)
            trainer.model.train()
            torch.cuda.empty_cache()

    found = largest_fitting_batch(fits, min_batch, max_batch)
    return align_batch_size(found)

import json
import math
import os
import re
import shutil
import signal
import tempfile
import threading
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import wandb
from matplotlib import pyplot as plt
from torch import Tensor
from torchvision.utils import make_grid
from tqdm import tqdm

from models.flow import FlowModel
from sampling.ode import EulerSimulator, FlowODE
from training.path import GaussianConditionalProbabilityPath

MiB = 1024**2
_COSINE_SCHEDULE = "cosine"
_NO_SCHEDULE = {None, "", "none", "constant"}


@contextmanager
def _deferred_stop_signals(enabled: bool):
    """Finish an in-flight optimizer/EMA update before honoring a stop signal."""
    state = {"signal": None}
    handlers = {}
    if enabled:
        if threading.current_thread() is not threading.main_thread():
            raise ValueError("handle_signals requires training on the main thread")

        def request_stop(signum, _frame):
            state["signal"] = signum

        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, request_stop)
    try:
        yield state
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


@contextmanager
def _isolated_seed(seed: int | None, device: torch.device):
    if seed is None:
        yield
        return
    devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        yield


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Resolve an immutable checkpoint pointer, falling back to a legacy file.

    Remote FUSE mounts can return stale bytes after replacing an existing large
    file. Immutable versions avoid that; a small JSON pointer is authoritative.
    """
    logical = Path(path)
    pointer = logical.with_name(logical.name + ".pointer.json")
    if not pointer.exists():
        return logical
    metadata = json.loads(pointer.read_text())
    filename = metadata["filename"]
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError(f"checkpoint pointer must contain a local basename: {pointer}")
    resolved = logical.parent / filename
    size = resolved.stat().st_size
    if size != metadata["size_bytes"]:
        raise ValueError(f"checkpoint size does not match pointer: {resolved}")
    return resolved


def model_size_b(model: nn.Module) -> int:
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size


class Trainer(ABC):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, **kwargs) -> Tensor:
        pass

    def get_val_loss(self, **kwargs) -> Tensor | None:
        return None

    def plot_trajectory(self, step: int, **kwargs) -> Path | None:
        return None

    def get_optimizer(
        self,
        lr: float,
        optimizer_name: str = "adam",
        weight_decay: float = 0.0,
        optimizer_betas: tuple[float, float] | None = None,
        optimizer_fused: bool = False,
    ) -> torch.optim.Optimizer:
        optimizers = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
        }
        if optimizer_name not in optimizers:
            raise ValueError(
                f"unknown optimizer {optimizer_name!r}, expected one of {sorted(optimizers)}"
            )
        options = {"betas": tuple(optimizer_betas)} if optimizer_betas is not None else {}
        if optimizer_fused:
            options["fused"] = True
        return optimizers[optimizer_name](
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **options,
        )

    def get_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        num_steps: int,
        lr: float,
        lr_milestones: list[int] | None = None,
        lr_gamma: float = 0.1,
        lr_schedule: str | None = None,
        lr_warmup_steps: int = 0,
        lr_min_ratio: float = 0.01,
        scheduler_num_steps: int | None = None,
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        if lr_milestones and lr_schedule == _COSINE_SCHEDULE:
            raise ValueError("use lr_milestones or lr_schedule='cosine', not both")
        if lr_warmup_steps < 0:
            raise ValueError("lr_warmup_steps must be non-negative")
        if not 0 <= lr_min_ratio < 1:
            raise ValueError("lr_min_ratio must be in [0, 1)")
        if scheduler_num_steps is not None and scheduler_num_steps < num_steps:
            raise ValueError("scheduler_num_steps must be >= num_steps")
        horizon = num_steps if scheduler_num_steps is None else scheduler_num_steps
        if lr_milestones:
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=sorted(lr_milestones),
                gamma=lr_gamma,
            )
        if lr_schedule in _NO_SCHEDULE:
            return None
        if lr_schedule != _COSINE_SCHEDULE:
            raise ValueError(
                f"unknown lr_schedule {lr_schedule!r}, expected one of "
                f"{sorted(s for s in _NO_SCHEDULE if isinstance(s, str)) + [_COSINE_SCHEDULE]}"
            )
        if lr_warmup_steps >= horizon:
            raise ValueError("lr_warmup_steps must be smaller than the scheduler horizon")
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(horizon - lr_warmup_steps, 1),
            eta_min=lr * lr_min_ratio,
        )
        if lr_warmup_steps == 0:
            return cosine
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=lr_warmup_steps,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[lr_warmup_steps],
        )

    def train(
        self,
        num_steps: int,
        device: torch.device,
        lr: float = 1e-3,
        optimizer_name: str = "adam",
        weight_decay: float = 0.0,
        max_grad_norm: float | None = None,
        lr_milestones: list[int] | None = None,
        lr_gamma: float = 0.1,
        lr_schedule: str | None = None,
        lr_warmup_steps: int = 0,
        lr_min_ratio: float = 0.01,
        scheduler_num_steps: int | None = None,
        abort_train_loss: float | None = None,
        ema_decay: float | None = None,
        ckpt_path: str | Path | None = None,
        best_ckpt_path: str | Path | None = None,
        checkpoint_every: int = 50,
        val_every: int = 50,
        val_batches: int = 8,
        plot_every: int = 50,
        n_plot_images: int = 10,
        n_plot_steps: int = 10,
        samples_dir: str | Path = "samples",
        show_plots: bool = True,
        resume_from: str | Path | None = None,
        wandb_run: Any | None = None,
        mixed_precision: str | None = None,
        optimizer_betas: tuple[float, float] | None = None,
        optimizer_fused: bool = False,
        ema_warmup: bool = False,
        validation_seed: int | None = None,
        plot_seed: int | None = None,
        n_sampling_steps: int | None = None,
        deadline_timestamp: float | None = None,
        handle_signals: bool = False,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Train with optional reproducible evaluation and a wall-clock deadline.

        ``step`` and history count completed optimizer/EMA/scheduler updates.
        A deadline or deferred SIGINT/SIGTERM ends between updates, then saves.
        BF16 applies to forward passes; FlowTrainer still reduces its MSE in FP32.
        """
        device = torch.device(device)
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        if val_every > 0 and val_batches < 1:
            raise ValueError("val_batches must be positive when validation is enabled")
        if lr_milestones and any(step < 1 for step in lr_milestones):
            raise ValueError("lr milestones must be positive")
        if lr_gamma <= 0:
            raise ValueError("lr_gamma must be positive")
        if ema_decay is not None and not 0 <= ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if abort_train_loss is not None and abort_train_loss <= 0:
            raise ValueError("abort_train_loss must be positive")
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if mixed_precision not in {None, "bf16"}:
            raise ValueError("mixed_precision must be None or 'bf16'")
        if mixed_precision == "bf16" and device.type != "cuda":
            raise ValueError("mixed_precision='bf16' requires CUDA")
        if deadline_timestamp is not None and not math.isfinite(deadline_timestamp):
            raise ValueError("deadline_timestamp must be finite")
        if n_sampling_steps is not None and n_sampling_steps < 1:
            raise ValueError("n_sampling_steps must be positive")

        size_b = model_size_b(self.model)
        print(f"Training model with size: {size_b / MiB:.3f} MiB")
        self.model.to(device)
        optimizer_options = {}
        if optimizer_betas is not None:
            optimizer_options["optimizer_betas"] = optimizer_betas
        if optimizer_fused:
            optimizer_options["optimizer_fused"] = True
        opt = self.get_optimizer(lr, optimizer_name, weight_decay, **optimizer_options)
        scheduler = self.get_scheduler(
            opt,
            num_steps=num_steps,
            lr=lr,
            lr_milestones=lr_milestones,
            lr_gamma=lr_gamma,
            lr_schedule=lr_schedule,
            lr_warmup_steps=lr_warmup_steps,
            lr_min_ratio=lr_min_ratio,
            scheduler_num_steps=scheduler_num_steps,
        )
        ema_model = deepcopy(self.model).eval() if ema_decay is not None else None
        if ema_model is not None:
            ema_model.requires_grad_(False)
        start_step = 0
        train_losses: list[Tensor] = []
        val_losses: list[Tensor] = []
        val_steps: list[int] = []
        self.last_stop_reason = None
        self.completed_steps = 0

        if resume_from is not None:
            # Keep serialized model copies off GPU and release them after restoring.
            state = torch.load(resolve_checkpoint_path(resume_from), map_location="cpu", weights_only=False)
            self.model.load_state_dict(state["model"])
            opt.load_state_dict(state["optimizer"])
            if scheduler is not None and state.get("scheduler") is not None:
                scheduler.load_state_dict(state["scheduler"])
            if ema_model is not None:
                ema_model.load_state_dict(state.get("ema_model") or state["model"])
            start_step = int(state["step"])
            train_losses = [
                value.detach().cpu() for value in state["history"]["train"][:start_step]
            ]
            val_losses = [
                value.detach().cpu() for value in state["history"].get("val", [])
            ]
            val_steps = [int(step) for step in state["history"].get("val_steps", [])]
            rng = state.get("rng_state")
            if rng is not None:
                torch.random.set_rng_state(rng["cpu"].cpu())
                if device.type == "cuda" and rng.get("cuda") is not None:
                    torch.cuda.set_rng_state_all([value.cpu() for value in rng["cuda"]])
            del state
            if start_step > num_steps:
                raise ValueError(
                    f"checkpoint step {start_step} exceeds requested num_steps {num_steps}"
                )
            print(f"resumed {resume_from} at step {start_step}")
        self.completed_steps = start_step

        best_train_loss = min((value.item() for value in train_losses), default=float("inf"))
        best_val_loss = min((value.item() for value in val_losses), default=float("inf"))

        def forward_context():
            return (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if mixed_precision == "bf16" else nullcontext()
            )

        def history() -> dict[str, Tensor]:
            train = torch.stack(train_losses).cpu() if train_losses else torch.empty(0)
            result = {"train": train}
            if val_losses:
                result["val"] = torch.stack(val_losses).cpu()
                result["val_steps"] = torch.tensor(val_steps, dtype=torch.long)
            return result

        @contextmanager
        def evaluation_model():
            training_model = self.model
            if ema_model is not None:
                self.model = ema_model
            try:
                yield
            finally:
                self.model = training_model

        def save_checkpoint(step: int, path: str | Path | None = ckpt_path) -> None:
            if path is None:
                return
            output = Path(path)
            output.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "model": self.model.state_dict(),
                "ema_model": ema_model.state_dict() if ema_model is not None else None,
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "history": history(),
                "step": step,
                "num_steps": num_steps,
                "lr": lr,
                "optimizer_name": optimizer_name,
                "optimizer_betas": optimizer_betas,
                "optimizer_fused": optimizer_fused,
                "weight_decay": weight_decay,
                "max_grad_norm": max_grad_norm,
                "lr_milestones": lr_milestones,
                "lr_gamma": lr_gamma,
                "lr_schedule": lr_schedule,
                "lr_warmup_steps": lr_warmup_steps,
                "lr_min_ratio": lr_min_ratio,
                "scheduler_num_steps": scheduler_num_steps,
                "abort_train_loss": abort_train_loss,
                "ema_decay": ema_decay,
                "ema_warmup": ema_warmup,
                "mixed_precision": mixed_precision,
                "validation_seed": validation_seed,
                "plot_seed": plot_seed,
                "n_sampling_steps": n_sampling_steps,
                "deadline_timestamp": deadline_timestamp,
                "stop_reason": self.last_stop_reason,
                "rng_state": {
                    "cpu": torch.random.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                },
                "train_kwargs": kwargs,
            }
            # Stage the torch zip locally, then publish a complete destination copy.
            # A failed copy must not truncate the previous usable checkpoint.
            tmp_dir = Path(os.getenv("DIFFUSION_CHECKPOINT_TMPDIR", "/tmp/diffusion-ckpts"))
            tmp_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=tmp_dir, prefix=f"{output.name}.", suffix=".partial", delete=False
            ) as temporary:
                tmp_path = Path(temporary.name)
            destination_tmp = None
            try:
                torch.save(payload, tmp_path)
                if os.getenv("DIFFUSION_IMMUTABLE_CHECKPOINTS", "").lower() in {"1", "true", "yes"}:
                    version = output.with_name(
                        f"{output.stem}.step{step:09d}.{uuid.uuid4().hex}{output.suffix}"
                    )
                    # Never replace an existing large FUSE object. Even rename
                    # can leave stale cached bytes under the destination path.
                    shutil.copyfile(tmp_path, version)
                    verified = torch.load(version, map_location="cpu", weights_only=False, mmap=True)
                    if int(verified["step"]) != step or len(verified["history"]["train"]) != step:
                        raise RuntimeError(f"published checkpoint failed step verification: {version}")
                    del verified
                    pointer = output.with_name(output.name + ".pointer.json")
                    metadata = {
                        "filename": version.name,
                        "step": step,
                        "size_bytes": version.stat().st_size,
                        "created_at": time.time(),
                    }
                    # Directly overwrite the tiny pointer: replacing its inode
                    # would reproduce the stale-name problem on these mounts.
                    with pointer.open("w") as stream:
                        json.dump(metadata, stream)
                        stream.write("\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    keep = int(os.getenv("DIFFUSION_CHECKPOINT_KEEP", "0"))
                    if keep > 0:
                        pattern = re.compile(
                            re.escape(output.stem) + r"\.step[0-9]{9}\.[0-9a-f]{32}" + re.escape(output.suffix)
                        )
                        versions = [
                            path for path in output.parent.glob(f"{output.stem}.step*{output.suffix}")
                            if pattern.fullmatch(path.name) and path != version
                        ]
                        versions.sort(key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True)
                        for obsolete in versions[max(0, keep - 1):]:
                            try:
                                obsolete.unlink(missing_ok=True)
                            except OSError as exc:
                                print(f"checkpoint cleanup skipped {obsolete}: {exc}")
                    return
                with tempfile.NamedTemporaryFile(
                    dir=output.parent, prefix=f".{output.name}.", suffix=".partial", delete=False
                ) as destination:
                    destination_tmp = Path(destination.name)
                shutil.copyfile(tmp_path, destination_tmp)
                os.replace(destination_tmp, output)
            finally:
                tmp_path.unlink(missing_ok=True)
                if destination_tmp is not None:
                    destination_tmp.unlink(missing_ok=True)

        last_step = start_step
        with _deferred_stop_signals(handle_signals) as stop_state:
            def should_stop() -> bool:
                if stop_state["signal"] is not None:
                    self.last_stop_reason = f"signal:{signal.Signals(stop_state['signal']).name}"
                    return True
                if deadline_timestamp is not None and time.time() >= deadline_timestamp:
                    self.last_stop_reason = "deadline"
                    return True
                return False

            def validation_loss() -> Tensor | None:
                values = []
                with evaluation_model(), _isolated_seed(validation_seed, device):
                    self.model.eval()
                    for _ in range(val_batches):
                        if should_stop():
                            return None  # Never compare an incomplete validation batch set.
                        with forward_context():
                            value = self.get_val_loss(**kwargs)
                        if value is not None:
                            values.append(value.float())
                return torch.stack(values).mean() if values else None

            try:
                if start_step == 0 and val_every > 0 and not should_stop():
                    initial_val = validation_loss()
                    if initial_val is not None:
                        val_losses.append(initial_val.detach().cpu())
                        val_steps.append(0)
                        best_val_loss = initial_val.item()
                        save_checkpoint(0, best_ckpt_path)
                        if wandb_run is not None:
                            wandb_run.log(
                                {"loss/val": initial_val.item(), "loss/val_best": best_val_loss},
                                step=0,
                            )

                pbar = tqdm(range(start_step + 1, num_steps + 1))
                for step in pbar:
                    if should_stop():
                        break
                    self.model.train()
                    opt.zero_grad()
                    with forward_context():
                        loss = self.get_train_loss(**kwargs)
                    loss_value = float(loss.item())
                    desc = f"Step {step}, train: {loss_value:.3f}"
                    log_data: dict[str, Any] = {
                        "loss/train": loss_value,
                        "learning_rate": opt.param_groups[0]["lr"],
                    }
                    exploded = not math.isfinite(loss_value) or (
                        abort_train_loss is not None and loss_value > abort_train_loss
                    )
                    if exploded:
                        self.last_stop_reason = "nonfinite_loss" if not math.isfinite(loss_value) else "loss_limit"
                        log_data["training/aborted"] = True
                        if wandb_run is not None:
                            wandb_run.log(log_data, step=step)
                        print(f"aborting: train loss {loss_value} at uncompleted step {step}")
                        break
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm if max_grad_norm is not None else float("inf")
                    )
                    grad_norm_value = float(grad_norm.item())
                    log_data["grad/norm"] = grad_norm_value
                    log_data["grad/clipped"] = float(
                        max_grad_norm is not None and grad_norm_value > max_grad_norm
                    )
                    if not math.isfinite(grad_norm_value):
                        self.last_stop_reason = "nonfinite_gradient"
                        log_data["training/aborted"] = True
                        if wandb_run is not None:
                            wandb_run.log(log_data, step=step)
                        print(f"aborting: non-finite gradient at uncompleted step {step}")
                        break
                    opt.step()
                    if ema_model is not None:
                        effective_decay = min(ema_decay, (1 + step) / (10 + step)) if ema_warmup else ema_decay
                        with torch.no_grad():
                            for ema_param, param in zip(ema_model.parameters(), self.model.parameters()):
                                ema_param.lerp_(param, 1 - effective_decay)
                            for ema_buffer, buffer in zip(ema_model.buffers(), self.model.buffers()):
                                ema_buffer.copy_(buffer)
                        log_data["ema/decay"] = effective_decay
                    if scheduler is not None:
                        scheduler.step()
                    # Commit counters only after all stateful pieces finish their update.
                    last_step = step
                    self.completed_steps = step
                    train_losses.append(loss.detach().cpu())
                    best_train_loss = min(best_train_loss, loss_value)
                    log_data["loss/train_best"] = best_train_loss
                    log_data["learning_rate"] = opt.param_groups[0]["lr"]
                    if not should_stop() and val_every > 0 and step % val_every == 0:
                        val = validation_loss()
                        if val is not None:
                            val_losses.append(val.detach().cpu())
                            val_steps.append(step)
                            is_best = val.item() < best_val_loss
                            best_val_loss = min(best_val_loss, val.item())
                            if is_best:
                                save_checkpoint(step, best_ckpt_path)
                            desc += f", val: {val.item():.3f}"
                            log_data["loss/val"] = val.item()
                            log_data["loss/val_best"] = best_val_loss

                    if not should_stop() and plot_every > 0 and step % plot_every == 0:
                        plot_options = {}
                        if n_sampling_steps is not None:
                            plot_options["n_sampling_steps"] = n_sampling_steps
                        with evaluation_model(), _isolated_seed(plot_seed, device), forward_context():
                            self.model.eval()
                            trajectory_path = self.plot_trajectory(
                                step=step,
                                n_images=n_plot_images,
                                n_steps=n_plot_steps,
                                samples_dir=samples_dir,
                                show=show_plots,
                                **plot_options,
                            )
                        if wandb_run is not None and trajectory_path is not None:
                            log_data["samples/trajectory"] = wandb.Image(
                                str(trajectory_path), caption=f"Step {step}: noise to image"
                            )

                    if checkpoint_every > 0 and step % checkpoint_every == 0 and not should_stop():
                        save_checkpoint(step)
                    if wandb_run is not None:
                        wandb_run.log(log_data, step=step)
                    pbar.set_description(desc)
                self.model.eval()
                save_checkpoint(last_step)
            except KeyboardInterrupt:
                # Legacy callers can retain immediate KeyboardInterrupt behavior;
                # campaign callers use handle_signals=True for complete update safety.
                self.last_stop_reason = "keyboard_interrupt"
                save_checkpoint(last_step)
                print(f"training interrupted; checkpoint saved at completed step {last_step}")
                raise
        if self.last_stop_reason is not None:
            print(f"training stopped ({self.last_stop_reason}) at completed step {last_step}")
        if ckpt_path is not None:
            print(f"saved {ckpt_path}")
        return history()


class FlowTrainer(Trainer):
    """Unconditional CFM. Lab CFGTrainer needs class labels; our UNet is unconditional."""

    def __init__(
        self,
        path: GaussianConditionalProbabilityPath,
        model: FlowModel,
        val_path: GaussianConditionalProbabilityPath | None = None,
    ):
        super().__init__(model)
        self.path = path
        self.val_path = val_path

    def _cfm_loss(self, path: GaussianConditionalProbabilityPath, batch_size: int) -> Tensor:
        z, _ = path.p_data.sample(batch_size)
        t = torch.rand(batch_size, 1, 1, 1, device=z.device)
        x, u_ref = path.sample_conditional_flow(z, t)
        u_pred = self.model(x, t.reshape(batch_size))
        return torch.mean((u_pred.float() - u_ref.float()) ** 2)

    def get_train_loss(self, batch_size: int) -> Tensor:
        return self._cfm_loss(self.path, batch_size)

    @torch.no_grad()
    def get_val_loss(self, batch_size: int) -> Tensor | None:
        if self.val_path is None:
            return None
        return self._cfm_loss(self.val_path, batch_size)

    @torch.no_grad()
    def plot_trajectory(
        self,
        step: int,
        n_images: int = 10,
        n_steps: int = 10,
        samples_dir: str | Path = "samples",
        show: bool = True,
        n_sampling_steps: int | None = None,
    ) -> Path:
        if n_steps < 2:
            raise ValueError("n_steps must include at least the initial and final frame")
        sampling_steps = n_steps - 1 if n_sampling_steps is None else n_sampling_steps
        if sampling_steps < 1:
            raise ValueError("n_sampling_steps must be positive")
        device = next(self.model.parameters()).device
        x0, _ = self.path.p_simple.sample(n_images)
        ts = (
            torch.linspace(0, 1, sampling_steps + 1, device=device)
            .view(1, -1, 1, 1, 1)
            .expand(n_images, -1, 1, 1, 1)
        )
        traj = EulerSimulator(FlowODE(self.model)).simulate_with_trajectory(
            x0, ts, use_tqdm=False
        )
        # Integrate at sampling resolution, then select only the displayed frames.
        frame_indices = torch.linspace(0, sampling_steps, n_steps, device=device).round().long()
        traj = traj[:, frame_indices]
        # (B, T, C, H, W) -> one row per image, columns are time
        frames = traj.reshape(n_images * n_steps, *traj.shape[2:])
        grid = make_grid(frames, nrow=n_steps, normalize=True, value_range=(-1, 1))
        fig, ax = plt.subplots(figsize=(n_steps, n_images))
        image = grid.permute(1, 2, 0).cpu().numpy()
        if image.shape[-1] == 1:
            ax.imshow(image.squeeze(-1), cmap="gray")
        else:
            ax.imshow(image.clip(0, 1))
        ax.axis("off")
        ax.set_title(f"step {step}: {n_images} trajectories, {sampling_steps} Euler updates (t=0 → 1)")
        fig.tight_layout()
        out = Path(samples_dir) / f"traj_step_{step:06d}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return out

from abc import ABC, abstractmethod
from contextlib import contextmanager
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
    ) -> torch.optim.Optimizer:
        optimizers = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
        }
        if optimizer_name not in optimizers:
            raise ValueError(
                f"unknown optimizer {optimizer_name!r}, expected one of {sorted(optimizers)}"
            )
        return optimizers[optimizer_name](
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
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
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        if lr_milestones and lr_schedule == _COSINE_SCHEDULE:
            raise ValueError("use lr_milestones or lr_schedule='cosine', not both")
        if lr_warmup_steps < 0:
            raise ValueError("lr_warmup_steps must be non-negative")
        if not 0 <= lr_min_ratio < 1:
            raise ValueError("lr_min_ratio must be in [0, 1)")
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
        if lr_warmup_steps >= num_steps:
            raise ValueError("lr_warmup_steps must be smaller than num_steps")
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(num_steps - lr_warmup_steps, 1),
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
        **kwargs,
    ) -> dict[str, Tensor]:
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

        size_b = model_size_b(self.model)
        print(f"Training model with size: {size_b / MiB:.3f} MiB")

        self.model.to(device)
        opt = self.get_optimizer(lr, optimizer_name, weight_decay)
        scheduler = self.get_scheduler(
            opt,
            num_steps=num_steps,
            lr=lr,
            lr_milestones=lr_milestones,
            lr_gamma=lr_gamma,
            lr_schedule=lr_schedule,
            lr_warmup_steps=lr_warmup_steps,
            lr_min_ratio=lr_min_ratio,
        )
        ema_model = deepcopy(self.model).eval() if ema_decay is not None else None
        if ema_model is not None:
            ema_model.requires_grad_(False)
        start_step = 0
        train_losses: list[Tensor] = []
        val_losses: list[Tensor] = []
        val_steps: list[int] = []

        if resume_from is not None:
            state = torch.load(resume_from, map_location=device, weights_only=False)
            self.model.load_state_dict(state["model"])
            opt.load_state_dict(state["optimizer"])
            if scheduler is not None and state.get("scheduler") is not None:
                scheduler.load_state_dict(state["scheduler"])
            if ema_model is not None and state.get("ema_model") is not None:
                ema_model.load_state_dict(state["ema_model"])
            start_step = int(state["step"])
            train_losses = [
                value.detach().cpu() for value in state["history"]["train"]
            ]
            val_losses = [
                value.detach().cpu() for value in state["history"].get("val", [])
            ]
            val_steps = [int(step) for step in state["history"].get("val_steps", [])]
            if start_step > num_steps:
                raise ValueError(
                    f"checkpoint step {start_step} exceeds requested num_steps {num_steps}"
                )
            print(f"resumed {resume_from} at step {start_step}")

        best_train_loss = min((value.item() for value in train_losses), default=float("inf"))
        best_val_loss = min((value.item() for value in val_losses), default=float("inf"))

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
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "ema_model": ema_model.state_dict() if ema_model is not None else None,
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "history": history(),
                    "step": step,
                    "num_steps": num_steps,
                    "lr": lr,
                    "optimizer_name": optimizer_name,
                    "weight_decay": weight_decay,
                    "max_grad_norm": max_grad_norm,
                    "lr_milestones": lr_milestones,
                    "lr_gamma": lr_gamma,
                    "lr_schedule": lr_schedule,
                    "lr_warmup_steps": lr_warmup_steps,
                    "lr_min_ratio": lr_min_ratio,
                    "ema_decay": ema_decay,
                    "train_kwargs": kwargs,
                },
                output,
            )

        if start_step == 0 and val_every > 0:
            with evaluation_model():
                self.model.eval()
                initial_values = [
                    self.get_val_loss(**kwargs)
                    for _ in range(val_batches)
                ]
            initial_values = [value for value in initial_values if value is not None]
            if initial_values:
                initial_val = torch.stack(initial_values).mean()
                val_losses.append(initial_val.detach().cpu())
                val_steps.append(0)
                best_val_loss = initial_val.item()
                save_checkpoint(0, best_ckpt_path)
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "loss/val": initial_val.item(),
                            "loss/val_best": best_val_loss,
                        },
                        step=0,
                    )

        last_step = start_step
        try:
            pbar = tqdm(range(start_step + 1, num_steps + 1))
            for step in pbar:
                self.model.train()
                opt.zero_grad()
                loss = self.get_train_loss(**kwargs)
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                opt.step()
                if ema_model is not None:
                    with torch.no_grad():
                        for ema_param, param in zip(
                            ema_model.parameters(), self.model.parameters()
                        ):
                            ema_param.lerp_(param, 1 - ema_decay)
                        for ema_buffer, buffer in zip(
                            ema_model.buffers(), self.model.buffers()
                        ):
                            ema_buffer.copy_(buffer)
                if scheduler is not None:
                    scheduler.step()
                train_losses.append(loss.detach().cpu())
                best_train_loss = min(best_train_loss, loss.item())
                last_step = step

                desc = f"Step {step}, train: {loss.item():.3f}"
                log_data: dict[str, Any] = {
                    "loss/train": loss.item(),
                    "loss/train_best": best_train_loss,
                    "learning_rate": opt.param_groups[0]["lr"],
                }
                if val_every > 0 and step % val_every == 0:
                    with evaluation_model():
                        self.model.eval()
                        values = [
                            self.get_val_loss(**kwargs)
                            for _ in range(val_batches)
                        ]
                    values = [value for value in values if value is not None]
                    if values:
                        val = torch.stack(values).mean()
                        val_losses.append(val.detach().cpu())
                        val_steps.append(step)
                        is_best = val.item() < best_val_loss
                        best_val_loss = min(best_val_loss, val.item())
                        if is_best:
                            save_checkpoint(step, best_ckpt_path)
                        desc += f", val: {val.item():.3f}"
                        log_data["loss/val"] = val.item()
                        log_data["loss/val_best"] = best_val_loss

                if plot_every > 0 and step % plot_every == 0:
                    with evaluation_model():
                        self.model.eval()
                        trajectory_path = self.plot_trajectory(
                            step=step,
                            n_images=n_plot_images,
                            n_steps=n_plot_steps,
                            samples_dir=samples_dir,
                            show=show_plots,
                        )
                    if wandb_run is not None and trajectory_path is not None:
                        log_data["samples/trajectory"] = wandb.Image(
                            str(trajectory_path),
                            caption=f"Step {step}: noise to MNIST",
                        )

                if checkpoint_every > 0 and step % checkpoint_every == 0:
                    save_checkpoint(step)
                if wandb_run is not None:
                    wandb_run.log(log_data, step=step)
                pbar.set_description(desc)
        except KeyboardInterrupt:
            save_checkpoint(last_step)
            print(f"training interrupted; checkpoint saved at step {last_step}")
            raise

        self.model.eval()
        save_checkpoint(last_step)
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
        return torch.mean((u_pred - u_ref) ** 2)

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
    ) -> Path:
        device = next(self.model.parameters()).device
        x0, _ = self.path.p_simple.sample(n_images)
        ts = (
            torch.linspace(0, 1, n_steps, device=device)
            .view(1, -1, 1, 1, 1)
            .expand(n_images, -1, 1, 1, 1)
        )
        traj = EulerSimulator(FlowODE(self.model)).simulate_with_trajectory(
            x0, ts, use_tqdm=False
        )
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
        ax.set_title(f"step {step}: {n_images} trajectories (t=0 → 1)")
        fig.tight_layout()
        out = Path(samples_dir) / f"traj_step_{step:06d}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return out

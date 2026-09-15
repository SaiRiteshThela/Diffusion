from abc import ABC, abstractmethod
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

    def train(
        self,
        num_steps: int,
        device: torch.device,
        lr: float = 1e-3,
        optimizer_name: str = "adam",
        weight_decay: float = 0.0,
        max_grad_norm: float | None = None,
        ckpt_path: str | Path | None = None,
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

        size_b = model_size_b(self.model)
        print(f"Training model with size: {size_b / MiB:.3f} MiB")

        self.model.to(device)
        opt = self.get_optimizer(lr, optimizer_name, weight_decay)
        start_step = 0
        train_losses: list[Tensor] = []
        val_losses: list[Tensor] = []
        val_steps: list[int] = []

        if resume_from is not None:
            state = torch.load(resume_from, map_location=device, weights_only=False)
            self.model.load_state_dict(state["model"])
            opt.load_state_dict(state["optimizer"])
            start_step = int(state["step"])
            train_losses = list(state["history"]["train"])
            val_losses = list(state["history"].get("val", []))
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

        def save_checkpoint(step: int) -> None:
            if ckpt_path is None:
                return
            output = Path(ckpt_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "history": history(),
                    "step": step,
                    "num_steps": num_steps,
                    "lr": lr,
                    "optimizer_name": optimizer_name,
                    "weight_decay": weight_decay,
                    "max_grad_norm": max_grad_norm,
                    "train_kwargs": kwargs,
                },
                output,
            )

        if start_step == 0 and val_every > 0:
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
                train_losses.append(loss.detach().cpu())
                best_train_loss = min(best_train_loss, loss.item())
                last_step = step

                desc = f"Step {step}, train: {loss.item():.3f}"
                log_data: dict[str, Any] = {
                    "loss/train": loss.item(),
                    "loss/train_best": best_train_loss,
                }
                if val_every > 0 and step % val_every == 0:
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
                        best_val_loss = min(best_val_loss, val.item())
                        desc += f", val: {val.item():.3f}"
                        log_data["loss/val"] = val.item()
                        log_data["loss/val_best"] = best_val_loss

                if plot_every > 0 and step % plot_every == 0:
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
        ax.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
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

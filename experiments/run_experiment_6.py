import argparse
import os
import sys
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parents[1]
os.chdir(project_root)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import wandb

from datasets.celeba import CelebASampler
from models.config import load_config
from models.flow import FlowModel
from sampling.generate_grid import generate_final_grid
from training.batch_size import probe_train_batch_size
from training.path import GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta
from training.trainer import FlowTrainer, model_size_b

CONFIG_PATH = "configs/experiment_6.yaml"


def resolve_data_root() -> Path:
    persistent_root = Path(os.getenv("DIFFUSION_DATA_ROOT", "/workspace-global/Diffusion-data"))
    data_root = Path("data/raw")
    if not data_root.exists():
        data_root = persistent_root / "datasets"
    return data_root


def make_path(
    cfg: dict[str, Any],
    data_root: Path,
    device: torch.device,
    split: str,
    horizontal_flip: bool,
) -> GaussianConditionalProbabilityPath:
    image_size = cfg["data"]["image_size"]
    return GaussianConditionalProbabilityPath(
        p_data=CelebASampler(
            root=str(data_root),
            split=split,
            image_size=image_size,
            seed=cfg["data"]["seed"],
            horizontal_flip=horizontal_flip,
        ),
        p_simple_shape=[3, image_size, image_size],
        alpha=LinearAlpha(),
        beta=LinearBeta(),
    ).to(device)


def build_trainer(
    cfg: dict[str, Any],
    config_path: str | Path,
    data_root: Path,
    device: torch.device,
) -> FlowTrainer:
    train_path = make_path(
        cfg,
        data_root,
        device,
        "train",
        horizontal_flip=bool(cfg["data"].get("horizontal_flip", False)),
    )
    val_path = make_path(cfg, data_root, device, "val", horizontal_flip=False)
    model = FlowModel.from_config(config_path).to(device)
    return FlowTrainer(path=train_path, model=model, val_path=val_path)


def configure_cuda() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Experiment 6 is intended to run on a CUDA GPU")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    return device


def choose_checkpoint(checkpoints_dir: Path, resume: bool) -> Path | None:
    if not resume:
        return None
    periodic = sorted(
        path for path in checkpoints_dir.glob("*.pt") if not path.name.endswith("_best.pt")
    )
    best_ckpts = sorted(checkpoints_dir.glob("*_best.pt"))
    return periodic[-1] if periodic else (best_ckpts[-1] if best_ckpts else None)


def run(
    cfg: dict[str, Any],
    config_path: str | Path = CONFIG_PATH,
    *,
    run_name: str = "experiment-6-bigger-unet",
    tags: list[str] | None = None,
    group: str | None = None,
    checkpoints_dir: Path | None = None,
    samples_dir: Path | None = None,
    resume: bool = True,
    save_checkpoints: bool = True,
    generate_grid: bool = True,
    scheduler_num_steps: int | None = None,
    abort_train_loss: float | None = None,
) -> dict[str, Any]:
    data_root = resolve_data_root()
    checkpoints_dir = checkpoints_dir or Path("checkpoints") / "experiment_6"
    samples_dir = samples_dir or Path("samples") / "experiment_6"
    wandb_dir = Path("/workspace/wandb")
    api_key = Path("/workspace-global/.wandb/api_key")
    if api_key.exists():
        os.environ.setdefault("WANDB_API_KEY", api_key.read_text().strip())
    for path in (data_root, checkpoints_dir, samples_dir, wandb_dir):
        path.mkdir(parents=True, exist_ok=True)

    device = configure_cuda()
    torch.manual_seed(cfg["data"]["seed"])
    trainer = build_trainer(cfg, config_path, data_root, device)
    print("device", device, torch.cuda.get_device_name(0))
    print(f"train images: {len(trainer.path.p_data.images):,}")
    print(f"val images: {len(trainer.val_path.p_data.images):,}")
    print(f"model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print(f"model size: {model_size_b(trainer.model) / 1024**2:.2f} MiB")
    print(f"batch size: {cfg['data']['batch_size']}")

    train_cfg = cfg["training"]
    sampling_cfg = cfg["sampling"]
    resume_ckpt = choose_checkpoint(checkpoints_dir, resume)
    wandb_id = resume_ckpt.stem.removesuffix("_best") if resume_ckpt else None

    try:
        wandb.login()
        with wandb.init(
            project="celeba-flow-matching",
            dir=str(wandb_dir),
            name=run_name,
            group=group,
            id=wandb_id,
            resume="must" if resume_ckpt else None,
            tags=tags or ["experiment-6", "celeba", "bigger-unet", "cosine", "ema", "4x4"],
            config=cfg,
        ) as wandb_run:
            checkpoint_path = checkpoints_dir / f"{wandb_run.id}.pt" if save_checkpoints else None
            best_checkpoint_path = (
                checkpoints_dir / f"{wandb_run.id}_best.pt" if save_checkpoints else None
            )
            history = trainer.train(
                num_steps=train_cfg["num_steps"],
                device=device,
                lr=train_cfg["learning_rate"],
                optimizer_name=train_cfg["optimizer"],
                weight_decay=train_cfg["weight_decay"],
                max_grad_norm=train_cfg["max_grad_norm"],
                lr_milestones=train_cfg.get("lr_milestones"),
                lr_gamma=train_cfg.get("lr_gamma", 0.1),
                lr_schedule=train_cfg.get("lr_schedule"),
                lr_warmup_steps=train_cfg.get("lr_warmup_steps", 0),
                lr_min_ratio=train_cfg.get("lr_min_ratio", 0.01),
                scheduler_num_steps=scheduler_num_steps,
                abort_train_loss=abort_train_loss,
                ema_decay=train_cfg["ema_decay"],
                batch_size=cfg["data"]["batch_size"],
                ckpt_path=checkpoint_path,
                best_ckpt_path=best_checkpoint_path,
                checkpoint_every=train_cfg["checkpoint_every"] if save_checkpoints else 0,
                val_every=train_cfg["val_every"],
                val_batches=train_cfg["val_batches"],
                plot_every=train_cfg["plot_every"],
                n_plot_images=sampling_cfg["n_plot_images"],
                n_plot_steps=sampling_cfg["n_plot_steps"],
                samples_dir=samples_dir,
                show_plots=False,
                wandb_run=wandb_run,
                resume_from=resume_ckpt,
            )
            train = history["train"]
            best_train_loss = train.min().item() if len(train) else float("nan")
            val = history.get("val")
            best_val_loss = val.min().item() if val is not None and len(val) else float("nan")
            wandb_run.summary["loss/train_best"] = best_train_loss
            wandb_run.summary["loss/val_best"] = best_val_loss
            wandb_run.summary["training_steps"] = train_cfg["num_steps"]
            wandb_run.summary["data/batch_size"] = cfg["data"]["batch_size"]
            result = {
                "history": history,
                "best_train_loss": best_train_loss,
                "best_val_loss": best_val_loss,
                "checkpoint_path": checkpoint_path,
                "best_checkpoint_path": best_checkpoint_path,
                "samples_dir": samples_dir,
                "wandb_id": wandb_run.id,
                "final_step": int(train.shape[0]),
            }
            if generate_grid and best_checkpoint_path is not None and best_checkpoint_path.exists():
                best_state = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
                grid_path = generate_final_grid(
                    trainer.model,
                    best_checkpoint_path,
                    samples_dir / "final",
                    num_samples=sampling_cfg["num_samples"],
                    ode_steps=sampling_cfg["ode_steps"],
                    seed=cfg["data"]["seed"],
                    image_size=cfg["data"]["image_size"],
                    in_channels=cfg["unet"]["in_channels"],
                    device=device,
                )
                wandb_run.log(
                    {"samples/final_grid": wandb.Image(str(grid_path))},
                    step=int(best_state["step"]),
                )
                wandb_run.summary["checkpoint"] = str(checkpoint_path)
                wandb_run.summary["best_checkpoint"] = str(best_checkpoint_path)
                wandb_run.summary["best_checkpoint_step"] = int(best_state["step"])
                result["best_checkpoint_step"] = int(best_state["step"])
                result["grid_path"] = grid_path
        print("best training loss", best_train_loss)
        print("best EMA validation loss", best_val_loss)
        return result
    finally:
        del trainer
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(CONFIG_PATH))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--max-batch",
        action="store_true",
        help="probe the largest train batch that fits on this GPU",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.max_batch or (
        args.batch_size is None and cfg["data"].get("batch_size") == "auto"
    ):
        device = configure_cuda()
        trainer = build_trainer(cfg, args.config, resolve_data_root(), device)
        probed = probe_train_batch_size(trainer, device=device)
        print(f"probed max batch size: {probed}")
        cfg["data"]["batch_size"] = probed
        del trainer
        torch.cuda.empty_cache()
    elif args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    result = run(cfg, args.config, resume=not args.no_resume)
    if result.get("best_checkpoint_path"):
        print("best checkpoint", result["best_checkpoint_path"])
    if result.get("best_checkpoint_step") is not None:
        print("best checkpoint step", result["best_checkpoint_step"])


if __name__ == "__main__":
    main()

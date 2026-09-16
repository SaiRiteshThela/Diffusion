import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
os.chdir(project_root)
sys.path.insert(0, str(project_root))

import torch
import wandb

from datasets.celeba import CelebASampler
from models.config import load_config
from models.flow import FlowModel
from sampling.generate_grid import generate_final_grid
from training.path import GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta
from training.trainer import FlowTrainer, model_size_b

CONFIG_PATH = "configs/experiment_6.yaml"


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    persistent_root = Path(os.getenv("DIFFUSION_DATA_ROOT", "/workspace-global/Diffusion-data"))
    data_root = Path("data/raw")
    if not data_root.exists():
        data_root = persistent_root / "datasets"
    checkpoints_dir = Path("checkpoints") / "experiment_6"
    samples_dir = Path("samples") / "experiment_6"
    wandb_dir = Path("/workspace/wandb")
    api_key = Path("/workspace-global/.wandb/api_key")
    if api_key.exists():
        os.environ.setdefault("WANDB_API_KEY", api_key.read_text().strip())
    for path in (data_root, checkpoints_dir, samples_dir, wandb_dir):
        path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Experiment 6 is intended to run on a CUDA GPU")
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfg["data"]["seed"])

    def make_path(split: str, horizontal_flip: bool) -> GaussianConditionalProbabilityPath:
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

    train_path = make_path("train", horizontal_flip=bool(cfg["data"].get("horizontal_flip", False)))
    val_path = make_path("val", horizontal_flip=False)
    model = FlowModel.from_config(CONFIG_PATH).to(device)
    trainer = FlowTrainer(path=train_path, model=model, val_path=val_path)
    print("device", device, torch.cuda.get_device_name(0))
    print(f"train images: {len(train_path.p_data.images):,}")
    print(f"val images: {len(val_path.p_data.images):,}")
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"model size: {model_size_b(model) / 1024**2:.2f} MiB")

    train_cfg = cfg["training"]
    sampling_cfg = cfg["sampling"]
    existing = sorted(checkpoints_dir.glob("*.pt"))
    resume_ckpt = next((path for path in existing if not path.name.endswith("_best.pt")), None)

    wandb.login()
    with wandb.init(
        project="celeba-flow-matching",
        dir=str(wandb_dir),
        name="experiment-6-bigger-unet",
        id=resume_ckpt.stem if resume_ckpt else None,
        resume="must" if resume_ckpt else None,
        tags=["experiment-6", "celeba", "bigger-unet", "cosine", "ema", "4x4"],
        config=cfg,
    ) as run:
        checkpoint_path = checkpoints_dir / f"{run.id}.pt"
        best_checkpoint_path = checkpoints_dir / f"{run.id}_best.pt"
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
            ema_decay=train_cfg["ema_decay"],
            batch_size=cfg["data"]["batch_size"],
            ckpt_path=checkpoint_path,
            best_ckpt_path=best_checkpoint_path,
            checkpoint_every=train_cfg["checkpoint_every"],
            val_every=train_cfg["val_every"],
            val_batches=train_cfg["val_batches"],
            plot_every=train_cfg["plot_every"],
            n_plot_images=sampling_cfg["n_plot_images"],
            n_plot_steps=sampling_cfg["n_plot_steps"],
            samples_dir=samples_dir,
            show_plots=False,
            wandb_run=run,
            resume_from=resume_ckpt,
        )
        best_train_loss = history["train"].min().item()
        best_val_loss = history["val"].min().item()
        best_state = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
        grid_path = generate_final_grid(
            model,
            best_checkpoint_path,
            samples_dir / "final",
            num_samples=sampling_cfg["num_samples"],
            ode_steps=sampling_cfg["ode_steps"],
            seed=cfg["data"]["seed"],
            image_size=cfg["data"]["image_size"],
            in_channels=cfg["unet"]["in_channels"],
            device=device,
        )
        run.log({"samples/final_grid": wandb.Image(str(grid_path))}, step=int(best_state["step"]))
        run.summary["loss/train_best"] = best_train_loss
        run.summary["loss/val_best"] = best_val_loss
        run.summary["checkpoint"] = str(checkpoint_path)
        run.summary["best_checkpoint"] = str(best_checkpoint_path)
        run.summary["best_checkpoint_step"] = int(best_state["step"])
        run.summary["training_steps"] = train_cfg["num_steps"]

    print("best training loss", best_train_loss)
    print("best EMA validation loss", best_val_loss)
    print("best checkpoint step", int(best_state["step"]))
    print("best checkpoint", best_checkpoint_path)


if __name__ == "__main__":
    main()

"""Short LR search for Experiment 6, then a full train with the winner.

Uses batch 128 on this A40. Peak LR is searched without linear-scaling 2e-4,
which exploded on this U-Net.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from copy import deepcopy
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
os.chdir(project_root)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml

from experiments.run_experiment_6 import (
    CONFIG_PATH,
    build_trainer,
    configure_cuda,
    resolve_data_root,
    run,
)
from models.config import load_config
from training.batch_size import align_batch_size, probe_train_batch_size
from training.tune import select_winner, trial_exploded

LEARNING_RATES = [3e-5, 5e-5, 1e-4]
WARMUP_STEPS = 2000
TUNE_STEPS = 4000
FULL_STEPS = 40000
ABORT_TRAIN_LOSS = 20.0
EMA_DECAY = 0.999


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(CONFIG_PATH))
    parser.add_argument("--tune-steps", type=int, default=TUNE_STEPS)
    parser.add_argument("--full-steps", type=int, default=FULL_STEPS)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--abort-train-loss", type=float, default=ABORT_TRAIN_LOSS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--min-batch", type=int, default=16)
    parser.add_argument("--max-batch", type=int, default=384)
    parser.add_argument(
        "--probe-batch",
        action="store_true",
        help="ignore --batch-size and probe the largest batch that fits",
    )
    parser.add_argument("--tune-only", action="store_true")
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=LEARNING_RATES,
    )
    return parser.parse_args()


def summarize_trial(
    history: dict,
    learning_rate: float,
    warmup_steps: int,
    abort_train_loss: float,
) -> dict:
    train = history["train"]
    val = history.get("val")
    val_steps = history.get("val_steps")
    max_train = float(train.max().item()) if len(train) else float("nan")
    final_val = float(val[-1].item()) if val is not None and len(val) else float("inf")
    best_val = float(val.min().item()) if val is not None and len(val) else float("inf")
    spike_count = int((train > 1).sum().item()) if len(train) else 0
    aborted = (not math.isfinite(max_train)) or max_train > abort_train_loss
    last_step = int(val_steps[-1].item()) if val_steps is not None and len(val_steps) else int(len(train))
    return {
        "learning_rate": learning_rate,
        "lr_warmup_steps": warmup_steps,
        "steps": int(len(train)),
        "last_logged_step": last_step,
        "max_train": max_train,
        "final_val": final_val,
        "best_val": best_val,
        "spike_count": spike_count,
        "aborted": aborted,
    }


def trial_config(
    base: dict,
    *,
    learning_rate: float,
    warmup_steps: int,
    batch_size: int,
    num_steps: int,
    ema_decay: float = EMA_DECAY,
) -> dict:
    cfg = deepcopy(base)
    cfg["data"]["batch_size"] = batch_size
    cfg["training"]["learning_rate"] = learning_rate
    cfg["training"]["lr_warmup_steps"] = warmup_steps
    cfg["training"]["num_steps"] = num_steps
    cfg["training"]["ema_decay"] = ema_decay
    cfg["training"]["checkpoint_every"] = 0
    cfg["training"]["plot_every"] = min(2000, num_steps)
    cfg["training"]["val_every"] = min(200, num_steps)
    return cfg


def write_winner_yaml(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def main() -> None:
    args = parse_args()
    if args.tune_steps < 1 or args.full_steps < args.tune_steps:
        raise ValueError("full-steps must be >= tune-steps >= 1")
    if args.warmup_steps >= args.full_steps:
        raise ValueError("warmup-steps must be smaller than full-steps")

    base_cfg = load_config(args.config)
    output_dir = Path("samples") / "experiment_6_tune"
    output_dir.mkdir(parents=True, exist_ok=True)
    winner_yaml = Path("configs") / "experiment_6_winner.yaml"

    if args.probe_batch:
        device = configure_cuda()
        probe_trainer = build_trainer(base_cfg, args.config, resolve_data_root(), device)
        batch_size = probe_train_batch_size(
            probe_trainer,
            device=device,
            min_batch=args.min_batch,
            max_batch=args.max_batch,
        )
        del probe_trainer
        torch.cuda.empty_cache()
    else:
        batch_size = args.batch_size
        if batch_size < 1:
            raise ValueError("batch-size must be positive")
    print(f"using batch size {batch_size}")

    results: list[dict] = []
    lr_index = 0
    while lr_index < len(args.learning_rates):
        learning_rate = args.learning_rates[lr_index]
        lr_name = f"{learning_rate:.0e}".replace("e-0", "e-")
        run_name = f"exp6-tune-lr{lr_name}-bs{batch_size}"
        print(f"\n=== tune {run_name} ===")
        cfg = trial_config(
            base_cfg,
            learning_rate=learning_rate,
            warmup_steps=args.warmup_steps,
            batch_size=batch_size,
            num_steps=args.tune_steps,
        )
        try:
            trial = run(
                cfg,
                args.config,
                run_name=run_name,
                tags=["experiment-6", "tune", "celeba", "bigger-unet"],
                group="experiment-6-tune",
                checkpoints_dir=Path("checkpoints") / "experiment_6_tune" / run_name,
                samples_dir=output_dir / run_name,
                resume=False,
                save_checkpoints=False,
                generate_grid=False,
                scheduler_num_steps=args.full_steps,
                abort_train_loss=args.abort_train_loss,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            smaller = align_batch_size(batch_size - 32)
            if smaller < args.min_batch or smaller >= batch_size:
                raise
            print(f"OOM at batch {batch_size}; retrying at {smaller}")
            batch_size = smaller
            continue
        summary = summarize_trial(
            trial["history"],
            learning_rate,
            args.warmup_steps,
            args.abort_train_loss,
        )
        summary["batch_size"] = batch_size
        summary["wandb_id"] = trial["wandb_id"]
        summary["exploded"] = trial_exploded(summary, args.abort_train_loss)
        results.append(summary)
        print("trial", json.dumps(summary, indent=2))
        torch.cuda.empty_cache()
        lr_index += 1

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps({"batch_size": batch_size, "trials": results}, indent=2) + "\n")
    winner = select_winner(results, abort_train_loss=args.abort_train_loss)
    if winner is None:
        print("no stable trial; not starting a full train")
        raise SystemExit(2)

    full_cfg = deepcopy(base_cfg)
    full_cfg["data"]["batch_size"] = batch_size
    full_cfg["training"]["learning_rate"] = winner["learning_rate"]
    full_cfg["training"]["lr_warmup_steps"] = winner["lr_warmup_steps"]
    full_cfg["training"]["num_steps"] = args.full_steps
    full_cfg["training"]["ema_decay"] = EMA_DECAY
    write_winner_yaml(full_cfg, winner_yaml)
    print("winner", json.dumps(winner, indent=2))
    print("wrote", winner_yaml)

    if args.tune_only:
        return

    print("\n=== full train with winner ===")
    result = run(
        full_cfg,
        args.config,
        run_name="experiment-6-bigger-unet",
        tags=["experiment-6", "celeba", "bigger-unet", "cosine", "ema", "4x4", "tuned"],
        checkpoints_dir=Path("checkpoints") / "experiment_6",
        samples_dir=Path("samples") / "experiment_6",
        resume=False,
        save_checkpoints=True,
        generate_grid=True,
    )
    print("full train best EMA val", result["best_val_loss"])
    print("full train best checkpoint", result.get("best_checkpoint_path"))


if __name__ == "__main__":
    main()

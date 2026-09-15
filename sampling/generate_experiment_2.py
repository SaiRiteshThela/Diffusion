import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from models.flow import FlowModel
from models.time_embedding import SinusoidalTimeEmbeddings
from models.unet import UNet
from sampling.ode import EulerSimulator, FlowODE


WINNER_CONFIG = {
    "start_dim": 96,
    "dim_mults": (1, 2, 4),
    "residual_blocks_per_group": 1,
    "group_norm_num_groups": 8,
    "n_heads": 2,
    "time_embedding_dim": 128,
    "conditioning_dim": 256,
    "d_ff": 512,
    "activation": "gelu",
    "attn_dropout": 0.2,
    "ffn_dropout": 0.1,
    "proj_dropout": 0.0,
}


def build_model() -> FlowModel:
    time_embed = SinusoidalTimeEmbeddings(
        time_emb_dim=WINNER_CONFIG["time_embedding_dim"],
        scaled_time_emb_dim=WINNER_CONFIG["conditioning_dim"],
    )
    unet = UNet(
        in_channels=1,
        start_dim=WINNER_CONFIG["start_dim"],
        dim_mults=WINNER_CONFIG["dim_mults"],
        residual_blocks_per_group=WINNER_CONFIG["residual_blocks_per_group"],
        group_norm_num_groups=WINNER_CONFIG["group_norm_num_groups"],
        time_emb_dim=WINNER_CONFIG["conditioning_dim"],
        n_heads=WINNER_CONFIG["n_heads"],
        d_ff=WINNER_CONFIG["d_ff"],
        attn_dropout=WINNER_CONFIG["attn_dropout"],
        ffn_dropout=WINNER_CONFIG["ffn_dropout"],
        proj_dropout=WINNER_CONFIG["proj_dropout"],
        activation=WINNER_CONFIG["activation"],
    )
    return FlowModel(unet=unet, time_embed=time_embed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/workspace-global/Diffusion-data/checkpoints/experiment_2/"
            "rpta0rn5-5000-steps.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/workspace-global/Diffusion-data/samples/"
            "experiment_2_winner_5000"
        ),
    )
    parser.add_argument("--num-samples", type=int, default=25)
    parser.add_argument("--ode-steps", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("num-samples must be positive")
    if args.ode_steps < 1:
        raise ValueError("ode-steps must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(
        args.num_samples,
        1,
        32,
        32,
        generator=generator,
        device=device,
    )
    ts = torch.linspace(
        0,
        1,
        args.ode_steps + 1,
        device=device,
    ).expand(args.num_samples, -1)

    samples = EulerSimulator(FlowODE(model)).simulate(noise, ts).clamp(-1, 1).cpu()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(samples, args.output_dir / "samples.pt")

    # Upscale only for convenient viewing; samples.pt retains the original 32x32 data.
    display_samples = F.interpolate(samples, size=(256, 256), mode="nearest")
    save_image(
        display_samples,
        args.output_dir / "grid_5x5.png",
        nrow=5,
        normalize=True,
        value_range=(-1, 1),
        padding=4,
        pad_value=1,
    )
    individual_dir = args.output_dir / "individual"
    individual_dir.mkdir(exist_ok=True)
    for index, sample in enumerate(display_samples):
        save_image(
            sample,
            individual_dir / f"sample_{index + 1:02d}.png",
            normalize=True,
            value_range=(-1, 1),
        )

    print(f"generated {args.num_samples} samples with {args.ode_steps} Euler steps")
    print(f"grid: {args.output_dir / 'grid_5x5.png'}")
    print(f"individual images: {individual_dir}")


if __name__ == "__main__":
    main()

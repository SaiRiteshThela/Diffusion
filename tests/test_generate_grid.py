from pathlib import Path

import torch

from sampling.generate_grid import generate_final_grid
from tests.helpers import TinyFlow


def test_generate_final_grid_loads_ema_not_raw(tmp_path: Path):
    raw = TinyFlow(channels=3)
    ema = TinyFlow(channels=3)
    with torch.no_grad():
        raw.conv.weight.fill_(0.0)
        ema.conv.weight.fill_(0.5)
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "model": raw.state_dict(),
            "ema_model": ema.state_dict(),
            "step": 12,
        },
        checkpoint,
    )
    loaded = TinyFlow(channels=3)
    grid = generate_final_grid(
        loaded,
        checkpoint,
        tmp_path / "final",
        num_samples=4,
        ode_steps=2,
        seed=0,
        image_size=8,
        in_channels=3,
        device=torch.device("cpu"),
        display_size=16,
    )
    assert grid.exists()
    assert (tmp_path / "final" / "samples.pt").exists()
    assert torch.allclose(loaded.conv.weight, ema.conv.weight)
    assert not torch.allclose(loaded.conv.weight, raw.conv.weight)

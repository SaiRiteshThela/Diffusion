from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from models.adm_unet import ADMUNet
from models.config import load_config
from models.time_embedding import SinusoidalTimeEmbeddings
from models.unet import UNet


class FlowModel(nn.Module):
    def __init__(self, unet: nn.Module, time_embed: nn.Module):
        super().__init__()
        self.unet = unet
        self.time_embed = time_embed

    @classmethod
    def from_config(cls, path: str | Path) -> "FlowModel":
        cfg = load_config(path)
        model_type = cfg.get("model_type", "custom")
        if model_type == "adm":
            # ADM embeds normalized times internally; legacy configs retain
            # their existing wrapper and checkpoint keys.
            return cls(ADMUNet(**cfg.get("adm_unet", {})), nn.Identity())
        if model_type != "custom":
            raise ValueError(f"unknown model_type {model_type!r}; expected 'custom' or 'adm'")
        time_embed = SinusoidalTimeEmbeddings(**cfg["time_embedding"])
        unet = UNet.from_config(path)
        return cls(unet, time_embed)

    def forward(self, xt: Tensor, t: Tensor) -> Tensor:
        if xt.shape[0] != t.numel():
            raise ValueError(f"batch xt={xt.shape[0]} != t={t.numel()}")
        return self.unet(xt, self.time_embed(t))


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parents[1] / "configs" / "dummy.yaml"
    cfg = load_config(config_path)
    model = FlowModel.from_config(config_path)
    batch = cfg["test"]["batch_size"]
    size = cfg["test"]["image_size"]
    channels = cfg["unet"]["in_channels"]
    xt = torch.randn(batch, channels, size, size)
    t = torch.rand(batch)
    u = model(xt, t)
    assert u.shape == xt.shape, u.shape
    print("FlowModel", tuple(u.shape))

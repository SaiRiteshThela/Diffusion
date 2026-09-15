from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from models.config import get_activation, load_config
from models.downsample import Downsample
from models.residual import ResidualBlock
from models.transformer import TransformerBlock
from models.upsample import Upsample


class LayerSpec(NamedTuple):
    kind: str
    channels: tuple[int, ...]


@dataclass(frozen=True)
class BlockSettings:
    group_norm_num_groups: int
    time_emb_dim: int
    n_heads: int
    d_ff: int
    attn_dropout: float
    ffn_dropout: float
    proj_dropout: float
    activation: Callable


def _channel_sizes(start_dim: int, dim_mults: Sequence[int]) -> list[int]:
    return [start_dim * m for m in dim_mults]


def build_encoder_spec(
    channel_sizes: Sequence[int],
    residual_blocks_per_group: int,
) -> list[LayerSpec]:
    spec: list[LayerSpec] = []
    for idx, width in enumerate(channel_sizes):
        for _ in range(residual_blocks_per_group):
            spec.append(LayerSpec("residual", (width, width)))
        spec.append(LayerSpec("downsample", (width, width)))
        spec.append(LayerSpec("attention", (width,)))
        if idx != len(channel_sizes) - 1:
            spec.append(LayerSpec("residual", (width, channel_sizes[idx + 1])))
    return spec


def build_bottleneck_spec(
    width: int,
    residual_blocks_per_group: int,
) -> list[LayerSpec]:
    return [LayerSpec("residual", (width, width)) for _ in range(residual_blocks_per_group)]


def build_decoder_spec(
    encoder_spec: Sequence[LayerSpec],
    bottleneck_width: int,
    input_width: int,
) -> list[LayerSpec]:
    spec: list[LayerSpec] = []
    out_dim = bottleneck_width
    for kind, channels in reversed(encoder_spec):
        if kind == "attention":
            spec.append(LayerSpec("attention", channels))
            continue
        ch_in, ch_out = channels
        spec.append(LayerSpec("residual", (out_dim + ch_out, ch_in)))
        if kind == "downsample":
            spec.append(LayerSpec("upsample", (ch_in, ch_in)))
        out_dim = ch_in
    spec.append(LayerSpec("residual", (input_width * 2, input_width)))
    return spec


def build_layer(spec: LayerSpec, settings: BlockSettings) -> nn.Module:
    kind, channels = spec
    if kind == "residual":
        ch_in, ch_out = channels
        return ResidualBlock(ch_in, ch_out, settings.group_norm_num_groups, settings.time_emb_dim)
    if kind == "downsample":
        ch_in, ch_out = channels
        return Downsample(ch_in, ch_out)
    if kind == "upsample":
        ch_in, ch_out = channels
        return Upsample(ch_in, ch_out)
    if kind == "attention":
        (width,) = channels
        return TransformerBlock(
            width,
            settings.n_heads,
            settings.d_ff,
            settings.attn_dropout,
            settings.ffn_dropout,
            settings.proj_dropout,
            settings.activation,
        )
    raise ValueError(f"unknown layer kind {kind!r}")


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        start_dim: int = 64,
        dim_mults: Sequence[int] = (1, 2, 4),
        residual_blocks_per_group: int = 1,
        group_norm_num_groups: int = 16,
        time_emb_dim: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        proj_dropout: float = 0.1,
        activation: str | Callable = "gelu",
    ):
        super().__init__()
        activation_fn = get_activation(activation)
        dim_mults = tuple(dim_mults)
        channel_sizes = _channel_sizes(start_dim, dim_mults)
        for width in channel_sizes:
            if width % n_heads != 0:
                raise ValueError(f"channels={width} must be divisible by n_heads={n_heads}")
            if width % group_norm_num_groups != 0:
                raise ValueError(
                    f"channels={width} must be divisible by num_groups={group_norm_num_groups}"
                )

        settings = BlockSettings(
            group_norm_num_groups=group_norm_num_groups,
            time_emb_dim=time_emb_dim,
            n_heads=n_heads,
            d_ff=d_ff,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
            proj_dropout=proj_dropout,
            activation=activation_fn,
        )

        encoder_spec = build_encoder_spec(channel_sizes, residual_blocks_per_group)
        bottleneck_spec = build_bottleneck_spec(channel_sizes[-1], residual_blocks_per_group)
        decoder_spec = build_decoder_spec(encoder_spec, channel_sizes[-1], channel_sizes[0])

        self.in_channels = in_channels
        self.time_emb_dim = time_emb_dim
        self.encoder_spec = encoder_spec
        self.decoder_spec = decoder_spec

        self.conv_in = nn.Conv2d(in_channels, channel_sizes[0], kernel_size=3, padding="same")
        self.encoder = nn.ModuleList(build_layer(spec, settings) for spec in encoder_spec)
        self.bottleneck = nn.ModuleList(build_layer(spec, settings) for spec in bottleneck_spec)
        self.decoder = nn.ModuleList(build_layer(spec, settings) for spec in decoder_spec)
        self.conv_out = nn.Conv2d(channel_sizes[0], in_channels, kernel_size=3, padding="same")

    @classmethod
    def from_config(cls, path: str | Path) -> "UNet":
        cfg = load_config(path)
        return cls(**cfg["unet"])

    def _run(self, layer: nn.Module, x: Tensor, time_emb: Tensor) -> Tensor:
        if isinstance(layer, ResidualBlock):
            return layer(x, time_emb)
        return layer(x)

    def _encode(self, x: Tensor, time_emb: Tensor) -> tuple[Tensor, list[Tensor]]:
        skips = [x]
        for layer in self.encoder:
            x = self._run(layer, x, time_emb)
            if isinstance(layer, (ResidualBlock, Downsample)):
                skips.append(x)
        return x, skips

    def _bottleneck(self, x: Tensor, time_emb: Tensor) -> Tensor:
        for layer in self.bottleneck:
            x = self._run(layer, x, time_emb)
        return x

    def _decode(self, x: Tensor, time_emb: Tensor, skips: list[Tensor]) -> Tensor:
        for layer in self.decoder:
            if isinstance(layer, ResidualBlock):
                x = torch.cat((x, skips.pop()), dim=1)
            x = self._run(layer, x, time_emb)
        if skips:
            raise RuntimeError(f"{len(skips)} unused encoder skips")
        return x

    def forward(self, x: Tensor, time_emb: Tensor) -> Tensor:
        if time_emb.shape[0] != x.shape[0]:
            raise ValueError(f"time batch {time_emb.shape[0]} != x batch {x.shape[0]}")
        if time_emb.shape[-1] != self.time_emb_dim:
            raise ValueError(f"expected time_emb_dim={self.time_emb_dim}, got {time_emb.shape[-1]}")

        x, skips = self._encode(self.conv_in(x), time_emb)
        x = self._bottleneck(x, time_emb)
        x = self._decode(x, time_emb, skips)
        return self.conv_out(x)


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parents[1] / "configs" / "dummy.yaml"
    model = UNet.from_config(config_path)
    cfg = load_config(config_path)
    batch = cfg["test"]["batch_size"]
    size = cfg["test"]["image_size"]
    channels = cfg["unet"]["in_channels"]
    x = torch.randn(batch, channels, size, size)
    t = torch.randn(batch, cfg["unet"]["time_emb_dim"])
    out = model(x, t)
    assert out.shape == x.shape, out.shape
    print("UNet", tuple(out.shape))

"""Compact unconditional ADM U-Net for continuous-time velocity prediction.

Adapted from OpenAI guided-diffusion's U-Net and timestep embedding:
https://github.com/openai/guided-diffusion

MIT License
Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _zero(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


class GroupNorm32(nn.GroupNorm):
    """Keep normalization arithmetic in FP32 when using autocast."""

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).to(x.dtype)


class FixedTimeEmbedding(nn.Module):
    def __init__(self, channels: int, time_scale: float = 1000.0):
        super().__init__()
        self.channels = channels
        self.time_scale = time_scale
        frequencies = torch.exp(-math.log(10000.0) * torch.arange(channels // 2).float() / (channels // 2))
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, t: Tensor) -> Tensor:
        angles = t.reshape(-1, 1).float() * self.time_scale * self.frequencies[None]
        result = torch.cat((angles.cos(), angles.sin()), dim=-1)
        return F.pad(result, (0, self.channels % 2))


class ADMResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, groups: int, dropout: float):
        super().__init__()
        self.input = nn.Sequential(
            GroupNorm32(groups, in_channels), nn.SiLU(), nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_channels, 2 * out_channels))
        self.norm = GroupNorm32(groups, out_channels)
        self.output = nn.Sequential(
            nn.SiLU(), nn.Dropout(dropout), _zero(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor, embedding: Tensor) -> Tensor:
        h = self.input(x)
        scale, shift = self.time(embedding).to(h.dtype).chunk(2, dim=1)
        h = self.norm(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.skip(x) + self.output(h)


class ADMAttention(nn.Module):
    def __init__(self, channels: int, heads: int, groups: int):
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels={channels} must be divisible by num_heads={heads}")
        self.heads = heads
        self.norm = GroupNorm32(groups, channels)
        self.qkv = nn.Conv1d(channels, 3 * channels, 1)
        self.projection = _zero(nn.Conv1d(channels, channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        flat = x.reshape(batch, channels, height * width)
        qkv = self.qkv(self.norm(flat)).reshape(batch, 3, self.heads, channels // self.heads, height * width)
        q, k, v = (part.transpose(-1, -2) for part in qkv.unbind(dim=1))
        h = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        h = h.transpose(-1, -2).reshape(batch, channels, height * width)
        return (flat + self.projection(h)).reshape(batch, channels, height, width)


class _TimedSequential(nn.Sequential):
    def forward(self, x: Tensor, embedding: Tensor) -> Tensor:
        for layer in self:
            x = layer(x, embedding) if isinstance(layer, ADMResBlock) else layer(x)
        return x


class _Upsample(nn.Sequential):
    def __init__(self, channels: int):
        super().__init__(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(channels, channels, 3, padding=1))


class ADMUNet(nn.Module):
    """Predict RGB velocity from an image and normalized times in [0, 1].

    attention_resolutions contains spatial sizes, not downsampling factors.
    Time rescaling affects only the embedding; the flow path/loss are unchanged.
    """

    def __init__(
        self,
        image_size: int = 64,
        in_channels: int = 3,
        model_channels: int = 64,
        channel_mult: Sequence[int] = (1, 2, 3, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Sequence[int] = (16, 8),
        num_heads: int = 4,
        dropout: float = 0.0,
        norm_groups: int = 32,
        time_scale: float = 1000.0,
    ):
        super().__init__()
        if not channel_mult or num_res_blocks < 1 or model_channels < 2:
            raise ValueError("channel_mult must be nonempty, num_res_blocks >= 1 and model_channels >= 2")
        if num_heads < 1 or norm_groups < 1 or image_size < 1 or not 0 <= dropout < 1:
            raise ValueError("invalid heads, normalization groups, image size or dropout")
        if image_size % (2 ** (len(channel_mult) - 1)):
            raise ValueError("image_size must be divisible by the total downsampling factor")
        widths = [model_channels * multiplier for multiplier in channel_mult]
        if any(width < 1 or width % norm_groups or width % num_heads for width in widths):
            raise ValueError("all channel widths must be positive and divisible by norm_groups and num_heads")
        self.image_size = image_size
        self.in_channels = in_channels
        time_channels = 4 * model_channels
        self.time_embedding = nn.Sequential(
            FixedTimeEmbedding(model_channels, time_scale),
            nn.Linear(model_channels, time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )
        attention_resolutions = set(attention_resolutions)
        possible_resolutions = {image_size // (2 ** i) for i in range(len(widths))}
        if not attention_resolutions <= possible_resolutions:
            raise ValueError("attention_resolutions must contain spatial sizes present in the U-Net")

        def residual(ch_in: int, ch_out: int) -> ADMResBlock:
            return ADMResBlock(ch_in, ch_out, time_channels, norm_groups, dropout)

        def attention(ch: int) -> ADMAttention:
            return ADMAttention(ch, num_heads, norm_groups)

        channels = widths[0]
        resolution = image_size
        self.encoder = nn.ModuleList([_TimedSequential(nn.Conv2d(in_channels, channels, 3, padding=1))])
        skip_channels = [channels]
        for level, width in enumerate(widths):
            for _ in range(num_res_blocks):
                layers = [residual(channels, width)]
                channels = width
                if resolution in attention_resolutions:
                    layers.append(attention(channels))
                self.encoder.append(_TimedSequential(*layers))
                skip_channels.append(channels)
            if level != len(widths) - 1:
                self.encoder.append(_TimedSequential(nn.Conv2d(channels, channels, 3, stride=2, padding=1)))
                skip_channels.append(channels)
                resolution //= 2

        self.middle = _TimedSequential(residual(channels, channels), attention(channels), residual(channels, channels))
        self.decoder = nn.ModuleList()
        for level in reversed(range(len(widths))):
            width = widths[level]
            for block in range(num_res_blocks + 1):
                layers = [residual(channels + skip_channels.pop(), width)]
                channels = width
                if resolution in attention_resolutions:
                    layers.append(attention(channels))
                if level > 0 and block == num_res_blocks:
                    layers.append(_Upsample(channels))
                    resolution *= 2
                self.decoder.append(_TimedSequential(*layers))
        if skip_channels:
            raise RuntimeError("unbalanced encoder/decoder skip construction")
        self.output = nn.Sequential(
            GroupNorm32(norm_groups, channels), nn.SiLU(), _zero(nn.Conv2d(channels, in_channels, 3, padding=1))
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1:] != (self.in_channels, self.image_size, self.image_size):
            raise ValueError(f"expected [batch, {self.in_channels}, {self.image_size}, {self.image_size}] input")
        if t.numel() != x.shape[0]:
            raise ValueError(f"batch x={x.shape[0]} != t={t.numel()}")
        embedding = self.time_embedding(t)
        skips = []
        h = x
        for block in self.encoder:
            h = block(h, embedding)
            skips.append(h)
        h = self.middle(h, embedding)
        for block in self.decoder:
            h = block(torch.cat((h, skips.pop()), dim=1), embedding)
        return self.output(h)

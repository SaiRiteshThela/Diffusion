import torch
import torch.nn as nn


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


if __name__ == "__main__":
    model = Downsample(in_channels=64, out_channels=64)
    out = model(torch.randn(2, 64, 16, 16))
    assert out.shape == (2, 64, 8, 8), out.shape
    print("Downsample", tuple(out.shape))

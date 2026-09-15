import torch
import torch.nn as nn


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same"),
        )

    def forward(self, x):
        return self.upsample(x)


if __name__ == "__main__":
    model = Upsample(in_channels=64, out_channels=64)
    out = model(torch.randn(2, 64, 16, 16))
    assert out.shape == (2, 64, 32, 32), out.shape
    print("Upsample", tuple(out.shape))

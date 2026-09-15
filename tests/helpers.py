import torch
import torch.nn as nn

from training.path import Sampleable


class TensorImages(nn.Module, Sampleable):
    """In-memory image set so tests do not touch torchvision MNIST."""

    def __init__(self, images: torch.Tensor):
        super().__init__()
        self.register_buffer("images", images)

    def sample(self, num_samples: int):
        idx = torch.randint(0, len(self.images), (num_samples,), device=self.images.device)
        return self.images[idx], None


class TinyFlow(nn.Module):
    def __init__(self, channels: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.conv(xt)

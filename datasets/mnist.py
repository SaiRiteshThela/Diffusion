from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm

from training.path import Sampleable


class MNISTSampler(nn.Module, Sampleable):
    """Cached MNIST in [-1, 1] at 32x32 so training is not bound by dataset __getitem__."""

    def __init__(
        self,
        root: str = "./data/raw",
        split: Literal["train", "val", "test"] = "train",
        val_size: int = 5000,
        seed: int = 0,
    ):
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")
        if val_size <= 0 or val_size >= 60_000:
            raise ValueError("val_size must be between 1 and 59,999")

        use_train_set = split != "test"
        dataset = datasets.MNIST(
            root=root,
            train=use_train_set,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            ),
        )
        if split == "test":
            indices = torch.arange(len(dataset))
        else:
            permutation = torch.randperm(
                len(dataset), generator=torch.Generator().manual_seed(seed)
            )
            indices = permutation[:-val_size] if split == "train" else permutation[-val_size:]

        images = torch.empty(len(indices), 1, 32, 32)
        labels = torch.empty(len(indices), dtype=torch.int64)
        for out_idx, dataset_idx in enumerate(tqdm(indices, desc=f"cache MNIST {split}")):
            images[out_idx], labels[out_idx] = dataset[int(dataset_idx)]
        self.register_buffer("images", images)
        self.register_buffer("labels", labels)

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if num_samples > len(self.images):
            raise ValueError(f"num_samples exceeds dataset size: {len(self.images)}")
        idx = torch.randint(0, len(self.images), (num_samples,), device=self.images.device)
        return self.images[idx], self.labels[idx]

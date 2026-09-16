from pathlib import Path
from typing import Literal, Optional, Tuple
from urllib.request import urlretrieve

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.datasets.utils import extract_archive
from tqdm import tqdm

from training.path import Sampleable

CELEBA_HF_BASE = "https://huggingface.co/datasets/Yuehao/celeba/resolve/main"
CELEBA_FILES = (
    "list_eval_partition.txt",
    "list_attr_celeba.txt",
    "identity_CelebA.txt",
    "list_bbox_celeba.txt",
    "list_landmarks_align_celeba.txt",
    "img_align_celeba.zip",
)
SPLIT_TO_TORCHVISION = {"train": "train", "val": "valid", "test": "test"}


def _celeba_dir(root: str | Path) -> Path:
    return Path(root) / "celeba"


def _extract_root() -> Path:
    return Path("/workspace/celeba-extract")


def download_celeba(root: str | Path) -> Path:
    """Download aligned CelebA into root/celeba and extract images onto local disk."""
    celeba_dir = _celeba_dir(root)
    celeba_dir.mkdir(parents=True, exist_ok=True)
    for filename in CELEBA_FILES:
        destination = celeba_dir / filename
        if destination.exists():
            continue
        url = f"{CELEBA_HF_BASE}/{filename}"
        partial = destination.with_suffix(destination.suffix + ".partial")
        urlretrieve(url, partial)
        partial.replace(destination)

    images_dir = celeba_dir / "img_align_celeba"
    if images_dir.exists():
        return celeba_dir

    extract_to = _extract_root()
    extract_to.mkdir(parents=True, exist_ok=True)
    extracted = extract_to / "img_align_celeba"
    if not extracted.exists():
        extract_archive(str(celeba_dir / "img_align_celeba.zip"), str(extract_to))
    if images_dir.is_symlink() or images_dir.exists():
        images_dir.unlink()
    images_dir.symlink_to(extracted)
    return celeba_dir


class CelebASampler(nn.Module, Sampleable):
    """Cached CelebA in [-1, 1]. Images are stored as uint8 to keep RAM reasonable."""

    def __init__(
        self,
        root: str = "./data/raw",
        split: Literal["train", "val", "test"] = "train",
        image_size: int = 64,
        seed: int = 0,
        download: bool = True,
        horizontal_flip: bool = False,
    ):
        super().__init__()
        if split not in SPLIT_TO_TORCHVISION:
            raise ValueError("split must be 'train', 'val', or 'test'")
        if image_size < 1:
            raise ValueError("image_size must be positive")

        cache_path = _celeba_dir(root) / f"cache_{split}_{image_size}.pt"
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            images = payload["images"]
            labels = payload["labels"]
        else:
            if download:
                download_celeba(root)
            dataset = datasets.CelebA(
                root=root,
                split=SPLIT_TO_TORCHVISION[split],
                download=False,
                transform=transforms.Compose(
                    [
                        transforms.CenterCrop(178),
                        transforms.Resize(image_size),
                        transforms.PILToTensor(),
                    ]
                ),
                target_type="identity",
            )
            images = torch.empty(len(dataset), 3, image_size, image_size, dtype=torch.uint8)
            labels = torch.empty(len(dataset), dtype=torch.int64)
            generator = torch.Generator().manual_seed(seed)
            order = torch.randperm(len(dataset), generator=generator)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=64,
                shuffle=False,
                num_workers=0 if len(dataset) < 256 else 4,
                pin_memory=False,
            )
            cursor = 0
            for batch_images, batch_labels in tqdm(loader, desc=f"cache CelebA {split} {image_size}"):
                n = batch_images.shape[0]
                images[cursor : cursor + n] = batch_images
                labels[cursor : cursor + n] = batch_labels.reshape(n)
                cursor += n
            images = images[order]
            labels = labels[order]
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"images": images, "labels": labels}, cache_path)

        self.image_size = image_size
        self.horizontal_flip = horizontal_flip
        self.register_buffer("images", images, persistent=False)
        self.register_buffer("labels", labels, persistent=False)

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if num_samples > len(self.images):
            raise ValueError(f"num_samples exceeds dataset size: {len(self.images)}")
        idx = torch.randint(0, len(self.images), (num_samples,), device=self.images.device)
        images = self.images[idx].to(dtype=torch.float32).div(127.5).sub(1.0)
        if self.horizontal_flip:
            flip = torch.rand(num_samples, device=images.device) < 0.5
            images[flip] = images[flip].flip(-1)
        return images, self.labels[idx]

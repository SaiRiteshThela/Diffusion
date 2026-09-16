import torch
import torch.nn as nn
import pytest

from datasets.celeba import CelebASampler


def _sampler_with_buffers(n: int = 10, size: int = 16) -> CelebASampler:
    sampler = CelebASampler.__new__(CelebASampler)
    nn.Module.__init__(sampler)
    sampler.image_size = size
    sampler.horizontal_flip = False
    sampler.register_buffer(
        "images",
        torch.linspace(0, 255, n * 3 * size * size).view(n, 3, size, size).to(torch.uint8),
    )
    sampler.register_buffer("labels", torch.arange(n))
    return sampler


def test_celeba_sample_shapes_and_range():
    sampler = _sampler_with_buffers(12, size=16)
    x, y = sampler.sample(5)
    assert x.shape == (5, 3, 16, 16)
    assert y.shape == (5,)
    assert y.dtype == torch.int64
    assert x.dtype == torch.float32
    assert x.min() >= -1
    assert x.max() <= 1


def test_celeba_sample_too_many_raises():
    sampler = _sampler_with_buffers(4)
    with pytest.raises(ValueError, match="num_samples exceeds dataset size"):
        sampler.sample(5)


def test_celeba_rejects_bad_split():
    with pytest.raises(ValueError, match="split must be"):
        CelebASampler(split="holdout", download=False)


def test_celeba_uses_official_split_names(monkeypatch, tmp_path):
    seen = {}

    class FakeCelebA:
        def __init__(self, root, split, download, transform, target_type):
            seen["split"] = split
            seen["root"] = root
            self.transform = transform
            self._n = 6

        def __len__(self):
            return self._n

        def __getitem__(self, index):
            from PIL import Image

            image = Image.new("RGB", (178, 218), color=(index, 40, 80))
            if self.transform is not None:
                image = self.transform(image)
            return image, index

    monkeypatch.setattr("datasets.celeba.datasets.CelebA", FakeCelebA)
    monkeypatch.setattr("datasets.celeba.download_celeba", lambda root: tmp_path / "celeba")
    sampler = CelebASampler(root=str(tmp_path), split="val", image_size=8, download=False)
    assert seen["split"] == "valid"
    assert sampler.images.shape == (6, 3, 8, 8)
    x, y = sampler.sample(2)
    assert x.shape == (2, 3, 8, 8)
    assert y.shape == (2,)


def test_celeba_horizontal_flip_can_mirror():
    sampler = _sampler_with_buffers(1, size=4)
    sampler.horizontal_flip = True
    sampler.images[0].zero_()
    sampler.images[0, :, :, 0] = 255
    found_original = False
    found_flipped = False
    torch.manual_seed(0)
    for _ in range(40):
        image, _ = sampler.sample(1)
        if image[0, 0, 0, 0].item() > 0.9:
            found_original = True
        if image[0, 0, 0, -1].item() > 0.9:
            found_flipped = True
    assert found_original
    assert found_flipped

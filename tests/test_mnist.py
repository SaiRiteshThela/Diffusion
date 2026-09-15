import torch
import torch.nn as nn
import pytest

from datasets.mnist import MNISTSampler


def _sampler_with_buffers(n: int = 10) -> MNISTSampler:
    sampler = MNISTSampler.__new__(MNISTSampler)
    nn.Module.__init__(sampler)
    sampler.register_buffer("images", torch.linspace(-1, 1, n * 32 * 32).view(n, 1, 32, 32))
    sampler.register_buffer("labels", torch.arange(n))
    return sampler


def test_mnist_sample_shapes_and_range():
    sampler = _sampler_with_buffers(12)
    x, y = sampler.sample(5)
    assert x.shape == (5, 1, 32, 32)
    assert y.shape == (5,)
    assert y.dtype == torch.int64
    assert x.min() >= -1
    assert x.max() <= 1


def test_mnist_sample_too_many_raises():
    sampler = _sampler_with_buffers(4)
    with pytest.raises(ValueError, match="num_samples exceeds dataset size"):
        sampler.sample(5)


def test_mnist_sampler_accepts_split():
    import inspect

    assert "split" in inspect.signature(MNISTSampler.__init__).parameters


def test_mnist_train_val_are_disjoint(monkeypatch):
    class FakeMNIST:
        def __init__(self, root, train, download, transform):
            self.train = train

        def __len__(self):
            return 20 if self.train else 8

        def __getitem__(self, index):
            return torch.full((1, 32, 32), float(index)), index

    monkeypatch.setattr("datasets.mnist.datasets.MNIST", FakeMNIST)
    train = MNISTSampler(split="train", val_size=5, seed=7)
    val = MNISTSampler(split="val", val_size=5, seed=7)
    test = MNISTSampler(split="test", val_size=5, seed=7)

    train_labels = set(train.labels.tolist())
    val_labels = set(val.labels.tolist())
    assert len(train_labels) == 15
    assert len(val_labels) == 5
    assert train_labels.isdisjoint(val_labels)
    assert train_labels | val_labels == set(range(20))
    assert test.labels.tolist() == list(range(8))


def test_mnist_sample_moves_with_module():
    sampler = _sampler_with_buffers(8)
    x, y = sampler.sample(3)
    assert x.device == sampler.images.device
    assert y.device == sampler.labels.device

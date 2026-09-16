import numpy as np
import pytest
import torch

from sampling.fid_celeba import (
    activation_statistics,
    frechet_distance,
    generate_uint8,
    load_eval_weights,
    to_uint8,
)
from tests.helpers import TinyFlow


def test_to_uint8_maps_minus_one_one():
    images = torch.tensor([-1.0, 0.0, 1.0]).view(1, 1, 1, 3)
    pixels = to_uint8(images)
    assert pixels.dtype == torch.uint8
    assert pixels.flatten().tolist() == [0, 128, 255]


def test_load_eval_weights_prefers_ema():
    raw = TinyFlow()
    ema = TinyFlow()
    with torch.no_grad():
        raw.conv.weight.fill_(0.1)
        ema.conv.weight.fill_(0.7)
    loaded = TinyFlow()
    load_eval_weights(
        loaded,
        {"model": raw.state_dict(), "ema_model": ema.state_dict()},
        use_ema=True,
    )
    assert torch.allclose(loaded.conv.weight, ema.conv.weight)


def test_load_eval_weights_falls_back_to_raw():
    raw = TinyFlow()
    with torch.no_grad():
        raw.conv.weight.fill_(0.3)
    loaded = TinyFlow()
    load_eval_weights(loaded, {"model": raw.state_dict(), "ema_model": None}, use_ema=True)
    assert torch.allclose(loaded.conv.weight, raw.conv.weight)


def test_generate_uint8_shape_and_dtype():
    model = TinyFlow(channels=3).eval()
    images = generate_uint8(
        model,
        num_samples=4,
        image_size=8,
        channels=3,
        ode_steps=2,
        batch_size=2,
        device=torch.device("cpu"),
        seed=0,
    )
    assert images.shape == (4, 3, 8, 8)
    assert images.dtype == torch.uint8


def test_activation_statistics_identity_covariance():
    torch.manual_seed(0)
    features = torch.eye(4).repeat(8, 1).numpy()
    mean, covariance = activation_statistics(features)
    assert mean.shape == (4,)
    assert covariance.shape == (4, 4)


def test_frechet_distance_identical_is_zero():
    mean = np.zeros(3)
    cov = np.eye(3)
    assert frechet_distance(mean, cov, mean, cov) == pytest.approx(0.0, abs=1e-6)

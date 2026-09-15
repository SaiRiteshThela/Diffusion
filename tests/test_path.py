import pytest
import torch

from training.path import (
    GaussianConditionalProbabilityPath,
    IsotropicGaussian,
    LinearAlpha,
    LinearBeta,
)

from tests.helpers import TensorImages


def test_linear_alpha_beta_endpoints():
    t0 = torch.zeros(2, 1, 1, 1)
    t1 = torch.ones(2, 1, 1, 1)
    alpha = LinearAlpha()
    beta = LinearBeta()
    assert torch.allclose(alpha(t0), t0)
    assert torch.allclose(alpha(t1), t1)
    assert torch.allclose(beta(t0), torch.ones_like(t0))
    assert torch.allclose(beta(t1), torch.zeros_like(t1))
    assert torch.allclose(alpha.dt(t0), torch.ones_like(t0))
    assert torch.allclose(beta.dt(t0), -torch.ones_like(t0))


def test_isotropic_gaussian_sample_shape_and_std():
    gauss = IsotropicGaussian(shape=[1, 4, 4], std=2.0)
    x, y = gauss.sample(256)
    assert x.shape == (256, 1, 4, 4)
    assert y is None
    assert x.std().item() == pytest.approx(2.0, rel=0.25)


def test_gaussian_path_shapes_and_linear_cfm_field():
    data = TensorImages(torch.randn(16, 1, 4, 4))
    path = GaussianConditionalProbabilityPath(
        p_data=data,
        p_simple_shape=[1, 4, 4],
        alpha=LinearAlpha(),
        beta=LinearBeta(),
    )
    z, _ = path.sample_conditioning_variable(5)
    t = torch.full((5, 1, 1, 1), 0.4)
    x = path.sample_conditional_path(z, t)
    u = path.conditional_vector_field(x, z, t)
    score = path.conditional_score(x, z, t)
    assert x.shape == z.shape == u.shape == score.shape == (5, 1, 4, 4)

    expected_u = (z - x) / (1 - t)
    expected_score = (z * t - x) / (1 - t) ** 2
    assert torch.allclose(u, expected_u, atol=1e-5)
    assert torch.allclose(score, expected_score, atol=1e-5)


def test_sample_conditional_flow_is_stable_near_t1(monkeypatch):
    data = TensorImages(torch.randn(16, 1, 4, 4))
    path = GaussianConditionalProbabilityPath(
        p_data=data,
        p_simple_shape=[1, 4, 4],
        alpha=LinearAlpha(),
        beta=LinearBeta(),
    )
    z, _ = data.sample(5)
    noise = torch.full_like(z, 0.25)
    monkeypatch.setattr(torch, "randn_like", lambda value: noise)
    t = torch.full((5, 1, 1, 1), 1 - 1e-7)
    x, velocity = path.sample_conditional_flow(z, t)
    assert x.shape == velocity.shape == z.shape
    assert torch.isfinite(velocity).all()
    assert torch.allclose(velocity, z - noise)


def test_sample_marginal_path():
    data = TensorImages(torch.randn(16, 1, 4, 4))
    path = GaussianConditionalProbabilityPath(
        p_data=data,
        p_simple_shape=[1, 4, 4],
        alpha=LinearAlpha(),
        beta=LinearBeta(),
    )
    t = torch.rand(3, 1, 1, 1)
    x = path.sample_marginal_path(t)
    assert x.shape == (3, 1, 4, 4)


def test_path_at_t0_is_noise_scale():
    data = TensorImages(torch.ones(8, 1, 4, 4))
    path = GaussianConditionalProbabilityPath(
        p_data=data,
        p_simple_shape=[1, 4, 4],
        alpha=LinearAlpha(),
        beta=LinearBeta(),
    )
    z, _ = data.sample(8)
    t = torch.zeros(8, 1, 1, 1)
    x = path.sample_conditional_path(z, t)
    assert x.shape == z.shape
    assert not torch.allclose(x, z)


def test_path_at_t1_is_data():
    data = TensorImages(torch.randn(8, 1, 4, 4))
    path = GaussianConditionalProbabilityPath(
        p_data=data,
        p_simple_shape=[1, 4, 4],
        alpha=LinearAlpha(),
        beta=LinearBeta(),
    )
    z, _ = data.sample(8)
    t = torch.ones(8, 1, 1, 1)
    x = path.sample_conditional_path(z, t)
    assert torch.allclose(x, z)


def test_p_simple_moves_with_module():
    gauss = IsotropicGaussian(shape=[2, 2])
    cpu = gauss.sample(1)[0].device
    assert cpu.type == "cpu"
    if not torch.cuda.is_available():
        pytest.skip("cuda not available")
    gauss = gauss.cuda()
    assert gauss.sample(1)[0].device.type == "cuda"

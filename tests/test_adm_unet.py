"""Behavioral checks for the optional ADM velocity backbone."""

import torch
import pytest
import yaml

from models.adm_unet import ADMAttention, ADMUNet, FixedTimeEmbedding
from models.flow import FlowModel


def _small_model():
    return ADMUNet(
        image_size=8, model_channels=8, channel_mult=(1, 2), num_res_blocks=1,
        attention_resolutions=(4,), num_heads=2, norm_groups=4,
    )


def test_adm_zero_initialization_and_learning():
    torch.manual_seed(7)
    model = _small_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(2, 3, 8, 8)
    t = torch.tensor([0.2, 0.8])
    target = torch.randn_like(x)
    initial = model(x, t)
    assert initial.shape == x.shape
    assert torch.count_nonzero(initial) == 0
    # Zero output/residual projections intentionally block some gradients at
    # initialization. After several updates, gradients should reach the torso.
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x, t) - target).square().mean()
        loss.backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        optimizer.step()
    assert torch.count_nonzero(model.encoder[0][0].weight.grad) > 0
    assert torch.count_nonzero(model.time_embedding[1].weight.grad) > 0
    assert (model(x, t) - target).square().mean() < target.square().mean()
    model.eval()
    assert torch.allclose(model(x, t).flip(0), model(x.flip(0), t.flip(0)), atol=1e-6)
    assert not torch.allclose(model(x, torch.zeros(2)), model(x, torch.ones(2)))


def test_adm_attention_matches_explicit_attention():
    torch.manual_seed(3)
    attention = ADMAttention(channels=8, heads=2, groups=4)
    torch.nn.init.normal_(attention.projection.weight, std=0.1)
    x = torch.randn(2, 8, 3, 3)
    flat = x.flatten(2)
    q, k, v = attention.qkv(attention.norm(flat)).chunk(3, dim=1)
    q, k, v = [value.reshape(2, 2, 4, 9).transpose(-1, -2) for value in (q, k, v)]
    probabilities = (q @ k.transpose(-1, -2) / 2).softmax(dim=-1)
    h = (probabilities @ v).transpose(-1, -2).reshape(2, 8, 9)
    expected = (flat + attention.projection(h)).reshape_as(x)
    torch.testing.assert_close(attention(x), expected)


def test_adm_time_embedding_is_fixed_and_scaled():
    embedding = FixedTimeEmbedding(8)
    assert list(embedding.parameters()) == []
    actual = embedding(torch.tensor([0.0, 0.001]))
    torch.testing.assert_close(actual[0], torch.tensor([1., 1., 1., 1., 0., 0., 0., 0.]))
    torch.testing.assert_close(actual[1, 0], torch.tensor(1.).cos())
    torch.testing.assert_close(actual[1, 4], torch.tensor(1.).sin())


def test_adm_config_dispatch_and_checkpoint_roundtrip(tmp_path):
    config_path = tmp_path / "adm.yaml"
    config_path.write_text(yaml.safe_dump({
        "model_type": "adm",
        "adm_unet": {"image_size": 8, "model_channels": 8, "channel_mult": [1, 2],
                     "num_res_blocks": 1, "attention_resolutions": [4], "num_heads": 2, "norm_groups": 4},
    }))
    model = FlowModel.from_config(config_path)
    # Exercise nonzero predictions; fresh zero outputs would hide state errors.
    torch.nn.init.normal_(model.unet.output[-1].weight, std=0.01)
    restored = FlowModel.from_config(config_path)
    restored.load_state_dict(model.state_dict(), strict=True)
    x, t = torch.randn(2, 3, 8, 8), torch.rand(2)
    torch.testing.assert_close(model(x, t), restored(x, t))
    assert isinstance(model.unet, ADMUNet)
    assert isinstance(model.time_embed, torch.nn.Identity)


def test_adm_rejects_invalid_inputs_and_configuration():
    model = _small_model()
    with pytest.raises(ValueError, match="expected"):
        model(torch.randn(2, 3, 9, 9), torch.rand(2))
    with pytest.raises(ValueError, match="batch"):
        model(torch.randn(2, 3, 8, 8), torch.rand(1))
    with pytest.raises(ValueError, match="divisible"):
        ADMUNet(model_channels=10)
    with pytest.raises(ValueError, match="spatial sizes"):
        ADMUNet(attention_resolutions=(7,))


def test_adm_proposed_64px_model_shapes_without_allocation():
    with torch.device("meta"):
        model = ADMUNet()
        output = model(torch.randn(2, 3, 64, 64), torch.rand(2))
    assert output.shape == (2, 3, 64, 64)
    assert 10_000_000 < sum(parameter.numel() for parameter in model.parameters()) < 50_000_000

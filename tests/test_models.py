from pathlib import Path

import torch
import yaml

import pytest

from models.attention import SelfAttention
from models.config import get_activation, load_config
from models.downsample import Downsample
from models.ffn import FFNet
from models.flow import FlowModel
from models.residual import ResidualBlock
from models.time_embedding import SinusoidalTimeEmbeddings
from models.transformer import TransformerBlock
from models.unet import UNet
from models.upsample import Upsample

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "dummy.yaml"
MNIST_CONFIG = ROOT / "configs" / "mnist.yaml"


def _config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def test_self_attention():
    x = torch.randn(2, 10, 128)
    out = SelfAttention(n_heads=4, d_embed=128)(x)
    assert out.shape == x.shape


def test_ffn():
    x = torch.randn(2, 10, 128)
    out = FFNet(d_embed=128, d_ff=512)(x)
    assert out.shape == x.shape


def test_transformer_block():
    x = torch.randn(2, 64, 8, 8)
    out = TransformerBlock(d_embed=64, n_heads=4, d_ff=256)(x)
    assert out.shape == x.shape


def test_time_embedding():
    cfg = _config()["time_embedding"]
    t = torch.tensor([0.1, 0.5, 0.9])
    out = SinusoidalTimeEmbeddings(**cfg)(t)
    assert out.shape == (3, cfg["scaled_time_emb_dim"])


def test_residual_block():
    x = torch.randn(2, 64, 16, 16)
    t = torch.randn(2, 256)
    out = ResidualBlock(64, 128, 16, 256)(x, t)
    assert out.shape == (2, 128, 16, 16)


def test_upsample():
    x = torch.randn(2, 64, 16, 16)
    out = Upsample(64, 64)(x)
    assert out.shape == (2, 64, 32, 32)


def test_downsample():
    x = torch.randn(2, 64, 16, 16)
    out = Downsample(64, 64)(x)
    assert out.shape == (2, 64, 8, 8)


def test_unet_from_dummy_config():
    cfg = _config()
    model = UNet.from_config(CONFIG_PATH)
    B = cfg["test"]["batch_size"]
    H = cfg["test"]["image_size"]
    C = cfg["unet"]["in_channels"]
    x = torch.randn(B, C, H, H)
    t = torch.randn(B, cfg["unet"]["time_emb_dim"])
    out = model(x, t)
    assert out.shape == x.shape


def test_time_embed_matches_unet():
    cfg = _config()
    embed = SinusoidalTimeEmbeddings(**cfg["time_embedding"])
    unet = UNet.from_config(CONFIG_PATH)
    B = cfg["test"]["batch_size"]
    H = cfg["test"]["image_size"]
    t = torch.rand(B)
    time_emb = embed(t)
    x = torch.randn(B, cfg["unet"]["in_channels"], H, H)
    out = unet(x, time_emb)
    assert time_emb.shape[-1] == cfg["unet"]["time_emb_dim"]
    assert out.shape == x.shape


def test_flow_model():
    cfg = _config()
    model = FlowModel.from_config(CONFIG_PATH)
    B = cfg["test"]["batch_size"]
    H = cfg["test"]["image_size"]
    xt = torch.randn(B, cfg["unet"]["in_channels"], H, H)
    t = torch.rand(B)
    u = model(xt, t)
    assert u.shape == xt.shape


def test_flow_model_batch_mismatch():
    model = FlowModel.from_config(CONFIG_PATH)
    xt = torch.randn(2, 3, 32, 32)
    with pytest.raises(ValueError, match="batch"):
        model(xt, torch.rand(3))


def test_flow_model_mnist_config():
    cfg = load_config(MNIST_CONFIG)
    model = FlowModel.from_config(MNIST_CONFIG)
    B = cfg["test"]["batch_size"]
    H = cfg["test"]["image_size"]
    C = cfg["unet"]["in_channels"]
    assert C == 1
    xt = torch.randn(B, C, H, H)
    t = torch.rand(B)
    u = model(xt, t)
    assert u.shape == xt.shape


def test_load_config_and_activation():
    cfg = load_config(CONFIG_PATH)
    assert "unet" in cfg
    assert get_activation("gelu") is not None
    assert get_activation(torch.nn.functional.relu) is torch.nn.functional.relu
    with pytest.raises(ValueError, match="unknown activation"):
        get_activation("not-an-act")


def test_self_attention_train_and_eval():
    x = torch.randn(2, 10, 128)
    attn = SelfAttention(n_heads=4, d_embed=128, attn_dropout=0.1)
    attn.train()
    assert attn(x).shape == x.shape
    attn.eval()
    assert attn(x).shape == x.shape


def test_residual_same_channels():
    x = torch.randn(2, 64, 8, 8)
    t = torch.randn(2, 256)
    out = ResidualBlock(64, 64, 16, 256)(x, t)
    assert out.shape == x.shape


def test_unet_rejects_bad_head_divisibility():
    with pytest.raises(ValueError, match="n_heads"):
        UNet(in_channels=3, start_dim=64, dim_mults=(1,), n_heads=3, group_norm_num_groups=16)

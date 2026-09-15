import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import SelfAttention
from models.ffn import FFNet


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_embed,
        n_heads,
        d_ff,
        attn_dropout=0.1,
        ffn_dropout=0.1,
        proj_dropout=0.1,
        activation=F.gelu,
    ):
        super().__init__()
        self.d_embed = d_embed
        self.ln1 = nn.LayerNorm(d_embed)
        self.attn = SelfAttention(n_heads, d_embed, attn_dropout, proj_dropout)
        self.ln2 = nn.LayerNorm(d_embed)
        self.ff = FFNet(d_embed, d_ff, ffn_dropout, activation)

    def forward(self, x):
        B, C, H, W = x.size()
        assert C == self.d_embed, f"expected channels={self.d_embed}, got {C}"
        x = x.reshape(B, C, H * W).transpose(1, 2)
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x.transpose(1, 2).reshape(B, C, H, W)


if __name__ == "__main__":
    model = TransformerBlock(d_embed=64, n_heads=4, d_ff=256)
    out = model(torch.randn(2, 64, 8, 8))
    assert out.shape == (2, 64, 8, 8), out.shape
    print("TransformerBlock", tuple(out.shape))

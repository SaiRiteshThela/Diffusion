import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, attn_dropout=0.1, proj_dropout=0.1):
        super().__init__()
        assert d_embed % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        self.attn_dropout = attn_dropout

        self.query = nn.Linear(d_embed, d_embed, bias=False)
        self.key = nn.Linear(d_embed, d_embed, bias=False)
        self.value = nn.Linear(d_embed, d_embed, bias=False)
        self.out = nn.Linear(d_embed, d_embed, bias=False)
        self.dropout = nn.Dropout(proj_dropout)

    def forward(self, x):
        B, T, C = x.size()
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        dropout_p = self.attn_dropout if self.training else 0.0
        if q.device.type == "mps" and dropout_p > 0:
            # MPS fused SDPA does not support dropout. Apply it to the
            # attention probabilities, matching SDPA semantics.
            scale = self.d_head**-0.5
            weights = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
            weights = F.dropout(weights, p=dropout_p, training=True)
            attn = weights @ v
        else:
            attn = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.out(attn))


if __name__ == "__main__":
    model = SelfAttention(n_heads=4, d_embed=128)
    out = model(torch.randn(2, 10, 128))
    assert out.shape == (2, 10, 128), out.shape
    print("SelfAttention", tuple(out.shape))

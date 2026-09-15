import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNet(nn.Module):
    def __init__(self, d_embed, d_ff, dropout=0.1, activation=F.gelu):
        super().__init__()
        self.linear1 = nn.Linear(d_embed, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_embed)
        self.activation = activation

    def forward(self, x):
        x = self.dropout(self.activation(self.linear1(x)))
        return self.linear2(x)


if __name__ == "__main__":
    model = FFNet(d_embed=128, d_ff=512)
    out = model(torch.randn(2, 10, 128))
    assert out.shape == (2, 10, 128), out.shape
    print("FFNet", tuple(out.shape))

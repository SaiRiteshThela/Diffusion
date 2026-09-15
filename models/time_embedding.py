import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, time_emb_dim, scaled_time_emb_dim):
        super().__init__()
        assert time_emb_dim % 2 == 0
        self.half_dim = time_emb_dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, scaled_time_emb_dim),
            nn.SiLU(),
        )

    def forward(self, time_steps):
        t = time_steps.view(-1, 1).float()
        freqs = t * self.weights * 2 * math.pi
        embeddings = torch.cat((torch.sin(freqs), torch.cos(freqs)), dim=-1) * math.sqrt(2)
        return self.time_mlp(embeddings)


if __name__ == "__main__":
    model = SinusoidalTimeEmbeddings(time_emb_dim=128, scaled_time_emb_dim=256)
    out = model(torch.tensor([0.1, 0.5, 0.9]))
    assert out.shape == (3, 256), out.shape
    print("SinusoidalTimeEmbeddings", tuple(out.shape))

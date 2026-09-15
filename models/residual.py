import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group_norm_num_groups, time_emb_dim):
        super().__init__()
        assert in_channels % group_norm_num_groups == 0, (
            f"in_channels={in_channels} must be divisible by num_groups={group_norm_num_groups}"
        )
        assert out_channels % group_norm_num_groups == 0, (
            f"out_channels={out_channels} must be divisible by num_groups={group_norm_num_groups}"
        )
        self.time_emb_dim = time_emb_dim
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        self.groupnorm1 = nn.GroupNorm(num_groups=group_norm_num_groups, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")
        self.groupnorm2 = nn.GroupNorm(num_groups=group_norm_num_groups, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same")

        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, time_emb):
        assert time_emb.shape[0] == x.shape[0], (
            f"time batch {time_emb.shape[0]} != x batch {x.shape[0]}"
        )
        assert time_emb.shape[-1] == self.time_emb_dim, (
            f"expected time_emb_dim={self.time_emb_dim}, got {time_emb.shape[-1]}"
        )
        identity = x
        time_emb = self.time_emb_proj(time_emb)
        x = self.conv1(F.relu(self.groupnorm1(x)))
        x = x + time_emb.unsqueeze(-1).unsqueeze(-1)
        x = self.conv2(F.relu(self.groupnorm2(x)))
        x = x + self.residual(identity)
        return x


if __name__ == "__main__":
    model = ResidualBlock(
        in_channels=64,
        out_channels=128,
        group_norm_num_groups=16,
        time_emb_dim=256,
    )
    out = model(torch.randn(2, 64, 16, 16), torch.randn(2, 256))
    assert out.shape == (2, 128, 16, 16), out.shape
    print("ResidualBlock", tuple(out.shape))

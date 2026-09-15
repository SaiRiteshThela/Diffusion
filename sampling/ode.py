from abc import ABC, abstractmethod

import torch
from torch import Tensor
from tqdm import tqdm

from models.flow import FlowModel


class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: Tensor, t: Tensor, **kwargs) -> Tensor:
        """
        Args:
            - xt: (bs, c, h, w)
            - t: (bs, 1, 1, 1) or (bs,)
        Returns:
            - drift: (bs, c, h, w)
        """
        pass


class Simulator(ABC):
    @abstractmethod
    def step(self, xt: Tensor, t: Tensor, dt: Tensor, **kwargs):
        pass

    @torch.no_grad()
    def simulate(self, x: Tensor, ts: Tensor, use_tqdm: bool = True, **kwargs):
        nts = ts.shape[1]
        steps = range(nts - 1)
        if use_tqdm:
            steps = tqdm(steps)
        for t_idx in steps:
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
        return x

    @torch.no_grad()
    def simulate_with_trajectory(self, x: Tensor, ts: Tensor, use_tqdm: bool = True, **kwargs):
        xs = [x.clone()]
        nts = ts.shape[1]
        steps = range(nts - 1)
        if use_tqdm:
            steps = tqdm(steps)
        for t_idx in steps:
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)


class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode

    def step(self, xt: Tensor, t: Tensor, h: Tensor, **kwargs):
        h = h.view([-1] + [1] * (len(xt.shape) - 1))
        return xt + self.ode.drift_coefficient(xt, t, **kwargs) * h


class FlowODE(ODE):
    """Wraps FlowModel so Euler can call it. Lab t is (B,1,1,1); FlowModel wants (B,)."""

    def __init__(self, flow: FlowModel):
        self.flow = flow

    def drift_coefficient(self, xt: Tensor, t: Tensor, **kwargs) -> Tensor:
        return self.flow(xt, t.reshape(xt.shape[0]))

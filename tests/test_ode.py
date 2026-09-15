import torch

from sampling.ode import EulerSimulator, FlowODE, ODE
from tests.helpers import TinyFlow


class ConstantODE(ODE):
    def drift_coefficient(self, xt, t, **kwargs):
        return torch.ones_like(xt)


def test_euler_constant_drift():
    sim = EulerSimulator(ConstantODE())
    x0 = torch.zeros(2, 1, 4, 4)
    ts = torch.linspace(0, 1, 11).view(1, -1, 1, 1, 1).expand(2, -1, 1, 1, 1)
    x1 = sim.simulate(x0, ts)
    assert x1.shape == x0.shape
    assert torch.allclose(x1, torch.ones_like(x1), atol=1e-5)


def test_euler_broadcasts_h():
    sim = EulerSimulator(ConstantODE())
    xt = torch.zeros(3, 2, 5, 5)
    t = torch.zeros(3, 1, 1, 1)
    h = torch.full((3,), 0.25)
    nxt = sim.step(xt, t, h)
    assert nxt.shape == xt.shape
    assert torch.allclose(nxt, torch.full_like(xt, 0.25))


def test_simulate_with_trajectory():
    sim = EulerSimulator(ConstantODE())
    x0 = torch.zeros(1, 1, 2, 2)
    ts = torch.linspace(0, 1, 5).view(1, -1, 1, 1, 1)
    traj = sim.simulate_with_trajectory(x0, ts)
    assert traj.shape == (1, 5, 1, 2, 2)
    assert torch.allclose(traj[:, 0], x0)
    assert torch.allclose(traj[:, -1], torch.ones_like(x0), atol=1e-5)


def test_flow_ode_flattens_time_and_matches_model():
    flow = TinyFlow().eval()
    ode = FlowODE(flow)
    xt = torch.randn(2, 1, 4, 4)
    t = torch.rand(2, 1, 1, 1)
    u = ode.drift_coefficient(xt, t)
    expected = flow(xt, t.reshape(2))
    assert u.shape == xt.shape
    assert torch.allclose(u, expected)


def test_euler_with_flow_ode():
    flow = TinyFlow().eval()
    sim = EulerSimulator(FlowODE(flow))
    x0 = torch.randn(2, 1, 4, 4)
    ts = torch.linspace(0, 1, 4).view(1, -1, 1, 1, 1).expand(2, -1, 1, 1, 1)
    x1 = sim.simulate(x0, ts)
    assert x1.shape == x0.shape
    assert torch.isfinite(x1).all()

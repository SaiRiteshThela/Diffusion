from pathlib import Path

import torch
import torch.nn as nn

from tests.helpers import TensorImages, TinyFlow
from training.path import GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta
from training.trainer import FlowTrainer, model_size_b


def _path(channels: int = 1, size: int = 4):
    data = TensorImages(torch.randn(32, channels, size, size))
    return GaussianConditionalProbabilityPath(
        p_data=data,
        p_simple_shape=[channels, size, size],
        alpha=LinearAlpha(),
        beta=LinearBeta(),
    )


def test_model_size_b():
    linear = nn.Linear(4, 4)
    assert model_size_b(linear) == sum(p.numel() * p.element_size() for p in linear.parameters())


def test_flow_trainer_loss_is_finite_and_trainable():
    path = _path()
    model = TinyFlow()
    trainer = FlowTrainer(path, model)
    loss = trainer.get_train_loss(batch_size=4)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_flow_trainer_train_and_checkpoint(tmp_path: Path):
    path = _path()
    val_path = _path()
    model = TinyFlow()
    ckpt = tmp_path / "flow.pt"
    samples = tmp_path / "samples"
    trainer = FlowTrainer(path, model, val_path=val_path)
    hist = trainer.train(
        num_steps=2,
        device=torch.device("cpu"),
        lr=1e-3,
        batch_size=4,
        ckpt_path=ckpt,
        checkpoint_every=1,
        val_every=1,
        val_batches=2,
        plot_every=1,
        n_plot_images=2,
        n_plot_steps=3,
        samples_dir=samples,
        show_plots=False,
    )
    assert hist["train"].shape == (2,)
    assert hist["val"].shape == (2,)
    assert hist["val_steps"].tolist() == [1, 2]
    assert not model.training
    assert list(samples.glob("traj_step_*.png"))
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "model" in saved
    assert "optimizer" in saved
    assert saved["num_steps"] == 2
    assert saved["step"] == 2
    assert saved["train_kwargs"]["batch_size"] == 4
    model.load_state_dict(saved["model"])

    resumed = FlowTrainer(path, TinyFlow(), val_path=val_path)
    resumed_hist = resumed.train(
        num_steps=3,
        device=torch.device("cpu"),
        lr=1e-3,
        batch_size=4,
        ckpt_path=ckpt,
        checkpoint_every=1,
        val_every=1,
        val_batches=1,
        plot_every=0,
        resume_from=ckpt,
    )
    assert resumed_hist["train"].shape == (3,)
    assert resumed_hist["val_steps"].tolist() == [1, 2, 3]


def test_flow_trainer_val_loss():
    trainer = FlowTrainer(_path(), TinyFlow(), val_path=_path())
    val = trainer.get_val_loss(batch_size=4)
    assert val is not None
    assert torch.isfinite(val)
    assert FlowTrainer(_path(), TinyFlow()).get_val_loss(batch_size=4) is None

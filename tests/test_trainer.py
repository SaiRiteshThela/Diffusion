from pathlib import Path

import pytest
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


def test_optimizer_selection():
    trainer = FlowTrainer(_path(), TinyFlow())
    assert isinstance(trainer.get_optimizer(1e-3, "adam"), torch.optim.Adam)
    assert isinstance(trainer.get_optimizer(1e-3, "adamw"), torch.optim.AdamW)
    with pytest.raises(ValueError, match="unknown optimizer"):
        trainer.get_optimizer(1e-3, "sgd")


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
    best_ckpt = tmp_path / "flow_best.pt"
    samples = tmp_path / "samples"
    logged = []

    class FakeRun:
        def log(self, data, step):
            logged.append((data, step))

    trainer = FlowTrainer(path, model, val_path=val_path)
    hist = trainer.train(
        num_steps=2,
        device=torch.device("cpu"),
        lr=1e-3,
        lr_milestones=[1],
        lr_gamma=0.5,
        ema_decay=0.9,
        batch_size=4,
        ckpt_path=ckpt,
        best_ckpt_path=best_ckpt,
        checkpoint_every=1,
        val_every=1,
        val_batches=2,
        plot_every=1,
        n_plot_images=2,
        n_plot_steps=3,
        samples_dir=samples,
        show_plots=False,
        wandb_run=FakeRun(),
    )
    assert hist["train"].shape == (2,)
    assert hist["val"].shape == (3,)
    assert hist["val_steps"].tolist() == [0, 1, 2]
    assert not model.training
    assert list(samples.glob("traj_step_*.png"))
    assert [step for _, step in logged] == [0, 1, 2]
    assert "loss/train" in logged[-1][0]
    assert "loss/val" in logged[-1][0]
    assert "samples/trajectory" in logged[-1][0]
    assert logged[1][0]["learning_rate"] == pytest.approx(5e-4)
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    best_saved = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    assert "model" in saved
    assert saved["ema_model"] is not None
    assert "optimizer" in saved
    assert saved["scheduler"] is not None
    assert saved["num_steps"] == 2
    assert saved["step"] == 2
    assert saved["train_kwargs"]["batch_size"] == 4
    assert best_saved["step"] in {0, 1, 2}
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
    assert resumed_hist["val_steps"].tolist() == [0, 1, 2, 3]


def test_flow_trainer_val_loss():
    trainer = FlowTrainer(_path(), TinyFlow(), val_path=_path())
    val = trainer.get_val_loss(batch_size=4)
    assert val is not None
    assert torch.isfinite(val)
    assert FlowTrainer(_path(), TinyFlow()).get_val_loss(batch_size=4) is None


def test_cosine_schedule_decays_learning_rate():
    logged = []

    class FakeRun:
        def log(self, data, step):
            if "learning_rate" in data:
                logged.append(data["learning_rate"])

    trainer = FlowTrainer(_path(), TinyFlow())
    trainer.train(
        num_steps=6,
        device=torch.device("cpu"),
        lr=1e-3,
        lr_schedule="cosine",
        lr_warmup_steps=0,
        batch_size=4,
        checkpoint_every=0,
        val_every=0,
        plot_every=0,
        wandb_run=FakeRun(),
    )
    assert logged
    assert logged[-1] < logged[0]


def test_abort_train_loss_stops_before_backward():
    trainer = FlowTrainer(_path(), TinyFlow())
    values = iter([0.5, 0.25, 99.0, 0.1])

    def fake_loss(batch_size: int):
        del batch_size
        return torch.tensor(next(values), requires_grad=True)

    trainer.get_train_loss = fake_loss  # type: ignore[method-assign]
    hist = trainer.train(
        num_steps=10,
        device=torch.device("cpu"),
        lr=1e-3,
        batch_size=4,
        abort_train_loss=10.0,
        checkpoint_every=0,
        val_every=0,
        plot_every=0,
    )
    assert hist["train"].shape == (3,)
    assert hist["train"][-1].item() == pytest.approx(99.0)


def test_scheduler_num_steps_keeps_higher_lr():
    short = []
    long = []

    class FakeRun:
        def __init__(self, bucket):
            self.bucket = bucket

        def log(self, data, step):
            if "learning_rate" in data:
                self.bucket.append(data["learning_rate"])

    kwargs = dict(
        num_steps=4,
        device=torch.device("cpu"),
        lr=1e-3,
        lr_schedule="cosine",
        lr_warmup_steps=0,
        batch_size=4,
        checkpoint_every=0,
        val_every=0,
        plot_every=0,
    )
    FlowTrainer(_path(), TinyFlow()).train(**kwargs, wandb_run=FakeRun(short))
    FlowTrainer(_path(), TinyFlow()).train(
        **kwargs,
        scheduler_num_steps=100,
        wandb_run=FakeRun(long),
    )
    assert long[-1] > short[-1]


def test_scheduler_num_steps_must_cover_training():
    trainer = FlowTrainer(_path(), TinyFlow())
    with pytest.raises(ValueError, match="scheduler_num_steps"):
        trainer.train(
            num_steps=4,
            device=torch.device("cpu"),
            lr=1e-3,
            lr_schedule="cosine",
            scheduler_num_steps=2,
            batch_size=4,
            checkpoint_every=0,
            val_every=0,
            plot_every=0,
        )


def test_cosine_rejects_milestones():
    trainer = FlowTrainer(_path(), TinyFlow())
    with pytest.raises(ValueError, match="not both"):
        trainer.train(
            num_steps=2,
            device=torch.device("cpu"),
            lr=1e-3,
            lr_milestones=[1],
            lr_schedule="cosine",
            batch_size=4,
            checkpoint_every=0,
            val_every=0,
            plot_every=0,
        )


def test_checkpoint_uses_configured_staging_directory(tmp_path, monkeypatch):
    staging = tmp_path / "staging"
    monkeypatch.setenv("DIFFUSION_CHECKPOINT_TMPDIR", str(staging))
    checkpoint = tmp_path / "saved.pt"
    trainer = FlowTrainer(_path(), TinyFlow())
    trainer.train(
        num_steps=1, device=torch.device("cpu"), batch_size=4,
        ckpt_path=checkpoint, checkpoint_every=0, val_every=0, plot_every=0,
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["step"] == 1
    assert staging.exists()
    assert list(staging.iterdir()) == []


def test_failed_checkpoint_save_cleans_staging_and_preserves_previous(tmp_path, monkeypatch):
    staging = tmp_path / "staging"
    monkeypatch.setenv("DIFFUSION_CHECKPOINT_TMPDIR", str(staging))
    checkpoint = tmp_path / "saved.pt"
    checkpoint.write_bytes(b"previous checkpoint")

    def failing_save(payload, path):
        assert Path(path).parent == staging
        Path(path).write_bytes(b"partial checkpoint")
        raise RuntimeError("simulated disk full")

    monkeypatch.setattr(torch, "save", failing_save)
    trainer = FlowTrainer(_path(), TinyFlow())
    with pytest.raises(RuntimeError, match="simulated disk full"):
        trainer.train(
            num_steps=1, device=torch.device("cpu"), batch_size=4,
            ckpt_path=checkpoint, checkpoint_every=0, val_every=0, plot_every=0,
        )
    assert checkpoint.read_bytes() == b"previous checkpoint"
    assert list(staging.iterdir()) == []

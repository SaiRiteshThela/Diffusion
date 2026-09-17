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
    # Rejected forward passes are not completed optimizer steps.
    assert hist["train"].shape == (2,)
    assert hist["train"][-1].item() == pytest.approx(0.25)
    assert trainer.completed_steps == 2


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


def test_validation_seed_is_repeatable_and_does_not_change_training_rng():
    torch.manual_seed(19)
    path, val_path = _path(), _path()
    starting = TinyFlow().state_dict()
    outcomes = []
    for val_every in (0, 1):
        model = TinyFlow()
        model.load_state_dict(starting)
        trainer = FlowTrainer(path, model, val_path=val_path)
        torch.manual_seed(123)
        hist = trainer.train(
            num_steps=3, device=torch.device("cpu"), lr=0.0, batch_size=4,
            val_every=val_every, val_batches=2, validation_seed=987,
            checkpoint_every=0, plot_every=0,
        )
        outcomes.append((hist, torch.random.get_rng_state()))
    assert torch.equal(outcomes[0][0]["train"], outcomes[1][0]["train"])
    assert torch.equal(outcomes[0][1], outcomes[1][1])
    vals = outcomes[1][0]["val"]
    assert torch.equal(vals, vals[0].expand_as(vals))


def test_resume_restores_rng_optimizer_scheduler_and_ema(tmp_path):
    torch.manual_seed(23)
    path, val_path = _path(), _path()
    initial = TinyFlow().state_dict()
    common = dict(
        device=torch.device("cpu"), lr=1e-3, batch_size=4,
        lr_schedule="cosine", lr_warmup_steps=1, scheduler_num_steps=6,
        ema_decay=0.9, ema_warmup=True, val_every=2, val_batches=2,
        validation_seed=1234, checkpoint_every=0, plot_every=0,
    )
    full_model = TinyFlow()
    full_model.load_state_dict(initial)
    torch.manual_seed(41)
    FlowTrainer(path, full_model, val_path).train(num_steps=6, ckpt_path=tmp_path / "full.pt", **common)
    partial_model = TinyFlow()
    partial_model.load_state_dict(initial)
    torch.manual_seed(41)
    FlowTrainer(path, partial_model, val_path).train(num_steps=3, ckpt_path=tmp_path / "split.pt", **common)
    # Model initialization consumes RNG; the checkpoint must undo that on resume.
    resumed = FlowTrainer(path, TinyFlow(), val_path)
    resumed.train(num_steps=6, ckpt_path=tmp_path / "split.pt", resume_from=tmp_path / "split.pt", **common)
    full = torch.load(tmp_path / "full.pt", weights_only=False)
    split = torch.load(tmp_path / "split.pt", weights_only=False)
    for field in ("model", "ema_model", "history"):
        for name in full[field]:
            assert torch.equal(full[field][name], split[field][name]), (field, name)
    assert full["scheduler"] == split["scheduler"]
    assert torch.equal(full["rng_state"]["cpu"], split["rng_state"]["cpu"])
    assert split["step"] == resumed.completed_steps == 6
    for key in full["optimizer"]["state"]:
        for field in full["optimizer"]["state"][key]:
            assert torch.equal(full["optimizer"]["state"][key][field], split["optimizer"]["state"][key][field])


def test_expired_deadline_saves_without_running_training_or_validation(tmp_path):
    trainer = FlowTrainer(_path(), TinyFlow(), val_path=_path())

    def unexpected(**kwargs):
        pytest.fail("expired deadline must not start a forward pass")

    trainer.get_train_loss = unexpected
    trainer.get_val_loss = unexpected
    checkpoint = tmp_path / "deadline.pt"
    hist = trainer.train(
        num_steps=4, device=torch.device("cpu"), batch_size=4,
        deadline_timestamp=0.0, handle_signals=True, ckpt_path=checkpoint,
    )
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["step"] == trainer.completed_steps == 0
    assert saved["stop_reason"] == "deadline"
    assert hist["train"].numel() == 0


@pytest.mark.parametrize("signal_name", ["SIGINT", "SIGTERM"])
def test_stop_signal_finishes_current_update_and_restores_handler(tmp_path, signal_name):
    import signal

    signum = getattr(signal, signal_name)
    previous_handler = signal.getsignal(signum)
    trainer = FlowTrainer(_path(), TinyFlow())
    original = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    real_loss = trainer.get_train_loss

    def signal_during_forward(**kwargs):
        value = real_loss(**kwargs)
        signal.raise_signal(signum)
        return value

    trainer.get_train_loss = signal_during_forward
    checkpoint = tmp_path / f"{signal_name}.pt"
    hist = trainer.train(
        num_steps=4, device=torch.device("cpu"), batch_size=4,
        handle_signals=True, ema_decay=0.9, ckpt_path=checkpoint,
        lr_schedule="cosine", val_every=0, plot_every=0, checkpoint_every=0,
    )
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["step"] == trainer.completed_steps == len(hist["train"]) == 1
    assert saved["scheduler"]["last_epoch"] == 1
    assert all(float(item["step"]) == 1 for item in saved["optimizer"]["state"].values())
    assert saved["stop_reason"] == f"signal:{signal_name}"
    assert signal.getsignal(signum) == previous_handler
    for key, value in saved["ema_model"].items():
        torch.testing.assert_close(value, original[key].lerp(saved["model"][key], 0.1))


def test_nonfinite_gradient_does_not_commit_optimizer_step(tmp_path):
    trainer = FlowTrainer(_path(), TinyFlow())
    original = {key: value.clone() for key, value in trainer.model.state_dict().items()}

    def invalid_gradient(**kwargs):
        value = trainer.model.conv.weight.sum()
        return (value - value.detach()).sqrt()

    trainer.get_train_loss = invalid_gradient
    checkpoint = tmp_path / "invalid.pt"
    trainer.train(
        num_steps=3, device=torch.device("cpu"), batch_size=4,
        ckpt_path=checkpoint, val_every=0, plot_every=0, checkpoint_every=0,
    )
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["step"] == 0
    assert saved["stop_reason"] == "nonfinite_gradient"
    assert saved["history"]["train"].numel() == 0
    for key, value in saved["model"].items():
        assert torch.equal(value, original[key])


def test_failed_destination_copy_keeps_previous_checkpoint(tmp_path, monkeypatch):
    import training.trainer as trainer_module

    staging = tmp_path / "staging"
    monkeypatch.setenv("DIFFUSION_CHECKPOINT_TMPDIR", str(staging))
    checkpoint = tmp_path / "saved.pt"
    checkpoint.write_bytes(b"previous checkpoint")

    def failing_copy(source, destination):
        assert Path(destination) != checkpoint
        Path(destination).write_bytes(b"incomplete transfer")
        raise OSError("simulated destination write failure")

    monkeypatch.setattr(trainer_module.shutil, "copyfile", failing_copy)
    trainer = FlowTrainer(_path(), TinyFlow())
    with pytest.raises(OSError, match="destination write"):
        trainer.train(
            num_steps=1, device=torch.device("cpu"), batch_size=4,
            ckpt_path=checkpoint, val_every=0, plot_every=0, checkpoint_every=0,
        )
    assert checkpoint.read_bytes() == b"previous checkpoint"
    assert not list(staging.iterdir())
    assert not list(tmp_path.glob(".*.partial"))


def test_preview_uses_sampling_steps_separate_from_display_frames(tmp_path):
    trainer = FlowTrainer(_path(), TinyFlow())
    calls = []
    hook = trainer.model.register_forward_hook(lambda _model, inputs, _out: calls.append(inputs[1].clone()))
    try:
        trainer.plot_trajectory(
            step=1, n_images=2, n_steps=4, n_sampling_steps=10,
            samples_dir=tmp_path, show=False,
        )
    finally:
        hook.remove()
    assert len(calls) == 10
    torch.testing.assert_close(torch.stack(calls)[:, 0], torch.arange(10) / 10)
    assert (tmp_path / "traj_step_000001.png").exists()


def test_bf16_velocity_prediction_reduces_loss_in_fp32():
    class ReducedPrecisionFlow(TinyFlow):
        def forward(self, xt, t):
            return super().forward(xt, t).to(torch.bfloat16)

    trainer = FlowTrainer(_path(), ReducedPrecisionFlow())
    loss = trainer.get_train_loss(batch_size=4)
    assert loss.dtype == torch.float32
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in trainer.model.parameters())


def test_optimizer_betas_and_gradient_logging():
    trainer = FlowTrainer(_path(), TinyFlow())
    opt = trainer.get_optimizer(1e-3, "adamw", optimizer_betas=(0.9, 0.95))
    assert opt.param_groups[0]["betas"] == (0.9, 0.95)
    logged = []

    class FakeRun:
        def log(self, data, step):
            logged.append(data)

    trainer.train(
        num_steps=1, device=torch.device("cpu"), batch_size=4,
        optimizer_betas=(0.9, 0.95), max_grad_norm=1e-8,
        val_every=0, plot_every=0, checkpoint_every=0, wandb_run=FakeRun(),
    )
    assert logged[0]["grad/norm"] > 1e-8
    assert logged[0]["grad/clipped"] == 1.0


def test_fixed_preview_noise_is_repeatable_and_isolated_from_training_rng():
    torch.manual_seed(51)
    path = _path()
    initial = TinyFlow().state_dict()
    records = []
    for plot_every in (0, 1):
        model = TinyFlow()
        model.load_state_dict(initial)
        trainer = FlowTrainer(path, model)
        noises = []

        def fake_plot(**kwargs):
            noises.append(torch.randn(8))
            return None

        trainer.plot_trajectory = fake_plot
        torch.manual_seed(92)
        hist = trainer.train(
            num_steps=2, device=torch.device("cpu"), batch_size=4,
            val_every=0, plot_every=plot_every, plot_seed=4321,
            n_sampling_steps=32, checkpoint_every=0,
        )
        records.append((hist, torch.random.get_rng_state(), noises))
    assert torch.equal(records[0][0]["train"], records[1][0]["train"])
    assert torch.equal(records[0][1], records[1][1])
    assert torch.equal(records[1][2][0], records[1][2][1])


def test_deadline_during_forward_finishes_update_and_skips_further_work(tmp_path, monkeypatch):
    import training.trainer as trainer_module

    clock = [10.0]
    monkeypatch.setattr(trainer_module.time, "time", lambda: clock[0])
    trainer = FlowTrainer(_path(), TinyFlow())
    real_loss = trainer.get_train_loss

    def expire_during_forward(**kwargs):
        loss = real_loss(**kwargs)
        clock[0] = 21.0
        return loss

    trainer.get_train_loss = expire_during_forward
    trainer.plot_trajectory = lambda **kwargs: pytest.fail("preview must not start after deadline")
    checkpoint = tmp_path / "deadline.pt"
    hist = trainer.train(
        num_steps=4, device=torch.device("cpu"), batch_size=4,
        deadline_timestamp=20.0, handle_signals=True,
        ema_decay=0.9, lr_schedule="cosine", ckpt_path=checkpoint,
        val_every=0, plot_every=1, checkpoint_every=0,
    )
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["step"] == trainer.completed_steps == len(hist["train"]) == 1
    assert saved["scheduler"]["last_epoch"] == 1
    assert saved["stop_reason"] == "deadline"
    assert all(float(item["step"]) == 1 for item in saved["optimizer"]["state"].values())

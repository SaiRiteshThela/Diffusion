import signal
import subprocess

from experiments import campaign_12h


def test_supervisor_kills_stuck_worker_before_global_deadline(monkeypatch, tmp_path):
    class StuckProcess:
        pid = 4321
        waits = 0

        def poll(self):
            return None

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                assert timeout == 7
                raise subprocess.TimeoutExpired("worker", timeout)
            return -9

    process = StuckProcess()
    killed = []
    monkeypatch.setattr(campaign_12h.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(campaign_12h.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(campaign_12h.time, "time", lambda: 90)
    result = campaign_12h.execute("train", until=50, deadline=100, log=tmp_path / "worker.log")
    assert result == -9
    assert killed == [(4321, signal.SIGTERM), (4321, signal.SIGKILL)]


def test_supervisor_preserves_successful_exit(monkeypatch, tmp_path):
    class Finished:
        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(campaign_12h.subprocess, "Popen", lambda *a, **k: Finished())
    assert campaign_12h.execute("benchmark", until=100, deadline=200, log=tmp_path / "ok.log") == 0


def test_json_publication_preserves_structured_state(tmp_path):
    path = tmp_path / "nested" / "status.json"
    payload = {"phase": "training", "deadline_unix": 123, "evaluations": []}
    campaign_12h.write_json(path, payload)
    assert campaign_12h.read_json(path) == payload
    assert not path.with_name("status.json.partial").exists()


def _screen(name, fid, seed=42):
    return {"name": name, "fid": fid, "num_generated": 2048, "ode_steps": 32, "seed": seed}


def test_promotion_selects_best_learning_rate_per_width_only_at_matching_protocol():
    candidates = {name: {"screen_complete": True} for name in campaign_12h.CANDIDATE_NAMES}
    candidates["adm96_lr3e4"]["failed"] = "nonfinite gradient"
    status = {"candidates": candidates, "evaluations": [
        _screen("adm64-screen", 50), _screen("adm64_lr2e4-screen", 35),
        _screen("adm64_lr3e4-screen", 40), _screen("adm96-screen", 38),
        _screen("adm96_lr2e4-screen", 36), _screen("adm96_lr3e4-screen", 1),
        _screen("adm64-screen", 0.1, seed=31415),
    ]}
    assert campaign_12h.best_screen_per_width(status) == ["adm64_lr2e4", "adm96_lr2e4"]
    assert not campaign_12h.is_screening_result(_screen("other", 0.1, seed=31415))


def test_promotion_ranking_uses_shared_seeds_and_can_change_winner():
    status = {
        "promoted": ["adm64", "adm96"],
        "candidates": {name: {"promotion_complete": True} for name in ("adm64", "adm96")},
        "evaluations": [_screen("adm64-promotion", 20), _screen("adm96-promotion", 30),
                        _screen("adm64-promotion-seed31415", 60, 31415),
                        _screen("adm96-promotion-seed31415", 25, 31415)],
    }
    ranking, seeds = campaign_12h.rank_promotions(status)
    assert seeds == [42, 31415]
    assert ranking == [(27.5, "adm96"), (40.0, "adm64")]
    status["evaluations"].pop()
    ranking, seeds = campaign_12h.rank_promotions(status)
    assert seeds == [42]
    assert ranking == [(20.0, "adm64"), (30.0, "adm96")]


def test_second_seed_is_enabled_by_default(monkeypatch):
    monkeypatch.setattr(campaign_12h.sys, "argv", ["campaign_12h.py"])
    assert campaign_12h.parse_args().promotion_two_seeds
    monkeypatch.setattr(campaign_12h.sys, "argv", ["campaign_12h.py", "--no-promotion-two-seeds", "--seed", "31415"])
    args = campaign_12h.parse_args()
    assert not args.promotion_two_seeds
    assert args.seed == 31415


import pytest


@pytest.mark.parametrize("already_screened", [False, True])
def test_six_pilot_controller_reuses_benchmarks_promotes_and_resumes_durably(tmp_path, monkeypatch, already_screened):
    import json
    import math
    from pathlib import Path
    from types import SimpleNamespace
    import yaml

    monkeypatch.setattr(campaign_12h, "ROOT", tmp_path)
    monkeypatch.setattr(campaign_12h, "LEGACY_REPORT", tmp_path / "missing-baseline.json")
    monkeypatch.setattr(campaign_12h, "report", lambda status: tmp_path / "report.md")
    monkeypatch.setattr(campaign_12h, "export_samples", lambda result, destination: None)
    clock = [1000.0]
    monkeypatch.setattr(campaign_12h.time, "time", lambda: clock[0])
    events = []
    campaign_dir = tmp_path / "campaign"
    (tmp_path / "configs").mkdir()
    for name in campaign_12h.CANDIDATE_NAMES:
        lr = 3e-4 if "lr3e4" in name else 2e-4 if "lr2e4" in name else 1e-4
        config = {"training": {"num_steps": 100000, "learning_rate": lr, "lr_warmup_steps": 500}}
        (tmp_path / "configs" / f"campaign_{name}.yaml").write_text(yaml.safe_dump(config))

    def fake_snapshot(checkpoint, destination):
        assert checkpoint.exists()
        step = campaign_12h.read_json(checkpoint.parent / "run.json")["completed_steps"]
        saved = destination.with_name(f"{destination.stem}.{step}.pt")
        saved.write_text(str(step))
        return saved

    monkeypatch.setattr(campaign_12h, "snapshot", fake_snapshot)
    screen_fids = {"adm64": 70, "adm96": 60, "adm64_lr2e4": 40,
                   "adm96_lr2e4": 45, "adm64_lr3e4": 90, "adm96_lr3e4": 50}

    def fake_execute(task, *, until, log, deadline, **kwargs):
        assert until < deadline
        assert clock[0] < until
        name = log.stem
        events.append((task, name, until, kwargs.copy(), clock[0]))
        if task == "benchmark":
            width = "adm64" if "adm64" in name else "adm96"
            campaign_12h.write_json(kwargs["result"], {
                "seconds_per_step": 0.35 if width == "adm64" else 0.65,
                "parameters": 23000000 if width == "adm64" else 52000000,
            })
            clock[0] += 10
            return 0
        if task == "train":
            run_dir = kwargs["run_dir"]
            info = campaign_12h.read_json(run_dir / "run.json", {})
            before = info.get("completed_steps", 0)
            cfg = yaml.safe_load(kwargs["config"].read_text())
            speed = 0.35 if run_dir.name.startswith("adm64") else 0.65
            delta = min(cfg["training"]["num_steps"] - before, max(1, math.floor((until - clock[0]) / speed)))
            clock[0] += delta * speed
            info.update(completed_steps=before + delta, stop_reason="deadline", finished_at=campaign_12h.utc())
            # A failed fresh pilot must not be eligible for promotion or FID.
            if run_dir.name == "adm64_lr3e4":
                info["stop_reason"] = "loss_limit"
            campaign_12h.write_json(run_dir / "run.json", info)
            (run_dir / "latest.pt").write_text("checkpoint")
            (run_dir / "best_val.pt").write_text("checkpoint")
            return 1 if info["stop_reason"] == "loss_limit" else 0
        assert task == "evaluate"
        if name.endswith("-screen"):
            fid = screen_fids[name.removesuffix("-screen")]
        elif kwargs["seed"] == 31415:
            fid = 0.01  # Must never enter the final seed-42 FID pool.
        elif "promotion" in name:
            fid = 30 if name.startswith("adm64") else 35
        elif "best-val" in name:
            fid = 18
        else:
            fid = 20
        result = {"fid": fid, "num_generated": kwargs["samples"], "ode_steps": kwargs["ode_steps"],
                  "seed": kwargs["seed"], "checkpoint_step": int(kwargs["checkpoint"].read_text()),
                  "generated_path": str(tmp_path / "missing-generated.pt")}
        campaign_12h.write_json(kwargs["result"], result)
        clock[0] += 30
        return 0

    monkeypatch.setattr(campaign_12h, "execute", fake_execute)
    args = SimpleNamespace(run_dir=campaign_dir, until=44200.0, publish_results=False, promotion_two_seeds=True)
    if already_screened:
        run_dir = campaign_dir / "adm64"
        run_dir.mkdir(parents=True)
        cfg = yaml.safe_load((tmp_path / "configs" / "campaign_adm64.yaml").read_text())
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
        (run_dir / "latest.pt").write_text("checkpoint")
        snap = run_dir / "screen_ema.4000.pt"
        snap.write_text("4000")
        campaign_12h.write_json(run_dir / "run.json", {"completed_steps": 4000, "stop_reason": "deadline"})
        campaign_12h.write_json(run_dir / "benchmark.json", {"seconds_per_step": 0.35, "parameters": 23000000})
        campaign_12h.write_json(campaign_dir / "status.json", {
            "deadline_unix": args.until, "phase": "paused", "errors": [],
            "candidates": {"adm64": {"run_dir": str(run_dir), "config": str(run_dir / "config.yaml"),
                                      "screen_complete": True, "screen_snapshot": str(snap), "screen_seconds_used": 1500,
                                      "promotion_seconds_used": 338.28336}},
            "evaluations": [{**_screen("adm64-screen", 5), "config": str(run_dir / "config.yaml"),
                             "snapshot": str(snap), "checkpoint_step": 4000}],
        })
    campaign_12h.controller(args)
    status = campaign_12h.read_json(campaign_dir / "status.json")
    assert list(status["candidates"]) == list(campaign_12h.CANDIDATE_NAMES)
    benchmarks = [event for event in events if event[0] == "benchmark"]
    assert len(benchmarks) == (1 if already_screened else 2)
    assert all("lr" not in event[1] for event in benchmarks)
    screens = [event[1] for event in events if event[0] == "train" and event[1].endswith("-screen-train")]
    expected = campaign_12h.CANDIDATE_NAMES[1:] if already_screened else campaign_12h.CANDIDATE_NAMES
    assert screens == [name + "-screen-train" for name in expected]
    expected_small = "adm64" if already_screened else "adm64_lr2e4"
    assert status["promoted"] == [expected_small, "adm96_lr2e4"]
    assert status["winner"] == expected_small
    if already_screened:
        promotion = next(event for event in events if event[1] == "adm64-promotion-train")
        assert promotion[2] - promotion[4] == pytest.approx(2700 - 338.28336)
    assert status["promotion_ranking_seeds"] == [42, 31415]
    assert status["candidates"]["adm64_lr3e4"]["failed"]
    assert not any(event[1] == "adm64_lr3e4-screen" for event in events)
    for width in ("adm64", "adm96"):
        horizons = [yaml.safe_load(Path(candidate["config"]).read_text())["training"]["num_steps"]
                    for name, candidate in status["candidates"].items() if campaign_12h.candidate_width(name) == width]
        assert len(set(horizons)) == 1
    for name in status["promoted"]:
        assert status["candidates"][name]["promotion_complete"]
        assert status["candidates"][name]["promotion_seconds_used"] >= 2699
    final_config = yaml.safe_load(Path(status["winner_config"]).read_text())
    assert final_config["training"]["scheduler_restart_id"] == "winner-final-cosine-v1"
    assert final_config["training"]["lr_warmup_steps"] == 0
    assert final_config["training"]["num_steps"] > status["winner_schedule"]["start_step"]
    final_events = [event for event in events if event[0] == "evaluate" and event[3]["samples"] == 10000]
    assert len(final_events) == 2
    assert all(event[3]["seed"] == 42 and "seed31415" not in event[1] for event in final_events)
    assert status["phase"] == "complete"
    # Reconnecting to an already-completed controller must not train/evaluate again.
    count = len(events)
    campaign_12h.controller(args)
    assert len(events) == count

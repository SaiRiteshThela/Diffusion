"""One-GPU, deadline-bounded architecture search with the existing FM objective.

Run the controller inside tmux; each GPU stage is a separate process. The
controller enforces stage/global deadlines independently of the training loop.
Large artifacts live under outputs/ (persistent storage), never in Git.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
CAMPAIGN = "campaign_20260917"
DEFAULT_DIR = ROOT / "outputs" / CAMPAIGN
DEFAULT_DEADLINE = datetime(2026, 9, 17, 15, 15, 44, tzinfo=timezone.utc).timestamp()
REAL_STATS = ROOT / "samples/experiment_5/fid/celeba_train_64_stats.npz"
LEGACY_REPORT = ROOT / "samples/experiment_5/fid/fid.json"
CANDIDATE_NAMES = ("adm64", "adm96", "adm64_lr2e4", "adm96_lr2e4", "adm64_lr3e4", "adm96_lr3e4")
SCREEN_SECONDS = 25 * 60
PROMOTION_SECONDS = 45 * 60
FINAL_RESERVE_SECONDS = 60 * 60
PROMOTION_SEEDS = (42, 31415)
UNHEALTHY_STOPS = {"loss_limit", "nonfinite_loss", "nonfinite_gradient"}


def candidate_width(name: str) -> str:
    width = name.split("_", 1)[0]
    if width not in {"adm64", "adm96"}:
        raise ValueError(f"unknown campaign candidate: {name}")
    return width


def is_screening_result(result: dict, seed: int = 42) -> bool:
    return (
        result.get("num_generated") == 2048
        and result.get("ode_steps") == 32
        and result.get("seed", 42) == seed
        and isinstance(result.get("fid"), (int, float))
        and math.isfinite(result["fid"])
    )


def best_screen_per_width(status: dict) -> list[str]:
    """Promote one learning rate per width using only the shared seed-42 protocol."""
    promoted = []
    for width in ("adm64", "adm96"):
        eligible = []
        for name, candidate in status["candidates"].items():
            if candidate_width(name) != width or not candidate.get("screen_complete") or candidate.get("failed"):
                continue
            result = next((value for value in status["evaluations"]
                           if value["name"] == name + "-screen" and is_screening_result(value)), None)
            if result is not None:
                eligible.append((result["fid"], name))
        if eligible:
            promoted.append(min(eligible)[1])
    return promoted


def rank_promotions(status: dict) -> tuple[list[tuple[float, str]], list[int]]:
    """Compare promotion checkpoints over exactly the same available seeds."""
    by_candidate = {}
    for name in status.get("promoted", []):
        candidate = status["candidates"][name]
        if not candidate.get("promotion_complete") or candidate.get("failed"):
            continue
        scores = {}
        for seed in PROMOTION_SEEDS:
            suffix = "-promotion" if seed == 42 else f"-promotion-seed{seed}"
            result = next((value for value in status["evaluations"]
                           if value["name"] == name + suffix and is_screening_result(value, seed)), None)
            if result is not None:
                scores[seed] = result["fid"]
        if scores:
            by_candidate[name] = scores
    if not by_candidate:
        return [], []
    common = sorted(set.intersection(*(set(scores) for scores in by_candidate.values())))
    if not common:
        return [], []
    return sorted((sum(scores[seed] for seed in common) / len(common), name)
                  for name, scores in by_candidate.items()), common



def utc(timestamp: float | None = None) -> str:
    return datetime.fromtimestamp(timestamp or time.time(), timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # GeeseFS can serve stale bytes after replacement; small metadata files
    # are written through their existing inode, with readers retrying a race.
    with path.open("w") as stream:
        stream.write(json.dumps(payload, indent=2, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    for attempt in range(4):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt == 3:
                raise
            time.sleep(0.05)


def configure_environment():
    # Configure only an explicitly launched campaign, never unrelated imports.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("DIFFUSION_CHECKPOINT_TMPDIR", "/dev/shm/diffusion-ckpts")
    os.environ.setdefault("DIFFUSION_IMMUTABLE_CHECKPOINTS", "1")
    os.environ.setdefault("DIFFUSION_CHECKPOINT_KEEP", "3")
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")


def configure_torch():
    configure_environment()
    import torch
    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError("Campaign requires its reserved CUDA GPU")
    if "A40" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"Expected A40, found {torch.cuda.get_device_name(0)}")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    return torch, torch.device("cuda")


def benchmark(args):
    torch, device = configure_torch()
    from copy import deepcopy
    from models.config import load_config
    from models.flow import FlowModel
    cfg = load_config(args.config)
    torch.manual_seed(42)
    model = FlowModel.from_config(args.config).to(device=device, memory_format=torch.channels_last)
    model.train()
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), fused=True)
    batch_size = cfg["data"]["batch_size"]
    # Reserve approximately the same space as the on-GPU uint8 data cache.
    cache_reservation = torch.empty(2_300_000_000, dtype=torch.uint8, device=device)
    torch.cuda.reset_peak_memory_stats()
    elapsed = []
    for step in range(25):
        if time.time() >= args.until:
            raise TimeoutError("Benchmark deadline reached")
        start = time.perf_counter()
        z = torch.randn(batch_size, 3, 64, 64, device=device).clamp(-1, 1)
        noise = torch.randn_like(z)
        t = torch.rand(batch_size, 1, 1, 1, device=device)
        xt = t * z + (1 - t) * noise
        target = z - noise
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted = model(xt, t.flatten())
        loss = (predicted.float() - target).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            for shadow, parameter in zip(ema.parameters(), model.parameters()):
                shadow.lerp_(parameter, 0.0005)
        torch.cuda.synchronize()
        if step >= 5:
            elapsed.append(time.perf_counter() - start)
    seconds = sum(elapsed) / len(elapsed)
    result = {
        "config": str(args.config), "parameters": sum(p.numel() for p in model.parameters()),
        "batch_size": batch_size, "seconds_per_step": seconds,
        "images_per_second": batch_size / seconds,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "final_loss": loss.item(), "completed_at": utc(),
    }
    write_json(args.result, result)
    print(json.dumps(result, indent=2), flush=True)


def train_worker(args):
    torch, device = configure_torch()
    import wandb
    from models.config import load_config
    from experiments.run_experiment_6 import build_trainer, resolve_data_root
    from training.trainer import resolve_checkpoint_path
    cfg = load_config(args.config)
    torch.manual_seed(cfg["data"]["seed"])
    trainer = build_trainer(cfg, args.config, resolve_data_root(), device)
    trainer.model.to(memory_format=torch.channels_last)
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = run_dir / "latest.pt"
    best = run_dir / "best_val.pt"
    saved_info = read_json(run_dir / "run.json", {})
    run_id = saved_info.get("wandb_id") or uuid.uuid4().hex[:8]
    key = Path("/workspace-global/.wandb/api_key")
    if key.exists():
        os.environ.setdefault("WANDB_API_KEY", key.read_text().strip())
    Path("/workspace/wandb").mkdir(exist_ok=True)
    run = None
    try:
        run = wandb.init(
            project="celeba-flow-matching", group=CAMPAIGN, id=run_id,
            name=f"{CAMPAIGN}-{run_dir.name}", resume="allow",
            dir="/workspace/wandb", config=cfg, allow_val_change=True,
            tags=["a40-12h", "unconditional", "same-fm-loss", "adm", "bf16"],
            settings=wandb.Settings(init_timeout=60),
        )
    except Exception as exc:
        print(f"W&B unavailable, continuing with local logs: {type(exc).__name__}", flush=True)
    info = {**saved_info, "wandb_id": run_id, "wandb_url": run.url if run else None,
            "config": str(args.config), "checkpoint": str(latest), "phase_until": utc(args.until),
            "parameters": sum(p.numel() for p in trainer.model.parameters()), "started_at": utc()}
    write_json(run_dir / "run.json", info)
    tr = cfg["training"]
    sp = cfg["sampling"]
    phase_started = time.time()
    phase_start_steps = int(saved_info.get("completed_steps", 0))
    try:
        history = trainer.train(
            num_steps=tr["num_steps"], device=device, lr=tr["learning_rate"],
            optimizer_name=tr["optimizer"], optimizer_betas=tuple(tr["optimizer_betas"]),
            optimizer_fused=True, weight_decay=tr["weight_decay"], max_grad_norm=tr["max_grad_norm"],
            lr_schedule=tr["lr_schedule"], lr_warmup_steps=tr["lr_warmup_steps"],
            lr_min_ratio=tr["lr_min_ratio"], ema_decay=tr["ema_decay"], ema_warmup=True,
            mixed_precision="bf16", batch_size=cfg["data"]["batch_size"],
            ckpt_path=latest, best_ckpt_path=best, checkpoint_every=tr["checkpoint_every"],
            val_every=tr["val_every"], val_batches=tr["val_batches"], validation_seed=20260917,
            plot_every=tr["plot_every"], plot_seed=42, n_sampling_steps=sp["preview_ode_steps"],
            n_plot_images=sp["n_plot_images"], n_plot_steps=sp["n_plot_steps"],
            samples_dir=run_dir / "previews", show_plots=False, wandb_run=run,
            resume_from=resolve_checkpoint_path(latest) if resolve_checkpoint_path(latest).exists() else None,
            deadline_timestamp=args.until, handle_signals=True, abort_train_loss=10.0,
            scheduler_restart_id=tr.get("scheduler_restart_id"),
        )
        info.update({"finished_at": utc(), "completed_steps": trainer.completed_steps, "stop_reason": trainer.last_stop_reason,
                     "best_fixed_val": float(history["val"].min()) if len(history.get("val", [])) else None,
                     "phase_started_unix": phase_started, "phase_elapsed_seconds": time.time() - phase_started,
                     "phase_start_steps": phase_start_steps, "phase_completed_steps": trainer.completed_steps - phase_start_steps})
        write_json(run_dir / "run.json", info)
        if trainer.last_stop_reason in UNHEALTHY_STOPS:
            raise RuntimeError(f"Unhealthy training stop: {trainer.last_stop_reason}")
    finally:
        if run is not None:
            run.finish()


def eval_worker(args):
    configure_torch()
    from sampling.fid_celeba import evaluate_checkpoint
    result = evaluate_checkpoint(
        args.config, args.checkpoint, args.run_dir,
        num_samples=args.samples, ode_steps=args.ode_steps, batch_size=128,
        real_stats_path=REAL_STATS, legacy_real_stats_report=LEGACY_REPORT,
        deadline_unix=args.until, seed=args.seed,
    )
    write_json(args.result, result)


def snapshot(checkpoint: Path, destination: Path):
    import torch
    from training.trainer import resolve_checkpoint_path
    checkpoint = resolve_checkpoint_path(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    weights = state.get("ema_model") or state["model"]
    payload = {"model": weights, "ema_model": weights, "step": state["step"],
               "source_checkpoint": str(checkpoint), "inference_only": True}
    destination = destination.with_name(f"{destination.stem}.step{state['step']:09d}.{uuid.uuid4().hex[:8]}.pt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path("/dev/shm/diffusion-ckpts") / f"{CAMPAIGN}-{destination.name}"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, temporary)
    import shutil
    shutil.copyfile(temporary, destination)
    temporary.unlink()
    return destination


def export_samples(result: dict, destination: Path):
    """Export the first 25 evaluated samples without using the GPU."""
    import torch
    from torchvision.utils import save_image
    generated = Path(result.get("generated_path", ""))
    if not generated.is_file():
        return None
    images = torch.load(generated, map_location="cpu", mmap=True, weights_only=False)[:25]
    destination.mkdir(parents=True, exist_ok=True)
    pixels = images.float().div(255)
    save_image(pixels, destination / "grid_5x5_native64.png", nrow=5)
    for index, sample in enumerate(pixels):
        save_image(sample, destination / f"sample_{index:02d}.png")
    return str(destination / "grid_5x5_native64.png")


def execute(task: str, *, until: float, log: Path, deadline: float, **kwargs):
    """Independent supervisor kills an unresponsive GPU stage before budget end."""
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--task", task, "--until", str(until)]
    for key, value in kwargs.items():
        if value is not None:
            command.extend(["--" + key.replace("_", "-"), str(value)])
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", buffering=1) as stream:
        stream.write(f"\n{utc()} START {task}\n")
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            hard_stop = min(until + 75, deadline - 15)
            while process.poll() is None:
                if time.time() >= hard_stop:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=max(1, min(10, deadline - time.time() - 3)))
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(min(5, max(0.1, hard_stop - time.time())))
            return process.wait()
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            raise


def report(status: dict):
    """Small, reviewable result artifact; checkpoints and credentials stay out of Git."""
    lines = [f"# {CAMPAIGN}: 12-hour A40 experiment", "",
             f"Updated: {utc()}", f"Status: {status['phase']}",
             f"Hard deadline: {utc(status['deadline_unix'])}", "",
             "Loss unchanged: uniform t; x_t = t*x + (1-t)*noise; velocity MSE against x-noise.",
             "Data unchanged: unconditional CelebA, crop178, resize64, horizontal flips, [-1,1].", "",
             "## Candidates", "", "| Candidate | Parameters | Step seconds |", "|---|---:|---:|"]
    for name, candidate in status.get("candidates", {}).items():
        bench = candidate.get("benchmark", {})
        lines.append(f"| {name} | {bench.get('parameters', '')} | {bench.get('seconds_per_step', '')} |")
    lines.extend(["", "Six 25-minute pilots: widths 64/96 × learning rates 1e-4/2e-4/3e-4. The best learning rate per width receives 45 more minutes; the winner uses the remaining training budget.",
                  "", "## Evaluations", "", "| Candidate/checkpoint | Samples | Euler updates | Seed | FID |", "|---|---:|---:|---:|---:|"])
    for result in status.get("evaluations", []):
        lines.append(f"| {result['name']} / step {result.get('checkpoint_step', '?')} | {result.get('num_generated', '?')} | {result.get('ode_steps', '?')} | {result.get('seed', 42)} | {result.get('fid', '?')} |")
    lines.extend(["", "Screening FID uses 2,048 samples and 32 Euler updates; compare only within that protocol.",
                  "Final evaluations use 10,000 samples and 100 Euler updates.",
                  "Real Inception statistics are reused from the prior, unchanged CelebA preprocessing with recorded provenance."])
    if status.get("winner"):
        lines.extend(["", f"Selected training candidate: {status['winner']}"])
    if status.get("selected_model"):
        lines.extend(["", "## Selected model", "", "```json", json.dumps(status["selected_model"], indent=2), "```"])
    if status.get("errors"):
        lines.extend(["", "## Stage errors", ""] + [f"- {message}" for message in status["errors"]])
    path = ROOT / "reports" / f"{CAMPAIGN}.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def controller(args):
    import yaml
    from training.trainer import resolve_checkpoint_path
    campaign_dir = args.run_dir
    campaign_dir.mkdir(parents=True, exist_ok=True)
    state_path = campaign_dir / "status.json"
    status = read_json(state_path, {
        "campaign": CAMPAIGN, "started_at": utc(), "deadline_unix": args.until,
        "phase": "preparing", "candidates": {}, "evaluations": [], "errors": [],
    })
    deadline = min(float(status["deadline_unix"]), args.until)
    status["deadline_unix"] = deadline
    train_end = deadline - FINAL_RESERVE_SECONDS
    status["search_version"] = 2
    status["candidate_order"] = list(CANDIDATE_NAMES)
    status.setdefault("promotion_two_seeds", getattr(args, "promotion_two_seeds", True))
    status.setdefault("pilot_horizons", {})

    def update(phase):
        status.update(phase=phase, updated_at=utc())
        write_json(state_path, status)
        report(status)
        print(f"{utc()} {phase}", flush=True)

    def error(message):
        if message not in status["errors"]:
            status["errors"].append(message)

    def stage(task, name, until, **kwargs):
        if time.time() >= min(until, deadline - 15):
            error(f"{name}: skipped because its deadline has passed")
            return 124
        update(name)
        code = execute(task, until=until, deadline=deadline,
                       log=campaign_dir / "logs" / f"{name}.log", **kwargs)
        if code != 0:
            error(f"{name}: exit {code}; see stage log")
        return code

    def evaluate(name, config, checkpoint, samples=2048, steps=32, seconds=900,
                 seed=42, phase_deadline=None):
        existing = next((value for value in status["evaluations"] if value["name"] == name), None)
        if existing is not None:
            if (existing.get("num_generated"), existing.get("ode_steps"), existing.get("seed", 42)) != (samples, steps, seed):
                raise ValueError(f"stored evaluation protocol does not match {name}")
            return existing
        out = campaign_dir / "evaluation" / name
        result_path = out / "result.json"
        until = min(time.time() + seconds, deadline - 90,
                    phase_deadline if phase_deadline is not None else deadline - 90)
        if until - time.time() < 60:
            return None
        code = stage("evaluate", name, until, config=config, checkpoint=checkpoint,
                     run_dir=out, result=result_path, samples=samples, ode_steps=steps, seed=seed)
        if code != 0 or not result_path.exists():
            error(f"{name}: no completed FID result")
            return None
        result = read_json(result_path)
        if ((result.get("num_generated"), result.get("ode_steps"), result.get("seed", 42)) != (samples, steps, seed)
                or not isinstance(result.get("fid"), (int, float)) or not math.isfinite(result["fid"])):
            error(f"{name}: invalid FID result or mismatched evaluation protocol")
            return None
        result.update(name=name, config=str(config), snapshot=str(checkpoint), seed=seed)
        status["evaluations"].append(result)
        update(name + "-complete")
        return result

    def checkpoint_exists(run_dir):
        return resolve_checkpoint_path(run_dir / "latest.pt").exists()

    def train_phase(name, phase, target_seconds):
        candidate = status["candidates"][name]
        if candidate.get("failed"):
            return False
        run_dir, config = Path(candidate["run_dir"]), Path(candidate["config"])
        complete_key, snapshot_key = f"{phase}_complete", f"{phase}_snapshot"
        if candidate.get(complete_key) and candidate.get(snapshot_key) and Path(candidate[snapshot_key]).exists():
            return True
        used_key, active_key = f"{phase}_seconds_used", f"{phase}_active"
        used = max(float(candidate.get(used_key, 0)), float(candidate.get(f"{phase}_seconds_credited", 0)))
        # Recover credit if the parent died after the worker saved and before
        # its phase bookkeeping completed. A stale run.json cannot add credit.
        active = candidate.pop(active_key, None)
        if active:
            info = read_json(run_dir / "run.json", {})
            if info.get("finished_at"):
                finished = datetime.fromisoformat(info["finished_at"]).timestamp()
                if finished >= active["started_at"]:
                    used += max(0, min(finished, active["until"]) - active["started_at"])
        candidate[used_key] = used
        remaining = max(0, target_seconds - used)
        if not candidate.get(complete_key) and remaining >= 30:
            if time.time() >= train_end - 300:
                return False
            before = read_json(run_dir / "run.json", {}).get("completed_steps", 0)
            started = time.time()
            until = min(started + remaining, train_end - 120)
            candidate[active_key] = {"started_at": started, "until": until, "before_steps": before}
            code = None
            try:
                code = stage("train", name + f"-{phase}-train", until, config=config, run_dir=run_dir)
            finally:
                candidate[used_key] = used + max(0, min(time.time(), until) - started)
                candidate.pop(active_key, None)
                write_json(state_path, status)
            info = read_json(run_dir / "run.json", {})
            if code != 0 or info.get("stop_reason") in UNHEALTHY_STOPS:
                candidate["failed"] = f"{phase} training failed: exit {code}, stop {info.get('stop_reason')}"
                error(f"{name}: {candidate['failed']}")
                update(name + f"-{phase}-failed")
                return False
            delta = int(info.get("completed_steps", 0)) - int(before)
            if delta <= 0:
                candidate["failed"] = f"{phase} made no optimizer progress"
                error(f"{name}: {candidate['failed']}")
                update(name + f"-{phase}-failed")
                return False
            elapsed = time.time() - started
            if elapsed >= 300:
                candidate["measured_seconds_per_step"] = elapsed / delta
        if candidate.get(used_key, 0) < target_seconds - 30 and not candidate.get(complete_key):
            error(f"{name}-{phase}: training budget ended before the full comparison phase")
            return False
        if not checkpoint_exists(run_dir):
            error(f"{name}-{phase}: no resumable checkpoint")
            return False
        info = read_json(run_dir / "run.json", {})
        if info.get("stop_reason") in UNHEALTHY_STOPS or info.get("completed_steps", 0) <= 0:
            error(f"{name}-{phase}: checkpoint is not from healthy completed training")
            return False
        candidate[snapshot_key] = str(snapshot(run_dir / "latest.pt", run_dir / f"{phase}_ema.pt"))
        candidate[complete_key] = True
        candidate[f"{phase}_completed_steps"] = int(info["completed_steps"])
        update(name + f"-{phase}-saved")
        return True

    # Benchmark each architecture once; LR variants share its throughput and
    # pilot cosine horizon, so a later launch does not change the LR schedule.
    for name in CANDIDATE_NAMES:
        width = candidate_width(name)
        run_dir = campaign_dir / name
        run_dir.mkdir(exist_ok=True)
        template = ROOT / "configs" / f"campaign_{name}.yaml"
        base_dir = campaign_dir / width
        benchmark_result = base_dir / "benchmark.json"
        if not benchmark_result.exists():
            if name != width:
                error(f"{name}: architecture benchmark unavailable")
                continue
            code = stage("benchmark", width + "-benchmark", min(time.time() + 600, train_end - 120),
                         config=template, result=benchmark_result)
            if code != 0:
                continue
        bench = read_json(benchmark_result)
        seconds_per_step = float(bench["seconds_per_step"])
        if not math.isfinite(seconds_per_step) or seconds_per_step <= 0:
            raise ValueError(f"invalid benchmark throughput: {width}")
        config_path = run_dir / "config.yaml"
        if width not in status["pilot_horizons"]:
            base_config = base_dir / "config.yaml"
            status["pilot_horizons"][width] = (
                int(yaml.safe_load(base_config.read_text())["training"]["num_steps"])
                if base_config.exists() else max(2000, int(8 * 3600 / seconds_per_step))
            )
        if not config_path.exists():
            config = yaml.safe_load(template.read_text())
            config["training"]["num_steps"] = status["pilot_horizons"][width]
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        status["candidates"][name] = {
            **status["candidates"].get(name, {}), "run_dir": str(run_dir), "config": str(config_path),
            "width": width, "benchmark": bench, "benchmark_source": str(benchmark_result),
        }
        update(name + "-ready")
    if not status["candidates"]:
        raise RuntimeError("No candidate passed its architecture benchmark")

    for name in CANDIDATE_NAMES:
        if name not in status["candidates"]:
            continue
        candidate = status["candidates"][name]
        if train_phase(name, "screen", SCREEN_SECONDS):
            evaluate(name + "-screen", Path(candidate["config"]), Path(candidate["screen_snapshot"]),
                     phase_deadline=train_end - 120)

    if "promoted" not in status:
        status["promoted"] = best_screen_per_width(status)
        # Explicit fallback only when that width has healthy training but no FID.
        for width in ("adm64", "adm96"):
            if any(candidate_width(name) == width for name in status["promoted"]):
                continue
            healthy = [name for name in CANDIDATE_NAMES if name in status["candidates"]
                       and candidate_width(name) == width and status["candidates"][name].get("screen_complete")
                       and not status["candidates"][name].get("failed")]
            if healthy:
                status["promoted"].append(healthy[0])
                error(f"{width}: screening FID unavailable; promoted first healthy learning rate")
        update("promotion-candidates-selected")
    promotion_seeds = PROMOTION_SEEDS if status["promotion_two_seeds"] else (42,)
    for name in status["promoted"]:
        candidate = status["candidates"][name]
        if train_phase(name, "promotion", PROMOTION_SECONDS):
            for seed in promotion_seeds:
                suffix = "-promotion" if seed == 42 else f"-promotion-seed{seed}"
                evaluate(name + suffix, Path(candidate["config"]), Path(candidate["promotion_snapshot"]),
                         seed=seed, phase_deadline=train_end - 120)

    if not status.get("winner"):
        ranking, seeds = rank_promotions(status)
        status["promotion_ranking_seeds"] = seeds
        status["promotion_ranking"] = [{"name": name, "mean_fid": score} for score, name in ranking]
        if ranking:
            status["winner"] = ranking[0][1]
        else:
            healthy_screens = [value for value in status["evaluations"] if is_screening_result(value)
                               and value["name"].endswith("-screen")
                               and not status["candidates"].get(value["name"].removesuffix("-screen"), {}).get("failed")]
            if healthy_screens:
                status["winner"] = min(healthy_screens, key=lambda value: value["fid"])["name"].removesuffix("-screen")
                error("Matched promotion FID unavailable; winner selected from seed-42 screening FID")
            else:
                healthy = [name for name, candidate in status["candidates"].items()
                           if candidate.get("screen_complete") and not candidate.get("failed")]
                if not healthy:
                    raise RuntimeError("No healthy candidate completed training")
                status["winner"] = healthy[0]
                error("All selection FIDs unavailable; defaulted to first healthy candidate")
        update("selected-" + status["winner"])
    winner = status["winner"]
    candidate = status["candidates"][winner]
    run_dir = Path(candidate["run_dir"])

    # Reset only the LR schedule once for the winner. The trainer preserves all
    # learned weights, optimizer moments, EMA and RNG when restart_id changes.
    if not status.get("winner_config"):
        config = yaml.safe_load(Path(candidate["config"]).read_text())
        current_steps = int(read_json(run_dir / "run.json", {}).get("completed_steps", 0))
        speed = float(candidate.get("measured_seconds_per_step", candidate["benchmark"]["seconds_per_step"]))
        remaining = max(0, train_end - time.time() - 120)
        config["training"]["num_steps"] = current_steps + max(1, int(remaining / speed * 0.97))
        config["training"]["lr_warmup_steps"] = 0
        config["training"]["scheduler_restart_id"] = "winner-final-cosine-v1"
        final_config = run_dir / "winner_final_config.yaml"
        final_config.write_text(yaml.safe_dump(config, sort_keys=False))
        status["winner_config"] = str(final_config)
        status["winner_schedule"] = {"start_step": current_steps, "estimated_seconds_per_step": speed,
                                      "num_steps": config["training"]["num_steps"], "created_at": utc()}
        update("winner-final-schedule-ready")
    config = Path(status["winner_config"])
    status.setdefault("round_snapshots", {})
    # Finish an evaluation saved just before a previous controller interruption.
    for round_name, saved_snapshot in status["round_snapshots"].items():
        evaluate(round_name, config, Path(saved_snapshot), phase_deadline=train_end - 120)
    round_index = int(status.get("completed_rounds", 0))
    while time.time() < train_end - 300:
        before = int(read_json(run_dir / "run.json", {}).get("completed_steps", 0))
        planned = int(yaml.safe_load(config.read_text())["training"]["num_steps"])
        if before >= planned:
            break
        round_index += 1
        round_name = f"{winner}-round{round_index}"
        code = stage("train", round_name + "-train", min(time.time() + 7200, train_end - 120),
                     config=config, run_dir=run_dir)
        info = read_json(run_dir / "run.json", {})
        if code != 0 or info.get("stop_reason") in UNHEALTHY_STOPS:
            break
        if int(info.get("completed_steps", 0)) <= before:
            error(f"{round_name}: no optimizer progress; stopped retries")
            break
        snap = snapshot(run_dir / "latest.pt", run_dir / f"round{round_index}_ema.pt")
        status["round_snapshots"][round_name] = str(snap)
        status["completed_rounds"] = round_index
        update(round_name + "-saved")
        evaluate(round_name, config, snap, phase_deadline=train_end - 120)
        update(round_name + "-complete")

    if resolve_checkpoint_path(run_dir / "best_val.pt").exists() and time.time() < deadline - 1800:
        existing_best = status.get("best_val_snapshot")
        best = Path(existing_best) if existing_best else snapshot(run_dir / "best_val.pt", run_dir / "best_val_ema.pt")
        status["best_val_snapshot"] = str(best)
        update("best-validation-snapshot-ready")
        evaluate(winner + "-best-val", config, best, phase_deadline=deadline - 1800)

    # Seed-31415 promotion checks must never enter the seed-42 finalist pool.
    screening = [value for value in status["evaluations"] if is_screening_result(value, seed=42)]
    finalists = sorted(screening, key=lambda value: value["fid"])[:2]
    for index, result in enumerate(finalists):
        if time.time() > deadline - 300:
            break
        available = deadline - 90 - time.time()
        seconds = min(2100, available) if index == 0 else available
        evaluate(result["name"] + "-final10k", Path(result["config"]), Path(result["snapshot"]),
                 samples=10000, steps=100, seconds=seconds, seed=42)
    final_results = [value for value in status["evaluations"]
                     if value.get("num_generated") == 10000 and value.get("ode_steps") == 100
                     and value.get("seed", 42) == 42 and math.isfinite(value.get("fid", float("nan")))]
    if final_results:
        selected = min(final_results, key=lambda value: value["fid"])
        status["selected_model"] = {key: selected[key] for key in ("name", "fid", "snapshot", "config", "checkpoint_step", "num_generated", "ode_steps")}
        status["selected_model"]["selection_protocol"] = "matched seed-42 10k Euler100 FID among evaluated campaign finalists"
    elif screening:
        selected = min(screening, key=lambda value: value["fid"])
        status["selected_model"] = {key: selected[key] for key in ("name", "fid", "snapshot", "config", "checkpoint_step", "num_generated", "ode_steps")}
        status["selected_model"]["selection_protocol"] = "provisional: seed-42 screening FID only; final10k unavailable"
    if status.get("selected_model"):
        status["selected_model"]["sample_grid"] = export_samples(selected, campaign_dir / "selected_samples")
    baseline = read_json(LEGACY_REPORT, {})
    status["historical_exp5_baseline"] = {**baseline, "provenance": "previously recorded run; not regenerated during this campaign"}
    if final_results and baseline.get("fid") is not None:
        status["selected_model"]["improves_recorded_exp5_fid"] = status["selected_model"]["fid"] < baseline["fid"]
        status["selected_model"]["recorded_exp5_fid"] = baseline["fid"]
    update("complete")
    if status.get("selected_model"):
        write_json(campaign_dir / "selected_model.json", status["selected_model"])
    if args.publish_results:
        path = report(status)
        subprocess.run(["git", "add", "--", str(path.relative_to(ROOT))], check=True)
        staged = subprocess.run(["git", "diff", "--cached", "--quiet", "--", str(path.relative_to(ROOT))])
        if staged.returncode == 1:
            subprocess.run(["git", "commit", "--only", "-m", "Record 12-hour A40 campaign results", "--", str(path.relative_to(ROOT))], check=True)
            subprocess.run(["git", "push", "origin", "HEAD"], check=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["controller", "benchmark", "train", "evaluate"], default="controller")
    parser.add_argument("--until", type=float, default=DEFAULT_DEADLINE)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--ode-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--promotion-two-seeds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--publish-results", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    configure_environment()
    args = parse_args()
    if time.time() >= args.until:
        raise SystemExit("Deadline has already passed; refusing GPU work")
    try:
        {"controller": controller, "benchmark": benchmark, "train": train_worker, "evaluate": eval_worker}[args.task](args)
    except BaseException as exc:
        if args.task == "controller":
            status_path = args.run_dir / "status.json"
            failed = read_json(status_path, {"deadline_unix": args.until, "errors": []})
            failed.update(phase="failed", updated_at=utc())
            failed.setdefault("errors", []).append(f"Controller stopped: {type(exc).__name__}")
            write_json(status_path, failed)
            report(failed)
        traceback.print_exc()
        raise

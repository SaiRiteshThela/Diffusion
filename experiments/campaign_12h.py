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
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("DIFFUSION_CHECKPOINT_TMPDIR", "/dev/shm/diffusion-ckpts")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
CAMPAIGN = "campaign_20260917"
DEFAULT_DIR = ROOT / "outputs" / CAMPAIGN
DEFAULT_DEADLINE = datetime(2026, 9, 17, 15, 15, 44, tzinfo=timezone.utc).timestamp()
REAL_STATS = ROOT / "samples/experiment_5/fid/celeba_train_64_stats.npz"
LEGACY_REPORT = ROOT / "samples/experiment_5/fid/fid.json"


def utc(timestamp: float | None = None) -> str:
    return datetime.fromtimestamp(timestamp or time.time(), timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    temporary.replace(path)


def read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def configure_torch():
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
            dir="/workspace/wandb", config=cfg,
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
            resume_from=latest if latest.exists() else None,
            deadline_timestamp=args.until, handle_signals=True, abort_train_loss=10.0,
        )
        info.update({"finished_at": utc(), "completed_steps": trainer.completed_steps, "stop_reason": trainer.last_stop_reason,
                     "best_fixed_val": float(history["val"].min()) if len(history.get("val", [])) else None})
        write_json(run_dir / "run.json", info)
        if trainer.last_stop_reason in {"loss_limit", "nonfinite_loss", "nonfinite_gradient"}:
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
        deadline_unix=args.until, seed=42,
    )
    write_json(args.result, result)


def snapshot(checkpoint: Path, destination: Path):
    import torch
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    weights = state.get("ema_model") or state["model"]
    payload = {"model": weights, "ema_model": weights, "step": state["step"],
               "source_checkpoint": str(checkpoint), "inference_only": True}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path("/dev/shm/diffusion-ckpts") / f"{CAMPAIGN}-{destination.name}"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, temporary)
    import shutil
    partial = destination.with_suffix(".partial")
    shutil.copyfile(temporary, partial)
    partial.replace(destination)
    temporary.unlink()
    return int(state["step"])


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
             "## Candidates", "", "| Candidate | Parameters | Step seconds | W&B |", "|---|---:|---:|---|"]
    for name, candidate in status.get("candidates", {}).items():
        bench = candidate.get("benchmark", {})
        info = read_json(Path(candidate["run_dir"]) / "run.json", {})
        lines.append(f"| {name} | {bench.get('parameters', '')} | {bench.get('seconds_per_step', '')} | {info.get('wandb_url', '')} |")
    lines.extend(["", "## Evaluations", "", "| Candidate/checkpoint | Samples | Euler updates | FID |", "|---|---:|---:|---:|"])
    for result in status.get("evaluations", []):
        lines.append(f"| {result['name']} / step {result.get('checkpoint_step', '?')} | {result.get('num_generated', '?')} | {result.get('ode_steps', '?')} | {result.get('fid', '?')} |")
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
    campaign_dir = args.run_dir
    campaign_dir.mkdir(parents=True, exist_ok=True)
    state_path = campaign_dir / "status.json"
    status = read_json(state_path, {
        "campaign": CAMPAIGN, "started_at": utc(), "deadline_unix": args.until,
        "phase": "preparing", "candidates": {}, "evaluations": [], "errors": [],
    })
    deadline = min(float(status["deadline_unix"]), args.until)
    train_end = deadline - 3600  # One hour reserved for matched final evaluation.

    def update(phase):
        status.update(phase=phase, updated_at=utc())
        write_json(state_path, status)
        report(status)
        print(f"{utc()} {phase}", flush=True)

    def stage(task, name, until, **kwargs):
        update(name)
        code = execute(task, until=until, deadline=deadline,
                       log=campaign_dir / "logs" / f"{name}.log", **kwargs)
        if code != 0:
            status["errors"].append(f"{name}: exit {code}; see stage log")
        return code

    def evaluate(name, config, checkpoint, samples=2048, steps=32, seconds=900):
        existing = next((v for v in status["evaluations"] if v["name"] == name), None)
        if existing:
            return existing
        out = campaign_dir / "evaluation" / name
        result_path = out / "result.json"
        until = min(time.time() + seconds, deadline - 90)
        if until - time.time() < 60:
            return None
        code = stage("evaluate", name, until, config=config, checkpoint=checkpoint,
                     run_dir=out, result=result_path, samples=samples, ode_steps=steps)
        if code == 0 and result_path.exists():
            result = read_json(result_path)
            result.update(name=name, config=str(config), snapshot=str(checkpoint))
            status["evaluations"].append(result)
            update(name + "-complete")
            return result
        return None

    for candidate_name in ("adm64", "adm96"):
        run_dir = campaign_dir / candidate_name
        run_dir.mkdir(exist_ok=True)
        template = ROOT / "configs" / f"campaign_{candidate_name}.yaml"
        benchmark_result = run_dir / "benchmark.json"
        if not benchmark_result.exists():
            code = stage("benchmark", candidate_name + "-benchmark", min(time.time()+600, train_end),
                         config=template, result=benchmark_result)
            if code != 0:
                continue
        bench = read_json(benchmark_result)
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            config = yaml.safe_load(template.read_text())
            # Budget a full winner run plus this candidate's screen. Leave one
            # competing screen and evaluation overhead out of the LR horizon.
            allocated = max(3600, train_end - time.time() - 2700 - 900)
            config["training"]["num_steps"] = max(2000, int(allocated / bench["seconds_per_step"] * 1.03))
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        status["candidates"][candidate_name] = {**status["candidates"].get(candidate_name, {}), "run_dir": str(run_dir), "config": str(config_path), "benchmark": bench}
        update(candidate_name + "-ready")

    if not status["candidates"]:
        raise RuntimeError("No candidate passed its benchmark")

    # Equal wall-clock screening. Each candidate retains its optimizer and EMA
    # so the winner continues from its screen rather than starting over.
    for name, candidate in status["candidates"].items():
        run_dir, config = Path(candidate["run_dir"]), Path(candidate["config"])
        if not candidate.get("screen_complete") and time.time() < train_end - 600:
            code = stage("train", name + "-screen-train", min(time.time()+2700, train_end-300),
                         config=config, run_dir=run_dir)
            if code == 0 and (run_dir / "latest.pt").exists():
                snapshot_path = run_dir / "screen_ema.pt"
                snapshot(run_dir / "latest.pt", snapshot_path)
                candidate["screen_complete"] = True
                candidate["screen_snapshot"] = str(snapshot_path)
                update(name + "-screen-saved")
        if candidate.get("screen_complete"):
            evaluate(name + "-screen", config, Path(candidate["screen_snapshot"]))

    valid = [v for v in status["evaluations"] if v["name"].endswith("-screen") and math.isfinite(v["fid"])]
    if not status.get("winner"):
        if valid:
            status["winner"] = min(valid, key=lambda v: v["fid"])["name"].removesuffix("-screen")
        else:
            finished = [name for name, candidate in status["candidates"].items() if candidate.get("screen_complete")]
            if not finished:
                raise RuntimeError("No candidate completed training")
            status["winner"] = finished[0]
            status["errors"].append("Screening FID unavailable; defaulted to first healthy candidate")
    winner = status["winner"]
    candidate = status["candidates"][winner]
    run_dir, config = Path(candidate["run_dir"]), Path(candidate["config"])
    update("selected-" + winner)
    round_index = int(status.get("completed_rounds", 0))
    while time.time() < train_end - 300:
        before_steps = read_json(run_dir / "run.json", {}).get("completed_steps", 0)
        planned_steps = yaml.safe_load(config.read_text())["training"]["num_steps"]
        if before_steps >= planned_steps:
            break
        round_index += 1
        code = stage("train", f"{winner}-round{round_index}-train", min(time.time()+7200, train_end-120),
                     config=config, run_dir=run_dir)
        if code != 0:
            break
        after_steps = read_json(run_dir / "run.json", {}).get("completed_steps", 0)
        if after_steps <= before_steps:
            status["errors"].append(f"{winner}-round{round_index}: no optimizer progress; stopped retries")
            break
        snap = run_dir / f"round{round_index}_ema.pt"
        snapshot(run_dir / "latest.pt", snap)
        evaluate(f"{winner}-round{round_index}", config, snap)
        status["completed_rounds"] = round_index
        update(f"{winner}-round{round_index}-complete")
        current = read_json(run_dir / "run.json", {})
        cfg = yaml.safe_load(config.read_text())
        if current.get("completed_steps", 0) >= cfg["training"]["num_steps"]:
            break

    # Include the fixed-validation best snapshot, which need not be the best FID.
    if (run_dir / "best_val.pt").exists() and time.time() < deadline - 1800:
        best = run_dir / "best_val_ema.pt"
        snapshot(run_dir / "best_val.pt", best)
        evaluate(winner + "-best-val", config, best)

    screening = [v for v in status["evaluations"] if v.get("num_generated") == 2048 and math.isfinite(v["fid"])]
    finalists = sorted(screening, key=lambda v: v["fid"])[:2]
    for index, candidate_result in enumerate(finalists):
        if time.time() > deadline - 300:
            break
        # First finalist gets up to 35 min, second uses remaining budget.
        available = deadline - 90 - time.time()
        seconds = min(2100, available) if index == 0 else available
        evaluate(candidate_result["name"] + "-final10k", Path(candidate_result["config"]),
                 Path(candidate_result["snapshot"]), samples=10000, steps=100, seconds=seconds)

    final_results = [v for v in status["evaluations"] if v.get("num_generated") == 10000 and v.get("ode_steps") == 100 and math.isfinite(v["fid"])]
    if final_results:
        selected = min(final_results, key=lambda v: v["fid"])
        status["selected_model"] = {k: selected[k] for k in ("name", "fid", "snapshot", "config", "checkpoint_step", "num_generated", "ode_steps")}
        status["selected_model"]["selection_protocol"] = "matched 10k Euler100 FID among evaluated campaign finalists"
    elif screening:
        selected = min(screening, key=lambda v: v["fid"])
        status["selected_model"] = {k: selected[k] for k in ("name", "fid", "snapshot", "config", "checkpoint_step", "num_generated", "ode_steps")}
        status["selected_model"]["selection_protocol"] = "provisional: screening FID only; final10k unavailable"
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
    parser.add_argument("--publish-results", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
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

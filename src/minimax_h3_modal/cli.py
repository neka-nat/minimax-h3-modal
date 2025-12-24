"""`h3`: command line client for the deployed Modal app.

Only needs `modal`, `typer` and `pydantic` locally; all GPU dependencies live in
the Modal image. Generations are spawned as Modal jobs so a dropped connection
never wastes GPU time: `h3 jobs fetch <job id>` picks the result up later.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import typer

from . import config as cfg
from .inputs import build_request
from .schemas import ASPECT_PRESETS, DEFAULT_STEPS, GenerationRequest, GenerationResult

app = typer.Typer(
    help="Generate video with MiniMax-H3 on Modal. Deploy first: modal deploy -m minimax_h3_modal.app",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
jobs_app = typer.Typer(help="Inspect and fetch spawned generation jobs.", no_args_is_help=True)
app.add_typer(jobs_app, name="jobs")

JOBS_FILE = Path.home() / ".cache" / "minimax-h3" / "jobs.jsonl"


# ----------------------------------------------------------------- helpers
def _fail(message: str) -> None:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _generator(partition: str) -> Any:
    import modal

    try:
        return modal.Cls.from_name(cfg.APP_NAME, "H3Generator")(partition=partition)
    except modal.exception.NotFoundError:
        _fail(f"app {cfg.APP_NAME!r} is not deployed; run: modal deploy -m minimax_h3_modal.app")


def _wait(function_call: Any, poll: float, what: str) -> Any:
    import modal

    t0 = time.monotonic()
    while True:
        try:
            return function_call.get(timeout=poll)
        except TimeoutError:
            typer.echo(f"\r  {what} ... {int(time.monotonic() - t0)}s", nl=False, err=True)
        except modal.exception.FunctionTimeoutError:
            typer.echo("", err=True)
            _fail("the remote call hit its timeout")
        except modal.exception.RemoteError as e:
            typer.echo("", err=True)
            _fail(f"remote failure: {e}")


def _record_job(job_id: str, req: GenerationRequest, out: Path | None) -> None:
    JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": req.task,
        "prompt": req.prompt[:120],
        "out": str(out) if out else None,
    }
    with JOBS_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _save_result(raw: Any, out: Path | None) -> Path:
    result = GenerationResult.model_validate(raw)
    path = out or Path(result.filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(result.video)
    typer.echo("", err=True)
    typer.echo(f"saved {path}  ({len(result.video) / 2**20:.1f} MiB)")
    typer.echo(
        f"  {result.task}  {result.width}x{result.height}  {result.num_frames} frames / {result.duration_s} s  "
        f"steps={result.num_inference_steps}  seed={result.seed}"
    )
    timings = "  ".join(f"{k}={v}" for k, v in result.timings.items())
    typer.echo(
        f"  {timings}  attention={result.attention_backend}  peak_gpu={result.peak_gpu_memory_gib} GiB  "
        f"container_load_s={result.container_load_s}"
    )
    typer.echo(f"  volume copy: {result.output_path}  (job {result.job_id})")
    return path


# ---------------------------------------------------------------- commands
@app.command()
def download(
    partition: Annotated[str, typer.Option(help="base (t2va/fl2va), ref (ref2va) or all")] = cfg.PARTITION_BASE,
    wait: Annotated[bool, typer.Option("--wait/--no-wait")] = True,
) -> None:
    """Download the model weights into the Modal Volume (one-off, ~144 GB per partition)."""
    import modal

    partitions = list(cfg.PARTITIONS) if partition == "all" else [partition]
    if any(p not in cfg.PARTITIONS for p in partitions):
        _fail(f"partition must be one of {cfg.PARTITIONS} or all")
    try:
        fn = modal.Function.from_name(cfg.APP_NAME, "download")
    except modal.exception.NotFoundError:
        _fail(f"app {cfg.APP_NAME!r} is not deployed; run: modal deploy -m minimax_h3_modal.app")
    for p in partitions:
        call = fn.spawn(p)
        typer.echo(f"download({p}) spawned as job {call.object_id}")
        if wait:
            info = _wait(call, poll=30.0, what=f"downloading {p}")
            typer.echo("", err=True)
            typer.echo(json.dumps(info))


@app.command()
def generate(
    prompt: Annotated[str, typer.Argument(help="What to generate; dialogue and sounds can be described too")],
    image: Annotated[Path | None, typer.Option("--image", "-i", help="First keyframe (fl2va)")] = None,
    last_image: Annotated[Path | None, typer.Option("--last-image", help="Last keyframe (fl2va)")] = None,
    ref: Annotated[
        list[str] | None,
        typer.Option("--ref", "-r", help="Reference file for ref2va, repeatable, in reading order. Prefix with image:/video:/audio: to force the kind"),
    ] = None,
    aspect: Annotated[str | None, typer.Option("--aspect", "-a", help=f"W:H, e.g. {', '.join(ASPECT_PRESETS)}; default: keyframe's ratio or 16:9")] = None,
    size: Annotated[str | None, typer.Option("--size", help="Explicit WIDTHxHEIGHT (multiples of 32), instead of --aspect")] = None,
    duration: Annotated[float, typer.Option("--duration", "-d", min=5.0, max=15.0, help="Seconds, 5 to 15 (snapped to the VAE's frame grid)")] = 5.0,
    steps: Annotated[int, typer.Option("--steps", min=2, max=200)] = DEFAULT_STEPS,
    seed: Annotated[int | None, typer.Option("--seed", min=0, help="Random if omitted")] = None,
    label: Annotated[str | None, typer.Option("--label", help="Tag used in the output file name")] = None,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Where to save the mp4")] = None,
    wait: Annotated[bool, typer.Option("--wait/--no-wait", help="Poll until the clip is ready and save it")] = True,
    poll: Annotated[float, typer.Option("--poll", help="Seconds between polls")] = 15.0,
) -> None:
    """Generate a video with stereo audio (t2va, fl2va with keyframes, ref2va with --ref)."""
    try:
        req = build_request(
            prompt,
            image=image,
            last_image=last_image,
            references=ref,
            aspect=aspect,
            size=size,
            duration=duration,
            steps=steps,
            seed=seed,
            label=label,
        )
    except (ValueError, FileNotFoundError) as e:
        _fail(str(e))
    call = _generator(req.partition).generate.spawn(req.model_dump())
    _record_job(call.object_id, req, out)
    typer.echo(f"job {call.object_id}: {req.task}, {req.num_frames} frames ({req.effective_duration_s:.2f} s), {req.num_inference_steps} steps")
    if not wait:
        typer.echo(f"fetch later with: h3 jobs fetch {call.object_id} -o {out or 'clip.mp4'}")
        return
    _save_result(_wait(call, poll, "generating"), out)


@jobs_app.command("status")
def jobs_status(job_id: str) -> None:
    """Show whether a job has finished (does not download the video)."""
    import modal

    call = modal.FunctionCall.from_id(job_id)
    try:
        raw = call.get(timeout=0)
    except TimeoutError:
        typer.echo(f"{job_id}: running")
        return
    except modal.exception.RemoteError as e:
        _fail(f"{job_id} failed: {e}")
    result = GenerationResult.model_validate(raw)
    typer.echo(f"{job_id}: done  {result.task} {result.width}x{result.height} {result.num_frames}f seed={result.seed} -> {result.output_path}")


@jobs_app.command("fetch")
def jobs_fetch(
    job_id: str,
    out: Annotated[Path | None, typer.Option("--out", "-o")] = None,
    poll: Annotated[float, typer.Option("--poll")] = 15.0,
) -> None:
    """Wait for a job (if needed) and save its mp4."""
    import modal

    _save_result(_wait(modal.FunctionCall.from_id(job_id), poll, "waiting"), out)


@jobs_app.command("list")
def jobs_list(limit: Annotated[int, typer.Option("--limit", "-n")] = 20) -> None:
    """List jobs spawned from this machine (newest last)."""
    if not JOBS_FILE.exists():
        typer.echo("no jobs recorded yet")
        return
    lines = JOBS_FILE.read_text(encoding="utf-8").splitlines()[-limit:]
    for line in lines:
        entry = json.loads(line)
        typer.echo(f"{entry['created_at']}  {entry['job_id']}  {entry['task']:6s}  {entry['prompt']}")


def main() -> None:  # console_scripts entry point
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()

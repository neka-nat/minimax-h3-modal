"""Modal app: weight download, the GPU inference class and a dev entrypoint.

    modal run -m minimax_h3_modal.app::download --partition base   # once, ~144 GB
    modal deploy -m minimax_h3_modal.app                            # then use the `h3` CLI
    modal run -m minimax_h3_modal.app --prompt "..." --duration 5   # dev / benchmarking
"""


import json
import re
import time
from pathlib import Path

import modal

from . import config as cfg

app = modal.App(cfg.APP_NAME)

weights_volume = modal.Volume.from_name(cfg.WEIGHTS_VOLUME, create_if_missing=True, version=2)
outputs_volume = modal.Volume.from_name(cfg.OUTPUTS_VOLUME, create_if_missing=True, version=2)

image = (
    modal.Image.debian_slim(python_version=cfg.IMAGE_PYTHON_VERSION)
    .uv_pip_install(*cfg.IMAGE_PACKAGES)
    .env(
        {
            "HF_HOME": str(cfg.HF_HOME),
            "HF_HUB_CACHE": str(cfg.HF_HUB_CACHE),
            "HF_XET_CACHE": "/tmp/hf-xet-cache",  # keep the xet chunk cache off the Volume
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("minimax_h3_modal")
)


@app.function(
    image=image,
    volumes={str(cfg.HF_HOME): weights_volume},
    cpu=8,
    memory=32 * 1024,
    timeout=6 * 3600,
)
def download(partition: str = cfg.PARTITION_BASE) -> dict:
    """Populate the hub-cache Volume with the diffusers layout of one partition."""
    from huggingface_hub import snapshot_download

    if partition not in cfg.PARTITIONS:
        raise ValueError(f"partition must be one of {cfg.PARTITIONS}, got {partition!r}")
    patterns = cfg.SHARED_PATTERNS + cfg.PARTITION_PATTERNS[partition]
    t0 = time.perf_counter()
    snapshot = snapshot_download(cfg.MODEL_ID, allow_patterns=patterns)
    weights_volume.commit()
    total = sum(p.stat().st_size for p in Path(snapshot).rglob("*") if p.is_file())
    return {
        "partition": partition,
        "snapshot": snapshot,
        "gib": round(total / 2**30, 1),
        "seconds": round(time.perf_counter() - t0),
    }


@app.function(image=image, cpu=2, memory=4 * 1024, timeout=600)
def env_check() -> dict:
    """Report the library versions inside the image (`modal run -m minimax_h3_modal.app::env_check`)."""
    import importlib
    import platform

    versions = {"python": platform.python_version()}
    for name in ("torch", "torchvision", "diffusers", "transformers", "accelerate", "kernels", "av", "huggingface_hub"):
        try:
            versions[name] = str(importlib.import_module(name).__version__)
        except Exception as e:  # noqa: BLE001
            versions[name] = f"MISSING ({type(e).__name__}: {e})"
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Blocks  # noqa: F401 - import check

    versions["minimax_h3_blocks"] = "ok"
    print(versions)
    return versions


@app.cls(
    image=image,
    gpu=cfg.GPU,
    cpu=cfg.CPU_CORES,
    memory=cfg.MEMORY_MIB,
    timeout=cfg.TIMEOUT_S,
    startup_timeout=cfg.STARTUP_TIMEOUT_S,
    scaledown_window=cfg.SCALEDOWN_S,
    max_containers=cfg.MAX_CONTAINERS,
    region=cfg.REGION,
    volumes={str(cfg.HF_HOME): weights_volume, str(cfg.OUTPUTS_DIR): outputs_volume},
)
class H3Generator:
    # One container pool per transformer partition: "base" (t2va, fl2va) or "ref" (ref2va).
    partition: str = modal.parameter(default=cfg.PARTITION_BASE)

    @modal.enter()
    def load(self) -> None:
        import logging

        from .pipeline import H3Pipeline

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)

        self.pipeline = H3Pipeline(
            self.partition,
            attention_backend=cfg.ATTENTION_BACKEND,
            memory_reserve_margin=cfg.MEMORY_RESERVE_MARGIN,
        )
        self.pipeline.load()

    @modal.method()
    def generate(self, request: dict) -> dict:
        """Generate one clip; returns a `GenerationResult` dump (mp4 bytes included)."""
        from .schemas import GenerationRequest, GenerationResult

        req = GenerationRequest.model_validate(request)
        if req.partition != self.partition:
            raise ValueError(f"{req.task} requests must go to H3Generator(partition={req.partition!r})")

        job_id = modal.current_function_call_id() or f"local-{int(time.time())}"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        label = re.sub(r"[^A-Za-z0-9_-]+", "-", req.label).strip("-")[:40] if req.label else ""
        stem = "-".join(part for part in (stamp, req.task, label, job_id[-8:]) if part)
        out_dir = Path(cfg.OUTPUTS_DIR) / job_id

        mp4, info = self.pipeline.generate(req, out_dir, stem)
        meta = {
            "job_id": job_id,
            "prompt": req.prompt,
            "aspect_ratio": req.aspect_ratio,
            "requested_duration_s": req.duration_s,
            "keyframes": [a.filename for a in (req.image, req.last_image) if a],
            "references": [f"{a.kind}:{a.filename}" for a in req.references],
            "container_load_s": self.pipeline.load_s,
            "gpu": cfg.GPU,
            **info,
        }
        (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        outputs_volume.commit()

        return GenerationResult(
            job_id=job_id,
            task=req.task,
            filename=mp4.name,
            video=mp4.read_bytes(),
            seed=info["seed"],
            height=info["height"],
            width=info["width"],
            num_frames=info["num_frames"],
            duration_s=info["duration_s"],
            num_inference_steps=req.num_inference_steps,
            output_path=str(mp4),
            attention_backend=info["attention_backend"],
            timings=info["timings"],
            peak_gpu_memory_gib=info["peak_gpu_memory_gib"],
            container_load_s=self.pipeline.load_s,
        ).model_dump()


@app.local_entrypoint()
def main(
    prompt: str = "A red fox trotting through a snowy pine forest, snow crunching underfoot",
    image: str = "",
    last_image: str = "",
    ref: str = "",  # comma-separated reference files, optionally prefixed image:/video:/audio:
    aspect: str = "",
    size: str = "",  # WIDTHxHEIGHT, multiples of 32
    duration: float = 5.0,
    steps: int = 50,
    seed: int = -1,
    label: str = "",
    out: str = "",
) -> None:
    """Dev entrypoint: run one generation synchronously and save the mp4 locally."""
    from .inputs import build_request
    from .schemas import GenerationResult

    req = build_request(
        prompt,
        image=Path(image) if image else None,
        last_image=Path(last_image) if last_image else None,
        references=[r for r in ref.split(",") if r],
        aspect=aspect or None,
        size=size or None,
        duration=duration,
        steps=steps,
        seed=None if seed < 0 else seed,
        label=label or None,
    )
    print(f"task={req.task} partition={req.partition} frames={req.num_frames} steps={req.num_inference_steps}")
    t0 = time.perf_counter()
    raw = H3Generator(partition=req.partition).generate.remote(req.model_dump())
    result = GenerationResult.model_validate(raw)
    out_path = Path(out or result.filename)
    out_path.write_bytes(result.video)
    print(
        f"saved {out_path} ({len(result.video) / 2**20:.1f} MiB) seed={result.seed} "
        f"{result.width}x{result.height} {result.num_frames}f/{result.duration_s}s "
        f"attention={result.attention_backend} peak_gpu={result.peak_gpu_memory_gib} GiB"
    )
    print(f"timings={result.timings} container_load_s={result.container_load_s} wall_s={time.perf_counter() - t0:.0f}")
    print(f"volume copy: {result.output_path}")

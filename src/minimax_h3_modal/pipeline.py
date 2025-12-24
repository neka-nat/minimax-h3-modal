"""Wrapper around diffusers' MiniMax-H3 ModularPipeline.

Runs inside the Modal container. torch / diffusers are imported lazily so this
module can be imported (and the rest of the package unit-tested) locally.
"""

from __future__ import annotations

import io
import logging
import random
import tempfile
import time
from pathlib import Path
from typing import Any

from .config import MODEL_ID, PARTITION_BASE, PARTITION_REF, PARTITIONS
from .schemas import FPS, GenerationRequest, MediaAttachment

log = logging.getLogger(__name__)

# Hub repositories behind diffusers' "*_hub" attention backends, fetched eagerly
# at load time so a missing kernel degrades to native attention instead of
# failing the first generation.
_HUB_KERNEL_REPOS = {
    "_flash_3_hub": "kernels-community/flash-attn3",
    "_flash_3_varlen_hub": "kernels-community/flash-attn3",
    "flash_hub": "kernels-community/flash-attn",
    "flash_varlen_hub": "kernels-community/flash-attn",
}

_DEFAULT_SUFFIX = {"image": ".png", "video": ".mp4", "audio": ".wav"}


class H3Pipeline:
    def __init__(
        self,
        partition: str,
        *,
        device: str = "cuda",
        attention_backend: str = "_flash_3_hub",
        memory_reserve_margin: str = "12GB",
    ) -> None:
        if partition not in PARTITIONS:
            raise ValueError(f"partition must be one of {PARTITIONS}, got {partition!r}")
        self.partition = partition
        self.device = device
        self.requested_backend = attention_backend
        self.memory_reserve_margin = memory_reserve_margin
        self.pipe: Any = None
        self.manager: Any = None
        self.attention_backend = "native"
        self.load_s: float | None = None

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        import torch
        from diffusers import ComponentsManager, ModularPipeline

        t0 = time.perf_counter()
        manager = ComponentsManager()
        if self.partition == PARTITION_REF:
            # Selecting the workflow up front keeps only its blocks and components,
            # so this never touches `transformer/`.
            pipe = ModularPipeline.from_pretrained(MODEL_ID, workflow="ref2va", components_manager=manager)
            pipe.load_components(dtype=torch.bfloat16)
            transformer = pipe.transformer_ref
        else:
            # The full pipeline picks t2va / fl2va per call from the inputs; loading
            # the fl2va workflow fetches `transformer/` plus every shared component.
            pipe = ModularPipeline.from_pretrained(MODEL_ID, components_manager=manager)
            pipe.load_components(workflow="fl2va", dtype=torch.bfloat16)
            transformer = pipe.transformer
        log.info("components loaded in %.0fs", time.perf_counter() - t0)

        # Weights live in host RAM; the manager moves onto the GPU what each block
        # needs and evicts the rest when room is short.
        manager.enable_auto_cpu_offload(device=self.device, memory_reserve_margin=self.memory_reserve_margin)
        self.attention_backend = self._set_attention_backend(transformer)
        # The two VAEs run their attention in float32 (their weights are pinned to fp32), which the
        # FlashAttention kernels refuse, so pin them to PyTorch's native attention explicitly instead of
        # letting them follow the dispatcher's default.
        for name in ("vae", "audio_vae"):
            model = getattr(pipe, name, None)
            if model is not None and hasattr(model, "set_attention_backend"):
                try:
                    model.set_attention_backend("native")
                except Exception as e:  # noqa: BLE001
                    log.warning("could not pin %s to native attention: %s", name, e)

        self.pipe = pipe
        self.manager = manager
        self.load_s = time.perf_counter() - t0
        log.info("pipeline ready in %.0fs (attention=%s)", self.load_s, self.attention_backend)

    def _set_attention_backend(self, transformer: Any) -> str:
        wanted = self.requested_backend
        if wanted in ("", "native"):
            return "native"
        try:
            repo = _HUB_KERNEL_REPOS.get(wanted)
            if repo:
                from kernels import get_kernel

                get_kernel(repo, version=1)  # downloads + imports the prebuilt kernel now
            transformer.set_attention_backend(wanted)
            return wanted
        except Exception as e:  # noqa: BLE001 - any failure here must not take the container down
            log.warning("attention backend %r unavailable (%s: %s); using native attention", wanted, type(e).__name__, e)
            try:
                transformer.reset_attention_backend()
            except Exception:  # noqa: BLE001
                pass
            return "native"

    # -------------------------------------------------------------- generate
    def generate(self, request: GenerationRequest, out_dir: Path, stem: str) -> tuple[Path, dict[str, Any]]:
        """Run one request and write `<out_dir>/<stem>.mp4` (H.264 + stereo AAC)."""
        import torch
        from diffusers.utils.export_utils import encode_video

        if self.pipe is None:
            raise RuntimeError("call load() first")
        if request.partition != self.partition:
            raise ValueError(f"{request.task} needs the {request.partition!r} partition, this container holds {self.partition!r}")

        seed = request.seed if request.seed is not None else random.randrange(2**31)
        kwargs: dict[str, Any] = {}
        keyframe_size: tuple[int, int] | None = None
        if request.image is not None:
            kwargs["image"] = self._open_image(request.image)
            keyframe_size = kwargs["image"].size
        if request.last_image is not None:
            kwargs["last_image"] = self._open_image(request.last_image)
            keyframe_size = keyframe_size or kwargs["last_image"].size
        height, width = request.resolve_canvas(keyframe_size)

        timings: dict[str, float] = {}
        use_cuda = torch.cuda.is_available()
        with tempfile.TemporaryDirectory() as tmp:
            if request.references:
                kwargs["references"] = self._build_references(request.references, Path(tmp))
            if use_cuda:
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            results = self.pipe(
                prompt=request.prompt,
                height=height,
                width=width,
                num_frames=request.num_frames,
                num_inference_steps=request.num_inference_steps,
                generator=torch.Generator().manual_seed(seed),
                output=["videos", "audio", "sampling_rate"],
                **kwargs,
            )
            timings["generate_s"] = round(time.perf_counter() - t0, 1)

        frames = results["videos"][0]
        audio = results["audio"][0]  # (2, num_samples)
        if hasattr(audio, "detach"):
            audio = audio.detach().float().cpu()
        sampling_rate = int(results["sampling_rate"])

        out_dir.mkdir(parents=True, exist_ok=True)
        mp4 = out_dir / f"{stem}.mp4"
        t0 = time.perf_counter()
        encode_video(frames, fps=FPS, output_path=str(mp4), audio=audio, audio_sample_rate=sampling_rate)
        timings["encode_s"] = round(time.perf_counter() - t0, 1)

        info: dict[str, Any] = {
            "task": request.task,
            "seed": seed,
            "height": height,
            "width": width,
            "num_frames": len(frames),
            "duration_s": round(len(frames) / FPS, 3),
            "num_inference_steps": request.num_inference_steps,
            "sampling_rate": sampling_rate,
            "attention_backend": self.attention_backend,
            "timings": timings,
            "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1) if use_cuda else None,
        }
        del results, frames, audio
        return mp4, info

    @staticmethod
    def _open_image(attachment: MediaAttachment):
        from PIL import Image

        return Image.open(io.BytesIO(attachment.data)).convert("RGB")

    @staticmethod
    def _build_references(attachments: list[MediaAttachment], tmp: Path) -> list[Any]:
        from diffusers.modular_pipelines.minimax_h3 import (
            MiniMaxH3AudioReference,
            MiniMaxH3ImageReference,
            MiniMaxH3VideoReference,
        )

        classes = {"image": MiniMaxH3ImageReference, "video": MiniMaxH3VideoReference, "audio": MiniMaxH3AudioReference}
        references = []
        for i, att in enumerate(attachments):
            suffix = Path(att.filename).suffix or _DEFAULT_SUFFIX[att.kind]
            path = tmp / f"ref{i:02d}{suffix}"
            path.write_bytes(att.data)
            # `from_file` decodes through PyAV and keeps the container's frame rate /
            # sample rate on the reference, which the model resamples from.
            references.append(classes[att.kind].from_file(str(path)))
        return references

"""Deploy-time configuration.

Everything here is read when `modal deploy` / `modal run` imports the app, so the
GPU, memory and timeout knobs can be overridden with environment variables
without touching the code, e.g. `H3_GPU=H100 modal deploy -m minimax_h3_modal.app`.
"""

from __future__ import annotations

import os
from pathlib import PurePosixPath

APP_NAME = "minimax-h3"
MODEL_ID = "MiniMaxAI/MiniMax-H3"

# Modal Volumes: one holds the Hugging Face hub cache (weights), one the outputs.
WEIGHTS_VOLUME = "minimax-h3-hf-cache"
OUTPUTS_VOLUME = "minimax-h3-outputs"
HF_HOME = PurePosixPath("/cache/huggingface")
HF_HUB_CACHE = HF_HOME / "hub"
OUTPUTS_DIR = PurePosixPath("/outputs")

# Which transformer partition a container loads. `base` serves t2va and fl2va,
# `ref` serves ref2va. Everything else (Qwen3-VL conditioner, VAEs) is shared.
PARTITION_BASE = "base"
PARTITION_REF = "ref"
PARTITIONS = (PARTITION_BASE, PARTITION_REF)

# Hub file patterns for the diffusers layout of the checkpoint (the FL2VA/ and
# Ref2VA/ folders are the SGLang/vLLM layout and are not needed here).
SHARED_PATTERNS = [
    "modular_model_index.json",
    "model_index.json",
    "LICENSE",
    "text_encoder/*",
    "tokenizer/*",
    "processor/*",
    "vae/*",
    "audio_vae/*",
    "scheduler/*",
    "audio_scheduler/*",
]
PARTITION_PATTERNS = {
    PARTITION_BASE: ["transformer/*"],
    PARTITION_REF: ["transformer_ref/*"],
}


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


# GPU container sizing. H200 (141 GB) keeps the transformer and both VAEs
# resident and only swaps the 62 GB text encoder; H100 (80 GB) also works via
# the same auto CPU offload but with much less headroom for long clips.
GPU = os.environ.get("H3_GPU", "H200")
CPU_CORES = _env_float("H3_CPU", 8)
MEMORY_MIB = int(_env_float("H3_MEMORY_GIB", 192) * 1024)  # host RAM for offloaded bf16 weights
TIMEOUT_S = int(_env_float("H3_TIMEOUT_MIN", 60) * 60)
STARTUP_TIMEOUT_S = int(_env_float("H3_STARTUP_TIMEOUT_MIN", 45) * 60)
SCALEDOWN_S = int(_env_float("H3_SCALEDOWN_MIN", 5) * 60)  # Modal caps this at 20 min
MAX_CONTAINERS = int(_env_float("H3_MAX_CONTAINERS", 1))
REGION = os.environ.get("H3_REGION") or None  # e.g. "jp"; costs 1.75x, see README

# Inference knobs.
ATTENTION_BACKEND = os.environ.get("H3_ATTENTION_BACKEND", "_flash_3_hub")
MEMORY_RESERVE_MARGIN = os.environ.get("H3_MEMORY_RESERVE_MARGIN", "12GB")

# Modal's default image builder for this workspace supports Python 3.10-3.12,
# so the container runs 3.12 even though the local project uses 3.13.
IMAGE_PYTHON_VERSION = os.environ.get("H3_IMAGE_PYTHON", "3.12")

# Pinned image dependencies. torch 2.12 on PyPI is a CUDA 13.0 build, which
# matches Modal's driver (CUDA 13.0) and the prebuilt FlashAttention-3 Hub
# kernel variant `torch212-cxx11-cu130`.
IMAGE_PACKAGES = [
    "torch==2.12.1",
    "torchvision==0.27.1",  # Qwen3VLProcessor (video processor) needs it, even for text-only prompts
    "diffusers==0.40.0",
    "transformers==5.17.0",
    "accelerate==1.15.0",
    "kernels==0.17.1",
    "av==18.1.0",
    "huggingface_hub[hf_xet]>=1.23,<2.0",
    "pydantic>=2,<3",
    "pillow",
    "numpy",
]

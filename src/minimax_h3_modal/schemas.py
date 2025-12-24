"""Request/response models shared by the local CLI and the Modal app.

The canvas and frame arithmetic mirrors diffusers' MiniMax-H3 implementation
(`diffusers.modular_pipelines.minimax_h3.modular_pipeline`) for the released
checkpoint, so that invalid requests fail locally before any GPU time is spent.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# Model constants of the released checkpoint.
FPS = 24
MIN_DURATION_S = 5.0
MAX_DURATION_S = 15.0
CANVAS_MULTIPLE = 32
CANVAS_SHORT_EDGE = 768
CANVAS_MAX_PIXELS = 768 * 1344
MIN_ASPECT_RATIO = 1 / 4
MAX_ASPECT_RATIO = 4.0
FRAMES_PER_CHUNK = 17  # video VAE clip_length
LATENTS_PER_CHUNK = 5  # video VAE tokens_chunk_size
MAX_PROMPT_CHARS = 7000
MAX_IMAGE_REFS = 9
MAX_VIDEO_REFS = 3
MAX_AUDIO_REFS = 3
MAX_REFS = 12
DEFAULT_STEPS = 50  # reference setting of the SGLang / vLLM recipes

ASPECT_PRESETS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")

Task = Literal["t2va", "fl2va", "ref2va"]
Partition = Literal["base", "ref"]
MediaKind = Literal["image", "video", "audio"]


def parse_aspect(text: str) -> tuple[float, float]:
    """Parse "16:9" (or "16x9" / "16/9") into (width, height) ratio terms."""
    for sep in (":", "x", "/"):
        if sep in text:
            left, _, right = text.partition(sep)
            try:
                w, h = float(left), float(right)
            except ValueError:
                break
            if w > 0 and h > 0:
                return w, h
            break
    raise ValueError(f"aspect ratio must look like W:H, e.g. 16:9, got {text!r}")


def resolve_canvas(aspect_w: float, aspect_h: float) -> tuple[int, int]:
    """Resolve an aspect ratio into the (height, width) canvas MiniMax-H3 uses.

    Same arithmetic as diffusers' `resolve_canvas_size`: short edge 768, area
    capped at 768*1344, both axes rounded to a multiple of 32.
    """
    if aspect_w <= 0 or aspect_h <= 0:
        raise ValueError(f"aspect ratio must be positive, got {aspect_w}:{aspect_h}")
    ratio = aspect_w / aspect_h
    if not MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO:
        raise ValueError(
            f"MiniMax-H3 supports aspect ratios from 1:{1 / MIN_ASPECT_RATIO:g} to "
            f"{MAX_ASPECT_RATIO:g}:1, got {aspect_w:g}:{aspect_h:g}"
        )
    if ratio >= 1.0:
        width, height = CANVAS_SHORT_EDGE * ratio, float(CANVAS_SHORT_EDGE)
    else:
        width, height = float(CANVAS_SHORT_EDGE), CANVAS_SHORT_EDGE / ratio
    area = width * height
    if area > CANVAS_MAX_PIXELS:
        scale = (CANVAS_MAX_PIXELS / area) ** 0.5
        width, height = width * scale, height * scale
    m = CANVAS_MULTIPLE
    return max(m, round(height / m) * m), max(m, round(width / m) * m)


def align_num_frames(num_frames: int) -> int:
    """Snap a frame count up to the next `17 * n + 5` the video VAE can decode."""
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    while num_frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        num_frames += 1
    return num_frames


def max_num_frames() -> int:
    """Largest aligned frame count whose duration still fits in 15 s (345 frames)."""
    n = int(MAX_DURATION_S * FPS)
    while n % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        n -= 1
    return n


def duration_to_num_frames(seconds: float) -> int:
    """Turn a requested duration into the aligned frame count the model will generate.

    The pipeline rounds up to `17n + 5` and then rejects durations above 15 s, so a
    request for the full 15 s is clamped to the last valid count (345 frames).
    """
    if not MIN_DURATION_S <= seconds <= MAX_DURATION_S:
        raise ValueError(
            f"duration must be between {MIN_DURATION_S:g} and {MAX_DURATION_S:g} seconds, got {seconds:g}"
        )
    return min(align_num_frames(round(seconds * FPS)), max_num_frames())


def num_frames_to_duration(num_frames: int) -> float:
    return num_frames / FPS


class MediaAttachment(BaseModel):
    """A media file sent inline to the container."""

    kind: MediaKind
    filename: str
    data: bytes

    @field_validator("data")
    @classmethod
    def _non_empty(cls, v: bytes) -> bytes:
        if not v:
            raise ValueError("attachment is empty")
        return v


class GenerationRequest(BaseModel):
    prompt: str
    image: MediaAttachment | None = None  # keyframe the video starts from
    last_image: MediaAttachment | None = None  # keyframe the video ends on
    references: list[MediaAttachment] = Field(default_factory=list)  # ref2va, in reading order
    aspect_ratio: str | None = None  # "16:9"; None = keyframe's ratio, else 16:9
    height: int | None = None
    width: int | None = None
    duration_s: float = 5.0
    num_inference_steps: int = Field(default=DEFAULT_STEPS, ge=2, le=200)
    seed: int | None = Field(default=None, ge=0)
    label: str | None = None  # free-form tag used in output file names

    @field_validator("prompt")
    @classmethod
    def _prompt(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("prompt is empty")
        if len(v) > MAX_PROMPT_CHARS:
            raise ValueError(f"prompt is longer than {MAX_PROMPT_CHARS} characters")
        return v

    @field_validator("image", "last_image")
    @classmethod
    def _keyframe_kind(cls, v: MediaAttachment | None) -> MediaAttachment | None:
        if v is not None and v.kind != "image":
            raise ValueError("keyframes must be images")
        return v

    @field_validator("duration_s")
    @classmethod
    def _duration(cls, v: float) -> float:
        if not MIN_DURATION_S <= v <= MAX_DURATION_S:
            raise ValueError(f"duration must be between {MIN_DURATION_S:g} and {MAX_DURATION_S:g} seconds")
        return v

    @field_validator("aspect_ratio")
    @classmethod
    def _aspect(cls, v: str | None) -> str | None:
        if v is not None:
            resolve_canvas(*parse_aspect(v))
        return v

    @model_validator(mode="after")
    def _consistency(self) -> GenerationRequest:
        if (self.height is None) != (self.width is None):
            raise ValueError("height and width must be given together")
        if self.height is not None:
            assert self.width is not None
            if self.height % CANVAS_MULTIPLE or self.width % CANVAS_MULTIPLE:
                raise ValueError(f"height and width must be multiples of {CANVAS_MULTIPLE}")
            ratio = self.width / self.height
            if not MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO:
                raise ValueError("width/height ratio must be between 1:4 and 4:1")
            if self.aspect_ratio is not None:
                raise ValueError("give either an aspect ratio or an explicit size, not both")
        if self.references and (self.image or self.last_image):
            raise ValueError("references (ref2va) cannot be combined with keyframes (fl2va)")
        if self.references:
            counts = {"image": 0, "video": 0, "audio": 0}
            for ref in self.references:
                counts[ref.kind] += 1
            if len(self.references) > MAX_REFS:
                raise ValueError(f"at most {MAX_REFS} references")
            if counts["image"] > MAX_IMAGE_REFS:
                raise ValueError(f"at most {MAX_IMAGE_REFS} image references")
            if counts["video"] > MAX_VIDEO_REFS:
                raise ValueError(f"at most {MAX_VIDEO_REFS} video references")
            if counts["audio"] > MAX_AUDIO_REFS:
                raise ValueError(f"at most {MAX_AUDIO_REFS} audio references")
            if counts["audio"] == len(self.references):
                raise ValueError("audio references must be paired with at least one image or video reference")
        return self

    @property
    def task(self) -> Task:
        if self.references:
            return "ref2va"
        if self.image is not None or self.last_image is not None:
            return "fl2va"
        return "t2va"

    @property
    def partition(self) -> Partition:
        return "ref" if self.task == "ref2va" else "base"

    @property
    def num_frames(self) -> int:
        return duration_to_num_frames(self.duration_s)

    @property
    def effective_duration_s(self) -> float:
        return num_frames_to_duration(self.num_frames)

    def resolve_canvas(self, keyframe_size: tuple[int, int] | None = None) -> tuple[int, int]:
        """(height, width) to generate at. `keyframe_size` is the first keyframe's (width, height)."""
        if self.height is not None and self.width is not None:
            return self.height, self.width
        if self.aspect_ratio is not None:
            return resolve_canvas(*parse_aspect(self.aspect_ratio))
        if keyframe_size is not None:
            return resolve_canvas(float(keyframe_size[0]), float(keyframe_size[1]))
        return resolve_canvas(16, 9)


class GenerationResult(BaseModel):
    job_id: str
    task: Task
    filename: str
    video: bytes  # the mp4 (H.264 + stereo AAC)
    seed: int
    height: int
    width: int
    num_frames: int
    duration_s: float
    num_inference_steps: int
    output_path: str  # where the same file sits in the outputs Volume
    attention_backend: str
    timings: dict[str, float] = Field(default_factory=dict)  # seconds
    peak_gpu_memory_gib: float | None = None
    container_load_s: float | None = None

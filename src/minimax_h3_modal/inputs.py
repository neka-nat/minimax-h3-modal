"""Turn local files and CLI options into a `GenerationRequest` (no Modal/typer here)."""

from __future__ import annotations

from pathlib import Path

from .schemas import GenerationRequest, MediaAttachment, MediaKind

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus"}


def infer_kind(path: Path) -> MediaKind:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    if suffix in AUDIO_SUFFIXES:
        return "audio"
    raise ValueError(f"cannot tell whether {path.name} is an image, video or audio file; use kind:path")


def parse_reference(spec: str) -> Path | tuple[MediaKind, Path]:
    """`image:/path/a.png` forces a kind; a bare path infers it from the suffix."""
    for kind in ("image", "video", "audio"):
        prefix = f"{kind}:"
        if spec.startswith(prefix):
            return kind, Path(spec[len(prefix) :])
    return Path(spec)


def attachment_from_path(path: Path, kind: MediaKind | None = None) -> MediaAttachment:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    return MediaAttachment(kind=kind or infer_kind(path), filename=path.name, data=path.read_bytes())


def parse_size(text: str) -> tuple[int, int]:
    """"1344x768" -> (height, width)."""
    try:
        w_txt, h_txt = text.lower().replace("×", "x").split("x")
        return int(h_txt), int(w_txt)
    except ValueError as e:
        raise ValueError(f"size must look like WIDTHxHEIGHT, e.g. 1344x768, got {text!r}") from e


def build_request(
    prompt: str,
    *,
    image: Path | None = None,
    last_image: Path | None = None,
    references: list[str] | None = None,
    aspect: str | None = None,
    size: str | None = None,
    duration: float = 5.0,
    steps: int = 50,
    seed: int | None = None,
    label: str | None = None,
) -> GenerationRequest:
    height = width = None
    if size:
        height, width = parse_size(size)
    refs: list[MediaAttachment] = []
    for spec in references or []:
        parsed = parse_reference(spec)
        if isinstance(parsed, tuple):
            refs.append(attachment_from_path(parsed[1], parsed[0]))
        else:
            refs.append(attachment_from_path(parsed))
    return GenerationRequest(
        prompt=prompt,
        image=attachment_from_path(image, "image") if image else None,
        last_image=attachment_from_path(last_image, "image") if last_image else None,
        references=refs,
        aspect_ratio=aspect,
        height=height,
        width=width,
        duration_s=duration,
        num_inference_steps=steps,
        seed=seed,
        label=label,
    )

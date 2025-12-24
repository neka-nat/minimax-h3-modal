import pytest
from pydantic import ValidationError

from minimax_h3_modal.inputs import infer_kind, parse_reference, parse_size
from minimax_h3_modal.schemas import (
    GenerationRequest,
    MediaAttachment,
    align_num_frames,
    duration_to_num_frames,
    max_num_frames,
    parse_aspect,
    resolve_canvas,
)
from pathlib import Path


@pytest.mark.parametrize(
    "aspect, expected",
    [
        ("16:9", (768, 1344)),
        ("9:16", (1344, 768)),
        ("1:1", (768, 768)),
        ("4:3", (768, 1024)),
        ("3:4", (1024, 768)),
        ("21:9", (672, 1536)),
    ],
)
def test_resolve_canvas_matches_released_checkpoint(aspect, expected):
    height, width = resolve_canvas(*parse_aspect(aspect))
    assert (height, width) == expected
    assert height % 32 == 0 and width % 32 == 0


def test_resolve_canvas_rejects_extreme_ratio():
    with pytest.raises(ValueError):
        resolve_canvas(5, 1)


def test_parse_aspect_accepts_common_separators():
    assert parse_aspect("16:9") == (16.0, 9.0)
    assert parse_aspect("16x9") == (16.0, 9.0)
    with pytest.raises(ValueError):
        parse_aspect("wide")


@pytest.mark.parametrize("requested, aligned", [(1, 5), (5, 5), (6, 22), (120, 124), (124, 124), (240, 243)])
def test_align_num_frames(requested, aligned):
    assert align_num_frames(requested) == aligned
    assert aligned % 17 == 5


@pytest.mark.parametrize("seconds, frames", [(5, 124), (6, 158), (10, 243), (14.375, 345), (15, 345)])
def test_duration_to_num_frames(seconds, frames):
    assert duration_to_num_frames(seconds) == frames
    assert frames / 24 <= 15.0


def test_max_num_frames_is_last_valid_grid_point():
    assert max_num_frames() == 345


@pytest.mark.parametrize("seconds", [4.9, 15.1])
def test_duration_out_of_range(seconds):
    with pytest.raises(ValueError):
        duration_to_num_frames(seconds)


def _png() -> MediaAttachment:
    return MediaAttachment(kind="image", filename="a.png", data=b"\x89PNG")


def _wav() -> MediaAttachment:
    return MediaAttachment(kind="audio", filename="a.wav", data=b"RIFF")


def test_task_detection():
    assert GenerationRequest(prompt="x").task == "t2va"
    assert GenerationRequest(prompt="x", image=_png()).task == "fl2va"
    assert GenerationRequest(prompt="x", last_image=_png()).task == "fl2va"
    req = GenerationRequest(prompt="x", references=[_png(), _wav()])
    assert req.task == "ref2va" and req.partition == "ref"


def test_default_canvas_and_frames():
    req = GenerationRequest(prompt="x")
    assert req.resolve_canvas() == (768, 1344)
    assert req.resolve_canvas(keyframe_size=(1080, 1920)) == (1344, 768)  # portrait keyframe
    assert req.num_frames == 124 and req.effective_duration_s == pytest.approx(124 / 24)
    assert GenerationRequest(prompt="x", aspect_ratio="1:1").resolve_canvas(keyframe_size=(1920, 1080)) == (768, 768)
    assert GenerationRequest(prompt="x", height=544, width=960).resolve_canvas() == (544, 960)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(prompt="   "),
        dict(prompt="x", height=544),
        dict(prompt="x", height=500, width=960),
        dict(prompt="x", height=544, width=960, aspect_ratio="16:9"),
        dict(prompt="x", aspect_ratio="9:1"),
        dict(prompt="x", duration_s=16),
        dict(prompt="x", num_inference_steps=1),
        dict(prompt="x", image=_png(), references=[_png()]),
        dict(prompt="x", references=[_wav()]),
        dict(prompt="x", references=[_png()] * 10),
        dict(prompt="x", image=_wav()),
    ],
)
def test_invalid_requests(kwargs):
    with pytest.raises(ValidationError):
        GenerationRequest(**kwargs)


def test_request_round_trips_through_dict():
    req = GenerationRequest(prompt="x", image=_png(), duration_s=8, seed=7)
    assert GenerationRequest.model_validate(req.model_dump()) == req


def test_parse_size_is_width_x_height():
    assert parse_size("1344x768") == (768, 1344)
    with pytest.raises(ValueError):
        parse_size("768")


def test_reference_parsing():
    assert infer_kind(Path("a.JPG")) == "image"
    assert infer_kind(Path("a.mov")) == "video"
    assert infer_kind(Path("a.flac")) == "audio"
    assert parse_reference("audio:/tmp/x.bin") == ("audio", Path("/tmp/x.bin"))
    assert parse_reference("/tmp/x.png") == Path("/tmp/x.png")
    with pytest.raises(ValueError):
        infer_kind(Path("a.bin"))

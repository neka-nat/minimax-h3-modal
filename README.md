# minimax-h3-modal

Run the open-weight [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) video + stereo-audio
model on [Modal](https://modal.com) and drive it from the command line.

* Inference: the official diffusers 0.40 `MiniMaxH3ModularPipeline`, one GPU per container
  (H200 by default), weights streamed from a Modal Volume that doubles as the Hugging Face cache.
* Tasks: `t2va` (text), `fl2va` (first and/or last keyframe), `ref2va` (image / video / audio references).
* Output: mp4 with H.264 video at 24 fps and 32 kHz stereo AAC, 5 to 15 s, 768p canvas.

Powered by MiniMax H3. MiniMax H3 is licensed under the
[MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE),
Copyright © 2026 MiniMax. Read it before using the model: it excludes use in the USA, EU, UK and
South Korea without a separate agreement, and Modal's default region is in the USA (see "Region").

## Setup

```bash
uv sync                                   # local deps: modal, typer, pydantic (no torch)
uv run modal token new                    # once, if ~/.modal.toml does not exist yet
uv run modal deploy -m minimax_h3_modal.app
uv run h3 download                        # ~144 GB into the Volume, one-off (add --partition ref for ref2va)
```

`h3 download` runs on a CPU container and writes the diffusers layout of the checkpoint into the
`minimax-h3-hf-cache` Volume. The `base` partition (transformer, Qwen3-VL conditioner, both VAEs)
serves `t2va` and `fl2va`; `ref` adds the 66 GB `transformer_ref/` for `ref2va`.

## Generate

```bash
uv run h3 generate "A red fox trotting through a snowy pine forest, snow crunching underfoot" -o fox.mp4
uv run h3 generate "..." --image first.png --last-image last.png --duration 8 --seed 42 -o clip.mp4
uv run h3 generate "..." --ref subject.png --ref motion.mp4 --ref voice.wav -o ref.mp4
uv run h3 generate "..." --aspect 9:16 --steps 40 --no-wait      # prints a job id
uv run h3 jobs fetch fc-XXXXXXXX -o clip.mp4                       # pick it up later
uv run h3 jobs list
```

Options: `--aspect W:H` (21:9, 16:9, 4:3, 1:1, 3:4, 9:16 or any ratio between 1:4 and 4:1; the
canvas keeps a 768 px short edge and at most 1344x768), `--size WxH` for an explicit canvas
(multiples of 32; 960x544 is roughly 2.3x faster per step than 1344x768), `--duration` 5 to 15 s
(snapped to the VAE's `17n + 5` frame grid, so the longest clip is 345 frames = 14.4 s),
`--steps` (50 is the reference setting) and `--seed`. There is no negative prompt or guidance
scale: the released weights are guidance-distilled.

Every generation is spawned as a Modal job. The mp4 is returned to the CLI and also kept in the
`minimax-h3-outputs` Volume under `/outputs/<job id>/` next to a JSON with the request and timings
(`modal volume ls minimax-h3-outputs`).

For development without deploying, `modal run` executes one generation synchronously. Each
`modal run` starts its own ephemeral app, so it pays the cold start every time; the deployed app
behind `h3 generate` keeps its container warm for `H3_SCALEDOWN_MIN` minutes between calls:

```bash
uv run modal run -m minimax_h3_modal.app --prompt "..." --size 960x544 --duration 5 --out test.mp4
```

## Measured performance (H200, 2026-09-28)

50 steps, t2va unless noted, FlashAttention-3 Hub kernel on the transformer, bf16 weights with diffusers' auto CPU offload
(`H3_MEMORY_RESERVE_MARGIN=12GB`). "Cold" is the first generation in a fresh container: the
weights are memory-mapped from the Volume and actually read during that first pass.

| Canvas | Clip | Container | Generation | Peak GPU memory |
|---|---|---|---|---|
| 960x544 | 5.2 s (124 frames) | cold | 207 s | 127 GiB |
| 960x544 | 5.2 s (124 frames) | warm | 155 s | 78 GiB |
| 1344x768 | 5.2 s (124 frames) | cold | 485 s | 130 GiB |
| 1344x768 | 5.2 s (124 frames) | warm | 380 s | 131 GiB |
| 1344x768 | 10.1 s (243 frames) | warm | 891 s | 85 GiB |
| 960x544, fl2va (first keyframe) | 5.2 s (124 frames) | warm | 208 s | 128 GiB |
| 960x544, ref2va (1 image + 1 video ref) | 5.2 s (124 frames) | cold | 1045 s | 85 GiB |

Pipeline construction itself takes 35 to 80 s on top of the container boot. ref2va is much slower
per clip: every reference is encoded at its own resolution and the packed sequence grows with each one,
so the cold 1045 s above is for a 1344x768 image plus a 5 s video reference at 960x544 output. At Modal's H200 rate
of $4.54/h a warm 1344x768 clip costs about $0.48 and a 960x544 one about $0.20; a cold start
adds roughly $0.30. The peak numbers show why an 80 GB card is tight for the full canvas: on
the H200 the offloader keeps nearly everything resident for 5 s clips and evicts the 62 GB
conditioner for the 10 s one (hence the lower peak), on an H100 it would swap every request.

## Configuration

Deploy-time knobs are environment variables read by `src/minimax_h3_modal/config.py`
(`H3_GPU=H100 uv run modal deploy -m minimax_h3_modal.app`):

| Variable | Default | Meaning |
|---|---|---|
| `H3_GPU` | `H200` | Modal GPU spec (`H100`, `H100!`, `B200`, ...) |
| `H3_MEMORY_GIB` | `192` | Host RAM; the bf16 weights that are not on the GPU live here (~135 GiB) |
| `H3_CPU` | `8` | CPU cores |
| `H3_TIMEOUT_MIN` | `60` | Per-generation timeout |
| `H3_STARTUP_TIMEOUT_MIN` | `45` | Time allowed for loading the weights on cold start |
| `H3_SCALEDOWN_MIN` | `5` | Idle time before a warm container is released (max 20) |
| `H3_MAX_CONTAINERS` | `1` | Spend cap: containers per partition |
| `H3_REGION` | unset | Modal region, e.g. `jp` (1.75x price) |
| `H3_ATTENTION_BACKEND` | `_flash_3_hub` | FlashAttention-3 from the Hub on Hopper for the transformer; falls back to `native`. The two VAEs always use native attention because they run in fp32 |
| `H3_MEMORY_RESERVE_MARGIN` | `12GB` | GPU memory kept free for activations by the auto offloader |

## How it works

```
src/minimax_h3_modal/
  config.py    deploy-time settings (env overrides), Hub file patterns
  schemas.py   GenerationRequest / GenerationResult, canvas + frame arithmetic (mirrors diffusers)
  inputs.py    local files + CLI options -> GenerationRequest
  pipeline.py  ModularPipeline load / generate / mp4 mux (runs in the container)
  app.py       modal.App: image, Volumes, download(), H3Generator, dev entrypoint
  cli.py       `h3` (typer): talks to the deployed app via modal.Cls.from_name
```

`H3Generator` is parameterised by `partition` (`base` or `ref`), so each partition gets its own
container pool and only loads its transformer. On start the container loads every component in
bf16 into host RAM and registers them with diffusers' `ComponentsManager`, which moves the
conditioner, transformer and VAEs onto the GPU as each block needs them.

## Region and license

Modal schedules containers in the USA unless `H3_REGION` is set. The MiniMax H3 Community License
excludes use in the USA, EU, UK and South Korea; whether running the weights on US hardware from
elsewhere counts as such use is your call, not something this repository can settle. Modal offers
`H3_REGION=jp`, at a 1.75x price multiplier and without a guarantee of GPU availability.

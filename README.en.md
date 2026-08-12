# minimax-h3

[日本語](README.md) | **English**

**MiniMax H3 (Hailuo 3.0) — running the 33B omnimodal model that generates video + stereo
audio in a single denoise pass, on consumer-tier GPUs.** From text, an image reference, or
an audio reference it produces **video + stereo audio**, and it can also emit still images
only (T2I / Ref2I). It is built on the diffusers Modular Pipeline (PR #14355, pinned to
commit **f37ab93**). **Verified to run on two 8GB cards, and on a single real RTX 4060 Ti
16GB card** (see the support table below). This is a preliminary verification workspace for
eventual integration into [diffusers-server](https://github.com/animede/diffusers-server)
(the diffusers-server codebase itself has not been touched at all).

![Main GUI](docs/images/ui_main_en.png)

*A single-page browser UI. Five modes on tabs (video: **T2VA** = text / **FL2VA** = frame /
**Ref2VA** = reference; still image: **T2I** / **Ref2I**). The lower section is a gallery of
outputs, from which checked videos can be concatenated and exported. Prompts can be enhanced
into the official H3 guide format via a local LLM, and FirstBlockCache / Sage / Turbo /
quantization / low-VRAM mode are toggled from this panel **without a restart**. Includes a
Japanese/English switch.*

## VRAM × feature matrix (measured 2026-08-11)

○ = completes (measured) / △ = derived estimate (not yet measured) / × = OOM. All speeds are
on this box's **PCIe Gen3 x4 slot + sm_89 (4060 Ti) + SDPA**; on a **Gen4 x16 slot the weight
transfer is roughly 1/8** (see "An honest note on speed" below).

| Config | t2i 768² | t2va 5s 768² | ref2i | i2va (image ref) | audio ref | 768×1344 5s |
|---|---|---|---|---|---|---|
| **Real RTX 4060 Ti 16GB, single** | ○ 498s / peak 7.4GB | ○ 25 min / 11.4GB | ○ | ○ 39 min / 9.41GB | ○ 54 min / 11.96GB | ○ 66 min / 13.37GB (15.2GB real = ceiling) |
| **8GiB × 2** (compute + TE, ballast-simulated) | ○ 512s / 6.4GB | ○ 25.6 min / 7.23GB | ○ 17.7 min / 6.69GB | × OOM | × OOM | × OOM |
| **12GB, single** | △ estimate | △ | △ | △ | close to × (14.7GB real) | × |
| **48GB + 20GB** (two cards, turbo LoRA) | ○ 9.7s | ○ 44.2s | ○ | ○ | ○ | ○ |

- 12GB single is not measured. t2va real peak 11.4GB / i2va 9.41GB (11.2GB real) fit in 12GiB
  by calculation, but audio ref (14.7GB real) and 768×1344 (15.2GB real) exceed it.
- The key to the 16GB / 8GB configs is the **projected TE** (Qwen3-VL-4B + a trained linear
  map, ClipProj, 3.11GB in NF4 — a stand-in for the 32B text_encoder) + `H3_LOWVRAM=group`
  (streams the int8 transformer block by block, ~1.4GB GPU-resident) + fp16 video-VAE decode.
  Details are in the dated sections 2026-08-10 to 08-11.
- **The measurement environment is shared across two machines**: the 16GB / 8GB rows are on
  the **96GB box (RTX PRO 6000 + an added 4060 Ti 16GB)**, the 48GB+20GB row on the **48GB box
  (RTX PRO 5000 48GB + RTX 4000 SFF Ada 20GB)**. Older measurements (from the 96GB-single era)
  are kept as-is in the individual dated sections.

## Quick start by VRAM tier

Complete "[Installation](#installation)" and "[Obtaining the model](#obtaining-the-model)"
first. After launch, open `http://<host>:8611/` in a browser.

```bash
# Real 16GB, single card (e.g. an sm_89 4060 Ti)
H3_LOWVRAM=group H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_VIDEO_VAE_FP16=1 \
  H3_ATTN_BACKEND=default \
  venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
# Note: anything other than sm_120 requires H3_ATTN_BACKEND=default (the default sage is an sm_120-only build)

# 8GB × 2 (put the TE on the second GPU)
H3_LOWVRAM=group H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_VIDEO_VAE_FP16=1 \
  H3_ATTN_BACKEND=default H3_TE_DEVICE=cuda:1 \
  venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611

# 48GB tier (recommended: full speedups)
H3_LOWVRAM=1 H3_TE_PRUNE=1 H3_TE_DEVICE=cuda:1 H3_VIDEO_VAE_FP16=1 H3_KEEP_TRANSFORMER=1 \
  venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611

# 96GB tier (everything resident, no env)
venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
```

For finer per-tier launch examples (80/48/32/18GB) and every environment variable, see
"[VRAM support table and main environment variables](#vram-support-table-and-main-environment-variables-quick-reference)".

## What the speedups bought (measured summary)

**This repository's central finding: the bottleneck was not denoising but the fixed cost of
loading and freeing models.** The numbers below are collected from the sections further
down; nothing here was re-measured for this table.

### Cumulative: from the stock configuration to today

48GB box (PRO 5000 48GB + RTX 4000 20GB), the `H3_LOWVRAM=1` family, 768².

| Stage | t2i (1 image) | t2va (5s) |
|---|---|---|
| Stock configuration (30 steps, no turbo) | 157s | 351.4s |
| + lightx2v turbo LoRA (4 steps) | 157s | 143s |
| + `H3_TE_PREQUANT` (on-disk cache of the quantized TE) | 83.2s | — |
| + `H3_TE_DEVICE` (TE parked on a second GPU) | about 35s | 60.5s |
| + `H3_KEEP_TRANSFORMER` (transformer stays resident) | **9.7s** | **44.2s** |
| | **16x** | **8.0x** |

Turbo — the one change that actually speeds up denoising — **tops out at 2.6x**. The
remaining 3x-plus all comes from eliminating fixed costs (even at 30 steps without turbo,
removing the fixed costs alone takes t2i from 157s to 51.1s, a 3x gain).

**The current best is t2i 7.65s / t2va 5s 28.56s on the 96GB box** (measured 2026-08-12
with everything stacked: an int8 transformer resident on GPU0, the projected TE resident on
GPU1, turbo, sage, fp16 decode). The peak is **42.5GB** — the speed comes from running with
zero fixed costs, not from the capacity (see the dated 2026-08-12 section).

### Per-technique: how much it bought, and whether the output changes

Equivalence: ◎ = **identical output MD5** at the same seed (mathematically a no-op) /
○ = epsilon-level drift (not bit-identical, visually equivalent) / △ = an approximation or
a different model (quality can change).

| Technique | Flag | Measured effect | Equiv. | Measured on |
|---|---|---|---|---|
| On-disk cache of the quantized TE | `H3_TE_PREQUANT` (default on) | TE load 53.0→**29.5s**, whole request 128.6→**83.2s** (-35%) | ◎ | 48GB box |
| Dropping the TE's unused upper layers | `H3_TE_PRUNE=1` | TE load 42.3→**35.0s** (-17%) | ◎ (byte-identical mp4) | 48GB box |
| Parking the TE on a second GPU | `H3_TE_DEVICE=cuda:1` | t2i steady-state 78.4→**about 35s** (-55%) | ○ (sm difference, 0.084% relative RMS) | 48GB+20GB |
| Keeping the transformer resident | `H3_KEEP_TRANSFORMER=1` | the per-request reload cost disappears → t2i **9.7s** / t2va **44.2s** | ◎ | 48GB+20GB |
| FirstBlockCache | `H3_CACHE=fbc` (default) | denoise 157→**118s** (-25%, 7 of 30 steps skipped) | △ (skips steps) | 96GB box |
| lightx2v turbo LoRA | `H3_TURBO_LORA=1` | t2va 351.4→**143s** (2.6x; denoise alone is 7.6x) | △ (4-step distillation) | 48GB box |
| Batched still images | `/api/t2i_batch` | sequential 157→**67.5s/image** (3 scenes; asymptote ~31s) | ◎ | 48GB box |
| Sharing the reference KV prefix across scenes | `H3_REF_PREFIX_CACHE` (default on) | reference encode 212.5→**83.1s**, ref2i batch 164.9→**116.7s/image** (-29%) | ○ (PSNR 21.9-27.4dB) | 48GB box |
| Ultra-short clips for reference stills | `still=1` (124→22 frames) | denoise 102.2→**13.5s** (1/7.6) | — (different length, not comparable) | 48GB box |
| Projected TE (replacing the 32B TE) | `H3_TE_PROJ` | t2i 65.7→**33.5s**, t2va 162.1→**143.5s** | △ (PSNR 22.4dB, visually on par) | 96GB box |

### What cut VRAM

| Change | Effect |
|---|---|
| int8 quantization of the transformer (`H3_TRANSFORMER_QUANT=int8`) | 66.3 → **34.0GB** |
| NF4 for the projected TE (`H3_TE_PROJ`, default bnb-4bit) | TE residency 21.02 → **3.11GB** |
| Dropping the TE's unused upper layers (`H3_TE_PRUNE=1`) | 21.01 → **17.44GB** |
| fp16 decode for the video VAE (`H3_VIDEO_VAE_FP16=1`) | decode phase 16.29 → **about 11.4GB** |
| Streaming the uint8 conversion (2026-08-10) | decode-phase intermediates +2.65 → **+0.03GB** |
| Moving decode denormalization to the CPU (2026-08-12) | the **838MiB** bulk fp32 upcast at 768²×124 frames leaves the GPU |

Stacking all of this is what produces the
"[VRAM × feature matrix](#vram--feature-matrix-measured-2026-08-11)" at the top — running on
a single real 16GB card, and on 8GB×2.

## An honest note on speed

**The low-VRAM configs run, but they are slow.** `H3_LOWVRAM=group` streams the int8 weights
(~34GB) from CPU to GPU every step, so transfer is the bottleneck. The 16GB / 8GB figures in
the table above are on this box's **PCIe Gen3 x4 slot**, where most of the `16.5s/step` (t2i)
to `51s/step` (t2va) is transfer time. On a **proper Gen4 x16 slot the transfer is about 1/8**,
so these are values of "this box's slot," not "the 16GB card's performance." Also, on the
int8+SDPA trajectory FirstBlockCache does not kick in (`cache_skipped_steps: 0`), leaving up
to a 2x speedup on the table via threshold tuning (unverified).

On quality: both the projected TE (PSNR 22.4dB vs 32B) and int8+SDPA shift the composition
even at the same seed (**a trajectory divergence, not degradation**). In fact, in some
measured cases int8+SDPA improved prompt fidelity, and the reference (vision) path was
visually confirmed to match the 32B ground truth in fidelity. **Cross-config quality cannot
be judged by PSNR/MD5 — judge it by eye.**

## Recent updates (2026-08-11 to 08-12)

- **Both final goals reached**: full functionality on a single real RTX 4060 Ti 16GB (t2va
  768² peak 11.4GB; the maximum is 768×1344 at a real 15.2GB) / t2va 5s 768² completes at
  full resolution on 8GB × 2 (peak 7.23GB).
- Moved the decode-tail denormalization to CPU (bit-identical, lowers the decode peak on all
  configs).
- Fixed a dtype instant-death on `H3_VIDEO_VAE_FP16` × reference (added an fp16 autocast on
  the encode side too).
- Freed the residual pinned RAM in group mode, so t2va ↔ ref2va mode switching now works.

## Where things stand and where to resume (as of 2026-08-12)

Read this first when resuming work.

| What you want to know | Where to look |
|---|---|
| **Spec, performance, design** (what is achieved and how) | [docs/TECHNICAL_OVERVIEW.en.md](docs/TECHNICAL_OVERVIEW.en.md) |
| **Pitfalls, failures, operational lessons hit** (to avoid the same holes) | [docs/internal/TECHNICAL_REPORT.en.md](docs/internal/TECHNICAL_REPORT.en.md) |
| **Decisions on incorporating community improvements** | [docs/COMMUNITY_IMPROVEMENTS.en.md](docs/COMMUNITY_IMPROVEMENTS.en.md) |
| Which mode loads/releases what, and when | [docs/RESIDENCY.en.md](docs/RESIDENCY.en.md) |
| **What to do next** (not yet started / unverified) | This README's "[Waiting on external events going forward](#waiting-on-external-events-going-forward-backlog-as-of-2026-08-06)" §3 |
| Regression baselines for when diffusers gets bumped | [docs/internal/regression_baselines.json](docs/internal/regression_baselines.json) |

**Current state**: diffusers is pinned to the merged version **f37ab93** (PR #14355 tracking
complete, all paths equivalent by identical-seed MD5). **The two final low-VRAM goals (real
16GB single / 8GB × 2) were reached on 2026-08-11**, broadening the pitch from just "speedup"
to "**runs on consumer-tier GPUs**." The recommended launch on the 48GB box is
`H3_LOWVRAM=1 H3_TE_PRUNE=1 H3_TE_DEVICE=cuda:1 H3_VIDEO_VAE_FP16=1 H3_KEEP_TRANSFORMER=1`
(t2i steady state 9.7s/image, t2va 5s = 44.2s). **When using ref2va, drop `H3_TE_DEVICE`**
(with only 20GB on the TE GPU, capacity is insufficient and it auto-rejects with 400). The
chronological work log is in this README's dated sections (read on below).

> **About the measurement environment**: this home is shared across two machines. The **96GB
> box** = RTX PRO 6000 Blackwell 96GB + an added RTX 4060 Ti 16GB (the low-VRAM goals were
> measured here); the **48GB box** = RTX PRO 5000 Blackwell 48GB + RTX 4000 SFF Ada 20GB (the
> recommended 48GB-tier config and turbo were measured here). Each dated section states which
> box its figures come from. A single 48GB card cannot physically load the default mode (bf16
> transformer, 66.3GB), so `H3_LOWVRAM=1` (48GB tier) or `H3_LOWVRAM=group` (24-32GB tier) is
> required.

## Structure

```
minimax-h3/
├── app.py               # FastAPI main body (port 8611)
├── core/
│   └── runner.py         # Core ModularPipeline load/generate logic
├── static/
│   └── index.html        # Single-page UI (Japanese)
├── scripts/
│   ├── download_t2va.py  # Script that downloads only the subfolders needed for T2VA verification
│   └── probe_t2va.py     # Regression script to verify operation before touching the UI
├── outputs/               # Generated outputs (.gitignore target)
├── logs/                  # Download monitoring logs etc. (.gitignore target)
└── venv/                  # Dedicated venv (.gitignore target, see below)
```

## Installation

### Requirements

| | Requirement |
|---|---|
| GPU | Minimum: a **single 16GB card** (all features, measured, slow) or **8GB×2** (up to t2i/t2va/ref2i). **48GB tier** recommended for comfortable use. Per-config measurements: "[VRAM × feature matrix](#vram--feature-matrix-measured-2026-08-11)" at the top |
| Host RAM | 64GB or more recommended (`H3_LOWVRAM=group` keeps ~34GB of int8 weights resident in RAM, so 48GB+ free is needed) |
| Disk | About 145GB (T2VA/FL2VA only) / about 207GB (if Ref2VA is also used) |
| CUDA | 12.8 series (to match torch 2.9.0+cu128). If building SageAttention, `nvcc` must also be the same series |
| Python | 3.12 |
| Other | `ffmpeg` (used for gallery concatenation and info retrieval; generation itself works without it) |

### Steps

```bash
git clone https://github.com/animede/Diffusers_minimax-h3.git
cd Diffusers_minimax-h3
python3.12 -m venv venv

# PyTorch (cu128)
venv/bin/pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128

# Install diffusers "pinned to a verified commit" (be sure to read the note below).
# The merged version f37ab93 (the final form of PR #14355) is assumed. All paths
# (t2i/t2va/batch/ref2va/ref-batch) have been regression-tested against the old pin
# abc5e9b with identical-seed MD5 (see the "Migration stage 1/stage 2" sections at the end).
venv/bin/pip install "git+https://github.com/huggingface/diffusers.git@f37ab93e621d5ce206c9662e8291ca8b67d9c555"

# transformers 5.14.1 or later is required
#   (5.1.0 lacks Qwen3VLProcessor.create_mm_token_type_ids, so the PR #14355 encoder won't run)
venv/bin/pip install "transformers==5.14.1" accelerate==1.12.0 safetensors huggingface_hub

# Video/audio muxing and the Web API
venv/bin/pip install av==16.0.1 fastapi==0.104.1 "uvicorn==0.24.0" python-multipart pillow numpy

# 4bit quantization of text_encoder (required for the default H3_TE_QUANT=bnb-4bit)
venv/bin/pip install bitsandbytes==0.49.0

# Only if using int8 quantization for the transformer (H3_TRANSFORMER_QUANT=int8 / H3_LOWVRAM)
#   0.18+ requires torch>=2.11, so pin to 0.17.0
venv/bin/pip install torchao==0.17.0
```

> **Important: keep diffusers pinned to the commit.**
> PR #14355 was **merged on 2026-08-05**, and this app has **completed migration to the
> merged final form f37ab93** (stage 1: t2i/t2va/batch, stage 2: ref2va family). All paths
> have been regression-tested against the old-pin abc5e9b baseline with **identical-seed
> MD5 exact match** (see the "Migration stage 1/stage 2" sections at the end).
> If you upgrade diffusers further from here, follow the same procedure (identical-seed
> MD5 regression).

### SageAttention (optional, enabled by default)

Using the default `H3_ATTN_BACKEND=sage` requires a build targeting sm_120 (no prebuilt
wheel exists for Linux). **If you don't build it, start with `H3_ATTN_BACKEND=default`
specified** (denoise is only about 12% slower; no functional impact).

```bash
CUDA_HOME=/usr/local/cuda-12.8 scripts/build_sageattention.sh
```

> The build **must always limit its parallelism** (the script runs with
> `MAX_JOBS=4 NVCC_THREADS=2` plus a systemd-run memory cap). Unrestricted parallel nvcc
> will exhaust host RAM and OOM the entire system.

### Starting

```bash
venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
```

Open `http://<host>:8611/` in a browser. The first startup takes a few minutes because it
loads the model.

## Obtaining the model

`MiniMaxAI/MiniMax-H3` is 498.6GB (2 checkpoints, FL2VA/Ref2VA + both transformers), but
T2VA/FL2VA verification only needs `transformer/` (for FL2VA, 66.3GB) +
`text_encoder/` (66.7GB) + `vae/` (10.4GB) + `audio_vae/` (0.6GB) + config files, **about
144GB in total**. `transformer_ref/` (for Ref2VA, 66.3GB) and the separate `Ref2VA/` /
`FL2VA/` packages (144GB each) are not needed this time, so **absolutely do not
`snapshot_download` the whole thing**.

```bash
venv/bin/python scripts/download_t2va.py
```

Internally it uses `allow_patterns` to fetch only the necessary subfolders. During the
download, cache size can be monitored via `logs/du_monitor.log` (warns above 170GB).

## VRAM support table and main environment variables (quick reference)

Launch examples per use case. The top five rows are ballast measurements on the 96GB box
(PRO 6000, 32B TE); the bottom three rows are **real-card measurements using the projected
TE** (2026-08-11; see the "[VRAM × feature matrix](#vram--feature-matrix-measured-2026-08-11)"
at the top and the dated 2026-08-11 sections for details).

| GPU | Startup flag | t2va measured (768², 5s) |
|---|---|---|
| 96GB | (none = default) | peak 92GB / about 160s |
| 80GB tier | `H3_TRANSFORMER_QUANT=int8` | peak 59.7GB / about 160s |
| 48GB tier | `H3_LOWVRAM=1` | peak 38.9GB / about 215s |
| 32GB tier | `H3_LOWVRAM=group` | peak 28.7GB / about 280s |
| 18GB tier | `H3_LOWVRAM=group H3_TE_PRUNE=1` | peak 17.7GB / about 280-320s |
| **Single 16GB** (real 4060 Ti) | `H3_LOWVRAM=group H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_VIDEO_VAE_FP16=1` | peak 11.4GB / about 25min* |
| **Single 12GB** (projected, unmeasured) | same as single 16GB | the measured 11.4GB peak should fit in 12GiB; reference modes and 768×1344 will not |
| **8GB×2** (ballast-measured) | single-16GB flags + `H3_TE_DEVICE=cuda:1` | peak 7.23GB / about 25.6min* |

\* Times in the low-VRAM real-card rows were measured on a PCIe Gen3 x4 slot with an sm_89
card (SDPA). On Gen4 x16 the weight transfer is about 1/8 (see
"[An honest note on speed](#an-honest-note-on-speed)"). Cards other than sm_120 also need
`H3_ATTN_BACKEND=default`.

```bash
# Example: launch on a 32GB-tier GPU
H3_LOWVRAM=group H3_TE_PRUNE=1 venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
# Example: launch on a single 16GB card (add H3_ATTN_BACKEND=default on sm_89)
H3_LOWVRAM=group H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_VIDEO_VAE_FP16=1 \
  venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
```

Main environment variables (defaults shown; most **can also be toggled from the UI**, so
only set these if you want to change them permanently):

| Variable | Default | Meaning |
|---|---|---|
| `H3_TE_QUANT` | `bnb-4bit` | text_encoder quantization (`none` = bf16, 66.7GB) |
| `H3_TE_PRUNE` | `0` | Remove unused upper layers of TE (output unchanged, -3.6GB) |
| `H3_TE_PROJ` | (off) | **Projected TE**: replaces the 32B TE with Qwen3-VL-4B + a linear map (repo id or .safetensors path; quality is an approximation, see the dated 2026-08-10 section) |
| `H3_TE_PROJ_QUANT` | `bnb-4bit` | Quantization of the projected 4B TE (NF4 = 3.11GB; `none` = bf16 8.88GB / `bnb-8bit`) |
| `H3_TE_DEVICE` | (off) | Park the TE on a second GPU (e.g. `cuda:1`; the 32B TE needs a 20GB-class card, the projected TE fits an 8GB-class one) |
| `H3_TRANSFORMER_QUANT` | `none` | `int8` shrinks the transformer from 66.3GB to 34GB |
| `H3_LOWVRAM` | `0` | `1` = 48GB-tier phase cycling / `group` = block offload for the 32GB tier and below |
| `H3_KEEP_TRANSFORMER` | `0` | Keep the transformer resident, removing the reload fixed cost (see its section for the preconditions) |
| `H3_CACHE` / `H3_CACHE_THRESHOLD` | `fbc` / `0.05` | FirstBlockCache (denoise -25%; measured ineffective on the int8+SDPA trajectory) |
| `H3_ATTN_BACKEND` | `sage` | The sage build is sm_120-only. **Any other GPU must use `default` (SDPA)** |
| `H3_TURBO_LORA` | `0` | 4/8-step distilled LoRA (downloads about 780MB on first use; incompatible with `group`) |
| `H3_VIDEO_VAE_FP16` | `0` | fp16-ify the video VAE (reduces decode peak; required at 16GB and below) |
| `H3_LLM_URL` | `http://127.0.0.1:64650` | Local LLM used for prompt enhancement (optional feature) |

## VRAM/RAM design (important, based on measurements)

This machine has 96GB VRAM against 94GB RAM. **text_encoder is distributed natively in
bf16 and measures 66.73GB** (the initial estimate of "half, 33GB, after bf16-ification"
assuming an fp32 distribution was wrong). Together with transformer bf16 66.3GB and
vae+audio_vae fp32 totaling 11GB, that comes to about 144GB, which **does not fit in VRAM
or RAM simultaneously**. diffusers'
`ComponentsManager.enable_auto_cpu_offload()` (keeps all components resident in RAM at
all times and pushes only the one active component to the GPU) is not viable with 94GB
RAM, so it was not adopted.

The TE loading method is selected via the `H3_TE_QUANT` environment variable (default
`bnb-4bit`, made default after A/B testing on 2026-08-04).

### `H3_TE_QUANT=bnb-4bit` (default)

Quantizes text_encoder to NF4 (bitsandbytes, compute_dtype=bf16) at startup time and
**keeps it resident on the GPU** (bnb 4-bit models cannot move between devices, so
residency is the only option). Measured size **21.0GB** (larger than the initial estimate
of ~17-18GB). transformer (66.3GB) also stays resident, eliminating the per-request
TE<->transformer swap.

- Steady-state residency: transformer + TE-nf4 = **~87.5GB**. Adding the 11GB VAE would
  bring this to ~98.5GB, exceeding 96GB, so **the VAE pair is not resident in this mode**
  (it stays on the CPU and round-trips to the GPU only during the keyframe
  encode/decode phase that needs it. Only the fp32 11GB PCIe round trip, no disk I/O).
- Only the decode window (~9s) is executed after freeing the transformer, then reloading
  it immediately afterward (because transformer+TE+VAE+decode buffers physically don't
  fit — confirmed OOM on real hardware). This is a single one-way trip x2 rather than a
  per-step swap, so it does not fall under prohibited pattern #33 of the
  diffusers-server CLAUDE.md.
- **Quality A/B (identical seed 12345)**: frame comparison shows equivalent composition,
  subject, and sharpness (only minor differences such as tree placement due to changes in
  conditioning values). Audio also has comparable levels (rms 0.0080->0.0061), with no
  -20dB-style collapse. **Judged as no degradation, so made the default.**

### `H3_TE_QUANT=none` (bf16 TE, legacy method)

**Swaps two 66GB models on the GPU per request**:

- `vae` + `audio_vae` (fp32, 11GB total) stays GPU-resident at all times.
- Encode phase: [VAE 11GB + text_encoder 66GB] (if transformer is resident, it is freed
  first)
- Denoise/decode phase: [VAE 11GB + transformer 66GB] (TE is freed immediately after
  encoding)
- Freeing is done by **directly dropping the CUDA model reference** (not evacuating to
  RAM via `.to("cpu")` — routing 66GB through RAM was measured to trigger swap in
  practice). Reload takes 11-40s/model from page cache/disk. Steady state between
  requests: transformer+VAE resident (77.5GB).

**Measured overhead**: per request, TE load ~37s + transformer reload ~26s.
It was this swap cost that `bnb-4bit` (default, above) eliminated, shortening the total
per-request time from 245s to **185s** (the 157s denoise is unchanged since it is
model-bound and common to both).

**Two implementation pitfalls (hit and fixed on real hardware)**:
1. `MiniMaxH3TextEncoderStep.encode_prompt` is a bare staticmethod, and `@torch.no_grad()`
   is only attached to the block's `__call__` side. When calling it directly, you must wrap
   it in `torch.no_grad()`. Forgetting this pins about 50GB of TE weights on the GPU via the
   autograd graph, and VRAM never comes back even after freeing the model (same pattern as
   diffusers-server CLAUDE.md #39).
2. Block outputs (`num_frames`, `keyframes`, latent shape, etc.) go into `PipelineState`.
   `get_block_state()` only maps declared inputs, so read outputs via `state.get(name)`.

Because video VAE decode internally uses `torch.autocast(dtype=torch.float16)` in
diffusers' `MiniMaxH3VideoDecodeStep`, the weights themselves can remain fp32.
**Never cast the audio VAE at all** (there is a known issue where bf16-ifying it reduces
the generated audio volume by about 20dB, so `runner.py` explicitly passes
`dtype=torch.float32` when loading `vae`/`audio_vae`).

## Measured values (RTX PRO 6000 Blackwell 96GB, 768x768, 124 frames = 5.17s, 30 steps)

| Item | Measured |
|---|---|
| Download size (T2VA-required portion only) | 135GiB (measured HF cache) |
| text_encoder load | 37.6s (cold) / 15.9s (page-cache warm) |
| transformer load | 37.7s (cold) / 10-26s (warm) |
| vae+audio_vae load | 10.0s |
| Prompt encoding | 0.7s |
| Denoise (30 steps) | 157-159s (about 5.4s/step, GPU 100%/600W) |
| VAE decode (video+audio) | 6.5-9s |
| Peak VRAM (during generation) | none: 83.4GB (at decode; 70.4GB during denoise) / bnb-4bit: 91.7GB |
| Request total (via server API, including load) | none: 245s / **bnb-4bit (default): 185s** |
| RAM | Stable at ~6.5GB usage, zero swap increase |

The bnb-4bit peak of 91.7GB fits within a 96GB card, but with only about 4GB of headroom.
If you want to prioritize headroom, you can revert to the legacy method with
`H3_TE_QUANT=none` (peak 83.4GB, +60s/request).

## Denoise speedup via FirstBlockCache (`H3_CACHE`, default `fbc`)

Enables diffusers' official step-to-step cache (FirstBlockCache), equivalent to the
ComfyUI community's EasyCache speedup, via `H3_CACHE=fbc` (default). It skips the
remaining computation when the residual change of the first transformer block between
steps is small.

- `H3_CACHE_THRESHOLD` (default 0.05): measured to reduce denoise from 157s to 118s
  (-25%, 7 of 30 steps skipped); output is nearly identical to the non-cached version
  (PSNR 31.8-34.3dB, audio correlation 0.979, hard to distinguish visually).
- threshold 0.1 gives 1.92x (denoise 81.5s, 14 skips) but causes visually noticeable
  compositional drift, so it is not the default (only for when speed is the top priority).
- `H3_CACHE=none` fully restores the traditional no-cache behavior (byte-match regression
  confirmed).
- Peak VRAM increases by +0.7GB for the residual cache (91.4->92.1GB).
- Implementation note: `MiniMaxH3TransformerBlock` is not registered in the PR branch's
  `TransformerBlockRegistry`, so the runner side registers `TransformerBlockMetadata`
  before calling `enable_cache()` (the venv's diffusers itself is unmodified). Each
  request is wrapped with `_reset_stateful_cache()` + `cache_context("h3")` (failing to
  reset causes mis-skips based on the previous request's residuals). Verified reset
  correctness via byte-identical mp4s across two consecutive same-seed runs.

## 2x upscale via two-pass generation (`upscale=1` on `/api/t2va`, default OFF)

A hires-fix in the same family as the ComfyUI community's MiniMaxH3_LatentUpscaler.
Denoise the first half at low resolution (768²) -> spatially upscale only the **x0
estimate** video latent 2x via bilinear -> re-inject fresh noise via
`scheduler.scale_noise()` -> finish the remaining low-sigma steps at 1536² -> decode.
`H3_HIRES_DENOISE` (default 0.35) is the denoise strength assigned to pass 2. The UI
exposes this as a checkbox in the T2VA tab.

Measured (768² -> 1536², 5s, 30 steps, seed=12345, fbc+bnb-4bit):

| | Total | Denoise | Decode | Peak VRAM | Output |
|---|---|---|---|---|---|
| upscale=0 | 181s | 125s | 6.5s | 92.1GB | 768² |
| upscale=1 | 645s | 533s (pass1 78s + pass2 455s) | 24.7s | 88.0GB | 1536² |

- Composition and subject match upscale=0, with actual detail (fur, grass, etc.) added on
  top. Background details (fences etc.) drift slightly due to the re-denoise in pass 2
  (an intrinsic property of hires-fix).
- Audio: the latent tensor itself is unchanged by the upscale processing, but since video
  and audio share self-attention within a single packed sequence, the audio output from
  pass 2 onward does not bit-match upscale=0 (correlation 0.89, non-silent, equivalent
  quality — this is a spec-level architectural constraint).
- VRAM: since pass 2 has roughly 4x the sequence length, requests with upscale=1 free
  TE-nf4 immediately after encoding, before denoise (lazily reloaded on the next request's
  encode).
- **Key implementation point (confirmed by hitting a bug on real hardware)**: the
  interpolation target must be **the x0 estimate, not the noisy latent** (interpolating
  the noisy latent amplifies checkerboard-like noise into full-frame noise; the ComfyUI
  reference implementation also uses denoised_output). When changing resolution,
  position_ids/token_tags/each index is rebuilt via `build_packed_sequence()`, and
  `row_timestep_plan` is also rebuilt for the remaining steps. Because ModularPipeline's
  `_execution_device` is determined by the first module in component registration order,
  after freeing TE you must explicitly use `components.transformer.device`
  (same pattern of pitfall as diffusers-server CLAUDE.md #23 and #47).
## Sage Attention (`H3_ATTN_BACKEND`, default `sage`)

Uses SageAttention 2.2.0, source-built for sm_120 (Blackwell), by default (build via
`scripts/build_sageattention.sh`, about 2 minutes. **Always run it with
`MAX_JOBS=4 NVCC_THREADS=2` plus a systemd-run memory cap** — unrestricted parallel nvcc
has a history of exhausting host RAM and taking down the whole system. Requires explicit
`CUDA_HOME=/usr/local/cuda-12.8`, since the default cuda-13.0 does not match torch's
cu128). No PyPI/community Linux sm_120 wheel existed (all were Windows-only).

- Measured: denoise 118s->104s (**-12%**). Fully deterministic (byte-identical across
  two same-seed runs). Quality is visually equivalent (the PSNR of 21dB is trajectory
  drift from the int8-QK approximation, not degradation)
- `H3_ATTN_BACKEND=default` reverts to the traditional SDPA
- Can be combined independently with FBC: sage + `H3_CACHE_THRESHOLD=0.1` gives denoise
  67s (-43%, request ~125s; FBC 0.1's compositional drift characteristic is as already
  known)
- The hub backends (`flash_hub`/`sage_hub`) don't work because no torch 2.9-targeted build
  exists on the Hub side (as of 2026-08-05, not an environment issue)

## Step-count guidance (distilled model, measured 2026-08-05)

`num_inference_steps` is an API/UI parameter. Relative to 30 (the verification default),
**20 gives -15%, 16 gives -31%** in denoise time. Both 16 and 20 show no breakdown in
single-frame quality or temporal stability (though composition does change with step
count). 16-20 is a reasonable guideline for drafts, 30 for production. Reducing steps also
reduces FBC skip opportunities (7 skips at 30 steps -> 0 at 16 steps), so the effect is not
simply proportional.

## transformer int8 quantization (`H3_TRANSFORMER_QUANT`, default `none`)

With `H3_TRANSFORMER_QUANT=int8`, the transformer / transformer_ref is int8-ified via
torchao (`Int8WeightOnlyConfig(version=2)`, the recipe from the PR #14355 docs, torchao
0.17.0). 66.3GB -> 34.0GB. Quality is visually equivalent (the PSNR of 19dB is trajectory
divergence, not degradation; int8-vs-int8 is fully deterministic, giving byte-identical
mp4 for the same seed), and denoise is about +5s (dequant cost).

**With int8, both transformers stay resident simultaneously** (34+34+TE-nf4 21 = ~89GB),
eliminating the ~66GB-class reload on ref2va<->t2va variant switching (only the first
ref2va incurs a ~36s cold load). Measured:

| | bf16 (default) | int8 both resident |
|---|---|---|
| t2va | 175-185s / peak 92.1GB | 177-196s / peak 59.7-91.1GB |
| ref2va (2nd request onward) | 523s / peak 87.6GB | **463-471s / peak 74.5GB** |
| Variant-switch reload | ~26-40s every time | **none** |

Phase control (with int8 both resident): before t2va denoise, TE is force-freed only when
transformer_ref is resident (because it would OOM at 89GB steady + activations — confirmed
on real hardware); ref2va no longer needs the TE force-free, and the decode window also
passes through while transformer_ref remains resident. The runner sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (because "only 54GB used but a 15GB
allocation failed" was reproduced on real hardware from fragmentation during the int8
load/free cycle — a setting with a track record in diffusers-server as well). The default
`none` remains byte-identical to before (regression confirmed).

## 48GB-tier VRAM support (`H3_LOWVRAM`, default `0`)

TE-nf4 (21GB) + transformer int8 (34GB) = 55GB cannot be co-resident on a 48GB-tier card
(neither the 96GB machine's default mode nor its int8-both-resident mode is viable).
`H3_LOWVRAM=1` supports the 48GB tier via a phase-cycling scheme that "never lets TE and
transformer be resident at the same time."

- **Forced**: if `H3_TRANSFORMER_QUANT` is unset, it is automatically overridden to
  `int8` (since bf16 66.3GB doesn't fit in 48GB even alone). If `H3_TRANSFORMER_QUANT=none`
  is explicitly specified, startup is rejected with a `RuntimeError`. Specifying
  `H3_TE_QUANT` as anything other than `bnb-4bit` (the default) is likewise rejected at
  startup. `H3_TRANSFORMER_BOTH_RESIDENT` (int8-both-resident) is unconditionally disabled.
- **Steady state**: nothing large stays resident between requests (only the VAE pair is
  CPU-resident; unlike other modes, there is no permanent transformer/TE squatting).
- **t2va/fl2va phases**: [entry: free any resident transformer/transformer_ref] ->
  load TE -> encode (+ keyframe encode for fl2va) -> **run layout/latents/timesteps
  first while TE is still resident** (workaround for the `_execution_device` resolution
  pitfall, see below) -> free TE -> load transformer (int8) -> denoise (~34GB +
  activations ~5GB ~= 39GB) -> free transformer -> VAE->GPU -> decode (~11GB + buffer)
  -> VAE->CPU (**the transformer is not reloaded for the next request** — the next
  request needs encode first).
- **ref2va phase**: following the same principle, reference VAE encode -> layout/latents/
  timesteps (again while TE stays resident) -> free TE -> load transformer_ref (int8) ->
  denoise -> and so on.
- **`_execution_device` resolution pitfall (discovered and fixed during implementation)**:
  the pipeline's component order is `text_encoder, tokenizer, processor, vae, scheduler,
  audio_scheduler, transformer, ...`. With a naive implementation that frees TE and then
  loads transformer, `vae` (an nn.Module that "exists" even when evacuated to CPU) gets
  resolved right after `text_encoder`, so `_execution_device` resolves to `cpu`, and the
  first transformer forward in denoise fails with
  `RuntimeError: Expected all tensors to be on the same device` (reproduced and
  root-caused on real hardware). As a countermeasure, **layout/latents/timesteps are run
  while TE is still GPU-resident, and TE is freed only after those output tensors have
  settled on the correct device** (the same idea as the `force_free_te` deferral pattern
  used in other modes). In ref2va, `reference_encoder_step` must likewise run while TE is
  still resident (because `layout_step` depends on the reference's latent shape).
- **upscale=1 (hires-fix) unsupported**: pass 2 requires roughly 4x the sequence length
  (~16x the attention activation cost), which is unverified within this mode's steady-state
  headroom (~9GB), so it is rejected with a `ValueError` (400).
- **Fix for a pre-existing bug found as a side effect**: `_sync_shared_components_to_ref()`
  inside `generate_ref2va()` was called **before** TE was loaded, so with TE unloaded
  (H3_LOWVRAM, or a case where ref2va is the first request in a `none`-family mode),
  `None` would be synced into `self._pipe_ref.text_encoder`, causing
  `AttributeError: 'NoneType' object has no attribute 'config'` (reproduced and
  root-caused on real hardware). Fixed the ordering to call it **after** TE loads (a fix
  common to all modes, not specific to H3_LOWVRAM).
- **Correctness verification**: since phase cycling shouldn't change the computation
  itself, the t2va output of `H3_LOWVRAM=1 H3_TRANSFORMER_QUANT=int8` and the t2va
  output of normal int8 mode were compared with the same seed (768², 5s, 30 steps, fox
  prompt, seed=12345), confirming **byte-identical mp4** (md5 match).
- **48GB-equivalent real-hardware verification** (constraining free VRAM to ~43.5GB via
  dummy VRAM allocation on the 96GB machine — a stricter condition than the ~47GB free
  typical of an actual 48GB card):

  | | Completed | Peak VRAM | Breakdown (approx.) |
  |---|---|---|---|
  | t2va #1 | Yes | 38.68GB | TE load ~52s + transformer load ~36s + denoise 108s + decode 6.6s |
  | t2va #2 (consecutive) | Yes | 38.94GB | Same fixed cost recurs every time (as expected from a design with no steady state) |
  | ref2va (1 image reference) | Yes | 43.84GB | Denoise 283s (somewhat higher since the reference rows lengthen the sequence and don't fit in the ~39GB range) |
  | upscale=1 | - | - | Judged an OOM risk at implementation time and explicitly rejected with 400 (not run) |

  None of these showed increased host RAM swap (`free -g` showed Swap used stable at
  ~6GB before and after the work). After the work, restarted with `H3_LOWVRAM` unset
  (fully default settings, bf16 transformer) and ran one t2va with the same prompt,
  confirming values like `peak_vram_gb: 91.94GB` / `cache_skipped_steps: 6` were at the
  same level as existing measurements (the table at the top of this README, and the int8
  quantization section), and judged there was no regression.
- ~~**Pre-saving quantized checkpoints to shorten load time**: not investigated~~
  -> **implemented for TE on 2026-08-08** (`H3_TE_PREQUANT`, default ON. TE load
  53.0s->29.5s, request total -35%, equivalence confirmed via MD5 match. See the
  "Disk cache of the quantized text_encoder" section above). Serializing the transformer
  int8 **remains unverified** — it would need about 34GB to save, which is impractical
  given this box's disk headroom (43GB free).

## Rounding of arbitrary size/duration (2026-08-06)

Added **"Custom (rounds to a multiple of 32)"** to the UI resolution select, allowing
free-form width/height input. Input values are **rounded to H3's rules rather than erroring**:

- **Canvas**: rounds to the nearest multiple of 32, clamped to 256-2048
  (`round_canvas_value` in `app.py`). H3's block spec raises a `ValueError` unless the
  value is a multiple of 32 (`MINIMAX_H3_CANVAS_MULTIPLE`), so sending a fractional value
  used to return 400. Values exceeding the native range (short side 768, max 768x1344)
  trigger a UI warning (VRAM/quality unverified for these)
- **Seconds -> frame count**: clamped to 5-15s, then rounded up to `17n+5`
  (`align_num_frames`). Example: 6.3s -> 158 frames (6.58s). The UI previews the actual
  frame count and duration before submission
- API: added `height`/`width` (optional; when specified, takes priority over the
  `resolution` preset) to `/api/t2va`/`/api/fl2va`. `/api/ref2va` already accepted these,
  but was changed to round them as well
- `/api/status`'s `constraints` exposes the rounding rules (canvas_multiple/min/max, fps,
  frame_step/offset), and the UI previews using the same rules (the server is the
  authority on rounding)

Measured confirmation: generating with `height=700 width=1000 seconds=6.3` -> response
`704x992 / 158 frames`; the output mp4 also matched 992x704, 158 frames via ffprobe.

**UI implementation pitfall**: attaching `min`/`max`/`step="32"` to the width/height
`<input type=number>` causes HTML5 input validation (stepMismatch/rangeOverflow) to
**treat the pre-rounding fractional input as invalid and block form submission entirely**
(only values that happen to be valid, like 1024x576, get through, while something like
1000x700 does nothing — a confusing symptom). Do not attach constraint attributes to
fields that are meant to be rounded.

## Turbo LoRA (`H3_TURBO_LORA`, default `0`, 2026-08-06 / **switched to the lightx2v variant on 2026-08-08**)

> **2026-08-08 update: switched the default LoRA to the lightx2v variant, making turbo
> usable even on 48GB (int8/low-VRAM).**
>
> - Default: `H3_TURBO_LORA_REPO=lightx2v/Minimax-h3-Turbo` /
>   `H3_TURBO_LORA_FILE=minimax_h3_fl2v_turbo_4step_v0.1.safetensors` (DMD distillation,
>   Apache 2.0, rank128, targets 312 Linear modules). Default step count is tied to the
>   format (lightx2v=4 / falls back to 8 if reverted to Ostris)
> - **Reason it works with int8**: the keys are diffusers-native (to_q/to_k/to_v
>   separated), so `fuse_projections()` is unneeded -> `torch.cat` is never called ->
>   avoids the `Int8Tensor` incompatibility. The apply function auto-detects the format
>   (`detect_turbo_lora_format`; checks the comfy signature `qkv_proj` first — order
>   matters since the Ostris variant also has `token_refiner.` keys)
> - **Applied scale factor** (`H3_TURBO_LORA_SCALE`, empty = measured per-format default):
>   for lightx2v it is **0.094** (Kijai's documented 0.75 assumes ComfyUI's alpha folding.
>   Multiplying the raw B*A by 0.75 directly produces complete noise even at 30 steps —
>   see the strength sweep in the spike section). Ostris remains 1.0 (scale=1.0 is
>   identity, so it bit-matches the old behavior)
> - **E2E measured** (RTX PRO 5000 48GB + `H3_LOWVRAM=1`, 768²): t2va 4 steps
>   **total 143s / denoise 29s** (non-turbo 30 steps is 351s). **t2i (still image) x
>   turbo is denoise 5.0s / total 94s**. The output of this implementation's path is a
>   **byte-identical mp4 md5 match** with the spike output. Two consecutive turbo runs
>   also correctly re-apply after lowvram's reload, and turning turbo=0 correctly returns
>   to the normal path (with FBC enabled)
> - **Known behavior**: with turbo, audio level tends to be higher than non-turbo (rms
>   0.018-0.042 vs 0.007; not white noise, but depending on intensity the peak can
>   approach 1.0)
> - **Combination restrictions**: cannot be combined with `H3_LOWVRAM=group` regardless of
>   format (`enable_group_offload`'s `cpu_param_dict` is fixed at enable time, and any LoRA
>   buffer added afterward risks being left out of the offload cycle — left disallowed
>   since this is unverified). comfy-format (Ostris) x int8 remains unsupported as before
>   (repo name is used for a preliminary check, with the final determination made from the
>   actual file's keys at apply time). Application to ref2va (transformer_ref) remains
>   unverified
> - Since this is v0.1 (published 2026-08-07), it remains **default OFF** (opt-in via the
>   UI/request turbo flag)

The following is a record from the era of the Ostris variant (comfy format), which remains
valid for the bf16 path:

Applying the 4/8-step distilled LoRA (Apache 2.0, rank64, targets 259 Linear modules)
being trained by Ostris via `H3_TURBO_LORA_REPO=larryvrh/MiniMax-H3-Turbo-Lora`, and
setting the default step count to 8. **Default OFF since this is a preview LoRA (still
in training)**. Will be re-evaluated once a finished version is released.

Measured (768²/5s/seed12345): 8 steps gives **87.7s (-46%)**, approaching the quality of
the 30-step baseline (163.5s); 16 steps at 98.4s matches the baseline; 4 steps at 39.6s is
softer but not broken.
**The community's "4-7 steps doesn't work" report is likely because ComfyUI's standard
sampler cannot handle the dual schedule (video shift12 / audio shift3)** — this
implementation (diffusers PR's scheduler/audio_scheduler separation + manual loop) showed
no audio corruption even at 4 steps. No change to the shift wiring was needed (12/3 are the
H3 baseline scheduler's defaults, and we confirmed the sigma grid bit-matches the author's
reference implementation).

Implementation note: LoRA keys use ComfyUI's fused-QKV naming, so it's applied via
`attn.fuse_projections()` + a runtime delta (W_eff=W+BA, not fused). **Pitfall:
`fuse_projections()` does not remove the old to_q/k/v, leaking +12.8GB** (addressed with
explicit deletes). Since AdaLN reads `linear.weight` directly, the wrapper needs
weight/bias etc. pass-through. FBC is automatically disabled while turbo is active.

### Verification of turbo x other features combined (2026-08-06)

Combinations with anything other than the default transformer path
(`transformer_quant=none`, `lowvram=0`) were initially "preventively rejected as
unverified," but have since been A/B tested on real hardware.

- **turbo x upscale (2x upscale hires-fix): verified working, unblocked.**
  768²->1536², seed12345, succeeded at both 8/16 steps. 8 steps: total 210.3s
  (denoise 82.6s + decode 24.4s), pass1=5/pass2=2 steps (`H3_HIRES_DENOISE=0.35`
  default split), peak VRAM 88.09GB — a large reduction from the non-turbo 30-step
  baseline of 645s. 16 steps: total 331.8s, pass1=10/pass2=5 steps, peak VRAM 88.35GB,
  clearly sharper than 8 steps. All frames (first/middle/last) were visually confirmed
  free of discoloration or checkerboard collapse, and audio RMS/peak were normal (no
  silence, no clipping). Since FBC is auto-disabled with turbo, hires-fix's FBC
  bookkeeping (`_fbc_last_step_was_skip()`) safely no-ops via try/except (confirming the
  concern of "unhandled FBC calls with turbo unsupported" causes no actual harm).
- **turbo x transformer int8 (`H3_TRANSFORMER_QUANT=int8`, including
  `transformer_both_resident`): confirmed unsupported by measurement, remains rejected.**
  `apply_turbo_lora()`'s `attn.fuse_projections()` executes
  `torch.cat([to_q.weight, to_k.weight, to_v.weight])`, but the int8-quantized
  `to_q`/`to_k`/`to_v` (`H3_INT8_MODULES_TO_NOT_CONVERT` does not skip these) are torchao
  `Int8Tensor`s, and since the `aten.cat` kernel is unimplemented, it reliably fails with
  `NotImplementedError: Int8Tensor dispatch: attempting to run unimplemented operator/
  function: func=<OpOverload(op='aten.cat', overload='default')>` (HTTP 500; fails
  cleanly per request with no VRAM leak — confirmed normal non-turbo generation
  immediately afterward).
- **turbo x lowvram=1: confirmed unsupported by measurement, remains rejected.**
  Since `lowvram=1` forces `transformer_quant=int8`, it fails with exactly the same
  `Int8Tensor`/`aten.cat` error as above (same error message confirmed on real hardware).
- **turbo x lowvram=group: confirmed unsupported by measurement, remains rejected.**
  Same reason (`lowvram=group` also implies `transformer_quant=int8`) yields the same
  error. **This failure is unrelated to the order in which group-offload hooks are
  applied** (the sibling project's fix pattern of "apply LoRA before
  `enable_group_offload()`" doesn't help here — `fuse_projections()` itself fails purely
  from `torch.cat`, without going through the group-offload hooks at all, so reordering
  wouldn't fix it and this was not pursued further).
- **ref2va: remains out of scope for this task's verification** (out of scope for the
  original task brief).

## 16GB-tier verification result: not supported (floor is ~18GB, confirmed 2026-08-06)

With a 16GB ballast (15.5GB free), **OOM occurs near the end of TE loading (nf4
quantization)** (fails to allocate +250MiB at 15.37GB used). The pruned TE-nf4's resident
17.45GB is itself the floor, and video VAE fp16 — a decode-stage countermeasure — cannot
move this floor. **Completes successfully with an 18GB ballast** (peak 17.72GB, total
302s) — i.e., the practical lower bound of the current architecture is **~18GB** (the
24GB-tier configuration `H3_LOWVRAM=group H3_TE_PRUNE=1` works as-is at the 18GB tier
too). Breaking through 16GB would require streaming execution of TE or sub-4-bit
quantization (a separate, not-yet-started task).

## fp16-ifying the video VAE (`H3_VIDEO_VAE_FP16`, default `0`)

With `H3_VIDEO_VAE_FP16=1`, only the video VAE weights are fp16-ified (9.70->4.85GB,
decode peak 16.29->~11.4GB). **Never touch the audio VAE** (known -20dB issue with
bf16-ification).
- Quality: mean PSNR across all 124 frames is **39.97dB** (min 39.08), visually
  indistinguishable. Since decode computation already used autocast fp16, the impact of
  fp16-ifying the weights themselves is small
- **Implementation pitfall**: `AutoencoderKLMiniMaxH3._keep_in_fp32_modules` forces the
  encoder/decoder etc. back to fp32, so `from_pretrained(dtype=fp16)` has no effect
  (confirmed on real hardware). You must explicitly call `.to(torch.float16)` after loading
- With default OFF, this MD5-matches the existing baseline (zero regression)

## 24-32GB-tier VRAM support (`H3_LOWVRAM=group`, added 2026-08-05)

`H3_LOWVRAM=1` (48GB tier) loads the full transformer (34GB) onto the GPU every request,
so it doesn't even fit alone on a 24-32GB-tier card. `H3_LOWVRAM=group` ports the
"block-level group offload" pattern established in diffusers-server (a sibling project)'s
CLAUDE.md #33/#34/#37 into this project, keeping the transformer **resident in host RAM**
while shuttling only the blocks needed for each denoise step (1-2 of 50 layers,
~0.68GB x 1-2) to and from the GPU. The transformer is **loaded exactly once at process
startup and stays resident across requests** (unlike `H3_LOWVRAM=1`'s per-request reload).

### Investigation results on the PR side's "load-time quantization during streamed offload"

At the time of this task, reading `TorchAoHfQuantizer`'s (in
`quantizers/torchao/torchao_quantizer.py`) `validate_environment()` showed that it only
sets `self.offload = True` when `device_map` is a **dict** (like accelerate's automatic
assignment) containing the **string value** `"cpu"`, and `check_if_quantized_param()`
skips quantization of CPU-placed parameters when this flag is set (i.e., by design,
parameters that get CPU-offloaded are intentionally left unquantized). This
implementation, however, uses `device_map={"transformer": "cpu"}`, which passes through
`load_components()` and ultimately reaches `from_pretrained()` as the **plain string**
`"cpu"`, which `modeling_utils.py`'s normalization code converts into a **single-key
dict** `{"": torch.device("cpu")}` (whose value is a `torch.device` object, not a string).
Since `torch.device("cpu") == "cpu"` evaluates to `False` in Python,
`"cpu" in device_map.values()` remains False, and `self.offload` is never set. In other
words, **loading onto the CPU does not skip quantization — it is correctly quantized as a
torchao Int8Tensor** (confirmed on real hardware via `scripts/probe_group_offload.py`
that 370/370 layers become Int8Tensor). Conclusion: **int8 quantization on the CPU works
without issue**.

### Implementation summary

- `_ensure_transformer_group()` (`core/runner.py`) quantization-loads onto the CPU with
  `device_map={"transformer": "cpu"}` + `TorchAoConfig(Int8WeightOnlyConfig)`, then calls
  `enable_group_offload(offload_type="block_level", num_blocks_per_group=1,
  use_stream=..., low_cpu_mem_usage=...)` (transformer_ref has an analogous
  `_ensure_transformer_ref_group()`).
- TE requires bnb-4bit just like `H3_LOWVRAM=1` (forced at startup). In t2va's steady
  state, TE stays resident across requests too (since the transformer is resident in the
  first place, it's better to keep TE resident as well and avoid per-request reload cost).

### [Major finding] `use_stream=True` + `low_cpu_mem_usage=True` combined has a bug that breaks torchao Int8Tensor

Actually running a forward pass in `scripts/probe_group_offload_forward.py` confirmed on
real hardware that it always fails at the first denoise block with
`RuntimeError: cannot pin 'torch.cuda.CharTensor' only dense CPU tensors can be pinned`.
`hooks/group_offloading.py`'s `_pinned_memory_tensors()` (called unconditionally every
step from `_onload_from_memory()` when `use_stream=True`) attempts `.pin_memory()`
regardless of `low_cpu_mem_usage`'s value, while `_init_cpu_param_dict()` (run once at
`enable_group_offload()` call time) skips pinning when `low_cpu_mem_usage=True` — an
asymmetric implementation where the two sides' assumptions don't line up. When this
mismatch occurs, torchao's `Int8Tensor.qdata` ends up in a broken state (internally
recognized as `torch.cuda.CharTensor`), and calling pin_memory() on it crashes. Results
of a controlled comparison in `scripts/probe_group_offload_fix.py`:

| Setting | Result | Onload/offload per block |
|---|---|---|
| `use_stream=True, low_cpu_mem_usage=True` (combined) | **crash** | - |

(Corrected 2026-08-10: this combination was originally described here as "the diffusers default", which was wrong — the `apply_group_offloading` API defaults are `use_stream=False, low_cpu_mem_usage=False`; the crash is hit when both are opted into for a memory-constrained setup.)
| `use_stream=False, low_cpu_mem_usage=True` | works | onload 0.1-0.26s / offload ~0.22s |
| `use_stream=True, low_cpu_mem_usage=False` | works | **onload 0.04-0.07s** / offload ~0s |

Adopted `low_cpu_mem_usage=False` (eagerly pins all parameters at
`enable_group_offload()` call time) as the new default (`H3_GROUP_OFFLOAD_LOW_CPU_MEM`,
default `0` = False). Reason: onload is 4-5x faster (pinned memory is not pageable, so
DMA transfer is faster). The cost is an extra ~14-16GB of host RAM pinned at load time
(page-locked, so it cannot be swapped), and `enable_group_offload()` itself taking about
22 seconds (measured on real hardware: 70s to load onto CPU + 22s to pin = ~90s total).
If you want to prioritize less RAM usage, explicitly set
`H3_GROUP_OFFLOAD_LOW_CPU_MEM=1` (in which case `H3_GROUP_OFFLOAD_USE_STREAM` also
automatically falls back to `0` unless explicitly specified, to avoid the broken
combination above).

### Final choreography (phase x resident items x peak)

| Phase | Large resident items | Notes |
|---|---|---|
| Startup preload | transformer (int8, CPU-resident + group-offload hooks) | About 90s (CPU load 70s + pinning 22s) |
| t2va encode | TE-nf4 (GPU, 21GB) + transformer (CPU) | |
| t2va denoise | TE-nf4 (GPU, 21GB) + 1-2 transformer blocks (GPU, ~1.4GB) | |
| t2va decode | VAE pair (GPU, ~11GB) + 1-2 transformer blocks | **TE is force-freed only for this window** (see below), reloaded after decode |
| ref2va reference encode | VAE pair (GPU, 11GB) | TE is force-freed only for this window (see below) |
| ref2va denoise | TE-nf4 (GPU, 21GB) + 1-2 transformer_ref blocks | |
| Steady state between requests | transformer (CPU) + TE-nf4 (GPU) | After ref2va, transformer_ref returns to unloaded (reloaded every t2va<->ref2va switch) |

### [Second bug discovered and fixed during implementation] TE needed to be force-freed for the decode window and reference-encode window

Initially designed on the assumption that "since group-offloaded transformer's actual
GPU footprint is tiny (~1.4GB), there's no need to free the transformer during decode,"
but a 32GB dummy-VRAM verification run reproduced `CUDA out of memory` on real hardware.
Enumerating actually-live CUDA tensors via `_log_gpu_tensor_diag()` (a temporary
diagnostic function enabled with `H3_DEBUG_MEM_DIAG=1`, left in `core/runner.py`) revealed
that TE-nf4's own embedding table / lm_head weights (shape `(151936, 5120)`, bf16,
1.556GB x 2 = a combined total of over 3.1GB out of 22.25GB total) had **remained
resident** right up until decode (not freed by `empty_cache()` alone, since they were
genuinely-referenced live tensors). In other words, the true requirement was TE-nf4
(21GB) + decode-only buffers (~16.3GB, see the VAE tiling investigation below) = 37GB —
a conflict between TE and VAE, unrelated to the transformer's footprint. Fix: **force-free
TE for the decode window (and ref2va's reference VAE encode window), reloading it after
exiting the window** (a dedicated piece of logic separate from `force_free_te`; the
`_execution_device` resolution order is secured by ensuring TE is freed before VAE goes
to the GPU).

### Result of the MD5-match check

Comparing the output of normal int8 mode (`H3_LOWVRAM` unset,
`H3_TRANSFORMER_QUANT=int8`, FBC `H3_CACHE=fbc` enabled) against the output of
`H3_LOWVRAM=group`, with the same seed (768², 5s, 30 steps, fox prompt, seed=12345),
**FBC's cache-skip decisions differed due to the difference in execution path**
(`cache_skipped_steps` was 6 vs 0), so a naive comparison found the mp4s mismatched.
Since FBC's decision is a numerically sensitive threshold on residual similarity to the
previous step, mathematically equivalent computations can still yield different skip
decisions depending on the path (not degradation). Re-comparing with both modes set to
`H3_CACHE=none`, confirmed **byte-identical mp4** (md5 match). This substantiates that
group offload's computation is mathematically identical to the existing path.

### Measurement tables: 32GB-limit and 24GB-limit probes

**32GB limit** (free VRAM constrained to ~30GB via dummy VRAM allocation, `H3_CACHE=fbc`
still enabled):

| | Completed | Peak VRAM | Time taken |
|---|---|---|---|
| t2va #1 (768², 5s, 30 steps) | Yes | **28.67GB** | denoise 220.79s / decode 6.31s / total 337.19s (including first-time TE load) |
| t2va #2 (consecutive) | Yes | **28.23GB** | denoise 220.85s / decode 6.01s / total 278.83s (shorter since TE stays resident) |
| ref2va (1 image reference, 768x1344) | **Rejected for insufficient RAM** (see below) | - | - |

Both runs produced an mp4 identical to run #1 (md5 match, `be3f32a84de074990208ad0d30f31a63`).
No increase in host RAM/swap in any phase (`free -g`'s Swap used remained stable at
~7-8GB before and after the work — this figure comes from other processes).

**24GB-limit probe** (free VRAM constrained to ~22GB via dummy VRAM allocation):

- 768²: OOM during denoise (onload of transformer blocks). Actual consumption 21.85GB, of
  which TE-nf4 alone accounts for 21GB. **TE-nf4's own fixed size (21GB) consumes the
  majority of the 22GB budget, leaving no room even for the additional onload of a single
  transformer block (~148MB).**
- 544x960 (temporarily added to RESOLUTION_PRESETS for the test, removed afterward):
  **OOM at the same point, same 21.85GB.** Lowering the resolution changed neither the
  failure point nor the consumption, **confirming, just as with the VAE tile shrink test,
  that a resolution-independent fixed cost is the bottleneck** (see the VAE tiling
  investigation below).
- **Conclusion**: with the current architecture (a design where TE is bnb-4bit and stays
  resident almost all the time), TE-nf4 alone consumes 21GB, the majority of a 24GB-tier
  card's effective budget (~22GB), and no resolution makes it viable. Supporting the
  24GB tier would require either freeing TE during denoise (reverting to an
  `H3_LOWVRAM=1`-like design) or further lightening TE itself (e.g. GGUF, though applying
  GGUF to transformers-family models is structurally difficult). Since this is outside the
  scope of this task (the goal was 48GB->24-32GB support, with 24GB an exploratory probe),
  it is concluded unsupported for now.

### VAE tiling investigation results

Benchmarking a 768², 124-frame VAE decode standalone (from a synthetic latent, bypassing
denoise) directly via `scripts/probe_vae_tile_size.py` showed that **shrinking
tile_sample_min_height/width from 256 (default) -> 192 -> 128 -> 96 made no difference
whatsoever to peak VRAM (16.29GB)** (time taken worsened from 5.9s to 10.5s as tile count
increased). This result suggests decode's peak is governed not by the spatial-tile
compositing buffer, but by **a buffer that decodes one whole chunk's worth of temporal
chunking (`tokens_chunk_size` units) at a time**, or by a fixed VAE-architecture overhead
(the VAE class doesn't expose the temporal chunk size as a public parameter, so further
tuning would require code changes and is out of this task's scope).
**Conclusion: adjusting spatial tile size is meaningless for 24-32GB-tier support**
(keep the default).

### Result of confirming FBC/sage coexistence

Throughout all ballast verification (both 32GB and 24GB), running with `H3_CACHE=fbc`
(default) and `H3_ATTN_BACKEND=sage` (default) enabled produced no conflicting errors with
the group-offload hooks (FBC makes block-level skip decisions, group offload manages
block GPU residency, and sage handles attention implementation inside each block — three
independent layers). Separate from the MD5-match check (done with `H3_CACHE=none`), a
full run with the default FBC-enabled settings (2 t2va runs under the 32GB limit) also
completed successfully.

### RAM constraint for Ref2VA (a known, unresolved limitation)

Confirmed on real hardware that calling ref2va after running t2va with `H3_LOWVRAM=group`
gets rejected by the host RAM guard regardless of VRAM budget (reproducible even on the
96GB machine with no dummy VRAM allocation applied):

```
H3_LOWVRAM=group requires at least 40.0GB of available host RAM before loading
the (~34GB, permanently CPU-resident) int8 transformer, but only 33.0GB is
available right now.
```

Root-cause narrowing: `avail_gb` recovers correctly (around 44.6GB) right after freeing
the transformer, but drops to around 33GB during the subsequent TE reload -> reference VAE
encode -> layout computation process (confirmed in real-hardware logs). Since
`swap_used_gb` consistently shows no increase, no actual swapping occurred — it is more
likely that `MemAvailable` (Linux's conservative buff/cache-inclusive heuristic estimate)
fluctuates more conservatively than the true free RAM. That said, even looking at `used`
from `free -g` — 62GB of 94GB used (32GB left) — securing an additional 34GB is
inherently tight, so we could not conclude the guard itself is wrong (even on the 96GB
machine, the design of "keeping the t2va transformer resident while also trying to pin the
ref2va transformer_ref onto the CPU" is already tight against the physical capacity of a
94GB-RAM machine). **We chose to err on the safe side and record this as a known
limitation rather than loosening the guard** (prioritizing the lesson learned from a past
swap-runaway incident). Future improvement candidates: stop eagerly loading the
transformer in `preload_all()` and lazily load it like TE (a latency tradeoff for the
first request), or design a path that consumes less RAM when switching between t2va and
ref2va. **On a machine with 48GB+ RAM, this problem likely does not occur (unverified, but
there should be more RAM budget headroom)** (this task only verified on a 94GB-RAM
machine; retesting on a machine with more RAM has not been done).

## Removing unused upper layers of text_encoder (`H3_TE_PRUNE`, default `0`, added 2026-08-06)

MiniMax-H3's text_encoder (Qwen3-VL-32B, 64 layers) only reads
`hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER]` (=50)
(`diffusers/modular_pipelines/minimax_h3/encoders.py`). `H3_TE_PRUNE=1` builds the
text_encoder with **only 51 layers** (0-50, `MINIMAX_H3_TEXT_ENCODER_LAYER + 1`), and
never loads the unused layers 52-64 + final `norm` + `lm_head` (equivalent to 14 layers'
worth of weight; measured ~3.6GB with bnb-4bit / ~13.6GB with bf16). The default `0` leaves
everything completely unchanged (this branch is never invoked).

### Why "51 layers," not exactly 50 (a transformers-side pitfall discovered and verified in this task)

The meaning of `hidden_states[k]` is determined by transformers' `_can_record_outputs =
{"hidden_states": Qwen3VLTextDecoderLayer}` hook mechanism (`output_capturing.py`):
`hidden_states[0]` = the embedding output (captures the input to layer 0),
`hidden_states[k]` (k=1..num_hidden_layers) = the output of `layers[k-1]`. In other words,
`hidden_states[50]` = the output of `layers[49]`, so it should have been sufficient to
just run `layers[0..49]` (50 layers). However, when `num_hidden_layers` is truncated to
**exactly 50**, `hidden_states[50]` becomes the **last element** of the captured tuple,
which triggers the behavior of `@capture_outputs(tie_last_hidden_states=True)` (the
default) wrapping `Qwen3VLTextModel.forward` — it forcibly overwrites the last element
with `outputs.last_hidden_state` (the value after the final `norm` is applied).
Real-hardware verification (`scripts/probe_te_prune*.py`) confirmed that when truncated to
exactly 50 layers, `hidden_states[50]` is **orders of magnitude different** from the true
(64-layer model's) value (max abs diff ~1.5e4 — not the level of quantization error, but a
completely different value). This is exactly the situation warned about by `encoders.py`'s
own guard (`if num_layers <= MINIMAX_H3_TEXT_ENCODER_LAYER: raise ValueError(...)`) —
"the final hidden state when truncated to exactly 50 layers is post-norm, not the value
MiniMax-H3 expects" (thanks to this guard, a mis-configuration of exactly 50 layers is
rejected with an exception via `encode_prompt()`). Using **51 layers** (where `layers[50]`
is executed but its output is never read — the only cost being one wasted layer's worth of
compute) puts `hidden_states[50]` in the middle of the captured tuple, avoiding the
overwrite. Confirmed on real hardware that the 51-layer version's `hidden_states[50]`
**exactly matches** (`torch.equal`, for both bf16 and bnb-4bit nf4) the 64-layer version's.

### Implementation

`core/runner.py`'s `_text_encoder_config_kwargs()` separately loads a `Qwen3VLConfig` from
the same location as text_encoder's `ComponentSpec`
(`pretrained_model_name_or_path="MiniMaxAI/MiniMax-H3"`, `subfolder="text_encoder"`), and
passes an object with `text_config.num_hidden_layers = 51` rewritten in as
`load_components(..., config={"text_encoder": pruned_config})`. `PreTrainedModel.
from_pretrained` skips its own automatic config loading and uses the given object as-is
when `config` is already a `PreTrainedConfig` instance (confirmed in `modeling_utils.py`).
The checkpoint's `layers.51-63.*` show up in `from_pretrained`'s load report as
`UNEXPECTED` and are simply ignored (since they are never constructed, they consume no
VRAM/RAM at all). The vision tower (`model.visual`) is left unchanged (since fl2va's
keyframe / ref2va's reference image and video pixel_values pass through it — explicitly
excluded from removal).

Composes with any combination of `H3_TE_QUANT` (bnb-4bit/none) and `H3_LOWVRAM`
(0/1/group).

### Measured size of the pruned TE

| Precision | Before pruning | After pruning (51 layers) | Reduction |
|---|---|---|---|
| bnb-4bit nf4 | 21.02GB | **17.45GB** | -3.57GB (-17%) |
| bf16 | 66.71GB | **53.06GB** | -13.65GB (-20%) |

Because nf4 quantization compresses per-layer size to roughly 1/4 of bf16, the absolute
reduction amount is also smaller than bf16 (the relative reduction ratio is roughly the
same).

### Result of the MD5-match check

Comparing the output of `H3_TE_PRUNE=0` (no pruning) against `H3_TE_PRUNE=1` (pruned),
with the same seed (768², 5s, 30 steps, fox prompt, seed=12345, using `H3_CACHE=none` to
eliminate FBC's path dependence), confirmed **byte-identical mp4 (md5 match) for both
t2va and ref2va (1 image reference, via the vision tower)**. Empirical proof that the
pruning is mathematically inconsequential. Also confirmed that outputs match exactly
regardless of pruning in `H3_LOWVRAM=1` and `H3_LOWVRAM=group` modes as well (described
below).

### 24GB-tier support: pruning alone wasn't enough (discovered on real hardware, additional fix on the `H3_LOWVRAM_GROUP` side)

The existing 24-32GB-tier mechanism (`H3_LOWVRAM=group`) was designed to keep TE-nf4
(21GB before pruning) resident even during denoise (since group-offloaded transformer's
actual footprint is small at ~1.4GB, this was not a problem at the 32GB tier). The pruned
TE (17.45GB) is still large, and **reproduced an OOM on real hardware at the 22GB limit**
(right at the start of denoise, failing to allocate 224MB with 21.73GB already in use).
Same story at the 24GB limit (also OOM, failing to allocate 1.16GB with 23.12GB in use,
occurring at step 1).

As a countermeasure, added the same "force-free TE only during the denoise loop and
reload it around the decode window" selection method as `H3_LOWVRAM=1`, but only when both
`H3_LOWVRAM_GROUP` and `H3_TE_PRUNE=1` are set (the `group_free_te_for_denoise` flag in
`core/runner.py`). The free point is placed **after** layout_step/latents_step/
timesteps_step (same reasoning as the existing `force_free_te`: since the outputs of
these steps are already materialized as tensors in `state`, they have no further influence
on `_execution_device` resolution from that point on). With `H3_TE_PRUNE=0` (default),
`H3_LOWVRAM_GROUP` is completely unchanged (this flag is only true when both
`H3_LOWVRAM_GROUP` and `H3_TE_PRUNE` are true).

After the fix, confirmed successful completion on real hardware at all of 22GB/24GB/20GB
VRAM limits:

| VRAM limit | Result | Peak VRAM (measured after reset) | Total time |
|---|---|---|---|
| 22GB (before fix, pruning only) | **OOM** (right at start of denoise, failing to allocate 224MB with 21.73GB in use) | - | - |
| 24GB (before fix, pruning only) | **OOM** (step 1, failing to allocate 1.16GB with 23.12GB in use) | - | - |
| 24GB (after fix, run #1) | Yes | 17.72GB | 321.7s (including first-time TE load) |
| 24GB (after fix, run #2, consecutive) | Yes | 18.68GB | 277.7s (shorter since TE stays resident) |
| 20GB (after fix) | Yes | 17.72GB | 320.3s |

The output mp4s across the 24GB x2 runs and 20GB x1 run were **all byte-identical** (md5
match, also matching the output of normal int8 mode (`H3_LOWVRAM` unset)). This
substantiates that group offload's computation is mathematically identical regardless of
VRAM budget (the same conclusion as the existing 32GB/24GB verification results). No
increase in host RAM/swap before/after any test (`free -h`'s Swap used remained
consistently at ~7.9GB throughout, an existing baseline).

### Shortened TE load time under `H3_LOWVRAM=1` (48GB tier)

Pruning shortens the per-request TE-load fixed cost that `H3_LOWVRAM=1` pays every time
(real hardware, free VRAM constrained to ~43.5GB via dummy VRAM allocation):

| | TE load time | TE size |
|---|---|---|
| Without pruning | 42.3s | 21.01GB |
| With pruning | **35.0s** (-17%) | 17.44GB |

The output mp4 is byte-identical (md5 match) regardless of pruning.

### Regression check

Ran t2va with the same conditions with `H3_TE_PRUNE` unset (default `0`), and confirmed it
byte-matches (md5 match) the baseline mp4 captured before this feature was added. Default
behavior is completely unchanged.

## Ref2VA (omni-reference generation, `/api/ref2va`)

Generates video + audio from ordered reference material (**up to 9 images, up to 3 videos,
up to 3 audio, 12 total**; audio alone is not allowed). The reference order matters, since
it corresponds to in-prompt labels (`<Picture i>` etc.) and the rotary layout. Video
references also have their soundtrack used for conditioning. **When the reference is
exactly one audio clip, the seconds parameter may be omitted** (the audio's length becomes
the generated duration; `seconds=0` in the API). Output canvas is not tied to the
reference; if unspecified, it is 16:9 (1344x768).

- Uses a **dedicated checkpoint `transformer_ref/` (61.7GB; same class/config as
  `transformer`, only the weights differ)**. Since it cannot be co-resident with the t2va/
  fl2va transformer, the runner manages it via variant switching (only the active one
  stays resident; free -> reload) (`active_variant` in `/api/status`). TE-nf4, VAE family,
  and processor are shared between both variants.
- VRAM countermeasures (finalized after hitting 3 real OOMs): load transformer_ref after
  reference VAE encode completes (the reverse order OOMs at 98.5GB); force-free TE-nf4
  before denoise (since the reference rows lengthen the sequence — same pattern as
  hires-fix). Freeing the shared text_encoder must drop the reference from both pipeline
  shells (dropping only one leaves a refcount, and VRAM doesn't come back).
- Measured (768x768 requested -> 1344x768 output, 30 steps, seed=12345): 1 image
  reference 523s/87.6GB; image+audio (duration derived from audio, 7.3s) 753s/88.1GB;
  2 images 635s/88.1GB. Visually confirmed subject identity of the reference and
  compositing of multiple references (a person seated in the reference scene's cafe).
  Round-trip switching between ref2va and t2va also works normally (t2va including the
  switch: 188s).
- The UI has a "Ref2VA (reference -> video)" tab (multi-file selection, selection order =
  reference order).

## Starting

```bash
venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
```

At startup, preloads the transformer and TE (by default NF4-quantized and GPU-resident)
(with `H3_TE_QUANT=none`, the legacy scheme applies: transformer/VAE stay resident, and TE
is loaded/freed on every request). Open `http://<host>:8611/` in a browser.

> **On the current box (48GB + 20GB, after the 2026-08-07 GPU swap), the default mode
> cannot be loaded**. Starting requires a low-VRAM mode:
> ```bash
> CUDA_VISIBLE_DEVICES=0 H3_LOWVRAM=1 venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
> ```
> (`CUDA_VISIBLE_DEVICES=0` pins the 48GB RTX PRO 5000 to cuda:0. The 20GB RTX 4000 SFF
> Ada only has 2GB of headroom over the ballast-verified floor of ~18GB, and being sm_89
> it also can't use the SageAttention built for sm_120 — H3 usage on it is unverified and
> not recommended)

## Regression-check probe (for verifying operation before using the UI)

```bash
venv/bin/python scripts/probe_t2va.py
```

Outputs `outputs/probe_t2va.mp4` and `outputs/probe_report.json` (load time, generation
time, peak VRAM, etc.).

## API

- `POST /api/t2va` (multipart/form-data): `prompt`, `resolution`
  (`768x768`|`768x1344`|`1344x768`), `seconds` (5-15), `num_inference_steps`, `seed`
- `POST /api/fl2va`: the above, plus `image` / `last_image` (at least one of the two)
- `POST /api/t2i`: still-image mode. `frames` (default 22, or 5) instead of `seconds`.
  Saves both an ultra-short mp4 and a center-frame PNG (see the "Still-image mode" section)
- `POST /api/t2i_batch`: batch generation of still images. Send `prompts` repeated for
  the number of scenes (up to 24). Amortizes the low-VRAM-mode load fixed cost across the
  whole batch, once (see the "Still-image batch generation" section)
- `POST /api/ref2va` + `still=1` + `frames`: reference-conditioned still image (ref2i).
  Saves a center-frame PNG
- `POST /api/ref2i_batch`: batch generation of reference-conditioned still images. Common
  `references` + `prompts` (up to 24). Generates a sequence of scene stills with a
  consistent character (see the "Reference-conditioned still images" section)
- `POST /api/ref2va_batch`: batch generation of reference-conditioned **videos**. Common
  `references` + `prompts` + `seconds` (common to all scenes, required). Generates a
  sequence of narrative scene videos (see the "Reference-conditioned video batch
  generation" section)
- `GET /api/status`: load state, VRAM/RAM
- `GET /api/progress`: for polling generation progress

## Residency reference (what is loaded/freed when, in each mode)

Since the number of combinations of mode, quantization, turbo, and TE placement has grown,
a table for looking up **"what is on the GPU at this point, under this setting"** at a
glance has been placed in **[docs/RESIDENCY.en.md](docs/RESIDENCY.en.md)**. It also
records an actual case of misreported peak-VRAM breakdown (mistaking the denoise-time peak
for the decode-time peak) as a "common misunderstanding."

## Technical documentation

This README is an operational document on "how to use it." The technical side is split in
two by purpose.

| Document | Contents | Audience |
|---|---|---|
| **[docs/TECHNICAL_OVERVIEW.en.md](docs/TECHNICAL_OVERVIEW.en.md)** | **Technical overview**: capabilities, architecture, how the various techniques are integrated, handling per VRAM tier, performance, configuration reference | Anyone who wants to know what this app does and how |
| [docs/internal/TECHNICAL_REPORT.en.md](docs/internal/TECHNICAL_REPORT.en.md) | **Internal document (work log)**: all work from 2026-08-04 to 08-08 — the background to design decisions, 16 pitfalls hit on real hardware and their resolutions, verification methodology | Developers who would rather not hit the same traps |

The technical report is **a record of the process, bugs, dead ends and all**, so if you only
want the specification and the performance, the technical overview is enough.

## List of community improvements incorporated

A record of work incorporating improvements from the ComfyUI community etc. into this app
(the diffusers path) is compiled in
**[docs/COMMUNITY_IMPROVEMENTS.en.md](docs/COMMUNITY_IMPROVEMENTS.en.md)** (what was
incorporated / what was investigated but not incorporated / what was independently
implemented after being inspired by something, each with its source, measurements,
verdict, and pitfalls hit).

## Switching settings from the UI (2026-08-06)

Opt-in settings that could previously only be changed via environment variables can now be
operated from the UI, split into two categories by nature.

### Instant apply (checkbox directly above the generate button, no reload needed)

FirstBlockCache (+ threshold), Sage Attention, Turbo LoRA. Sent as per-request parameters
(`cache` / `cache_threshold` / `attn` / `turbo`), applied by `MiniMaxH3Runner.
apply_instant_settings()` to the resident transformer after acquiring the generation lock
and before denoise (`disable_cache()`/`enable_cache()`, `set_attention_backend()`,
`_TurboLoRALinear.enabled`). **If unspecified, the process default applies as before**, so
existing curl/scripts work unchanged. Enabling turbo automatically disables FBC (following
the original safety rule).

Measured (same seed, switched without restart): FBC on 100.8s (6 skips) / off 129.3s
(0 skips), Sage on 129.3s / off (native) 158.5s, Turbo on (8 steps) 38.9s.

### Requires reload (collapsible header + "Apply (reload)" button)

transformer int8, TE quantization, TE layer pruning, low-VRAM mode, video VAE fp16.
`POST /api/settings/apply` calls `apply_reload_settings()` in `core/settings.py`, which
frees all models within the runner and reloads them with the new settings **without
restarting the process** (self-killing the process would leave no one able to restart it,
making the UI permanently unrecoverable, so no os.execv/self-kill approach is implemented).
Returns 409 during generation; 400 for unverified combinations.

Measured: transformer_quant none->int8->none took 56.0s / 55.0s (GPU 87.5->55.0->87.3GB),
lowvram 0->1->0 also round-trips correctly. `GET /api/settings` returns current values and
choices, and the UI initializes from this.

**UI implementation note**: unchecking a checkbox must send an explicit `turbo=0`, not an
empty string (an empty string can be interpreted as "unspecified = server default," which
could mean unchecking does not actually disable turbo on a server started with
`H3_TURBO_LORA=1`). turbo and upscale are mutually exclusive — selecting one automatically
deselects/disables the other (consistent with the server-side 400).

## Gallery of generated videos (2026-08-06)

Tiles the mp4s directly under `outputs/` below the result display. Thumbnails are not
generated server-side; instead, `<video preload="metadata" src="....mp4#t=0.1">` is used to
have the browser draw the first frame (no added dependency; the same technique used for
Ref2VA's reference tiles).

- `GET /api/outputs`: **the *.mp4 and *.png directly under it** (still-image mode's
  outputs; distinguished via the `type` field "video"/"image". Verification artifacts like
  `outputs/ab_*` are excluded). Duration/resolution obtained via ffprobe and cached in
  memory keyed by mtime+size
- `POST /api/outputs/delete`: **path-traversal countermeasure** (rejects `/`, `\`, `..`,
  and after resolving verifies the result is directly under `outputs/` — blocking
  escapes via symlinks too). The UI requires `confirm()`. Confirmed on real hardware that
  `../app.py`, `/etc/passwd`, `ab_*/...`, etc. return 400
- `POST /api/outputs/concat`: **concatenation order follows "the order checked"**
  (display order is newest-first). If all inputs' parameters match, uses
  `concat demuxer + -c copy` (**no re-encoding = zero quality loss**); if they don't
  match, re-encodes via `filter_complex`, matching the resolution of the first video (a
  silent `anullsrc` is synthesized for silent inputs). Fewer than 2 inputs returns 400,
  concurrent concat requests return 409 (does not take generation_lock since it doesn't
  use the GPU). **PNGs (still images) are excluded from concatenation** (because ffprobe
  can read PNG as a video stream too, so it's explicitly rejected with 400 by extension;
  the UI also disables the concat button if any PNG is among the selection)

**A note on dependencies**: this app's generated-output muxing **uses PyAV**
(`av.open()`, writing libx264+aac directly, `_mux_mp4()` in `core/runner.py`) — **it never
used the ffmpeg command**. The gallery's ffprobe/ffmpeg calls are **the app's first ever
external-command dependency** (this environment has `/usr/bin/ffmpeg`; if absent, it
catches `FileNotFoundError` and returns an explicit error). If you want to eliminate the
external-command dependency, a realistic alternative is "packet re-muxing via PyAV
(equivalent to `-c copy`), limited to identical parameters, erroring on any mismatch"
(implementing full re-encoding of mismatched parameters in PyAV would essentially be
re-implementing ffmpeg, so this is not recommended).

**UI implementation note**: rebuilding all tiles on every selection causes all `<video>`
elements to be regenerated, producing flicker (noticeable at 95 tiles). Selection state
should be updated as a **diff** on badges/checkmarks (`updateGallerySelectionUI()`; the
list itself is only rebuilt when `/api/outputs` is fetched).

## Real-hardware verification of the official skill (h3-prompt-writing) mode (2026-08-07)

Results of measuring `h3-official` mode against a local LLM (gemma4-31B) connection.

**Structural conformance**: the generated prompts fully complied with the official spec.
- T2VA: 3 fields (`integrated_multimodal_description` / `overall_soundscape` /
  `non_diegetic_music`), `[Shot 1]` with no timestamp, `[Shot 2] At 00:05.000` in 3-digit
  notation, dialogue preserved verbatim in the original language via
  `<d>[Japanese] おかえり</d>`, speaker ID `(S1)` also attached. Response time 6.2s
- Ref2VA: all 6 fields output, using `<Subject n>`/`<Picture n>`/`<Audio n>` labels and
  relationship markers such as `fully_preserved` / `fully_copy`. Response time 12.8s

**Measured cut positions** (768², 10s, turbo 8 steps, LLM-generated prompt, 2 seeds each.
ffmpeg scene detection `gt(scene,0.15)`):

| Notation | Instruction | Detected (seed 12345 / 777) | Deviation |
|---|---|---|---|
| Official `[Shot 2] At 00:05.000` | 5.000s | 4.875 / 4.875 | **-0.125s (identical for both seeds)** |
| Custom `CUT 2 [6-10s]` | 6.0s | 6.083 / 6.333 | +0.083 / +0.333s |

The official notation shows **less variance** (identical across the two seeds). That said,
the custom CUT notation also stays within a 0.1-0.3s deviation, **not nearly as large a
gap as the earlier single hand-written-prompt trial (custom notation, +1.0s)**. Both are
practically usable; the measured conclusion is that the official notation is more
reproducible.

**Implementation pitfall (reproduced and fixed on real hardware)**: specifying `lang=ja`
**still returns English**. Placing the language instruction at the start of the system
prompt gets overwritten by the 15.8KB English reference that follows it (which is packed
with English output examples). Resolved by **placing the language instruction after the
reference body, as the final instruction** (the same principle established as "the end of
the prompt is strongest" in diffusers-server's T-pose implementation). After the fix, only
the narrative text becomes Japanese, while field names, `[Shot n]`, timecodes, and dialogue
inside `<d>` tags remain in English/the original language.

## h3-official quality assurance: verification + repair loop (2026-08-08)

To avoid arguing by speculation about "is this an LLM capability limit, or fixable via
prompting," **first measured the failure rate** (`scripts/probe_h3official_compliance.py`,
5 inputs x 3 runs), then applied countermeasures based on the results. LLM: gemma4-31B
Q4_K_M / llama.cpp, **n_ctx=7680**.

### Baseline measurement: failures were concentrated in a single class

| Failure class | Occurrence rate |
|---|---|
| Structure/notation (field names, `[Shot n]`, timecodes, `<d>` tags, speaker IDs) | **0/15 (0%)** |
| Time allocation | 6/15 (40%) |
| Context overflow | 0/15 (2,900+ tokens of headroom) |

**The concern that "a different input would surface a different problem" turned out to be
unfounded, in a good way** — violations were not scattered, but concentrated in the single
class of time allocation. Moreover, those 6 cases split into two natures:

- **Input physically impossible (3/6)**: the estimated speaking duration of the dialogue
  exceeds the requested total duration. One example was a 38-character monologue
  (estimated 9.5s) requested with a 9s duration, where **the LLM had already allocated
  the best possible split, giving all 9s to a single shot**. Since the official spec
  requires dialogue to be preserved verbatim, shortening isn't an available escape route
  either -> **unsolvable by any model. Information that should be returned to the user**
- **LLM allocation mistake (3/6)**: fits within the duration, but the placement is poor ->
  the kind of thing that can be fixed by pointing it out

Context: t2va's system prompt uses 4,397 of 7,680 tokens, with headroom to spare. **ref2va
uses 6,191 tokens (81%), leaving only about 1,100 tokens of headroom** (in measurements,
all 6 fields completed normally, but caution is warranted with longer inputs).

### Countermeasures (3 layers)

1. **Validator** (`core/prompt_check.py`): implements deterministically-checkable rules
   (F1 3 fields / F2 no timestamp on the first shot / F3 strictly increasing cut times /
   F4 within duration / F5 minimum shot-duration floor / F6 `<d>` tags and language tags /
   F7 dialogue fits within its shot / F8 speaker ID). F5 and F7 are **this app's practical
   rules, not present in the official spec** — the official spec only says "cut times must
   be within the duration," which permits a literal-but-unusable output such as cutting
   at 4.5s within a 5s duration, leaving the final shot only 0.5s (this actually occurred).
   Semantic consistency (e.g. mixed shot framing within one shot) is only approximately
   checked and left as a **warning**. **This also helps hand-written prompts**
2. **System prompt improvement**: the English wrapper **had nothing after the guide body**
   (only the Japanese wrapper had a trailing block, from the `lang=ja` fix). Since the
   default is `lang=en`, all instructions including the duration constraint were buried
   inside the 15.8KB guide. Added 6 time-allocation rules (minimum shot-duration floor,
   estimating dialogue speaking time, guideline shot count, one composition per shot,
   the boilerplate phrase for off-screen audio, speaker ID) at the **end** of both versions
3. **Repair loop** (`enhance_prompt_checked` in `core/llm.py`): upon detecting a
   violation, present its content in Japanese and regenerate (up to 2 times,
   `H3_OFFICIAL_MAX_REPAIRS`). **Discards any repair candidate that increases the
   violation count** (preventing an accident where a repair breaks something else).
   Input infeasibility is judged **before handing off to the LLM**, raising
   `InfeasibleInputError` -> 400 + advice

### Measured results after improvement (same conditions)

| | clean | infeasible input detected | **violations remaining** |
|---|---|---|---|
| Baseline | 9/15 | 0 | **6/15** |
| **After improvement** | **12/15** | **3/15** (correct behavior) | **0/15** |

**Unresolved violations dropped to zero.** Every case now either "returns a valid prompt"
or "returns an infeasibility reason with advice." Median time went from 8.5s to 9.2s
(+8%); the repair loop only fires when needed (about 21s, roughly double, when it does).

The API adds `violations` / `warnings` / `check_report` / `attempts` / `repaired` to the
h3-official response of `/api/prompt/enhance`. The UI displays any remaining findings in
the status area (**generation is not blocked** — since the prompt is editable, the human
is the final gate).

## Disk cache of the quantized text_encoder (`H3_TE_PREQUANT`, default `1`, 2026-08-08)

`H3_LOWVRAM=1` reloads TE on every request, and that time becomes a straight fixed cost.
Most of that time is spent "reading the original bf16 weights + quantizing them to
bnb-4bit on the fly," so **saving the already-quantized weights once means only having to
read them from then on**. Saved automatically on first load, and read from there afterward.

**Measured (RTX PRO 5000 48GB + `H3_LOWVRAM=1 H3_TE_PRUNE=1`, t2i turbo 4 steps, 4 runs
each)**:

| | TE load | Request total |
|---|---|---|
| Cache disabled (legacy) | 46.5-55.8s (avg **53.0s**) | 108.7-150.0s (avg **128.6s**) |
| **Cache enabled (default)** | 21.1-34.3s (avg **29.5s**) | 75.3-90.6s (avg **83.2s**) |
| Difference | **-23.5s (1.8x)** | **-45.4s (-35%)** |

**Equivalence**: outputs for the same seed are **byte-identical PNG MD5** (confirmed for
seeds 3 and 4). In addition, the probe (`scripts/probe_prequant_equivalence.py`) confirmed
`hidden_states[50]` is bit-identical via `torch.equal` (max_abs_diff 0.0). Since bnb-4bit
quantization is deterministic, this is the expected result, but this project's convention
is to not adopt something just because it's faster.

- Cache location: `models/prequant/te_<quant>_prune<0|1>/` (already in .gitignore).
  **A separate directory per setting combination** — since changing TE_QUANT / TE_PRUNE
  changes the weight contents, reusing the same location would risk reading stale weights
  after a setting switch
- Disk footprint: **17.44GB** (`H3_TE_PRUNE=1`) / ~21GB if unpruned
- `H3_TE_PREQUANT=0` fully disables this (identical behavior to before this feature). If
  free disk space falls below `H3_TE_PREQUANT_MIN_FREE_GB` (default 25GB), **saving is
  skipped and generation continues** (the cache is a speedup, not a required feature).
  Saving writes to a temp directory then renames, so a mid-write crash can never leave a
  half-written cache looking "valid"

**A measurement pitfall hit during this verification**: measuring right after saving
shows an unrealistic value of **2.6 seconds**, since the 17.44GB is still in the page
cache. In practice, reading the transformer (34GB) from disk between requests evicts TE's
page cache, so real-world measurements settle around 21-34 seconds (getting slower with
each additional request, plateauing around 34s). **Still faster than the legacy path's
46-56 seconds**, but the probe's raw numbers should not be reported as-is as real
performance.

## Keeping text_encoder resident on a separate GPU (`H3_TE_DEVICE`, default ``, 2026-08-09)

The root cause of `H3_LOWVRAM=1` reloading TE on every request is that **there is nowhere
to put TE during denoise** (on 48GB, transformer-int8 34GB + activations 5GB = 39GB,
leaving only 9GB, which doesn't fit TE's 17.45GB). Moving TE to a second GPU eliminates
this reload.

```bash
# Example: keep TE resident on cuda:1 (do not set CUDA_VISIBLE_DEVICES = expose both GPUs)
H3_LOWVRAM=1 H3_TE_PRUNE=1 H3_TE_DEVICE=cuda:1 venv/bin/python -m uvicorn app:app --port 8611
```

**Measured (RTX PRO 5000 48GB + RTX 4000 SFF Ada 20GB, t2i turbo 4 steps, 768²)**:

| | Run #1 | Steady state (run #2 onward) |
|---|---|---|
| Legacy (TE also on cuda:0) | 81.3s | 67.6-86.2s (avg **78.4s**) |
| **TE on cuda:1** | 88.2s (including first-time TE load) | 33.6-36.7s (**about 35s, -55%**) |

TE is loaded **only once, right after process startup**. It stays resident on GPU1 at
17GB, while GPU0 drops to as low as 2.3GB when idle.

- **ref2va is automatically rejected** (in the 20GB-tier case): confirmed on real hardware
  that the activations from running a 2048px-short-side reference through the vision
  tower don't fit and OOM (19.25GB in use, short by 204MB). If the TE GPU has less than
  24GB, this returns 400 with the reason — rather than letting it run and OOM under an
  assumption of "it should work." Automatically allowed if 24GB or more. t2va/fl2va/t2i
  and their respective batches can be used together
- **PCIe width is not an issue**: GPU1 is Gen4 x4 (a card rated for x16 but connected at 4
  lanes), but TE is only loaded once at startup, and the only per-request transfer is
  prompt_embeds at about 42MB (about 6ms)

**Output does not bit-match the legacy configuration** (PNG MD5 differs). The cause is
**rounding due to the architectural difference between sm_120 (Blackwell) and sm_89
(Ada)**; the measured relative RMS difference of `hidden_states[50]` is **0.084%**
(max_abs_diff 3.5, value RMS 57.4). This is **20x smaller** than the difference from the
already-adopted reference-prefix sharing (1.5%), and orders of magnitude away from
bug-level territory (negative control 27-30%). It's the same kind of trajectory drift as
Sage Attention or int8, and equally imperceptible visually. **Remove `H3_TE_DEVICE` for
any A/B test requiring bit reproducibility.**

### Implementation summary (the `_execution_device` pitfall, again)

`_execution_device` returns the device of the first nn.Module in component order
(`text_encoder, tokenizer, processor, vae, ...`) (confirmed by reading the
implementation). **Placing TE on cuda:1 makes this return cuda:1**, and layout/latents/
timesteps as well as decode all build tensors on the wrong GPU as a result.

The first attempt was to "only close the window around layout," but **it also fired during
decode** (`latents = latents * latents_std + latents_mean` failed with `Expected all
tensors to be on the same device, cuda:0 and cuda:1`, reproduced on real hardware). Since
closing windows individually is leaky, switched to the inverted design of **"detach TE
from the pipeline by default, and attach it only while encoding"**
(`_te_attached()`). This stays safe even as new code paths are added. The module instance
itself is kept alive by `self._te_module`, so detaching it doesn't free it.

During the layout/latents/timesteps window, `_pin_execution_device_to_compute()` also
temporarily detaches vae (since, after TE is detached, the next nn.Module found would be
the CPU-resident vae, making `_execution_device` return cpu instead — this ensures
transformer is found first).

## Still-image mode (T2I, `/api/t2i`, production implementation 2026-08-07)

A mode that uses H3's "ultra-short video -> extract still frame" as a substitute for image
generation. This is the production implementation of the conclusion from the earlier
ultra-short-clip probe (below). The value proposition is being able to produce **still
images whose art style exactly matches H3's**, for use as FL2VA's first frame or Ref2VA's
reference image (speed cannot beat a dedicated T2I model, since it's dominated by fixed
costs).

- **API**: `POST /api/t2i` — `prompt` / `frames` (default 22, or 5) / `resolution` or
  `height`+`width` / `num_inference_steps` / `seed` / instant-apply parameters
  (cache/cache_threshold/attn/turbo). Response includes both an ultra-short mp4 and the
  **center frame's PNG** (`image_url`, `t2i_<ts>.png`)
- **UI**: the "T2I" tab in the still-image row (seconds/upscale fields hidden, only frame
  count selectable). PNGs are also tiled in the gallery (deletable, not concatenable)
- **Frame count**: 22 (0.917s) is the default. 5 (0.208s) is experimental but works thanks
  to the VAE fix below
- **The three-part implementation** (`core/runner.py`):
  1. `generate(still=True, still_frames=...)` — diffusers' minimum-duration-5-seconds
     validation (`MINIMAX_H3_MIN_DURATION`) is relaxed via `_relaxed_min_duration()` for
     exactly one call of `MiniMaxH3SetupStep` (safe since generation is serialized by
     generation_lock, so no leakage)
  2. **VAE small-clip decode fix** (`H3_VAE_SMALLCLIP_FIX`, default 1):
     `AutoencoderKLMiniMaxH3._decode()` has an upstream bug where a latent of 1-2 frames
     yields num_chunks=0, which crashes on `torch.cat([])` (reproduced on real hardware
     with the probe). Added a branch via monkeypatch on the runner side: "if less than one
     chunk, decode the entire token set via a single `_decode_clip()` call, and trim
     `frame_pre_padding` (3) and the trailing padding." A latent of 2 frames x4 - 3 = 5
     pixel frames results in matching geometry. Normal video (num_chunks>=1) is delegated
     entirely to the original implementation, so this has no effect there (the venv's
     diffusers itself is unmodified)
  3. **Restoring steady state on decode exceptions**: as a countermeasure against the
     "decode exception -> transformer already dropped -> restore code never runs ->
     ~98.5GB stays leaked -> subsequent requests OOM in a chain" behavior observed with
     the probe, wrapped the decode portion of `generate()`/`generate_ref2va()` in
     try/except, so that `_restore_decode_steady_state()` (moves VAE to CPU, reloads
     transformer/TE per mode) runs before re-raising even on exception. **Verified on real
     hardware** (2026-08-07): using the one-shot failure-injection hook
     `H3_DEBUG_FAIL_DECODE=1` (read via `os.environ.pop`, so normal behavior resumes from
     the next request in the same process) to deliberately fail decode, confirmed that
     after the 500 response the GPU is freed down to ~2GB, and **the next request in the
     same process returns 200 normally**

**Measured (2026-08-07, RTX PRO 5000 48GB + `H3_LOWVRAM=1`, 768², 30 steps, fbc+sage,
seed 12345)**:

| frames | Denoise | Decode | Peak VRAM | Total (including per-request reload) | Quality (visual) |
|---|---|---|---|---|---|
| 22 (0.917s) | 29.0s (1.0s/step) | 1.8s | 35.0GB | 157s | No breakdown, high quality |
| 5 (0.208s) | 9.1s | 0.67s | 35.2GB | 125s | No breakdown (first-ever successful real decode) |

The 5-frame audio (0.208s) is also non-silent (rms 0.055).

### Measured result for consecutive generation (a sequence of narrative scene images with different prompts): **group mode is worse for still images**

Measured the hypothesis that "resident mode (`H3_LOWVRAM=group`) should be faster for
consecutive generation since the reload fixed cost disappears," via 3 consecutive runs of
3 narrative-scene prompts (frames=22, 768², 30 steps, fixed seed)
(2026-08-07, RTX PRO 5000 48GB). **The result was the opposite: about 1.5x slower than
lowvram=1**:

| Mode | 1st image | Steady state (2nd onward) | Steady-state breakdown |
|---|---|---|---|
| `H3_LOWVRAM=1` | 157s | **~157s** | TE+transformer load ~120s + denoise 29s + decode 2s |
| `H3_LOWVRAM=group` | 310s (including cold TE) | **~240s** | denoise **130s** + decode 2s + post-decode TE reload **75-97s** |

- **The denoise degradation from 29s to 130s is the core issue**: group offload transfers
  the entire ~34GB of int8 transformer over PCIe, block by block, every single step. This
  transfer cost is fixed regardless of sequence length, so it completely dominates when
  computation is light (an ultra-short 22-frame clip) (with a normal 124-frame video, this
  is hidden within the heavier computation — the design assumption that breaks down for
  still images)
- Group mode force-frees TE during the decode window and reloads it afterward (headroom
  design for the 24-32GB tier). On this box, the NF4 TE load measures 75-97s (much slower
  than the 15-40s of the 96GB era), so this reload also lands on the steady-state cost
- Quality is as good as lowvram=1 (visually). Peak VRAM is 25.1GB
- **Conclusion**: on this box, consecutive still-image generation is **fastest kept on
  `H3_LOWVRAM=1` (~157s/image)**. Further reduction comes from `/api/t2i_batch` below
  (implemented as a result of this measurement)

### Batch generation of still images (`/api/t2i_batch`, production implementation 2026-08-07)

Building on the measurement above, implemented an endpoint that **amortizes the fixed
cost of `H3_LOWVRAM=1`** (~110s: TE load 75-97s + transformer load ~20-35s) **once across
the whole batch** (`generate_still_batch()` in `core/runner.py`). Reorders `generate()`'s
lowvram choreography by phase:

    entry   : [nothing big resident]
    encode  : [TE-nf4]         setup/encode/layout/latents/timesteps for all scenes
    denoise : [transformer]    denoise all scenes in sequence
    decode  : [vae pair]       decode all scenes in sequence -> save PNG/mp4 (saved
                               per scene as it progresses, so a mid-batch failure
                               still leaves completed scenes intact)

- **API**: `POST /api/t2i_batch` — send `prompts` repeated via multipart for the number
  of scenes (up to 24 scenes). `frames`/`resolution`/`num_inference_steps`/`seed`/
  instant-apply parameters are common to all scenes (only the prompt can vary). The UI's
  "Batch continuous generation (1 line = 1 scene)" checkbox on the T2I tab sends each
  line of the prompt field as one scene
- **Resetting shared state between scenes** (the key implementation detail): since the
  scheduler's sigmas/timesteps values are identical across all scenes (same geometry, same
  step count), it suffices to reset `_step_index = None` right before denoise
  (`MiniMaxH3Scheduler.step()` re-derives the index from the timestep value).
  FirstBlockCache does `_reset_stateful_cache()` + `cache_context` per scene (a per-scene
  version of `generate()`'s per-request reset)
- **Proof of equivalence**: the sequential `/api/t2i` and the batch's scene 1, with the
  same prompt and seed, produce **byte-identical mp4 and PNG** (md5 match). Phase
  reordering is mathematically inconsequential (note: this does not match group mode's
  output — a known effect of FBC's decisions changing with execution path, not something
  specific to batching)
- Modes other than `H3_LOWVRAM=1` (where the big models stay resident) gain nothing from
  phase reordering, so they fall back to sequential `generate()` calls under the same API
  (identical response format)

**Measured (2026-08-07, RTX PRO 5000 48GB + `H3_LOWVRAM=1`, 3 scenes, frames=22, 768²,
30 steps)**:

| Method | Total | Per image | Breakdown |
|---|---|---|---|
| Sequential `/api/t2i` x3 | ~471s | **157s** | reload ~120s + denoise 29s + decode 2s, every time |
| `/api/t2i_batch` (3 scenes) | **202.6s** | **67.5s** | load ~110s (once) + encode 0.9s + denoise 84.8s + decode 6.9s |

Since the marginal cost of an additional scene is **~31s/image** (denoise 28-29s + decode
2.3s), the per-image cost asymptotically approaches 31s as the scene count grows (computed
as ~42s/image at 10 scenes, ~36s/image at 24 scenes). Peak VRAM is 35.0GB, same as
single-image generation (only tens of MB grow per scene, for latents+prompt_embeds).

### Reference-conditioned still images (ref2i, `still=1` on `/api/ref2va` and `/api/ref2i_batch`, production implementation 2026-08-07)

Implemented in production following the success of the spike below. A mode for producing
**scene-by-scene stills with a consistent character** from a character reference. The
resulting stills can be fed back in as FL2VA's first frame or the next Ref2VA's reference
(building material for generating multiple narrative videos).

- **Single**: pass `still=1` + `frames` (default 22, or 5) as extra parameters to
  `POST /api/ref2va`. `seconds` is ignored, and the center frame's PNG
  (`ref2i_<ts>.png`) is attached to the response's `image_url`. Implementation reuses the
  same three-part set as t2i (duration-gate relaxation only during the setup step, the VAE
  small-clip fix, and the existing decode-exception cleanup)
- **Batch**: `POST /api/ref2i_batch` — common `references` + `prompts` (up to 24 scenes).
  With `H3_LOWVRAM=1`, uses `generate_ref2i_batch()` (the ref2va version of t2i_batch's
  same phase reordering: encode all scenes with TE resident -> a single VAE window to
  encode the reference for all scenes -> layout/timesteps -> a single transformer_ref
  load, denoising everything -> decode everything together). Other modes fall back to
  sequential calls. The between-scene reset of scheduler/FBC uses the same technique
  already proven to md5-match in t2i_batch
- **UI**: the "Ref2I" tab in the still-image row (moved on 2026-08-09 from a "still image"
  checkbox inside the Ref2VA tab to its own independent tab; see "Two-row tab layout"
  below) + "Batch continuous generation (1 line = 1 scene)" checkbox

**Measured (2026-08-07, RTX PRO 5000 48GB + `H3_LOWVRAM=1`, 1 Little Red Riding Hood
reference, 3 scenes, frames=22, 768², 30 steps)**: total 494.7s = **164.9s/image**
(1.6x faster than sequential's ~265s/image). Breakdown: encode phase 212.5s + denoise
178.1s (57.7-60.2s/scene) + decode 6.6s. Peak VRAM 36.8GB. Quality and character
consistency both good (all 3 scenes kept the red cloak and followed each prompt).

- ~~Known room for improvement: reference vision encoding runs per scene~~ ->
  **resolved by "Shared KV cache for the reference prefix" below (2026-08-08)**

### Shared KV cache for the reference prefix (`H3_REF_PREFIX_CACHE`, default 1, production implementation 2026-08-08)

Resolves the problem, in the encode phase of ref batches (`generate_ref_batch` =
ref2i_batch/ref2va_batch), where the Qwen3-VL encoding of the reference labels+vision
(~4104 tokens, ~65s/scene) was duplicated per scene. In ref2va's token sequence, "the
reference is prepended, and the prompt is appended verbatim at the end" (confirmed via
`build_ref2va_presentation` in packing_ref2va), and since the conditioning source Qwen3-VL
is a causal LM, **the text representation of the reference prefix does not depend on the
prompt** — encode the prefix once with `use_cache=True`, bake it into a `DynamicCache`,
and for each scene continue the cache with only the prompt's tail (14-33 tokens, ~0.2s)
(`_encode_ref_prompts_shared_prefix()` in `core/runner.py`).

**Verification** (`scripts/probe_ref_prefix_cache.py`, based on real-hardware measurement
after reading transformers 5.14.1's modeling_qwen3_vl.py):

- The prefix portion's hidden_states[50] is **bit-identical** (`torch.equal`) to the full
  computation
- The prompt-tail portion retains a relative RMS rounding difference of ~1.5%. The cause
  is that the kernel/GEMM's tiling path changes with sequence length — this holds even
  with eager fixed (not sdpa-specific). **Confirmed not a logic bug via a negative
  control**: a continuation deliberately corrupted with a broken position offset jumps to
  27-30% relative RMS (20x). 1.5% is at the level of "rounding noise from a correct
  computation"
- Recipe for the continuation call (3 pitfalls): `attention_mask=None` (rides on
  `compute_3d_position_ids`'s arange branch = sequential numbering from the past length +
  the rope_deltas addition), `mm_token_type_ids`/`pixel_values`/`grid_thw`-family args are
  **all None** (passing `image_grid_thw` recomputes and overwrites `model.rope_deltas`),
  and reuse the cache serially per scene by cropping back with
  `DynamicCache.crop(prefix_len)`. Since `rope_deltas` is **instance state** on
  Qwen3VLModel, no other TE call may be interleaved between the prefix call and all the
  continuations (designed to complete within a single helper call)

**E2E measured (ref2i_batch, 1 Little Red Riding Hood reference, same 3 scenes,
seed 12345, 768², 30 steps)**:

| | Encode phase | Total | Per image |
|---|---|---|---|
| No sharing (equivalent to H3_REF_PREFIX_CACHE=0) | 212.5s | 494.7s | 164.9s |
| **With sharing (default)** | **83.1s** | **350.1s** | **116.7s (-29%)** |

Quality: PSNR 21.9-27.4dB against the same-seed output without sharing. Batch output does
not bit-match the legacy path (the ~1.5% rounding difference on the prompt tail drifts the
trajectory — the same kind and level of epsilon-class drift as when Sage Attention was made
default), but composition, quality, and character consistency are all visually equivalent.
For A/B tests requiring bit reproducibility, revert to the legacy path with
`H3_REF_PREFIX_CACHE=0`. Since ref2va_batch (video) uses the same code for its encode
phase, the same reduction applies directly (~130s/3 scenes). The reference VAE encode
(~a few seconds/scene) is not shared and stays per-scene (the benefit is small, and this
avoids increasing the risk of state aliasing).

### Batch generation of reference-conditioned videos (`/api/ref2va_batch`, production implementation 2026-08-08)

Applies the same phase reordering as ref2i (implemented via the same method,
`generate_ref_batch(still=False)`) to **normal-duration video**. Generates each scene's
video for a narrative, sharing a common reference with varying prompts, continuously.
Duration (`seconds`) is common and required across all scenes (since the scheduler's
sigmas/timesteps values being identical across scenes is a precondition for phase
reordering; automatic duration derivation from an audio reference is also unavailable in
batch mode). The UI's Ref2VA / Ref2I tabs have a "Batch continuous generation (1 line = 1
scene)" checkbox (usable independently of the tab: Ref2I+batch = ref2i_batch, Ref2VA+batch
= ref2va_batch).

**Measured (2026-08-08, RTX PRO 5000 48GB + `H3_LOWVRAM=1`, 1 Little Red Riding Hood
reference, 2 scenes, 5s, 768², 30 steps)**: total 803.2s = **401.6s/video** (17% shorter
than sequential's ~485s/video @2 scenes). Breakdown: encode phase 151.3s + denoise 503.4s
(247.9-255.5s/scene) + decode 25.4s. Peak VRAM 40.5GB. The marginal cost of an additional
scene is ~330s/video (vision encode ~65s + denoise ~250s + decode ~13s), so as scene count
grows this asymptotically approaches **about a 32% reduction**. Since video is
denoise-dominated, this is less dramatic than the still-image batch, but quality and
character consistency are as good as sequential (both scenes kept the red cloak and
followed their prompts).

### Spike: Ref2VA x ultra-short clips (reference-conditioned still images, 2026-08-07, measured -> productionized as described above)

Measured whether "the ultra-short 22-frame clip, already verified in t2va, still holds up
in quality when combined with reference packing (packing_ref2va: reference rows outnumber
generation rows)" (`scripts/probe_ref2va_short.py`, main codebase unmodified, monkeypatch
confined to the probe). If this held up, it would enable mass production of
**character-consistent scene stills**, speeding up building material for generating
multiple narrative videos.

Conditions: reference = 1 PNG of a Little Red Riding Hood-style girl (this repo's own t2i
output), the same prompt (with a `<Picture 1>` reference), seed 12345, 30 steps, 768².
RTX PRO 5000 48GB + `H3_LOWVRAM=1`.

| Condition | Denoise | Decode | Peak VRAM | Wall (sequential, including load) | Quality (visual, center frame) |
|---|---|---|---|---|---|
| short22 (0.917s) | 58.1s | 1.7s | 36.7GB | 265s | **No breakdown, high quality** |
| baseline (5.0s) | 243.1s | 10.6s | 40.7GB | 485s | No breakdown (anchor) |

- **Verdict: GO**. Even in this out-of-distribution overlap (ultra-short clip x reference
  packing), quality holds up, and the reference's costume/props (red cloak, lantern, white
  dress) and art style were preserved even at short22. Composition also correctly followed
  the prompt (seated on a moss-covered rock)
- Denoise is about 2x t2va's 22-frame value (29s -> 58s), since the packed sequence
  lengthens by the reference rows. Still 1/4 of the 5-second baseline
- The strictness of facial identity depends on how the reference photo looks (this
  experiment used a somewhat back-facing shot) — a property of the reference, not a flaw
  of the ultra-short clip
- The production implementation (API/UI, the ref2va version of `generate_still_batch`) was
  **not started at the time** — the same three-part set as t2i (duration-gate relaxation,
  VAE small-clip fix, exception cleanup) is reusable, and the batch phase-reordering
  approach looks reusable too, following t2i_batch's method (scheduler `_step_index` reset
  + per-scene FBC reset, with a single shared reference encode if the reference is common
  across all scenes)

### The preceding ultra-short-clip probe (2026-08-07, measurement record)

Frame counts step in units of 17n+5 (minimum 5 = 0.208s, next 22 = 0.917s). The probe was
verified via monkeypatch only (`scripts/probe_short_frames.py` + `_one.py`). The
environment at the time was RTX PRO 6000 96GB, default mode.

| Frame count | Denoise (30 steps) | Decode | Result |
|---|---|---|---|
| 5 (0.208s) | completes | **fails** | VAE chunked-decode boundary bug (2-frame latent gives num_chunks=0 -> `torch.cat([])`) -> **fixed by item 2 above** |
| **22 (0.917s)** | **13.5s** (1/7.6 of the 124f case) | 1.2s | **success, quality on par with the 5-second baseline** (visually confirmed) |
| 124 (5s, baseline) | 102.2s | 6.3s | success |

- 22 frames is out-of-distribution (1/5 of the official 5-second minimum), yet quality
  holds up, and audio is non-silent too
- **Lesson**: during the probe, the 5-frame exception occurred before the post-decode
  cleanup (transformer reload), and a chain of subsequent-request OOMs from ~98.5GB of
  leaked residency was observed -> reflected into the production implementation as the
  cleanup path described in item 3 above

## 2026-08-09: H3_KEEP_TRANSFORMER — eliminates the per-request reload fixed cost by keeping the transformer resident

`H3_LOWVRAM=1` frees the transformer right before decode and reloads it **every single**
request (measured fixed cost of 14.8-32.7s, `docs/RESIDENCY.en.md` §5.5). `H3_KEEP_TRANSFORMER=1`
skips this freeing, keeping the transformer resident on the GPU and carrying it over to the
next request, converging this fixed cost down to a single occurrence at the initial load.

**All three preconditions are mandatory** (enforced by an import-time guard in
`core/runner.py`; a `RuntimeError` is raised if any is missing). The VRAM-budget reasoning:

1. `H3_LOWVRAM=1` (raw `"1"` only; `group` is a different design that keeps transformer
   host-RAM-resident with block offload, and is out of scope)
2. `H3_TE_DEVICE` set (keeps TE resident on a separate GPU) — **without this, the encode
   phase breaks down first**: TE-nf4 17.45GB + resident transformer-int8 34.3GB = 51.75GB
   exceeds the effective budget
3. `H3_VIDEO_VAE_FP16=1` — the encode phase can avoid the issue by offloading TE to
   another GPU, but if the decode phase is left at its fp32 decode peak of 16.29GB, then
   transformer 34.3 + 16.29 = 50.6GB exceeds the effective budget (~49.8GB). With the VAE
   fp16-ified, the decode peak drops to ~11.4GB, so 34.3 + 11.4 = 45.7GB fits (the quality
   impact of video VAE fp16 is already measured at PSNR **39.97dB** with no visible
   difference — see the "fp16-ifying the video VAE" section for details)

Default (`H3_KEEP_TRANSFORMER=0`) leaves behavior unchanged. **ref2va (`transformer_ref`)
is out of scope** — it remains outside this flag and continues to be freed every time as
before.

```bash
# Recommended launch command (48GB GPU0 + 20GB GPU1)
H3_LOWVRAM=1 H3_TE_PRUNE=1 H3_TE_DEVICE=cuda:1 H3_VIDEO_VAE_FP16=1 H3_KEEP_TRANSFORMER=1 \
  venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
```

**Real-hardware E2E measurement (2026-08-09, 48GB GPU0 + 20GB GPU1, the launch
configuration above)**:

- transformer (int8) is loaded exactly once, on the first request (32.0s). No reload on
  subsequent requests (confirmed in server logs)
- t2i turbo 4 steps, steady state: **9.7s/image** (denoise 4.32s, decode 1.5s). Peak VRAM
  41.97GB (during denoise)
- t2i steps=30, steady state: 51.1s (denoise 45.7s). Peak 41.97GB
- t2va 5s turbo 4 steps: **44.2s** (denoise 26.05s, decode 10.81s). **Peak VRAM
  44.15GB = the decode phase** (transformer 34.03GB resident + fp16 decode). Derived
  prediction was 45.7GB, measured 44.15GB, versus the cataloged 48.9GB with ~4.8GB
  headroom
- nvidia-smi measured peak 42,620 MiB (1-second sampling; torch's measured 44.15GB is the
  authoritative instantaneous peak)
- Same-seed output equivalence: PNG MD5 **exact match** against the flag-OFF baseline
  (all else equal) (seed=11, md5 `665eadddea8f34298a1b5b89e69d4bd0`). Baseline side:
  total 63.27s (including transformer load) / peak 36.4GB

**Speedup lineage** (48GB GPU0, `H3_LOWVRAM=1` family, t2va at 768², 5s):

| | t2i | t2va 5s |
|---|---|---|
| No turbo, 30 steps (bare configuration right after GPU swap, 08-07) | 157s | 351.4s (denoise 197.7s) |
| lightx2v turbo 4 steps introduced (08-07 morning) | 157s | 143s |
| + `H3_TE_PREQUANT` | 83.2s | — |
| + `H3_TE_DEVICE` | ~35s | 60.5s |
| + `H3_KEEP_TRANSFORMER` (this section) | **9.7s** | **44.2s** |

t2va is **8.0x** the bare configuration (351.4s -> 44.2s). Turbo alone gives 2.6x (denoise
shortens 7.6x, but the ~110s load fixed cost remains); the rest is the contribution of
eliminating fixed costs (`H3_TE_PREQUANT`/`H3_TE_DEVICE`/`H3_KEEP_TRANSFORMER`). Note that
in this section's configuration, non-turbo 30 steps for t2i is 51.1s (measured above) —
against the bare 157s, eliminating fixed costs alone gives a 3x speedup even at the
unchanged 30 steps.

See `docs/RESIDENCY.en.md` §5.5 and §5.6 for the detailed VRAM-budget derivation and the
phase x residency table.

## 2026-08-09: Migration to the PR #14355 merged version (f37ab93), stage 1 — t2i/t2va/batch, identical-seed MD5 exact match

Branch migrate-pr14355. Ported runner.py's t2i/t2va/still_batch/hires/turbo/FBC paths to
the merged version's new contract (see the impact investigation in §1), and swapped the
venv to f37ab93 (`--no-deps --force-reinstall`, other dependencies unchanged). **The
ref2va family is deferred to stage 2** (guarded in both app.py/runner.py, returning 503
with a clear message).

**Regression results (all compared at the same seed against the baseline recorded at
abc5e9b)**:

| Path | Result | MD5 |
|---|---|---|
| t2i turbo 4 steps seed=11 | steady 9.24s (9.7s before migration) | **PNG exact match** |
| t2i turbo 4 steps seed=12 | 9.24s | **PNG exact match** |
| t2i 30 steps seed=1 | 49.65s | **PNG exact match** |
| t2va 5s turbo seed=21 | 41.54s, peak 44.15GB (same value) | **MP4 exact match** |
| t2i_batch 2 scenes seed=11 | 16.47s (8.2s/image) | scene 1 = **exact match** with the single-shot seed=11 run |

As the impact investigation concluded that the numerical path (scheduler/transformer/VAE)
is unchanged, **the migration is bit-equivalent**. Performance and VRAM peak also match
pre-migration values exactly.

**Two new pitfalls hit during the port (both variants of the `_execution_device` issue)**:

1. **`audio_vae` moved ahead of `transformer` in the component order** (old:
   `text_encoder, tokenizer, processor, vae, scheduler, audio_scheduler, transformer,
   video_processor, audio_vae` -> new: `image_processor, text_encoder, tokenizer,
   processor, vae, audio_vae, scheduler, audio_scheduler, transformer_ref, transformer,
   video_processor`). Since `_pin_execution_device_to_compute()` only detached
   text_encoder and vae, the CPU-resident audio_vae was picked up as the first nn.Module,
   and the new layout step (whose contract places output on `_execution_device`) built
   position_ids on the CPU, causing a device mismatch inside rope() (reproduced on real
   hardware). -> fixed by also detaching audio_vae for that window
2. **In the batch's phase reordering, the layout stage's `_execution_device` becomes
   unresolvable** (with TE externally resident, transformer is not yet loaded and TE has
   already been detached) -> tensors end up created on the CPU. Since the values
   themselves are correct, explicitly move them to the compute GPU right before denoise via
   `_scene_state_to_compute()` (a no-op if already on the GPU)

## 2026-08-09: Merged-version migration stage 2 — the ref2va family also completes with identical-seed MD5 exact match

Continuation of stage 1 (above). Ported ref2va / ref2i / ref-batches to f37ab93's new
contract, and compared against **the baseline taken at the old pin abc5e9b** (temporarily
reverted to main + abc5e9b to record it), using the same seed, same reference images, and
same configuration (`H3_LOWVRAM=1 H3_TE_PRUNE=1`, no TE_DEVICE):

| Path (steps=8) | Baseline | After migration | MD5 |
|---|---|---|---|
| ref2i seed=101 | 206.1s | 201.4s | **PNG exact match** |
| ref2va 5s seed=102 | 332.5s | 322.5s (steady state; the first run's 496.8s was a cold disk cache) | **MP4 exact match** |
| ref2i_batch 2 scenes seed=101 (KV prefix-cache path) | 257.9s | 246.7s | **both scenes' PNGs exact match** |

**Two design decisions**:

1. **Consolidated the pipe shell into one**. The merged version's `MiniMaxH3Blocks` has
   merged component specs combining sub-blocks from all 3 workflows (t2va/fl2va/ref2va),
   with both the `transformer_ref` and `transformer` slots present in the same shell
   (`MiniMaxH3AutoDenoiseStep` branches from state **per call**). The old two-shell
   design's `_ensure_pipe_ref_shell` / `_sync_shared_components_to_ref` were reduced to
   no-ops, keeping only the `self._pipe_ref = self._pipe` alias — the existing VRAM
   choreography (the many places touching `_pipe_ref.transformer_ref`) survives unmodified
2. **The KV-prefix-cache split point is unchanged**. Since the new
   `MiniMaxH3Ref2VATextEncoderStep` also assembles the prompt as an `emit(text(prompt))`
   at the end of the presentation, the old design of "share the reference prefix + encode
   the prompt tail as a continuation" still holds as-is. Rebuilt on top of the new step's
   own instance methods (`_gather_vision_features` / `_build_presentation`) in place of the
   removed packing_ref2va function family (the DynamicCache / rope_deltas / continuation
   call argument conventions carried over unchanged from the old implementation's pitfall
   notes)

The rest of the port follows the same pattern as stage 1 (inserting AfterDenoiseStep,
inserting the new `MiniMaxH3PrepareConditionLatentsStep` /
`MiniMaxH3Ref2VAPrepareLatentsStep` steps, `reference_kind()` -> `entry.kind` /
`entry.has_audio`, moving app.py's reference construction to the
`MiniMaxH3ImageReference.from_file()` family). Since the new SetupStep no longer derives
duration from an audio reference internally when `seconds=None`, this was implemented on
the runner side as `_num_frames_from_audio_reference`.

This makes **all paths bit-equivalent on the merged version f37ab93**. There is no longer
any reason to revert to the old pin abc5e9b.

## 2026-08-09: Two-row tab layout (top row = video, bottom row = still image) and making Ref2I its own tab

To keep the tabs from growing into an ever-longer single row, **split them into two rows
by output type**:

| Row | Tabs (input in parentheses) |
|---|---|
| **Video** | T2VA (text) / FL2VA (frame) / Ref2VA (reference) |
| **Still image** | T2I (text) / Ref2I (reference) |

The row heading represents "output," and the tab name represents "input," so row x tab
reads as "video x reference = Ref2VA," "still image x reference = Ref2I."

**Why Ref2I was made its own tab**: it used to be a "still image" checkbox inside the
Ref2VA tab (only T2I had its own tab). This asymmetry stemmed from the fact that "putting
a still-image checkbox on the T2VA tab would conflict with that tab's keyframe input
fields (`still=True` cannot be combined with image/last_image — raises ValueError in
runner.py)," but **once split into two rows, it becomes more unnatural for "the still-image
row to have no reference-conditioned mode"**, so it was made independent. The reference
upload field is shared with Ref2VA (`isRefMode()`); the only differences are how duration
is decided (seconds vs. frame count) and the `still` flag.

**Batch is not made into a tab**: batching is **orthogonal** to the still-image/video axis
(Ref2I+batch = `/api/ref2i_batch`, Ref2VA+batch = `/api/ref2va_batch`, T2I+batch =
`/api/t2i_batch`). Turning it into a tab would balloon the layout to 3 columns x 2 rows, so
it remains a checkbox within each tab.

**Tab buttons use a 2-line structure** (line 1 = model name, line 2 = input type,
`.tab-btn small`). At a panel width of 378px, the button's inner width is 77px, while
"T2VA (text)" is 95px and would wrap, making button heights uneven, so it was designed as
2 lines from the start, independent of width (measured in-browser: all buttons align at
50px, and line 2's position also aligns at y=26px).

### Lazy-loading gallery thumbnails

A problem where accumulating outputs made page rendering heavy (measured: at 165 outputs,
browser rendering choked and screenshots timed out at 30 seconds). The cause is that
**`<video preload="metadata">` fetches metadata for every tile all at once, as soon as it's
displayed**.

`<img>` respects `loading="lazy"`, but **`<video>` does not honor the same attribute**, so
the src is evacuated to `data-src`, and set only when the element approaches the viewport
via `IntersectionObserver` (`rootMargin: 300px`, pre-loading one screen ahead). On every
re-render, `disconnect()` is called before re-observing (since the observer holds
references to elements it's watching, failing to detach it would keep old tiles from being
freed).

**Measured (browser network log)**: mp4 requests on page load went from **165 to 0**. PNGs
(55 of them) are left to the browser's standard lazy loading.

### UI language switching (i18n)

The toggle at the top right (`日本語` / `English`) switches **without a page reload**. The
choice is stored in `localStorage['h3_lang']`; the first visit is decided by
`navigator.language`.

- **One dictionary** (`I18N = { ja, en }`, 165 keys). Lookups go through `t('key', {vars})`,
  and a missing key falls back to Japanese
- Static text is marked with `data-i18n` / `data-i18n-html` (for text containing `<b>`) /
  `data-i18n-placeholder` / `data-i18n-title`, and `applyI18n()` rewrites them in one pass
- **Text generated by JS does not follow an attribute rewrite**, so on switching,
  `rerenderDynamicI18n()` **re-renders** the gallery, the result panel, the reference tiles
  and the various hint lines (the most recent result is held in `lastResultData` and passed
  back to the same render function)
- Splitting the HTML into two per-language files was rejected: the UI keeps changing, and
  double editing guarantees one side rots

**Not translated**: JS comments (103 lines, design notes for developers), server-returned
error messages (`data.detail`; app.py stays Japanese), technical identifiers such as
`te_quant:`, and the language toggle's own labels.

**Verified in the browser**: switching JA↔EN also swaps gallery tiles (`静止画` ↔ `Still`),
the result placeholder and the hint lines; the choice survives a reload; the six submit
routes (`/api/t2i`, `/api/ref2va`+still=1, `/api/ref2i_batch`, `/api/ref2va_batch`,
`/api/ref2va`, `/api/t2va`) are unchanged after switching; and no console errors occur.

## 2026-08-10: Lip sync from an audio reference (`fully_copy`) — measurements, and the bug found along the way

**Motivation**: the technique of replacing the text encoder with a small model + a
projection matrix
([ClipProj-MiniMax-H3](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3)) can shrink
the TE from 15.7GB to 5.2GB, but by the author's own measurements **speech degrades** (4B is
unintelligible, 8B drops the language and speaks English). So the hypothesis: if the dialogue
is fed **from an audio reference, it no longer depends on the TE's speech ability**. This
section verifies that.

**Result: the hypothesis holds.**

| Verified item | Result |
|---|---|
| Does the dialogue content carry over from the audio reference | **Holds** — detected language ja (0.964), recognized "今日は良い天気だね", **100% character match** |
| Lip sync | **Holds** — correlation between mouth opening and the audio envelope **+0.745** (offset 0ms) |
| Character consistency | Holds (face, hairstyle, uniform as in the reference) |
| 1:1 copy of the audio | **Does not hold** — waveform correlation 0.112, duration 5.22s → 5.88s |

The decisive point is that **not a single character of Japanese dialogue was written in the
prompt** (no `<d>` tag; the only instruction was "match the mouth to the dialogue of
`<Audio 1>`"). The same Japanese came out anyway = **the dialogue does not pass through the
text encoder**.

**Despite its name, `fully_copy` is not a signal copy.** The official guide says *"The
complete source audio serves as the target video's complete final audio track"* / *"reused
1:1"*, but the actual behavior is "**regenerate the same content**". The voice timbre
changes, and the duration is decided by the model's own constraints (141 frames = 5.88s). If
you want the original audio verbatim, you have to swap it in after generation (with +0.745
sync correlation this looks practical; unverified).

Also, the mouth is open for 4.1s against 1.6s of actual audio — the mouth tends to keep
moving after the speech ends. Acceptable for anime-style footage, but not a strict
phoneme-level match.

Conditions: 96GB box, TE nf4 + transformer bf16 (unquantized), 768×1344, 141 frames,
30 steps, seed 777, total 553.9s, peak 87.67GB.

### [Bug fix] ref2va with an audio reference always crashed under sage attention

This fired during the verification above. **Passing a reference containing audio crashes
reliably**:

```
sageattention/core.py: assert dtype in [torch.float16, torch.bfloat16]
AssertionError: Input tensors must be in dtype of torch.float16 or torch.bfloat16
  (origin: autoencoder_kl_minimax_h3_audio.py -> dispatch_attention_fn)
```

**Cause**: `MiniMaxH3AudioAttnProcessor` calls `dispatch_attention_fn` with
`backend=self._attention_backend` (default `None`), so **the backend is resolved
globally**. This app only calls `set_attention_backend()` on transformer /
transformer_ref, but with `H3_ATTN_BACKEND=sage` (the default) even the audio_vae's
attention flows into sage. The audio_vae, however, is **fixed to fp32 by design** (bf16
drops the volume by about 20dB), and sage only accepts fp16/bf16. **The per-request
`attn=` override cannot work around it either** (it only affects the transformer family).

**Fix**: call `audio_vae.set_attention_backend("native")` right after loading the
audio_vae, pinning just this module to native. The audio VAE's compute is small and sage
gains nothing there.

**This was a hole in the tests**: the path is only exercised by "references with audio",
so every existing ref2va regression (**48GB box, int8, image references only**) slipped past
it. The unquantized ref2va on the 96GB box had also gone unverified since the merged-version
migration. The fix was confirmed both by ref2va with an audio reference passing with sage
still the default (186.3s, Japanese ja 0.976, 100% character match) and by image-only ref2i
remaining intact (133.9s).

## 2026-08-10: Projected TE (Qwen3-VL-4B + a trained linear map) — implementation, measurements, unverified items

Implemented as `H3_TE_PROJ` (default OFF; existing behavior does not change by a single
byte). It replaces the TE path from Qwen3-VL-32B with **Qwen3-VL-4B + a trained projection
matrix** ([ClipProj-MiniMax-H3](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3)). The
aim is to keep the TE and the transformer **simultaneously resident** on the 48GB box,
eliminating the swap fixed cost.

```
cond = ((h - mean_in) / std_in) @ W * std_out + mean_out     # h = the 4B's hidden_states[24]
cond[:, 0] = sink_out                                        # token 0 is the attention sink
```

### Measurements (96GB box, t2i 768², 30 steps, seed 4242, identical prompt)

| | 32B TE (current) | **Projected 4B TE** |
|---|---|---|
| PSNR | — | **22.64 dB** |
| Sharpness (Laplacian variance) | 48 | **37** (-23%) |
| Generation time | 65.7s | **43.7s** |
| Peak VRAM | 88.73GB | **76.1GB** |
| **TE's real GPU occupancy** | 21.02GB (nf4) | **8.88GB** |

**The quality is not "the same picture" but "a different picture of comparable quality".**
Composition, palette and the reading of the time of day match, but fine-grained
specification drops (a real example, outside what the prompt asked for: the foreground
water-lily leaves and grass present in the 32B version were replaced by open water in the
projected version). No breakdown; practical quality. This matches the nature of the
projection (the distributor reports test cosine 0.712): read it as **the gist of the prompt
is preserved, the details are lost**.

**This change is different in kind from every optimization so far.** Quantization,
residency control, turbo etc. could all be proven "mathematically inert" by identical-seed
MD5 match, but the projected TE is **an approximation by principle**, so MD5 is unusable.
The verdict rests on PSNR + visual inspection + measured VRAM.

### Tokenizer investigation (the basis of the implementation policy)

| Target | Result |
|---|---|
| Ordinary text, Japanese | **IDs match exactly** (the H3-side tokenizer can be used as-is) |
| `<d>` / `</d>` | H3 has them as **single tokens** 151669/151670; the 4B lacks them and splits them apart |
| `<\|cutoff\|>` `<\|lyrics_*\|>` `<\|caption_*\|>` | H3-specific (151671-675) but **unused by both the official guide and this app** |
| `[Shot n]` `<cutoff>` `<scenetrans>` | **Ordinary text** (not special tokens). Multi-shot structure is unaffected |

→ The only thing affected in real use is **the dialogue tag `<d>`**. The implementation
**explicitly rejects** prompts containing id >= 151669 with a `ValueError` (never silently
sends something different). Dialogue can be fed via an audio reference (`fully_copy`), as
demonstrated the same day (see the section above), so an operational workaround exists.

### The second-GPU requirement drops

Plugging the measured 8.88GB back into the derivation in `docs/RESIDENCY.md` §5.4, the card
requirement for `H3_TE_DEVICE` drops from **20GB to 16GB (probably 12GB)**.

| 2nd GPU | Effective budget | 32B TE (17.45GB) | Projected 4B TE (8.88GB) |
|---|---|---|---|
| 12GB | ~10.5GB | No | **Expected to hold** (needs ~9.2GB, ~1.3GB headroom) |
| 16GB | ~14.5GB | No | **Expected to hold** (~5.3GB headroom) |
| 20GB | ~19.7GB | t2va only; ref2va OOMs | Holds (~10.5GB headroom) |

**12GB is not asserted without a real-hardware check.** We previously experienced "fits by
derivation, OOMs in practice" at 20GB (the units pitfall, §5.1), and 1.3GB of headroom is
thin.

### Addendum (same day): the NF4 quantization option, and quality confirmation on video

Added **`H3_TE_PROJ_QUANT`** (default `none` / `bnb-4bit` / `bnb-8bit`). It quantizes the
4B itself — a **4B-only flag**, distinct from the 32B's `H3_TE_QUANT` (see the intent of the
exclusivity guard).

**Using the bf16 projection matrix as-is is the right call** (measured). The conditioning
drift from quantization is 0.61-0.96% relative RMS under NF4 (cosine 1.0000). Using the
distributor's int8_convrot matrix actually increases the drift (1.02-2.97%) — that one is
calibrated specifically for ComfyUI's quantization scheme.

| Config | TE resident | t2i (30 steps) | t2va 5s (30 steps) | Peak (t2va) |
|---|---|---|---|---|
| 32B TE | 21.02GB | 65.7s | 162.1s | 91.9GB |
| Projected 4B bf16 | 8.88GB | 43.7s | — | — |
| **Projected 4B NF4** | **3.11GB** | **33.5s** | **143.5s** | **74.3GB** |

**Quality confirmed on video too** (768², 5s, 124 frames, identical seed 555): the
sharpness drop seen on stills (-23%) does not appear on video (187 vs 191), and **flicker
(second-order difference) is actually 11% lower** (7.51 → 6.67). Direct bf16 vs NF4
comparison is PSNR 34.45dB — quantization has essentially zero effect. The PSNR 14.98dB
against the 32B is not "degradation" but "a different take on the same instruction" (the
32B version has an approaching passer-by, the projected version has the girl alone — a
divergence of interpretation).

**Relation to the settings API**: the projected TE is currently **env-only**
(`H3_TE_PROJ`/`H3_TE_PROJ_QUANT`). Attempting to change te_quant/te_prune via
`/api/settings/apply` slipped past the runner's import-time guard and had a hole where
**nothing was applied yet the snapshot reported a change**, so with the projected TE
enabled these fields are now rejected with 400 (read-only `te_proj`/`te_proj_quant` were
added to the `/api/settings` snapshot).

### Addendum (same day, part 2): NF4 made the default, switchable from the UI

- **Changed the default of `H3_TE_PROJ_QUANT` to `bnb-4bit`** (based on the measured
  quality; none/bnb-8bit remain selectable explicitly). Along with the default change, the
  "quantization specified + projection OFF" guard now only fires **when explicitly set
  (`in os.environ`)** — so ordinary users on the defaults are not knocked over by mistake
- **The projected TE can now be toggled from the UI's reload-settings panel**
  (`te_proj` / `te_proj_quant` on `/api/settings/apply`). While ON, the te_quant/te_prune
  controls are disabled and the API also rejects them with 400 (the check is based on the
  "post-apply values": turning it OFF while changing te_quant is legal)

**Round-trip reload E2E (measured, 96GB box)**: OFF→ON 47.9s / ON→OFF 29.4s / re-ON 27.3s.
At each step the PNG MD5 of an identical-seed generation was compared: **the UI-path ON
matches the env-path NF4 exactly**, **OFF matches the 32B baseline exactly**, and **re-ON
matches the first ON exactly** (no state residue; the projection matrix reloads correctly).
Changing te_quant while ON returns 400.

### Unverified (important)

- **TE + transformer simultaneously resident on the 48GB box** — **the main prize**.
  8.88 + 34 (int8) = 42.9GB is expected to fit but is unmeasured. Only once this holds does
  the adoption pay off (on the 96GB box both fit anyway, so no difference shows)
- **The reference path (ref2va)** — the projection matrix was **calibrated on text only**,
  and whether vision features map correctly was unknown. The implementation is wired to
  work, but emits a one-time `logger.warning`
  → **Visually verified on 2026-08-11: references clearly take effect** (face, hair,
  clothes, accessories all reflected, on par with the 32B ground truth; see the section
  "2026-08-11: How far do ref2va/i2va go"). The warning is kept as a record of the
  calibration fact
- **Combination with an audio reference + `fully_copy`** (does the feed-dialogue-from-audio
  workflow still hold with the projected TE)
- **The 4B's TE load of 80.5s** (including the first download). Steady-state load time is
  unmeasured

## 2026-08-10: Breaking down the decode-phase peak VRAM — the composition, and the 15% we took from it

To decide whether to port ComfyUI's
[PR #15446](https://github.com/Comfy-Org/ComfyUI/pull/15446) (chunk-streaming the H3 VAE so
the decode VRAM becomes duration-independent), **we first broke down what our own
decode-phase peak of 16.29GB is made of**.

### The breakdown (768×1344, 107 frames, fp32, measured)

| Component | Measured | Share |
|---|---|---|
| **The weights of the two VAEs (resident)** | **11.02 GB** | 66% |
| video decode activations | 3.08 GB | 19% |
| `postprocess_video` | 0.00 GB | 0% |
| audio decode | 0.00 GB | 0% |
| **uint8 conversion + CPU transfer** | **2.49 GB** | 15% |
| Total | **16.59 GB** | (nearly matches the README's measured 16.29GB) |

**What we learned**: two-thirds of the peak is **weights**, and chunking removes not a
single byte of that. What PR #15446 targets is the 3.08GB of activations — **porting it
saves at most 19%**.

That the decode peak scales with duration was confirmed separately (latent 32 → 48 frames
takes 3.08 → 4.80GB, linear at about 30MB/frame). So the PR's observation itself is
correct, and it helps more at longer durations.

### The 15% taken first — the uint8 conversion's intermediate tensors

Only the breakdown revealed that **our own code was stacking up 2.49GB**:

```python
frames_uint8 = (video_tensor.permute(0,2,3,1).float().clamp(0,1) * 255).round().to(torch.uint8).cpu().numpy()
```

`float()` / `clamp` / `*255` / `round()` each return an **intermediate tensor of the full
length**, and finally the uint8 copy is made on top. This was consolidated into
`frames_to_uint8()`, which converts 8 frames at a time and writes directly into the CPU
output array (the same code had existed in 4 places; now one).

| | Peak |
|---|---|
| Current (all at once) | +2.65 GB |
| **Improved (8 frames at a time)** | **+0.03 GB (-99%)** |

The order of operations is identical to before, so the rounding does not change either.
**The PNG MD5 of an identical-seed production generation matches exactly**
(`66a59ff92d653f1284cabe76bdb6501c`) — confirmed.

### Porting PR #15446 is on hold

Against 3.08GB of activations (19%), it would require a monkeypatch replacing the
upstream `_decode` wholesale, incurring a tracking cost. Having taken the cheaper 15%
first, the next candidate with better expected value is **fp16-ifying the audio_vae (part
of the 11GB of weights)** (the current fp32-fixed rationale is the measured "bf16 drops the
volume by about 20dB", but **fp16 is unverified** — bf16 has a 7-bit mantissa, fp16 has 10
bits, and small-amplitude audio can behave differently).

## 2026-08-11: Added an RTX 4060 Ti 16GB to the 96GB box — stage 1 toward the low-VRAM goals (TE onto the second GPU)

An **RTX 4060 Ti 16GB (sm_89) was added as cuda:1** to the 96GB box (RTX PRO 6000). The
final goals are **(A) running on a single 16GB card, and (B) running on a two-card 8GB×2
setup**. As stage 1, "the projected TE (4B NF4) resident on the 4060 Ti" was measured.

Launch: `H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_TE_DEVICE=cuda:1` (no lowvram;
transformer in bf16, resident on GPU0).

| Measurement | Value | Note |
|---|---|---|
| GPU1 resident | **3213 MiB (3.37GB)** | Matches the derived 3.11GB+ε. Does not grow after generation |
| t2i 768² steady state | 43.94s (denoise 13.48s, decode 0.88s) | ~10s more than the previous day's single-GPU 33.5s (see below) |
| t2va 5s 768² | 134.58s (denoise 102.57s), peak 71.19GB | |
| PSNR vs the 32B baseline | **22.36 dB** (sharpness 37 vs baseline 48) | Same level as the old matrix's 22.49dB |
| PSNR vs the old-matrix NF4 (single-GPU) | **39.29 dB** | Essentially the same picture despite recalibration + a different GPU |
| t2i PNG MD5 | `3cd088df882e37547219b9816a217b91` | New matrix + sm_89, so it differs from the old anchor (as expected) |

### Two bugs found and fixed (surfaced by a Sonnet agent's verification)

1. **A rope device-mismatch crash in the combination of plain mode (no lowvram) +
   `H3_TE_DEVICE`**. Detaching the TE makes `_execution_device` fall to the CPU-resident
   audio_vae — a known pitfall — but this combination had never run before, and the
   layout-through-timesteps span of the two non-lowvram branches (catch-all /
   bnb-4bit+fl2va) sat outside `_pin_execution_device_to_compute()`. Fixed by wrapping
   them in the pin only when `self._te_external` (the usual TE-cohabiting path is
   byte-for-byte unchanged).
2. **The projection matrix's default filename 404'd**. The distributor had moved
   `h3_qwen3vl_4b_tap24.safetensors` to obsolete/ and replaced it with the **recalibrated**
   `mmh3-4b-ClipProj.safetensors` (training 1,666 → 5,664 prompts, cos_test 0.711 → 0.717,
   W's cosine 0.9596 vs the old — effectively a different function). The default was
   updated to the new filename. The old matrix survives in the local HF cache (snapshot
   3f762f19); passing an absolute path via `H3_TE_PROJ` reproduces it.

### Observation: steady-state t2i is +10s vs single-GPU

Denoise + decode stay at 14.4s; **the per-request fixed cost is about 29s** (vs about 19s
single-GPU). The main suspects for the difference are the encode running on the sm_89
4060 Ti (NF4 dequant is slower) + the PCIe transfer of the cond. Meanwhile t2va did not get
worse at 134.6s — the longer the duration, the thinner the fixed cost spreads. On the 96GB
box the TE fits on GPU0 anyway, so **there is no practical benefit to using the second GPU
on this box** (the real targets are the 16GB-single and 8GB×2 configurations).

### Next stages (unmeasured)

- **Stage 2 = goal A**: cap GPU0 to a 16GB equivalent with ballast, `H3_LOWVRAM=group` +
  projected TE **cohabiting** (no TE_DEVICE). Derived 3.11+1.4+6.6 = 11.1GB should fit, but
  **the te_proj × group combination has never run**
- **Stage 3 = goal B**: 8GB×2. The TE's 3.41GB fits on GPU1, but GPU0's blocks +
  activations of 8.0GB exceed the effective budget of 7.1GB by 0.9GB — trimming the
  activations via resolution/duration needs consideration

## 2026-08-11: Stage 2 = goal A achieved — t2i / t2va complete on a real RTX 4060 Ti 16GB **alone**

Verified not with ballast simulation but on the real added card alone, via
`CUDA_VISIBLE_DEVICES=1`. Launch: `H3_LOWVRAM=group H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3
H3_VIDEO_VAE_FP16=1 H3_ATTN_BACKEND=default` (sage is an sm_120-only build, so back to SDPA.
No TE_DEVICE = projected TE cohabiting. **te_proj × group ran here for the first time** —
it worked as-is with no fix).

| Measurement | Value | Verdict |
|---|---|---|
| Resident right after startup | 251 MiB (transformer on CPU, group offload) | |
| Peak during t2i | 7.4GB (nvidia-smi 7615MiB) | |
| Peak during t2va 5s denoise | **11.4GB (nvidia-smi 11681MiB)** | **~4.7GB left** on a 16GiB card |
| During t2va decode | ~7.0GB (fp16 decode) | |
| t2i steady state | 498.5s (denoise 479s, **16.5s/step**) | |
| t2va 5s | 1499s ≈ 25 min (denoise 1445s, 49.8s/step) | |

**The derived 11.1GB of VRAM landed almost exactly on the measured 11.4GB, and goal A
(single 16GB) holds with plenty of headroom.** Not a single OOM or error.

### The time is heavy mainly because "this box's second slot is PCIe x4"

group offload streams ~34GB of int8 weights CPU→GPU every step, but the slot holding the
4060 Ti is **Gen3 x4 (effective ~3.5GB/s)**. The transfer alone comes to ~10s/step, which
is consistent with the measured 16.5s/step. **On a proper Gen4 x16 slot the transfer is
~1/8**, so this time is a value of "this box's slot", not "the 16GB card's performance".
The t2va's 49.8s/step is that plus the SDPA compute for 124 frames (sm_89) on top.

One more thing: **the FBC cache is not saving a single step** (`cache_skipped_steps: 0`,
threshold 0.05). In contrast to the PRO 6000 + sage + bf16 trajectory, where most steps
were saved. On the int8+SDPA trajectory the residual apparently never drops below the
threshold. Threshold tuning leaves up to a 2x speedup on the table (unverified; a quality
trade-off).

### Quality: not "degradation" but a trajectory divergence — and prompt fidelity actually improved

PSNR is 7.40dB vs the 32B baseline / 7.43dB vs the previous stage — numerically
catastrophic, but **visually it is a different story**:

- The 32B baseline and previous stage (bf16+sage) outputs: at seed 4242 they had converged
  on an **anime-style sunset lake** (letterboxed). The prompt's "photorealistic" and
  "snow-capped peaks" were not reflected
- This run (int8+group+SDPA): **a photorealistic snowy mountain and a misty morning lake**
  — faithful to the prompt and high quality

So the int8+SDPA combination moved the generation trajectory to a different attractor, and
prompt adherence actually came out better. Cross-config PSNR/MD5 comparisons never were
valid (beyond the known int8 trajectory divergence of ~19dB), so **judge cross-config
quality by eye**.

## 2026-08-11: Stage 3 = goal B achieved — **t2va 5s 768² completes at full resolution on 8GB×2**

Both GPUs were capped to headless-8GiB-card equivalents with
`scripts/vram_ballast.py --target-free-gb 7.9` (GiB) to simulate. Compute side = 4060 Ti,
TE side = PRO 6000 (`CUDA_VISIBLE_DEVICES=1,0` maps them to the app's cuda:0/cuda:1).
Launch is the same env as goal A + `H3_TE_DEVICE=cuda:1` (**te_proj × group × TE_DEVICE
also ran here for the first time** — worked without modification).

| Measurement (8GiB×2) | Value |
|---|---|
| t2i 768² | Success. Peak 6.4GB, total 512s |
| t2va 5s 768² denoise | **29/29 completed** (51.3s/step — the prior prediction of "0.9GB over" was wrong; it fit) |
| t2va 5s 768² overall | **Success**. Total 1534s, decode 35.2s, API peak **7.23GB**, measured workload ~8069MiB |

**No reduction of resolution or duration was needed** (the 640²/3s/512² search ladder went
unused).

### But one thing was fixed: an 838MiB all-at-once fp32 conversion at the decode tail OOM'd → denormalization moved to CPU

The first attempt OOM'd on **the final line of the decode tail** after the denoise had
completed. The last line of the upstream `decoders.py`'s
`MiniMaxH3VideoDecodeStep.__call__`,
`(video.float() * pixel_std + pixel_mean).clamp(0,1)`, converts the full-length fp16
decode result to fp32 on the GPU all at once — at 768², 124 frames that is
124×768×768×3×4B = **838MiB** of temporary allocation (matching the OOM message's "Tried
to allocate 838.00 MiB" exactly). Same family of problem as the all-at-once conversion
crushed by `frames_to_uint8`.

The remedy follows the no-venv-modification rule: a runner-side subclass
(`_cpu_norm_video_decode_step()`, duplicating f37ab93's `__call__` with exactly one
change): **move to CPU while still fp16, then denormalize**. Elementwise fp32
mul/add/clamp round identically under IEEE754 on CPU and GPU (no reductions, no FMA
fusion), so the output is bit-identical — **confirmed by measurement: the identical-seed
PNG MD5 matches exactly (`1a2a136b61234b4917465604ac35cca2`) before and after applying**.
The only added cost is one PCIe transfer of the full-length fp16 ~420MiB. Applied
unconditionally to every path (t2va/t2i/ref2va/batch), lowering the decode-phase peak by a
few full-length fp32 buffers on every configuration (the decode-phase numbers in
docs/RESIDENCY.md will be re-measured at the next update).

### Summary: both final goals achieved

- **Goal A (single 16GB)**: full functionality on a real RTX 4060 Ti 16GB, peak 11.4GB
  (~4.7GB headroom)
- **Goal B (8GB×2)**: compute 8GiB + TE 8GiB completes t2i / t2va 5s 768² (peak 7.23GB)

The remaining caveats are speed only (this box's second slot is Gen3 x4, hence
16.5-51s/step; on Gen4 x16 the transfer is ~1/8) and the FBC not engaging on the int8+SDPA
trajectory (threshold tuning unverified).

## 2026-08-11: How far do ref2va/i2va go — two latent bugs found and fixed, and the low-VRAM boundary settled

The reference family (ref2i / i2va = image-reference ref2va / audio reference / 768×1344)
was verified on the goal A/B configurations. The first pass was a **total wipeout**, but
the cause was not VRAM — it was **two latent bugs that only fire in combinations never
exercised together until today**. After the fixes, the boundary is decided plainly by VRAM
amount.

### Bug 1: `H3_VIDEO_VAE_FP16=1` × any reference always crashes, regardless of VRAM (dtype mismatch)

`H3_VIDEO_VAE_FP16=1` permanently casts the VAE weights to fp16. **Decode** is consistent
because the upstream step raises its own fp16 autocast, but the **encode** side
(`encode_vae_condition` in encoders.py — used by ref2va's references and fl2va's keyframe
conditioning) has no autocast and passes pixels it explicitly promoted to fp32 into
`vae.encode()` → instant death with
`Input type (float) and bias type (c10::Half) should be the same`. It crashes even at
96GB. It stayed latent because every ref2va regression so far had used the fp32-VAE
configuration.
**Fix**: in `_load_vae`, immediately after the fp16 cast, wrap `vae.encode` in an fp16
autocast symmetric with the decode side. Precision is within design
(`encode_vae_condition` already rounds its result down to fp16 itself before returning).

### Bug 2: in group mode, the t2va→ref2va mode switch was permanently impossible (residual pinned RAM)

group offload (use_stream=True) places the ~34GB of int8 weights in **pinned memory**. The
del+gc in `_free_transformer` leaves the pages **held by torch's host-side caching
allocator without returning them to the OS** (`torch.cuda.empty_cache()` is device-side
only), so MemAvailable stays ~34GB short and the subsequent transformer_ref load is
rejected by the RAM guard (which asks for 40GB).
Measured: after freeing, avail was still 38.6GB (RssShmem 45.6GB residual).
**Fix**: in `_free_transformer` / `_free_transformer_ref`, only in group mode, add
`torch._C._host_emptyCache()` (a private API, so guarded with getattr). After the fix,
avail recovers fully to **85.8GB** immediately after freeing, and the t2va→ref2va switch
succeeded in group mode for the first time.

### The boundary after the fixes (measured)

| Test | 8GiB×2 | 16GB single (TE cohabiting) |
|---|---|---|
| ref2i (reference still 768²) | **○** peak 6.69GB, 1059s | (not run — established at 8GB) |
| i2va (768² 5s) | × denoise OOM | **○** peak 9.41GB, ~39 min |
| Audio reference (looped 6.9s → 7.29s generated) | × denoise OOM | **○** peak 11.96GB, ~54 min |
| 768×1344 5s | × denoise OOM | **○** peak 13.37GB (nvidia measured 15.2GB), ~66 min |

- **8GiB×2 goes as far as ref2i**. Video with a reference has a longer sequence than t2va
  (7.23GB) by the reference tokens, and even the shortest 768²/5s cannot fit the denoise
  activations (the required amount is a measured 9.41GB).
- **16GB single runs the entire reference family too**. 768×1344/5s (real peak 15.2GB) is
  the practical ceiling.
- A note on audio references: supplying audio shorter than 5 seconds returns 400 from the
  minimum-duration check (per spec).

### Quality (the first visual verification of the projected TE's vision path): references clearly take effect

Since "the projection matrix was calibrated on text only", vision quality was unknown — but
visually no degradation could be detected: the reference person's face, bangs, hair length,
cardigan color, white ribbon and necklace are all consistently reflected, and the i2va
compositions (profile shots) are nearly identical to the 32B ground truth
`test1_walk_park.mp4`. With an audio reference the lip movement is clear too. The only
blemish is a light compositional drift late in the 768×1344 run (not a breakdown).
Outputs: `outputs/ref2i_1786447720.png` / `ref2va_1786449645.mp4` /
`ref2va_1786452054.mp4` / `ref2va_1786455323.mp4`.

## 2026-08-12: Everything stacked on the 96GB box — t2i 7.65s / t2va 28.56s, and two wasted costs it exposed

With the ballast from the low-VRAM verification removed, the 96GB box (RTX PRO 6000 plus the
second-slot 4060 Ti 16GB) was measured with **every available speedup turned on**, to find
the ceiling.

Configuration: `H3_LOWVRAM=1 H3_KEEP_TRANSFORMER=1 H3_VIDEO_VAE_FP16=1
H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_TE_DEVICE=cuda:1 H3_TURBO_LORA=1` — an int8
transformer resident on GPU0 (with the turbo LoRA applied), the projected TE in NF4 (3.11GB)
resident on GPU1, sage attention (sm_120), fp16 decode. Turbo is toggled per request.

| Setting | t2i steady-state | t2va 5s | Peak |
|---|---|---|---|
| **turbo, 4 steps** (FBC is auto-disabled by turbo) | **7.65s** | **28.56s** | 42.5GB |
| 30 steps + FBC (8 skipped for t2i, 7 for t2va) | 21.41s | 121.5s | 41.8GB |
| 30 steps, no FBC | 27.42s | 155.0s | 42.5GB |
| (reference) 96GB default, 30 steps with FBC — the old baseline | — | about 160s | 92GB |

Breakdown with turbo: t2i = denoise 2.40s + decode 1.07s + about 4.2s of remainder; t2va =
denoise 14.81s + decode 7.20s. **This beats the 48GB box's record** (t2i 9.7s / t2va 44.2s),
and is **5.6x** faster than the old 96GB default (about 160s).

**Note the 42.5GB peak** — the fastest configuration does not come close to filling 96GB. The
speed comes from running with zero fixed costs, not from capacity (the same configuration
fits a 48GB-class card).

### Waste 1: a co-resident projected TE gets freed on every request

The `H3_LOWVRAM=1` choreography loads the TE, encodes, and frees it on every request. That was
designed around the 32B TE (21GB), but **the projected TE is only 3.11GB and still gets the
same treatment, paying a 7.6s load every time** — with 55GB sitting free. Parking it on the
second GPU with `H3_TE_DEVICE=cuda:1`:

| Where the TE lives | t2i steady-state | t2va 5s |
|---|---|---|
| Co-resident on GPU0 (loaded/freed each request) | 15.23s | 35.69s |
| **Resident on GPU1** (`H3_TE_DEVICE=cuda:1`) | **7.65s** | **28.56s** |

Even with everything else identical, **where the TE lives is a 2x difference for t2i**. Note
that the 96GB box's second card is 16GB, so the 32B TE (21GB in nf4, 17.45GB pruned) would not
fit — **this residency only works because of the projected TE**.

### Waste 2: a bf16 transformer is freed and reloaded for 12s every request

Since 96GB can hold a bf16 transformer (66.3GB), it should be possible to avoid int8's
dequantization overhead, so plain mode (no `H3_LOWVRAM`) was measured too. **Denoising is
indeed faster in bf16** (t2i 2.07s vs int8's 2.40s; t2va 14.05s vs 14.81s — 5-14% faster). Yet
it loses on the total:

| Transformer | t2i steady-state | t2va 5s | Peak |
|---|---|---|---|
| int8, resident (`H3_KEEP_TRANSFORMER=1`) | **7.65s** | **28.56s** | 42.5GB |
| bf16, plain mode | 19.9s | 40.0s | 68.9-73.1GB |

The log shows plain mode **freeing the transformer for every decode and reloading it in
11.9-12.3s right after**. That free exists because of the "TE-nf4 21GB + transformer 66.3GB +
VAE 11GB = 98.5GB > 96GB" premise — but **with the TE no longer on GPU0, it is 66.3 + 6.1
(fp16 VAE) = 72.4GB**, which fits, so the free is unnecessary in this configuration. Removing
it should make bf16 the fastest option (roughly t2i 8s / t2va 28s at full-precision weights).
`H3_KEEP_TRANSFORMER` currently requires `H3_LOWVRAM=1`, so the right fix is an equivalent
"don't free" decision on the plain-mode side. **Not implemented.**

## Waiting on external events going forward (backlog, as of 2026-08-06)

### 1. diffusers PR #14355 — **merged (2026-08-05), migration also complete (2026-08-09)**

> **This section is the impact audit written before the migration.** For the actual
> outcome see "Migration to the merged version (f37ab93) — stage 1/stage 2" above (every
> path is equivalent by same-seed MD5, and it is merged into main). It is kept as a
> template for the next time diffusers is bumped.

PR #14355 was **merged at 2026-08-05 17:00Z** (merge commit `f53d552`, the PR's final head
was `f37ab93`). **The venv has been updated to f37ab93.** There is no stable release
including H3 yet (the latest v0.39.0 is from 7/3; H3 arrives starting with v0.40.0), so the
pin stays on a SHA.

**The diff between abc5e9b and f37ab93 spans 27 commits and 70 files**, and as feared,
includes a large-scale refactor via `8ab3662` (review & refactor, #14371). On 2026-08-09,
an impact investigation cross-referencing every diffusers touchpoint in runner.py
(15 imported modules, about 40 symbols) against the merged version's source found the
following:

**Points that break immediately (ImportError, 6 import sites)**:
- `packing.py` / `packing_ref2va.py` were **removed**. The runner imports
  `MINIMAX_H3_TEXT_ENCODER_LAYER` (-> became the `components.text_encoder_layer`
  property, default 50), `MINIMAX_H3_TEXT_TAG` (-> moved to modular_pipeline.py),
  `MINIMAX_H3_KEYFRAME_NOISE_AUG` / `MINIMAX_H3_MIN_DURATION` (-> became the
  `components.keyframe_noise_aug` / `.min_duration` properties — **the "monkeypatch a
  module constant" approach no longer works**), `build_packed_sequence` /
  `build_row_timesteps` / `patchify_video_latents` (-> moved to before_denoise.py),
  `unpatchify_video_tokens` (**no replacement** — needed by the hires path, requiring a
  self-implemented replacement), `build_ref2va_presentation` / `reference_kind` /
  `sample_reference_video_frames` (**no replacement** — redesigned into references.py's
  class hierarchy `MiniMaxH3ImageReference/VideoReference/AudioReference`)
- `MiniMaxH3SetupStep` -> gone (reorganized into `MiniMaxH3ResizeStep`),
  `MiniMaxH3AutoKeyframeVaeEncoderStep` -> reorganized into
  `MiniMaxH3KeyframeVaeEncoderStep` / `MiniMaxH3AutoVaeEncoderStep`,
  `MiniMaxH3TextEncoderStep.encode_prompt` (staticmethod) -> changed to the module
  function `get_qwen3vl_prompt_embeds` (signature also changed),
  `MiniMaxH3Ref2VABlocks` -> **gone from the public API** (merged into a single
  `MiniMaxH3Blocks` + branching inside `MiniMaxH3AutoDenoiseStep` — affects the
  pipe/pipe_ref two-shell design)

**What survived**: the main step classes (`MiniMaxH3SetTimestepsStep` /
`MiniMaxH3LoopDenoiser` / `MiniMaxH3DenoiseStep` / the 2 decode variants / the Ref2VA
denoise/encoder steps) persist under the same names. The `row_timestep_plan` state key
also persists. decoders.py's latents normalization (`latents * latents_std +
latents_mean`, which the H3_TE_DEVICE implementation depends on) also persists in the same
form.

**Additional findings from a diff-level audit (2026-08-09, full-line audit by a Sonnet
agent)**:
- **The numerical path's foundation is unchanged**: scheduling_minimax_h3.py's diff
  (36+/41-) is **docstring formatting only, zero code changes** (confirmed via diff). The
  transformer only had padding-row handling removed (QKV structure/forward signature
  unchanged), and video VAE's `_decode` is byte-identical. -> **the possibility of an
  identical-seed MD5 match remains open**. Regression should check MD5 first, falling back
  to PSNR+visual if it doesn't match
- **The decode contract change ripples through every generation function**: the old
  `MiniMaxH3VideoDecodeStep` received packed sequence rows and unpatchified internally,
  but the new version has the newly-added **`MiniMaxH3AfterDenoiseStep` handle
  unpatchify, with the decode step receiving an already-unpatchified 5D tensor** as its
  contract. All 4 of generate / generate_still_batch / generate_ref2va /
  generate_ref_batch **need an AfterDenoiseStep-equivalent insertion right before decode**
- `build_packed_sequence` / `build_row_timesteps` didn't just move to `@staticmethod`s on
  before_denoise's step classes — **their signatures also changed** (`audio_channels` /
  `audio_tag` / `video_tag` became required arguments) — the hires path's
  `_upscale_block_state_2x` needs a full rewrite
- ref2va is now integrated into `ModularPipeline.from_pretrained(MODEL_ID,
  workflow="ref2va")`'s **workflow mechanism** (changing the premise of the pipe/pipe_ref
  two-shell design. Whether the `transformer_ref` component name persists also needs
  checking). The `_execution_device` resolution algorithm itself (first nn.Module in
  component insertion order) is preserved
- `encode_prompt` disappeared as a staticmethod from both the t2va and Ref2VA sides
  (fl2va is split into `MiniMaxH3FL2VATextEncoderStep`). Since the runner calls it
  directly in order to self-manage `@torch.no_grad()`, the same optimization needs to be
  rebuilt on top of the new module function `get_qwen3vl_prompt_embeds`

**Estimated migration effort: large** (revised upward from the initial "medium-to-large").
It's not just a matter of chasing renames — needed are (a) repointing every import
reference, (b) inserting AfterDenoiseStep before decode x4 functions, (c) rebuilding the
direct encode_prompt call sites x2, (d) converting constant monkeypatches into property
overrides, (e) a full rewrite of the hires path, (f) redesigning ref2va's two-shell design
into the workflow mechanism, (g) reimplementing the KV-prefix cache on top of
references.py. The silver lining is that since the numerical path's foundation is
unchanged, **the likelihood of being able to prove equivalence via per-feature MD5/PSNR
regression after migration is high**.

**Current stance**: keep the current configuration (abc5e9b, all features A/B-tested) as
is for now. If migrating, target the minimal-diff f37ab93 in two stages: (1) t2i/t2va path
-> (2) ref2va path.

Note that the int8 recipe (`TorchAoConfig` + `Int8WeightOnlyConfig`) is already included in
abc5e9b and usable without waiting for the merge (already implemented as
`H3_TRANSFORMER_QUANT=int8`).

### 2. Waiting for a finished release of Turbo LoRA -> **resolved via the lightx2v variant (production-implemented 2026-08-08, see the "Turbo LoRA" section)**

The `H3_TURBO_LORA=1` wiring is complete (see the section above). The then-current LoRA
(Ostris's) was explicitly labeled "demo/preview, still in training," hence default OFF,
and **unusable with int8/low-VRAM** (fused-QKV assumption -> `Int8Tensor` lacks `aten.cat`).

On 2026-08-08, spiked the alternative candidate
**[lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo)**
(`minimax_h3_fl2v_turbo_4step_v0.1.safetensors`, 1.38GB, Apache 2.0, DMD distillation;
Kijai/MiniMax-H3_comfy is a ComfyUI repack of this), and **confirmed by measurement that it
works on 48GB (int8) and reaches usable quality at 4 steps**
(`scripts/probe_lightx2v_turbo.py`, main codebase unmodified).

**The decisive difference: the keys are diffusers-native**. Reading the safetensors header
directly confirmed the keys look like
`transformer_blocks.N.attn.to_q.lora_A.default.weight`, with **to_q/to_k/to_v separated**
(rank 128, 312 modules = 50 blocks x6 + token_refiner 2 blocks x6, covering only attn and
ff, not adaln/final_layer), meaning **`fuse_projections()` is unnecessary**. Since
`torch.cat` is never called, it avoids the main cause of int8 incompatibility that blocked
the Ostris variant. In fact, **all 312 modules applied to the int8 transformer in 0.6
seconds with no exceptions**. `ff.net.0.proj`'s `lora_B` has shape `(28672, 128)`, exactly
matching diffusers' SwiGLU (gated, `dim_out*2`), confirming this LoRA was trained directly
against diffusers' own module structure.

**Pitfall (most important): the strength is ~0.094, not Kijai's documented 0.75**. In the
implementation's approach of applying directly to the raw `B*A`, 0.75 is **too strong,
producing fully-noised output even at 30 steps** (demonstrated at 30 steps to rule out a
step-count issue). Since the ComfyUI side applies with alpha folded in, the hypothesis
that `0.75 x (alpha/rank) = 0.75 x 16/128 ~= 0.094` is the corresponding value matched the
measurements.

| strength | 4-step result | audio rms / peak |
|---|---|---|
| 0.75 (Kijai's documented value as-is) | **fully noise** (same even at 30 steps) | 0.083 / 0.43 |
| 0.15 | good (background slightly soft) | 0.069 / 0.79 |
| 0.10 | good | 0.065 / **1.05 (clipping)** |
| **0.094** (= 0.75 x 16/128) | **best** (fur, pine needles all sharp) | 0.039 / 0.70 |

**Speed (RTX PRO 5000 48GB + `H3_LOWVRAM=1`, 768², 5s, seed 12345, same prompt)**: denoise
**197.7s -> 26.1s (7.6x)**, total **351.4s -> 135.2s (2.6x)**. The smaller reduction in
total time is because the low-VRAM mode's load fixed cost (~110s) remains — combining with
the batch path can amortize this too. Temporal consistency also confirmed via 4-frame
sampling (no breakdown).

**Remaining concern**: audio is **about 5x louder** than baseline (baseline rms 0.0073 ->
0.039). Spectral flatness is 0.24, lower than baseline's 0.34, meaning **it is not white
noise** (there is structure), but depending on intensity the peak can exceed 1.0 and clip.
Would need level checking if implemented. Also, since `token_refiner` is on the int8
exclusion list, it ends up in a mixed state — **transformer_blocks are int8-base +
bf16-delta, while token_refiner is bf16-base + bf16-delta** (no observed harm, since both
apply and generate succeed).

**Production-implemented on 2026-08-08** (see the update in the "Turbo LoRA" section):
(1) the diffusers-native apply function + key-format auto-detection, (2) `_TurboLoRALinear`'s
scale factor (per-format default 1.0/0.094), (3) format-based combination guards (comfy x
int8 remains rejected, group rejects regardless of format). (4) Only applicability to
ref2va remains unverified.

### 3. Unstarted improvement candidates (in priority order, none urgent)

- ~~**Pre-saving quantized checkpoints**~~ -> **implemented for TE** (`H3_TE_PREQUANT`,
  2026-08-08. TE load 53.0s->29.5s, request total -35%). **What remains is transformer
  int8**, which would need about 34GB to save, impractical given this box's disk headroom
  (43GB free). On a machine with more disk headroom, the same technique could shave off
  the transformer's load time (~32.5s) as well
- ~~**Keeping TE resident on a second GPU**~~ -> **implemented on 2026-08-09**
  (`H3_TE_DEVICE`, steady state 78.4s->about 35s = -55%. See the "Keeping text_encoder
  resident on a separate GPU" section above). The following is the record from spike
  time: offloading TE to a separate card lets GPU0 keep the transformer resident, and
  **the fixed cost disappears entirely**. In measurements, the current RTX 4000 SFF Ada
  20GB **works for t2va** (peak 17.76GB / 3.23GB headroom), but **ref2va OOMs** (short by
  1-2GB from running a 2048px reference through the vision tower). A 24GB tier is needed
  to cover ref2va as well. PCIe Gen4 x4 is not an issue (TE is loaded once at startup; the
  only per-request transfer is prompt_embeds at 42MB). Implementation requires careful
  changes around `_execution_device` (this project's single biggest source of pitfalls)
  (`scripts/probe_te_on_second_gpu.py`)
- **`ref2va` × turbo is unverified**: turbo LoRA has only been measured applied to
  `transformer` (the t2va family); applying it to `transformer_ref` has **never been
  tried**. Wiring-wise it should go through the same path, but whether a distilled LoRA
  holds up on a reference-conditioned trajectory is a separate question (the strength
  0.094 validity is also a t2va-only measurement). If speeding up reference-conditioned
  generation becomes desirable, spike this first
- **A 4B-only quantization option for the projected TE (`H3_TE_PROJ`)** → **implemented
  (same day, 2026-08-10)**: added as `H3_TE_PROJ_QUANT` and, after measurement (NF4 3.11GB,
  quality on par), **the default was switched to bnb-4bit** (see "Addendum (same day, part 2)"
  in the dated 2026-08-10 section). What follows is the pre-implementation analysis: the
  initial projected TE loads Qwen3-VL-4B **in bf16, with a real GPU occupancy of 8.88GB** (measured
  2026-08-10; matches the 8.88GB checkpoint). The projection matrix's distributor writes
  "15.7GB → 5.2GB", but **5.2GB does not hold in bf16** (it presumably refers to a
  quantized variant). Quantizing the 4B to int8/nf4 would put it in the 4-5GB class,
  lowering the second-GPU requirement further.
  **Caution**: the current implementation makes `H3_TE_PROJ` and
  `H3_TE_QUANT`/`H3_TE_PRUNE`/`H3_TE_PREQUANT` **mutually exclusive** via an import-time
  guard. That is a measure to "keep 32B-TE settings from leaking onto the 4B", not a ban on
  quantizing the 4B. If implementing, add it as a **separate 4B-only flag** (e.g.
  `H3_TE_PROJ_QUANT`) without breaking the existing guard's intent. Note also that the
  projection matrix must be applied in fp32 (W is 2560×5120 fp32)
- **16GB-tier support** → **achieved (2026-08-11)**: with the projected TE in NF4 (3.11GB)
  co-resident alongside `H3_LOWVRAM=group`, **full functionality was measured on a real
  RTX 4060 Ti 16GB alone** (no TE streaming needed after all). 8GiB×2 also holds up through
  t2va. See the dated 2026-08-11 sections. What follows is the pre-achievement analysis:
  would need streaming execution of TE (flowing it to the GPU block
  by block). The current floor is the pruned TE-nf4's resident 17.45GB (see the section
  above). **With the projected TE this floor drops to 8.88GB**, so the 16GB tier may be
  reachable by a different route (see the recalculation of the second-GPU requirement
  above)
- **torch.compile**: unverified. Compatibility with FBC/group-offload hooks (graph breaks)
  needs checking
- **torchao's C++ kernel**: could eliminate int8 mode's dequant cost (+5s), but requires
  torch>=2.11, carrying a large regression risk for the whole venv (not recommended)

## License

**This repository's code is licensed under Apache License 2.0** ([LICENSE](LICENSE)).
Model weights are not included.

The licenses of the models/weights used must each be followed separately:

| Target | License |
|---|---|
| MiniMax-H3's own weights | MiniMax Community License (free for non-commercial use; commercial use free up to $20M/year revenue, credit required) |
| Turbo LoRA (`larryvrh/MiniMax-H3-Turbo-Lora`) | Apache-2.0 |
| Qwen3-VL-32B (text_encoder) | Follows Qwen's official license |
| diffusers / transformers / torchao / SageAttention etc. | Each package's own license |

## LLM prompt enhancement (added 2026-08-04)

Uses a local LLM (gemma4-31B, OpenAI-compatible `/v1/chat/completions`) to reformat input
prompts into the H3 official guide's format. A local reproduction of the cloud Hailuo AI's
internal prompt-formatting layer.

- Connection target: environment variable `H3_LLM_URL` (default
  `http://127.0.0.1:64650`). Returns 502 if unreachable (does not affect the generation
  feature)
- `POST /api/prompt/enhance` {text, mode, seconds, task, lang}
- Modes (the UI's "LLM enhance" button + mode selector):
  - `storyboard` (default): expands into a multi-shot CUT-timecode format (total duration
    = seconds, 2-3 cuts, hard cuts, subject-identity preservation, per-cut sound
    instructions; focal length restricted to 35/50/65/100mm). This is not the official H3
    documentation's notation but this app's own proposal (see `h3-official` below)
  - `brief`: single-shot elaboration in the official brief format (scene -> subject ->
    action -> camera -> sound -> ending)
  - `h3-official` (added 2026-08-07): strict conformance to the field structure/notation
    of MiniMax's official skill `h3-prompt-writing` (`skills/h3-prompt-writing/` in the
    `MiniMax-AI/MiniMax-H3` repository). Switches the reference guide by `task` (the UI's
    current tab = t2va/fl2va/ref2va): t2va/fl2va use the 3-field format from
    `references/base-en.txt` (`integrated_multimodal_description` /
    `overall_soundscape` / `non_diegetic_music`, the `[Shot N] At MM:SS.SS` cut notation,
    dialogue preserved verbatim via `<d>[language] ...</d>`), while ref2va uses the
    6-field format from `references/ref-en.txt` (`subject_definitions` / `summary` /
    `retention_analysis` / `detailed_description` / `overall_soundscape` /
    `non_diegetic_music`). Output is the official default of English (`lang=en`);
    specifying `lang=ja` can also localize just the narrative body into Japanese (field
    names, `[Shot n]`-style labels, timecode notation, and dialogue inside `<d>` remain in
    English/the original language per the official rules). The reference body text is not
    bundled in this repository (since the source repository has no license notice).
    `venv/bin/python scripts/fetch_h3_skill.py` must be run beforehand to fetch it into
    `skills_cache/` (in .gitignore); if not yet fetched, returns a 400 with the fetch
    command guidance. The system prompt feeds in SKILL.md + the full reference guide
    without summarization (about 18.8KB for t2va/fl2va, about 26.6KB for ref2va)
  - `translate`: a literal English translation without embellishment (preserves CUT
    structure)
- The enhancement result replaces the prompt field (editable), with "revert" restoring one
  generation back. The generated result shows the full prompt actually used in a
  collapsible section (for comparing with/without enhancement, and across modes)
- The UI bundles a hand-writing "prompt guide" cheat sheet (the official brief structure,
  CUT notation, measured cut accuracy of +-1s)
- Verified against a real LLM (gemma4-31B Q4_K_M): confirmed 3 modes (brief/storyboard/
  translate) all follow the format (storyboard also respects total timecode sum and focal
  length constraints, response time 11-17s). As of the 2026-08-07 addition of
  `h3-official`, the LLM server was not connected, so only generation A/B was performed
  (see above); actual response verification via the LLM is still needed separately

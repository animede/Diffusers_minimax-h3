# minimax-h3

[日本語](README.md) | **English**

**Latest: 2026-11-19 18:23 JST** — Windows: an automatic VRAM cap of card size minus 1.2GB keeps torch's pool from spilling past physical VRAM (traced on an RTX 3050 6GB) / pinned-RAM transfers no longer used (always file streaming) / real RTX 4060 Laptop 8GB figures added. Change history in [docs/HISTORY.en.md](docs/HISTORY.en.md).

**MiniMax H3 (Hailuo 3.0) — running the 33B omnimodal model that generates video + stereo audio in a single denoise pass, on consumer-tier GPUs.**
From text, an image reference or an audio reference it produces **video + stereo audio**, and it can also emit still images only (T2I / Ref2I).
It is built on the diffusers Modular Pipeline (PR #14355, pinned to commit **f37ab93**).
**Verified on a single 6GB card, on two 8GB cards, on a single real RTX 4060 Ti 16GB and on Windows 11 + RTX 4060 8GB (16GB RAM)**, and a single 32GB card reaches realtime conversational generation (0.95x real time).

![Main GUI](docs/images/ui_main_en.png)

*A single-page browser UI. Five modes on tabs (video: **T2VA** = text / **FL2VA** = frame / **Ref2VA** = reference; still image: **T2I** / **Ref2I**). The lower section is a gallery of outputs, from which checked videos can be concatenated and exported. Prompts can be enhanced into the official H3 guide format via a local LLM, and FirstBlockCache / Sage / Turbo / quantization / low-VRAM mode are toggled from this panel **without a restart**. Includes a Japanese/English switch.*

## Quick start (Windows, 6 to 16GB of VRAM)

The shortest path for people arriving from social media. Details are under [Installation](#installation) and [Starting](#starting).

```powershell
winget install --id Git.Git -e --source winget
winget install --id Python.Python.3.12 -e --source winget
winget install --id Gyan.FFmpeg -e --source winget
git clone https://github.com/animede/Diffusers_minimax-h3.git
cd Diffusers_minimax-h3
.\scripts\install.bat
.\venv\Scripts\python.exe scripts\prepare_lowvram.py
.\run.bat
```

- Reopen PowerShell after the winget installs. ffmpeg is used for the gallery's length / resolution display and for video concatenation (generation works without it; the log then shows a hint). `install.bat` creates the venv and installs the dependencies (a few minutes).
- `prepare_lowvram.py` prepares the model. It does not fetch 144GB: it downloads the 66GB of weights one shard at a time, quantizes each and deletes it, so **about 60GB of disk** and 16GB of RAM are enough. The time is dominated by your connection (hours). If it stops, rerun it and it continues. Add `--ref` if you also want Ref2VA / Ref2I (video from reference material; +32GB).
- `run.bat` starts the server; open `http://127.0.0.1:8611` in a browser. The first start takes a few minutes to build the projected text encoder's quantized cache. The settings for your GPU size (low-VRAM mode, 4-step turbo) are applied automatically.
- Expect a 768², 5-second clip (4-step turbo) to take 211 s on an A4000 under a 6GB cap (NVMe 3.3GB/s), with a peak of about 2.8GB of VRAM (768²) or 3.5GB (1024²). A 16GB-RAM PC reads the 30GB of weights from the NVMe on every step, so SSD speed matters directly. 6GB cards (RTX 3050 etc.) run 768² too. Real-machine figures (2026-11-19, real machine (Windows 11 gaming laptop, RTX 4060 Laptop 8GB / 16GB RAM, NVMe, 4-step turbo, 5 s): 512² 194 s (denoise 117 + decode 16), 768x512 279 s (191 + 65), 768² 412 s (313 + 85)). The RTX 3050 6GB is verified to work only; its times are not measured.
- Low-VRAM speed is set by two numbers: host-to-GPU transfer speed and SSD read speed. If it feels slow, measure both with `.\venv\Scripts\python.exe scripts\check_bandwidth.py` (expect about 10GB/s pinned transfers on PCIe 4.0 x8, about 5GB/s on x4; under 1GB/s points at the link or BIOS settings).
- If something fails, see the troubleshooting notes in the [Windows](#windows) section and the [Issues](https://github.com/animede/Diffusers_minimax-h3/issues).

## Highlights

- **Runs on consumer GPUs**: block-wise streaming of the int8 transformer (`H3_LOWVRAM=group`), a projected TE (Qwen3-VL-4B + a linear map standing in for the 32B TE), an fp16 video VAE and streaming of the TE / token_refiner / VAE decoder bring 768² x 5 s down to **4.6GB** and 1024² x 5 s to **6.4GB**. With 32GB or 16GB of host RAM the weights are streamed straight from the quantized cache.
- **Fixed costs removed**: the bottleneck was never the denoise but the model load/free. A resident transformer, quantized caches, pinned parking and a KV cache for the reference prefix give t2i 9.7 s / t2va (5 s) 44 s on the 48GB tier and t2i 7.4 s on the 96GB tier.
- **Speed options**: FirstBlockCache, SageAttention (sm_120), the lightx2v turbo LoRA (4 / 8-step distillation), ck-w4a8 quantization and the AdaLN-pruned transformer_ref.
- **Five generation modes plus batches**: T2VA / FL2VA / Ref2VA (ordered references: up to 9 images, 3 videos and 3 audio clips; lip sync from an audio reference) / T2I / Ref2I. Still images and reference-conditioned videos batch up to 24 scenes. **The still-image modes (T2I / Ref2I) do not use the dedicated image model MiniMax-H3-Image: they generate a 22-frame ultra-short video with this video model (MiniMax-H3) and save the center frame as a PNG.** Slower than a dedicated image model, but the look matches exactly when the result is used as an FL2VA first frame or a Ref2VA reference. MiniMax-H3-Image lives in a separate repository, [minimax-h3-image-server](https://github.com/animede/minimax-h3-image-server).
- **LLM prompt enhancement**: an external LLM (OpenAI-compatible API, `H3_LLM_URL`) rewrites the input into the format of the official `h3-prompt-writing` skill, with a verify + repair loop before generation. When the external LLM is unreachable it falls back automatically to a built-in CPU LLM (Gemma 4 E4B, GGUF Q4_0, about 4.4GB, downloaded on first use, thinking off), the same idea as ACE-Step-1.5-fast's easy-music; the built-in LLM is llama.cpp's prebuilt `llama-server` started as a child process, so no compiler is needed and it works as-is on Windows.
- **Quality guarded**: diffusers upgrades and quantized caches are regression-checked by identical-seed MD5.

## Sample

[![t2va 768², 5 s, 4-step turbo, RTX 4060 Laptop 8GB](docs/samples/t2va_768_turbo4_rtx4060laptop.gif)](docs/samples/t2va_768_turbo4_rtx4060laptop.mp4)

*t2va generated on a user's Windows 11 + RTX 4060 Laptop 8GB / 16GB RAM (768², 5 s, 4-step turbo). The GIF is a downscaled, reduced-colour preview; click it for the original mp4 with audio (1MB).*

## Supported GPUs (at a glance)

Measured for t2va 5 s at 768². The detailed rows (80GB / 18GB / 12GB tiers etc.) and the measurement conditions are in [docs/FEATURES.en.md](docs/FEATURES.en.md#vram-configurations-and-residency).

| GPU | Startup flags | Result |
|---|---|---|
| 96GB | (none = default) | peak 92GB / about 160 s |
| 48GB tier | `H3_LOWVRAM=1 H3_TE_PRUNE=1 H3_TE_DEVICE=cuda:1 H3_VIDEO_VAE_FP16=1 H3_KEEP_TRANSFORMER=1` (recommended two-card setup) | t2i 9.7 s / t2va 44.2 s (turbo, 4 steps) |
| single 24 / 32GB | ck-w4a8 + AdaLN-pruned + pinned parking ([details](docs/FEATURES.en.md#oct-2026-update-realtime-conversational-generation-on-a-single-24gb32gb-gpu)) | 320×448, 141 frames in **5.57 s = 0.95x real time** (32GB) / 7.1 s (24GB) |
| 32GB tier | `H3_LOWVRAM=group` | peak 28.7GB / about 280 s |
| single 16GB (real 4060 Ti / A4000) | `H3_LOWVRAM=group H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_VIDEO_VAE_FP16=1` | 4060 Ti: peak 11.4GB (Aug 2026). A4000 (2026-11-17): 30 steps peak 8.0GB / 790 s, turbo 4 steps (`H3_TURBO_LORA=1`) peak 7.0GB / 151 s * |
| single 6GB / 8GB | the row above + `H3_TE_STREAM=1 H3_TE_DIET=1` | 30 steps: 768² 4.62GB / 1024² 6.41GB (272 / 552 s on the 48GB box). Turbo 4 steps (A4000 under an 8GB cap, VAE decoder streamed): 768² 3.08GB / 209 s, 1024² 3.73GB / 376 s; fl2va 3.37GB / 225 s, ref2va 3.02GB / 299 s (all 768², 5 s). `file` streaming (host RAM under 48GB) + turbo under a 6GB cap on the A4000 (LoRA streamed from host RAM): 768² 2.82GB / 211 s, 1024² 3.48GB / 380 s (ref2va measured with the LoRA still resident: 768² 4.13GB / 276 s). **Real machine: RTX 4060 Laptop 8GB / 16GB RAM (Windows 11, NVMe): 512² 194 s, 768x512 279 s, 768² 412 s** (4-step turbo, 5 s; see the note below). RTX 3050 6GB verified to work only |
| 8GB x 2 | single-16GB flags + `H3_TE_DEVICE=cuda:1` | peak 7.23GB * |

\* Times depend heavily on PCIe bandwidth (low-VRAM setups are bound by the per-step weight transfer). On a 4060 Ti (Gen3 x4, sm_89, 30 steps) both the single 16GB and the 8GB x 2 setups took about 25 minutes for t2va 5 s; the same configuration takes 790 s on an A4000 (Gen4 x4) and 272 s on GPU0 of the 48GB box, so expect roughly 4-5 minutes on Gen4 x16. **Since the turbo LoRA (`H3_TURBO_LORA=1`, 4 steps) now works in group mode too, the transfer-bound "runs but slowly" caveat is largely gone**: even on the x4-connected A4000, t2va 5 s takes 151 s (16GB tier), and with the 8GB-tier flags (VAE decoder streamed) 209 s at 768² / 376 s at 1024² (the number of transfers scales with the step count). The 8GB-tier figures were measured on the A4000 under an 8GB allocator cap (`H3_VRAM_LIMIT_GB`); the denoise peak of 3.1-3.7GB also fits a 6GB card. Cards other than sm_120 also need `H3_ATTN_BACKEND=default` (the default sage build is sm_120-only; `H3_PROFILE=auto` sets it when unset). That crossing configurations changes the composition even at the same seed (a trajectory divergence, not degradation) is covered in the [performance overview](docs/FEATURES.en.md#performance-overview). On a 16GB-RAM PC (file streaming) about 30GB of weights are read from the SSD on every step, so SSD read speed matters too (211 s on the A4000 with a 3.3GB/s NVMe under a 6GB cap; on a 500MB/s-class SSD the reads alone take about a minute per step). Real-machine figures (2026-11-19, real machine (Windows 11 gaming laptop, RTX 4060 Laptop 8GB / 16GB RAM, NVMe, 4-step turbo, 5 s): 512² 194 s (denoise 117 + decode 16), 768x512 279 s (191 + 65), 768² 412 s (313 + 85)). The RTX 3050 6GB is verified to work only; its times are not measured.

## Installation

### Requirements

| | Requirement |
|---|---|
| GPU | Minimum: a **single 6GB card** (768²) / **single 8GB** (1024²) / **single 16GB** (all features). **48GB tier** recommended for comfortable use |
| Host RAM | 64GB or more recommended (`H3_LOWVRAM=group` pins ~34GB of int8 weights, so 48GB+ free is needed). Below 48GB the weights are streamed from the quantized cache instead (`H3_GROUP_STREAM_HOST=file`; on a 32GB box 1K runs at the same speed, 512² about 2x slower) |
| Disk | About 145GB (T2VA / FL2VA only) / about 207GB (if Ref2VA is also used). The low-VRAM / low-RAM preparation path (`scripts/prepare_lowvram.py`) keeps only the quantized caches, about 60GB (+32GB with `--ref` for Ref2VA) |
| CUDA | 12.8 series (to match torch 2.9.0+cu128). If building SageAttention, `nvcc` must be the same series |
| Python | 3.12 |
| Other | `ffmpeg` (used for gallery concatenation and info retrieval; generation itself works without it) |

### Steps

```bash
git clone https://github.com/animede/Diffusers_minimax-h3.git
cd Diffusers_minimax-h3
scripts/install.sh
```

`scripts/install.sh` creates `venv/`, installs [requirements.txt](requirements.txt) and finally prints the torch / diffusers / transformers versions. It needs only Python 3.12 and `git`; nothing is compiled against CUDA, so it finishes in a few minutes. Add `--sage` to also install SageAttention (below), and `--local-llm` to pre-fetch the prebuilt `llama-server` for the built-in LLM (Gemma 4 E4B; below. Even without it, the binary is fetched automatically on the first fallback).

<details>
<summary>Installing by hand (same contents as requirements.txt)</summary>

```bash
python3.12 -m venv venv

# PyTorch (cu128)
venv/bin/pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128

# ============================================================================
# [IMPORTANT] Pin diffusers to the verified commit f37ab93 (the final merged form of PR #14355).
#             Do not install the PyPI release or the latest main (see the note below)
# ============================================================================
venv/bin/pip install "git+https://github.com/huggingface/diffusers.git@f37ab93e621d5ce206c9662e8291ca8b67d9c555"

# transformers 5.14.1 or later is required (5.1.0 lacks Qwen3VLProcessor.create_mm_token_type_ids)
venv/bin/pip install "transformers==5.14.1" accelerate==1.12.0 safetensors huggingface_hub

# Video/audio muxing and the Web API
venv/bin/pip install av==16.0.1 fastapi==0.104.1 "uvicorn==0.24.0" python-multipart pillow numpy

# 4bit quantization of text_encoder (required for the default H3_TE_QUANT=bnb-4bit)
venv/bin/pip install bitsandbytes==0.49.0

# Only if using int8 quantization for the transformer (H3_TRANSFORMER_QUANT=int8 / H3_LOWVRAM)
#   0.18+ requires torch>=2.11, so pin to 0.17.0
venv/bin/pip install torchao==0.17.0
```

</details>

> **About upgrading diffusers**: installing it is just the one line above (pure Python, no build; it takes well under a minute with `git` available). Care is needed only **when upgrading diffusers later**: a newer version can change the compute path and therefore the output, so keep the commit pinned. The move to f37ab93 itself was made only after every path (t2i / t2va / batch / ref2va / reference batch) was verified equivalent to the old pin abc5e9b by identical-seed MD5. Any further upgrade should go through the same regression check (`scripts/run_regression.py` with the baselines in docs/internal). Not needed for ordinary use.

### Windows

All dependencies have Windows wheels (torch 2.9.0+cu128 from the PyTorch index, torchao as PyPI's pure-Python wheel, bitsandbytes / av / psutil as PyPI Windows wheels). No compiler or Visual Studio Build Tools is needed. SageAttention has no Windows build and is not used; `H3_PROFILE=auto` sets `H3_ATTN_BACKEND=default` (SDPA) at startup. The OS-specific code paths (RAM readout, file read-ahead hints, temp-file location, the built-in LLM child process) have Windows branches. On Windows the Hugging Face cache is set not to use symlinks (`HF_HUB_DISABLE_SYMLINKS=1`) automatically, to avoid a huggingface_hub race that aborts the first download with `WinError 1314` on PCs without Developer Mode (disk usage is unchanged).

Requirements: Python 3.12 (callable as `py -3.12`), git, an NVIDIA driver for CUDA 12.8. `ffmpeg` is optional (gallery concatenation). If git or Python is missing, install them with winget (then reopen PowerShell):

```powershell
winget install --id Git.Git -e --source winget
winget install --id Python.Python.3.12 -e --source winget
winget install --id Gyan.FFmpeg -e --source winget
```

```powershell
git clone https://github.com/animede/Diffusers_minimax-h3.git
cd Diffusers_minimax-h3
.\scripts\install.bat
```

`install.bat` runs `scripts\install.ps1` regardless of PowerShell's execution policy (Windows does not run unsigned `.ps1` files directly by default). It creates the venv and installs torch (cu128) and `requirements.txt`; add `-LocalLLM` to also fetch the built-in LLM binary. Then fetch the model:

```powershell
.\venv\Scripts\python.exe scripts\prepare_lowvram.py
```

(On a PC with 48GB+ VRAM and 64GB+ RAM, `scripts\download_t2va.py` works as well; see "Preparing a low-VRAM / low-RAM PC" below.) To install by hand, run these in order (PyPI's Windows torch is CPU-only, so torch must come first from the PyTorch index):

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

For starting the server (passing environment variables, per-tier examples) see the Windows part of [Starting](#starting). To run the `.ps1` files directly, either add Bypass (`powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1`) or run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

The built-in LLM (the fallback when the external LLM is unreachable) can be added later even if you installed without `-LocalLLM`. It is fetched automatically on the first fallback; to fetch it ahead of time run the following (line 1 = the prebuilt `llama-server.exe`, about 25MB; line 2 = the Gemma 4 E4B GGUF, about 4.4GB, optional):

```powershell
.\venv\Scripts\python.exe scripts\fetch_llama_server.py
.\venv\Scripts\python.exe scripts\fetch_llama_server.py --model
```

Running `.\scripts\install.bat -LocalLLM` later does the same (already-installed packages are left as they are). Details: [Built-in LLM (optional)](#built-in-llm-optional) below.

> Windows status (2026-11-18): a user's Windows 11 + RTX 4060 8GB / 16GB RAM machine got through install, `prepare_lowvram.py`, startup and **video generation (both 30 steps and 4-step turbo)**. Task Manager during generation showed 6.3 / 8.0GB of dedicated GPU memory and 2.0GB of shared GPU memory (a laptop RTX 4060). The dedicated figure is above the peak the server reports (about 4.2GB) because it includes memory torch keeps reserved for reuse plus the desktop; the shared figure is the pinned host RAM used as the streaming staging area, which Windows counts as shared GPU memory. Windows 10 + RTX 3050 6GB / 16GB RAM also generated a turbo 768², 5 s clip (the Hub cache symlink, source-shard deletion, execution-policy and encoding problems found along the way are fixed). On that machine VRAM overflowed (dedicated 5.8 / 6.0GB, shared 3.4GB) and Windows spilled to shared memory, which makes generation extremely slow without any hang or error. The turbo LoRA (about 1.2GB) is now streamed from host RAM instead of staying on the GPU, bringing the peak from 4.19GB down to 2.82GB (versions from 2026-11-18 15:00 JST on), so it fits in 6GB. 4060 Laptop generation times (after moving to an NVMe, 4-step turbo, 5 s): 512² 194 s, 768x512 279 s, 768² 412 s. In practice 512²-class sizes, or 768x512 for landscape, are the realistic choice. The 3050's times are not measured. The image server built the same way, [minimax-h3-image-server](https://github.com/animede/minimax-h3-image-server), generates fine on Windows 11 + RTX 3050. Please report problems in the [Issues](https://github.com/animede/Diffusers_minimax-h3/issues).

### SageAttention (optional, enabled by default; Linux only)

The default `H3_ATTN_BACKEND=sage` needs a build for sm_120 (there is no prebuilt Linux wheel). **If you do not build it, start with `H3_ATTN_BACKEND=default`** (denoise is about 12% slower; no functional difference).

```bash
CUDA_HOME=/usr/local/cuda-12.8 scripts/install.sh --sage     # or scripts/build_sageattention.sh
```

`--sage` installs a prebuilt wheel from `third_party/wheels/` if one is there, and otherwise builds from source.

### Built-in LLM (optional)

LLM enhancement uses an external OpenAI-compatible API (`H3_LLM_URL`) by default and falls back automatically to a built-in CPU LLM (Gemma 4 E4B, GGUF Q4_0, thinking off) when it is unreachable. The built-in LLM is llama.cpp's prebuilt `llama-server` from the [GitHub Releases](https://github.com/ggml-org/llama.cpp/releases) (CPU build, about 25MB, pinned tag `b12475`, override with `H3_LLAMA_CPP_TAG`): the server starts it as a child process and talks to it over the OpenAI-compatible API (default `127.0.0.1:64661`, `H3_LLM_LOCAL_PORT`). **No compiler is needed**; the Linux x64 / Windows x64 / macOS binary is chosen automatically.

```bash
scripts/install.sh --local-llm            # pre-fetch llama-server into tools/llama/ (optional)
venv/bin/python scripts/fetch_llama_server.py --model   # also pre-fetch the GGUF, about 4.4GB (optional)
```

On low-RAM PCs (16GB class) video generation and the built-in LLM compete for RAM, so llama-server is stopped 300 s after its last use (`H3_LLM_LOCAL_IDLE_S`) and, on hosts under 32GB RAM, also when a generation starts (`H3_LLM_LOCAL_STOP_ON_GENERATE=auto`). The next LLM enhancement restarts it in a few seconds.

Both are optional and can be run later on an existing installation. On the first fallback `llama-server` is fetched into `tools/llama/` and the model into `models/llm/` automatically. The child process stops with the server, and if the server is killed hard it is taken down with it (Windows: Job Object with KILL_ON_JOB_CLOSE; Linux: `PR_SET_PDEATHSIG`), so it never lingers as an orphan (verified with SIGKILL on Linux; Windows untested). To use existing files set `H3_LLAMA_SERVER=<path to llama-server>` and `H3_LLM_LOCAL_GGUF=<path to the GGUF>` (a llama-server you started yourself that answers on `H3_LLM_LOCAL_PORT` is used as-is).


### Obtaining the model

**PCs with 32GB of VRAM or less, or less than 48GB of host RAM, should use "Preparing a low-VRAM / low-RAM PC" below instead** (the steps here fetch the full 144GB, which is wasteful on the low-VRAM tiers, and a low-RAM PC cannot build the quantized cache and stops at startup).

`MiniMaxAI/MiniMax-H3` is 498.6GB, but T2VA / FL2VA only need `transformer/` (66.3GB) + `text_encoder/` (66.7GB) + `vae/` (10.4GB) + `audio_vae/` (0.6GB) + configs, **about 144GB**. Fetch `transformer_ref/` (Ref2VA, 66.3GB) only when needed, and **never `snapshot_download` the whole repo**.

```bash
venv/bin/python scripts/download_t2va.py
```

It uses `allow_patterns` internally to fetch only the needed subfolders; `logs/du_monitor.log` tracks the cache size while downloading. When the bf16 transformer / transformer_ref weights are not cached, they are downloaded only with `H3_ALLOW_WEIGHT_DOWNLOAD=1` (the default fails with a hint).

### Preparing a low-VRAM / low-RAM PC

The `H3_LOWVRAM=group` configuration (32GB tier and below) that streams the weights from the quantized cache files **requires the int8 quantized cache (`models/prequant/transformer_int8`, about 32GB) to exist first**. Without it the server falls back to quantizing from bf16 (66GB) and stops at the 40GB host-RAM guard (this is exactly what happens on a 16GB-RAM PC). `scripts/prepare_lowvram.py` builds that cache by fetching the bf16 shards (14 × 4.7GB) one at a time, quantizing each and deleting it: a few GB of RAM, no VRAM needed (CPU quantization by default, `--device cuda` optional), and disk peaks at one shard plus the output. The result is **bit-identical** to the cache the server would build itself (verified). It also builds the VAE decoder fp16 cache and pre-fetches the projected TE (Qwen3-VL-4B + ClipProj). The 32B text_encoder is not downloaded.

```bash
venv/bin/python scripts/prepare_lowvram.py
```

Windows:

```powershell
.\venv\Scripts\python.exe scripts\prepare_lowvram.py
```

The time is dominated by downloading 66GB (quantization takes seconds to tens of seconds per shard). The cache is checked against the Hugging Face snapshot, but on a PC without the bf16 originals (where a rebuild is impossible anyway) the check is skipped and the cache is used as-is. Reruns are idempotent: an existing cache is only inspected, which takes seconds. `--keep-source` keeps the bf16 shards (+66GB, faster rebuilds). The projected TE's NF4 cache is built by the server on first start (fine with 8GB VRAM / 16GB RAM).

**Add `--ref` if you also use Ref2VA / Ref2I.** The reference modes use a different set of weights, `transformer_ref/` (66GB; every shard differs from `transformer`), so its cache `models/prequant/transformer_ref_int8` (about 32GB) is built the same way. On a host with less than 48GB RAM the reference modes stop at startup without it (the error tells you how to build it). It can be added later (reruns are idempotent; an existing `transformer_int8` is only inspected, which takes seconds).

```bash
venv/bin/python scripts/prepare_lowvram.py --ref
```

Windows:

```powershell
.\venv\Scripts\python.exe scripts\prepare_lowvram.py --ref
```

## Starting

### Linux

```bash
venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
```

With nothing set, **`H3_PROFILE=auto`** picks the 96 / 80 / 48 / 32 / 16 / 8GB-tier defaults (the same flags as the launch examples below) from the compute GPU's total VRAM. On the 32GB-and-below tiers (group) the turbo LoRA (4 steps) is on by default as well, because in group mode it must be applied at startup to be usable from the GUI at all (untick turbo in the GUI for 30 steps, or set `H3_TURBO_LORA=0` to disable it at startup). On 48GB and above, use the GUI's turbo checkbox per request. Setting any of the main profile variables (`H3_LOWVRAM`, `H3_TE_PROJ`, ...) switches to manual mode and nothing is filled in automatically. When the default sage build cannot be used (a GPU other than sm_120, or sageattention not installed) `H3_ATTN_BACKEND=default` is also set automatically. The chosen tier is shown in `/api/status` under `profile` and in the startup log.

Open `http://<host>:8611/` in a browser. The first start takes a few minutes to load the models (by default the TE is NF4-quantized and kept on the GPU). `run.sh` is the launcher used by the diffusers-movie-server integration: it starts the same server on internal port 8631 with the conversation-oriented defaults (turbo LoRA etc.).

Launch examples per VRAM tier:

```bash
# Real 16GB, single card (e.g. an sm_89 4060 Ti; anything other than sm_120 needs H3_ATTN_BACKEND=default)
H3_LOWVRAM=group H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_VIDEO_VAE_FP16=1 \
  H3_ATTN_BACKEND=default \
  venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611

# 6GB / 8GB single card (the 16GB flags plus TE streaming and diet; H3_CACHE=none goes lower still)
H3_LOWVRAM=group H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_VIDEO_VAE_FP16=1 H3_TE_STREAM=1 H3_TE_DIET=1 \
  H3_ATTN_BACKEND=default \
  venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611

# 8GB x 2 (TE on the second GPU)
H3_LOWVRAM=group H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_VIDEO_VAE_FP16=1 \
  H3_ATTN_BACKEND=default H3_TE_DEVICE=cuda:1 \
  venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611

# 32GB tier
H3_LOWVRAM=group H3_TE_PRUNE=1 venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611

# 48GB tier (recommended: full speedups; drop H3_TE_DEVICE when using ref2va)
H3_LOWVRAM=1 H3_TE_PRUNE=1 H3_TE_DEVICE=cuda:1 H3_VIDEO_VAE_FP16=1 H3_KEEP_TRANSFORMER=1 \
  venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611

# 96GB tier (everything resident, no env)
venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
```

The full environment set and caveats for realtime conversational generation on a single 24 / 32GB card (ck-w4a8 quantization + pinned parking) are in [docs/FEATURES.en.md](docs/FEATURES.en.md#oct-2026-update-realtime-conversational-generation-on-a-single-24gb32gb-gpu).

To verify operation before using the UI, run the regression probe (writes `outputs/probe_t2va.mp4` and `outputs/probe_report.json`):

```bash
venv/bin/python scripts/probe_t2va.py
```

### Windows

`run.bat` starts the same server (it just runs `run.ps1` regardless of the execution policy; default 127.0.0.1:8611; `H3_HOST` / `H3_PORT` change the bind address and port):

```powershell
.\run.bat
```

Set environment variables one per line with `$env:` before starting. The Linux launch examples above carry over directly: turn each `H3_X=value` into `$env:H3_X = "value"` and then call `.\run.ps1`. Single-16GB example (cards other than sm_120 need `H3_ATTN_BACKEND=default`; `H3_PROFILE=auto` sets it when unset):

```powershell
$env:H3_LOWVRAM = "group"
$env:H3_TE_PROJ = "NicoLab28/ClipProj-MiniMax-H3"
$env:H3_VIDEO_VAE_FP16 = "1"
$env:H3_ATTN_BACKEND = "default"
.\run.bat
```

With nothing set, `H3_PROFILE=auto` picks the tier from the GPU's total VRAM, so plain `.\run.bat` is a fine first start (on the 32GB-and-below tiers the turbo LoRA is enabled automatically; add `$env:H3_TURBO_LORA = "1"` only if you want turbo as the default on 48GB and above). Variables stay set until that PowerShell window is closed (to start again with a different configuration, remove them with e.g. `Remove-Item Env:H3_LOWVRAM` or open a new window).

If you run `run.ps1` directly and the execution policy blocks it (`run.bat` does this internally):

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

The regression probe:

```powershell
.\venv\Scripts\python.exe scripts\probe_t2va.py
```

## Usage

### UI

A single-page browser UI: inputs and settings on the left, results and the gallery on the right. The "日本語 / English" buttons at the top right switch the display language.

**Mode tabs**

- The upper row is video (**T2VA** = text, **FL2VA** = first / last frame, **Ref2VA** = reference material), the lower row is stills (**T2I**, **Ref2I**). Switching tabs shows only the inputs that mode needs.
- The still-image tabs do not use MiniMax-H3-Image: the video model generates a 22-frame (or 5-frame) ultra-short clip and the center frame is saved as a PNG, with the mp4 kept alongside.

**Prompt and LLM enhancement**

- Prompts work in Japanese as well as English. "Prompt guide" opens a cheat sheet for hand-written prompts: the official brief structure (scene → subject → action → camera → sound → ending), the `CUT n [X-Y s]:` notation and the dialogue tag `<d>`.
- "LLM enhance" rewrites the prompt with the external LLM (or the built-in Gemma 4 E4B when it is unreachable) and replaces the prompt field. Four modes: `storyboard` (expands into multi-cut timecodes), `brief` (single-shot detailing in the official brief format), `h3-official` (strict adherence to MiniMax's `h3-prompt-writing` skill; the current tab is sent as the task, output language English or Japanese, and a verify + auto-repair pass runs before generation with warnings for unresolved issues), `translate` (English translation only). "Undo" restores the previous prompt once.
- The projected TE (the default on the low-VRAM tiers) cannot handle the `<d>` dialogue tag, so prompts containing it are flagged before generation (use an audio reference for dialogue instead).

**Inputs per mode**

- FL2VA: drag and drop the first / last frame image (either one is enough; with both, the clip is generated between them).
- Ref2VA / Ref2I: add multiple reference files (up to 9 images, 3 videos, 3 audio clips, 12 in total). **The order you add them is the reference order**, and each can be referred to in the prompt by the label shown on its tile, such as `<Picture 1>`. With exactly one audio reference the duration can be left blank and follows the audio. Ref2I picks a frame count (22 / 5) and saves the center frame as a PNG.
- T2I: frame count (22 = the quality-verified default, 5 = experimental).
- Batch generation (T2I / Ref2I / Ref2VA): ticking it turns the prompt field into "one line = one scene" (up to 24 scenes); references, seed, resolution and steps are shared across scenes. Model loading and reference encoding are shared between scenes, which pays off especially in low-VRAM mode and with references. Ref2VA batches require the duration.

**Size, duration, steps, seed**

- Resolution: the presets 768² / 1344×768 (landscape) / 768×1344 (portrait), "custom" for an explicit width × height (rounded to a multiple of 32, 256-2048, with a warning beyond the native range), or "auto" for the server default (16:9 for Ref2VA).
- Duration 5-15 s. The actual frame count is rounded up to 17n + 5 and shown as "actual length" (e.g. 5 s → 124 frames, 5.17 s).
- Steps switch automatically to the recommended value (4 / 30) when Turbo LoRA is toggled; a hand-edited value is used as-is. Leave seed blank for random.

**Instant settings (no reload, sent per request)**

- FirstBlockCache (+ threshold), Sage Attention (sm_120 only), Turbo LoRA (4 steps recommended), mute (no audio track).
- Combinations the server's configuration cannot run are greyed out with the reason in a tooltip (for example turbo cannot be selected in group mode unless it was applied at startup). Enabling turbo switches FirstBlockCache off automatically.

**Generation and results**

- "Generate" shows a progress bar with the phase (TE encode / denoise n/N / decode) and elapsed seconds. On completion the right side shows the video (or PNG plus mp4) with mode, resolution, length, steps, seed, denoise / decode times, peak VRAM, audio presence, the instant settings and the full prompt used (collapsible).
- The LLM-enhancement message says whether the external or the built-in LLM handled it.

**Settings that need a reload (collapsible header)**

- transformer quantization (none / int8; ck-w4a8 is startup-only), text_encoder quantization, TE layer pruning, projected TE and its quantization, low-VRAM mode (0 / 1 / group), fp16 video VAE. Changing any of them enables "Apply (reload)" with an estimated time; applying frees and reloads the models without restarting the process.
- Controls are greyed out to match the server's constraints, e.g. text_encoder quantization and TE pruning are disabled while the projected TE is on (the 32B TE is not loaded at all).

**Server status**

- A status block under the inputs refreshes every 4 seconds: load state (transformer / transformer_ref / VAE / TE), GPU allocation and peak, host RAM, `profile` (what the automatic profile chose), the lowvram / quantization / attention / turbo / TE configuration, where the streamed weights live, and the LLM endpoint plus built-in LLM state.

**Gallery**

- Generated videos and stills, newest first. Click a thumbnail for the lightbox view; each item can be downloaded.
- Tick items and use "Delete selected" or "Concatenate selected". Concatenation joins **in the order ticked** (stills are excluded) and saves `outputs/concat_<timestamp>.mp4`.

### API

| Endpoint | What it does |
|---|---|
| `POST /api/t2va` | Text to video + audio. `prompt`, `resolution` (`768x768` / `768x1344` / `1344x768`), `seconds` (5-15), `num_inference_steps`, `seed` |
| `POST /api/fl2va` | The above plus `image` / `last_image` (at least one) |
| `POST /api/ref2va` | Reference-conditioned generation. `references` (up to 9 images, 3 videos, 3 audio clips; order matters). With exactly one audio reference, `seconds=0` uses the audio length. `still=1` + `frames` gives a reference-conditioned still (ref2i) |
| `POST /api/t2i` | Still image. `frames` (default 22, or 5) instead of `seconds`. Saves an ultra-short mp4 and the center-frame PNG |
| `POST /api/t2i_batch` / `POST /api/ref2i_batch` / `POST /api/ref2va_batch` | Batch generation. Repeat `prompts` per scene (up to 24). The reference variants share `references`; `ref2va_batch` requires a common `seconds` |
| `POST /api/prompt/enhance` | LLM prompt enhancement. `{text, mode, seconds, task, lang}` |
| `GET /api/status` / `GET /api/progress` / `POST /api/interrupt` | Load state and VRAM / RAM, progress of the running job, interrupt |
| `GET /api/settings` / `POST /api/settings/apply` | Read and apply (reload) the settings switchable from the UI |
| `POST /api/admin/unload` / `POST /api/admin/reload` | Free and reload the models |
| `GET /api/outputs` / `POST /api/outputs/delete` / `POST /api/outputs/concat` | Gallery listing, deletion and concatenation |

All endpoints take multipart/form-data. Each mode is described in [Generation modes in docs/FEATURES.en.md](docs/FEATURES.en.md#generation-modes).

## Main environment variables

Defaults shown. Most **can also be toggled from the UI**, so set them only to change them permanently. The full list with measurements is in [docs/FEATURES.en.md](docs/FEATURES.en.md).

| Variable | Default | Meaning |
|---|---|---|
| `H3_PROFILE` | `auto` | Automatic main profile. `auto` picks a tier from total VRAM only when none of the main profile variables is set (the turbo LoRA is also enabled on the 32GB-and-below tiers) / `96gb`..`8gb` forces a tier (use this when emulating a smaller card with `H3_VRAM_LIMIT_GB`) / `off` |
| `H3_LOWVRAM` | `0` | `1` = 48GB-tier phase cycling / `group` = block offload for the 32GB tier and below (streams the int8 weights block by block) |
| `H3_TRANSFORMER_QUANT` | `none` | `int8` shrinks the transformer from 66.3GB to 34GB. `ck-w4a8` uses ComfyUI's W4A8 kernel (needs `comfy-kitchen`) |
| `H3_TE_QUANT` | `bnb-4bit` | text_encoder quantization (`none` = bf16, 66.7GB) |
| `H3_TE_PRUNE` | `0` | Remove unused upper layers of the TE (output unchanged, -3.6GB) |
| `H3_TE_PROJ` | (off) | **Projected TE**: replaces the 32B TE with Qwen3-VL-4B + a linear map (repo id or .safetensors path; quality is an approximation) |
| `H3_TE_PROJ_QUANT` | `bnb-4bit` | Quantization of the projected TE (NF4 = 3.11GB; `none` = bf16 8.88GB / `bnb-8bit`) |
| `H3_TE_DEVICE` | (off) | Park the TE on a second GPU (e.g. `cuda:1`; the 32B TE needs a 20GB-class card, the projected TE fits an 8GB-class one) |
| `H3_TE_STREAM` / `H3_TE_DIET` | `0` / `0` | Streaming and diet of the projected TE / token_refiner (the key to the 6GB / 8GB setups) |
| `H3_GROUP_STREAM` / `H3_GROUP_STREAM_HOST` | `own` / `auto` | Weight transfer scheme in group mode. `HOST` streams straight from the quantized cache files (`file`) when the host has less than 48GB of RAM; on Windows always `file` regardless of RAM (WDDM moves pinned RAM pathologically slowly) |
| `H3_VIDEO_VAE_FP16` | `0` | fp16-ify the video VAE (reduces the decode peak; required at 16GB and below) |
| `H3_KEEP_TRANSFORMER` | `0` | Keep the transformer resident, removing the reload fixed cost |
| `H3_PRUNED` / `H3_PRUNED_QUANT` | `0` / `int8wo` | Load the AdaLN-pruned transformer_ref for ref2va (requires `H3_TRANSFORMER_QUANT=int8`) and its quantization scheme |
| `H3_CACHE` / `H3_CACHE_THRESHOLD` | `fbc` / `0.05` | FirstBlockCache (denoise -25%; measured ineffective on the int8+SDPA trajectory) |
| `H3_ATTN_BACKEND` | `sage` | The sage build is sm_120-only. **Any other GPU must use `default` (SDPA)** |
| `H3_VRAM_LIMIT_GB` | (unset) | Cap on the VRAM this process may allocate (GB). **On Windows, unset means an automatic cap of card size minus 1.2GB** (`H3_VRAM_AUTO_CAP_MARGIN_GB`): Windows torch cannot shrink its grown pool, so 6GB cards spill past physical VRAM and crawl; the cap stops the growth (at the cap the allocator releases unused blocks and reuses them). To disable, set a value at or above the card size |
| `H3_GPU_POOL_DEFRAG` | `auto` | Right before the decode, move the live GPU tensors (about 1.2GB) to the CPU, release torch's whole pool, and move them back (`auto` = Windows only: Windows torch cannot shrink its grown pool, so 6GB cards spill during the decode and crawl). `1` / `0` |
| `H3_VAE_DECODER_RESIDENT` | `auto` | In file-streaming decodes, keeps as many VAE decoder blocks (36, 4.5GB) on the GPU as free VRAM allows for the decode and streams the rest from file (every resident block removes its share of the per-pass transfer; same output). A4000 emulation: 30/36 resident under an 8GB cap, 768² decode 86 to 34 s (peak 6.2GB); 15/36 under a 6GB cap, decode 51 s (peak 4.2GB). `1` forces all, `0` disables. Headroom via `H3_VAE_DECODER_RESIDENT_MARGIN_GB` (default 1.0 on Linux, 2.5 on Windows, whose torch pool grows more) |
| `H3_TURBO_LORA` | `0` | lightx2v 4 / 8-step distilled LoRA (downloads about 780MB on first use). Works in `group` too (needs the in-house streamer `H3_GROUP_STREAM=own` and the flag at startup; with `file` streaming the LoRA, about 1.2GB, is streamed from host RAM block by block; tensors missing from the files and at least `H3_FILE_STREAM_HOST_MIN_MB` (default 1) large are handled this way) |
| `H3_ALLOW_WEIGHT_DOWNLOAD` | `0` | Download the bf16 weights (~66GB each) from HF only when `1` if they are not cached |
| `H3_LLM_URL` / `H3_LLM_API_KEY` | `http://127.0.0.1:64660` / none | External LLM for prompt enhancement and its API key (the key is also read from the git-ignored `config.local.env`) |
| `H3_LLM_LOCAL_IDLE_S` / `H3_LLM_LOCAL_STOP_ON_GENERATE` | `300` / `auto` | Memory saving for the built-in LLM: stop llama-server 300 s after its last use (0 = never) / stop it when a generation starts (`auto` = only on hosts under 32GB RAM, `1` / `0` to force). It restarts automatically on the next LLM enhancement |
| `H3_LLM_LOCAL` / `H3_LLM_LOCAL_GGUF` | `auto` / none | Built-in LLM (Gemma 4 E4B, CPU, a prebuilt `llama-server` child process). `auto` = only when the external LLM is unreachable / `1` = built-in only / `0` = never. The GGUF is downloaded to `models/llm/`; point `H3_LLM_LOCAL_GGUF` at an existing file instead. Also `H3_LLAMA_SERVER` (binary path), `H3_LLAMA_CPP_TAG` (llama.cpp tag to fetch), `H3_LLM_LOCAL_PORT` (default 64661) |

## Documentation

| Document | Contents |
|---|---|
| [docs/FEATURES.en.md](docs/FEATURES.en.md) | **Features and settings in detail**: how each environment variable works, its preconditions and measurements, per-VRAM-tier configurations, generation modes, UI, LLM enhancement |
| [docs/TECHNICAL_OVERVIEW.en.md](docs/TECHNICAL_OVERVIEW.en.md) | **Technical overview**: architecture, how the techniques are integrated, handling per VRAM tier, performance, configuration reference |
| [docs/RESIDENCY.en.md](docs/RESIDENCY.en.md) | **Residency reference**: what is on the GPU at each phase, per mode and setting |
| [docs/COMMUNITY_IMPROVEMENTS.en.md](docs/COMMUNITY_IMPROVEMENTS.en.md) | Decisions on community improvements (adopted / investigated and declined / reimplemented in-house) |
| [docs/HISTORY.en.md](docs/HISTORY.en.md) | **Development history**: dated work log, bugs and fixes, A/B measurements, backlog |
| [docs/internal/TECHNICAL_REPORT.en.md](docs/internal/TECHNICAL_REPORT.en.md) | Internal document: the background to design decisions from 2026-08-04 to 08-08 and the pitfalls hit on real hardware |
| [docs/internal/regression_baselines_dev96.json](docs/internal/regression_baselines_dev96.json) | Regression baselines for diffusers upgrades (run with `scripts/run_regression.py`) |

For a conversational application built on this server see [Realtime_Narration_Video](https://github.com/animede/Realtime_Narration_Video) and the voice client [Realtime_Conversation_Video](https://github.com/animede/Realtime_Conversation_Video); the production integration lives in [diffusers-movie-server](https://github.com/animede/diffusers-movie-server).

## Layout

```
minimax-h3/
├── app.py                 # FastAPI server (port 8611)
├── run.sh                 # launcher for the diffusers-movie-server integration (port 8631)
├── run.ps1, run.bat       # Windows launcher (port 8611; the .bat runs the .ps1 bypassing the execution policy)
├── core/
│   ├── runner.py          # ModularPipeline loading / generation logic
│   ├── blockstream.py     # windowed prefetch streamer for transformer blocks (H3_GROUP_STREAM=own)
│   ├── filestream.py      # streams weights straight from the quantized cache files (H3_GROUP_STREAM_HOST=file)
│   ├── leanblock.py       # memory-lean block forward
│   ├── attnchunk.py       # query-chunked / head-grouped attention
│   ├── testream.py        # streaming of the projected TE and token_refiner
│   ├── vaecache.py        # fp16 cache of the VAE decoder
│   ├── pruned.py          # AdaLN-pruned transformer_ref and its quantized cache
│   ├── adaln_precompute.py, h3_cudagraph.py
│   ├── llm.py, prompt_check.py   # LLM prompt enhancement and verification
│   ├── settings.py, gallery.py
├── static/index.html      # single-page UI (ja / en)
├── scripts/               # install.sh / install.ps1 / install.bat, download, regression and probe scripts
├── docs/                  # documentation (table above); docs/samples/ holds the sample video
├── models/                # quantized caches etc. (git-ignored)
├── outputs/, logs/        # outputs and logs (git-ignored)
└── venv/                  # dedicated venv (git-ignored)
```

## License

Copyright holder of the source code in this repository: © 2026 YUZUKI (animede). Keep the copyright notice and the contents of [NOTICE](NOTICE) when redistributing, as required by Apache-2.0 section 4.

**This repository's code is licensed under Apache License 2.0** ([LICENSE](LICENSE)). Model weights are not included.

The licenses of the models / weights used must each be followed separately:

| Target | License |
|---|---|
| MiniMax-H3's own weights | MiniMax Community License (free for non-commercial use; commercial use free up to $20M/year revenue, credit required) |
| Turbo LoRA (`larryvrh/MiniMax-H3-Turbo-Lora`, `lightx2v/Minimax-h3-Turbo`) | Apache-2.0 |
| Qwen3-VL-32B (text_encoder) / Qwen3-VL-4B (projected TE) | Follows Qwen's official license |
| diffusers / transformers / torchao / SageAttention etc. | Each package's own license |

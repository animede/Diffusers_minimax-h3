# Features and settings in detail (minimax-h3)

[日本語](FEATURES.md) | **English**

Detailed descriptions and measurements for each feature and environment variable, split out of the README. Each section is the original README text moved verbatim; dates inside a section mark when that feature was implemented and measured. See [README.en.md](../README.en.md) for how to use the app and [HISTORY.en.md](HISTORY.en.md) for the chronological work log.

- [Performance overview](#performance-overview)
- [VRAM configurations and residency](#vram-configurations-and-residency)
- [text_encoder](#text_encoder)
- [transformer and denoising](#transformer-and-denoising)
- [Generation modes](#generation-modes)
- [UI](#ui)
- [LLM prompt enhancement](#llm-prompt-enhancement)

## Performance overview

### What the speedups bought (measured summary)

**This repository's central finding: the bottleneck was not denoising but the fixed cost of
loading and freeing models.** The numbers below are collected from the sections further
down; nothing here was re-measured for this table.

#### Cumulative: from the stock configuration to today

48GB box (PRO 5000 48GB + RTX 4000 20GB), the `H3_LOWVRAM=1` family, 768², **steady-state**
(second and later requests, excluding the initial model load; t2va is 5s = 124 frames).

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

#### Where each mode landed (all measured in one configuration, 2026-08-12)

Configuration: **int8 transformer resident + projected TE (NF4) + no decode-window free +
turbo at 4 steps + sage + fp16 decode**, on **a single GPU** (measured on the 96GB box:
`H3_TRANSFORMER_QUANT=int8 H3_KEEP_TRANSFORMER=1 H3_VIDEO_VAE_FP16=1 H3_TE_PROJ=…
H3_TURBO_LORA=1`).

| Mode | Single (fastest) | **Per item in a run** | Denoise | Peak | What the remaining time goes to |
|---|---|---|---|---|---|
| **t2i** 768² | **7.40s** | 7.9s (**0.94x — not worth using**) | 2.40s | 45.0GB | VAE CPU↔GPU round trips, ~3.3s |
| **t2va** 5s 768² | **28.13s** | (no batch API) | 14.94s | 45.6GB | decode, 7.05s |
| **ref2i** 768² (reference still) | 79.3s | **47.0s (1.69x)** | 7.8s | 45.4GB | reference vision encode, **~47s** |
| **i2va** 5s 768² (image reference to video) | 103.1s | **75.0s (1.37x)** | 22.0s | 45.9GB | the same ~47s, plus ~13s to reload the t2va transformer |

**How to read these numbers** (measurement conditions):

- **All steady-state; the initial load at server startup is excluded** (second and later
  requests). Startup spends **about 50s** making the transformer and VAE resident, but that is
  paid once per process.
- **The reference modes differ on their first request only**: `transformer_ref` is loaded on the
  first reference request rather than at startup, costing **+55s once** (measured for ref2i:
  134.7s first, 79.3s steady). It stays resident afterwards.
- Resolution pinned to **768×768**. Video is 5s = **124 frames** (24fps); stills are **22 frames**.
- **4 steps with the turbo LoRA** (`H3_TURBO_LORA=1`). FirstBlockCache is auto-disabled by turbo;
  attention is sage. Seed and prompt are fixed per mode.
- "Per item in a run" is the cost per item when several are generated together with the same
  reference and settings (`/api/t2i_batch`, `/api/ref2i_batch`, `/api/ref2va_batch`), measured
  with 3 scenes for t2i/ref2i and 2 for i2va. **The batch path is `H3_LOWVRAM=1`-only**, so that
  column runs a different configuration from the single column — the table pairs "the fastest way
  to make one" with "the fastest way to make N".

- **Every mode fits in 45-46GB**, so a single 48GB card runs the full feature set at
  near-peak speed. But **mixing the t2va family and the reference family in one process keeps
  both transformers resident, at 74.3GB** (measured; the peak on returning to t2va is 77.3GB).
  To run on one 48GB card, **keep the modes in separate processes**.
- **The reference modes are bound by the reference vision encode (~47s), not by denoising** —
  which is why turbo buys only 2.8x for i2va against 5.5x for t2va.
- Two GPUs with a bf16 transformer reach **t2i 6.89s / t2va 26.8s**, the fastest measured, but
  need a 77GB-class card for about 7%. See the dated 2026-08-12 sections.

#### What the batch measurements established

- **Batching t2i is no longer worth it** (0.94x, marginally slower). Batching existed to
  amortize model-load fixed costs, and **residency removed the cost there was to amortize**
  (before residency it was 2.3x: 157s to 67.5s).
- **Only the reference modes still benefit**, because what gets shared is **not a load but the
  ~47s reference vision encode**. **The batch's per-step time matches a single request exactly**
  (ref2i 2.598s vs 2.599s; i2va 7.321s vs 7.323s), so the batch path adds no overhead of its
  own — the entire difference is how many times that encode is paid.
- **More scenes, more benefit**: the saving is `47x(scenes-1)/scenes`. For i2va that is 1.37x at
  2 scenes, 1.44x at 3, 1.57x at 5 — so it favours long stories built around one character.
- **The "sharing shrinks it" model holds for stills and video alike** (measured vs predicted:
  ref2i at 3 scenes, 32.3 vs 31.3s/image; i2va at 2 scenes, 28.1 vs 23.5s/video). → **Caching
  the reference encode across requests should deliver the same saving to repeated single
  requests** (not implemented; see the dated 2026-08-12 section).
- **Two gates**: the batch path is `H3_LOWVRAM=1`-only and **silently falls back to sequential**
  otherwise; and reference batches reject `H3_TE_DEVICE` via a "TE GPU needs 24GB or more" guard
  — **a threshold sized for the 32B TE's vision activations**, far too large for the 3.11GB
  projected TE (it rejects a 16GB 4060 Ti). Both **worth revisiting**.

> **A measurement trap (walked into on 2026-08-12)**: comparing a batch against singles without
> passing `height`/`width` on the batch side means **the batch generates on the server's default
> 16:9 canvas (1344×768)** — 1.75x the pixels — which manufactures a nonexistent "the batch's
> steps are 1.75x slower" effect. It was caught because the pixel ratio 1344×768÷768² = 1.75
> matched the measured step-time ratio of 1.753 exactly. **Always pin the resolution when
> comparing modes or code paths.**

### An honest note on speed

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

> **Addendum (2026-11-17)**: the text above reflects August 2026; three things have changed.
> - "The support table above" now refers to the next section's "[VRAM × feature matrix](#vram--feature-matrix-measured-2026-08-11)" after the split.
> - **The turbo LoRA (4 steps) now works with `H3_LOWVRAM=group`** (in-house streamer + `H3_TURBO_LORA=1` at startup). Because the number of weight transfers scales with the step count, the transfer-bound "runs but slowly" caveat is largely gone. Measured on an A4000 16GB (Ampere sm_86, PCIe Gen4 x4, SDPA): t2va 5 s at 768² went from 790 s at 30 steps (peak 8.0GB) to 151 s at 4 steps (peak 7.0GB). With the 8GB-tier flags (`H3_TE_STREAM=1 H3_TE_DIET=1`, VAE decoder streamed, emulated with `H3_VRAM_LIMIT_GB=8`): 768² 209 s / 3.08GB, 1024² 376 s / 3.73GB, fl2va 225 s / 3.37GB, ref2va 299 s / 3.02GB (see the README's support table).
> - "FirstBlockCache does not fire on the int8+SDPA trajectory" no longer holds: on the same A4000 (int8 + SDPA, 30 steps) t2i skipped 7 steps and t2va 6 (`cache_skipped_steps`).
> The quality paragraph (crossing configurations changes the composition even at the same seed; that is a trajectory divergence, not degradation, and must be judged visually) is unchanged.

## VRAM configurations and residency

### VRAM × feature matrix (measured 2026-08-11)

○ = completes (measured) / △ = derived estimate (not yet measured) / × = OOM. **Times are
steady-state** (second and later requests, excluding the initial model load); **peaks are
torch's allocated peak**. The 16GB and 8GiB rows run 30 steps, the 48GB+20GB row turbo at 4
steps. All speeds are on this box's **PCIe Gen3 x4 slot + sm_89 (4060 Ti) + SDPA**; on a
**Gen4 x16 slot the weight transfer is roughly 1/8** (see "An honest note on speed" below).

| Config | t2i 768² | t2va 5s 768² | ref2i | i2va (image ref) | audio ref | 768×1344 5s |
|---|---|---|---|---|---|---|
| **Real RTX 4060 Ti 16GB, single** | ○ 498s / peak 7.4GB | ○ 25 min / 11.4GB (2026-11-15: 4.62GB with `H3_TE_STREAM=1 H3_TE_DIET=1`, 6.41GB at 1024²; 4.79GB with head-grouped attention, 3.03GB with `H3_CACHE=none`) | ○ | ○ 39 min / 9.41GB | ○ 54 min / 11.96GB | ○ 66 min / 13.37GB (15.2GB real = ceiling) |
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

### VRAM support table and main environment variables (quick reference)

Launch examples per use case. The top five rows are ballast measurements on the 96GB box
(PRO 6000, 32B TE); the bottom three rows are **real-card measurements using the projected
TE** (2026-08-11; see the "[VRAM × feature matrix](#vram--feature-matrix-measured-2026-08-11)"
at the top and the dated 2026-08-11 sections for details).

| GPU | Startup flag | t2va measured (768², 5s) |
|---|---|---|
| 96GB | (none = default) | peak 92GB / about 160s |
| 80GB tier | `H3_TRANSFORMER_QUANT=int8` | peak 59.7GB / about 160s |
| 48GB tier | `H3_LOWVRAM=1` | peak 38.9GB / about 215s |
| 32GB tier | `H3_LOWVRAM=group` | peak 28.7GB / about 280s (denoise about 1.7x faster since the 2026-11-15 streamer, see `H3_GROUP_STREAM` below) |
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
| `H3_PROFILE` | `auto` | Added 2026-11-17. When none of the main profile variables is set, fills in the 96/80/48/32/16/8GB-tier defaults from total VRAM (`core/profile.py`). On the 32GB-and-below tiers it also enables the turbo LoRA (ref2v / fl2v 4-step files), since in group mode it must be applied at startup; also with `file` streaming (the LoRA, about 1.2GB, is streamed from host RAM per unit; 2026-11-18). `96gb`..`8gb` forces a tier, `off` disables. Also sets `H3_ATTN_BACKEND=default` when sage cannot be used |
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
| `H3_TURBO_LORA` | `0` | 4/8-step distilled LoRA (downloads about 780MB on first use). Since 2026-11-17 also works in `group` (`H3_GROUP_STREAM=own` + the flag at startup; with `file` streaming the LoRA, about 1.2GB, first stayed on the GPU: 8GB-tier flags under an 8GB cap gave 768² 205 s / 4.19GB, 1024² 376 s / 4.85GB, ref2va 768² 276 s / 4.13GB. At 15:00 the same day it was switched to per-unit streaming from host RAM (`H3_FILE_STREAM_HOST_MIN_MB`); under a 6GB cap 768² takes 211 s at 2.82GB and 1024² 380 s at 3.48GB. Verified to work on real Windows 11 + RTX 4060 Laptop 8GB / 16GB RAM and Windows 10 + RTX 3050 6GB / 16GB RAM machines (times to be published once measured under fair conditions; weight reads are the bottleneck, so SSD speed dominates). A4000 16GB: t2va 5 s 768² in 151 s at 4 steps; with the 8GB-tier flags under an 8GB cap, 768² 209 s / 3.08GB, 1024² 376 s / 3.73GB ref2va 768² 299 s / 3.0GB and fl2va 768² 225 s / 3.4GB. Note: ref2va in group mode with the in-house streamer crashed on a layout device mismatch, fixed the same day) |
| `H3_ALLOW_WEIGHT_DOWNLOAD` | `0` | When the bf16 transformer / transformer_ref weights (~66GB each) are not cached, download them from HF only if `1` (default: no download, fail with a hint) |
| `H3_PIKA_RENOISE` | `0` | Renoise sampler for the Pika H3 Flex turbo LoRA (experimental) |
| `H3_BLOCK_COMPILE` | `0` | `1` torch.compiles the 50 transformer_ref blocks after the turbo LoRA is applied (attention itself is a graph break). About -13% per step at 1024×768 on an RTX PRO 6000; +5-14s on the first call only |
| `H3_LORA_FUSE` | `0` | Cuts element-wise work in the turbo LoRA wrapper: `1` skips the multiply when scale is 1.0 (bit-identical) / `2` also fuses the add into the GEMM. No extra gain on top of `H3_BLOCK_COMPILE` |
| `H3_VIDEO_VAE_FP16` | `0` | fp16-ify the video VAE (reduces decode peak; required at 16GB and below) |
| `H3_LLM_URL` | `http://127.0.0.1:64660` | Local LLM used for prompt enhancement (optional feature) |
| `H3_LLM_API_KEY` | none | API key for the LLM server (sent as `Authorization: Bearer`). If unset, read from the same-named line in the git-ignored `config.local.env` at the repo root |
| `H3_LLM_LOCAL` / `H3_LLM_LOCAL_GGUF` | `auto` / none | Added 2026-11-17. Falls back to a built-in CPU LLM (Gemma 4 E4B, GGUF Q4_0) when the external LLM is unreachable; it is llama.cpp's prebuilt `llama-server` run as a child process (`core/local_llm.py` fetches and starts it, no compiler needed). `1` = built-in only, `0` = off. `H3_LLM_LOCAL_IDLE_S` (default 300) stops it when idle, `H3_LLM_LOCAL_STOP_ON_GENERATE` (auto = hosts under 32GB RAM) stops it when a generation starts |

### Oct 2026 update: realtime conversational generation on a single 24GB/32GB GPU

Since the initial August release, major speed / low-VRAM work for conversational use
(continuous ref2va) was done in the production repository (diffusers-movie-server) and
re-synced here. Where it landed:

| Setup | Condition | Result |
|---|---|---|
| **Single 32GB (RTX 5090 class)** | 320x448, 141f (5.875s of video) | **5.57s = 0.95x realtime**, peak 25.6GB |
| Single 24GB (RTX 4090 class) | 320x448, 141f | 7.1s = 1.21x (usable for short replies), peak 20.4GB |

Key additions:
- **ck-w4a8 quantization** (`H3_PRUNED_QUANT=ck-w4a8` / `H3_TRANSFORMER_QUANT=ck-w4a8`):
  ComfyUI's W4A8 int8 kernels (`pip install comfy-kitchen==0.2.37`). Combined with the
  AdaLN-pruned checkpoints (multimodalart), the 33B transformer fits in **12.5-14.9GB
  resident with -28% denoise time and visually equal quality**
- **Pinned parking** (`H3_VAE_PINNED` / `H3_BASE_PINNED` / `H3_REF_PINNED`): model
  eviction becomes a pinned-memory pointer swap (park 0.3s / restore 2s)
- **TE reduction** (`H3_TE_DIET` / `H3_TE_STREAM`), prequantized weight caches,
  reference latent / prefix caches
- lightx2v 4-step turbo distillation (`H3_TURBO_LORA=1`)

Env set for single-GPU operation (32GB class; for 24GB change to `H3_KEEP_REF2VA_VAE=0`):
see the Japanese section above for the full command line.

**Caveats**: the GPU must be dedicated (TTS/LLM cannot co-reside); first-run cache
creation needs ~38GB of downloads and **45GB of free host RAM** for the ref side, plus
converting Kijai's `minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` (12.5GB) with
`scripts/convert_kijai_w4a8.py` for the base side; pinned host RAM totals ~45GiB
(on 64GB boxes drop `H3_REF_PINNED` or `H3_TE_STREAM`); on 24GB do NOT set
`H3_KEEP_REF2VA_VAE=1` (OOMs on the speech/idle prompt switch).

### VRAM/RAM design (important, based on measurements)

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

#### `H3_TE_QUANT=bnb-4bit` (default)

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

#### `H3_TE_QUANT=none` (bf16 TE, legacy method)

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

### Measured values (RTX PRO 6000 Blackwell 96GB, 768x768, 124 frames = 5.17s, 30 steps)

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

### 48GB-tier VRAM support (`H3_LOWVRAM`, default `0`)

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

### 24-32GB-tier VRAM support (`H3_LOWVRAM=group`, added 2026-08-05)

#### 2026-11-15: block transfer replaced by an in-house windowed prefetch streamer (`H3_GROUP_STREAM`, default `own`)

The block transfer of `H3_LOWVRAM=group` moved from diffusers' `enable_group_offload(block_level, use_stream=True)`
to the streamer in `core/blockstream.py` (`H3_GROUP_STREAM=diffusers` restores the old path). It follows the
design of the text encoder's `H3_TE_STREAM` (`core/testream.py`): the 50 int8 blocks (30.1 GiB) live in
page-locked host RAM and a copy stream sends `window` blocks ahead of the computation (`H3_GROUP_OFFLOAD_BLOCKS`+1,
default 2). Everything that is not a block (embedders, token_refiner, output layers, 1.6 GiB in bf16) stays on the
GPU. torchao `Int8Tensor`s are moved through their inner `qdata` / `scale` / `zero_point` tensors rather than the
outer wrapper (the same approach as diffusers' `_restore_torchao_tensor`). This is the design proven in
flux3-klein-studio's 6 GB work, ported here.

Measured (GPU0 of an RTX PRO 5000 Blackwell 48 GB, `H3_LOWVRAM=group H3_TE_PRUNE=1`, t2va 512² x 5 s = 124 frames,
30 steps, seed 1, two runs each):

| | diffusers (`H3_GROUP_STREAM=diffusers`) | in-house (`own`, default) |
|---|---:|---:|
| Denoise, `H3_CACHE=none` | 164.9-165.0 s (5.68 s/step) | **96.4-96.5 s (3.32 s/step)** |
| Denoise, `H3_CACHE=fbc` (default) | 165.5-167.6 s (FBC never skips a step) | **85.6-86.1 s (FBC skips 6 of 30 steps)** |
| Peak VRAM | 17.7-18.3 GB | 19.5-20.0 GB (+1.6 GiB of resident non-block modules) |
| Output (MD5) | **identical** to the in-house streamer with `H3_CACHE=none` | same |
| Load (pinning) | 20 s (34 GB into a pinned dict) | 15 s (one buffer per block) |

- The speed difference is the transfer scheme itself (1.71x with FBC off on both sides): the stock hooks pay a
  fixed cost per group and let the CPU run ahead of the GPU, which delays the frees (section 4.2 of
  `docs/lowvram-techniques.md`).
- **FirstBlockCache now works in group mode.** Under the diffusers path FBC never skipped a step (the
  `cache_skipped_steps: 0` noted under "An honest note on speed"); the in-house streamer skips 6 of 30 at the
  default threshold 0.05, so the output differs from FBC-off by exactly FBC's intended approximation. Use
  `H3_CACHE=none` when the exact result matters.
- The group-mode transformer now loads from the prequantized cache (`models/prequant/transformer_int8`) onto the CPU
  when present (the bf16 shards no longer need to be in the HF cache; same fail-open as the other modes). Int8Tensors
  read from the cache are pinned to `requires_grad=False` (diffusers' `swap_tensors` otherwise walks autograd and
  fails with `aten.view unimplemented`). On a box whose bf16 shards were deleted, the `source_snapshot` check of
  `meta.json` is skipped.
- Release goes through `BlockStreamer.release()` (`cudaHostUnregister`) instead of relying on `_host_emptyCache`.
- Host RAM: 30.1 GiB pinned (about the same as the diffusers path's 34 GB pinned dict). The RAM guard
  (`H3_GROUP_OFFLOAD_MIN_RAM_GB`) is unchanged.

#### 2026-11-15: where low VRAM stands — from the 16GB configuration to 8GB / 6GB (three streaming stages)

The same design was applied to the three things that still stayed on the GPU in group mode. All are placement
changes only, and **the outputs match by MD5**.

1. **The projection TE's LM layers** (`H3_TE_STREAM=1` now also applies to `H3_TE_PROJ`): the 4B model's 36 layers
   (1.9 GiB) live in pinned host memory and stream only during encoding. Group mode keeps the TE resident through
   denoise, so this lowers the denoise peak.
2. **The token_refiner's two blocks** (`H3_GROUP_STREAM_REFINER`, default 1): excluded from int8, 1.5 GiB of bf16 stayed resident.
3. **The video VAE decoder's 36 blocks** (`H3_GROUP_STREAM_VAE`, default `auto` = stream when less than 8 GB of VRAM is
   free at the first decode): 4.8 GiB of the VAE's 5.2 GiB (fp16) is the decoder, and after 1 and 2 this was the peak.
   Blocks are re-sent per tile, so decoding gets slower.

Measured (GPU0 of an RTX PRO 5000 48 GB, `H3_LOWVRAM=group H3_TE_PROJ=… H3_VIDEO_VAE_FP16=1 H3_TE_STREAM=1`,
t2va 5 s, 30 steps, seed 1):

| | baseline (in-house streamer only) | +1 projection TE | +2 token_refiner | +3 VAE decoder |
|---|---:|---:|---:|---:|
| 512² x 5 s peak | 8.59 GB | 8.33 GB | 6.79 GB | **4.67 GB** |
| 768² x 5 s peak | 10.78 GB | 9.54 GB | 7.99 GB | **7.45 GB** |
| 512² denoise / decode | 90 / 6 s | 86 / 6 s | 90 / 6 s | 89 / 17 s |
| 768² denoise / decode | 200 / 11 s | 193 / 11 s | 195 / 11 s | 195 / 30 s |

Resident VRAM right after startup is 0.18 GB. What remains at the peak is the denoise activations, which depend on
resolution and length.

4. **Token-chunked feed-forward** (`H3_FF_CHUNK`, default `auto` = 8192 tokens in group mode, `0` disables). The largest
   transient in a block is the SwiGLU projection (2 × ffn_dim 14336 bf16 = 57 KB per token); running the feed-forward
   8192 tokens at a time bounds that transient. It is a per-token operation, so the result is the same (MD5 match).

| Activation peak incl. 1K (denoise phase, same configuration) | unchunked | `H3_FF_CHUNK=8192` |
|---|---:|---:|
| 768² × 5 s | 7.45 GB | **6.26 GB** |
| 1024² × 5 s (38k tokens, 13.9 s/step) | 10.73 GB | **9.34 GB** |

What remains is the attention's K and V (0.55 GB each at 1024²) plus the output and the residual stream, close to the
floor for an exact result.

5. **Query-chunked attention** (`H3_ATTN_Q_CHUNK`, default 0 = off, `core/attnchunk.py`). K and V are built in full; the
   Q projection, QK norm, RoPE, the kernel and the output projection run 8192 tokens at a time (MD5 match). It only
   helps at 1K and above: the 1024² × 5 s denoise peak goes 9.34 → **8.53 GB**, 768² is unchanged (6.26 → 6.38). It is
   2-4% slower (13.9 → 14.4 s/step at 1024²), so it is off by default; add `H3_ATTN_Q_CHUNK=8192` for 1K on a small card.

That left 1024² × 5 s at 8.5 GB. Three more changes closed the gap (all MD5-identical):

6. **A memory-lean block forward** (`H3_LEAN_BLOCK`, default `auto` = 8192 tokens in group mode, `core/leanblock.py`).
   The original forward materializes the norm1 output, the attention output, the norm2 output and the FF output for
   all tokens (0.41 GB each at 1024²) before adding them. The lean version runs normalization → modulation →
   projections/attention/FF → gate → residual add in chunks, so the only full-length tensors are the input, the
   output, K and V. It subsumes 4 (FF chunking) and 5 (query chunking); recomputing the norm per chunk costs about 7%
   at 1024².
7. **Streaming in thirds of a block** (`H3_GROUP_STREAM_UNITS=half`, default): adaln_proj → attn → ff as 150 units, so
   the prefetched residency drops from two blocks (1.3 GiB) to two small units.
8. **`H3_TE_DIET=1` for the projection TE** (opt-in): releasing lm_head and bridging the embedding table
   (151936 × 2560) to the CPU frees 1.45 GiB.

| Peak VRAM (same configuration + `H3_TE_DIET=1`, overall, VAE streamed as on a small card) | at first (in-house streamer only) | 1-3 (three streaming stages) | everything (4-8) |
|---|---:|---:|---:|
| 512² × 5 s | 8.59 GB | 4.67 GB | **3.27 GB** |
| 768² × 5 s | 10.78 GB | 7.45 GB | **4.85 GB** |
| 1024² × 5 s | 10.73 GB (denoise) | — | **6.82 GB** |

That was the morning. In the afternoon the 1024² × 5 s breakdown was re-measured (`H3_LEAN_DEBUG=1` logs allocated / peak
around blocks 0-2): 38,321 tokens (31,744 video + audio + text), one residual-stream buffer 0.41 GB, K and V 0.55 GB each,
and the largest transient inside a block is the attention phase (K + V 1.1 GB + the int8 copies sage makes inside its
kernel + the chunk). Four more changes:

9. **Rounding the fp32 patch embeddings** (`shrink_prologue` in `core/leanblock.py`, installed together with
   `H3_LEAN_BLOCK`). The transformer forward computes the `proj_in` output (a layer kept in fp32) for all tokens and
   rounds it to bf16 when packing, but the fp32 version survives as a local variable for the whole block loop (0.68 GB at
   1024²). Rounding at the projection's exit gives the same values with half the residency. 1024² 6.82 → **6.41 GB**,
   768² 4.85 → **4.62 GB** (MD5 match).
10. **In-place residual update** (same module). From block 2 on, each block writes into its input buffer (every chunk
    finishes reading its own rows before writing them, so the result is the same; MD5 match). FBC references the input
    and output of block 0, so blocks 0 and 1 write new buffers; together with the three residual tensors FBC keeps
    (head_block_output / head_block_residual / tail_block_residuals, 1.2 GB in total at 1024²) the peak stays at block 1,
    so the overall figure does not move with FBC on. With `H3_CACHE=none` every block is in place and one buffer is saved.
11. **Head-grouped attention** (`H3_ATTN_HEAD_GROUPS`, default `auto`). K, V and Q are built for the 56 heads in G groups
    (row slices of the to_q/to_k/to_v weights; the streamer keeps only the inner tensors of torchao's Int8Tensor on the
    GPU, so the slice is rebuilt from them). Each group's attention output goes into one full-width buffer (attn_out) and
    the output projection runs from it for all heads at once. The resident set goes from two buffers (K, V) to
    attn_out + K/G + V/G, and the K/V copies sage makes inside its kernel shrink by G as well. Recomputing the norm and
    modulation G times costs time (G=4 +13%, G=8 +15-35%, G=28 +70%). **Same math, but not bit-identical**: changing the
    GEMM width (7168/G) and the head count seen by the attention kernel changes kernel selection and summation order,
    so the last bf16 bit can move (sage's K-mean centering matches or not depending on the head count and token count).
    Thirty steps amplify that into a different sample of the same video (PSNR against the baseline 34.4 dB at 1024²,
    38.7 dB at 768²; FBC on vs off differs by 34.2 dB, so this is the same order as the FBC approximation that is on by
    default, not a quality loss). Therefore **auto always picks G=1 when G=1 fits** and otherwise the smallest of
    4 → 7 → 8 → 14 → 28 that fits the free VRAM (a `H3_VRAM_LIMIT_GB` cap is honored; decided at the first forward of a
    generation and kept for that generation, after reserving FBC's three residuals and 1 GiB of slack). Fixed G=4: 1024² 5.09 GB (16.7 s/step, +8%).
    Fixed G=8: 1024² **4.79 GB** (17.9 s/step), 768² **3.95 GB** (8.8 s/step). Under a 5.8 GB cap, auto chose G=4 → 28 and
    peaked at 4.79 GB (26 s/step). Even a combination that matched bit-for-bit in a single-block check (G=4 at 1024²) gave a
    different MD5 after 30 real steps, so treat any G>1 as a different sample.
12. **Turning FBC off** (`H3_CACHE=none`) drops the three FBC residuals and makes every block in place. With G=8 the
    1024² × 5 s denoise peaks at **3.03 GB** (all 30 steps are computed, 23 s/step; FBC itself is an approximation, so the
    output changes).

| Peak VRAM (same configuration, overall) | up to 8 (morning) | 9-10 (MD5 match, default) | + head groups G=8 (different sample) | + `H3_CACHE=none` |
|---|---:|---:|---:|---:|
| 512² × 5 s | 3.27 GB | **2.90 GB** | — | — |
| 768² × 5 s | 4.85 GB | **4.62 GB** | 3.95 GB | — |
| 1024² × 5 s | 6.82 GB | **6.41 GB** | 4.79 GB | 3.03 GB |

Exact (MD5-identical) results bottom out at **512² 2.90 GB / 768² 4.62 GB / 1024² 6.41 GB**; what remains is K and V, the residual stream
and the three FBC residuals. Allowing head groups puts **1024² × 5 s on a 6 GB card** (auto uses them only when needed).
`/api/status` reports `group_stream_refiner`, `group_stream_vae`, `group_stream_units`, `ff_chunk`, `attn_q_chunk`,
`lean_block`, `attn_head_groups` and `lean_block_last` (token count, G and the first in-place block of the latest forward;
the generation result JSON carries `lean_block_last` too).

#### PCs with 32 GB / 16 GB of host RAM: streaming the weights straight from the files (`H3_GROUP_STREAM_HOST=file`, 2026-11-15)

Everything above keeps the whole int8 transformer (30 GiB) in pinned host RAM, which assumes a 48 GB-class machine.
`H3_GROUP_STREAM_HOST=file` (`core/filestream.py`, the same scheme as minimax-h3-image-server's `app/filestream.py`) reads
each unit (a third of a block) from the prequantized cache (safetensors; `from_pretrained` maps the files, so nothing is
expanded into RAM) with 8 threads in 32 MB chunks into a small pinned staging ring (3 × 0.24 GiB, two units of read-ahead)
and copies from there to the GPU. The only mandatory host RAM is that ring plus the token_refiner's (2 × 0.36 GiB); whatever
else fits in the OS page cache makes it faster. The tail units that do not fit the cache are dropped with
`posix_fadvise(DONTNEED)` after every read so the cyclic access does not flush everything through LRU (the number is `auto`:
weights + the TE / VAE pinning that comes later − usable RAM, honoring a cgroup limit). In file mode the VAE decoder's
streaming source is not pinned either (the fp16 copy is used as is). The int8 weights are the same, so the output is the same.

- `H3_GROUP_STREAM_HOST`: `auto` (default: file when the installed RAM is below `H3_GROUP_STREAM_FILE_BELOW_GB` = 48 GB),
  `pinned`, `file`. Falls back to pinned when the weights were quantized from bf16 at load (no cache file to read).
  `H3_GROUP_STREAM_DROP_UNITS` (auto / a number), `H3_GROUP_STREAM_RING` (staging buffers, default 3). `H3_FILE_STREAM_HOST_MIN_MB` (default 1): tensors missing from the files (the turbo LoRA's
  lora_a / lora_b, 1.24 GiB in total) at least this large are not kept on the GPU but held in pageable host RAM and
  memcpy'd into the staging buffer per unit (2026-11-18; resident, they made a 6GB card spill into Windows shared memory
  and crawl, measured on an RTX 3050). `/api/status` reports
  `group_stream_host` / `group_stream_file`.
- Windows has no `fadvise`, so the drop control does not apply (reading itself works, opened with
  `FILE_FLAG_SEQUENTIAL_SCAN`; the same scheme was verified on the image server's RTX 3050 / 16 GB RAM machine).
  2026-11-18: `FILE_FLAG_SEQUENTIAL_SCAN` is now applied only to weight sets that do not fit in RAM (the 30 GB transformer);
  sets that fit (the 4.5 GB VAE decoder, the 1.9 GB projected TE) are opened with normal caching. Applying it to everything
  made every per-tile re-read of the decoder hit the NVMe on an RTX 4060 Laptop / 16 GB RAM machine (decode 124 s, NVMe
  pinned at about 1 GB/s). The rule is "weights + 1 GiB + reserve <= usable RAM"; the log says `page cache kept / not kept`.
- Same day: a path that keeps the decoder **resident on the GPU** (`H3_VAE_DECODER_RESIDENT`, default auto, `ResidentLoader` in
  `core/filestream.py`). A 768², 5 s decode is 7 clips x 16 tiles, about 150 passes, and re-sending the 36 blocks (4.5 GB) from the
  host per tile made that transfer (page cache -> staging -> GPU) most of the decode time on 16 GB-RAM PCs (RTX 4060 Laptop: 0.68 s
  of each 0.77 s pass waiting for reads, decode about 90 s). When free VRAM is at least 4.5 + 0.5 + output + 0.5 GiB headroom the
  decoder is read once and kept. Forced on the A4000 under an 8 GB cap: decode 86 -> 34 s, total 201 -> 152 s, peak 7.03 GB,
  identical output hash. When the whole decoder does not fit, only as many blocks as fit are kept and the rest is streamed
  (`SplitDecoder`; default headroom 1.0 GiB). A4000 emulation: 30/36 resident under an 8 GB cap, decode 33.8 s, peak 6.22 GB,
  total 149 s; 15/36 under a 6 GB cap, decode 51 s, peak 4.21 GB. Identical output hashes in both. The streamed remainder shrinks
  to 0.75-2.6 GB, so it also stays in the page cache (the 90-300 s decode swings on 16 GB-RAM PCs came from the 4.5 GB decoder
  falling in and out of the cache).
- 2026-11-19, tried and rejected: 512 px VAE tiles instead of 256 px to cut the pass count to a quarter (on 8 GB laptops the
  decode is GPU-compute-bound and this was the only remaining lever). Under an 8 GB cap on the A4000 the 768² decode drops from
  86 to 38 s, but **the whole frame gets a grid-like artifact** (PSNR 22 dB, SSIM 0.60 against the 256 px output; everywhere, not
  just at seams). The decoder's transformer blocks appear to be trained for 256 px tiles, so the tile size cannot be changed. Not
  added to the code. On the RTX 4060 Laptop (26/36 resident, zero read wait) the 85.7 s decode, about 0.57 s x 150 passes, is
  the floor for that GPU.
- The block count is decided from "driver free + torch's reserved-but-unused memory". Windows torch has no
  `expandable_segments`, so a few GB stay reserved after the denoise (real RTX 4060 Laptop: allocated 1.46 / reserved 3.33 GB,
  driver free 3.8 GB, so only 10/36 blocks fit at first); that memory is reused by new allocations and may be counted.
- The quantized cache is built beforehand with `scripts/prepare_lowvram.py` (fetches the bf16 shards one at a time,
  quantizes and deletes them; fine with 16 GB RAM). Ref2VA / Ref2I use a different set of weights, `transformer_ref/`
  (66 GB, every shard differs from `transformer`), so `--ref` also builds `transformer_ref_int8` (+32 GB). Until
  2026-11-18 the ref2va group load ran the 40 GB RAM guard before looking for the cache, so the reference modes always
  stopped on low-RAM hosts; it now matches the base side (no guard when the cache exists and is streamed from file)
  and, when the cache is missing, stops with a message pointing at `--ref`.

Measured on GPU0 of the 48 GB machine with the RAM limit emulated by a cgroup and the cache's page cache dropped before
each run (this machine's NVMe reads 3.3 GB/s directly; every output is MD5-identical to pinned):

| Where the weights live | Mandatory host RAM | 512² × 5 s | 1024² × 5 s | Notes |
|---|---:|---:|---:|---|
| pinned (reference) | about 38 GiB (transformer 30 + refiner 1.4 + TE 1.9 + VAE 4.5) | 2.98 s/step | 15.1 s/step | VRAM 2.90 / 6.41 GB |
| file, no RAM limit (warm cache) | about 8 GiB | 3.11 s/step | — | VRAM 2.49 GB (one less unit prefetched on the GPU) |
| file, 28 GB RAM cap (a 32 GB PC) | same | 6.1 s/step | **14.7 s/step** | 94 of 150 units read from disk; at 1K the reads hide behind compute |
| file, 16 GB RAM cap | same | 8.0 s/step | — | all 150 units read from disk (30 GiB/step), MD5 match |
| file, 12 GB RAM cap (a 16 GB PC), VAE and TE from files too (below) | about 2 GiB | 8.0 s/step | **15.4 s/step** | MD5 match; the VAE is pinned only during the decode when RAM allows |

At 1K the reads (worst case 30 GiB/step ÷ 3.3 GB/s ≈ 9 s) hide behind 15 s of compute, so it is as fast as pinned; 512² is
read-bound and takes about twice as long. **A 32 GB PC runs 1024² × 5 s at the pinned speed, and a 16 GB PC runs 1K too** (speed depends on the disk's sequential
read speed). In the first 12 GB-cap test the process's anonymous memory reached about 12 GB right before the decode (`proc_anon_gb`:
the fp16 copy of the VAE 4.8, staging 1.6, the TE's pinned layers 1.9-2.4, baseline 0.6, plus the decode's frame buffers).
The two remaining pieces were moved to the same scheme, and 1K now runs with 16 GB of RAM:

- **An fp16 cache of the VAE decoder** (`core/vaecache.py`, `models/prequant/vae_decoder_fp16/`, 4.5 GiB, written once in
  2 s). In file mode the decoder blocks are not cast to fp16 at startup (that copy was 4.8 GiB of anonymous memory); they
  become 0-element placeholders. At decode time the free RAM decides: **with 4.5 GiB + 2 GiB available the cache is read
  into pinned memory in 1.4 s and kept only for that decode** (same speed as pinned); otherwise it is streamed from the
  file per tile (1K decode 48 → 95 s). The pinned copy is released afterwards. The values are the same rounding as
  `vae.to(float16)`, so the output is unchanged.
- **A quantized cache of the projection TE** (`H3_TE_PROJ_PREQUANT=1`, default, `models/prequant/te_proj_<model>_bnb-4bit/`).
  The bnb-4bit TE is saved once (2 s) and loaded from there afterwards (on this machine the load takes the same 9-10 s as
  quantizing from bf16, but it reads 2.2 GB of nf4 instead of 8 GB of bf16, which matters on a slow disk). In file mode its 36 LM layers (1.9 GiB) are streamed straight from that file instead of being
  pinned (staging 2 × 0.06 GiB).

With that, **1024² × 5 s completes under the 12 GB cap** (a 16 GB PC): 15.4 s/step, decode 97 s streamed from the file,
MD5 match; 768² runs at 8.3 s/step (the VAE is pinned for the decode, 30 s; MD5 match). The mandatory host RAM is the staging ring plus the TE's
(about 1 GiB) and the process itself, about 2 GiB in total.

##### VRAM class × host RAM class at a glance (t2va 5 s, 30 steps, the default low-VRAM configuration)

Speeds are seconds per step on this machine (RTX PRO 5000); an RTX 3050 takes several times longer. ○ = fits with
MD5-identical output, △ = fits with head groups (a different sample) or `H3_CACHE=none`, ✗ = does not fit.

| VRAM \ host RAM | 16 GB (file, every unit from disk) | 32 GB (file, part from disk) | 48 GB+ (pinned) |
|---|---|---|---|
| **6 GB** | 512² ○ 8.0 / 768² ○ 8.3. 1K △ about 17 (head groups; decode streamed from the file, 97 s) | 512² ○ 6.1 / 768² ○ 7.1. 1K △ 16.8 (auto picks G=4, 4.4 GB) | 512² ○ 3.0 / 768² ○ 7.1. 1K △ (head groups 4.8 GB, `H3_CACHE=none` 3.0 GB) |
| **8 GB** | same; 1K ○ 15.4 (5.8 GB, decode 97 s) | same; 1K ○ 14.7 (5.8 GB) | same; 1K ○ 15.1 (6.4 GB) |
| **12-16 GB** | same | same; with 8 GB+ of free VRAM the VAE is not streamed and the decode is faster (768²: 31 → about 11 s) | same |
| **24 GB+** | same | the earlier tables (`H3_LOWVRAM=1` family) also apply | same |

- VRAM decides what fits: 512² (2.9 GB) and 768² (4.6 GB) fit 6 GB; 1K stays MD5-identical on 8 GB (pinned 6.4 GB /
  file 5.8 GB) and needs head groups (auto uses them only when required) or `H3_CACHE=none` on 6 GB.
- Host RAM decides the speed: 48 GB+ pins the weights and is fastest, 32 GB runs 768² and 1K at the same speed and 512²
  at 2×, 16 GB runs 512² at 2.7× and 768² at 1.1×, and 1K runs as well (the VAE and the TE are streamed from files too).
- File mode prefetches one unit less on the GPU, so the VRAM peak drops too (512² 2.49 / 768² 3.71 / 1K 5.76 GB).
  Whether the streamed VAE decoder is pinned in file mode is `H3_GROUP_STREAM_VAE_PIN` (auto = RAM 24 GB+; without it
  the 768² decode goes 31 → 53 s).

`H3_LOWVRAM=1` (48GB tier) loads the full transformer (34GB) onto the GPU every request,
so it doesn't even fit alone on a 24-32GB-tier card. `H3_LOWVRAM=group` ports the
"block-level group offload" pattern established in diffusers-server (a sibling project)'s
CLAUDE.md #33/#34/#37 into this project, keeping the transformer **resident in host RAM**
while shuttling only the blocks needed for each denoise step (1-2 of 50 layers,
~0.68GB x 1-2) to and from the GPU. The transformer is **loaded exactly once at process
startup and stays resident across requests** (unlike `H3_LOWVRAM=1`'s per-request reload).

#### Investigation results on the PR side's "load-time quantization during streamed offload"

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

#### Implementation summary

- `_ensure_transformer_group()` (`core/runner.py`) quantization-loads onto the CPU with
  `device_map={"transformer": "cpu"}` + `TorchAoConfig(Int8WeightOnlyConfig)`, then calls
  `enable_group_offload(offload_type="block_level", num_blocks_per_group=1,
  use_stream=..., low_cpu_mem_usage=...)` (transformer_ref has an analogous
  `_ensure_transformer_ref_group()`).
- TE requires bnb-4bit just like `H3_LOWVRAM=1` (forced at startup). In t2va's steady
  state, TE stays resident across requests too (since the transformer is resident in the
  first place, it's better to keep TE resident as well and avoid per-request reload cost).

#### [Major finding] `use_stream=True` + `low_cpu_mem_usage=True` combined has a bug that breaks torchao Int8Tensor

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

#### Final choreography (phase x resident items x peak)

| Phase | Large resident items | Notes |
|---|---|---|
| Startup preload | transformer (int8, CPU-resident + group-offload hooks) | About 90s (CPU load 70s + pinning 22s) |
| t2va encode | TE-nf4 (GPU, 21GB) + transformer (CPU) | |
| t2va denoise | TE-nf4 (GPU, 21GB) + 1-2 transformer blocks (GPU, ~1.4GB) | |
| t2va decode | VAE pair (GPU, ~11GB) + 1-2 transformer blocks | **TE is force-freed only for this window** (see below), reloaded after decode |
| ref2va reference encode | VAE pair (GPU, 11GB) | TE is force-freed only for this window (see below) |
| ref2va denoise | TE-nf4 (GPU, 21GB) + 1-2 transformer_ref blocks | |
| Steady state between requests | transformer (CPU) + TE-nf4 (GPU) | After ref2va, transformer_ref returns to unloaded (reloaded every t2va<->ref2va switch) |

#### [Second bug discovered and fixed during implementation] TE needed to be force-freed for the decode window and reference-encode window

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

#### Result of the MD5-match check

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

#### Measurement tables: 32GB-limit and 24GB-limit probes

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

#### VAE tiling investigation results

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

#### Result of confirming FBC/sage coexistence

Throughout all ballast verification (both 32GB and 24GB), running with `H3_CACHE=fbc`
(default) and `H3_ATTN_BACKEND=sage` (default) enabled produced no conflicting errors with
the group-offload hooks (FBC makes block-level skip decisions, group offload manages
block GPU residency, and sage handles attention implementation inside each block — three
independent layers). Separate from the MD5-match check (done with `H3_CACHE=none`), a
full run with the default FBC-enabled settings (2 t2va runs under the 32GB limit) also
completed successfully.

#### RAM constraint for Ref2VA (a known, unresolved limitation)

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

### 2026-08-09: H3_KEEP_TRANSFORMER — eliminates the per-request reload fixed cost by keeping the transformer resident

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

### Keeping text_encoder resident on a separate GPU (`H3_TE_DEVICE`, default ``, 2026-08-09)

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

#### Implementation summary (the `_execution_device` pitfall, again)

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

### fp16-ifying the video VAE (`H3_VIDEO_VAE_FP16`, default `0`)

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

## text_encoder

### Removing unused upper layers of text_encoder (`H3_TE_PRUNE`, default `0`, added 2026-08-06)

MiniMax-H3's text_encoder (Qwen3-VL-32B, 64 layers) only reads
`hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER]` (=50)
(`diffusers/modular_pipelines/minimax_h3/encoders.py`). `H3_TE_PRUNE=1` builds the
text_encoder with **only 51 layers** (0-50, `MINIMAX_H3_TEXT_ENCODER_LAYER + 1`), and
never loads the unused layers 52-64 + final `norm` + `lm_head` (equivalent to 14 layers'
worth of weight; measured ~3.6GB with bnb-4bit / ~13.6GB with bf16). The default `0` leaves
everything completely unchanged (this branch is never invoked).

#### Why "51 layers," not exactly 50 (a transformers-side pitfall discovered and verified in this task)

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

#### Implementation

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

#### Measured size of the pruned TE

| Precision | Before pruning | After pruning (51 layers) | Reduction |
|---|---|---|---|
| bnb-4bit nf4 | 21.02GB | **17.45GB** | -3.57GB (-17%) |
| bf16 | 66.71GB | **53.06GB** | -13.65GB (-20%) |

Because nf4 quantization compresses per-layer size to roughly 1/4 of bf16, the absolute
reduction amount is also smaller than bf16 (the relative reduction ratio is roughly the
same).

#### Result of the MD5-match check

Comparing the output of `H3_TE_PRUNE=0` (no pruning) against `H3_TE_PRUNE=1` (pruned),
with the same seed (768², 5s, 30 steps, fox prompt, seed=12345, using `H3_CACHE=none` to
eliminate FBC's path dependence), confirmed **byte-identical mp4 (md5 match) for both
t2va and ref2va (1 image reference, via the vision tower)**. Empirical proof that the
pruning is mathematically inconsequential. Also confirmed that outputs match exactly
regardless of pruning in `H3_LOWVRAM=1` and `H3_LOWVRAM=group` modes as well (described
below).

#### 24GB-tier support: pruning alone wasn't enough (discovered on real hardware, additional fix on the `H3_LOWVRAM_GROUP` side)

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

#### Shortened TE load time under `H3_LOWVRAM=1` (48GB tier)

Pruning shortens the per-request TE-load fixed cost that `H3_LOWVRAM=1` pays every time
(real hardware, free VRAM constrained to ~43.5GB via dummy VRAM allocation):

| | TE load time | TE size |
|---|---|---|
| Without pruning | 42.3s | 21.01GB |
| With pruning | **35.0s** (-17%) | 17.44GB |

The output mp4 is byte-identical (md5 match) regardless of pruning.

#### Regression check

Ran t2va with the same conditions with `H3_TE_PRUNE` unset (default `0`), and confirmed it
byte-matches (md5 match) the baseline mp4 captured before this feature was added. Default
behavior is completely unchanged.

### Disk cache of the quantized text_encoder (`H3_TE_PREQUANT`, default `1`, 2026-08-08)

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

### 2026-08-10: Projected TE (Qwen3-VL-4B + a trained linear map) — implementation, measurements, unverified items

Implemented as `H3_TE_PROJ` (default OFF; existing behavior does not change by a single
byte). It replaces the TE path from Qwen3-VL-32B with **Qwen3-VL-4B + a trained projection
matrix** ([ClipProj-MiniMax-H3](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3)). The
aim is to keep the TE and the transformer **simultaneously resident** on the 48GB box,
eliminating the swap fixed cost.

```
cond = ((h - mean_in) / std_in) @ W * std_out + mean_out     # h = the 4B's hidden_states[24]
cond[:, 0] = sink_out                                        # token 0 is the attention sink
```

#### Measurements (96GB box, t2i 768², 30 steps, seed 4242, identical prompt)

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

#### Tokenizer investigation (the basis of the implementation policy)

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

#### The second-GPU requirement drops

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

#### Addendum (same day): the NF4 quantization option, and quality confirmation on video

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

#### Addendum (same day, part 2): NF4 made the default, switchable from the UI

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

#### Unverified (important)

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

## transformer and denoising

### transformer int8 quantization (`H3_TRANSFORMER_QUANT`, default `none`)

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

### Denoise speedup via FirstBlockCache (`H3_CACHE`, default `fbc`)

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

### Sage Attention (`H3_ATTN_BACKEND`, default `sage`)

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

### Turbo LoRA (`H3_TURBO_LORA`, default `0`, 2026-08-06 / **switched to the lightx2v variant on 2026-08-08**)

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

#### Verification of turbo x other features combined (2026-08-06)

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

### Step-count guidance (distilled model, measured 2026-08-05)

`num_inference_steps` is an API/UI parameter. Relative to 30 (the verification default),
**20 gives -15%, 16 gives -31%** in denoise time. Both 16 and 20 show no breakdown in
single-frame quality or temporal stability (though composition does change with step
count). 16-20 is a reasonable guideline for drafts, 30 for production. Reducing steps also
reduces FBC skip opportunities (7 skips at 30 steps -> 0 at 16 steps), so the effect is not
simply proportional.

### 2x upscale via two-pass generation (`upscale=1` on `/api/t2va`, default OFF)

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

### Rounding of arbitrary size/duration (2026-08-06)

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

## Generation modes

### Ref2VA (omni-reference generation, `/api/ref2va`)

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

### Still-image mode (T2I, `/api/t2i`, production implementation 2026-08-07)

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

#### Measured result for consecutive generation (a sequence of narrative scene images with different prompts): **group mode is worse for still images**

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

#### Batch generation of still images (`/api/t2i_batch`, production implementation 2026-08-07)

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

#### Reference-conditioned still images (ref2i, `still=1` on `/api/ref2va` and `/api/ref2i_batch`, production implementation 2026-08-07)

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

#### Shared KV cache for the reference prefix (`H3_REF_PREFIX_CACHE`, default 1, production implementation 2026-08-08)

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

#### Reusing the prefix across single requests (`H3_REF_PREFIX_CACHE_SINGLE`, default 0, 2026-08-24)

Carries the shared prefix above across **separate single `/api/ref2va` requests**, not just
within one batch call. For workflows that pair one reference image with many different audio
tracks and prompts (exactly what mv_studio's lip-sync generation does), the ~52s reference
vision encode disappears from the second request onward.

**Default OFF.** It keeps a process-lifetime KV cache (measured 1.04GiB of VRAM) alive between
requests, so "do not change existing behaviour" wins by default. Enable it through the gateway
overrides: `{"backend":"h3","preset":"96gb","overrides":{"H3_REF_PREFIX_CACHE_SINGLE":"1"}}`.

**When it applies** (decided mechanically; anything else silently falls back to the legacy path):

- **Image references only** (any number of audio references is fine). `_build_presentation` emits
  only the `"<Audio j>: "` label for audio and the waveform never reaches the conditioner, so
  **changing the audio's content and length leaves the prefix token sequence bit-identical**.
- **Never used when any video reference is present** (a video's pixels do enter the prefix). That
  case drops the cache and falls back to the full encode.

The **key** is a SHA-256 of what actually enters the prefix forward (token ids + the image tensor's
bytes + `image_grid_thw`), not of filenames — the count, order, resolution or pixels of the
references all change the key. Key computation measured at 0.14-0.35s.

**Safeguards the batch version does not need** (see `_encode_ref2va_prompt_prefix_cached`'s
docstring):

1. `model.rope_deltas` is *instance state* on `Qwen3VLModel`: the prefix writes it, the
   continuation reads it. Across requests, a t2va encode or `/api/prompt/enhance` can overwrite it
   in between, so the value is cloned at prefix time and **written back right before every
   continuation**.
2. A `weakref` checks text_encoder identity — a freed/reloaded/re-quantized TE invalidates the
   cache. `_free_text_encoder()` also drops it actively so the VRAM comes back immediately.
3. `DynamicCache.crop(prefix_len)` after every continuation (plus a length check before it, as
   insurance against an exception unwinding mid-continuation).

**Measured** (96GB box, preset=96gb + `H3_TRANSFORMER_QUANT=int8`, 768x448 / 8s / turbo / 4 steps /
seed 777, one image + one audio reference):

| | 1st | 2nd | Peak VRAM |
|---|---|---|---|
| OFF (default, legacy path) | 136.6s | **94.5s** | 73.79GB |
| **ON** | 140.4s (MISS, prefix 52.5s) | **41.9s (-56%)** | 74.92GB (+1.1GB) |

- Same image + **different audio** (a different 6.6s file vs 8.0s) also hits: 37.0s.
- Swapping the reference image logs `cache cleared` and misses (93.6s); swapping back misses again
  (94.1s) but restores the original key (keys are deterministic).
- The ref batch endpoints benefit too, since in resident mode they loop `generate_ref2va()`:
  the second scene of ref2i_batch went 71s -> **21s**.

**Numerics**: a MISS (prefix computed in this request) and a HIT (prefix reused) produce
**bit-identical** output (`max diff = 0` over 192 frames and the audio), so cross-request reuse
itself is exactly equivalent. Against the legacy path (one monolithic encode) it is *not*
bit-identical — the ~1.5% rounding difference from splitting prefix/continuation is amplified by
turbo's 4 steps, the same property and the same cause the batch path (`H3_REF_PREFIX_CACHE`)
already accepts:

| Comparison | Mean pixel diff | PSNR | Audio correlation |
|---|---|---|---|
| ON (HIT) vs OFF | 2.46/255 | 31.7dB | 0.9909 |
| ON (MISS) vs ON (HIT) | **0 (bit-identical)** | inf | 1.0 |

Visually the composition, character identity and even mouth shape are indistinguishable. Keep
`H3_REF_PREFIX_CACHE_SINGLE=0` (default) for A/B tests that need bit reproducibility.

**Interaction with H3_LOWVRAM**: lowvram frees the TE every request, so the cache is dropped every
time (no benefit). Verified on hardware with 48gb-lowvram + ON: every request logs
`cache cleared: text_encoder freed`, the next one misses, and there is no VRAM leak or exception
(with TE prune the 51-layer TE makes the cache 0.84GiB, sized automatically).

#### Batch generation of reference-conditioned videos (`/api/ref2va_batch`, production implementation 2026-08-08)

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

#### Spike: Ref2VA x ultra-short clips (reference-conditioned still images, 2026-08-07, measured -> productionized as described above)

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

#### The preceding ultra-short-clip probe (2026-08-07, measurement record)

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

## UI

### Switching settings from the UI (2026-08-06)

Opt-in settings that could previously only be changed via environment variables can now be
operated from the UI, split into two categories by nature.

#### Instant apply (checkbox directly above the generate button, no reload needed)

FirstBlockCache (+ threshold), Sage Attention, Turbo LoRA. Sent as per-request parameters
(`cache` / `cache_threshold` / `attn` / `turbo`), applied by `MiniMaxH3Runner.
apply_instant_settings()` to the resident transformer after acquiring the generation lock
and before denoise (`disable_cache()`/`enable_cache()`, `set_attention_backend()`,
`_TurboLoRALinear.enabled`). **If unspecified, the process default applies as before**, so
existing curl/scripts work unchanged. Enabling turbo automatically disables FBC (following
the original safety rule).

Measured (same seed, switched without restart): FBC on 100.8s (6 skips) / off 129.3s
(0 skips), Sage on 129.3s / off (native) 158.5s, Turbo on (8 steps) 38.9s.

#### Requires reload (collapsible header + "Apply (reload)" button)

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

### Gallery of generated videos (2026-08-06)

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

## LLM prompt enhancement

### LLM prompt enhancement (added 2026-08-04)

Uses a local LLM (gemma4-31B, OpenAI-compatible `/v1/chat/completions`) to reformat input
prompts into the H3 official guide's format. A local reproduction of the cloud Hailuo AI's
internal prompt-formatting layer.

- Connection target: environment variable `H3_LLM_URL` (default
  `http://127.0.0.1:64660`). Returns 502 if unreachable (does not affect the generation
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

### Real-hardware verification of the official skill (h3-prompt-writing) mode (2026-08-07)

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

### h3-official quality assurance: verification + repair loop (2026-08-08)

To avoid arguing by speculation about "is this an LLM capability limit, or fixable via
prompting," **first measured the failure rate** (`scripts/probe_h3official_compliance.py`,
5 inputs x 3 runs), then applied countermeasures based on the results. LLM: gemma4-31B
Q4_K_M / llama.cpp, **n_ctx=7680**.

#### Baseline measurement: failures were concentrated in a single class

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

#### Countermeasures (3 layers)

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

#### Measured results after improvement (same conditions)

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


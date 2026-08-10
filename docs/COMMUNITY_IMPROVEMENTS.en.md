# Community Improvements Adoption Log (MiniMax H3)

[日本語](COMMUNITY_IMPROVEMENTS.md) | **English**

A record of work adopting improvements from the ComfyUI community and elsewhere into this
app (the diffusers path), starting from the original (MiniMaxAI/MiniMax-H3 + diffusers PR #14355). 2026-08-04 through 08-06.

All verdicts are based on **real-machine A/B comparisons at the same seed**. "Quality parity"
means confirmed by visual inspection + PSNR + audio correlation; "equivalent" means confirmed
down to mp4 MD5/byte match.

---

## A. Adopted (community-originated)

### A-1. FirstBlockCache (equivalent to ComfyUI's EasyCache) — ON by default

| | |
|---|---|
| Source | ComfyUI community (kijai's EasyCache usage; posts by @gosrum / @umiyuki_ai) |
| Community reports | 1.67x with EasyCache; 6min → 2.5min combined with Sage |
| Adoption approach | Adopted diffusers' official `FirstBlockCache` (a similar step-to-step caching scheme). `H3_CACHE`/`H3_CACHE_THRESHOLD` |
| Measured | Denoise 157s → **118s (-25%)**, 7 of 30 steps skipped. At threshold 0.1, 81.5s (1.92x) but composition drift |
| Quality | PSNR 31.8–34.3dB, audio correlation 0.979. Indistinguishable by eye |
| Verdict | **threshold 0.05 made default**. 0.1 is opt-in |
| Gotcha | H3 blocks are not registered in the PR branch's `TransformerBlockRegistry` → requires own registration. Must call `_reset_stateful_cache()` + `cache_context()` per request (verified via byte match across two consecutive runs at the same seed) |
| Commit | `14afdfc` |

### A-2. Sage Attention — ON by default

| | |
|---|---|
| Source | Same post as above ("Sage alone cuts 25%") |
| Adoption approach | **Built thu-ml/SageAttention from source** for sm_120 (no prebuilt wheel exists for Linux; only a Windows build is published). `H3_ATTN_BACKEND` |
| Measured | Denoise 118s → **104s (-12%)**. Falls short of the community-reported -25% |
| Quality | Fully deterministic (byte match across two runs at the same seed). Visually equivalent (the 21dB PSNR is trajectory drift from the int8-QK approximation, not degradation) |
| Verdict | **Made default** (`H3_ATTN_BACKEND=default` reverts to conventional SDPA) |
| Gotcha | Build requires `MAX_JOBS=4 NVCC_THREADS=2` plus a systemd-run memory cap (unrestricted parallel nvcc has a history of exhausting host RAM and taking the whole system down). Must explicitly set `CUDA_HOME=/usr/local/cuda-12.8` (the default cuda-13.0 mismatches torch cu128 and the build fails) |
| Commit | `9c7e6a6`, `scripts/build_sageattention.sh` |

### A-3. Latent upscaler (two-stage generation / hires-fix) — opt-in

| | |
|---|---|
| Source | [Tr1dae/ComfyUI-MiniMaxH3_LatentUpscaler](https://github.com/Tr1dae/ComfyUI-MiniMaxH3_LatentUpscaler) (via a post by @umiyuki_ai) |
| Mechanism | First-half denoise at low resolution → spatial 2x interpolation of the video latent only → renoise → finish at high resolution. No trained upscaler used |
| Adoption approach | `upscale=1` on `/api/t2va` / `/api/i2v`, `H3_HIRES_DENOISE` |
| Measured | 768² → **1536²** takes 645s / peak 88.0GB (upscale=0 is 181s / 92.1GB) |
| Quality | Composition stays consistent while real detail (e.g. fur) is added. Background detail drifts slightly from the re-denoise (inherent to hires-fix) |
| Verdict | **opt-in** (OFF by default) |
| Gotcha | **The interpolation target must be the x0 estimate, not the noisy latent.** Interpolating the noisy latent amplifies checkerboard noise into full-frame noise (reproduced on real hardware → confirmed the reference implementation also uses `denoised_output` and fixed accordingly). Changing resolution requires rebuilding `build_packed_sequence()` and `row_timestep_plan` |
| Commit | `e9a45a7` |

### A-4. Turbo LoRA (4/8-step distillation) — opt-in

| | |
|---|---|
| Source | [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) (trained by Ostris, Apache 2.0). Practical info obtained from a post by @PhotogenicWeekE |
| Community reports | 1280×736/10s at 8 steps goes from 272s → 160s. **"4–7 steps don't work"** |
| Adoption approach | `H3_TURBO_LORA=1` (8 steps by default). Uses the original, not the ComfyUI repack |
| Measured | 8 steps **87.7s (-46%)** / 16 steps 98.4s / 4 steps 39.6s (baseline 30 steps is 163.5s) |
| Quality | 8 steps approaches the baseline. 16 steps is on par with the baseline. 4 steps is somewhat soft, with weaker audio |
| Verdict | **opt-in** (OFF by default because the author explicitly labels the LoRA "demo/preview, still training"; set up so that once a finished version is released, it can be swapped in and the default decided from A/B alone) |
| New finding | **"4–7 steps don't work" is likely caused by ComfyUI's standard sampler.** In this implementation (which correctly integrates the dual schedule of video shift 12 / audio shift 3), no audio corruption occurred even at 4 steps. Note that no fix was needed for the shift wiring — bit-exact match of the sigma grid confirmed the PR implementation's default values were already correct |
| Gotcha | LoRA keys are in ComfyUI's fused-QKV naming convention → applied via `fuse_projections()` plus a runtime delta. **`fuse_projections()` does not delete the old to_q/k/v, leaking +12.8GB** (discovered via an actual OOM, fixed with an explicit delete) |
| Commit | `2ab3100` |

### A-5. Block-level group offload (24–32GB class support) — opt-in

| | |
|---|---|
| Source | ComfyUI's "runs on 24GB with INT8 + layerwise offload" report + PR #14355's streamed offload support + knowledge established in the sister project (diffusers-server) |
| Adoption approach | `H3_LOWVRAM=group`. Load the int8 transformer on CPU → `enable_group_offload(block_level, num_blocks_per_group=1, use_stream=True)` |
| Measured | **peak 28.7GB** under a 32GB cap, denoise 220.8s (~2.1x the resident-mode figure) |
| Equivalence | **mp4 MD5 match** with the normal int8 mode |
| Verdict | **opt-in** |
| Gotcha | **`use_stream=True` combined with `low_cpu_mem_usage=True` crashes on torchao `Int8Tensor`'s pin_memory** (`cannot pin 'torch.cuda.CharTensor'`). Worked around with `low_cpu_mem_usage=False` (also speeds up onload 4–5x). → A real bug worth reporting upstream to diffusers |
| Commit | `26bf434` |

---

## B. Investigated but not adopted

| Item | Source | Conclusion |
|---|---|---|
| **VAE device/dtype cast fix** | [ComfyUI commit 16e3f30](https://github.com/Comfy-Org/ComfyUI/commit/16e3f3034f2bba1fff6c70cbd759339778555cd6) (@PhotogenicWeekE) | **Not needed.** This fixes a crash caused by ComfyUI's own weight management (per-layer casting at compute time) letting a raw `nn.Parameter` pass through unchanged. diffusers moves entire modules with `.to(device)`, and since VAE fp32 is fixed by the PR's explicit contract, the same kind of mismatch cannot occur (confirmed in the actual code) |
| **NVFP4 TE variant** | Comfy-Org (14.6GB, ComfyUI format) / RedHatAI (20.4GB, compressed-tensors) | **Rejected.** The ComfyUI format requires an in-house implementation of fp4 arithmetic; the RedHatAI version is nearly the same size as nf4's 21GB and does not solve the 24GB problem (its runtime also assumes vLLM) |
| **Hub-based attention backends** | diffusers `flash_hub` / `sage_hub` | **Not viable.** No torch 2.9-targeted build exists on the Hub side (only 2.10–2.12 are available). Not an environment issue |
| **ComfyUI's pruned TE file (47.97GB)** | Comfy-Org/MiniMax-H3 | Did not use the file itself; **independently derived the same reduction and implemented it in-house** (C-1 below). This avoids the ComfyUI format conversion and lets the existing bnb-nf4 path be used as-is |
| **torchao's C++ kernel** | torchao 0.18 | **Passed on.** Requires torch>=2.11, posing a large regression risk to the whole venv. Currently running on the 0.17 pure-Python fallback |

---

## C. Own implementations inspired by community findings

### C-1. Removing unused upper layers of text_encoder (`H3_TE_PRUNE`)

Investigated why the bf16 TE distributed by ComfyUI is 47.97GB (-14.2GB versus the full
62.13GB) and discovered that **H3 only reads `hidden_states[50]` of the TE** (the 14 layers
from layer 51 onward, ≈13GB, are dead weight), then implemented the equivalent in-house.

- Measured: TE-nf4 **21.02 → 17.45GB (-17%)**, bf16 66.71 → 53.06GB
- Equivalence: both t2va and ref2va produce an **mp4 MD5 match** against the unpruned version (also confirmed `torch.equal` against the 64-layer version)
- **Gotcha (most important): pruning to exactly 50 layers causes transformers'
  `tie_last_hidden_states` to silently overwrite `hidden_states[50]` with the value after
  the final norm is applied, producing numerically different results.
  Pruning to 51 layers is correct** (this is exactly why diffusers has a guard of
  `num_layers <= 50 → raise`)
- This completes support for the 24GB class (confirmed working even at a measured 20GB)
- Commit: `2d31424`

### C-2. Transformer int8 quantization + keeping both variants resident simultaneously

Applied the recipe documented in PR #14355 (`Int8WeightOnlyConfig`), and additionally kept
both the transformer and transformer_ref resident at the same time to eliminate the switching
cost between Ref2VA and T2VA.

- Measured: transformer 66.3 → **34.0GB**. ref2va 523s → **463–471s**; the 66GB-class reload on variant switching is eliminated
- Gotcha: `expandable_segments:True` is required (int8 load/free cycles fragment memory, and "only 54GB in use yet a 15GB allocation fails" was reproduced on real hardware)
- Commit: `236a424`, `435f831`

### C-3. Other in-house implementations

| Item | Details | Commit |
|---|---|---|
| TE bnb-4bit | TE 66.7GB → 21.0GB. 245s → 185s (eliminates TE⇔transformer swapping) | `6526b61` |
| 48GB-class phase rotation | Never keep TE and transformer resident simultaneously. t2va peak 38.9GB | `2bb3127` |
| Video VAE fp16 | Decode peak 16.3 → 11.4GB, PSNR 39.97dB. Gotcha: `_keep_in_fp32_modules` overrides the dtype setting | `a6c5ffa` |
| Local LLM prompt enhancement | Three modes: storyboard / brief / translate | `02e311f` |
| Ref2VA | Ordered references — 9 images/3 videos/3 audio (wired up the PR's feature; resolved 3 real-machine OOMs) | `ca5d912` |

---

## D. Current state (768², 5 seconds)

| Configuration | Request time |
|---|---|
| Initial (bf16 TE swapping) | 245s |
| Current default (bnb-4bit + FBC 0.05 + Sage) | **~160s** |
| + FBC 0.1 (opt-in) | ~125s |
| + Turbo LoRA 8 steps (opt-in) | **~88s** |
| + Turbo 4 steps (for draft use) | ~40s |

VRAM floor: 96GB → **~18GB** (`H3_LOWVRAM=group H3_TE_PRUNE=1`).
Every step has been confirmed equivalent via either an MD5 match at the same seed or a quality A/B.

# MiniMax-H3 Test App — Technical Overview

[日本語](TECHNICAL_OVERVIEW.md) | **English**

## 1. What This App Does

MiniMax H3 (Hailuo 3.0) is an omnimodal 33B model that **generates video and stereo audio simultaneously in a single denoising pass**. Unlike conventional pipelines that overlay audio in a later stage, video and audio are denoised together, as separate "rows" on the same packed sequence, within a shared self-attention.

This app is a test app that runs this model via diffusers' **Modular Diffusers** path (the implementation provided in PR #14355). diffusers support for MiniMax-H3 is provided only in this PR, and the app was built to verify operation on diffusers independently of the ComfyUI implementation. It is a preliminary workspace for eventually integrating the functionality into [diffusers-server](https://github.com/animede/diffusers-server); diffusers-server itself has not been touched at all.

The server is built with FastAPI and listens on port **8611**. The UI is a single page (`static/index.html`) that supports switching between Japanese and English.

### Dependencies

| Dependency | Version / Pin | Reason |
|---|---|---|
| diffusers | `f37ab93e621d5ce206c9662e8291ca8b67d9c555` (final state after PR #14355 merge) | The Modular Pipeline implementation for MiniMax-H3 exists only in this PR |
| transformers | `5.14.1` or later | `Qwen3VLProcessor.create_mm_token_type_ids` is required (not present in 5.1.0) |
| torch | `2.9.0` (cu128) | Compatible with the CUDA 12.8 series |
| accelerate / safetensors / huggingface_hub | ordinary latest series | Model loading |
| bitsandbytes | `0.49.0` | NF4 quantization of text_encoder (required for the default path) |
| torchao | `0.17.0` | int8 quantization of the transformer (`0.18` and later were not adopted because they require torch>=2.11) |
| av / fastapi / uvicorn | `16.0.1` / `0.104.1` / `0.24.0` | Video/audio muxing and the Web API |

diffusers is operated as a **commit pin**. All paths (t2i/t2va/batch/ref2va/ref batch) have been regression-tested against the old pin via matching same-seed MD5s, and the policy is to follow the same procedure when advancing further.

---

## 2. Features Provided

### Generation Modes

| Mode | Input | Output | Endpoint |
|---|---|---|---|
| T2VA | Text prompt | Video + stereo audio | `POST /api/t2va` |
| FL2VA | Text + first/last frame image(s) (at least one of the two) | Video + stereo audio | `POST /api/fl2va` |
| Ref2VA | Text + ordered references (up to 9 images, up to 3 videos, up to 3 audio, 12 total) | Video + stereo audio | `POST /api/ref2va` |
| T2I (still image) | Text prompt | Still image (PNG) + ultra-short mp4 | `POST /api/t2i` |
| Ref2I (still image with reference) | Text + reference | Still image (PNG) | `POST /api/ref2va` (`still=1`) |

T2I and Ref2I are modes that substitute for image generation by "generating an ultra-short video and extracting the center frame." Their value is not speed relative to a dedicated T2I model, but that they can produce **still images whose style exactly matches H3**, for use as the first frame of FL2VA or as a reference for Ref2VA.

### Batch Generation

| Endpoint | Content | Shared across the batch | Variable |
|---|---|---|---|
| `POST /api/t2i_batch` | Batch of still images (up to 24 scenes) | frames, resolution, steps, seed | prompt (1 line = 1 scene) |
| `POST /api/ref2i_batch` | Batch of still images with references | references, frames, resolution, steps | prompt (per scene) |
| `POST /api/ref2va_batch` | Batch of videos with references | references, seconds (same for all scenes, required) | prompt (per scene) |

All of these are designed to amortize, once over the entire batch, the fixed cost of loading/freeing the model under `H3_LOWVRAM=1` (details in §4). In modes other than `H3_LOWVRAM=1` (large model resident), there is no gain from phase reordering, so the same API falls back to sequential generation.

### LLM Prompt Enhancement

`POST /api/prompt/enhance` uses a local LLM (default `H3_LLM_URL=http://127.0.0.1:64650`, assuming gemma4-31B Q4_K_M) to reformat prompts into the notation of the official H3 skill (`h3-official`).

- **Structure**: T2VA has 3 fields (`integrated_multimodal_description` / `overall_soundscape` / `non_diegetic_music`); Ref2VA has 6 fields. It outputs the `[Shot n]` cut notation and the `<d>[language] line</d>` speaker-tagged dialogue notation according to the official specification.
- **Validator** (`core/prompt_check.py`, rules F1–F8): mechanically checks 8 rules — field order, absence of a timestamp on the first shot, strictly increasing cut timestamps, whether it fits within the duration, minimum shot duration, `<d>` tag consistency, whether dialogue fits within the duration, and speaker ID. F5 (minimum shot duration) and F7 (dialogue fitting within duration) are not in the official spec; they are rules this app added for practical purposes.
- **Repair loop** (`enhance_prompt_checked` in `core/llm.py`): when a violation is detected, it is presented back to the model for regeneration (up to 2 times, `H3_OFFICIAL_MAX_REPAIRS`). A repair candidate that increases violations is discarded. If the input itself is infeasible (e.g., dialogue cannot fit in the duration), this is judged before sending it to the LLM, and it is rejected with a reason.
- Generation is not blocked. The validator's findings are only displayed in the status area; the final decision is left to the human (prompt editing).

### hires-fix and turbo

- **Two-stage generation (hires-fix)**: `upscale=1` on `/api/t2va` (default OFF). The first half is denoised at low resolution, only the x0 estimate of the video latent is spatially interpolated 2x, and fresh noise is re-injected to finish at high resolution.
- **turbo LoRA** (`H3_TURBO_LORA`, default OFF): applies a 4/8-step distillation LoRA, reducing the number of denoising iterations itself.

Detailed figures and the conditions under which each holds are covered in §4 and §6.

### UI

The single-page UI consists of 5 tabs in a 2-column layout (video tabs: T2VA / FL2VA / Ref2VA, still-image tabs: T2I / Ref2I). Each tab has a batch-generation checkbox (1 line = 1 scene). Generated results are collected in a gallery that tiles the mp4/PNG files under `outputs/`, and videos support lossless concatenation in selection order (`concat demuxer + -c copy`, re-encoding only when parameters mismatch). It supports switching between Japanese and English.

---

## 3. Architecture

### A design that manually calls the Modular Pipeline blocks individually

diffusers support for MiniMax-H3 is provided as a set of Modular Diffusers blocks. Rather than calling the `ModularPipeline` as a whole, this app adopts a design where **individual blocks are called in sequence from its own code**.

```
MiniMaxH3SetupStep            Resolve canvas / align frame count to 17n+5 / prepare keyframes
MiniMaxH3TextEncoderStep      Prompt encoding
MiniMaxH3PrepareLayoutStep    Packed sequence layout / rotary positions
MiniMaxH3PrepareLatentsStep   Latent initialization
MiniMaxH3SetTimestepsStep     Sigma schedules for the two video/audio tracks
MiniMaxH3DenoiseStep          Denoising loop
MiniMaxH3VideoDecodeStep /
MiniMaxH3AudioDecodeStep      Decoding
```

The reason for this decomposition is that **it is necessary to control, phase by phase, which model is currently loaded on the GPU**. With the standard usage of calling the pipeline as a whole, this control point does not even exist. In environments where VRAM cannot hold all components simultaneously (text_encoder + transformer + VAEs together are about 144GB), the design presupposes being able to insert model load/free at phase boundaries. For the same reason, a modification such as hires-fix, which inserts processing partway through the denoising loop, cannot be implemented unless the blocks are driven manually.

The cost of this design is a strong dependency on diffusers-internal state contracts such as `get_block_state()` / `set_block_state()` / `PipelineState`. A block's outputs (`num_frames`, `keyframes`, latent shapes, etc.) are stored in `PipelineState`, and since `get_block_state()` only maps declared inputs, outputs must be read via `state.get(name)`.

### Phase Structure

A single generation passes through the following phases in order. The boundary of each phase is the unit at which model load/free is inserted.

```
setup → encode → layout/latents/timesteps → denoise → after-denoise → decode
```

- **setup**: Aligns the canvas size and frame count to H3's rules (multiple of 32, `17n+5` frames).
- **encode**: Encodes the text_encoder (and the FL2VA keyframes, the Ref2VA references).
- **layout/latents/timesteps**: Assembles the packed sequence layout and rotary positions, the latent initialization, and the sigma schedules for the two video/audio tracks. Depending on the mode, information dependent on text_encoder may still be needed here, so the text_encoder is kept resident through this phase in some modes (see §4, §5).
- **denoise**: The denoising loop by transformer (or transformer_ref). Under VRAM constraints, this phase produces the peak VRAM.
- **decode**: Decodes with the video VAE and audio VAE.

### A structure in which both the transformer and transformer_ref slots live in a single pipe shell

Ref2VA uses a dedicated checkpoint, `transformer_ref/` (same class and config as `transformer`, weights only differ). The text_encoder, VAEs, and processor are shared between both variants, and a single pipeline shell holds both the `transformer` and `transformer_ref` slots. In configurations with ample VRAM (int8 both resident, see §5), both are kept resident simultaneously, eliminating the T2VA⇔Ref2VA switching cost. In VRAM-constrained configurations, the approach switches to "keep only the active one resident, and free→reload on variant switch" (the currently resident variant can be checked via `active_variant` in `/api/status`).

### Server Configuration

- Single FastAPI process. Generation is serialized with a **single global lock allowing only one concurrent request** (to avoid running GPU-occupying processes concurrently).
- Progress can be polled for long-running generations via `GET /api/progress`.
- `GET /api/status` returns the load state and measured VRAM/RAM values.
- Settings that apply immediately (FirstBlockCache, Sage Attention, Turbo LoRA) can be sent as request parameters and are applied after acquiring the generation lock and before denoising. Settings that require a reload (quantization scheme, low-VRAM mode, video VAE precision) are switched explicitly via `POST /api/settings/apply` (the process does not restart; the runner frees and reloads the model internally).

---

## 4. Integration of the Various Techniques

### Quantization

| Target | Method | Effect |
|---|---|---|
| transformer | torchao `Int8WeightOnlyConfig(version=2)` (`H3_TRANSFORMER_QUANT=int8`) | 66.3GB → **34.0GB** |
| text_encoder | bitsandbytes NF4 (`H3_TE_QUANT=bnb-4bit`, default, compute_dtype=bf16) | 66.71GB → **21.02GB** |
| Removal of unused upper layers of text_encoder | `H3_TE_PRUNE=1` | nf4 21.02GB → **17.45GB** (-17%), bf16 66.71GB → 53.06GB (-20%) |

Of text_encoder (Qwen3-VL-32B, 64 layers), only `hidden_states[50]` is actually read. `H3_TE_PRUNE=1` builds 51 layers (0 to 50; the output of `layers[50]` itself is not read, but the computation is still performed), and never loads the unused layers 52–64, the final `norm`, or `lm_head`. **Truncating to exactly 50 layers produces an incorrect value** (because transformers' `tie_last_hidden_states` mechanism overwrites the last element of the captured tuple with the value after the final `norm` is applied). Reducing to 51 layers is the correct boundary, and it has been confirmed to be bit-identical (`torch.equal`) to `hidden_states[50]` of the 64-layer version.

For int8 quantization, NF4 quantization, and layer removal alike, it has been confirmed that the output mp4/PNG is byte-identical (MD5-identical) with and without the removal/quantization, and these are treated as mathematically inconsequential optimizations.

### Attention

| Method | Environment variable | Effect |
|---|---|---|
| Sage Attention 2.2.0 (source build for sm_120) | `H3_ATTN_BACKEND=sage` (default) | Denoise 118s → **104s (-12%)** |
| FirstBlockCache | `H3_CACHE=fbc` (default), `H3_CACHE_THRESHOLD=0.05` (default) | Denoise 157s → **118s (-25%)**, 7 of 30 steps skipped |

FirstBlockCache is diffusers' official caching mechanism that skips the remaining computation when the change in the transformer's first block's residual between steps is small. Raising the threshold to 0.1 gives up to a 1.92x speedup, but it is not the default (opt-in) because the composition visibly drifts. Quality is judged visually indistinguishable at PSNR 31.8–34.3dB and audio correlation 0.979. Sage Attention is fully deterministic (two runs with the same seed are byte-identical); its PSNR of 21dB is judged to be trajectory drift from the int8-QK approximation, not degradation.

The two operate on independent layers and can be combined (sage + threshold 0.1 gives -43% denoise time).

### Distillation (Turbo LoRA)

`H3_TURBO_LORA` (default OFF, can also be opted into via the request's `turbo=1`) applies a 4/8-step distillation LoRA, reducing the number of denoising iterations itself. The default is the **lightx2v** format (`lightx2v/Minimax-h3-Turbo`, DMD distillation, Apache 2.0, rank128, targeting 312 Linear layers, default 4 steps).

- **Applied scale** (`H3_TURBO_LORA_SCALE`) is **0.094**. The 0.75 listed by the LoRA distributor assumes ComfyUI's alpha folding; applying 0.75 directly to the raw B·A turns the output into complete noise even at 30 steps.
- **Why it can be combined with an int8-quantized transformer**: the lightx2v format's keys are diffusers-native (to_q/to_k/to_v separated), so applying it does not require `fuse_projections()` (which requires `torch.cat`). The older-generation comfy format (Ostris version, fused `qkv_proj`) requires `torch.cat`, and since `aten.cat` kernels are not implemented for int8-quantized `Int8Tensor`, it remains unusable in int8/low-VRAM mode. The apply function auto-detects the key format.
- **Combination restriction**: it cannot be combined with `H3_LOWVRAM=group`, regardless of format (because `enable_group_offload`'s `cpu_param_dict` is fixed at the time it is enabled).
- When turbo is enabled, FBC is automatically disabled.

### Offloading

`H3_LOWVRAM=group` (24–32GB class) uses diffusers' `enable_group_offload(offload_type="block_level", num_blocks_per_group=1, use_stream=...)`, keeping the int8-quantized transformer **resident in host RAM** while streaming only the blocks needed at each denoise step (1–2 of 50 layers, about 0.68GB each) to and from the GPU — block-level group offload. The transformer is loaded only once at process startup and stays resident across requests.

Even when loaded onto the CPU with `device_map={"transformer": "cpu"}`, int8 quantization is applied correctly (confirmed on real hardware that 370/370 layers become `Int8Tensor`). The combination of diffusers' default `use_stream=True` + `low_cpu_mem_usage=True` has a bug that reliably crashes against torchao's `Int8Tensor` with `cannot pin 'torch.cuda.CharTensor'`; this is avoided by adopting `low_cpu_mem_usage=False` (`H3_GROUP_OFFLOAD_LOW_CPU_MEM`, default 0=False). This setting also has the side effect of making onload 4–5x faster (0.04–0.07s/block versus 0.1–0.26s/block).

### Three-Stage Reduction of the Fixed Cost

`H3_LOWVRAM=1` (48GB class) cannot keep TE (17.45–21GB) and transformer-int8 (34GB) resident at the same time, so it repeats load/free on every request. This fixed cost was reduced in three stages.

1. **On-disk cache of the quantized text_encoder** (`H3_TE_PREQUANT`, default ON): saves the bnb-4bit quantized weights once, so subsequent runs only need to load them. TE load average 53.0s → **29.5s**.
2. **Keeping TE resident on a second GPU** (`H3_TE_DEVICE=cuda:1`): keeps TE resident on a second GPU, reducing TE load time to zero for subsequent requests. Steady-state time for t2i turbo 4steps averages 78.4s → **about 35s (-55%)**.
3. **Keeping transformer resident** (`H3_KEEP_TRANSFORMER=1`): does not free the transformer even during the decode phase. There are 3 conditions for this to hold (see §5.5). t2i turbo 4steps drops to a steady-state **9.7s/image**.

Output equivalence has been confirmed at every stage via same-seed MD5/PNG exact match (only for moving TE to a second GPU, architecture differences between sm_120 and sm_89 cause bit mismatches from rounding error, but the relative RMS difference of 0.084% stays within the level of trajectory drift).

### fp16-ification of the video VAE

`H3_VIDEO_VAE_FP16=1` converts only the video VAE weights to fp16 (9.70GB → 4.85GB, decode peak 16.29GB → about 11.4GB). The audio VAE stays in fp32 and is never cast (because converting it to bf16 has a known issue of reducing the generated audio's volume by about 20dB). Quality is a mean PSNR of **39.97dB** (min 39.08) across all 124 frames, visually indistinguishable.

### KV Prefix Sharing for Reference Batches

`H3_REF_PREFIX_CACHE` (default 1) resolves a problem in the encode phase of ref batches (ref2i_batch / ref2va_batch), where the Qwen3-VL encoding of the reference labels + vision (about 4,104 tokens, about 65 seconds/scene) was being duplicated for every scene. The token sequence for ref2va has a structure where "references come first, and the prompt is appended verbatim at the end," and since the conditioning source is a causal LM, **the representation of the reference prefix does not depend on the prompt**. The prefix is run once with `use_cache=True`, baked into a `DynamicCache`, and for each scene only the prompt tail (14–33 tokens, about 0.2 seconds) is continued against the cache.

The `hidden_states[50]` of the prefix portion is bit-identical (`torch.equal`) to the full computation. The prompt-tail side retains a relative RMS difference of about 1.5% from rounding, but a negative control that intentionally breaks the position offset jumps to a relative RMS of 27–30% (20x), confirming that this is genuine rounding noise from a correct computation, not a logic bug. Effect: encode phase of the ref2i batch 212.5s → **83.1s**, per-image 164.9s → **116.7s (-29%)**.

### Batch Phase Reordering

The per-request fixed cost of `H3_LOWVRAM=1` (TE load + transformer load, about 90–110 seconds) is amortized once over the entire batch by **reordering the phases from per-request to per-batch**.

```
entry   : [nothing resident]
encode  : [TE-nf4]         setup/encode/layout/latents/timesteps for all scenes
denoise : [transformer]    denoise all scenes in sequence
decode  : [VAE pair]       decode all scenes → save (saved as each scene finishes)
```

The key implementation detail is resetting the mutable state shared across scenes. Because the scheduler's sigma/timestep values are identical across all scenes (same geometry, same step count), it suffices to reset `_step_index = None` (since `MiniMaxH3Scheduler.step()` re-derives the index from the timestep value), and FirstBlockCache calls `_reset_stateful_cache()` + `cache_context` per scene. Matching mp4/PNG MD5s against sequential generation demonstrates that the phase reordering is mathematically inconsequential.

---

## 5. Handling by VRAM Capacity

### How to Derive a Configuration From Capacity

The mode can be derived as a function of VRAM capacity. When the GPU changes, re-derive from the following parts table and inequalities rather than looking up the table from memory.

**Parts table (all measured)**

| Part | Size |
|---|---|
| text_encoder bf16 | 66.71GB (53.06GB with 51-layer removal) |
| text_encoder nf4 | 21.02GB (17.45GB with 51-layer removal) |
| transformer bf16 | 66.3GB |
| transformer int8 | 34.0GB |
| transformer_ref bf16 / int8 | 61.7GB / about 34GB |
| vae + audio_vae (fp32) | 11.0GB |
| Denoise activations | about 5–6.6GB (measured 6.6GB at 768², 5 seconds) |
| Decode peak | 16.29GB (about 11.4GB with video VAE fp16) |
| Additional cost of ref2va reference encoding | +3.2GB or more against TE (vision tower at 2048px short side, measured lower bound) |
| CUDA context etc. (non-PyTorch) | about 1GB |

**Inequalities to satisfy (independent per phase)**. Only things within the same phase need to be resident simultaneously; there is no need to sum across phases.

```
Effective budget = Catalog capacity − unit difference (about 0.5GB) − CUDA context etc. (about 1GB)

Encode : TE                                      ≤ Effective budget
Denoise: transformer + activations (about 6.6GB) ≤ Effective budget
Decode : decode peak (16.29 / 11.4 if fp16)       ≤ Effective budget
```

If something needs to stay resident across requests, add its size to each phase (e.g., if you want to denoise while keeping TE resident, `TE + transformer + activations ≤ capacity`).

> **Unit trap**: `nvidia-smi` reports MiB, PyTorch's OOM messages report GiB, and this app's logs report GB (decimal); a 20GB card shows as 21.47GB (decimal) in `nvidia-smi`, but the effective capacity visible from PyTorch is about 20.99GB (decimal). With a further ~1GB subtracted for non-PyTorch overhead on top of that, using the catalog capacity directly as the budget overestimates it by about 1.5GB.

### Recommended Configuration Table by Capacity

| Capacity | Configuration that holds | Bottleneck (why it can't go further) |
|---|---|---|
| 96GB | TE+transformer resident in bf16 (87.5GB) | 3 things at once during decode (98.5GB) don't fit, so transformer must be surrendered |
| 80GB | TE+transformer+transformer_ref resident in int8 (89GB) | Same as above |
| 48GB | `H3_LOWVRAM=1`. TE(17.45)+transformer(34)+activations(6.6)=58GB doesn't fit → swapped every time | No room to place TE is the cause of the fixed cost |
| 32GB | `H3_LOWVRAM=group`. transformer resident in RAM, streamed block by block | TE(21)+decode(16.3)=37GB > 32GB, so TE must be surrendered during decode |
| 24GB | `group` + `H3_TE_PRUNE=1` | TE-nf4's 17.45GB is most of the budget. Won't fit without removal |
| 18GB | Same as above (measured floor) | The TE-nf4-removed version's 17.45GB itself is the floor |
| 16GB | Not possible | OOM near the end of TE loading. Breaking through this requires streaming execution of TE |

### Requirements for Placing TE on a Second GPU

Specifying a GPU via `H3_TE_DEVICE` keeps TE resident on that GPU continuously; it is never freed (since staying resident is the point). The usable purposes depend on the effective budget of the TE GPU.

| Purpose | Required amount (measured) | Feasible on a 20GB card? |
|---|---|---|
| t2va / fl2va / t2i | 17.76GB | Feasible (about 1.9GB headroom) |
| ref2va | 20.67GB or more (TE 17.45 + reference encoding 3.22 or more) | Not feasible (OOMs by 204MB against an effective budget of about 19.7GB) |

It follows that ref2va requires an effective 20.7GB or more, i.e. a GPU with a catalog capacity of 22.2GB or more (a 24GB card would have an effective ~22.4GB, giving about 1.7GB headroom, but this is not guaranteed since 2 or more references increase the requirement further).

Layering `H3_KEEP_TRANSFORMER=1` on top makes it possible to keep the transformer resident even during the decode phase. All 3 of the following conditions are required:

1. `H3_LOWVRAM=1` (`group` is not eligible)
2. `H3_TE_DEVICE` is set (without TE on a separate GPU, the resident transformer-int8 34.3GB + TE-nf4 17.45GB = 51.75GB would break the encode phase first)
3. `H3_VIDEO_VAE_FP16=1` (with fp32 decode, transformer 34.3GB + decode peak 16.29GB = 50.6GB doesn't fit in 48GB; with fp16 it's 45.7GB, which fits)

### Recommended Launch Command (Current 48GB + 20GB Configuration)

```bash
H3_LOWVRAM=1 H3_TE_PRUNE=1 H3_TE_DEVICE=cuda:1 venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
```

To nearly eliminate the fixed cost further, add `H3_VIDEO_VAE_FP16=1 H3_KEEP_TRANSFORMER=1`.

This assumes a two-GPU configuration where GPU0 (48GB) is fixed for transformer duties, and GPU1 (20GB) is used to keep TE resident. To show only GPU0 (a configuration where TE is not placed on a separate GPU), start with `CUDA_VISIBLE_DEVICES=0` and `H3_LOWVRAM=1` alone.

For more detailed, per-phase patterns of what stays resident, see [docs/RESIDENCY.en.md](RESIDENCY.en.md).

---

## 6. Performance

### Measurements by Mode and Configuration

**Baseline is 768×768, 5 seconds (124 frames), 30 steps.**

| Configuration | Peak VRAM | t2va time |
|---|---|---|
| 96GB (default) | 92GB | about 160s |
| 80GB class (`H3_TRANSFORMER_QUANT=int8`) | 59.7GB | about 160s |
| 48GB class (`H3_LOWVRAM=1`) | 38.9GB | about 215s |
| 32GB class (`H3_LOWVRAM=group`) | 28.7GB | about 280s |
| 18GB class (`H3_LOWVRAM=group H3_TE_PRUNE=1`) | 17.7GB | about 280–320s |

**Measurements for each mode on RTX PRO 5000 48GB + `H3_LOWVRAM=1` (current configuration, since 2026-08-07)**:

| Mode | Time | Notes |
|---|---|---|
| t2va single (quality-focused, 30 steps) | 351s | |
| t2va single (turbo 4 steps) | **143s** | |
| t2i (still image, turbo 4 steps) | **94s** | |
| t2i (turbo + `H3_TE_DEVICE` + `H3_KEEP_TRANSFORMER`) | **9.7s/image** (steady state) | denoise 4.32s + decode 1.5s |
| t2i_batch (still image batch, 3 scenes) | 67.5s/image | marginal cost about 31s/image |
| ref2i_batch (still image with reference, 3 scenes) | 116.7s/image | including KV prefix sharing |
| ref2va_batch (video with reference, 2 scenes, 5 seconds) | 401.6s/clip | marginal cost about 330s/clip (asymptotically about 32% shorter as scene count grows) |

### Lineage of Speedups

| Stage | Request time (768², 5 seconds) |
|---|---|
| Initial (bf16 TE swap) | 245s |
| + TE bnb-4bit conversion | **185s** |
| + FirstBlockCache (0.05) | denoise 157→**118s** |
| + Sage Attention | denoise 118→**104s** |
| Current default (96GB machine) | **about 160s** |
| + FBC 0.1 (opt-in) | about 125s |
| + Turbo LoRA 8 steps (opt-in) | **about 88s** |
| + Turbo 4 steps (draft use) | about 40s |

Lineage of fixed-cost reduction on the 48GB machine (t2i turbo 4steps): 157s (right after the GPU swap) → 83.2s (`H3_TE_PREQUANT`) → about 35s (`H3_TE_DEVICE`) → **9.7s** (`H3_KEEP_TRANSFORMER`). For t2va at 5 seconds, 768²: 351.4s without turbo at 30 steps → 143s with turbo → 60.5s → **44.2s** (8.0x).

### Measured Peak VRAM

| Phase | Breakdown | Measured (48GB machine, `H3_LOWVRAM=1 H3_TE_DEVICE=cuda:1`) |
|---|---|---|
| Denoise (peak) | transformer-int8 34.3GB + activations about 6.6GB | 40.9GB |
| Decode | VAE pair 11.3GB + buffer | (after denoise, transformer already freed) |
| Decode with `H3_KEEP_TRANSFORMER=1` | transformer 34.03GB resident + fp16 decode | 44.15GB (measured 44.15GB against a derived prediction of 45.7GB) |

Denoise and decode do not overlap in time (the transformer is always freed immediately before decode, except with `H3_KEEP_TRANSFORMER=1`). The peak normally occurs during denoise.

---

## 7. Ensuring Quality and Equivalence

### Regression Confirmation via Same-Seed MD5 Matching

Modifications that should be mathematically inconsequential (layer removal, phase reordering, cache resets, the determinism of quantization itself) are verified up to byte-exact match (MD5 match) of the output (mp4/PNG) generated with the same seed. This lets equivalence be shown as "byte-identical" rather than "probably the same." Examples where this is applied: text_encoder 51-layer removal, batch phase reordering, FBC reset handling, matching between the turbo production implementation and the spike verification, switching between int8 quantization and bf16, the TE preload cache, and others. The policy is to follow the same procedure (same-seed MD5 match for t2va) for regression confirmation when upgrading the diffusers version as well.

### Distinguishing Degradation From Drift via PSNR

Sage Attention's PSNR is 21dB relative to baseline, and int8 quantization's is 19dB, but both are treated as **trajectory drift**, not degradation. Because diffusion models cause tiny initial computational errors to diverge across all subsequent steps, PSNR functions as a measure of "is it the same trajectory," not "is it the same picture." This is judged to be drift rather than degradation by combining visual indistinguishability with the fact that two runs with the same seed are byte-identical (fully deterministic). For modifications that do not involve quantization, such as fp16-ification of the video VAE, the high PSNR value of 39.97dB itself is treated as the quality metric.

### Language Verification of Audio (ASR)

For the audio of a generated video containing dialogue, verification confirms whether the speech is in the specified language (ASR-based verification). In structural conformance verification of the h3-official mode, whether the language specification in the dialogue tag `<d>[Japanese] ...</d>` corresponds to the actual audio output is a target of verification.

### Policy of Not Judging Numbers Visually

All claims about quality or performance — VRAM, time required, PSNR, MD5, ASR judgment, etc. — are based on measured values or confirmed byte matches. Visual confirmation is one piece of corroborating information used alongside these, and is never the sole basis for a judgment.

---

## 8. Configuration Reference

### Key Environment Variables

| Variable | Default | Effect |
|---|---|---|
| `H3_TE_QUANT` | `bnb-4bit` | text_encoder quantization method (`none` is bf16 at 66.7GB) |
| `H3_TE_PRUNE` | `0` | Removes unused upper layers of TE (output unchanged, -3.6GB with nf4) |
| `H3_TE_DEVICE` | (empty) | Keeps TE resident on the specified GPU, never freed (e.g. `cuda:1`) |
| `H3_TE_PREQUANT` | `1` | On-disk cache of the quantized TE weights (reduces load time) |
| `H3_TE_PREQUANT_DIR` | `models/prequant` | Cache save location |
| `H3_TE_PREQUANT_MIN_FREE_GB` | `25` | Skips saving if free disk space falls below this (generation continues) |
| `H3_TRANSFORMER_QUANT` | `none` | `int8` quantizes the transformer from 66.3GB→34GB |
| `H3_LOWVRAM` | `0` | `1` = phase-cycling for 48GB class / `group` = block-level offload for 24-32GB class |
| `H3_KEEP_TRANSFORMER` | `0` | Under `H3_LOWVRAM=1`, keeps the transformer resident even during the decode phase (3 conditions required, see §5) |
| `H3_VIDEO_VAE_FP16` | `0` | Converts the video VAE to fp16 (audio VAE is excluded) |
| `H3_CACHE` | `fbc` | Enables FirstBlockCache (`none` disables it) |
| `H3_CACHE_THRESHOLD` | `0.05` | FBC's cache-skip decision threshold |
| `H3_ATTN_BACKEND` | `sage` | Uses Sage Attention (`default` falls back to SDPA) |
| `H3_HIRES_DENOISE` | `0.35` | Denoise strength for pass 2 of hires-fix |
| `H3_TURBO_LORA` | `0` | Default enablement of the 4/8-step distillation LoRA |
| `H3_TURBO_LORA_REPO` | `lightx2v/Minimax-h3-Turbo` | Distribution source of the turbo LoRA |
| `H3_TURBO_LORA_FILE` | `minimax_h3_fl2v_turbo_4step_v0.1.safetensors` | File name of the turbo LoRA |
| `H3_TURBO_LORA_SCALE` | (measured default per format, 0.094 for lightx2v) | LoRA application scale |
| `H3_GROUP_OFFLOAD_BLOCKS` | `1` | Number of blocks transferred concurrently under group offload |
| `H3_GROUP_OFFLOAD_USE_STREAM` | `1` | Stream transfer for group offload |
| `H3_GROUP_OFFLOAD_LOW_CPU_MEM` | `0` | `1` prioritizes RAM savings (onload becomes slower) |
| `H3_GROUP_OFFLOAD_MIN_RAM_GB` | `40` | Minimum free RAM required to start group mode |
| `H3_VAE_SMALLCLIP_FIX` | `1` | VAE decode fix for ultra-short clips (still-image mode) |
| `H3_REF_PREFIX_CACHE` | `1` | KV prefix sharing for reference batches |
| `H3_LLM_URL` | `http://127.0.0.1:64650` | Local LLM used for prompt enhancement |

There are other environment variables for diagnostics/debugging as well, but the above are the ones typically changed in normal operation. For items that apply immediately from the UI (FBC, Sage, Turbo), specify them via environment variables only if you want to change them permanently.

### API Endpoint List

| Path | Main parameters | Return value |
|---|---|---|
| `GET /` | — | UI (index.html) |
| `GET /api/status` | — | Load state, measured VRAM/RAM |
| `GET /api/progress` | — | Progress during generation |
| `GET /api/settings` | — | Current reload-affecting settings and their choices |
| `POST /api/settings/apply` | Quantization method, low-VRAM mode, etc. | Result of freeing/reloading the model |
| `POST /api/t2va` | `prompt`, `resolution`/`height`+`width`, `seconds`, `num_inference_steps`, `seed`, `upscale` | Video + audio (mp4) |
| `POST /api/fl2va` | Above + `image` / `last_image` | Video + audio (mp4) |
| `POST /api/t2i` | `prompt`, `frames` (default 22 \| 5), `resolution`/`height`+`width`, `seed` | Ultra-short mp4 + center-frame PNG |
| `POST /api/t2i_batch` | `prompts` (up to 24) + shared parameters | PNG/mp4 per scene |
| `POST /api/ref2va` | `prompt`, reference files (image/video/audio), `seconds`, `still`, `frames` | Video + audio, or PNG if `still=1` |
| `POST /api/ref2i_batch` | `references` + `prompts` (up to 24) | PNG per scene |
| `POST /api/ref2va_batch` | `references` + `prompts` + `seconds` (required) | Video + audio per scene |
| `POST /api/prompt/enhance` | `prompt`, `mode`, `task`, `lang` | Enhanced prompt + `violations`/`warnings`/`check_report` |
| `GET /api/outputs` | — | List of mp4/PNG directly under `outputs/` |
| `POST /api/outputs/delete` | File name | Deletion result (path-traversal protected) |
| `POST /api/outputs/concat` | Order of selected files | Concatenated mp4 (lossless or re-encoded) |

---

For the background of design decisions and details of pitfalls encountered during implementation, see the internal document [docs/internal/TECHNICAL_REPORT.en.md](internal/TECHNICAL_REPORT.en.md). For operational procedures and primary-source measured values, see [README.en.md](../README.en.md); for details of the VRAM residency design, see [docs/RESIDENCY.en.md](RESIDENCY.en.md).

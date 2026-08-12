# Running MiniMax-H3 on diffusers — Technical Report

[日本語](TECHNICAL_REPORT.md) | **English**

> **Internal document (work log).** This records the development process itself, including
> the bugs, dead ends and detours hit along the way -- it is not the outward-facing
> technical description. For the specification, design and performance, read
> **[docs/TECHNICAL_OVERVIEW.en.md](../TECHNICAL_OVERVIEW.en.md)** instead.
> The value here is "why the design ended up this way" and "so the same trap is not hit twice".

**Period covered**: the main body covers 2026-08-04 to 2026-08-08 (44 commits). **Everything after
that is covered by addendum A (08-09) and addendum B (08-11 to 12)** — the main body's numbers and
open items are a snapshot of that period, and anything since resolved or updated is annotated
inline (summarized in §11.1).
**Subject**: this repository `minimax-h3` (a standalone app for verifying MiniMax H3 / Hailuo 3.0 functionality)
**Implementation size**: approx. 10,700 lines (`core/runner.py` 4,943 / `static/index.html` 1,839 / `README.md` 1,441 / `app.py` 1,003 / others)

Separately from the operational documentation in [README.md](../../README.en.md), this report exists to record **not "what was built" but "why it was built that way, what went wrong, and how it was verified."** All numbers are measured on real hardware, with measurement conditions noted alongside. For step-by-step operating instructions, see the README.

---

## 1. Problem Setting

MiniMax H3 is an omni-modal 33B model that **generates video and stereo audio simultaneously in a single denoising pass**. Rather than overlaying audio afterward, video and audio are denoised together as separate rows on the same packed sequence — the decisive difference from existing video-generation models.

The goal of this project was to run this model **through the diffusers path**, gaining knowledge ahead of time for a future integration into `diffusers-server`. A ComfyUI implementation already existed, but since the integration target is diffusers-based, it was necessary to verify **whether diffusers alone could match or exceed** the ComfyUI assets, rather than simply reusing them.

### 1.1 The Biggest Constraint, Known From the Start

The measured component sizes are as follows, and **they do not fit into VRAM or RAM at the same time**.

| Component | Precision | Measured size |
|---|---|---|
| text_encoder (Qwen3-VL-32B) | bf16 (native distribution) | **66.73 GB** |
| transformer | bf16 | 66.3 GB |
| transformer_ref (for ref2va, separate checkpoint) | bf16 | 61.7 GB |
| vae + audio_vae | fp32 | 11 GB total |

Against a total of approx. 144 GB, the initial environment had **96 GB VRAM / 94 GB RAM**.

One early mistake is worth recording here. The initial estimate was "the text_encoder is probably distributed as fp32, so converting to bf16 would give 33GB" — but **when measured, it was already bf16 at distribution time, at 66.73GB**. This pattern of "designing from an estimate, then having reality break it" recurred throughout the project, so the project adopted an early policy of **never using estimated values as a basis for design**.

---

## 2. Design for Running on diffusers

### 2.1 Calling Modular Diffusers Blocks Directly

diffusers support for MiniMax-H3 is **provided only in PR #14355**, and as of 2026-08-06 it is unmerged (draft). This app **pins to commit `abc5e9b` (2026-08-02)** of that PR branch.

Furthermore, rather than calling the `ModularPipeline` as a whole, the design **calls the individual blocks in sequence, driven directly by the app itself**.

```
MiniMaxH3SetupStep            → canvas resolution, 17n+5 frame-count alignment, keyframe prep
MiniMaxH3TextEncoderStep      → prompt encoding (calls the staticmethod directly)
MiniMaxH3PrepareLayoutStep    → packed sequence layout, rotary positions
MiniMaxH3PrepareLatentsStep   → latent initialization
MiniMaxH3SetTimestepsStep     → sigma schedules for the two systems, video/audio
MiniMaxH3DenoiseStep          → denoising loop
MiniMaxH3VideoDecodeStep / MiniMaxH3AudioDecodeStep → decoding
```

**Why decompose it**: solving the memory constraints required controlling "what is resident on the GPU during which phase" at single-step granularity (§3). Calling the pipeline as a whole provides no such control point. In addition, modifications such as hires-fix (§5.1), which **inject processing partway through the loop**, cannot be implemented unless the blocks are driven directly.

**The cost**: this created a **strong dependency on the state contract** — `get_block_state()` / `set_block_state()` / `state.get(...)` / `row_timestep_plan`. The PR had a state-contract refactor (`8ab3662`, `99ced1b`) land on 8/4–8/5, and there is a high likelihood that tracking the latest version will break things. For this reason the README explicitly states "**do not casually upgrade**," and defines the procedure "first confirm regression via identical-seed MD5 match on t2va" for whenever tracking is done.

### 2.2 Options Not Adopted

`ComponentsManager.enable_auto_cpu_offload()` is the standard diffusers solution, but it was **not adopted**. This approach keeps all components resident on the CPU and moves only the active one to the GPU, meaning it tries to hold the full 144GB in RAM simultaneously. That does not work on this machine's 94GB of RAM.

---

## 3. Memory Design — Phase × Resident Set

The core of this project is the design of "**what is placed on the GPU during which phase**," which ultimately settled into five modes.

### 3.1 Mode List and Measurements

| Assumed GPU | Launch flag | Peak VRAM | t2va time (768², 5s, 30steps) |
|---|---|---|---|
| 96GB | (none = default) | 92 GB | approx. 160 s |
| 80GB class | `H3_TRANSFORMER_QUANT=int8` | 59.7 GB | approx. 160 s |
| 48GB class | `H3_LOWVRAM=1` | 38.9 GB | approx. 215 s |
| 32GB class | `H3_LOWVRAM=group` | 28.7 GB | approx. 280 s |
| 18GB class | `H3_LOWVRAM=group H3_TE_PRUNE=1` | 17.7 GB | approx. 280–320 s |

**The VRAM floor was pushed from 96GB down to approx. 18GB.** Each tier was verified equivalent either by identical-seed MD5 match or by A/B quality comparison.

> **Since then (2026-08-11) the floor dropped further**: with the projected TE (4B + a trained
> linear map, 3.11GB at NF4), **a real RTX 4060 Ti 16GB on its own** runs every feature to
> completion (peak 7.4–15.2GB), and **8GiB×2** gets as far as t2va 5s at 768² (peak 7.23GB). The
> table above is the five-mode lineup under the 32B TE; **the current floor is the 16GB class**.
> Details in addendum B and [docs/RESIDENCY.en.md](../RESIDENCY.en.md) §5.3.

### 3.2 Design Philosophy: Allow Only "Short, One-Way" Moves

Drawing on lessons from the sister project `diffusers-server`, a constraint was set from the very start: **the pattern "swap out a 60GB-class module on every single step" is forbidden**. This was because of a past incident where runaway swapping took down the whole system.

The principle adopted instead was "short window, one-way." Every module move must be a **one-way move confined to a single, specific phase**.

For example, in the default `bnb-4bit` mode:

```
steady state : [transformer 66.3GB + TE-nf4 21GB] = 87.5GB resident
encode       : unchanged (VAE evacuated to CPU)
denoise      : unchanged
decode       : drop transformer, bring [vae pair 11GB] to GPU  ← approx. 9s window
after decode : move VAE back to CPU and reload transformer (return to steady state)
```

transformer (66.3) + TE-nf4 (21.0) + vae pair (11.0) = **98.5GB**, which exceeds 95.6GB even before counting activation buffers. Attempting to keep all three resident at once and run decode actually OOM'd with `Tried to allocate 30.00 MiB` (the allocator saturated at 93.7GB). Since decode does not use the transformer at all, **the transformer yielding** was the correct answer.

### 3.3 `H3_LOWVRAM=1` (48GB Class): Phase Rotation

At 48GB, even TE-nf4 (21GB) plus transformer-int8 (34GB), 55GB combined, does not fit at once. So the design became **keep nothing resident between requests**.

```
entry    : [nothing resident]
encode   : [TE-nf4 21GB]                      ← transformer already released
(release TE)
denoise  : [transformer-int8 34GB + activations ~5GB ≈ 39GB]
(release transformer)
decode   : [vae pair ~11GB + decode buffers]
(VAE moved to CPU. No prefetch/preload for the next request.)
```

The cost is explicit: **a fixed cost of approx. 90–100 seconds per request** for loading TE plus loading the transformer. This fixed cost later became the motivation for the batch feature (§5.4).

> **That fixed cost was later eliminated**: first `H3_TE_PREQUANT`, then `H3_TE_DEVICE` / the
> projected TE, then `H3_KEEP_TRANSFORMER` — after which **both the transformer and the TE can stay
> resident between requests** (addendum A1 and [docs/RESIDENCY.en.md](../RESIDENCY.en.md) §5.5).
> The "nothing stays resident between requests" shape above is the behavior of
> `H3_KEEP_TRANSFORMER=0` (the default) and is still the design in use for configurations that
> **decode in fp32**.

### 3.4 `H3_LOWVRAM=group` (24–32GB Class): Block-Level Streaming

To go lower still, the 34GB transformer itself cannot be placed on the GPU. Using diffusers' `enable_group_offload(block_level, num_blocks_per_group=1, use_stream=True)`, the transformer is **kept resident in host RAM, with only 1–2 of its 50 blocks (approx. 0.68GB each) streamed to the GPU at a time**.

This does not fall under the CLAUDE.md prohibition on "swapping out a whole module." The design's rationale is that a group-offloaded module's "resident location" is the CPU side, and the hooks automatically manage small, block-level visits to the GPU.

---

## 4. Pitfalls Encountered — Reproduced and Fixed on Real Hardware

This chapter is probably the most valuable record in the whole project. Every item here was **reproduced on real hardware, root-caused, and fixed** — none are speculation.

### 4.1 Device Mismatch From the `_execution_device` Resolution Order

**Symptom**: immediately after releasing TE, a cuda/cpu device mismatch occurs inside the transformer's `rope()`.

**Cause**: `MiniMaxH3ModularPipeline._execution_device` returns **the device of the first `nn.Module` found**, in `self.components` insertion order. That order is `text_encoder, tokenizer, processor, vae, scheduler, ...`. Once TE is released, the next one found is the **`vae`, which is evacuated to the CPU**, so `_execution_device` silently starts returning `cpu`.

**Fix**: delay releasing TE until **after** `layout_step` / `latents_step` / `timesteps_step` finish. These steps are the last places that need `_execution_device`; once tensors are materialized on the correct device, releasing TE afterward has no effect.

This pitfall was nastier under `H3_LOWVRAM=1`. In that mode, the trick of "just load the transformer first" doesn't work either — `vae` is never released and **stays present** on the CPU, so the `_execution_device` scan never even reaches the transformer.

### 4.2 The Autograd Graph Pins TE to the GPU

**Symptom**: even though TE should have been released, approx. 50GB remains on the GPU.

**Cause**: `MiniMaxH3TextEncoderStep.encode_prompt` is a plain `staticmethod`, and `@torch.no_grad()` is attached to the `__call__` side of the block that was bypassed. As a side effect of decomposing the blocks ourselves (§2.1), **the no_grad guard had been dropped**.

**Fix**: explicitly wrap the call in `with torch.no_grad():`. This pitfall was a side effect of the decomposition design; it would not have occurred if the pipeline had been called as a whole.

### 4.3 `use_stream=True` + `low_cpu_mem_usage=True` Always Breaks With torchao int8

**Symptom**: in group offload mode, the forward pass of the **very first** transformer block during denoising fails with
`RuntimeError: cannot pin 'torch.cuda.CharTensor' only dense CPU tensors can be pinned`.

**Cause**: `low_cpu_mem_usage=True` (the API default is False) skips pre-pinning at `enable_group_offload()` time and switches to a path that **pins on every onload instead**. This deferred-pinning path does not work for torchao's `Int8Tensor`. It only fires when combined with `use_stream=True`.

**Fix**: default to `low_cpu_mem_usage=False`. Despite its name, **this is the safe side**. A minimal real-hardware reproduction (a stack of int8-quantized nn.Linear layers) confirmed that this combination triggers the bug unconditionally.

As a side benefit, **onload became 4–5x faster** (0.1–0.26s → 0.04–0.07s). Pinned memory cannot be paged out, so DMA transfers are faster. The cost is the up-front pinning cost (approx. 22 seconds, approx. 15.7GB RAM), which is guarded by `H3_GROUP_OFFLOAD_MIN_RAM_GB`.

**This is worth reporting as a real bug upstream to diffusers.**

### 4.4 `fuse_projections()` Does Not Remove the Original Weights

**Symptom**: after applying the turbo LoRA, the transformer's resident size grows from 66.3GB → **79.08GB** (+12.8GB), and the subsequent TE load (21GB) OOMs.

**Cause**: `fuse_projections()` **copies** `to_q`/`to_k`/`to_v` into `to_qkv`, but **does not delete the original three**. Reading `unfuse_projections()` shows it only removes `to_qkv`, with no logic to restore the originals. The result is that both the fused and unfused versions end up held simultaneously.

**Fix**: `del module.to_q / to_k / to_v` immediately after fusion. The `to_qkv` weights are an independent copy made via `torch.cat`, not a view, so this is safe.

### 4.5 int8 Fragmentation — "Only 54GB in Use, Yet a 15GB Allocation Fails"

**Symptom**: in the int8-dual-resident mode, the transformer reload for a second ref2va request fails inside `from_pretrained`'s `_caching_allocator_warmup` (`Tried to allocate 15.43 GiB`, actual allocation only 54.44GB out of 92.55GB). The total resident budget (89GB) fits comfortably within 95.6GB.

**Cause**: not a capacity shortage but **fragmentation**. Repeated int8 load/free cycles left the allocator with approx. 37GB of oddly-shaped "reserved but unallocated" blocks.

**Fix**: set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` **before `import torch`** (PyTorch only reads this once, at allocator initialization). This switches to a scheme that grows/shrinks a single virtual address reservation, resolving the issue. Because this project's headroom is always tight, this is set **unconditionally**, not as an opt-in.

### 4.6 FirstBlockCache Does Not Know About H3 Blocks

**Symptom**: after `enable_cache()`, a `ValueError` occurs on the first denoising step.

**Cause**: FBC looks up per-block-class metadata from `TransformerBlockRegistry`, but the PR branch has no registration for `MiniMaxH3TransformerBlock`.

**Fix**: register `TransformerBlockMetadata(return_hidden_states_index=0, return_encoder_hidden_states_index=None)` on the runner side. Since H3 has no cross-attention (text is treated as a row of the packed sequence), `None` for the `encoder_hidden_states` slot is correct. The principle of **leaving the venv's diffusers itself unmodified** was maintained.

### 4.7 hires-fix: Interpolating a Noisy Latent Produces a Checkerboard Pattern

**Symptom**: in two-pass generation, pass 2's output is entirely covered in checkerboard-like noise and completely collapses.

**Cause**: **the noisy `x_t` was being spatially interpolated.** Interpolation amplifies the high-frequency components of noise.

**Fix**: change the interpolation target to the **x0 estimate** (equivalent to `denoised_output`), and re-apply noise at pass 2's starting sigma. This was cross-checked against a reference implementation (a ComfyUI node), which was also confirmed to use `denoised_output`.

### 4.8 `_keep_in_fp32_modules` Overrides the dtype Argument

**Symptom**: even loading the VAE with `from_pretrained(dtype=torch.float16)`, all parameters remain fp32 (9.70GB).

**Cause**: `AutoencoderKLMiniMaxH3._keep_in_fp32_modules = ["encoder", "decoder", "quant_conv", "post_quant_conv"]` effectively covers the entire module tree, so diffusers' `load_model_dict_into_meta` ignores `dtype=` and forces fp32.

**Fix**: call `.to(torch.float16)` explicitly **after** `from_pretrained`.

### 4.9 Pruning TE to 50 Layers Silently Produces a Different Value

This was the hardest-to-discover pitfall in the whole project.

**Background**: H3 only reads `hidden_states[50]` from TE. Layer 51 onward (14 layers, approx. 13GB) is dead weight.

**Symptom**: naively pruning to "50 layers" makes `hidden_states[50]` differ from the original 64-layer model by a **maximum absolute difference of approx. 1.5e4** — a wholly different value, not at the level of quantization noise.

**Cause**: `Qwen3VLTextModel.forward` is wrapped in `@capture_outputs`, whose default `tie_last_hidden_states=True` **unconditionally overwrites the last hidden_states entry with `outputs.last_hidden_state` (the value after the final norm is applied)**. In the 64-layer model, index 50 is far from the end so this has no effect; but once pruned to 50 layers, index 50 becomes the **sole and final** entry, and gets swapped for the post-norm value.

**Fix**: prune to **51 layers** instead. `layers[50]` still executes, but its output is not read. This puts index 50 back in the middle of the stack. Confirmed with `torch.equal` that both bf16 and nf4 are **bit-identical** to the original.

Note that diffusers has a guard of `num_layers <= 50 → raise` precisely because of this boundary. **51 is the smallest value that passes that guard while also avoiding the substitution.**

**Result**: TE-nf4 21.02GB → **17.45GB** (-17%), bf16 66.71GB → 53.06GB (-20%). For both t2va and ref2va, **the resulting mp4 is MD5-identical with and without pruning**.

### 4.10 VAE Decode Fails on Ultra-Short Clips

**Symptom**: with a 5-frame (0.208-second) video, denoising completes fine but decode fails with `ValueError: torch.cat(): expected a non-empty list of Tensors`.

**Cause**: `AutoencoderKLMiniMaxH3._decode()`'s chunk-count computation `num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)` evaluates to **0** when the latent has 2 frames (exactly `tokens_chunk_size(5)` − `token_drop(3)`), so the loop never runs even once.

**Fix**: a monkeypatch on the runner side (`H3_VAE_SMALLCLIP_FIX`, default ON) adds a branch: "if less than one chunk, decode all tokens in a single `_decode_clip()` call, and trim the `frame_pre_padding` (3) and trailing padding." 2 latent frames × temporal_ratio(4) − 3 = 5 pixel frames, which is geometrically consistent. Normal-length videos are delegated straight through to the original implementation, so there's no impact there.

### 4.11 Missing Cleanup on Decode Exceptions (Cascading OOM)

**Symptom**: after the exception from 4.10 above occurred, **two unrelated subsequent requests both OOM'd back to back** (with approx. 98.5GB still resident).

**Cause**: when an exception is raised during the decode phase, `generate()`'s own post-processing — "reload transformer and return to steady state" — **never runs**. This leaves both TE-nf4 and the transformer resident, a state where bookkeeping and actual occupancy have diverged.

**Fix**: wrap the decode section in try/except, and **always** route through `_restore_decode_steady_state()` before re-raising, even on exception. Applied to both `generate()` and `generate_ref2va()`. A one-shot fault-injection hook (`H3_DEBUG_FAIL_DECODE`) was used to confirm on real hardware: "500 response → GPU frees down to approx. 2GB → the next request in the same process succeeds normally with 200."

### 4.12 A Lingering Reference Count Keeps VRAM From Being Returned

**Symptom**: `_free_text_encoder(force=True)` runs according to the logs, yet `allocated_gb` stays at 87.5GB and doesn't drop, and the very next denoise OOMs.

**Cause**: `_sync_shared_components_to_ref()` makes `self._pipe_ref.text_encoder` point to the **same object** as `self._pipe.text_encoder`. Deleting only one side leaves a reference count behind.

**Fix**: explicitly delete both references. "Thinking you freed it" is the most dangerous pattern here, so the operating procedure now always measures `allocated_gb` after every release to confirm.

### 4.13 Misattribution From a Swallowed Exception

**Symptom**: `AttributeError: 'NoneType' object has no attribute 'enable_cache'`.

**Cause**: `load_components()` internally swallows exceptions with `try/except Exception: continue`, and does not propagate a load failure to the caller. As a result, `self._pipe.transformer` remains `None` and subsequent processing proceeds, causing the error to surface **at a location unrelated to the actual failure**.

**Fix**: add an explicit None check right after loading, and report the failure on the spot. When an upstream layer swallows failures, **the caller must establish its own boundary.**

### 4.14 The transformer_ref Load Ordering in ref2va

**Symptom**: on the first ref2va request, `vae._encode_clip()` OOMs 8 seconds in with `Tried to allocate 98.00 MiB`. On subsequent requests, it OOMs at `_vae_to_gpu()` with `allocated_gb: 98.81`.

**Cause**: reference VAE encoding is done with `vae` on the GPU, but `transformer_ref` (66.3GB) + TE-nf4 (21GB) + vae pair (11GB) = 98.3GB was already over budget. From the second request onward, the `transformer_ref` restored after the previous decode also remained resident.

**Fix**: change the ordering so `transformer_ref` is explicitly released every time **before** reference VAE encoding, and loaded only after encoding completes. This is the same kind of ordering constraint as fl2va's keyframe encoding.

### 4.15 Non-GPU Pitfalls

- **HTML5 form validation rejects input before it gets rounded**: attaching `min`/`max`/`step="32"` to the width/height `<input type=number>` fields caused unrounded fractional values (e.g. 1000×700) to be blocked by `stepMismatch`, **so submission never even happened**. The confusing symptom was "only coincidentally-valid values go through." Resolved by not attaching constraint attributes to fields that are meant to be rounded.
- **A checkbox turned OFF must explicitly send `0`, not an empty string**: an empty string can be interpreted as "unspecified — use the server default," which risks turbo not being disabled by unchecking the box on a server started with `H3_TURBO_LORA=1`.
- **Rebuilding every gallery tile causes flicker**: rebuilding all tiles on every selection regenerates every `<video>` element (noticeable at 95 tiles). Selection state should instead be a **differential update** to badges/checkmarks.

### 4.16 Miscellaneous

- **Ordering of `_sync_shared_components_to_ref()`**: syncing shared components to the ref pipeline must happen **after** TE is loaded. `components` is a live attribute read, not a promise, so syncing beforehand bakes in `text_encoder = None`. Because `H3_LOWVRAM` does not preload TE, this only surfaced in that mode, as `AttributeError: 'NoneType' object has no attribute 'config'`.
- **`MiniMaxH3AdaLayerNormModulation` reads `.weight.dtype` directly**: replacing the Linear with a wrapper causes an `AttributeError`. Resolved by giving `_TurboLoRALinear` transparent `.weight` / `.bias` properties. This is the same shape of issue the sister project hit with JoyAI's `PatchifyLinear`.
- **audio_vae must never be run in bf16**: a known upstream bug makes generated audio approx. 20dB quieter. Fixed at fp32, strictly.
- **VAE tile size has no effect**: shrinking it 256 → 192 → 128 → 96 to reduce decode peak left peak VRAM **unchanged at 16.29GB** (only making timing worse, 5.9s → 10.5s). Concluded that a fixed cost independent of spatial tiling is the bottleneck. **That 16.29GB predates both the 2026-08-10 uint8 intermediate-tensor fix and the 2026-08-12 move of de-normalization to the CPU (addendum B6)**, so the current decode peak is lower (**the post-fix re-measurement has not been done**). The conclusions (tiling does not help; two thirds of the peak is VAE weights) are unchanged.
- **Turbo LoRA format detection order matters**: checking for the comfy signature (`qkv_proj`) first is essential, or misdetection occurs. The Ostris version also has `token_refiner.blocks.*` keys, so checking prefix alone would misclassify it as diffusers format (reproduced and fixed using an actual file).

---

## 5. Feature Extensions

### 5.1 2x Upscaling via Two-Pass Generation (hires-fix)

Denoise the first half at low resolution → spatially interpolate only the video latent's x0 estimate by 2x → re-noise → finish at high resolution. No trained upscaler is used.

- 768² → **1536²** takes 645s / peak 88.0GB (`upscale=0` is 181s / 92.1GB)
- The audio latent has no spatial axis, so it is left untouched
- Changing resolution requires rebuilding `build_packed_sequence()` and `row_timestep_plan`
- The scheduler's `_step_index` is not reset between passes (to keep the sigma trajectory continuous)

### 5.2 Ref2VA (Omni-Reference Generation)

Generates video + audio from an ordered set of references: 9 images, 3 videos, 3 audio clips (12 total). Wired up the PR's functionality, and resolved 3 OOM issues along the way. Reference VAE encoding must happen with `vae` on the GPU, which makes the ordering — **before** loading transformer_ref — mandatory.

### 5.3 Still-Image Mode (T2I / ref2i)

Substitutes for image generation via "generate an ultra-short video → extract the center frame."

| Frame count | Denoise | Decode | Quality |
|---|---|---|---|
| 22 (0.917s) | 29.0 s | 1.8 s | No breakdown, high quality |
| 5 (0.208s) | 9.1 s | 0.67 s | No breakdown (succeeded for the first time only after the 4.10 fix) |
| 124 (5s, baseline) | 197.7 s | 10.6 s | — |

Even out-of-distribution for training (1/5 of the official 5-second minimum), quality did not degrade, and audio was non-silent as well. The value here is not speed, but the ability to produce **still images whose style perfectly matches H3's own**, for use as an FL2VA opening frame or as a Ref2VA reference.

Still images with references (ref2i) also worked. Even when reference packing (more reference rows than generation rows) overlapped with the ultra-short-clip path, quality did not degrade, and the reference's costume, props, and style were preserved (denoise 58.1s, 1/4 of the 5-second baseline).

### 5.4 Batch Generation — Amortizing the Fixed Cost Once Across the Whole Batch

Amortizes `H3_LOWVRAM=1`'s per-request fixed cost (approx. 110 seconds) by **reordering phases**.

```
entry   : [nothing resident]
encode  : [TE-nf4]        encode all scenes together, in one pass
denoise : [transformer]   load once and denoise all scenes in sequence
decode  : [vae pair]      decode all scenes (saving each one)
```

The key to the implementation is resetting the mutable state shared across scenes. Since sigma values are identical across all scenes, the scheduler only needs `_step_index = None` reset; FirstBlockCache calls `_reset_stateful_cache()` per scene. **This equivalence was proven by an mp4/PNG MD5 match against sequential generation.**

| Path | Sequential | Batch | Reduction |
|---|---|---|---|
| Still image (t2i, 3 scenes) | 157 s/image | **67.5 s/image** | marginal cost ~31 s/image |
| Still image w/ reference (ref2i, 3 scenes) | 164.9 s/image | **116.7 s/image** | -29% |
| Video w/ reference (ref2va, 2 scenes) | ~485 s/clip | **401.6 s/clip** | -17% (marginal ~32%) |

> **Making models resident changed what batching is for (2026-08-12)**: this feature existed to
> amortize **the load fixed cost** across one batch, and `H3_KEEP_TRANSFORMER` removed that fixed
> cost entirely — so **t2i batching no longer helps** (0.94x at 3 scenes, i.e. marginally slower;
> before residency it was 157s → 67.5s, a 2.3x win). **The reference family still benefits**,
> because what is shared there is not the load but the **~47s reference vision encode** (ref2i
> 1.69x at 3 scenes, i2va 1.37x at 2 scenes). **Per-step times in a batch match single requests
> exactly** (ref2i 2.598s vs 2.599s, i2va 7.321s vs 7.323s), so the batch path itself adds zero
> overhead. **Lesson**: an amortization optimization is worth exactly as much as the cost it
> amortizes. **State explicitly what is being shared**, and you can reason about whether the
> benefit grows or vanishes when the premises change.

### 5.5 KV Cache Sharing for the Reference Prefix

During the encoding phase of a ref batch, the Qwen3-VL encoding of the reference labels + vision content (approx. 4,104 tokens, approx. 65 seconds per scene) was being duplicated across every scene.

The ref2va token sequence has references prepended, with the prompt appended verbatim at the end, and the conditioning model is a **causal LM**. Therefore **the representation of the reference prefix does not depend on the prompt**. The prefix is run once with `use_cache=True` and baked into a `DynamicCache`; for each scene, only the prompt tail (14–33 tokens, approx. 0.2 seconds) is fed through as a cache continuation.

**Verification**: the prefix portion's `hidden_states[50]` is **bit-identical** to the full computation, via `torch.equal`. A relative RMS discrepancy of approx. 1.5% remains in the prompt-tail portion, and a **negative control** confirmed this is not a logic bug — a deliberately broken positional-offset continuation spikes the relative RMS to 27–30% (a 20x jump). 1.5% is at the level of "rounding noise from a correct computation."

**Pitfall** (discovered by reading the transformers 5.14.1 implementation): for the continuation call, `attention_mask=None`, and the `mm_token_type_ids` / `pixel_values` / `grid_thw` family must **all be None** (passing `image_grid_thw` causes `model.rope_deltas` to be recomputed and overwritten). `rope_deltas` is **instance state** on `Qwen3VLModel`, so no other TE call may be interleaved between the prefix pass and any of its continuations.

**Result**: the ref2i batch's encode phase went from 212.5s → **83.1s**, and per-image time went from 164.9s → **116.7s** (-29%).

---

## 6. Speed Improvements, Stacked

| Stage | Request time (768², 5s) | Quality verification |
|---|---|---|
| Initial (bf16 TE swap) | 245 s | — |
| + TE bnb-4bit | **185 s** | Frames and audio at the same level, same seed |
| + FirstBlockCache (0.05) | denoise 157→**118 s** | PSNR 31.8–34.3dB, audio correlation 0.979 |
| + Sage Attention | denoise 118→**104 s** | Fully deterministic (byte-identical for the same seed) |
| **the default at the time (2026-08-08)** | **~160 s** | |
| + FBC 0.1 (opt-in) | ~125 s | Composition drifts noticeably by eye |
| + Turbo LoRA 8steps (opt-in) | **~88 s** | Close to baseline |
| + Turbo 4steps (draft) | ~40 s | Somewhat soft |

### 6.1 Not Reading PSNR as "Degradation"

Sage Attention's PSNR is **21dB** vs. baseline, and int8 quantization is **19dB**. Taken at face value these look like large degradations, but **neither is actually degradation — both are trajectory drift**. In diffusion models, tiny early differences fork the entire trajectory, so PSNR here measures "is it the same trajectory," not "is it the same picture." In practice, the two are visually indistinguishable, and two runs with the same seed are byte-identical (fully deterministic).

Misreading this would cause valid optimizations to be rejected as "quality degradation." This project judged such cases using **visual inspection + audio correlation + determinism, together.**

### 6.2 Not Breaking the Environment While Building

No pre-built Linux wheel exists for SageAttention (only a Windows build is published), so it had to be built from source for sm_120. Two practical cautions came out of this:

- **Unlimited parallel nvcc jobs exhaust host RAM and can take down the whole system** (this has happened before). Restrict to `MAX_JOBS=4 NVCC_THREADS=2` and run under `systemd-run` with a memory cap
- **`CUDA_HOME=/usr/local/cuda-12.8` must be set explicitly.** The default cuda-13.0 is incompatible with torch's cu128 build and the build fails

Also, diffusers' hub-based attention backends (`flash_hub` / `sage_hub`) turned out **not to be viable**, since the Hub has no build for torch 2.9 (only 2.10–2.12 are available). This is not an environment problem.

---

## 7. Verification Methodology

To eliminate guesswork, four methods were used, chosen per situation.

| Method | Purpose | Example application |
|---|---|---|
| **mp4 MD5 match, same seed** | Mathematical proof of equivalence | TE 51-layer pruning, batch phase reordering, FBC per-request reset, turbo real implementation vs. spike match |
| **PSNR + audio correlation + visual inspection** | Quality judgment for approximate methods | FBC, Sage, int8, video VAE fp16 (PSNR 39.97dB) |
| **VRAM ballast** | Verifying behavior on GPU capacities not physically owned | 32GB / 24GB / 22GB / 20GB / 18GB / 16GB tiers |
| **Probe scripts** | Hypothesis testing without touching the main codebase | 17 scripts (`scripts/probe_*.py`) |

**The point of the probe approach** is that verification can happen without changing the main codebase at all. For instance, verifying ultra-short clips avoided both diffusers' and the app's own 5-second validation entirely via monkeypatches confined to the probe script. If a hypothesis is rejected, the main codebase is left untouched.

**Subprocess isolation** was also an important lesson. Since a 5-frame exception contaminated subsequent requests in the same process (§4.11), all later probes isolate one generation per process. The goal was to measure a given condition **in isolation**, not "behavior after an exception has already corrupted GPU state."

### 7.1 Probe Script Inventory (`scripts/`)

| Script | What it verifies |
|---|---|
| `probe_t2va.py` | Regression check on the basic T2VA path. Run before checking via the UI |
| `probe_group_offload.py` | Whether `device_map="cpu"` + torchao int8 actually gets quantized on the CPU (confirmed 370/370 layers) |
| `probe_group_offload_forward.py` | Isolates and reproduces the §4.3 pin_memory crash using a dummy int8 stack |
| `probe_group_offload_fix.py` | Controlled comparison of 2 candidate fixes (`use_stream=False` / `low_cpu_mem_usage=False`) |
| `probe_vae_tile_size.py` | Benchmarks VAE decode in isolation, measuring the effect of tile size (found to be unrelated) |
| `probe_te_prune.py` through `_4.py` | §4.9's 50 vs. 51 layers. `_3` checks byte-identical weights, `_4` saves all hidden_states to pinpoint the divergence |
| `probe_turbo_lora_apply.py` | Isolated verification of applying the Ostris-version turbo LoRA (259 layers wrapped, forward diff) |
| `probe_lightx2v_turbo.py` | Whether the lightx2v version can be applied under int8 (Q1), and a 4-step quality/strength sweep (Q2) |
| `probe_short_frames.py` / `_one.py` | Ultra-short clips at 5/22/124 frames. Uses subprocess isolation |
| `probe_ref2va_short.py` | Quality of Ref2VA × ultra-short clips (overlap with reference packing) |
| `probe_ref_prefix_cache.py` | Bit-match verification and negative control for the §5.5 prefix sharing |
| `probe_hires_vae_upscale.py` | Isolated check of whether the VAE can decode hires-fix's interpolated latent |
| `probe_h3official_compliance.py` | The §10.2 baseline failure-rate measurement (mechanical judgment rules F1–F9) |
| `vram_ballast.py` | Reduces free VRAM to an arbitrary value using dummy CUDA tensors (the ballast itself) |

All probes were written under the policy of "no changes to the main codebase, only monkeypatches within the probe" or "an independent reimplementation after reading the main codebase" — **the venv's diffusers itself was never touched.**

---

## 8. Environmental Change — GPU Swap (2026-08-07)

Partway through the work period, the experiment machine's GPU was swapped from a single **RTX PRO 6000 Blackwell 96GB** to a two-card configuration of **RTX PRO 5000 Blackwell 48GB + RTX 4000 SFF Ada 20GB**.

The impact was significant. With only 48GB available, the default mode (bf16 transformer, 66.3GB) can **no longer physically load**, so launching now requires `H3_LOWVRAM=1` or `group`. Even with the same settings, things got slower compared to the old GPU (`H3_LOWVRAM=1`'s t2va went from 215s → 351s). The 20GB RTX 4000 has only 2GB of headroom above the floor (approx. 18GB), and being sm_89, it cannot even use the SageAttention build built for sm_120.

**This swap shaped subsequent design decisions.** As the fixed cost of low-VRAM modes became dominant, the value of batch generation (§5.4) and prefix sharing (§5.5) grew proportionally larger, and the question "can turbo be used at 48GB" led to the turbo LoRA reconsideration in §9.

---

## 9. Turbo LoRA — From "Impossible" to "Unlocked"

### 9.1 Why the First LoRA (Ostris Version) Didn't Work Under int8

`larryvrh/MiniMax-H3-Turbo-Lora` targets the **fused QKV** (`qkv_proj`) that originates from ComfyUI. Applying it requires `attn.fuse_projections()`, whose implementation is `torch.cat([to_q.weight, to_k.weight, to_v.weight])`. Under int8 quantization, `to_q`/`to_k`/`to_v` are torchao `Int8Tensor` instances, and **`Int8Tensor` has no registered `aten.cat` kernel.**

As a result, this fails reliably with `NotImplementedError` under any of `H3_LOWVRAM=1` / `group` / int8-dual-resident. This is not an ordering problem (it fails before even reaching the group offload hooks) — it is a **missing kernel**, judged unavoidable, so the app now rejects it at launch time.

### 9.2 Resolved With the lightx2v Version

`lightx2v/Minimax-h3-Turbo` (DMD distillation, Apache 2.0) has **diffusers-native keys**, with `to_q`/`to_k`/`to_v` kept separate. Reading the safetensors header directly confirmed that `ff.net.0.proj`'s `lora_B` is `(28672, 128)`, exactly matching diffusers' SwiGLU (`dim_out*2` including the gate) — backing up that **this was trained directly against diffusers' own module layout.**

No fusion needed → no call to `torch.cat` → **applicable even under int8.** In practice, it was applied to the int8 transformer across 312 modules in 0.6 seconds, with no exceptions.

### 9.3 Pitfall: the Strength Is 0.094, Not 0.75

The distributor's documentation says "4 steps, LoRA strength 0.75," but **applying that value directly to the raw B·A delta produces pure noise output even at 30 steps.**

The cause is that ComfyUI folds alpha into how it applies the LoRA, whereas this implementation multiplies the raw delta directly. A conversion hypothesis of `0.75 × (alpha/rank) = 0.75 × 16/128 ≈ 0.094` was formed and confirmed with a strength sweep.

| strength | 4-step result | audio rms / peak |
|---|---|---|
| 0.75 (documented value as-is) | **pure noise** (same even at 30 steps) | 0.083 / 0.43 |
| 0.15 | good (background slightly soft) | 0.069 / 0.79 |
| 0.10 | good | 0.065 / **1.05 (clipping)** |
| **0.094** | **best** (sharp down to fur and pine needles) | 0.039 / 0.70 |

Confirming that it still breaks even at 30 steps was the key point that isolated this as "a strength problem, not a step-count problem."

**Result** (RTX PRO 5000 48GB + `H3_LOWVRAM=1`, 768², 5 seconds): denoise **197.7s → 26.1s (7.6x)**, total time **351.4s → 135.2s (2.6x)**. Combined with still-image mode, denoise drops to 5.0 seconds.

**Remaining limitations**: incompatible with `H3_LOWVRAM=group` in any form (because `enable_group_offload`'s `cpu_param_dict` is fixed at activation time, the LoRA buffers added afterward risk being left out of the offload cycle; not unlocked while unverified). Audio level runs somewhat higher than non-turbo.

> **Application to ref2va was confirmed on 2026-08-12 (the old "unverified" is resolved)**: even at
> 4 steps the reference subject's fidelity holds (bangs, hairstyle, cardigan, ribbon and necklace
> all match), and the video keeps the person consistent from the first frame through the middle to
> the last. **Strength 0.094 carries over to reference-conditioned trajectories unchanged.**
> The same test also revealed that **only the three reference endpoints in `app.py`
> (`/api/ref2va`, `/api/ref2i_batch`, `/api/ref2va_batch`) hard-coded
> `num_inference_steps: int = Form(30)`** and therefore never picked up turbo's step default
> (`DEFAULT_NUM_INFERENCE_STEPS` = 4). The turbo LoRA itself was being applied to
> `transformer_ref` correctly, so what had been silently running for a long time was the
> contradictory combination of **a distilled LoRA running 30 steps**. Fixed by making all three use
> `DEFAULT_NUM_INFERENCE_STEPS`. **Lesson**: when you retrofit a "single source of truth" default,
> **grep for every endpoint carrying the same form parameter and convert them all**. A leftover
> literal still works, so the only symptom is "slow" — nothing surfaces as an error.

---

## 10. LLM Prompt Enhancement and Its Quality Assurance

### 10.1 "The End of the Prompt Wins"

While implementing a mode that feeds MiniMax's official `h3-prompt-writing` skill (SKILL.md + reference material, approx. 15.8KB) into the system prompt in full, a problem arose where specifying `lang=ja` still **returned English**.

The cause was placing the language instruction at the **beginning** of the system prompt. It was overwritten by the 15.8KB of English reference material that followed (which is full of English-output examples). This was fixed by **placing the language instruction after the reference body, as the final instruction.**

This discovery — that "instructions later in the prompt take precedence" — later led to finding the flaw in the English-language wrapper (§10.3).

### 10.2 Measure "Is This a Model Limitation" Before Deciding

Rather than debating by guesswork whether an LLM output defect was "a limit of the model's capability" or "fixable via prompting," the failure rate was **measured first** (5 inputs × 3 runs, gemma4-31B Q4_K_M, n_ctx 7680).

| Failure class | Rate |
|---|---|
| Structure/notation (field names, `[Shot n]`, timecodes, `<d>` tags, speaker IDs) | **0/15 (0%)** |
| Time allocation | 6/15 (40%) |
| Context overflow | 0/15 |

**The worry that "different inputs would surface different problems" turned out to be unfounded, in a good way** — violations did not scatter, they concentrated in a single class. Furthermore, those 6 cases split into two distinct natures.

- **Input was physically impossible (3/6)**: the estimated speaking time for a line exceeds the shot's duration. In one actual example, a 38-character monologue (estimated 9.5 seconds) was requested within a 9-second shot, and **the LLM had already made the best possible allocation, giving all 9 seconds to a single shot.** Since the official spec requires verbatim preservation of dialogue, there is no escape hatch via shortening either. **Unsolvable by any model**
- **LLM allocation mistake (3/6)**: it fits within the duration, but the placement is poor. The kind of thing that can be fixed by pointing it out

In other words, only **one fifth of the total** was attributable to "an LLM limitation" — the rest were either an inevitable consequence of the spec or a problem on the input side.

### 10.3 Countermeasures (3 Layers)

1. **Validator** (`core/prompt_check.py`): 8 rules that can be judged deterministically. Of these, "minimum shot duration" and "dialogue fits within its shot" are **practical rules this app adds that are not in the official spec.** The official spec only says "cut times must fit within the duration," which permits a literal-but-unusable output such as cutting a 5-second duration down to 4.5 seconds (leaving a final shot of 0.5 seconds). This also benefits hand-written prompts
2. **System prompt improvements**: investigation found that **the English-language wrapper had zero instructions placed after the guide body.** Only the Japanese version had the trailing block from the §10.1 fix, and since the default is `lang=en`, all instructions — including the duration constraint — were being buried inside the 15.8KB guide. The 6 time-allocation rules were added to the end of both versions
3. **Repair loop**: violations are presented back to the model in Japanese for regeneration (up to 2 retries). **A repair candidate that increases violations is discarded.** Input-side infeasibility is judged before ever reaching the LLM, returning a 400 plus advice

**Result** (same conditions):

| | clean | infeasible input detected | **violations remaining** |
|---|---|---|---|
| baseline | 9/15 | 0 | **6/15** |
| after improvement | **12/15** | **3/15** (correct behavior) | **0/15** |

Median time went from 8.5s → 9.2s (+8%). The repair loop only fires when needed.

### 10.4 Context Budget

The LLM's n_ctx is **7,680**. Measured system-prompt usage is 4,832 tokens (63%) for t2va, and **6,626 tokens (86%) for ref2va**. t2va has headroom, but **ref2va has only about 1,050 tokens left to cover both input and output.** It currently works correctly, but behavior with longer inputs is unconfirmed, and this is recorded as a known risk.

---

## 11. State of the Art as of 2026-08-08

> **This section is the state of the art for the main body's period (2026-08-04 to 08-08), not
> today's numbers.** Later records live in addendum A (08-09), addendum B (08-11 to 12) and the
> README's dated sections. For reference, the current fastest is **t2i 7.40s / t2va 5s 28.13s** on
> the 96GB box with int8 on a single GPU and freeing stopped (`H3_KEEP_TRANSFORMER=1`), at turbo
> 4steps, 768², steady state. The table below is from the **48GB box, in the era when the
> transformer was still being swapped in and out on every request**.

**Environment**: RTX PRO 5000 Blackwell 48GB / 94GB RAM / `H3_LOWVRAM=1`

| Use case | Time required |
|---|---|
| t2va single (quality-focused, 30 steps) | 351 s |
| t2va single (turbo 4 steps) | **143 s** |
| still image t2i (turbo 4 steps) | **94 s** |
| still image batch (3 scenes) | 67.5 s/image (marginal ~31 s/image) |
| still image w/ reference batch (3 scenes) | 116.7 s/image |
| video w/ reference batch (2 scenes) | 401.6 s/clip (marginal ~330 s/clip) |

As a storytelling production pipeline, the flow "**establish the character with t2i → produce still images for every scene with ref2i_batch → turn each scene into video with ref2va_batch**" now works end to end.

### 11.1 Remaining Work (the 2026-08-08 list, annotated with what happened since)

**Waiting on external events**
- Merging of diffusers PR #14355 (per §2.1, not tracked casually; when tracked, confirm regression via identical-seed MD5)
  → **【resolved 2026-08-09】** tracked to the merged `f37ab93`. Every path is equivalent at
  identical seeds by MD5 (addenda A2 / A5, and the README's dated sections)
- lightx2v turbo LoRA is v0.1. Reconsider enabling it by default once the community has more track record (**still open**)

**Untouched improvement candidates**
- Pre-saving quantized checkpoints (would eliminate the low-VRAM mode's 90–100 second fixed cost)
  → **【resolved】** `H3_TE_PREQUANT` (an on-disk cache of the quantized TE) took 90–100s down to
  ~55s, and parking the TE on another GPU / using the projected TE together with
  `H3_KEEP_TRANSFORMER` **removed the fixed cost entirely** (addendum A1 and
  [docs/RESIDENCY.en.md](../RESIDENCY.en.md) §5.5)
- 16GB-class support (needs streaming execution of TE; the current floor is 17.45GB)
  → **【resolved 2026-08-11】** solved not by streaming the TE but by the **projected TE** (4B + a
  trained linear map, 3.11GB at NF4). A real RTX 4060 Ti 16GB runs every feature **on its own**, and
  8GiB×2 gets as far as t2va (addendum B)
- `torch.compile` (compatibility with the FBC / group offload hooks is unverified) → **still untouched**

**Known unresolved issues**
- With `H3_LOWVRAM=group`, running ref2va after t2va gets rejected by the host RAM guard (confirmed only on the 94GB machine; likely, but not verified, not to occur on machines with 48GB+ RAM headroom)
  → **【resolved 2026-08-12】** the cause was not RAM capacity but **leftover pinned host cache**
  (fixed by adding `torch._C._host_emptyCache()`, addendum B8). It happens **even on machines with
  plenty of RAM headroom** — the guess recorded here at the time was **wrong**
- ref2va's system prompt occupies 86% of n_ctx (§10.4) → **still unresolved**
- Turbo applied to ref2va, and turbo × group offload, are both unverified
  → turbo × ref2va is **【resolved 2026-08-12】** (the callout in §9.3).
  **turbo × group offload is still unverified** and the combination remains rejected in code

**Items added since** (details in addendum B and the README)
- Caching the **~47s reference vision encode** across requests, which is the reference family's
  bottleneck (**not implemented**)
- Removing the **~13s** reload of the t2va transformer at the end of a ref2va request (**not implemented**)
- The reference family's `H3_TE_DEVICE` guard ("the TE GPU needs 24GB+") is still a **32B-TE-based
  threshold**, far too strict for the 3.11GB projected TE (**needs revisiting, not yet fixed**)
- The batch path's phase reordering is `H3_LOWVRAM=1`-only and **silently falls back to sequential**
  execution otherwise
- **Threshold tuning** for FBC skipping 0 steps on the int8+SDPA trajectory (**unverified**, up to 2x on the table)
- **Re-measuring the decode peak** after the 2026-08-12 move of de-normalization to the CPU (not done)

---

## 12. Summary — Reusable Lessons

Lessons from this project worth carrying forward into the next one:

1. **Never use an estimated value as a design premise.** The text_encoder's size (estimated 33GB → measured 66.73GB) and TE-nf4's size (estimated 17–18GB → measured 21GB) were both wrong as estimates. Measure first, then design.

2. **Mathematical equivalence can be proven with MD5.** For any modification where you can show "byte-identical" rather than "probably the same" (removing layers, reordering phases, resetting a cache), always verify all the way to that point. This made it possible to make large changes without fear of regression.

3. **A drop in PSNR is not necessarily degradation.** With diffusion models, trajectory divergence must be distinguished from quality degradation. Use visual inspection, determinism, and audio correlation together.

4. **Making failure detectable is more realistic than making failure rate zero.** LLM output quality does not reach 100% through prompt improvement alone, but adding a validator and a repair loop means the system can always return either "a valid output" or "an explanation of why it's infeasible" (§10.3).

5. **Cleanup on the exception path matters just as much as on the normal path.** The cascading OOM in §4.11 was caused by a single exception taking down two unrelated subsequent requests.

6. **The order spike → measure → implement.** Rather than entering full implementation on "this should work," each question was resolved with a probe that doesn't touch the main codebase before proceeding. Turbo LoRA's int8 support (§9.2) is a case where following this order made it possible to discover the decisive difference — key format — early.

7. **Read the upstream code.** Most of the pitfalls in this report were only understood by actually reading the implementation of diffusers / transformers / torchao. §4.9 (`tie_last_hidden_states`) and §4.3 (`pin_memory`) in particular could never have been figured out from documentation alone.

---

# Addendum: work on 2026-08-09 (12 commits)

The main body covers 2026-08-04 to 08-08. The pitfalls and lessons that arose after that
are added here. The content covers "tracking the diffusers merged version", "keeping the
transformer resident", and "UI reorganization and internationalization".
**The settled specification lives in [docs/TECHNICAL_OVERVIEW.en.md](../TECHNICAL_OVERVIEW.en.md)**,
so only pitfalls and lessons are written here.

## A1. `H3_KEEP_TRANSFORMER` -- the estimate was 1.55GB more conservative than the measurement

A configuration that does not release the transformer even during the decode phase. The
estimate was `transformer 34.3 + fp16 decode peak 11.4 = 45.7GB` against an effective
budget of 49.8GB, judged to have about 4GB of headroom, and work proceeded on that basis.
**The measured peak was 44.15GB** (t2va, 5 seconds), 1.55GB smaller than the estimate.
An estimate skewing conservative is safe, but since "4GB of headroom" was the basis for
the go/no-go decision, it was still the right call not to declare it viable until measured.

t2i steady state 9.7s/image, t2va 5 seconds = 44.2s. PNG/MP4 MD5 exact match for the same seed.

> **Since then (2026-08-11 / 08-12): the conditions were relaxed twice.** The original guard was
> "`H3_LOWVRAM=1` **and** `H3_TE_DEVICE` set **and** `H3_VIDEO_VAE_FP16=1`", but both of the first
> two came from the premise of **co-locating the 32B TE (17.45GB at nf4) with the compute GPU**.
> (1) With `H3_TE_PROJ` (the projected TE, 3.11GB at NF4) co-location fits the budget, so
> `H3_TE_DEVICE` is exempted; (2) plain mode (`H3_LOWVRAM=0`) also frees and reloads the
> transformer for every decode window (**a measured 11.9–12.3s per request**) and gains exactly the
> same way, so the condition was relaxed to "`H3_LOWVRAM` is not `group`". **(2) took zero extra
> implementation** — the branch that skips the free was already shared and the restoring
> `_ensure_transformer` was idempotent. Equivalence was proven on the plain side too, by a
> **same-seed PNG MD5 match** (`596a718e4b5cf9a0b907d2ec479225d2`, 19.58s → 7.40s).
> **Lesson**: a guard's conditions tend to encode **the configuration you happened to have at the
> time**, not the inequality you actually need. Write *why* each value is required in a comment,
> and when the premise changes you will **notice that it can be relaxed** (here, the justification
> for relaxing it was literally the budget arithmetic already written in the guard's own comment).

## A2. Two `_execution_device` pitfalls hit while tracking the merged version (both a recurrence of §4 in the main body)

`_execution_device`, which the main body called "the single biggest source of pitfalls" in
§4, produced two more incidents on the merged version. A real example that **even when the
same function is the cause, the shape of the cause is different every time**.

1. **The order of components changed.** Old: `text_encoder, tokenizer, processor, vae,
   scheduler, audio_scheduler, transformer, video_processor, audio_vae` ->
   New: `image_processor, text_encoder, tokenizer, processor, vae, **audio_vae**,
   scheduler, audio_scheduler, transformer_ref, transformer, video_processor`.
   Because `audio_vae` now comes before transformer, if `_pin_execution_device_to_compute()`
   only excludes text_encoder and vae, **the CPU-resident audio_vae gets picked up as the
   first nn.Module** and `_execution_device` returns `cpu`. The new layout step's contract
   is to place its output on `_execution_device`, so position_ids end up created on CPU,
   causing a device mismatch inside rope(). -> Fixed by also excluding audio_vae in the window.
   **Lesson**: this kind of pinning should have been written with the intent of excluding
   "every nn.Module that comes before transformer", not "modules named so-and-so". Ordering
   changes at the library's discretion.

2. **On the batch path, `_execution_device` cannot be resolved to begin with.** Batch
   reorders the phases so that "layout/latents/timesteps are finished during the encode
   phase, and loading the transformer is deferred until the denoise phase". At that point
   the transformer is not yet loaded and TE has already been detached, so the scan falls
   back to whatever CPU-resident module is found, or to a fallback `cpu`. The value itself
   is built correctly, so it is enough to explicitly carry it via `_scene_state_to_compute()`
   right before denoise.
   **Lesson**: a design that reorders phases also shifts the implicit assumptions tied to
   phase order (here, "the compute GPU can be resolved at this point").

## A3. A constraint of the verification environment -- the preview page does not render while `hidden`

Recorded because UI verification burned 15 minutes on this. The preview browser sits in a
state where `document.visibilityState === 'hidden'`, and **rendering (compositing) does not
run**. Consequences:

| Symptom | Cause | Workaround |
|---|---|---|
| `preview_screenshot` times out at 30 seconds | Nothing renders, so compositing can't happen. Guaranteed to stall when the gallery has 165 `<video>` elements | Measure the DOM directly via `preview_eval` (text, position, height, visibility) |
| `IntersectionObserver` never fires | Intersection calculation doesn't run on a hidden page | Verify lazy loading via **network logs** instead (mp4 request count 165->0) |
| `preview_click` (coordinate click) doesn't land | Hit testing requires layout/compositing | Call `element.click()` from `preview_eval` instead |

**Lesson**: UI verification in this environment must be done via **DOM measurement and
network logs**, not "appearance". Don't mistake an inability to take a screenshot for a
code defect (in fact, lazy loading was briefly misjudged as "not working" once because of this).

## A4. An agent-operation incident -- a chain of re-delegation (**recurrence**)

The "grandchild agent incident" already recorded in [[stdgen-spike-verdict]] recurred. An
agent delegated to investigate did not do the work itself and instead **spawned yet another
agent**, which then reported completion as nothing more than "spawned an agent" -- this
chain happened three times. A stop instruction was issued, but one instance survived and
ran to completion, burning roughly 230,000 tokens redundantly.

**Countermeasure (which actually worked for the later delegations)**: state at the very top
of the delegation prompt that **"you must complete the work yourself via Read/Edit/Bash.
Spawning sub-agents (the Agent tool) is forbidden."** Across the six delegations that
followed, this never recurred once.

**A secondary lesson**: the output of the incident-prone investigation was itself useful (a
diff-level audit deeper than my own direct investigation, which corrected two points in my
own report). Still, the token cost was duplicated, so **writing the prohibition before
delegating is far cheaper**.

## A5. For a migration, take the regression baseline first

Before bumping diffusers, **the old pin was restored and baseline MD5s were captured first**
(t2i/t2va/batch reused the same-day measurements already on hand; the ref family was
restored to main + abc5e9b and captured over 15 minutes). Without this ordering, there is no
way to judge "is the post-migration output correct".

The captured baselines were saved to
**[docs/internal/regression_baselines.json](regression_baselines.json)** (since scratchpad
gets wiped). Next time diffusers is bumped, it is enough to regenerate this file's `cases`
under the same conditions and diff the md5s. Captured at abc5e9b, and after tracking f37ab93
all cases still matched, so **it also serves as the expected values for the current f37ab93**.

## A6. Layering the documentation

This report has been demoted to a "work log (internal document)", and the outward-facing
technical description has been split out into
[docs/TECHNICAL_OVERVIEW.en.md](../TECHNICAL_OVERVIEW.en.md). The two have different
readers: someone who wants to know the spec and performance shouldn't have to read a
narrative mixed with failures, while conversely **for a developer who wants to avoid the
same pitfalls, the narrative itself is the value** -- that's the split.

Japanese/English parity was also done at the same time (separate files with cross-links;
keeping both languages in a single file was rejected, since in a repository where measured
values keep getting appended, one language would inevitably go stale).

---

# Addendum: 2026-08-11 to 12 (the campaign to hit the low-VRAM goals)

The main body and the A-addendum cover through fixed-cost reduction on the 48GB box. Here
we record the pitfalls and lessons hit while, **with the projected TE (Qwen3-VL-4B + a
trained linear map) as the axis, achieving the two final goals: full functionality on a
single real 16GB card, and up to t2va on 8GiB×2.** The settled specification lives in
[docs/TECHNICAL_OVERVIEW.en.md](../TECHNICAL_OVERVIEW.en.md) and
[docs/RESIDENCY.en.md](../RESIDENCY.en.md), so only pitfalls and lessons are written here.

## B1. plain mode × `H3_TE_DEVICE` device mismatch (a new variant of the `_execution_device` pitfall)

On the very first run of the projected TE (4B NF4) parked on `H3_TE_DEVICE=cuda:1` under
**no lowvram (plain)**, it crashed with a device mismatch inside rope(). The cause was the
same `_execution_device` as in the main body §4 and A2, but **the firing site was different
again**.

- Detaching TE drops `_execution_device` onto the CPU-resident audio_vae (same shape as the
  first A2 incident). A2 added audio_vae exclusion to `_pin_execution_device_to_compute()`,
  but only for the window on the lowvram path.
- plain (non-lowvram) has two other branches (catch-all / bnb-4bit+fl2va), and their
  layout–timesteps sat **outside the pinning window**. plain × `H3_TE_DEVICE` was a first run
  here, and this is the only combination that traverses those branches.
- **Fix**: wrap those branches in `_pin_execution_device_to_compute()` too, but only when
  `self._te_external` is true (the ordinary path with TE co-resident stays byte-for-byte
  unchanged).

**Lesson**: the intent gained in A2 — "exclude every nn.Module that comes before
transformer" — was correct, but we overlooked that **there are multiple windows where that
intent must be applied, one per path**. When introducing pinning, verify "which paths
traverse this window" by enumerating all paths. A new env combination traverses, for the
first time, a branch of existing code that has never been exercised.

## B2. Silent recalibration of the upstream projection matrix (an update that is not same-named but is effectively a different function)

The default file for `H3_TE_PROJ` started 404ing. The distributor had moved the old
`h3_qwen3vl_4b_tap24.safetensors` into `obsolete/` and replaced it with a **recalibrated
version** `mmh3-4b-ClipProj.safetensors` (training 1,666→5,664 prompts,
cos_test 0.711→0.717).

- The rename made the 404 noticeable, but **the contents were also a different thing**. The
  cosine against the old matrix W was 0.9596, which is at the level of "effectively a
  different function". Had we looked only at the name and treated it as "a re-upload of the
  same thing", we could have confused the quality delta with the nature of the projection.
- **How it was confirmed**: the old matrix was still in the local HF cache
  (snapshot `3f762f19`), so running old vs new on the same seed let us confirm they were
  **different by PNG MD5, and different tensors by the hash of W**. The old matrix is
  reproducible by passing its absolute path to `H3_TE_PROJ`.
- Both old and new have comparable PSNR against the 32B baseline (22.49→22.36dB); the
  new-vs-old NF4 PSNR is 39.29dB — "effectively the same image even with recalibration + a
  different GPU".

**Lesson**: **distrust same-named/renamed updates of distributed artifacts**. Files in an HF
repo silently change contents. If the old version is still in the local cache, you can
hash-compare old vs new — so don't wipe `~/.cache/huggingface` snapshots before tracking an
update (note the hashes before wiping).

## B3. sage attention is an sm_120-only build → sm_89 requires `H3_ATTN_BACKEND=default`

The added RTX 4060 Ti is **sm_89**. This app's sage is 2.2.0 source-built for sm_120 and
cannot be used on an sm_89 card. The single-16GB verification must launch with
`H3_ATTN_BACKEND=default` (SDPA).

**Lesson**: the attention backend is tied to the card's compute capability. When the GPU
changes, first check sage's supported arch, and if unsupported, fall back to SDPA (launching
with the default `sage` will crash). This also ties into B4 below: falling back to SDPA
changes the trajectory and stops FBC from working.

## B4. On the int8+SDPA trajectory, FBC skips 0 steps + the composition changes at the same seed → cross-configuration quality is judged visually

In the single-16GB run (int8 group + SDPA), **FBC skipped not a single step**
(`cache_skipped_steps: 0`, threshold 0.05). In contrast to the PRO 6000 + sage + bf16
trajectory where most steps were skipped, on the int8+SDPA trajectory the residual never
drops below the threshold.

At the same time, PSNR is numerically catastrophic (7.40dB vs the 32B baseline / 7.43dB vs
the prior stage), but **visually it was not degradation**: the prior stage (bf16+sage)
converged at seed 4242 to an anime-style lake at dusk (not reflecting the prompt's
"photorealistic"/"snow-capped peaks"), whereas this run (int8+group+SDPA) produced a
photorealistic snowy mountain and a lake in morning mist — **prompt fidelity actually
improved**.

**Lesson**: int8+SDPA moves generation to a different attractor. **Cross-configuration
PSNR/MD5 comparison is simply invalid to begin with** (even further out than the known ~19dB
int8 trajectory divergence). Judge cross-configuration quality visually. FBC's skip rate is
also trajectory-dependent, and "doesn't work" ≠ "broken" (there is up to a 2x headroom from
threshold tuning, but it is unverified and trades against quality).

## B5. Isolating the cause of slowness — this box's 2nd slot is Gen3 x4 (don't confuse it with the card's performance)

On single-16GB, t2i was 16.5s/step and t2va 49.8s/step, very slow. group offload streams
~34GB of int8 weights CPU→GPU every step, but the slot the 4060 Ti sits in is
**Gen3 x4 (effective ~3.5GB/s)**. Transfer alone comes to ~10s/step, consistent with the
measurement.

- **How it was confirmed**: `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current`
  directly confirmed the slot is Gen3 x4. The bottleneck is the per-step transfer, not the
  denoise computation itself.
- On **a proper Gen4 x16 slot, transfer would be ~1/8**, so this time is a value of "this
  box's slot", not "the 16GB card's performance". t2va's 49.8s/step is that plus SDPA
  compute (sm_89) for 124 frames.

**Lesson**: group-mode speed is PCIe-bandwidth-bound. When you see slowness, **suspect the
slot's Gen/width, not the card's compute** (checkable via nvidia-smi). Don't record a value
specific to this box as the card's general performance.

## B6. 838MiB whole-length fp32 conversion at the end of decode OOMs after denoise completes → move de-normalization to the CPU

The first 8GiB×2 t2va attempt OOMed on **the last line of decode, after denoise ran
29/29 to completion**. The final line of upstream `decoders.py`
`MiniMaxH3VideoDecodeStep.__call__`,
`(video.float() * pixel_std + pixel_mean).clamp(0,1)`, converts the whole-length fp16 decode
result to fp32 in one shot on the GPU — at 768², 124 frames that is
124×768×768×3×4B = **838MiB** temporary (exactly matching the OOM message
"Tried to allocate 838.00 MiB"). The same family as the "whole-length conversion at the end
of decode" we crushed with `frames_to_uint8` in the main body was still present on the
normalization side.

- **Handling**: per the no-touching-venv rule, a runner-side subclass
  (`_cpu_norm_video_decode_step()`, cloning f37ab93's `__call__` with a single change):
  **move to CPU while still fp16, then de-normalize on the CPU**.
- **The equivalence argument**: element-wise fp32 mul/add/clamp has no reduction and no FMA
  fusion, so IEEE754 rounding agrees between CPU and GPU → the output is bit-identical.
  **Measured and confirmed a same-seed PNG MD5 exact match
  (`1a2a136b61234b4917465604ac35cca2`) before and after applying** (proven by byte match,
  not "should be the same"). The only added cost is one PCIe transfer of the whole-length
  fp16, ~420MiB.
- Applied unconditionally to all paths (t2va/t2i/ref2va/batch). In every configuration, a
  few whole-length fp32 tensors disappear from the decode-phase peak.

**Lesson**: the "denoise passes but decode falls over" pattern points to a whole-length dtype
conversion at the tail. CPU/GPU equivalence of element-wise ops holds when there is no
reduction, and when it holds the rounding matches, so **it can be proven by MD5** (an
equivalent relocation, not an approximation).

## B7. `H3_VIDEO_VAE_FP16=1` × references instant death by dtype (decode has autocast, encode does not)

Running the reference family (ref2va / fl2va) under `H3_VIDEO_VAE_FP16=1` dies instantly
(even at 96GB, regardless of VRAM amount) with
`Input type (float) and bias type (c10::Half) should be the same`.

- **Why**: `H3_VIDEO_VAE_FP16=1` permanently casts the VAE weights to fp16. **Decode** is
  fine because the upstream step wraps its own fp16 autocast, but the **encode** side
  (encoders.py `encode_vae_condition` — used by ref2va's references and fl2va's keyframe
  conditioning) has no autocast and passes explicitly fp32-ified pixels to `vae.encode()`.
  The cause is the upstream **asymmetry of decode having autocast and encode not**.
- **Why it stayed latent**: every ref2va regression so far was on an fp32 VAE configuration,
  so **it could never have been found by fp32 VAE regressions alone**. It surfaced only when
  the low-VRAM goal combined `H3_VIDEO_VAE_FP16=1` × with-references for the first time.
- **Fix**: in `_load_vae`, right after the fp16 cast, wrap `vae.encode` in an fp16 autocast
  symmetric with the decode side. Precision is within design (`encode_vae_condition`
  originally rounds its result to fp16 itself via `latents.to(torch.float16).float()` before
  returning).

**Lesson**: upstream can be **asymmetric in how it applies autocast between decode/encode**.
Running regressions on only one path (fp32 VAE) makes bugs specific to the other path
structurally undetectable. The regression matrix is the Cartesian product of "flag ×
feature", and at least the dangerous combinations (fp16 VAE × references) must be included
explicitly.

## B8. group's pinned ~34GB stays in the host caching allocator (empty_cache is device-side only)

In group mode, **switching modes t2va→ref2va was permanently impossible**. The cause was not
VRAM but **residual host-side pinned RAM**.

- group offload (use_stream=True) places ~34GB of int8 weights in **pinned memory**.
  `_free_transformer`'s del+gc leaves the pages **held in torch's host-side caching allocator,
  never returned to the OS** (`torch.cuda.empty_cache()` is device-side only), so
  MemAvailable stays ~34GB short. The subsequent transformer_ref load is rejected by the RAM
  guard (which requires 40GB).
- **How it was confirmed**: measuring avail after freeing showed 38.6GB (RssShmem 45.6GB
  still held). "RAM doesn't come back even though it was freed" was confirmed numerically.
- **Fix**: add `torch._C._host_emptyCache()` (a private API, so `getattr`-guarded) to
  `_free_transformer` / `_free_transformer_ref`, only in group mode. After the fix, avail
  recovers fully to **85.8GB** right after freeing, and t2va→ref2va switching works in group
  mode for the first time.

**Lesson**: **the host-side pinned cache is a separate ledger from device freeing.**
`empty_cache()` is device-side only, and pinned host pages are returned to the OS only by
`_host_emptyCache` (a private API). In a configuration that uses lots of pinning, like group
mode, return this explicitly at mode switch. Since it's a private API, check its existence
via `getattr` before calling (it can disappear across torch versions).

## B9. Methodology for VRAM-ballast simulation (why it skews slightly stricter than a real card)

The 8GiB×2 verification simulated headless 8GiB cards by capping the added real 16GB card
and the 96GB card with `scripts/vram_ballast.py --target-free-gb 7.9` (**in GiB**).
Compute side = 4060 Ti, TE side = PRO 6000 (assigned to the app's cuda:0/cuda:1 via
`CUDA_VISIBLE_DEVICES=1,0`).

- The ballast fills the rest while leaving the specified free amount. Since `--target-free-gb`
  is in GiB, it is **slightly stricter** than the effective budget of an 8GiB card (the
  ballast's own allocator fragmentation and the CUDA context eat a bit more than a real
  card's "raw free"). In other words it skews toward "if it passes with ballast, it passes on
  a real card" — safe as a go/no-go check.
- We ran both the real card (Goal A) and ballast (Goal B) to **back up whether the ballast
  correctly approximates a real card** on the real-card side.

**Lesson**: ballast simulation skews stricter than a real card, so it is **safe for go/no-go
decisions**, but **it does not simulate speed** (PCIe bandwidth and compute capability are
specific to the real slot/real card). Measure capacity feasibility with ballast, and speed on
the real card.

## B10. Operating sub-agent verification (Bash 10-min cap → nohup + polling)

Low-VRAM generation takes 25–66 minutes per run (because of this box's Gen3 x4 slot). A
delegated sub-agent's Bash has a 10-minute cap and can't wait for long-running generation as-is.

- **Handling**: launch generation in the background with `nohup ... &` and poll
  `logs/server.log` to detect completion. Even if curl disconnects midway, **generation
  continues server-side** and the result is recoverable from `outputs/` and `server.log`
  (generation is serialized by a global lock, so a disconnect doesn't lose the job).
- The "no re-delegation, complete it yourself" at the top of the delegation prompt continues
  to apply (the A4 lesson).

**Lesson**: verify long-running jobs with "launch + poll + collect logs", not "wait
synchronously". Don't confuse a curl disconnect with the life/death of the server-side job
(an HTTP connection is not the job's lifetime).

## B11. Comparing code paths without pinning resolution "discovered" an effect that did not exist (2026-08-12)

**What happened**: comparing the reference batch endpoints (`/api/ref2i_batch`,
`/api/ref2va_batch`) against single requests (`/api/ref2va`) showed **the batch denoising 1.75x
slower per step**. The ratio held at 1, 2 and 3 scenes, held with `H3_REF_PREFIX_CACHE=0`, and
both paths were confirmed to be on the sage backend — so the conclusion was nearly "the batch
path carries a scene-count-independent overhead".

**Why**: the single-request curl passed `height=768 width=768`; **the batch curl did not**. When
omitted, the server resolves H3's own 16:9 canvas, so only the batch generated at **1344×768** —
**1.75x the pixels** of 768².

**How it was pinned down**: reading the output mp4 dimensions with `av` showed 1344×768 versus
768×768. The clincher was that **the pixel ratio 1344×768 ÷ 768² = 1.750 matched the measured
step-time ratio 12.84 ÷ 7.323 = 1.753** almost exactly. Re-measured at matched resolution, the
batch's per-step time matched a single request **exactly** (ref2i 2.598s vs 2.599s; i2va 7.321s
vs 7.323s) — the batch path's overhead is actually zero.

**Fallout**: that bad measurement was read as "video gets no benefit from sharing the reference
encode", and the `i2va 103s → 45s` estimate was withdrawn from the README on the strength of it.
Re-measured at matched resolution, the sharing model holds for stills and video alike (+1.0s and
+4.6s against prediction), and the estimate was reinstated.

**Lessons**:
1. **When comparing speed across code paths or modes, pin resolution, duration and step count
   explicitly.** "Let the defaults apply" is not a comparison when the defaults differ per path.
2. When a ratio lands on **a clean number like 1.75**, suspect geometry (pixel count, token
   count) first. Checking input size is faster than auditing implementation differences.
3. Output dimensions are readable from the artifact even when the result JSON's `height`/`width`
   are `null`. **Verify measurement conditions from the artifacts, not the request.**
4. Conclusions drawn from a bad measurement get retracted even after they are written down. Here
   the "withdraw → re-measure → reinstate" history was kept in the README rather than quietly
   edited away.

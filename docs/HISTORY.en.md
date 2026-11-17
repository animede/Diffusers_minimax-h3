# Development history (minimax-h3)

[日本語](HISTORY.md) | **English**

Dated work log, verification notes and backlog, split out of the README. Each section is the original README text moved verbatim and describes the state at that time (where things changed later, [FEATURES.en.md](FEATURES.en.md) and [README.en.md](../README.en.md) take precedence).

- [Where things stood as of 2026-08-12](#where-things-stood-as-of-2026-08-12)
- [Chronological work log](#chronological-work-log)
- [Backlog and external events](#backlog-and-external-events)

## Where things stood as of 2026-08-12

### Recent updates (2026-08-11 to 08-12)

- **Both final goals reached**: full functionality on a single real RTX 4060 Ti 16GB (t2va
  768² peak 11.4GB; the maximum is 768×1344 at a real 15.2GB) / t2va 5s 768² completes at
  full resolution on 8GB × 2 (peak 7.23GB).
- Moved the decode-tail denormalization to CPU (bit-identical, lowers the decode peak on all
  configs).
- Fixed a dtype instant-death on `H3_VIDEO_VAE_FP16` × reference (added an fp16 autocast on
  the encode side too).
- Freed the residual pinned RAM in group mode, so t2va ↔ ref2va mode switching now works.

### Where things stand and where to resume (as of 2026-08-12)

Read this first when resuming work.

| What you want to know | Where to look |
|---|---|
| **Spec, performance, design** (what is achieved and how) | [docs/TECHNICAL_OVERVIEW.en.md](TECHNICAL_OVERVIEW.en.md) |
| **Pitfalls, failures, operational lessons hit** (to avoid the same holes) | [docs/internal/TECHNICAL_REPORT.en.md](internal/TECHNICAL_REPORT.en.md) |
| **Decisions on incorporating community improvements** | [docs/COMMUNITY_IMPROVEMENTS.en.md](COMMUNITY_IMPROVEMENTS.en.md) |
| Which mode loads/releases what, and when | [docs/RESIDENCY.en.md](RESIDENCY.en.md) |
| **What to do next** (not yet started / unverified) | This README's "[Waiting on external events going forward](#waiting-on-external-events-going-forward-backlog-as-of-2026-08-06)" §3 |
| Regression baselines for when diffusers gets bumped | [docs/internal/regression_baselines.json](internal/regression_baselines.json) |

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

## Chronological work log

### 16GB-tier verification result: not supported (floor is ~18GB, confirmed 2026-08-06)

With a 16GB ballast (15.5GB free), **OOM occurs near the end of TE loading (nf4
quantization)** (fails to allocate +250MiB at 15.37GB used). The pruned TE-nf4's resident
17.45GB is itself the floor, and video VAE fp16 — a decode-stage countermeasure — cannot
move this floor. **Completes successfully with an 18GB ballast** (peak 17.72GB, total
302s) — i.e., the practical lower bound of the current architecture is **~18GB** (the
24GB-tier configuration `H3_LOWVRAM=group H3_TE_PRUNE=1` works as-is at the 18GB tier
too). Breaking through 16GB would require streaming execution of TE or sub-4-bit
quantization (a separate, not-yet-started task).

### 2026-08-09: Migration to the PR #14355 merged version (f37ab93), stage 1 — t2i/t2va/batch, identical-seed MD5 exact match

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

### 2026-08-09: Merged-version migration stage 2 — the ref2va family also completes with identical-seed MD5 exact match

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

### 2026-08-09: Two-row tab layout (top row = video, bottom row = still image) and making Ref2I its own tab

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

#### Lazy-loading gallery thumbnails

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

#### UI language switching (i18n)

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

### 2026-08-10: Lip sync from an audio reference (`fully_copy`) — measurements, and the bug found along the way

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

#### [Bug fix] ref2va with an audio reference always crashed under sage attention

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

### 2026-08-10: Breaking down the decode-phase peak VRAM — the composition, and the 15% we took from it

To decide whether to port ComfyUI's
[PR #15446](https://github.com/Comfy-Org/ComfyUI/pull/15446) (chunk-streaming the H3 VAE so
the decode VRAM becomes duration-independent), **we first broke down what our own
decode-phase peak of 16.29GB is made of**.

#### The breakdown (768×1344, 107 frames, fp32, measured)

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

#### The 15% taken first — the uint8 conversion's intermediate tensors

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

#### Porting PR #15446 is on hold

Against 3.08GB of activations (19%), it would require a monkeypatch replacing the
upstream `_decode` wholesale, incurring a tracking cost. Having taken the cheaper 15%
first, the next candidate with better expected value is **fp16-ifying the audio_vae (part
of the 11GB of weights)** (the current fp32-fixed rationale is the measured "bf16 drops the
volume by about 20dB", but **fp16 is unverified** — bf16 has a 7-bit mantissa, fp16 has 10
bits, and small-amplitude audio can behave differently).

### 2026-08-11: Added an RTX 4060 Ti 16GB to the 96GB box — stage 1 toward the low-VRAM goals (TE onto the second GPU)

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

#### Two bugs found and fixed (surfaced by a Sonnet agent's verification)

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

#### Observation: steady-state t2i is +10s vs single-GPU

Denoise + decode stay at 14.4s; **the per-request fixed cost is about 29s** (vs about 19s
single-GPU). The main suspects for the difference are the encode running on the sm_89
4060 Ti (NF4 dequant is slower) + the PCIe transfer of the cond. Meanwhile t2va did not get
worse at 134.6s — the longer the duration, the thinner the fixed cost spreads. On the 96GB
box the TE fits on GPU0 anyway, so **there is no practical benefit to using the second GPU
on this box** (the real targets are the 16GB-single and 8GB×2 configurations).

#### Next stages (unmeasured)

- **Stage 2 = goal A**: cap GPU0 to a 16GB equivalent with ballast, `H3_LOWVRAM=group` +
  projected TE **cohabiting** (no TE_DEVICE). Derived 3.11+1.4+6.6 = 11.1GB should fit, but
  **the te_proj × group combination has never run**
- **Stage 3 = goal B**: 8GB×2. The TE's 3.41GB fits on GPU1, but GPU0's blocks +
  activations of 8.0GB exceed the effective budget of 7.1GB by 0.9GB — trimming the
  activations via resolution/duration needs consideration

### 2026-08-11: Stage 2 = goal A achieved — t2i / t2va complete on a real RTX 4060 Ti 16GB **alone**

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

#### The time is heavy mainly because "this box's second slot is PCIe x4"

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

#### Quality: not "degradation" but a trajectory divergence — and prompt fidelity actually improved

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

### 2026-08-11: Stage 3 = goal B achieved — **t2va 5s 768² completes at full resolution on 8GB×2**

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

#### But one thing was fixed: an 838MiB all-at-once fp32 conversion at the decode tail OOM'd → denormalization moved to CPU

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

#### Summary: both final goals achieved

- **Goal A (single 16GB)**: full functionality on a real RTX 4060 Ti 16GB, peak 11.4GB
  (~4.7GB headroom)
- **Goal B (8GB×2)**: compute 8GiB + TE 8GiB completes t2i / t2va 5s 768² (peak 7.23GB)

The remaining caveats are speed only (this box's second slot is Gen3 x4, hence
16.5-51s/step; on Gen4 x16 the transfer is ~1/8) and the FBC not engaging on the int8+SDPA
trajectory (threshold tuning unverified).

### 2026-08-11: How far do ref2va/i2va go — two latent bugs found and fixed, and the low-VRAM boundary settled

The reference family (ref2i / i2va = image-reference ref2va / audio reference / 768×1344)
was verified on the goal A/B configurations. The first pass was a **total wipeout**, but
the cause was not VRAM — it was **two latent bugs that only fire in combinations never
exercised together until today**. After the fixes, the boundary is decided plainly by VRAM
amount.

#### Bug 1: `H3_VIDEO_VAE_FP16=1` × any reference always crashes, regardless of VRAM (dtype mismatch)

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

#### Bug 2: in group mode, the t2va→ref2va mode switch was permanently impossible (residual pinned RAM)

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

#### The boundary after the fixes (measured)

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

#### Quality (the first visual verification of the projected TE's vision path): references clearly take effect

Since "the projection matrix was calibrated on text only", vision quality was unknown — but
visually no degradation could be detected: the reference person's face, bangs, hair length,
cardigan color, white ribbon and necklace are all consistently reflected, and the i2va
compositions (profile shots) are nearly identical to the 32B ground truth
`test1_walk_park.mp4`. With an audio reference the lip movement is clear too. The only
blemish is a light compositional drift late in the 768×1344 run (not a breakdown).
Outputs: `outputs/ref2i_1786447720.png` / `ref2va_1786449645.mp4` /
`ref2va_1786452054.mp4` / `ref2va_1786455323.mp4`.

### 2026-08-12: Everything stacked on the 96GB box — t2i 7.65s / t2va 28.56s, and two wasted costs it exposed

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

#### Waste 1: a co-resident projected TE gets freed on every request

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

#### Waste 2: a bf16 transformer is freed and reloaded for 12s every request

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

### 2026-08-12 (part 2): stopping the decode-window free — t2i 7.4s on a single GPU at 45.6GB, and the 48GB-tier goal falls out

"Waste 2" above was fixed on the spot. **The change is one import-time guard condition**:
`H3_KEEP_TRANSFORMER` used to require `H3_LOWVRAM=1`; that was relaxed to "anything but
`group`", so plain mode (`H3_LOWVRAM=0`) can skip the decode-window free too. The branch that
skips the free (`if H3_KEEP_TRANSFORMER: pass`) was already shared, and the restore path's
`_ensure_transformer` is idempotent, so **no additional implementation was needed**. The other
two conditions (the TE is not on GPU0, i.e. `H3_TE_DEVICE` or `H3_TE_PROJ`; and
`H3_VIDEO_VAE_FP16=1`) are exactly what makes plain mode fit as well (66.3 + 11.4 fp16 decode
= 77.7GB).

#### All configurations compared (96GB box, turbo 4 steps, 768²)

| Configuration | t2i steady-state | t2va 5s | Peak | GPUs |
|---|---|---|---|---|
| bf16 + TE on GPU1 + **no free** (B) | **6.89s** | **26.8s** | 74.2GB + 3.2GB | 2 |
| bf16 **single GPU** (TE on GPU0) + no free (A) | 7.08s | 27.04s | 77.3GB | 1 |
| **int8 single GPU + no free (C)** | 7.40s | 28.13s | **45.6GB** | 1 |
| int8 + `H3_LOWVRAM=1` + KEEP + TE on GPU1 (previous best) | 7.65s | 28.56s | 42.5GB | 2 |
| int8 single GPU, **with the free** (C at KEEP=0) | 19.58s | — | 39.8GB | 1 |
| bf16 + TE on GPU1, with the free (previous section) | 19.9s | 40.0s | 68.9GB | 2 |

**Equivalence**: configuration C and the same configuration at `H3_KEEP_TRANSFORMER=0` produce
a PNG with an **identical MD5** (`596a718e4b5cf9a0b907d2ec479225d2`). Skipping the free is
mathematically a no-op, and the same image comes out in **19.58s → 7.40s (2.6x)**.

#### What this establishes

- **Where the TE lives no longer matters.** Single GPU (TE on GPU0) at 7.08s versus two GPUs
  (TE on GPU1) at 6.89s is a 2.7% difference. **The second GPU only mattered while the free was
  still there** — the "2x difference from TE placement" in the previous section was an artifact
  of that free.
- **The free itself was the cost** (2.6x under otherwise identical settings).
- **The practical optimum is configuration C**: int8, single GPU, **45.6GB peak**. Against a
  48GB-tier effective budget of about 49.8GB that leaves roughly 4.2GB of headroom, so **a
  single 48GB card can run within 7% of the fastest configuration**. The backlog item
  "[TE + transformer co-resident on the 48GB box](FEATURES.en.md#unverified-important)" is satisfied in this
  form (not yet confirmed on the physical 48GB box — this is the 96GB box's measured peak
  shown to fit the 48GB budget).
- bf16 denoises faster than int8 (t2i 2.05-2.07s vs 2.39-2.40s) but **needs a 77GB-class
  card**. Whether that beats int8's 45.6GB at a 7% penalty depends on the card you have.

### 2026-08-12 (part 3): reference-mode speed (ref2i / i2va) — a turbo bug, and the fixed costs that remain

The reference modes were measured in the fastest configuration above (int8, single GPU, no
decode-window free, 45.6GB peak). Reference generation uses `transformer_ref` — a separate
model — so both its budget and its bottleneck differ from t2va.

#### First, a bug: the reference endpoints never picked up turbo's step default

Requests were running at `steps=30` despite `turbo=1`. The cause was in `app.py`: **the three
reference endpoints (`/api/ref2va`, `/api/ref2i_batch`, `/api/ref2va_batch`) hardcoded
`num_inference_steps: int = Form(30)`** while t2va/t2i/t2i_batch/fl2va use
`DEFAULT_NUM_INFERENCE_STEPS` (which is 4 under turbo). The turbo LoRA itself was being applied
to `transformer_ref` correctly (the log shows `turbo LoRA lazily applied to transformer_ref
(312 layers wrapped)`), so the combination running was **a distilled LoRA driven for 30 steps** —
a mismatch. Fixed by aligning all three to `DEFAULT_NUM_INFERENCE_STEPS`.

#### Measurements (96GB box, int8, single GPU, 768²)

| Mode | turbo, 4 steps | 30 steps (no turbo) | Denoise (4 / 30) | Peak |
|---|---|---|---|---|
| **ref2i** (reference still, 22 frames) | **79.3s** | 148.4s | 7.8s / 72.1s | 45.4GB |
| **i2va** (image reference to 5s video, 124 frames) | **103.1s** | 290.3s | 22.0s / 209.0s | 45.9GB |

**The peak is 45.9GB, essentially the same as the t2va family's 45.6GB** — the reference modes
also fit a single 48GB-class card. Turbo helps less here (2.8x for i2va) than for t2va (5.5x)
because, as below, **the non-denoise fixed costs dominate**.

#### The backlog item "ref2va × turbo" holds up (verified visually)

The turbo LoRA had only ever been measured on `transformer`, never on `transformer_ref`. **At 4
steps the reference fidelity holds**: bangs, hairstyle, light-green cardigan, white ribbon and
necklace all match, and across the video the subject stays consistent from first to last frame
with the camera tracking and the park setting as prompted, with no breakdown
(`outputs/ref2i_1786509275.png`, `outputs/ref2va_1786509457.mp4`). **Strength 0.094 carries over
to reference-conditioned trajectories unchanged.**

#### The fixed costs that remain (i2va's 103.1s, from log timestamps)

| Phase | Time | Note |
|---|---|---|
| **Reference vision encode** | **about 47s** | **The dominant cost.** The 4B projected TE does not shrink it (though it is an improvement on the 32B era's ~65s/scene) |
| Denoise (4 steps) | 22.0s | |
| Decode + VAE round trips | about 10s | |
| Reference VAE encode | about 6s | |
| **Reloading the t2va transformer at the end** | **about 13s** | **Pure waste for consecutive ref2va requests** |

- **The 47s vision encode is the real target** for reference modes. The batch paths already share
  it across scenes (`H3_REF_PREFIX_CACHE`), but **repeated single requests do not share anything**
  — a cross-request cache would pay off whenever the same reference image is reused.
  → **Implemented on 2026-08-24 as `H3_REF_PREFIX_CACHE_SINGLE` (default 0)**; see the section
  "Reusing the prefix across single requests" above.
- **The 13s reload** comes from restoring the "t2va steady state" when a ref2va request finishes.
  If the next request is also ref2va it is unnecessary, and the same "don't restore" judgment as
  `H3_KEEP_TRANSFORMER` could apply (**not implemented**).

#### Confirming the estimate (same day, via the batch measurements)

The estimate that removing these two costs takes **i2va from 103s to about 45s** is **supported
by the batch measurements**. The batch path is literally the experiment "what happens if the
reference encode is shared", and with everything pinned to 768²:

| | Single | Per item, batched | Measured saving | Model's prediction | Difference |
|---|---|---|---|---|---|
| ref2i (3 scenes) | 79.3s | 47.0s | 32.3s/image | 31.3s/image | **+1.0s** |
| i2va (2 scenes) | 103.1s | 75.0s | 28.1s/video | 23.5s/video | **+4.6s** |

**Stills and video both land on prediction** (slightly better, in fact). The batch's per-step
time matches a single request (i2va 7.321s vs 7.323s), so the difference is purely how many
times the 47s encode is paid. A cross-request cache (47s) plus dropping the reload (13s) gives
**103.1 − 47 − 13 ≈ 43s**, consistent with the evidence as it stands (**still not implemented**).

> **This was briefly withdrawn in error**: the first batch measurement omitted `height`/`width`
> on the batch side only, so it generated at 1344×768 — 1.75x the pixels of 768² — which read as
> "video gets no sharing benefit" and prompted a retraction. Matching the resolution made the
> numbers agree. **Pin the resolution when comparing code paths** — see the boxed note in the
> batching section above.

## Backlog and external events

### Waiting on external events going forward (backlog, as of 2026-08-06)

#### 1. diffusers PR #14355 — **merged (2026-08-05), migration also complete (2026-08-09)**

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

#### 2. Waiting for a finished release of Turbo LoRA -> **resolved via the lightx2v variant (production-implemented 2026-08-08, see the "Turbo LoRA" section)**

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

#### 3. Unstarted improvement candidates (in priority order, none urgent)

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


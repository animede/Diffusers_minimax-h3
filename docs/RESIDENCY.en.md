# What is loaded and freed, and when — residency reference

[日本語](RESIDENCY.md) | **English**

The combinations of mode, quantization, turbo, and TE placement have grown, so this document lets you look up **"what is on the GPU, in this phase, for this configuration"** on a single page. Everything is derived from the `core/runner.py` code and cross-checked against measured logs.

> **Why this document exists**: on 2026-08-09, the peak VRAM breakdown was explained as "transformer + VAE during decode", but that was **wrong** (in fact the transformer is freed before decode). There are too many combinations to answer from memory without making a mistake — this is a real example of that. Check here from now on.

---

## 0. Bottom line first — two representative configurations

> **Two measurement machines**: the **96GB box** = RTX PRO 6000 Blackwell 96GB + an added
> RTX 4060 Ti 16GB, and the **48GB box** = RTX PRO 5000 Blackwell 48GB + RTX 4000 SFF Ada 20GB.
> Every measurement below states which box it came from (same split as the note at the top of
> the README).

### ① `H3_LOWVRAM=1` + TE on the 2nd GPU (freeing enabled) — 48GB box

```bash
H3_LOWVRAM=1 H3_TE_PRUNE=1 H3_TE_DEVICE=cuda:1
```

| Phase | On GPU0 (48GB) | GPU1 (20GB) |
|---|---|---|
| Idle | (almost empty, ~1.5GB) | **TE-nf4 17.4GB (stays resident)** |
| Encode | — | TE-nf4 17.4GB |
| **Denoise** | **transformer-int8 34.3GB + activations ~6.6GB = 40.9GB** ← **peak** | TE-nf4 |
| (transformer freed) | 0.3GB | TE-nf4 |
| Decode | vae pair 11.3GB + buffers | TE-nf4 |
| At exit | vae moved to CPU → 1.2GB | TE-nf4 (not freed) |

**Important**: denoise and decode **never overlap in time**. With `H3_KEEP_TRANSFORMER=0` (the default) the transformer is always freed immediately before decode (the `_free_transformer()` right before `generate()`'s decode phase). The 40.9GB peak is from **denoise**.

### ② int8 on a single GPU with freeing disabled (the practical optimum as of 2026-08-12) — measured on the 96GB box

```bash
H3_TRANSFORMER_QUANT=int8 H3_KEEP_TRANSFORMER=1 H3_VIDEO_VAE_FP16=1 \
  H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_TURBO_LORA=1
```

**A single GPU** holds both the transformer (int8) and the projected TE (NF4) **resident at all
times**, and nothing is freed even for the decode window. `H3_LOWVRAM` is not used (plain mode).

| Phase | On GPU0 |
|---|---|
| Idle (between requests) | transformer-int8 34.0GB + projected TE NF4 3.11GB (both resident) |
| Encode | same as above (the TE is not freed) |
| Denoise | same as above + activations |
| Decode | same as above + fp16 decode (**the transformer is not freed**) |
| At exit | vae moved to CPU. **Nothing else is freed** |

**Measured peaks** (96GB box, turbo 4steps, 768², steady state): t2i 45.0GB / **t2va 5s 45.6GB** /
ref2i 45.4GB / i2va 45.9GB. **Which phase the peak falls in has not been measured** (no per-phase
breakdown like configuration 1 above). Against a 48GB-class effective budget of ~49.8GB that
leaves ~4.2GB, so it is **expected to fit on a single 48GB card** (**not yet confirmed on real
48GB hardware** — all that was confirmed is that peaks measured on the 96GB box fit inside a 48GB
budget). See §5.5 for speeds and conditions.

---

## 1. What's "resident" and what's "per-request" — four axes

The decision is governed by four independent axes. The source of confusion is that these are orthogonal.

| Axis | Env var | Effect |
|---|---|---|
| **A. How much stays resident** | `H3_LOWVRAM` (0/1/group) | Most important. Determines the table in §2 below |
| **B. Which GPU the TE lives on** | `H3_TE_DEVICE` | Once set, the TE is **never freed at all** (stays resident on the other GPU) |
| **C. Whether loading is faster** | `H3_TE_PREQUANT` | Does not change **when** it loads. Only makes **the load itself** faster |
| **D. Whether freeing happens at all** | `H3_KEEP_TRANSFORMER` | **The largest speed effect of the four** (2.6x under otherwise identical conditions). It **overrides** A by stopping every transformer free (§5.5). Combines with any `H3_LOWVRAM` value except `group` |

Note that **`H3_TE_PROJ` (the projected TE) is also an alternative to axis B**: it shrinks the TE
to 3.11GB (NF4), so it can sit on the compute GPU instead of being parked on a second card. Axis
D's guard treats it as equivalent to `H3_TE_DEVICE`.

**turbo (`H3_TURBO_LORA` / request-level `turbo=1`) does not affect this table.** LoRA merely wraps the transformer and causes no change in residency (only a one-time 1.38GB weight load and wrap on first use). It forces FBC off and changes the default step count, nothing more.

---

## 2. Phase × residency, by `H3_LOWVRAM` value

### `H3_LOWVRAM=0` + `H3_TE_QUANT=bnb-4bit` (default for the 96GB-class box)

Both TE and transformer **stay resident at all times**. Only the VAE shuttles on and off the GPU per phase.

| Phase | Resident | Notes |
|---|---|---|
| Steady state (between requests) | transformer 66.3GB + TE-nf4 21GB = **87.5GB** | VAE stays on CPU |
| Encode | same as above | |
| Denoise | same as above + activations | |
| **Decode** | **transformer freed** → vae pair 11GB | because all three at once would be 98.5GB, over the 96GB budget |
| After decode | transformer reloaded (back to steady state) | |

> **This freeing can be stopped in plain mode too (2026-08-12)**: the "free the transformer for
> decode, reload it right after" above is a **measured 11.9–12.3s per request** of fixed cost, and
> it was why the bf16 plain configuration lost to the int8 resident one on the 96GB box. That
> freeing came from a budget that **assumes the 32B TE sits on GPU0**: TE-nf4 21GB + transformer
> bf16 66.3GB + VAE fp32 11GB = 98.5GB > 96GB. With the TE off GPU0 (parked via `H3_TE_DEVICE`, or
> the 3.11GB projected TE) that premise no longer holds: 66.3 + fp16 decode 11.4 = **77.7GB**
> (+3.11 if the projected TE is co-resident). `H3_KEEP_TRANSFORMER=1`'s import-time guard was
> relaxed so it applies to plain mode as well (§5.5). **Caveat**: stopping it in bf16 needs 77.7GB,
> i.e. an 80GB-class card in practice.

### `H3_LOWVRAM=1` (48GB-class)

The rule is **nothing stays resident between requests**. The one exception is the TE, if `H3_TE_DEVICE` is set.

| Phase | `H3_TE_DEVICE` unset | `H3_TE_DEVICE=cuda:1` |
|---|---|---|
| Entry | transformer freed (leftover from previous request) | same |
| Encode | **TE loaded** 17.4GB | TE already resident on GPU1 (**no load needed**) |
| layout/latents/timesteps | TE stays resident | TE detached from pipe, transformer loaded |
| (TE freed) | **TE freed** | **not freed** |
| Denoise | transformer-int8 34.3GB + activations | same |
| (transformer freed) | **freed** | **freed** (as of now) |
| Decode | vae pair 11.3GB | same |
| Exit | vae moved to CPU. **Nothing is reloaded** | same |

**Fixed cost per request**:

| | TE load | transformer load | Total |
|---|---|---|---|
| Plain configuration | 46.5–55.8s | 32.5s | **~85s** |
| + `H3_TE_PREQUANT=1` | 21–34s | 32.5s | **~55s** |
| + `H3_TE_DEVICE=cuda:1` | **0s** | 14.8–32.7s | **~25s** |

**When `H3_KEEP_TRANSFORMER=1` is layered on top** (either `H3_TE_DEVICE` **or** `H3_TE_PROJ` is required, details in §5.5): the "(transformer freed)" row and the "Decode" row in the table above change — the transformer is **no longer freed at the decode phase either**, and it **stays resident across requests too** (the "Entry" freeing step no longer happens either). It stays resident from the single initial load only, so the transformer's load fixed cost (14.8–32.7s) collapses to a one-time cost paid only on the first request. The decode-phase peak becomes transformer 34.3GB resident + fp16 decode ~11.4GB (`H3_VIDEO_VAE_FP16=1` required). See §5.5 for measurements.

### `H3_LOWVRAM=group` (24-32GB-class)

The transformer **stays resident in host RAM** and shuttles to the GPU block by block (1-2 blocks at a time, ~1.4GB).

| Phase | On GPU | Notes |
|---|---|---|
| At startup | transformer resident on CPU (int8, with group-offload hooks) | for the entire process lifetime |
| Encode | TE-nf4 21GB | |
| Denoise | TE-nf4 21GB + 1-2 blocks of the transformer ~1.4GB | |
| **Decode** | **TE forcibly freed** → vae pair 11GB | because TE (21GB) + decode (16.3GB) = 37GB exceeds 32GB |
| After decode | TE reloaded | because the next request needs the TE first |

**Note**: this is the only mode where it's the TE, not the transformer, that gets freed for decode. Since a group-offloaded transformer's GPU footprint is already small, it's the TE that has to give way instead.

> **group freeing requires _host_emptyCache (added 2026-08-12)**: group offload places ~34GB
> of int8 weights in **pinned host memory**. `_free_transformer` / `_free_transformer_ref`'s
> del+gc leaves the pages held in torch's host-side caching allocator, never returned to the
> OS (`torch.cuda.empty_cache()` is device-side only). **The host-side pinned cache is a
> separate ledger from device freeing**, and without returning it, the t2va→ref2va mode
> switch is rejected by the RAM guard (after freeing, avail 38.6GB / RssShmem 45.6GB still
> held → the subsequent transformer_ref load fails against its 40GB requirement). The fix adds
> `torch._C._host_emptyCache()` (a private API, so getattr-guarded), only in group mode, and
> avail recovers fully to **85.8GB** right after freeing. Details in addendum B8 of
> [docs/internal/TECHNICAL_REPORT.en.md](internal/TECHNICAL_REPORT.en.md).

---

## 3. Additional guarantees when `H3_TE_DEVICE` is set

- The TE is **not freed even when `_free_text_encoder()` is called** (even with `force=True`) — not freeing it is the whole point
- The TE is **normally detached from the pipe**. It's only connected during the `_te_attached()` window (while encoding)
  - Reason: `_execution_device` returns the device of the first nn.Module found in components order, so if it stayed attached, layout and **even decode** would end up building tensors on the wrong GPU and crash (reproduced on real hardware)
- During the layout/latents/timesteps window, `_pin_execution_device_to_compute()` **also temporarily detaches the vae** (otherwise, once the TE is detached, the next thing found would be the vae sitting on CPU, and `cpu` would be returned instead)
- **ref2va is rejected with a 400 if the TE's GPU has less than 24GB** (OOM measured at 20GB).
  The reference batch endpoints (`/api/ref2i_batch` / `/api/ref2va_batch`) are rejected by the same guard

> **That 24GB threshold needs revisiting (not yet fixed)**: `_te_external_usable_for("ref2va")`
> only looks at **whether the TE GPU's total capacity is ≥24GB**; it does not distinguish whether
> the resident TE is the 32B one or the projected 4B one. The threshold was derived from "32B TE
> (17.45GB) + ≥3.22GB of vision activations for a 2048px short side", which is **far too strict
> for the 3.11GB projected TE**. The practical damage: on a configuration that parks the projected
> TE on the 96GB box's second card (a 4060 Ti 16GB), **`H3_TE_DEVICE` cannot be combined with the
> reference family at all** (single or batch). The projected TE's vision path itself was verified
> visually on 2026-08-11, so this is **not a question of whether it works — it is the guard's
> criterion that is wrong**.

---

## 4. Common misconceptions

| Misconception | Reality |
|---|---|
| "The 40.9GB peak is transformer + decode VAE" | **Wrong.** The peak is during denoise (transformer 34.3 + activations 6.6). By decode time the transformer is already freed, leaving just VAE 11.3GB |
| "`H3_TE_PREQUANT` makes the TE stay resident" | **Wrong.** It doesn't change when it loads. **It only makes the load faster** (53s→29.5s) |
| "turbo increases/decreases VRAM" | Barely changes at all. LoRA is just a wrapper, only reading 1.38GB on first use |
| "`H3_LOWVRAM=group` is always more memory-efficient but slower than `1`" | It is more memory-efficient, but **for still images `group` is actually slower** (block transfer is a fixed cost, and for very short/lightweight compute, transfer dominates) |
| "Putting the TE on a separate GPU makes the fixed cost zero" | Only the TE's share. **The transformer load (14.8–32.7s) remains** (because it's still freed before decode). However, layering `H3_KEEP_TRANSFORMER=1` on top collapses this too, down to a one-time cost (§5.5) |
| "`H3_KEEP_TRANSFORMER` requires `H3_LOWVRAM=1` + `H3_TE_DEVICE`" | **Not any more (relaxed 2026-08-12)**. What it requires is: `H3_LOWVRAM` is not `group`, `H3_TE_DEVICE` **or** `H3_TE_PROJ`, and `H3_VIDEO_VAE_FP16=1`. It works in plain mode (`H3_LOWVRAM=0`) and in a single-GPU setup with the TE co-resident on GPU0 (§5.5) |
| "Where the TE lives changes speed by 2x" | **That was an artifact of the era when freeing was still happening.** Once freeing is stopped, single-GPU (TE also on GPU0) 7.08s vs two GPUs (TE@GPU1) 6.89s — a 2.7% difference (96GB box, bf16, t2i steady state). What mattered was **stopping the freeing**, not the TE's location |

---

## 5. When VRAM capacity changes — derive the table, don't memorize it

**The mode is a function of VRAM capacity.** If you swap the GPU, re-derive from the component table and inequalities below
(this very table has a history of being built on a 96GB machine and then rebuilt on a 48GB machine).

### 5.1 Component table (all measured)

| Component | Size |
|---|---|
| text_encoder bf16 | 66.71GB (53.06GB with 51 layers pruned) |
| **text_encoder nf4** | **21.02GB (17.45GB with 51 layers pruned)** |
| transformer bf16 | 66.3GB |
| **transformer int8** | **34.0GB** |
| transformer_ref bf16 / int8 | 61.7GB / ~34GB |
| vae + audio_vae (fp32) | 11.0GB |
| Denoise activations | ~5–6.6GB (measured 6.6GB at 768², 5 seconds) |
| **Decode peak** | **16.29GB** (~11.4GB with video VAE fp16). **Both predate the 2026-08-10 uint8 fix** (see the lineage note below) |
| ref2va's extra reference-encoding cost | **+3.2GB or more** on top of the TE (vision tower with 2048px short side. measured lower bound) |
| **CUDA context etc. (non-PyTorch)** | **~1GB** (easy to forget. see the pitfall below) |

> **Decode-phase de-normalization moved to the CPU (2026-08-12)**: the tail of upstream
> `MiniMaxH3VideoDecodeStep`, `(video.float() * pixel_std + pixel_mean).clamp(0,1)`, converted
> the whole-length fp16 decode result to fp32 in one shot on the GPU (an **838MiB** temporary
> at 768², 124 frames). The runner-side subclass `_cpu_norm_video_decode_step()` changes it to
> **move to CPU while still fp16, then de-normalize**, and this whole-length fp32 temporary
> disappears from the GPU (element-wise ops are bit-identical CPU vs GPU, proven by a same-seed
> PNG MD5 match). Applied unconditionally to all paths (t2va/t2i/ref2va/batch). The decode-peak
> 16.29GB / fp16 11.4GB above are the **pre-fix values**; after the fix a few whole-length fp32
> tensors come off the top. **The post-fix re-measurement has not been done**, so every derivation
> in this document still uses the pre-fix (i.e. conservative) numbers — in reality there is a
> little more headroom. This was the direct condition that let t2va fit on 8GiB×2. Details in
> addendum B6 of [docs/internal/TECHNICAL_REPORT.en.md](internal/TECHNICAL_REPORT.en.md).

> **Lineage of the decode-peak numbers (easy to mix up)**: three generations coexist in this
> document. (1) **16.29GB (fp32) / ~11.4GB (fp16)** = **before** the 2026-08-10 uint8 intermediate-
> tensor fix. The component table in §5.1, the inequalities in §5.2, and the `H3_KEEP_TRANSFORMER`
> derivation in §5.5 all use this generation. (2) **14.2GB (fp32) / 7.53GB (fp16)** = **after** the
> uint8 fix; the capacity tables in §5.3 use these. (3) After the 2026-08-12 move of
> de-normalization to the CPU = **not measured**. Every fit/no-fit derivation still uses (1) or (2),
> i.e. the conservative side, so no verdict can drift in the optimistic direction. **When you
> discuss whether a new configuration fits, always state which generation of numbers you used.**

> **The unit pitfall (an actual mistake made while building this table)**: `nvidia-smi` reports
> **MiB**, PyTorch's OOM messages report **GiB**, and this app's own logs (`gpu_mem_gb()`) report
> **GB (decimal)**. A 20GB card is 20475 MiB on `nvidia-smi` = **21.47 GB (decimal)**, but the
> capacity PyTorch actually sees is 19.55 GiB = **20.99 GB (decimal)**. That's **about a 0.5GB gap**,
> and on top of that another ~1GB is subtracted for non-PyTorch overhead. **Treating the catalog
> capacity as the budget as-is overestimates it by about 1.5GB**
> (this oversight almost led to wrongly concluding "ref2va fits even at 20GB" — it OOM'd in practice).

### 5.2 Inequalities to satisfy (independent per phase)

What needs to be resident together only needs to fit **within each phase**. There is no need to sum across phases.

```
effective budget = catalog capacity − unit gap (~0.5GB) − CUDA context etc. (~1GB)

Encode  : TE                                        ≤ effective budget
Denoise : transformer + activations (~6.6GB)         ≤ effective budget
Decode  : decode peak (16.29 / 11.4 with fp16)        ≤ effective budget
```

**If you want something to stay resident across requests, add its cost to every phase.** This is where the design branches.

- Want the TE to stay resident through denoise → `TE + transformer + activations ≤ capacity`
- Want the transformer to stay resident through decode → `transformer + decode peak ≤ capacity`

### 5.3 Modes derived from capacity (fully revised 2026-08-10 / 48GB row updated with measurements 2026-08-12)

**Three premises changed** (all measured the same day): (1) the projected TE at NF4 is
**3.11GB** resident (replacing the pruned 32B nf4's 17.45GB; caveats: no `<d>` dialogue
tags — use an audio reference instead — approximate detail; the ref2va vision path was
verified visually on 2026-08-11), (2) the decode phase is **7.53GB** (video VAE fp16 + the uint8 fix; 14.2GB
even at fp32), (3) denoise stays 34.03+6.6=**40.6GB**. The tables below are re-derived
from those. **Everything not marked "measured" is a derivation.**

**Single GPU (no 2nd card)**

| Capacity (effective) | 32B TE route | Projected TE (NF4) route |
|---|---|---|
| 96GB | bf16 TE+transformer resident (measured) | unnecessary (plenty of room) |
| 48GB (~49.8) | `H3_LOWVRAM=1`, swap every request (measured) | **everything resident at once, and it works**: derived 3.11+34.03+7.53=**44.7GB**, against a measured peak of **45.6GB** on the 96GB box (t2va 5s, §0 ②), leaving ~4.2GB. The guard change is done (below). **Not yet confirmed on real 48GB hardware** |
| 32GB (~30.5) | `group` (nf4 21 + blocks 1.4 + activations 6.6 = 29, barely) | `group` with room to spare (~11.1GB incl. TE) |
| 24GB (~22.4) | `group`+`H3_TE_PRUNE=1` required (measured, the old floor) | `group` with room to spare (~11.1GB) |
| **16GB (~15.2)** | **not possible** (the 17.45GB TE doesn't fit) | **Measured (2026-08-11)**: a real RTX 4060 Ti 16GB **alone** runs t2i/t2va/ref2i/i2va/audio-reference/768×1344, all to completion. Peak 7.4–15.2GB (t2va denoise 11.4GB, i2va 9.41GB, audio-reference 11.96GB, 768×1344 measured 15.2GB by nvidia). Opens up the previously-impossible 16GB tier |

> **Single-16GB launch (measured 2026-08-11)**: `H3_LOWVRAM=group H3_TE_PROJ=... H3_VIDEO_VAE_FP16=1
> H3_ATTN_BACKEND=default` (sage is an sm_120-only build, so fall back to SDPA. no TE_DEVICE =
> projected TE co-resident). The derived 11.1GB matched the measured t2va denoise 11.4GB
> closely. The reference family (i2va/audio-reference/768×1344) all runs on the same card too,
> with 768×1344/5s (measured peak 15.2GB) the practical ceiling. **Speed is 16.5–51s/step
> because the 96GB box's 2nd slot is Gen3 x4** (a value of the slot, not the card; a Gen4 x16 slot
> would transfer ~1/8). **FBC skips 0 steps on the int8+SDPA trajectory** (threshold tuning
> unverified).

> **48GB single-GPU (the guard change is done, 2026-08-11/12)**: `H3_KEEP_TRANSFORMER`'s
> import-time guard used to require `H3_TE_DEVICE` (it was designed around the 32B TE). It was
> relaxed to **accept `H3_TE_PROJ` as well**, so the projected TE can be housed **on the same GPU**
> while the transformer stays resident (3.11 + 34.03 = 37.1GB; even with 6.6GB of denoise
> activations that is 43.7GB, inside the budget). Condition 1 was relaxed too — now only
> "`H3_LOWVRAM` is not `group`" — so it works in plain mode as well (§5.5).

**With a 2nd GPU (`H3_TE_DEVICE` parks the TE off-card)**

| Main | 2nd GPU requirement | What works |
|---|---|---|
| 48GB | 20GB (32B pruned, 17.76 measured) | **measured on the 48GB box**: t2i **9.7s** / t2va 5s **44.2s** (turbo 4steps). ref2va needs a 24GB 2nd card (see also the callout in §3) |
| 48GB | **8GB-class** (projected NF4, 3.11) | the same residency on a much cheaper card (derived) |
| 32GB | 20GB (32B) / 8GB-class (projected NF4) | still `group`, just with more headroom. **Non-group stays impossible even with a 2nd card** (denoise 40.6 > 30.5 is the main GPU's problem) |
| **8GiB×2** | **8GiB (projected NF4 3.11+ε)** | **Measured (2026-08-11, ballast-simulated)**: compute-side 8GiB + TE-side 8GiB runs t2i/t2va 5s 768²/ref2i to completion (t2i peak 6.4GB, t2va peak **7.23GB**, ref2i peak 6.69GB). **The video reference family (i2va/audio-reference) does not fit**: reference tokens lengthen the sequence, so even the shortest 768²/5s has no room for denoise activations (i2va requires a measured 9.41GB) |
| 24/16GB | **6–8GB-class** (projected NF4) | the main GPU runs `group` alone (blocks + activations ~8GB); parking the TE elsewhere lowers the main-GPU floor further |

**The 8GiB×2 boundary (measured 2026-08-11)**: still-image references (ref2i) work, but the
**video reference family does not fit because of the reference-token VRAM addition**. i2va
requires 9.41GB against t2va's 7.23GB (+2.18GB from references), and audio-reference and
768×1344 also exceed the 8GiB budget. These run only on the single 16GB card.

ref2va needs care at every tier: with the 32B TE, reference encoding adds **≥3.22GB
(measured)** on the TE's GPU. The projected TE's vision path was **verified visually on
2026-08-11** (face/hair/clothing/props all reflected, on par with the 32B reference; after
fixing the two latent bugs).

### 5.4 Putting the TE on a separate GPU (`H3_TE_DEVICE`)

The capacity needed for the TE's GPU depends on the use case.

The current 20GB card's **effective budget is about 19.7GB** (catalog 21.47GB − unit gap 0.5 − non-PyTorch ~1).

| TE | Required | Card that fits |
|---|---|---|
| 32B pruned nf4 / t2va-family | 17.76GB (measured) | **20GB+** (margin ~1.9GB) |
| 32B pruned nf4 / ref2va | **20.67GB+** (measured: TE 17.45 + reference encoding ≥3.22) | **24GB+** (20GB OOMs, short by 204MB; multiple references not guaranteed even at 24GB) |
| Projected 4B bf16 | 8.88GB (measured) + ε | 12GB (thin) / **16GB+** |
| **Projected 4B NF4** | **3.11GB (measured) + ε** | **measured at 8GiB-class (2026-08-11)**: resident on the TE side of an 8GiB×2 setup (t2i/t2va/ref2i complete). 6GB-class is derived. The ref2va vision path was verified visually |

**→ For ref2va with the 32B TE you need an effective 20.7GB or more, i.e. a catalog capacity of 22.2GB or more.** A 24GB card
(effective ~22.4GB) is expected to fit with a margin of ~1.7GB. This is the concrete basis for swapping
to an RTX PRO 4000 Blackwell 24GB (**capacity alone is the reason** — PCIe width or generation are irrelevant, see §3).

However, a 1.7GB margin is thin, and requiring 2 or more reference images increases the demand further
(2 images OOM'd even in testing). **Even at 24GB, multiple references for ref2va with the 32B TE cannot be guaranteed to work.**

**This capacity arithmetic does not apply to the projected TE** (3.11GB + vision activations), yet
the guard in the code rejects anything under 24GB regardless. **Needs revisiting, not yet fixed**
(see the callout in §3).

### 5.5 `H3_KEEP_TRANSFORMER=1` — decoding while the transformer stays resident

A configuration where **the transformer stays resident through decode**. **This is the single most
effective flag in the repository**: under `H3_LOWVRAM=1` it removes the per-request transformer load
(14.8–32.7s), and in plain mode it removes the per-decode-window free + reload
(**11.9–12.3s per request**). Per the inequality in 5.3, this doesn't fit with fp32 decode, but it
does fit with `H3_VIDEO_VAE_FP16=1`:

```
transformer 34.3 + decode peak 16.29 = 50.6GB  > 48GB   ← doesn't fit (fp32 VAE)
transformer 34.3 + 11.4 (H3_VIDEO_VAE_FP16=1)  = 45.7GB  < 48GB  ← derived prediction
```

Implemented in `core/runner.py` as `H3_KEEP_TRANSFORMER=1`. The conditions for it to be valid are enforced
by an import-time guard (all three are required; missing any raises `RuntimeError`. **Conditions 1 and 2
were relaxed in two rounds, on 2026-08-11 and 2026-08-12** — beware of older write-ups of "the three
conditions"):

1. **`H3_LOWVRAM` must not be `group`** (`1` is fine, and so is `0`/plain — relaxed 2026-08-12).
   Only `group` is out of scope: it keeps the transformer resident on the CPU and shuttles it
   block by block, a separate design that is simply unrelated
2. **Either `H3_TE_DEVICE` or `H3_TE_PROJ`** (relaxed 2026-08-11). What the condition really
   demands is "**do not co-locate the 32B TE with the compute GPU**": doing so breaks the
   **encode** phase before decode ever becomes an issue (TE-nf4 17.45GB + resident
   transformer-int8 34.3GB = 51.75GB > the ~49.8GB effective budget). The projected TE
   (`H3_TE_PROJ`) is only 3.11GB at NF4, so co-locating it gives 3.11+34.03 = 37.1GB, or 43.7GB
   with 6.6GB of denoise activations — inside the budget. There is therefore no reason to demand
   a second GPU, and the guard exempts it
3. `H3_VIDEO_VAE_FP16=1` (with fp32 it doesn't fit at 50.6GB, as shown above)

The default (`H3_KEEP_TRANSFORMER=0`) leaves behavior unchanged.

**This flag only touches `transformer`** (the t2va/fl2va/t2i side); it has **nothing to do with
`transformer_ref`** (the ref2va/ref2i side). In the code, the skipped frees are the four sites in
`generate()` and `generate_still_batch()` — the "entry" and the "decode window" of each — and all
four only skip `self._free_transformer()`. `self._free_transformer_ref()` is still called
**unconditionally** at those entries (entering a t2va request always drops the ref side).
What decides whether `transformer_ref` stays resident is a different flag,
**`H3_TRANSFORMER_BOTH_RESIDENT`** (automatically true when `H3_TRANSFORMER_QUANT=int8`): under
plain + int8, ref2va restores the t2va `transformer` when it finishes (**~13 seconds**, measured),
so **both transformers end up resident** and the peak jumps (measured 74.3GB on the 96GB box, and
77.3GB as the peak once you switch back to t2va). **If you run on a single 48GB card, keep the
t2va family and the reference family in separate processes.**

#### Equivalence and effect in plain mode (`H3_LOWVRAM=0`) (96GB box, measured 2026-08-12)

Relaxing condition 1 made this applicable to plain mode. **No additional implementation was
needed** — the branch that skips the free was already shared, and the restoring
`_ensure_transformer` is idempotent, so it becomes a no-op.

- **Same-seed PNGs match by MD5** (`596a718e4b5cf9a0b907d2ec479225d2`). Stopping the free is
  mathematically inert, and the identical image comes out in **19.58s → 7.40s (2.6x)**
  (int8, single GPU, t2i 768², turbo)
- The peak for that configuration goes 39.8GB (KEEP=0) → **45.6GB** (KEEP=1): the transformer now
  also occupies the decode phase (**the per-phase breakdown has not been measured**)

#### Measurements by configuration (96GB box, turbo 4steps, 768², steady state)

| Configuration | t2i | t2va 5s | Peak | GPUs |
|---|---|---|---|---|
| bf16 + TE@GPU1 + freeing stopped | **6.89s** | **26.8s** | 74.2GB + 3.2GB | 2 |
| bf16 single GPU (TE also on GPU0) + freeing stopped | 7.08s | 27.04s | 77.3GB | 1 |
| **int8 single GPU + freeing stopped (practical optimum)** | **7.40s** | **28.13s** | **45.6GB** | 1 |
| int8 + `H3_LOWVRAM=1` + KEEP + TE@GPU1 | 7.65s | 28.56s | 42.5GB | 2 |
| int8 single GPU, freeing enabled (KEEP=0) | 19.58s | — | 39.8GB | 1 |
| bf16 + TE@GPU1, freeing enabled | 19.9s | 40.0s | 68.9GB | 2 |

bf16 denoises faster than int8 (t2i 2.05–2.07s vs 2.39–2.40s, t2va 14.05s vs 14.81s), but it
**needs a 77GB-class card**. Whether that beats int8 at 45.6GB and 7% slower depends on the card
you own.

#### Per-mode results (96GB box, int8 single GPU + freeing stopped, turbo 4steps, 768², steady state)

| Mode | Single request | Per item in a batch | Denoise | Peak |
|---|---|---|---|---|
| t2i 768² | 7.40s | 7.9s (0.94x = no benefit) | 2.40s | 45.0GB |
| t2va 5s | 28.13s | (no batch API) | 14.94s | 45.6GB |
| ref2i 768² | 79.3s | 47.0s (1.69x) | 7.8s | 45.4GB |
| i2va 5s | 103.1s | 75.0s (1.37x) | 22.0s | 45.9GB |

The 30-step versions (no turbo): t2i 27.42s / t2va 155.0s (no FBC), 21.41s / 121.5s (with FBC),
ref2i 148.4s / i2va 290.3s.

#### First-request costs (not included in the steady-state numbers)

Every number above is a **steady-state** value (second request onwards). Once per process, the
following one-time costs apply:

- **~50 seconds at startup** to make the transformer + VAE resident
- **The reference family additionally cold-loads `transformer_ref` on its first request (+55
  seconds)**. `transformer_ref` is loaded on the first reference request, not at startup, so
  ref2i takes 134.7s the first time and 79.3s in steady state

#### Breakdown of i2va's 103.1s (96GB box, from log timestamps)

| Phase | Time | Notes |
|---|---|---|
| Reference vision encoding | **~47s** | **the dominant cost**. Does not shrink with the 4B projected TE |
| Denoise (4 steps) | 22.0s | |
| Decode + VAE round trip | ~10s | |
| Reference VAE encoding | ~6s | |
| Reloading the t2va transformer at the tail | ~13s | pure waste for back-to-back ref2va (**not yet fixed**) |

#### Real-hardware E2E measurements on the 48GB box (2026-08-09, the `H3_LOWVRAM=1` route)

The following was measured on the 48GB GPU0 + 20GB GPU1 box with `H3_LOWVRAM=1 H3_TE_PRUNE=1
H3_TE_DEVICE=cuda:1 H3_VIDEO_VAE_FP16=1 H3_KEEP_TRANSFORMER=1` (i.e. the pruned 32B TE parked on
the second GPU). It is **no longer the fastest configuration**, but it is the only measurement
taken on real 48GB-class hardware, so it is kept as-is:

- The transformer (int8) loads exactly once, on the first request (32.0s). No reload on subsequent
  requests (confirmed in server logs)
- t2i turbo 4steps steady state: **9.7s/image** (denoise 4.32s, decode 1.5s). peak VRAM 41.97GB
  (during denoise)
- t2i steps=30 steady state: 51.1s (denoise 45.7s). peak 41.97GB
- t2va 5-second turbo 4steps: **44.2s** (denoise 26.05s, decode 10.81s). **peak VRAM 44.15GB =
  decode phase** (transformer 34.03GB resident + fp16 decode). against the derived prediction of
  45.7GB above, measured 44.15GB, leaving a margin of ~4.8GB against the 48.9GB catalog capacity
- nvidia-smi measured peak 42,620 MiB (1-second sampling. the instantaneous peak of 44.15GB from
  torch's own measurement is the correct one)
- Same-seed output equivalence: PNG MD5 is a **perfect match** against the flag-OFF baseline
  (otherwise identical conditions) (seed=11, md5 `665eadddea8f34298a1b5b89e69d4bd0`). The baseline
  side was total 63.27s (including transformer load) / peak 36.4GB
- Lineage of speedups: t2i turbo 157s (morning of 08-07) → 83.2s (`H3_TE_PREQUANT`) → ~35s
  (`H3_TE_DEVICE`) → **9.7s** (`H3_KEEP_TRANSFORMER`). t2va 5s (768²):
  no turbo, 30 steps **351.4s** (plain 48GB configuration) → turbo 143s → 60.5s → **44.2s** (8.0x)

### 5.6 Checking the derivation formula (matches measurements in every case)

Running the known measurements through the formula above, the fit/no-fit verdict **matched the
measurements in every single case** (the derived numbers themselves are off by up to ~1GB — usually
on the conservative side, but sometimes the measurement exceeds the derivation — by 0.9GB in the
plain+KEEP row below. **Good enough for a fit/no-fit verdict, but never declare a configuration with under 1GB
of margin without measuring it.**)

| Case | Required / effective budget | Verdict | Measured |
|---|---|---|---|
| TE GPU 20GB, t2va | 17.76 / 19.97 | OK | ○ fits |
| TE GPU 20GB, ref2va | 20.67 / 19.97 | NG | ○ OOM |
| TE GPU 24GB, ref2va | 20.67 / 24.27 | OK | (not yet measured, expected) |
| GPU0 48GB, denoise | 40.60 / 49.81 | OK | ○ peak 40.89 |
| GPU0 48GB, denoise with TE resident | 58.05 / 49.81 | NG | ○ this is why we swap every time |
| GPU0 48GB, decode with transformer resident | 50.29 / 49.81 | NG | (never implemented for fp32 VAE) |
| Same, + video VAE fp16 (`H3_KEEP_TRANSFORMER=1`) | 45.70 predicted / 49.81 | OK | ○ measured 44.15GB (§5.5, 2026-08-09, 48GB box) |
| Single GPU, projected TE co-resident, transformer resident too (plain + KEEP) | 44.7 predicted / 49.81 | OK | ○ measured 45.6GB (§5.5, 2026-08-12, 96GB box. **not confirmed on real 48GB hardware**) |
| t2va family and reference family mixed in one process | 74.3 measured / 49.81 | NG | ○ both transformers resident (96GB box; 77.3GB when switching back to t2va) |
| Single 16GB (projected TE NF4 co-resident, group), t2va denoise | 11.1 predicted / ~15.2 | OK | ○ measured 11.4GB (2026-08-11) |
| 8GiB×2 (projected TE off-card), t2va | 7.23 / ~7.1 | OK (barely) | ○ measured 7.23GB (2026-08-11) |
| 8GiB×2, i2va (with references) | 9.41 / ~7.1 | NG | ○ OOM (+2.18GB from reference tokens) |

**VRAM addition from reference tokens (measured 2026-08-11, single 16GB)**: against t2va (no
references) 7.23GB — i2va (image reference) 9.41GB, audio-reference 11.96GB, 768×1344/5s
13.37GB (nvidia measured 15.2GB). The reference family lengthens the sequence by the reference
tokens, increasing denoise activations. On 8GiB×2 the ceiling is still-image references
(ref2i peak 6.69GB); the video reference family exceeds the budget from this addition.

**If you swap the GPU, plug the new capacity into this formula and re-derive.** No need to re-memorize the table.

## 6. How to check while running

```bash
# what the runner currently believes
curl -s http://127.0.0.1:8611/api/status | python3 -m json.tool | grep -E "loaded|on_gpu|peak|allocated"

# actual GPU usage (per process)
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv

# load/free history (this is the most reliable)
grep -E "transformer (loaded|freed)|text_encoder|vae/audio_vae ->" logs/server.log | tail -20
```

Every log line carries `gpu={'allocated_gb':..., 'peak_gb':...}`, so **read that instead of guessing**. The table in §0 was also built from these logs.

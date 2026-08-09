# What is loaded and freed, and when — residency reference

[日本語](RESIDENCY.md) | **English**

The combinations of mode, quantization, turbo, and TE placement have grown, so this document lets you look up **"what is on the GPU, in this phase, for this configuration"** on a single page. Everything is derived from the `core/runner.py` code and cross-checked against measured logs.

> **Why this document exists**: on 2026-08-09, the peak VRAM breakdown was explained as "transformer + VAE during decode", but that was **wrong** (in fact the transformer is freed before decode). There are too many combinations to answer from memory without making a mistake — this is a real example of that. Check here from now on.

---

## 0. Bottom line first — current configuration (this box)

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

**Important**: denoise and decode **never overlap in time**. The transformer is always freed immediately before decode (`generate()` lines 3792-3793). The 40.9GB peak is from **denoise**.

---

## 1. What's "resident" and what's "per-request" — three axes

The decision is governed by three independent axes. The source of confusion is that these are orthogonal.

| Axis | Env var | Effect |
|---|---|---|
| **A. How much stays resident** | `H3_LOWVRAM` (0/1/group) | Most important. Determines the table in §2 below |
| **B. Which GPU the TE lives on** | `H3_TE_DEVICE` | Once set, the TE is **never freed at all** (stays resident on the other GPU) |
| **C. Whether loading is faster** | `H3_TE_PREQUANT` | Does not change **when** it loads. Only makes **the load itself** faster |

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

### `H3_LOWVRAM=1` (48GB-class) ← **this box**

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

**When `H3_KEEP_TRANSFORMER=1` is layered on top** (`H3_TE_DEVICE` must also be set, details in §5.5): the "(transformer freed)" row and the "Decode" row in the table above change — the transformer is **no longer freed at the decode phase either**, and it **stays resident across requests too** (the "Entry" freeing step no longer happens either). It stays resident from the single initial load only, so the transformer's load fixed cost (14.8–32.7s) collapses to a one-time cost paid only on the first request. The decode-phase peak becomes transformer 34.3GB resident + fp16 decode ~11.4GB (`H3_VIDEO_VAE_FP16=1` required). See §5.5 for measurements.

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

---

## 3. Additional guarantees when `H3_TE_DEVICE` is set

- The TE is **not freed even when `_free_text_encoder()` is called** (even with `force=True`) — not freeing it is the whole point
- The TE is **normally detached from the pipe**. It's only connected during the `_te_attached()` window (while encoding)
  - Reason: `_execution_device` returns the device of the first nn.Module found in components order, so if it stayed attached, layout and **even decode** would end up building tensors on the wrong GPU and crash (reproduced on real hardware)
- During the layout/latents/timesteps window, `_pin_execution_device_to_compute()` **also temporarily detaches the vae** (otherwise, once the TE is detached, the next thing found would be the vae sitting on CPU, and `cpu` would be returned instead)
- **ref2va is rejected with a 400 if the TE's GPU has less than 24GB** (OOM measured at 20GB)

---

## 4. Common misconceptions

| Misconception | Reality |
|---|---|
| "The 40.9GB peak is transformer + decode VAE" | **Wrong.** The peak is during denoise (transformer 34.3 + activations 6.6). By decode time the transformer is already freed, leaving just VAE 11.3GB |
| "`H3_TE_PREQUANT` makes the TE stay resident" | **Wrong.** It doesn't change when it loads. **It only makes the load faster** (53s→29.5s) |
| "turbo increases/decreases VRAM" | Barely changes at all. LoRA is just a wrapper, only reading 1.38GB on first use |
| "`H3_LOWVRAM=group` is always more memory-efficient but slower than `1`" | It is more memory-efficient, but **for still images `group` is actually slower** (block transfer is a fixed cost, and for very short/lightweight compute, transfer dominates) |
| "Putting the TE on a separate GPU makes the fixed cost zero" | Only the TE's share. **The transformer load (14.8–32.7s) remains** (because it's still freed before decode). However, layering `H3_KEEP_TRANSFORMER=1` on top collapses this too, down to a one-time cost (§5.5) |

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
| **Decode peak** | **16.29GB** (~11.4GB with video VAE fp16) |
| ref2va's extra reference-encoding cost | **+3.2GB or more** on top of the TE (vision tower with 2048px short side. measured lower bound) |
| **CUDA context etc. (non-PyTorch)** | **~1GB** (easy to forget. see the pitfall below) |

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

### 5.3 Modes derived from capacity

| Capacity | Configuration that works | Bottleneck (why you can't go further) |
|---|---|---|
| **96GB** | bf16 as-is, TE+transformer resident (87.5GB) | at decode time all three at once (98.5GB) don't fit, so the transformer has to give way |
| **80GB** | int8, TE+transformer+transformer_ref resident (89GB) | same as above |
| **48GB** | `H3_LOWVRAM=1`. TE(17.45)+transformer(34)+activations(6.6) = **58GB, doesn't fit** → swap every time | **having nowhere to park the TE** is the cause of the fixed cost |
| **32GB** | `H3_LOWVRAM=group`. transformer resident in RAM, streamed block by block | TE(21)+decode(16.3)=37GB > 32GB, so TE gives way at decode time |
| **24GB** | `group` + `H3_TE_PRUNE=1` | TE-nf4 17.45GB is most of the budget. Won't fit without pruning |
| **18GB** | same as above (measured floor) | **the pruned TE-nf4's 17.45GB itself is the floor** |
| 16GB | **not possible** | OOMs near the end of the TE load. Breaking through this would require streaming execution of the TE |

### 5.4 Putting the TE on a separate GPU (`H3_TE_DEVICE`)

The capacity needed for the TE's GPU depends on the use case.

The current 20GB card's **effective budget is about 19.7GB** (catalog 21.47GB − unit gap 0.5 − non-PyTorch ~1).

| Use case | Required (measured) | Fits on 20GB? |
|---|---|---|
| t2va / fl2va / t2i | 17.76GB | **Fits** (margin ~1.9GB) |
| ref2va | **20.67GB or more** (TE 17.45 + reference encoding 3.22 or more) | **Doesn't fit** (OOM, short by 204MB) |

**→ ref2va needs an effective 20.7GB or more, i.e. a catalog capacity of 22.2GB or more.** A 24GB card
(effective ~22.4GB) is expected to fit with a margin of ~1.7GB. This is the concrete basis for swapping
to an RTX PRO 4000 Blackwell 24GB (**capacity alone is the reason** — PCIe width or generation are irrelevant, see §3).

However, a 1.7GB margin is thin, and requiring 2 or more reference images increases the demand further
(2 images OOM'd even in testing). **Even at 24GB, multiple references for ref2va cannot be guaranteed to work.**

### 5.5 `H3_KEEP_TRANSFORMER=1` — decoding while the transformer stays resident (measured and working, 2026-08-09)

A configuration where **the transformer stays resident through decode**. The fixed cost (transformer load
14.8–32.7s, incurred every request under `H3_LOWVRAM=1`) disappears. Per the inequality in 5.3, this doesn't
fit with fp32 decode, but it does fit with `H3_VIDEO_VAE_FP16=1`:

```
transformer 34.3 + decode peak 16.29 = 50.6GB  > 48GB   ← doesn't fit (fp32 VAE)
transformer 34.3 + 11.4 (H3_VIDEO_VAE_FP16=1)  = 45.7GB  < 48GB  ← derived prediction
```

Implemented in `core/runner.py` as `H3_KEEP_TRANSFORMER=1`. The conditions for it to be valid are enforced
by an import-time guard (all three are required; missing any raises `RuntimeError`):

1. `H3_LOWVRAM=1` (raw `"1"` only. `group` is out of scope — irrelevant since it's a separate design)
2. `H3_TE_DEVICE` set (TE on a separate GPU. **Otherwise the encode phase breaks first**:
   TE-nf4 17.45GB + resident transformer-int8 34.3GB = 51.75GB > effective budget)
3. `H3_VIDEO_VAE_FP16=1` (with fp32 it doesn't fit at 50.6GB, as shown above)

The default (`H3_KEEP_TRANSFORMER=0`) leaves behavior unchanged. `transformer_ref` (the ref2va path) is out
of scope and continues to be freed every time as before.

**Real-hardware E2E measurements (2026-08-09, 48GB GPU0 + 20GB GPU1, `H3_LOWVRAM=1 H3_TE_PRUNE=1
H3_TE_DEVICE=cuda:1 H3_VIDEO_VAE_FP16=1 H3_KEEP_TRANSFORMER=1`)**:

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
measurements in every single case**.

| Case | Required / effective budget | Verdict | Measured |
|---|---|---|---|
| TE GPU 20GB, t2va | 17.76 / 19.97 | OK | ○ fits |
| TE GPU 20GB, ref2va | 20.67 / 19.97 | NG | ○ OOM |
| TE GPU 24GB, ref2va | 20.67 / 24.27 | OK | (not yet measured, expected) |
| GPU0 48GB, denoise | 40.60 / 49.81 | OK | ○ peak 40.89 |
| GPU0 48GB, denoise with TE resident | 58.05 / 49.81 | NG | ○ this is why we swap every time |
| GPU0 48GB, decode with transformer resident | 50.29 / 49.81 | NG | (never implemented for fp32 VAE) |
| Same, + video VAE fp16 (`H3_KEEP_TRANSFORMER=1`) | 45.70 predicted / 49.81 | OK | ○ measured 44.15GB (§5.5, 2026-08-09) |

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

"""
MiniMax-H3 T2VA/FL2VA runner.

Loading strategy (see dev_notes/handoff-minimax-h3.md and diffusers-server CLAUDE.md
#33/#46/#47 for the constraints this follows):

- This box has 96GB VRAM but only ~94GB host RAM. The big components add up to ~144GB
  (text_encoder bf16-native ~66.7GB -- measured on GPU, the checkpoint shards are
  already bf16 -- transformer bf16 ~66.3GB, vae+audio_vae fp32 ~11GB), which fits in
  neither VRAM nor host RAM at once. `ComponentsManager.enable_auto_cpu_offload()`
  keeps every component CPU-resident as its steady state (accelerate hooks only move
  the *active* one to GPU), so it would try to hold all ~144GB in RAM simultaneously --
  not possible here.

There are two loading strategies, selected by the `H3_TE_QUANT` env var:

`H3_TE_QUANT=none`: the two 66GB models cycle through GPU per request, with
  the small fp32 VAEs (~11GB) permanently resident:
    encode phase : [vae 11GB + text_encoder 66GB]   (transformer dropped if resident)
    denoise/decode: [vae 11GB + transformer 66GB]   (TE dropped right after encoding)
  Each drop frees the CUDA model in place (no .to("cpu") staging -- that would take
  ~30s, evict page cache and push the box into swap, observed on the first probe run).
  Reloads are served from disk/page cache at ~16-40s per model, i.e. ~1 load/free cycle
  per generation for each big model -- the "short window" pattern CLAUDE.md sanctions,
  not the banned "swap the whole module every step" pattern. The steady state between
  requests keeps transformer + VAEs resident (77GB). Overhead: ~37s TE reload +
  ~26s transformer reload per request.

`H3_TE_QUANT=bnb-4bit` (default; A/B verified 2026-08-04 -- same-seed frames and audio
  show no visible/audible degradation vs bf16 TE, and requests drop 245s -> 185s):
  the text_encoder is quantized to NF4 (bitsandbytes,
  compute_dtype=bf16) at startup and stays GPU-resident permanently -- bnb 4bit models
  cannot be moved between devices, so "load once, keep forever" is the only option for
  them anyway. Measured size: ~21.0GB (not the ~18GB originally estimated). The
  transformer (66.3GB) also stays resident between requests: no more per-request TE<->
  transformer swap. That leaves transformer+TE-nf4 = ~87.5GB resident during encode/
  denoise, which does not leave enough headroom for vae+audio_vae(11GB, permanently
  resident in the `none` path) plus activation buffers within this card's ~95.6GB. So
  in this mode the VAEs are NOT permanently resident: they live on CPU by default and
  are moved to GPU only for their active phase (keyframe encode / video+audio decode),
  then moved back to CPU right after.
  A second, sharper constraint was found by measurement, not by the original estimate:
  transformer(66.3) + TE-nf4(21.0) + vae pair(11.0) = ~98.5GB *before* any decode
  activation buffer is even counted -- already over the card's ~95.6GB. Keeping all
  three resident through decode OOM'd in practice ("Tried to allocate 30.00 MiB" with
  the allocator already pinned at 93.7GB). Since the transformer is not touched by
  either decode step (MiniMaxH3VideoDecodeStep / MiniMaxH3AudioDecodeStep only use
  vae/audio_vae/video_processor), it is dropped for the ~9s decode window and reloaded
  right after, restoring the transformer+TE-nf4 steady state before the next request.
  None of this is the banned "swap a 60GB+ module every step" pattern (CLAUDE.md #33):
  every move is a single one-way trip bounded to one specific phase (keyframe encode,
  decode, or the reload right after), the same "short window, small object" shape the
  `none` path already uses for its own TE/transformer cycle -- just sliced along a
  different phase boundary (decode instead of encode) and applied to the VAEs plus,
  when decode is the phase in question, the transformer too.
  Overhead avoided: no more per-request TE reload (was ~37s) and no more per-request
  transformer reload *around encode* (was ~26s). Overhead added: ~1 VAE round trip
  in/out of GPU per request (small, fp32, ~11GB, PCIe-bound, no disk I/O) plus one
  transformer drop+reload around the decode window specifically (~10-26s, page-cache
  warm) -- still net faster per request since the TE load is fully eliminated and it
  replaces what used to be *two* full big-model reloads with one.

- video VAE decode runs under a float16 autocast internally (diffusers' own
  MiniMaxH3VideoDecodeStep) even though its weights are float32. audio_vae must stay
  float32 end-to-end: casting it to bf16 is a known upstream bug that makes generated
  audio ~20dB too quiet, so we never touch its dtype after loading fp32.
- The video VAE ships with spatial tiling enabled by default (`use_tiling=True`,
  256px tiles, verified in autoencoder_kl_minimax_h3.py) and runner.py never disables
  it, so tiled decode is already active in both modes -- there is no extra "enable
  tiling" step needed for decode-peak reduction here.

`H3_LOWVRAM=1` (opt-in, default "0" leaves every mode above byte-for-byte unchanged):
  a third loading strategy, orthogonal to `H3_TE_QUANT`/`H3_TRANSFORMER_QUANT` (it
  forces TE_QUANT=bnb-4bit's VAE-parks-on-CPU behaviour and requires
  H3_TRANSFORMER_QUANT=int8, see H3_LOWVRAM's own module-level comment), for
  48GB-class cards where TE-nf4 (21GB) + transformer-int8 (34GB) = 55GB already does
  not fit together. Steady state between requests is "nothing big resident" (only the
  small VAE pair, parked on CPU). Phase x resident-set table for a t2va request
  (`generate()`'s lowvram branch):

    entry         : [nothing big -- any resident transformer/transformer_ref is freed]
    encode        : [TE-nf4 21GB]                      (transformer freed if resident)
    (TE freed)
    denoise       : [transformer-int8 34GB + ~5GB activations ~= 39GB]  (TE freed)
    (transformer freed)
    decode        : [vae pair ~11GB + decode buffers]  (transformer freed, TE freed)
    (vae parked back on CPU; nothing reloaded "for next time")

  ref2va is the same shape with an extra reference-VAE-encode phase between text-encode
  and denoise (needs `vae` on GPU while TE is *already* freed -- see
  `generate_ref2va()`'s lowvram branch for the `_execution_device` resolution note this
  requires, same "freeing TE makes `vae` the next resolved module, so bring vae onto
  GPU either before or in the same breath as freeing TE" pattern `force_free_te`
  already established for bnb-4bit/int8-both-resident mode above) and denoises against
  `transformer_ref` instead of `transformer`.

  This pays TE-load (~15-40s) + transformer-load (~35-40s, torchao int8 quantization
  happens inline during this load) on *every* request -- there is no cross-request
  steady state to amortize against, unlike every mode above. See README.md for the
  measured per-phase timing breakdown and the peak-VRAM verification against a VRAM
  ballast.
"""
from __future__ import annotations

import gc
import io
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# Must be set before `import torch` (PyTorch reads it once, at CUDA-allocator init
# time). Reproduced by this task's own verification: in H3_TRANSFORMER_QUANT=int8 +
# H3_TRANSFORMER_BOTH_RESIDENT mode, repeated int8 transformer/transformer_ref
# load+free cycles (the decode-window drop/reload pattern used throughout this file)
# left the allocator holding ~37GB reserved-but-unallocated in odd-sized fragments --
# a *second* ref2va request's post-decode `transformer` reload then failed inside
# `from_pretrained`'s `_caching_allocator_warmup` ("Tried to allocate 15.43 GiB" with
# only 54.44GB actually allocated out of 92.55GB in use), even though the *total*
# resident budget (transformer_ref 34 + TE-nf4 21 + transformer 34 = 89GB) was well
# within this card's ~95.6GB -- a fragmentation failure, not an over-budget one.
# `expandable_segments:True` lets the allocator grow/shrink a single virtual-address
# reservation instead of caching many fixed-size blocks, which is the fix PyTorch's own
# OOM error message suggests for exactly this "reserved but unallocated memory is
# large" symptom. This card's ~95.6GB-vs-89GB steady-state headroom is tight enough
# (see H3_TRANSFORMER_BOTH_RESIDENT's module-level comment) that this project needs it
# unconditionally now, not just as an opt-in workaround -- so it is set here rather
# than left for the operator to export before launching uvicorn (bf16/none mode is
# unaffected either way: it never has this file's tightest headroom margins).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image

# Re-exported so callers (app.py) can do `from core.runner import MiniMaxH3Reference`
# without reaching into diffusers' modular_pipelines package themselves. Cheap import
# (no model loading, just dataclass/PyAV/numpy/torch glue) -- safe at module level,
# unlike the actual big-model loading calls in this file, which all stay lazy/on-demand.
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference

logger = logging.getLogger("minimax_h3.runner")

MODEL_ID = "MiniMaxAI/MiniMax-H3"
DEVICE = torch.device("cuda:0")
CPU = torch.device("cpu")

# "none" (default) = current per-request TE<->transformer GPU swap.
# "bnb-4bit" = TE quantized NF4, TE+transformer both resident permanently, VAEs cycle
# through GPU per-phase instead. See module docstring above.
TE_QUANT = os.environ.get("H3_TE_QUANT", "bnb-4bit").strip().lower()
if TE_QUANT not in ("none", "bnb-4bit"):
    raise ValueError(f"H3_TE_QUANT must be 'none' or 'bnb-4bit', got {TE_QUANT!r}")

# "fbc" (default; A/B verified 2026-08-04: threshold 0.05 gives -25% denoise time with
# near-identical output -- PSNR 31.8-34.3dB vs no-cache, audio corr 0.979, no visible drift.
# threshold 0.1 reaches 1.92x but composition drifts visibly; not recommended as default).
# "none" = no caching, byte-for-byte identical to pre-FBC behaviour (enable_cache
# is never called). "fbc" = FirstBlockCache (see diffusers/hooks/first_block_cache.py):
# skips the remaining transformer blocks on a denoise step when the first block's residual
# is close enough to the previous step's, reusing the cached tail-block residual instead.
H3_CACHE = os.environ.get("H3_CACHE", "fbc").strip().lower()
if H3_CACHE not in ("none", "fbc"):
    raise ValueError(f"H3_CACHE must be 'none' or 'fbc', got {H3_CACHE!r}")
H3_CACHE_THRESHOLD = float(os.environ.get("H3_CACHE_THRESHOLD", "0.05"))

# EXPERIMENTAL, opt-in, not yet A/B'd against the committed default at task-write time
# (this env var and its wiring are themselves the subject of that pending A/B -- see
# dev_notes/ or the task that added this comment). "none" (default) = transformer stays
# bf16, byte-for-byte identical to pre-int8 behaviour (quantize_ is never called).
# "int8" = the transformer is weight-only int8-quantized in place via torchao
# (Int8WeightOnlyConfig(version=2), diffusers' TorchAoConfig plumbing) right after its
# bf16 load, using the modules_to_not_convert list from the upstream PR's documented
# recipe (small projection/embedding/norm layers that are numerically sensitive or tiny
# enough that quantizing them buys no memory and risks more error than it is worth).
# Only the transformer is affected; transformer_ref (ref2va) and the text_encoder
# (H3_TE_QUANT, already bnb-4bit nf4 by default) are untouched by this flag.
H3_TRANSFORMER_QUANT = os.environ.get("H3_TRANSFORMER_QUANT", "none").strip().lower()
if H3_TRANSFORMER_QUANT not in ("none", "int8"):
    raise ValueError(f"H3_TRANSFORMER_QUANT must be 'none' or 'int8', got {H3_TRANSFORMER_QUANT!r}")

# Upstream PR #14355's documented int8 recipe for the MiniMax-H3 transformer: skip
# quantizing these modules (small, and/or numerically sensitive input/output
# projections rather than the bulk attention/MLP weight that dominates the 66GB).
# Applied identically to `transformer` and `transformer_ref` -- both are the exact same
# `MiniMaxH3Transformer3DModel` class/config (see `_enable_fbc_ref`'s docstring: their
# config.json files are byte-identical in the downloaded snapshot), so there is no
# reason for the quantization recipe to differ between them.
H3_INT8_MODULES_TO_NOT_CONVERT = [
    "proj_in", "audio_proj_in", "context_embedder",
    "time_embedder", "time_proj", "token_refiner",
    "norm_out", "proj_out", "audio_proj_out",
]

# int8 shrinks each big transformer from ~66.3GB (bf16) to ~34.0GB (measured, see
# logs/server_int8.log), so transformer(34.0) + transformer_ref(~34, same recipe) +
# TE-nf4(21.0) = ~89GB steady state fits (barely -- ~6.6GB headroom) in this card's
# ~95.6GB. In this mode both big transformers stay GPU-resident permanently once
# loaded (loaded lazily, on first use of each variant), eliminating the ~62GB-class
# free+reload (~26-40s) that a t2va<->ref2va switch previously incurred every time in
# `none`/bf16 mode (see `_switch_to_variant`/`_free_other_variant_transformer`, both
# skip freeing the other variant's transformer when this is True). Only meaningful
# together with `H3_TRANSFORMER_QUANT=int8`; bf16 mode (~66.3GB each) cannot fit both
# at once and keeps the existing one-resident-at-a-time behaviour unchanged.
H3_TRANSFORMER_BOTH_RESIDENT = H3_TRANSFORMER_QUANT == "int8"

# EXPERIMENTAL, opt-in. "0" (default) = every mode above is untouched -- this flag is
# read nowhere else unless it is "1". "1" = 48GB-class low-VRAM mode: TE (bnb-4bit
# nf4, ~21GB) and the big transformer (int8, ~34GB) are never allowed to be
# GPU-resident *at the same time* -- 21+34=55GB alone already exceeds a 48GB card, so
# unlike every mode above (which all keep at least one 60GB+ class model resident
# between requests), this mode's steady state between requests is "nothing big"
# (transformer/transformer_ref/TE all freed; only the small VAE pair, ~11GB, and only
# while parked on CPU -- same as bnb-4bit's own VAE placement, see `_ensure_vaes`).
# Each request pays TE-load + transformer-load from scratch (see `generate()`'s
# lowvram branch): encode with TE resident -> free TE -> load transformer -> denoise
# (transformer alone, ~34+~5GB activations -> ~39GB) -> free transformer -> VAE to GPU
# -> decode (~11GB + buffers) -> VAE back to CPU. No transformer is reloaded at the
# end "for next time" (CLAUDE.md #33: only short, one-way trips -- never a standing
# swap -- and there is nothing useful to preload anyway since the *next* request needs
# TE first, not transformer). See the module docstring addendum below H3_HIRES_DENOISE
# for the full phase x resident-set table.
#
# Requires H3_TRANSFORMER_QUANT=int8 (bf16's 66.3GB transformer alone is already
# larger than a 48GB card with headroom for anything else) -- if the transformer quant
# was left at its own default ("none") while H3_LOWVRAM=1 is set, this is auto-upgraded
# to "int8" below (rather than silently running an unfittable bf16 config) UNLESS the
# operator *explicitly* set H3_TRANSFORMER_QUANT=none, in which case this raises at
# import time instead of silently overriding an explicit choice.
# H3_TRANSFORMER_BOTH_RESIDENT (both transformer AND transformer_ref resident at once,
# 34+34=68GB) is incompatible with this mode and is force-disabled below regardless of
# H3_TRANSFORMER_QUANT.
# upscale=1 (hires-fix) is rejected with a 400-mapped ValueError in this mode (see
# `generate()`) -- pass 2's ~4x-longer packed sequence was not verified to fit in the
# ~9GB of headroom this mode's steady state leaves at 48GB-class VRAM.
H3_LOWVRAM = os.environ.get("H3_LOWVRAM", "0").strip() == "1"
if H3_LOWVRAM:
    _explicit_transformer_quant = "H3_TRANSFORMER_QUANT" in os.environ
    if _explicit_transformer_quant and H3_TRANSFORMER_QUANT == "none":
        raise RuntimeError(
            "H3_LOWVRAM=1 requires an int8 transformer (bf16's 66.3GB does not fit a "
            "48GB-class card even alone) but H3_TRANSFORMER_QUANT=none was explicitly "
            "set. Drop H3_TRANSFORMER_QUANT (it will default to int8 under "
            "H3_LOWVRAM=1) or set H3_TRANSFORMER_QUANT=int8 explicitly."
        )
    H3_TRANSFORMER_QUANT = "int8"
    H3_TRANSFORMER_BOTH_RESIDENT = False
    # Every `H3_LOWVRAM` branch further down in this file (generate()/generate_ref2va())
    # is written assuming TE_QUANT == "bnb-4bit" (it is the only TE loading strategy
    # that produces a small-enough, movable-only-by-full-reload TE that this mode's
    # "never resident together with the transformer" choreography can work with --
    # `none` mode's 66.3GB bf16-native TE would not fit alongside anything else on a
    # 48GB-class card even on its own). Reject the combination explicitly rather than
    # silently mis-choreograph an unfittable 66.3GB TE.
    if TE_QUANT != "bnb-4bit":
        raise RuntimeError(
            f"H3_LOWVRAM=1 requires H3_TE_QUANT=bnb-4bit (default), got "
            f"H3_TE_QUANT={TE_QUANT!r}. bf16 TE (~66.3GB) cannot coexist with anything "
            "else on a 48GB-class card."
        )

# EXPERIMENTAL, opt-in. "" (default) = whatever diffusers' attention_dispatch picks
# natively (native/SDPA today) -- `set_attention_backend()` is never called, byte-for-byte
# identical to pre-this-flag behaviour. Any other value is passed straight to
# `transformer.set_attention_backend(...)` / `transformer_ref.set_attention_backend(...)`
# right after each big transformer loads (see `_ensure_transformer`/`_ensure_transformer_ref`)
# -- e.g. "sage" for SageAttention (see AttentionBackendName in diffusers/models/
# attention_dispatch.py for the full list of valid strings: "sage", "sage_varlen",
# "flash", "flash_hub", "xformers", ...). This project's stock `sageattention` install
# (comfy-env's 2.2.0, inherited via venv/site-packages/comfy_env.pth) has no sm_120
# (Blackwell) kernel compiled in -- confirmed by task-time probe: `sageattn(q,k,v)` raises
# "no kernel image is available for execution on the device". A source rebuild with
# `TORCH_CUDA_ARCH_LIST=12.0` (see third_party/SageAttention, scripts/build_sageattention.sh)
# targeting this box's actual arch is required before "sage"/"sage_varlen" can work; if the
# import-time sm_120 kernel is missing, `set_attention_backend("sage")` itself will not
# raise (it only stores the backend name on `self.processor._attention_backend`) but the
# first denoise step will, inside `sageattn()`. FBC (`H3_CACHE`) and this flag are
# independent and compose: FBC skips whole blocks based on residual similarity, this flag
# only changes how the *executed* blocks compute attention internally.
# "sage" (default; A/B verified 2026-08-05): SageAttention 2.2.0 built from source for
# sm_120 (scripts/build_sageattention.sh, ~2min build). Denoise 118s -> 104s (-12%) vs
# SDPA, fully deterministic (two same-seed runs byte-identical), visual quality
# equivalent (the ~21dB PSNR vs SDPA is trajectory drift from the int8-QK approximation,
# not degradation -- same phenomenon as H3_TRANSFORMER_QUANT=int8). Set
# H3_ATTN_BACKEND=default to revert to the pre-sage SDPA path.
H3_ATTN_BACKEND = os.environ.get("H3_ATTN_BACKEND", "sage").strip().lower()
if H3_ATTN_BACKEND in ("default", "none"):
    H3_ATTN_BACKEND = ""

# Two-pass hires-fix (see generate(..., upscale=1)): fraction of the *sigma schedule*
# (not step count) that pass 2 (high-res) is responsible for finishing. E.g. 0.35 with
# num_inference_steps=30 means pass 1 runs steps 0..18 (round(29*0.65)=19 of the 29 model
# evaluations -- MiniMaxH3Scheduler.set_timesteps() drives num_inference_steps - 1 model
# calls, see scheduling_minimax_h3.py) at the requested resolution. The video latent's x0
# estimate (not the noisy x_t -- see _upscale_block_state_2x's docstring for why: an
# earlier version upscaled x_t directly and reliably produced checkerboard-corrupted
# output) is then spatially upscaled 2x and re-noised with fresh noise at pass 2's
# starting sigma, and pass 2 runs the remaining steps at 2x resolution, continuing that
# freshly-noised trajectory. The scheduler's internal `_step_index` is not reset between
# passes (no new `set_timesteps()` call), so `step()`'s x_t/x0 blend uses the correct
# sigma/sigma_next pair for step N1 onward automatically.
H3_HIRES_DENOISE = float(os.environ.get("H3_HIRES_DENOISE", "0.35"))

# MINIMAX_H3_MIN_DURATION..MAX_DURATION = 5..15s at 24fps, aligned to 17*n+5.
MIN_SECONDS = 5.0
MAX_SECONDS = 15.0
FPS = 24


def _register_minimax_h3_block_for_fbc() -> None:
    """Register `MiniMaxH3TransformerBlock` with diffusers' `TransformerBlockRegistry`.

    FirstBlockCache (diffusers/hooks/first_block_cache.py) looks up per-block-class metadata
    (which forward arg/return slot is `hidden_states`) via `TransformerBlockRegistry.get()`.
    This diffusers version (PR #14355 branch) registers metadata for Wan/Flux/LTX/etc. blocks
    in `diffusers/hooks/_helpers.py::_register_transformer_blocks_metadata()` but does not yet
    include `MiniMaxH3TransformerBlock` -- `TransformerBlockRegistry.get()` raises `ValueError`
    for unregistered classes, so `transformer.enable_cache(FirstBlockCacheConfig(...))` would
    crash on the very first denoise step without this.

    `MiniMaxH3TransformerBlock.forward(hidden_states, temb, adaln_indices, rotary_emb,
    attention_mask) -> hidden_states` (see transformer_minimax_h3.py) returns a single tensor,
    not a tuple, and there is no encoder_hidden_states slot (H3 has no cross-attention -- text
    tokens are just rows in the packed sequence) -- same shape as `BasicTransformerBlock` /
    `WanTransformerBlock` / `LTXVideoTransformerBlock`'s registration:
    `return_hidden_states_index=0, return_encoder_hidden_states_index=None`.

    This only touches this project's runner code -- the venv's diffusers package itself is not
    modified (CLAUDE.md rule). Registration is idempotent (dict assignment), so calling this
    more than once (e.g. across server restarts within the same process, or defensively before
    every enable_cache call) is harmless.
    """
    from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock

    TransformerBlockRegistry.register(
        model_class=MiniMaxH3TransformerBlock,
        metadata=TransformerBlockMetadata(
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=None,
        ),
    )


def align_num_frames(num_frames: int) -> int:
    while num_frames % 17 != 5:
        num_frames += 1
    return num_frames


def seconds_to_num_frames(seconds: float) -> int:
    seconds = max(MIN_SECONDS, min(MAX_SECONDS, seconds))
    return align_num_frames(round(seconds * FPS))


def gpu_mem_gb() -> dict:
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
        "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }


def ram_gb() -> dict:
    meminfo = {}
    with open("/proc/meminfo") as f:
        for line in f:
            parts = line.split()
            meminfo[parts[0].rstrip(":")] = int(parts[1])
    total = meminfo["MemTotal"] / 1e6
    avail = meminfo["MemAvailable"] / 1e6
    swap_total = meminfo.get("SwapTotal", 0) / 1e6
    swap_free = meminfo.get("SwapFree", 0) / 1e6
    return {
        "avail_gb": round(avail, 1),
        "total_gb": round(total, 1),
        "swap_used_gb": round(swap_total - swap_free, 2),
        "swap_total_gb": round(swap_total, 1),
    }


class _NullContext:
    """A no-op context manager, used where FBC's `cache_context` is conditionally absent
    (H3_CACHE == "none") but the calling code wants one `with` statement either way."""

    def __enter__(self):
        return None

    def __exit__(self, *exc_info):
        return False


@dataclass
class ProgressState:
    """Simple polling-friendly progress snapshot, in the spirit of diffusers-server's core/progress.py."""

    job_id: str = ""
    phase: str = "idle"  # idle | loading_text_encoder | encoding | loading_transformer | denoising | decoding | done | error
    step: int = 0
    total_steps: int = 0
    started_at: float = 0.0
    updated_at: float = 0.0
    message: str = ""
    error: str | None = None
    result_path: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)
            self.updated_at = time.time()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "job_id": self.job_id,
                "phase": self.phase,
                "step": self.step,
                "total_steps": self.total_steps,
                "elapsed_s": round(time.time() - self.started_at, 1) if self.started_at else 0.0,
                "message": self.message,
                "error": self.error,
                "result_path": self.result_path,
            }


class MiniMaxH3Runner:
    """
    Holds the ModularPipeline shell and manages component residency.

    Not thread-safe by itself -- callers must serialize generate() calls (the app does
    this with a single global lock, matching diffusers-server's one-generation-at-a-time
    design).
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        self._pipe = None
        self._transformer_loaded = False
        self._vae_loaded = False
        self._text_encoder_loaded = False
        # bnb-4bit mode only: whether the (permanently-loaded-in-RAM-terms, but
        # phase-cycled-on-GPU) VAEs are currently placed on GPU or parked on CPU.
        self._vae_on_gpu = False
        self._load_lock = threading.Lock()

        # --- ref2va (omni-reference) additions ---
        # A second ModularPipeline shell, built from MiniMaxH3Ref2VABlocks (the only way
        # to get a pipe whose `_component_specs` know about `transformer_ref` -- the
        # default `ModularPipeline.from_pretrained(MODEL_ID)` shell above is built from
        # MiniMaxH3Blocks (t2va/fl2va) and its spec table has no `transformer_ref` entry
        # at all, confirmed by probe: `pipe.load_components(names=["transformer_ref"])`
        # on the t2va shell logs "Unknown components will be ignored: {'transformer_ref'}"
        # and leaves `pipe.transformer_ref` unset. `transformer`/`transformer_ref` are
        # each ~66.3GB bf16 and cannot coexist in this card's ~96GB (same constraint as
        # TE vs transformer above), so only one of `self._pipe.transformer` /
        # `self._pipe_ref.transformer_ref` is ever GPU-resident at a time -- tracked by
        # `self._active_variant`. Every other component (text_encoder, tokenizer,
        # processor, vae, audio_vae, scheduler, audio_scheduler, video_processor) is
        # loaded once on `self._pipe` and shared onto `self._pipe_ref` by plain attribute
        # assignment (`ModularPipeline.components` is just `{name: getattr(self, name)
        # for name in self._component_specs if hasattr(self, name)}` -- confirmed by
        # reading modular_pipeline.py -- so this is not a hack, it is the documented shape
        # of that dict) -- avoids a second ~66GB TE / ~11GB VAE load.
        self._pipe_ref = None
        self._transformer_ref_loaded = False
        # "t2va" | "ref2va" | None (nothing loaded yet). Only one of `transformer` /
        # `transformer_ref` may be GPU-resident at a time; this is the single source of
        # truth callers check before a cross-variant swap.
        self._active_variant: str | None = None

    # ------------------------------------------------------------------
    # Component lifecycle
    # ------------------------------------------------------------------
    def _ensure_pipe_shell(self):
        if self._pipe is not None:
            return
        from diffusers import ModularPipeline

        logger.info("building ModularPipeline shell from %s (H3_TE_QUANT=%s)", MODEL_ID, TE_QUANT)
        self._pipe = ModularPipeline.from_pretrained(MODEL_ID)
        logger.info("pipe shell built: blocks=%s components=%s",
                     self._pipe._blocks.__class__.__name__, self._pipe.component_names)

    def _ensure_pipe_ref_shell(self):
        """Build the second ModularPipeline shell (MiniMaxH3Ref2VABlocks), whose spec table
        knows about `transformer_ref`. See the `_pipe_ref` field comment in `__init__` for
        why a second shell is required at all (the t2va shell's spec table has no
        `transformer_ref` entry). Idempotent, and does not load any component weights.
        """
        self._ensure_pipe_shell()
        if self._pipe_ref is not None:
            return
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Ref2VABlocks

        logger.info("building ref2va ModularPipeline shell from %s", MODEL_ID)
        self._pipe_ref = MiniMaxH3Ref2VABlocks().init_pipeline(MODEL_ID)
        logger.info("ref2va pipe shell built: blocks=%s components=%s",
                     self._pipe_ref._blocks.__class__.__name__, self._pipe_ref.component_names)

    def _sync_shared_components_to_ref(self):
        """Mirror every component the two shells have in common (everything except
        `transformer` / `transformer_ref` themselves) from `self._pipe` onto
        `self._pipe_ref`, by plain attribute assignment -- confirmed safe by reading
        `ModularPipeline.components`'s implementation (see `_pipe_ref` field comment).
        Called before any ref2va block runs, so text_encoder/vae/audio_vae/schedulers are
        loaded exactly once regardless of which variant a request asks for. Safe to call
        repeatedly (e.g. once per generate() call): each assignment just re-points the
        same already-loaded module, it never re-loads or copies weights.
        """
        self._ensure_pipe_ref_shell()
        for name in ("text_encoder", "tokenizer", "processor", "vae", "audio_vae",
                     "scheduler", "audio_scheduler", "video_processor"):
            component = getattr(self._pipe, name, None)
            if component is not None:
                setattr(self._pipe_ref, name, component)

    def _ensure_vaes(self, progress: ProgressState | None = None):
        """Load vae + audio_vae (~11GB fp32) component weights (host RAM/disk -> not GPU yet
        in bnb-4bit mode). In `none` mode these are placed on GPU immediately and stay there
        permanently, matching the original behaviour.
        """
        self._ensure_pipe_shell()
        if self._vae_loaded:
            return
        if progress:
            progress.update(phase="loading_vae", message="vae/audio_vae をロード中...")
        t1 = time.time()
        # video VAE must stay fp32 (decode step applies its own fp16 autocast);
        # audio VAE must stay fp32 end-to-end (bf16 causes ~20dB volume loss, see
        # module docstring / handoff doc).
        self._pipe.load_components(names=["vae", "audio_vae"], dtype=torch.float32)
        if TE_QUANT == "bnb-4bit":
            # Parked on CPU by default in this mode -- moved to GPU only for the phase
            # that needs them (keyframe encode / decode). See module docstring.
            self._pipe.vae.to(CPU)
            self._pipe.audio_vae.to(CPU)
            self._vae_on_gpu = False
        else:
            self._pipe.vae.to(DEVICE)
            self._pipe.audio_vae.to(DEVICE)
            self._vae_on_gpu = True
        self._pipe.load_components(names=["scheduler", "audio_scheduler"])
        self._vae_loaded = True
        logger.info("vae/audio_vae loaded (%s) in %.1fs. gpu=%s ram=%s",
                     "GPU" if self._vae_on_gpu else "CPU", time.time() - t1, gpu_mem_gb(), ram_gb())

        from diffusers.video_processor import VideoProcessor

        if getattr(self._pipe, "video_processor", None) is None:
            self._pipe.video_processor = VideoProcessor(vae_scale_factor=16, do_normalize=False)

    def _vae_to_gpu(self):
        """bnb-4bit mode only: move the (small, fp32, ~11GB) VAEs onto GPU for their active
        phase. A single short one-way trip, not a standing swap -- see module docstring.
        """
        if TE_QUANT != "bnb-4bit" or self._vae_on_gpu:
            return
        t0 = time.time()
        self._pipe.vae.to(DEVICE)
        self._pipe.audio_vae.to(DEVICE)
        self._vae_on_gpu = True
        logger.info("vae/audio_vae -> GPU in %.2fs. gpu=%s", time.time() - t0, gpu_mem_gb())

    def _vae_to_cpu(self):
        """bnb-4bit mode only: move the VAEs back off GPU once their phase is done, to make
        room for the permanently-resident transformer + TE-nf4 during denoise.
        """
        if TE_QUANT != "bnb-4bit" or not self._vae_on_gpu:
            return
        t0 = time.time()
        self._pipe.vae.to(CPU)
        self._pipe.audio_vae.to(CPU)
        self._vae_on_gpu = False
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("vae/audio_vae -> CPU in %.2fs. gpu=%s", time.time() - t0, gpu_mem_gb())

    def _ensure_transformer(self, progress: ProgressState | None = None):
        """Load the 66GB bf16 transformer to GPU (or, with `H3_TRANSFORMER_QUANT=int8`,
        weight-only int8-quantize it via torchao in the same `from_pretrained` call).

        `none` mode: frees the text_encoder first if resident (they cannot coexist).
        `bnb-4bit` mode: TE-nf4 is permanently resident, nothing to free here. Called at
        startup, and again after every request's decode phase (which drops the
        transformer for its ~9s window -- see the decode section of `generate()`) to
        restore the transformer+TE-nf4 steady state between requests.

        int8 path: `quantization_config` is passed straight into `load_components`
        (same per-component-kwarg dict shape `_load_text_encoder` already uses for TE's
        `BitsAndBytesConfig`), so `from_pretrained` quantizes the module as it materializes
        each shard on `device_map="cuda"` -- there is no separate "load bf16 to GPU, then
        quantize in place" step, matching the component-wise cuda-direct loading pattern
        this file uses everywhere else (never a CPU-wide staging pass for a 60GB+ module,
        per CLAUDE.md #33 as referenced in the module docstring).
        """
        self._ensure_pipe_shell()
        if self._transformer_loaded:
            # int8 both-resident mode: this can be a "just mark it active again" call
            # (transformer already resident, transformer_ref was the one last used) --
            # `_switch_to_variant`'s early-return check reads `_active_variant` alongside
            # the loaded flags, so this must still update it even on the cached-return
            # path, or a t2va request right after a ref2va one would leave
            # `_active_variant == "ref2va"` despite `transformer` being the one actually
            # about to be used for denoising.
            self._active_variant = "t2va"
            return
        if TE_QUANT != "bnb-4bit":
            # TE (66GB) + transformer (66GB) cannot coexist in 96GB VRAM.
            self._free_text_encoder()
        if progress:
            progress.update(phase="loading_transformer", message="transformer をロード中...")
        t0 = time.time()
        if H3_TRANSFORMER_QUANT == "int8":
            from diffusers import TorchAoConfig
            from torchao.quantization import Int8WeightOnlyConfig

            quant_config = TorchAoConfig(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=H3_INT8_MODULES_TO_NOT_CONVERT,
            )
            self._pipe.load_components(
                names=["transformer"],
                dtype=torch.bfloat16,
                quantization_config={"transformer": quant_config},
                device_map={"transformer": "cuda"},
            )
        else:
            self._pipe.load_components(names=["transformer"], dtype=torch.bfloat16)
            self._pipe.transformer.to(DEVICE)
        # `ModularPipeline.load_components()` swallows the underlying exception
        # internally (`modular_pipeline.py`'s `try/except Exception: ... logger.warning
        # (...); continue` around each component's `spec.load()`) and does NOT
        # re-raise -- a failed load (e.g. CUDA OOM inside `from_pretrained`) just logs a
        # warning and leaves `self._pipe.transformer` unset, with no exception for this
        # method to catch. Reproduced during this task's own verification: an int8-mode
        # OOM inside `from_pretrained`'s `_caching_allocator_warmup` (a fragmentation
        # issue, not an over-budget one -- "Tried to allocate 15.43 GiB" with the
        # allocator already holding 37GB reserved-but-unallocated) surfaced only as a
        # confusing `AttributeError: 'NoneType' object has no attribute 'enable_cache'`
        # three lines below, with `self._transformer_loaded` about to be wrongly marked
        # `True` for a component that was never actually loaded. Checking explicitly
        # here turns that into a clear, correctly-attributed error instead.
        if getattr(self._pipe, "transformer", None) is None:
            raise RuntimeError(
                "transformer load failed (see the diffusers 'Failed to create component "
                "transformer' warning above for the underlying error, often CUDA OOM) -- "
                "self._pipe.transformer is still None after load_components()."
            )
        self._transformer_loaded = True
        self._active_variant = "t2va"
        if H3_ATTN_BACKEND:
            self._pipe.transformer.set_attention_backend(H3_ATTN_BACKEND)
            logger.info("transformer attention backend set to %r", H3_ATTN_BACKEND)
        if H3_CACHE == "fbc":
            self._enable_fbc()
        logger.info(
            "transformer loaded to GPU in %.1fs (quant=%s). gpu=%s ram=%s",
            time.time() - t0, H3_TRANSFORMER_QUANT, gpu_mem_gb(), ram_gb(),
        )

    def _fbc_last_step_was_skip(self) -> int:
        """Best-effort introspection of whether the just-finished transformer forward skipped
        the remaining blocks (cache hit). Reads `FBCSharedBlockState.should_compute` off the
        head block's hook (see first_block_cache.py) -- `should_compute=False` means the tail
        blocks were skipped and the cached residual was reused instead. This is diagnostic only
        (for the A/B measurement task): wrapped in try/except so a diffusers-internals change
        degrades to "unknown" (0) rather than breaking generation.
        """
        try:
            from diffusers.hooks.first_block_cache import _FBC_LEADER_BLOCK_HOOK

            head_block = self._pipe.transformer.transformer_blocks[0]
            hook = head_block._diffusers_hook.get_hook(_FBC_LEADER_BLOCK_HOOK)
            shared_state = hook.state_manager.get_state()
            return 0 if shared_state.should_compute else 1
        except Exception:
            return 0

    def _enable_fbc(self):
        """Attach FirstBlockCache hooks to the (freshly loaded) transformer.

        Called once right after every transformer load (startup preload, and any reload that
        happens after `bnb-4bit` mode drops the transformer around its decode window -- see
        `_free_transformer`/decode section of `generate()`). A freshly-loaded transformer has
        no `_diffusers_hook` yet, so this always starts from a clean slate; there is no stale
        state to worry about across a drop+reload cycle in bnb-4bit mode. Per-*request* reset
        (for the more common case where the transformer stays resident across requests) is
        handled separately in `generate()` via `_reset_stateful_cache()` + `cache_context()`.
        """
        from diffusers.hooks import FirstBlockCacheConfig

        _register_minimax_h3_block_for_fbc()
        self._pipe.transformer.enable_cache(FirstBlockCacheConfig(threshold=H3_CACHE_THRESHOLD))
        logger.info("FirstBlockCache enabled on transformer (threshold=%s)", H3_CACHE_THRESHOLD)

    def _free_transformer(self):
        if not self._transformer_loaded:
            return
        # Drop in place, no CPU staging (same reasoning as _free_text_encoder).
        del self._pipe.transformer
        self._pipe.transformer = None
        self._transformer_loaded = False
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("transformer freed. gpu=%s ram=%s", gpu_mem_gb(), ram_gb())

    # ------------------------------------------------------------------
    # ref2va transformer_ref lifecycle (mirrors transformer's, above)
    # ------------------------------------------------------------------
    def _ensure_transformer_ref(self, progress: ProgressState | None = None):
        """Load the transformer_ref (66GB bf16, or ~34GB int8-quantized -- see
        `H3_TRANSFORMER_QUANT`) to GPU, onto the ref2va pipe shell.

        bf16 mode: `transformer` and `transformer_ref` are each ~66.3GB and cannot
        coexist in this card's ~96GB (same one-big-model-at-a-time constraint the
        t2va/fl2va path already enforces between TE and transformer, in `none` TE mode).
        Callers must go through `_switch_to_variant("ref2va")` rather than calling this
        directly, so the t2va transformer is freed first when it is the one resident --
        this method itself only handles the transformer_ref side of that swap.

        int8 mode (`H3_TRANSFORMER_BOTH_RESIDENT`): both transformers fit at once
        (~34GB each), so `transformer` is left alone here -- this is called directly by
        `generate_ref2va()` without going through `_switch_to_variant`/
        `_free_other_variant_transformer` in that mode (see those methods' docstrings).

        int8 quantization uses the exact same recipe as `_ensure_transformer` (same
        model class/config, see `H3_INT8_MODULES_TO_NOT_CONVERT`'s comment).
        """
        self._ensure_pipe_ref_shell()
        if self._transformer_ref_loaded:
            # See `_ensure_transformer`'s matching comment: must update
            # `_active_variant` even on the cached-return path, for int8 both-resident
            # mode's `_switch_to_variant` early-return check.
            self._active_variant = "ref2va"
            return
        if progress:
            progress.update(phase="loading_transformer", message="transformer_ref (ref2va) をロード中...")
        t0 = time.time()
        if H3_TRANSFORMER_QUANT == "int8":
            from diffusers import TorchAoConfig
            from torchao.quantization import Int8WeightOnlyConfig

            quant_config = TorchAoConfig(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=H3_INT8_MODULES_TO_NOT_CONVERT,
            )
            self._pipe_ref.load_components(
                names=["transformer_ref"],
                dtype=torch.bfloat16,
                quantization_config={"transformer_ref": quant_config},
                device_map={"transformer_ref": "cuda"},
            )
        else:
            self._pipe_ref.load_components(names=["transformer_ref"], dtype=torch.bfloat16)
            self._pipe_ref.transformer_ref.to(DEVICE)
        # See `_ensure_transformer`'s matching check/comment: `load_components()` does
        # not re-raise on a failed component load (e.g. CUDA OOM), it only logs a
        # warning and leaves the attribute unset -- verify explicitly rather than let a
        # `None` transformer_ref surface later as a confusing AttributeError.
        if getattr(self._pipe_ref, "transformer_ref", None) is None:
            raise RuntimeError(
                "transformer_ref load failed (see the diffusers 'Failed to create "
                "component transformer_ref' warning above for the underlying error, "
                "often CUDA OOM) -- self._pipe_ref.transformer_ref is still None after "
                "load_components()."
            )
        self._transformer_ref_loaded = True
        self._active_variant = "ref2va"
        if H3_ATTN_BACKEND:
            self._pipe_ref.transformer_ref.set_attention_backend(H3_ATTN_BACKEND)
            logger.info("transformer_ref attention backend set to %r", H3_ATTN_BACKEND)
        if H3_CACHE == "fbc":
            self._enable_fbc_ref()
        logger.info(
            "transformer_ref loaded to GPU in %.1fs (quant=%s). gpu=%s ram=%s",
            time.time() - t0, H3_TRANSFORMER_QUANT, gpu_mem_gb(), ram_gb(),
        )

    def _enable_fbc_ref(self):
        """Attach FirstBlockCache hooks to the (freshly loaded) transformer_ref.

        `transformer_ref` is the very same `MiniMaxH3Transformer3DModel` class as
        `transformer` (confirmed: `transformer/config.json` and
        `transformer_ref/config.json` are byte-identical in the downloaded snapshot), so
        the block-class registration `_register_minimax_h3_block_for_fbc()` performs is
        shared -- no separate registration needed, just a separate `enable_cache()` call
        against this transformer instance's own submodules.
        """
        from diffusers.hooks import FirstBlockCacheConfig

        _register_minimax_h3_block_for_fbc()
        self._pipe_ref.transformer_ref.enable_cache(FirstBlockCacheConfig(threshold=H3_CACHE_THRESHOLD))
        logger.info("FirstBlockCache enabled on transformer_ref (threshold=%s)", H3_CACHE_THRESHOLD)

    def _fbc_last_step_was_skip_ref(self) -> int:
        """Same as `_fbc_last_step_was_skip`, against `transformer_ref`'s own hook state."""
        try:
            from diffusers.hooks.first_block_cache import _FBC_LEADER_BLOCK_HOOK

            head_block = self._pipe_ref.transformer_ref.transformer_blocks[0]
            hook = head_block._diffusers_hook.get_hook(_FBC_LEADER_BLOCK_HOOK)
            shared_state = hook.state_manager.get_state()
            return 0 if shared_state.should_compute else 1
        except Exception:
            return 0

    def _free_transformer_ref(self):
        if not self._transformer_ref_loaded:
            return
        # Drop in place, no CPU staging -- same reasoning as _free_transformer /
        # _free_text_encoder (CLAUDE.md #33: no whole-module CPU-staging trips for
        # 60GB+ modules on this box).
        del self._pipe_ref.transformer_ref
        self._pipe_ref.transformer_ref = None
        self._transformer_ref_loaded = False
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("transformer_ref freed. gpu=%s ram=%s", gpu_mem_gb(), ram_gb())

    def _free_other_variant_transformer(self, variant: str):
        """Free the *other* variant's big transformer (if resident) so this request's own
        variant has room to load its own -- without loading anything itself.

        bf16 mode: `transformer`/`transformer_ref` are each ~66.3GB and never coexist in
        this card's ~96GB. This is split out from actually loading `variant`'s own
        transformer (see `_switch_to_variant`'s docstring for why) so a caller can free
        the other side early -- before a vae-heavy encode step that itself needs
        headroom -- and defer its own ~66.3GB load until after that step, mirroring the
        ordering `generate()` already uses for the fl2va keyframe-encode-then-
        transformer-load sequence. Idempotent: a no-op when the other variant's
        transformer was not resident.

        int8 mode (`H3_TRANSFORMER_BOTH_RESIDENT`): a deliberate no-op. Both
        transformers fit in VRAM at once (~34GB each + TE-nf4 21GB = ~89GB steady
        state), so there is no "other variant" to evict any more -- this is the whole
        point of int8 mode, eliminating the ~62GB-class free+reload a t2va<->ref2va
        switch previously required every time.
        """
        if variant not in ("t2va", "ref2va"):
            raise ValueError(f"variant must be 't2va' or 'ref2va', got {variant!r}")
        if H3_TRANSFORMER_BOTH_RESIDENT:
            return
        if variant == "ref2va":
            self._free_transformer()
        else:
            self._free_transformer_ref()

    def _switch_to_variant(self, variant: str, progress: ProgressState | None = None):
        """Ensure the requested variant's big transformer is the one GPU-resident *right
        now*, freeing the other one first if it is currently loaded.

        `variant`: "t2va" (serves t2va/fl2va requests, `self._pipe.transformer`) or
        "ref2va" (serves ref2va requests, `self._pipe_ref.transformer_ref`).
        `_active_variant` is only ever updated here or inside `_ensure_transformer`/
        `_ensure_transformer_ref` themselves, so it always reflects which one is actually
        GPU-resident.

        CAUTION: this loads `variant`'s transformer immediately -- correct for
        `generate()`'s t2va path (whose text encoding happens with the TE resident and
        does not additionally need the transformer's own vae, so there is no headroom
        conflict to defer around), but **not** used for ref2va's entry any more: ref2va's
        reference-encoder step needs `vae`/`audio_vae` on GPU before `transformer_ref` is
        loaded (transformer_ref(66.3) + TE-nf4(21.0) + vae pair(11.0) already exceeds this
        card's ~95.6GB -- the identical three-way conflict `generate()`'s own fl2va/decode
        comments document). `generate_ref2va()` instead calls
        `_free_other_variant_transformer("ref2va")` early (frees `transformer` only, if
        resident) and `_ensure_transformer_ref()` later, after the reference encoder step
        and (in bnb-4bit mode) after `_vae_to_cpu()` -- the same split
        `_free_other_variant_transformer`/`_ensure_transformer_ref` this method is built
        from, just not fused into one call for that path. Kept for `generate()`'s t2va
        entry point, where the fused "free other + load mine now" shape is safe.

        int8 mode (`H3_TRANSFORMER_BOTH_RESIDENT`): `_free_other_variant_transformer` is
        a no-op (see its docstring), so this degrades to "load `variant`'s transformer
        if not already resident, and update `_active_variant`" -- both transformers
        end up loaded (lazily, on each one's first use) and stay loaded from then on.
        `_active_variant` still tracks "most recently used" in this mode: it is read by
        the decode-window drop/reload logic in `generate()`/`generate_ref2va()`, which
        (even in int8 mode) still drops the *just-used* transformer for the short decode
        window to make room for the VAE pair -- see those methods' decode sections.
        """
        if variant not in ("t2va", "ref2va"):
            raise ValueError(f"variant must be 't2va' or 'ref2va', got {variant!r}")
        if self._active_variant == variant and (
            self._transformer_loaded if variant == "t2va" else self._transformer_ref_loaded
        ):
            return
        t0 = time.time()
        self._free_other_variant_transformer(variant)
        if variant == "ref2va":
            self._ensure_transformer_ref(progress)
        else:
            self._ensure_transformer(progress)
        logger.info("switched active variant -> %s in %.1fs. gpu=%s ram=%s",
                     variant, time.time() - t0, gpu_mem_gb(), ram_gb())

    def _load_text_encoder(self, progress: ProgressState | None = None):
        """Load the text_encoder to GPU.

        `none` mode: ~66GB bf16-native TE, loaded/freed per request, frees the
        transformer first (they cannot coexist).
        `bnb-4bit` mode: ~18GB NF4-quantized TE, loaded once at startup and kept
        resident forever (bnb 4bit models cannot be moved between devices, so
        `device_map="cuda"` places it directly and there is nothing to cycle).
        """
        self._ensure_pipe_shell()
        if self._text_encoder_loaded:
            return
        if TE_QUANT == "bnb-4bit":
            if progress:
                progress.update(phase="loading_text_encoder", message="text_encoder (Qwen3-VL-32B, NF4) をロード中...")
            t0 = time.time()
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            # Per-component kwargs: `load_components` broadcasts a plain (non-dict) kwarg
            # value to every named component, but tokenizer/processor do not accept
            # `quantization_config` or `device_map`. Use the dict form (component name ->
            # value) so only `text_encoder` gets them.
            self._pipe.load_components(
                names=["text_encoder", "tokenizer", "processor"],
                dtype=torch.bfloat16,
                quantization_config={"text_encoder": quant_config},
                device_map={"text_encoder": "cuda"},
            )
            self._text_encoder_loaded = True
            logger.info(
                "text_encoder (NF4) loaded to GPU in %.1fs. gpu=%s ram=%s", time.time() - t0, gpu_mem_gb(), ram_gb()
            )
            return
        # TE (66GB) + transformer (66GB) cannot coexist in 96GB VRAM: measured 66.73GB
        # for the TE alone (the checkpoint shards are bf16-native, not fp32). The two
        # big models therefore cycle: TE on GPU only during prompt encoding.
        self._free_transformer()
        if progress:
            progress.update(phase="loading_text_encoder", message="text_encoder (Qwen3-VL-32B) をロード中...")
        t0 = time.time()
        self._pipe.load_components(names=["text_encoder", "tokenizer", "processor"], dtype=torch.bfloat16)
        self._pipe.text_encoder.to(DEVICE)
        self._text_encoder_loaded = True
        logger.info("text_encoder loaded to GPU in %.1fs. gpu=%s ram=%s", time.time() - t0, gpu_mem_gb(), ram_gb())

    def _free_text_encoder(self, force: bool = False):
        """Free the resident text_encoder.

        `force=False` (default): in `bnb-4bit` mode this is a no-op (TE-nf4 is normally
        kept permanently resident -- see `_load_text_encoder` docstring); in `none` mode
        it always frees (TE/transformer already cycle every request there).

        `force=True`: used only by the hires-fix upscale path (`generate(..., upscale=1)`)
        to actually drop the nf4 TE (~21GB) after prompt encoding, buying headroom for
        pass 2's much larger attention activations at 2x spatial resolution (sequence
        length is 4x -> full self-attention cost is ~16x). bnb 4bit modules cannot be
        `.to()`-moved between devices (CLAUDE.md-style constraint carried over from
        diffusers-server, see module docstring point 33/47 lineage: only "drop in place,
        reload from disk/page-cache later" is available for a quantized module, never a
        host-RAM staging trip) -- `del` + a later `_load_text_encoder()` call (which
        re-quantizes from the safetensors shards straight to CUDA) is the only option,
        exactly like the transformer drop/reload the decode window already does in this
        mode.
        """
        if not self._text_encoder_loaded:
            return
        if TE_QUANT == "bnb-4bit" and not force:
            # Permanently resident in this mode -- never freed mid-run (see
            # _load_text_encoder docstring). Guard so a stray call is a harmless no-op
            # rather than silently dropping the model.
            logger.debug("bnb-4bit text_encoder is permanently resident; ignoring free request")
            return
        # Drop the CUDA model directly: releasing the last reference frees the VRAM in
        # place. Do NOT stage through .to("cpu") first -- the text_encoder is ~21-66GB
        # depending on quantization, and a host-RAM transit would both waste time and
        # evict the page-cached model shards that make the next per-request reload fast.
        #
        # BUG FOUND DURING THIS TASK'S OWN VERIFICATION: `self._pipe_ref.text_encoder`
        # (set by `_sync_shared_components_to_ref` via plain attribute assignment, so it
        # is the *same* module object as `self._pipe.text_encoder`, not a copy) also has
        # to be cleared here, or it keeps the refcount above zero and `del
        # self._pipe.text_encoder` frees nothing -- reproduced on this task's own second
        # ref2va attempt: `_free_text_encoder(force=True)` logged as having run, but
        # `gpu.allocated_gb` did not drop (stayed at ~87.5GB, TE-nf4's ~21GB never
        # released), and the very next denoise step OOM'd identically to the un-freed
        # case. Only `text_encoder` needs this (the only shared component this class ever
        # `del`s outright -- vae/audio_vae are `.to()`-moved, never deleted, and
        # transformer/transformer_ref are never shared between the two shells).
        del self._pipe.text_encoder
        self._pipe.text_encoder = None
        if self._pipe_ref is not None and getattr(self._pipe_ref, "text_encoder", None) is not None:
            del self._pipe_ref.text_encoder
            self._pipe_ref.text_encoder = None
        self._text_encoder_loaded = False
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("text_encoder freed (force=%s). gpu=%s ram=%s", force, gpu_mem_gb(), ram_gb())

    def preload_all(self):
        """Load the steady-state residents once at startup.

        `none` mode: transformer + VAEs (the text_encoder cycles per request, so
        preloading it would only be churn).
        `bnb-4bit` mode: transformer + text_encoder(NF4) + VAEs are ALL loaded here --
        the VAEs' weights are loaded now (onto CPU, see _ensure_vaes) and the TE is
        loaded straight to GPU permanently, since nothing cycles anymore in this mode.
        `H3_LOWVRAM=1`: this mode's whole point is that TE (21GB) and transformer
        (34GB) are never GPU-resident together, so neither is preloaded here -- both
        are loaded fresh, per-request, by `generate()`/`generate_ref2va()` (see the
        H3_LOWVRAM module comment's phase table). Only the VAE pair's *weights* are
        preloaded (onto CPU, same as bnb-4bit -- `_ensure_vaes` already parks them on
        CPU whenever TE_QUANT=="bnb-4bit", which H3_LOWVRAM always implies), so the
        per-request decode phase only pays a CPU->GPU move, not a disk/HF-cache load.
        """
        with self._load_lock:
            self._ensure_vaes()
            if not H3_LOWVRAM:
                self._ensure_transformer()
                if TE_QUANT == "bnb-4bit":
                    self._load_text_encoder()

    def status(self) -> dict:
        return {
            "pipe_built": self._pipe is not None,
            "transformer_loaded": self._transformer_loaded,
            "pipe_ref_built": self._pipe_ref is not None,
            "transformer_ref_loaded": self._transformer_ref_loaded,
            # True once both big transformers are simultaneously GPU-resident (only
            # possible in H3_TRANSFORMER_QUANT=int8 mode, see H3_TRANSFORMER_BOTH_RESIDENT) --
            # i.e. the t2va<->ref2va switch cost has actually been eliminated for the
            # *current* process, not just "the flag that requests it is set".
            "both_transformers_resident": self._transformer_loaded and self._transformer_ref_loaded,
            "transformer_both_resident_mode": H3_TRANSFORMER_BOTH_RESIDENT,
            "active_variant": self._active_variant,
            "vae_loaded": self._vae_loaded,
            "vae_on_gpu": self._vae_on_gpu,
            "text_encoder_loaded": self._text_encoder_loaded,
            "te_quant": TE_QUANT,
            "transformer_quant": H3_TRANSFORMER_QUANT,
            "lowvram": H3_LOWVRAM,
            "attn_backend": H3_ATTN_BACKEND or "default",
            "cache_mode": H3_CACHE,
            "cache_threshold": H3_CACHE_THRESHOLD if H3_CACHE == "fbc" else None,
            "gpu": gpu_mem_gb(),
            "ram": ram_gb(),
        }

    # ------------------------------------------------------------------
    # Hires-fix (two-pass upscale) helpers
    # ------------------------------------------------------------------
    def _upscale_block_state_2x(self, components, block_state, state, pass1_steps: int, last_step_info: dict):
        """Spatially upscale the video latent of `block_state` 2x between pass 1 and pass 2
        of hires-fix, and rebuild the packed-sequence layout (row_timestep_plan, position_ids,
        token_tags, video/audio/text_indices) for the new resolution's *remaining* timesteps.

        IMPORTANT (found during this task's own verification, not assumed from the reference
        up front): this upscales the pass-1 **x0 estimate** (the model's denoised prediction),
        not the noisy `x_t` sample directly, then re-noises the upscaled x0 at the pass-2
        starting sigma with fresh noise. The first implementation bilinear-interpolated
        `block_state.latents` (the noisy `x_t`) directly, matching a naive reading of "spatial
        2x upscale of the video latent between passes" -- this reliably produced a checkerboard/
        moire-corrupted decode (reproduced and isolated with `scripts/debug_vae_direct.py`:
        the corruption persists even with `vae.disable_tiling()`, so it is not a VAE tiling-seam
        artifact, and it is present in a *direct* decode of the interpolated latent with zero
        pass-2 steps run, so it is not something pass 2 could ever "clean up" -- if anything pass
        2 amplifies it into total noise because the model is being asked to denoise a `x_t` whose
        noise component has been low-pass-filtered by the bilinear resize, which is off-distribution
        for what the model expects a genuine forward-process sample to look like at that sigma).
        Re-reading the ComfyUI reference's own description in light of this (`utils.py`, fetched
        during this task): its pass 1 is read from `SamplerCustomAdvanced`'s `denoised_output`
        (already the x0 estimate, not the noisy latent) and its upscale node explicitly re-noises
        via `model_sampling.noise_scaling(sigma_start, fresh_noise, upscaled_latent)` afterward --
        i.e. the reference *never* interpolates a noisy sample either. This implementation follows
        that shape once translated to this scheduler's own `scale_noise(sample, timestep, noise)`
        API (`x_t = t*x0 + (1-t)*noise`, this repo's rectified-flow convention, see
        scheduling_minimax_h3.py): x0 is reconstructed here from the last pass-1 step's
        `(sample, model_output, t)` via the same formula `MiniMaxH3Scheduler.step()` uses
        internally (`denoised = sample + (1-t)*model_output`) since the block wrapper discards
        it, that x0 is what gets bilinear-interpolated, and the result is re-noised with **fresh**
        noise at pass 2's first timestep before pass 2's loop begins.

        Only the *video* rows have spatial extent (`(t, h, w)` -> `F.interpolate`); the audio
        rows are channel-major and carry no height/width coordinate at all (see
        `build_packed_sequence` in packing.py -- their rotary position only has a time axis
        and a fixed left/right width-grid endpoint pin), so they are left completely
        untouched here, matching the reference ComfyUI node's audio pass-through
        (`audio_denoise=0` behaviour) -- this task's design choice per the brief.

        This function assumes `num_condition_video_rows == 0` / `num_condition_audio_rows
        == 0` (t2va only, no keyframe conditioning rows) -- enforced by the `ValueError`
        `generate()` raises for fl2va + upscale before this is ever reached.
        """
        from diffusers.modular_pipelines.minimax_h3.packing import (
            build_packed_sequence,
            build_row_timesteps,
            patchify_video_latents,
            unpatchify_video_tokens,
            MINIMAX_H3_KEYFRAME_NOISE_AUG,
        )
        # NOTE: deliberately NOT using `components._execution_device` here (unlike the
        # rest of this file's calls into the modular blocks). By the time this runs, TE
        # has already been force-freed (see `generate()`'s H3_HIRES_DENOISE comment) and
        # `vae` is parked on CPU (bnb-4bit mode, outside its decode-phase window) --
        # `_execution_device` would resolve to `vae`'s CPU location the same way it did
        # for the layout_step bug this task found and fixed earlier in `generate()`. The
        # transformer is the one component guaranteed to be GPU-resident throughout the
        # whole denoise loop, so its device is used directly instead.
        device = components.transformer.device

        num_latent_frames = state.get("num_latent_frames")
        latent_height = state.get("latent_height")
        latent_width = state.get("latent_width")
        num_audio_latents = state.get("num_audio_latents")
        patch_size = components.patch_size
        vae_latent_channels = components.vae_latent_channels

        # 1. Reconstruct the x0 (denoised) estimate from the last pass-1 step, using the same
        # formula `MiniMaxH3Scheduler.step()` uses internally (see scheduling_minimax_h3.py):
        # `denoised = sample + (1 - t) * model_output`, i.e. `sample + sigma_from_timestep *
        # model_output`. `last_step_info["sample"]` is the *pre-step* video sample (x_t at the
        # last pass-1 timestep) and `last_step_info["noise_pred"]` is the velocity the model
        # predicted for it; both captured by `run_steps(..., capture_last=True)` in generate()
        # before the scheduler folded them into the next (already-stepped) `x_t`.
        last_sample = last_step_info["sample"]
        last_noise_pred = last_step_info["noise_pred"]
        last_t = last_step_info["t"]
        sigma_from_timestep = 1.0 - last_t
        x0_rows = last_sample.float() + sigma_from_timestep * last_noise_pred.float()

        # 2. Unpack the x0 rows into a 5D latent tensor.
        video_latent = unpatchify_video_tokens(
            x0_rows, num_latent_frames, latent_height, latent_width, vae_latent_channels, patch_size
        )

        # 3. F.interpolate the spatial (H, W) axes only -- temporal axis untouched. bilinear
        # (not nearest, not trilinear over T) per the task brief; align_corners=False is
        # torch's numerically-recommended default for this kind of resize (avoids the corner-
        # alignment bias nearest/bilinear-align_corners=True introduces). This is safe to do
        # on the x0 estimate (smooth, image-like content) in a way it was not on the noisy
        # `x_t` (see the docstring above).
        b, c, t_dim, h_dim, w_dim = video_latent.shape
        video_latent_2d = video_latent.permute(0, 2, 1, 3, 4).reshape(b * t_dim, c, h_dim, w_dim)
        video_latent_2d = torch.nn.functional.interpolate(
            video_latent_2d.float(), scale_factor=2, mode="bilinear", align_corners=False
        )
        new_h, new_w = video_latent_2d.shape[-2:]
        x0_upscaled = video_latent_2d.reshape(b, t_dim, c, new_h, new_w).permute(0, 2, 1, 3, 4)

        # 4. Re-patchify the upscaled x0 back into rows, draw fresh noise at the new (larger)
        # row count, and re-noise via the scheduler's own forward process
        # (`x_t = t*x0 + (1-t)*noise`, this repo's rectified-flow convention) at pass 2's
        # first timestep -- restoring proper `x_t` noise statistics for the model to continue
        # denoising from, instead of handing it a low-pass-filtered `x_t` it never would have
        # produced itself (root cause of the checkerboard corruption, see docstring).
        x0_upscaled_rows = patchify_video_latents(x0_upscaled.to(x0_rows.dtype), patch_size).to(device)
        pass2_start_t = float(state.get("timesteps")[pass1_steps])
        # `randn_tensor` (not raw torch.randn) so a CPU generator (the request's own,
        # `torch.Generator(device="cpu")` in generate()) works the same way it does for
        # every other noise draw in this pipeline (prepare_latents, keyframe_condition_noise
        # both use it for exactly this reason -- CUDA generators are not what the request
        # seed is defined against). Reuses the same generator object pass 1's/the initial
        # draw's noise came from, so this draw is deterministic per-request-seed but is a
        # *new, independent* sample from it (not a reuse of any earlier noise tensor).
        from diffusers.utils.torch_utils import randn_tensor

        fresh_noise = randn_tensor(
            x0_upscaled_rows.shape, generator=state.get("generator"), device=device, dtype=x0_upscaled_rows.dtype
        )
        block_state.latents = components.scheduler.scale_noise(x0_upscaled_rows, pass2_start_t, fresh_noise)

        # 5. Rebuild the packed layout at the new latent geometry (position_ids/token_tags/
        # indices all key off latent_height/latent_width -- see build_packed_sequence).
        # text_token_tags/num_audio_latents are unchanged (audio + text are untouched by the
        # spatial upscale), only the video row count and its rotary grid change. Calls
        # `build_packed_sequence` directly (the same function `MiniMaxH3PrepareLayoutStep.
        # __call__` calls internally) instead of going through the block, so `device` can
        # be passed explicitly instead of resolved via `components._execution_device`
        # (unsafe here -- see the NOTE at the top of this function).
        new_layout = build_packed_sequence(
            state.get("text_token_tags"),
            num_latent_frames,
            new_h,
            new_w,
            num_audio_latents,
            patch_size,
            (),  # keyframe_anchors: t2va only, enforced by the caller.
        )

        block_state.token_tags = new_layout.token_tags.to(device)
        block_state.position_ids = new_layout.position_ids.to(device)
        block_state.video_indices = new_layout.video_indices.to(device)
        block_state.audio_indices = new_layout.audio_indices.to(device)
        block_state.text_indices = new_layout.text_indices.to(device)
        # t2va only (enforced by the caller): no conditioning rows, so both stay 0.
        block_state.num_condition_video_rows = 0
        block_state.num_condition_audio_rows = 0

        # 6. Rebuild row_timestep_plan against the new (larger) sequence_length -- the old
        # plan was sized for the pass-1 sequence_length and would misindex if reused.
        # video_timesteps/audio_timesteps themselves are resolution-independent (the sigma
        # schedule does not depend on latent geometry), only their *row broadcast* does.
        #
        # `_predict_velocity` (denoise.py) indexes `block_state.row_timestep_plan[i]` with
        # the *absolute* step index (0..num_inference_steps-1), not a pass-relative one --
        # `run_steps()` in generate() keeps calling the loop blocks with the original `i`
        # across the pass-1/pass-2 splice. So this replaces the plan entries from `pass1_steps`
        # onward (pass 2's own steps) with plans built against `new_layout`, while the
        # earlier entries (never read again -- pass 2 only iterates i >= pass1_steps) are
        # left as-is, just to keep the list the same full length the denoiser indexes into.
        video_timesteps = state.get("timesteps")
        audio_timesteps = block_state.audio_timesteps
        old_plan = block_state.row_timestep_plan
        new_plan = list(old_plan)
        for i in range(pass1_steps, len(video_timesteps)):
            new_plan[i] = tuple(
                tensor.to(device)
                for tensor in build_row_timesteps(
                    new_layout,
                    float(video_timesteps[i]),
                    float(audio_timesteps[i]),
                    max(float(video_timesteps[i]), MINIMAX_H3_KEYFRAME_NOISE_AUG),
                    1.0,
                )
            )
        block_state.row_timestep_plan = new_plan

        # Update state's own latent_height/latent_width/layout too, in case anything reads
        # them again downstream (decode step reads latent_height/latent_width off `state`,
        # not `block_state` -- see the caller in generate()).
        state.set("latent_height", new_h)
        state.set("latent_width", new_w)
        state.set("layout", new_layout)
        return block_state

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        height: int = 768,
        width: int = 768,
        seconds: float = 5.0,
        num_inference_steps: int = 30,
        seed: int | None = None,
        image: Image.Image | None = None,
        last_image: Image.Image | None = None,
        progress: ProgressState | None = None,
        upscale: int = 0,
    ) -> dict:
        """
        Runs T2VA (image=None, last_image=None) or FL2VA (either/both given).

        `upscale=1` enables two-pass hires-fix: pass 1 denoises `round(num_inference_steps
        * (1 - H3_HIRES_DENOISE))` steps at the requested (height, width), the video latent's
        x0 estimate is then spatially upscaled 2x with `F.interpolate` and re-noised (audio
        latent is left untouched -- it has no spatial axes, see `_upscale_block_state_2x`
        docstring), and pass 2 continues the same sigma trajectory for the remaining steps
        at 2x resolution. The returned `height`/`width` reflect the actual (2x) output
        resolution in that case.

        Returns a dict with mp4_path, frame counts, timing and VRAM/RAM stats.
        """
        from diffusers.modular_pipelines.minimax_h3.before_denoise import (
            MiniMaxH3PrepareLatentsStep,
            MiniMaxH3PrepareLayoutStep,
            MiniMaxH3SetTimestepsStep,
        )
        from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3SetupStep
        from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (
            MiniMaxH3AutoKeyframeVaeEncoderStep,
        )
        from diffusers.modular_pipelines.minimax_h3.decoders import (
            MiniMaxH3AudioDecodeStep,
            MiniMaxH3VideoDecodeStep,
        )
        from diffusers.modular_pipelines.minimax_h3.denoise import (
            MiniMaxH3DenoiseStep,
            MiniMaxH3LoopDenoiser,
            MiniMaxH3LoopSchedulerStep,
        )
        from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3TextEncoderStep
        from diffusers.modular_pipelines.modular_pipeline import PipelineState

        t_start = time.time()
        num_frames = seconds_to_num_frames(seconds)
        do_upscale = bool(upscale)
        if do_upscale and (image is not None or last_image is not None):
            # Scope of this task's hires-fix is t2va only. fl2va's keyframe conditioning
            # rows are prepared once (at the requested resolution) before the loop and are
            # never denoised, only re-anchored into the packed sequence every step -- a
            # spatial upscale mid-loop would need those condition latents upscaled too and
            # their (fixed) rotary anchor position recomputed against the new geometry,
            # which is unverified territory this task did not have time to check against
            # the reference. Fail loudly rather than silently mis-render.
            raise ValueError("upscale=1 (hires-fix) is only supported for t2va requests, not fl2va.")
        if do_upscale and H3_LOWVRAM:
            # Not verified to fit: pass 2 runs full self-attention over a ~4x longer
            # packed sequence (~16x pass 1's attention activation cost), and this mode's
            # whole steady state is already sized to leave only ~9GB of headroom above
            # the int8 transformer alone (see H3_LOWVRAM's module comment) on a
            # 48GB-class card. Fail loudly rather than risk an OOM mid-request.
            raise ValueError("upscale=1 (hires-fix) is not supported with H3_LOWVRAM=1.")

        with self._load_lock:
            if H3_LOWVRAM:
                # This mode's whole point is TE (21GB) and transformer (34GB) are never
                # GPU-resident together (55GB already exceeds a 48GB-class card) -- so
                # unlike the branch below, do NOT call `_switch_to_variant`/
                # `_ensure_transformer` here: that would load the (int8) transformer
                # *before* TE, and TE has not even encoded the prompt yet. Just free
                # whichever big transformer happens to be resident (leftover from a
                # previous request -- lowvram's own steady state never leaves one
                # resident, but a mode-flag flip mid-process or a request that errored
                # out mid-denoise could) without loading a replacement; the transformer
                # is loaded further down, after TE has already finished encoding and
                # been freed again.
                self._free_transformer()
                self._free_transformer_ref()
                self._active_variant = None
                self._ensure_vaes(progress)
                self._load_text_encoder(progress)
            else:
                # Ensure `transformer` (not `transformer_ref`) is the GPU-resident big
                # model before anything else in this method touches it. A no-op when
                # t2va is already the active variant (the common case -- most requests
                # do not interleave with ref2va ones); when the previous request was a
                # ref2va one, this frees the ~66.3GB transformer_ref first. Must run
                # before `_load_text_encoder` below: in `none` mode that method's own
                # `_free_transformer()` call only knows about `transformer`, not
                # `transformer_ref`, so without this line a ref2va -> t2va switch in
                # `none` mode would try to hold transformer_ref(66.3) + TE(66.7) at once
                # and OOM.
                self._switch_to_variant("t2va", progress)
                # `none` mode: VAEs (permanent residents) + text encoder.
                # _load_text_encoder frees the transformer internally if it is resident
                # (TE 66GB + transformer 66GB cannot coexist in 96GB VRAM).
                # `bnb-4bit` mode: everything is already resident from preload_all()
                # except the VAEs, which are parked on CPU -- nothing to do here, they
                # get moved to GPU right before the phase that needs them, below.
                self._ensure_vaes(progress)
                self._load_text_encoder(progress)

        # Reset peak stats after loading so the reported peak reflects this
        # generation's encode+denoise+decode, not the (much larger, one-time) model
        # loading peak from a cold start.
        torch.cuda.reset_peak_memory_stats()

        pipe = self._pipe

        state = PipelineState()
        state.set("prompt", prompt)
        state.set("image", image)
        state.set("last_image", last_image)
        state.set("height", height)
        state.set("width", width)
        state.set("num_frames", num_frames)
        state.set("generator", torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None)
        state.set("num_inference_steps", num_inference_steps)
        state.set("output_type", "pt")
        state.set("attention_kwargs", None)
        state.set("latents", None)
        state.set("audio_latents", None)
        state.set("condition_latents", None)
        state.set("audio_condition_latents", None)

        is_fl2va = image is not None or last_image is not None
        if is_fl2va:
            # fl2va's keyframe VAE-encode step needs `vae` on GPU; bring it in now (no-op
            # in `none` mode, where it is already permanently resident).
            self._vae_to_gpu()

        # --- setup (canvas / frame count / keyframe prep) ---
        setup_step = MiniMaxH3SetupStep()
        _, state = setup_step(pipe, state)
        # The setup step's *outputs* (num_frames after alignment, prepared keyframes,
        # latent geometry) live in the PipelineState -- get_block_state only maps
        # declared inputs, so read outputs via state.get().
        actual_num_frames = state.get("num_frames")
        keyframes = state.get("keyframes")

        # --- text encode (still has text_encoder on GPU at this point) ---
        if progress:
            progress.update(phase="encoding", message="プロンプトをエンコード中...")
        # encode_prompt is a bare staticmethod; the @torch.no_grad() lives on the block
        # __call__ we bypass. Without no_grad the autograd graph pins ~50GB of TE
        # weights on GPU past the free below (observed on the first probe run).
        with torch.no_grad():
            prompt_embeds, text_token_tags = MiniMaxH3TextEncoderStep.encode_prompt(
                pipe, prompt, keyframes or None, device=DEVICE, dtype=torch.bfloat16
            )
        state.set("prompt_embeds", prompt_embeds)
        state.set("text_token_tags", text_token_tags)

        # upscale (hires-fix) requests: force-free the TE-nf4 even in bnb-4bit mode.
        # Pass 2 runs full self-attention over a ~4x longer packed sequence (2x spatial ->
        # 4x video rows), i.e. ~16x the attention activation cost of pass 1 -- bnb-4bit's
        # normal 87.7GB steady state (transformer 66.3GB + TE-nf4 21.0GB) only leaves
        # ~4-8GB of headroom (measured 91.7GB peak at 768x768, see README), nowhere near
        # enough for that. Freeing TE-nf4 here (and reloading it after decode, in the
        # decode section below) is the same one-way "short window" pattern the transformer
        # already uses around the decode step in this mode -- not a standing swap.
        #
        # int8 both-resident mode (`H3_TRANSFORMER_BOTH_RESIDENT`): force-free TE-nf4
        # here too, but ONLY when `transformer_ref` also happens to be resident right
        # now (i.e. a ref2va request has run at some point in this process's life).
        # Reproduced by this task's own verification, exactly the OOM this comment
        # predicts: transformer(34) + transformer_ref(34) + TE-nf4(21) = ~89GB measured
        # as 90.84GB allocated (steady state, see `status()`'s `allocated_gb`) left only
        # ~4-6GB of headroom, and t2va's own denoise activations (measured ~4.9GB peak
        # over the transformer+TE-only 55GB baseline in this same task's test 1, i.e.
        # the *same* activation footprint t2va always had) pushed it over: "Tried to
        # allocate 1.16 GiB" with 92.05GB already in use, ~1.2GB free. When
        # `transformer_ref` is NOT resident (fresh process, or this process has never
        # served a ref2va request yet), this is unnecessary churn -- t2va's own resident
        # set is just transformer(34) + TE-nf4(21) = 55GB, the same safe budget it always
        # ran at before this task (see test 1's 59.71GB peak, well under 95.6GB).
        # Reloaded after decode, below -- same "restore the steady state for the next
        # request" shape the pre-existing `do_upscale` force_free_te reload uses.
        #
        # IMPORTANT: this free is deliberately deferred until *after* layout_step/
        # latents_step/timesteps_step below, not done here alongside the transformer load.
        # `MiniMaxH3ModularPipeline._execution_device` (used by all three of those blocks)
        # resolves to the device of the *first* `nn.Module` in `self.components` insertion
        # order (`text_encoder, tokenizer, processor, vae, scheduler, audio_scheduler,
        # transformer, ...`) that is actually still set. Freeing text_encoder here would
        # make `vae` (parked on CPU in bnb-4bit mode outside its active phase) the new
        # first hit, silently resolving `_execution_device` to `cpu` and producing a
        # cuda/cpu device-mismatch inside the transformer's rope() -- reproduced and
        # confirmed by traceback during this task's own verification run. Freeing TE only
        # once those position_ids/layout tensors already exist on the correct device (set
        # once, from the layout step, and never touched again) sidesteps the whole
        # resolution question for the rest of the request.
        # H3_LOWVRAM: TE is force-freed unconditionally, but -- same
        # `_execution_device` resolution trap the comment above describes -- this
        # cannot happen until *after* layout_step/latents_step/timesteps_step have run
        # (see the dedicated H3_LOWVRAM branch below, which runs those three steps
        # *before* freeing TE/loading the transformer, unlike every other branch here,
        # which loads its big transformer up front and only then runs those steps).
        # `vae` sits between `text_encoder` and `transformer` in this pipe's own
        # component order (`text_encoder, tokenizer, processor, vae, scheduler,
        # audio_scheduler, transformer, ...`), and it is a resident `nn.Module`
        # (just CPU-placed, not freed) throughout t2va in this mode -- so simply
        # loading the transformer first would NOT fix this the way it does for
        # `none`/plain `bnb-4bit` mode: `_execution_device` would still resolve to
        # `vae`'s CPU location the instant TE is freed, `transformer` never being
        # reached in the scan. Reproduced by this task's own verification (t2va OOM'd
        # -- no, worse, silently produced a device-mismatch `RuntimeError` deep inside
        # the transformer's own forward, not caught until the first denoise step) the
        # first time this branch tried to free TE right before `_ensure_transformer`,
        # mirroring `none` mode's own ordering naively.
        force_free_te = TE_QUANT == "bnb-4bit" and not H3_LOWVRAM and (
            do_upscale or (H3_TRANSFORMER_BOTH_RESIDENT and self._transformer_ref_loaded)
        )

        if H3_LOWVRAM:
            # fl2va's keyframe step (if any) runs here too, while TE is still resident
            # (harmless -- it only touches vae/scheduler, not TE) and `vae` is already
            # on GPU from the `is_fl2va` block above this function's setup section.
            keyframe_step = MiniMaxH3AutoKeyframeVaeEncoderStep()
            _, state = keyframe_step(pipe, state)
            if is_fl2va:
                self._vae_to_cpu()

            # --- layout / latents / timesteps, run NOW (TE still GPU-resident) ---
            # `_execution_device` resolves via `text_encoder` (still resident on GPU)
            # here, exactly like every non-lowvram bnb-4bit branch's own
            # `force_free_te`-deferred ordering achieves -- see the long comment above.
            layout_step = MiniMaxH3PrepareLayoutStep()
            _, state = layout_step(pipe, state)
            latents_step = MiniMaxH3PrepareLatentsStep()
            _, state = latents_step(pipe, state)
            timesteps_step = MiniMaxH3SetTimestepsStep()
            _, state = timesteps_step(pipe, state)

            # Only now is it safe to free TE and load the (int8) transformer: every
            # tensor that would have needed `_execution_device` to resolve correctly
            # already exists, materialized on the right device, on `state`.
            with self._load_lock:
                self._free_text_encoder(force=True)
                self._ensure_transformer(progress)
        elif TE_QUANT == "bnb-4bit" and is_fl2va:
            # bnb-4bit + fl2va only: transformer(66.3) + TE-nf4(21.0) + vae pair(11.0)
            # already sums to ~98.3GB before any activation buffer, over this card's
            # ~95.6GB (the same three-way conflict measured for decode, see the decode
            # section below and the module docstring) -- so the keyframe VAE-encode step
            # (which needs `vae` on GPU, already brought in above) has to run *before*
            # the transformer is loaded, not after. TE stays resident throughout (it is
            # not involved in this step); fl2va + upscale is rejected earlier in this
            # function, so force_free_te is always False on this branch.
            keyframe_step = MiniMaxH3AutoKeyframeVaeEncoderStep()
            _, state = keyframe_step(pipe, state)
            self._vae_to_cpu()
            with self._load_lock:
                self._ensure_transformer(progress)

            # --- layout / latents / timesteps ---
            layout_step = MiniMaxH3PrepareLayoutStep()
            _, state = layout_step(pipe, state)
            latents_step = MiniMaxH3PrepareLatentsStep()
            _, state = latents_step(pipe, state)
            timesteps_step = MiniMaxH3SetTimestepsStep()
            _, state = timesteps_step(pipe, state)
        else:
            # `none` mode: TE's job is done for this request -- free it and bring in the
            # transformer (which stays resident until the next request's encode phase
            # kicks it out again).
            # `bnb-4bit` + t2va: TE is normally permanently resident and the transformer
            # is normally already resident too -- except right after a previous request's
            # decode phase dropped it (see the decode section below), in which case this
            # is the reload that restores it before denoise. No vae conflict here since
            # t2va's vae never went to GPU in the first place.
            with self._load_lock:
                # `none` mode always frees here (force is irrelevant -- _free_text_encoder
                # frees unconditionally when TE_QUANT != "bnb-4bit"); `bnb-4bit` mode's
                # force-free (force_free_te) is deferred past layout/latents/timesteps
                # below, see the comment above, so this call is a no-op for it here.
                self._free_text_encoder()
                self._ensure_transformer(progress)

            # --- keyframe VAE conditioning (fl2va + `none` mode only; vae already
            # permanently resident in `none` mode) ---
            keyframe_step = MiniMaxH3AutoKeyframeVaeEncoderStep()
            _, state = keyframe_step(pipe, state)

            # --- layout / latents / timesteps ---
            layout_step = MiniMaxH3PrepareLayoutStep()
            _, state = layout_step(pipe, state)
            latents_step = MiniMaxH3PrepareLatentsStep()
            _, state = latents_step(pipe, state)
            timesteps_step = MiniMaxH3SetTimestepsStep()
            _, state = timesteps_step(pipe, state)

        # --- denoise loop, instrumented for progress polling ---
        if progress:
            progress.update(phase="denoising", step=0, total_steps=num_inference_steps, message="デノイズ中...")
        t_denoise = time.time()
        step_times = []
        cache_skips = [0]
        pass1_time = None
        interpolate_time = None
        pass2_time = None
        # Canonical (post-setup) resolution -- MiniMaxH3SetupStep resolves `None` and
        # snaps to the canvas rules, so this is not necessarily identical to the raw
        # `height`/`width` args.
        out_height, out_width = state.get("height"), state.get("width")

        def _fbc_reset_and_context():
            # Same reasoning as the single-pass path below: per-request/per-pass reset is
            # required so a stale residual from a previous call (previous request, or
            # pass 1 of *this* request) cannot make step 0 of the new call wrongly skip.
            self._pipe.transformer._reset_stateful_cache()
            return self._pipe.transformer.cache_context("h3")

        if force_free_te and not do_upscale:
            # int8 both-resident mode only (the only way `force_free_te` can be True
            # here -- `do_upscale` always takes the hires-fix branch below, which has
            # its own force_free_te handling already). Safe to free now for the same
            # reason the hires-fix branch's own comment gives: layout_step/latents_step/
            # timesteps_step have already run above and their outputs are already
            # materialized as tensors on `state`, so `_execution_device` resolution is
            # no longer touched by freeing text_encoder from here on.
            with self._load_lock:
                self._free_text_encoder(force=True)

        if not do_upscale:
            denoise_step = MiniMaxH3DenoiseStep()
            orig_loop_step = denoise_step.loop_step

            def timed_loop_step(components, bstate, i, t):
                ts = time.time()
                result = orig_loop_step(components, bstate, i=i, t=t)
                step_times.append(time.time() - ts)
                if H3_CACHE == "fbc":
                    cache_skips[0] += self._fbc_last_step_was_skip()
                if progress:
                    progress.update(step=i + 1, message=f"デノイズ中 {i + 1}/{num_inference_steps}")
                return result

            denoise_step.loop_step = timed_loop_step
            if H3_CACHE == "fbc":
                # Per-request reset: `FirstBlockCache`'s hooks are stateful (cached head-block
                # residual/output + tail-block residuals persist on the transformer submodules
                # between calls, see FBCSharedBlockState in first_block_cache.py). Without this
                # reset, the *first* denoise step of this request would see the *previous*
                # request's leftover `head_block_residual` from its own final step and could
                # incorrectly decide to skip computation on step 0 (which should always compute,
                # since there is no prior-step residual within this request to compare against).
                # `_reset_stateful_cache()` -> `HookRegistry.reset_stateful_hooks()` ->
                # `FBCHeadBlockHook.reset_state()` -> `StateManager.reset()`, which empties the
                # per-context state cache entirely (a fresh `FBCSharedBlockState()` is created on
                # the next `get_state()`), not just the partial fields `FBCSharedBlockState.reset()`
                # touches -- so this clears `head_block_output`/`head_block_residual` too, not only
                # `tail_block_residuals`/`should_compute`.
                self._pipe.transformer._reset_stateful_cache()
                # `cache_context(...)` is required, not optional: `StateManager.get_state()` raises
                # `ValueError("No context is set...")` if no context has been entered, so the very
                # first transformer forward of this request would crash without it. H3 is
                # guidance-distilled (no CFG, no cond/uncond branches -- confirmed in
                # modular_pipelines/minimax_h3/denoise.py and encoders.py docstrings), so unlike
                # Wan/Flux's per-branch "cond"/"uncond" contexts there is only one branch here; a
                # single fixed context name for the whole request's denoise loop is correct.
                with self._pipe.transformer.cache_context("h3"):
                    _, state = denoise_step(pipe, state)
            else:
                _, state = denoise_step(pipe, state)
        else:
            # --- two-pass hires-fix ---
            # This bypasses `MiniMaxH3DenoiseStep.__call__` (which owns the whole
            # `for i, t in enumerate(timesteps)` loop internally) and instead drives the
            # per-step sub-blocks (`MiniMaxH3LoopDenoiser`, `MiniMaxH3LoopSchedulerStep`)
            # directly through one shared `BlockState`, so a resolution change (new
            # layout/position_ids/row_timestep_plan) can be spliced in mid-loop while the
            # scheduler's internal `_step_index` keeps incrementing across the splice --
            # see the module-level H3_HIRES_DENOISE docstring for why this needs no
            # separate renoise/DisableNoise step, unlike the ComfyUI reference node this
            # was modeled after (which has to cross a KSamplerAdvanced node boundary and
            # therefore re-injects noise at the pass-2 starting sigma instead).
            denoiser_block = MiniMaxH3LoopDenoiser()
            scheduler_block = MiniMaxH3LoopSchedulerStep()
            denoise_wrapper = MiniMaxH3DenoiseStep()  # only used for get/set_block_state plumbing
            block_state = denoise_wrapper.get_block_state(state)

            if force_free_te:
                # Safe to free now: layout_step/latents_step/timesteps_step (which all
                # depend on `components._execution_device` resolving correctly, see the
                # long comment above) have already run and their outputs are already
                # materialized as tensors on `state`/`block_state`. Nothing from here to
                # the end of the denoise loop touches `components._execution_device`
                # again except the transformer's own forward (which resolves its device
                # from its own parameters, not from pipe-level component scanning).
                with self._load_lock:
                    self._free_text_encoder(force=True)

            timesteps = state.get("timesteps")
            # `MiniMaxH3Scheduler.set_timesteps()` builds a sigma grid of
            # `num_inference_steps` points *including* the terminal 0, then exposes
            # `self.timesteps = 1 - sigmas[:-1]` -- i.e. `len(timesteps) ==
            # num_inference_steps - 1` model evaluations, one fewer than the requested
            # step count (confirmed against scheduling_minimax_h3.py). The single-pass
            # path never has to know this (it just does `for i, t in
            # enumerate(block_state.timesteps)`), but this loop's bounds are computed
            # from `num_inference_steps` directly, so it must use `len(timesteps)`, not
            # `num_inference_steps`, or the last step indexes past the end (reproduced:
            # "IndexError: index 29 is out of bounds for dimension 0 with size 29" when
            # this used the raw request value of 30 as the pass-2 end bound).
            actual_steps = len(timesteps)
            n1 = max(1, min(actual_steps - 1, round(actual_steps * (1.0 - H3_HIRES_DENOISE))))
            logger.info(
                "hires-fix: %d model evaluations, pass1=%d steps @ %dx%d, pass2=%d steps @ %dx%d "
                "(H3_HIRES_DENOISE=%s)",
                actual_steps, n1, out_width, out_height, actual_steps - n1, out_width * 2, out_height * 2,
                H3_HIRES_DENOISE,
            )

            # Populated by run_steps() with the *last* step's pre-step video sample and
            # predicted velocity, so the caller can reconstruct an x0 estimate for the
            # hires splice (see the long comment above _upscale_block_state_2x's call
            # site below for why this is needed instead of upscaling the noisy x_t
            # directly).
            last_step_info = {}

            def run_steps(bstate, i_start, i_end, phase_label, capture_last=False):
                # `MiniMaxH3LoopDenoiser`/`MiniMaxH3LoopSchedulerStep.__call__` both mutate
                # and return the *same* `BlockState` object (see `BlockState.__setitem__` /
                # the plain `setattr` pattern every block writes its outputs through) -- so
                # reassigning `bstate` here every iteration is just documenting that fact,
                # not actually swapping to a different object.
                #
                # `num_condition_video_rows` is always 0 here (t2va only, enforced earlier
                # in generate()), so `bstate.latents`/`bstate.noise_pred[0]` are entirely
                # generated video rows with no conditioning-row prefix to skip.
                fbc_cm = _fbc_reset_and_context() if H3_CACHE == "fbc" else None
                cm = fbc_cm if fbc_cm is not None else _NullContext()
                with cm:
                    for i in range(i_start, i_end):
                        t = timesteps[i]
                        ts = time.time()
                        pre_step_video_sample = bstate.latents.clone() if (capture_last and i == i_end - 1) else None
                        _, bstate = denoiser_block(pipe, bstate, i=i, t=t)
                        if capture_last and i == i_end - 1:
                            last_step_info["sample"] = pre_step_video_sample
                            last_step_info["noise_pred"] = bstate.noise_pred[0].clone()
                            last_step_info["t"] = float(t)
                        _, bstate = scheduler_block(pipe, bstate, i=i, t=t)
                        step_times.append(time.time() - ts)
                        if H3_CACHE == "fbc":
                            cache_skips[0] += self._fbc_last_step_was_skip()
                        if progress:
                            progress.update(
                                step=i + 1,
                                message=f"デノイズ中 {phase_label} {i + 1}/{actual_steps}",
                            )
                return bstate

            t_pass1 = time.time()
            block_state = run_steps(block_state, 0, n1, "pass1", capture_last=True)
            pass1_time = time.time() - t_pass1

            # --- spatial 2x upscale of the video latent between passes ---
            if progress:
                progress.update(message="潜在空間を2xアップスケール中...")
            t_interp = time.time()
            block_state = self._upscale_block_state_2x(
                components=pipe, block_state=block_state, state=state, pass1_steps=n1,
                last_step_info=last_step_info,
            )
            interpolate_time = time.time() - t_interp
            out_height, out_width = out_height * 2, out_width * 2

            t_pass2 = time.time()
            block_state = run_steps(block_state, n1, actual_steps, "pass2")
            pass2_time = time.time() - t_pass2

            denoise_wrapper.set_block_state(state, block_state)
        denoise_time = time.time() - t_denoise

        # --- decode ---
        if progress:
            progress.update(phase="decoding", message="動画/音声をデコード中...")
        # bnb-4bit mode: transformer(66.3GB) + TE-nf4(~21GB) + vae pair(11GB) = ~98.5GB
        # already exceeds this card's ~95.6GB before any decode activation buffers are
        # even counted (measured: an attempt to keep all three resident OOM'd during
        # decode, "Tried to allocate 30.00 MiB" with the allocator already at 93.7GB).
        # The transformer is not used by either decode step (MiniMaxH3VideoDecodeStep /
        # MiniMaxH3AudioDecodeStep only touch vae/audio_vae/video_processor), so it is
        # the thing that gives here: drop it for this short (~9s) window, then reload it
        # right after so the steady state between requests is unchanged. This is the
        # same bounded "short window" pattern as the `none` mode's per-request TE/
        # transformer cycle, just applied to the transformer around decode instead of
        # around encode. `none` mode does not need this at all -- its vae is already
        # permanently resident and its transformer/TE never coexist in the first place,
        # so dropping the transformer here would only add pointless reload churn.
        if TE_QUANT == "bnb-4bit":
            self._free_transformer()
        self._vae_to_gpu()
        t_decode = time.time()
        video_decode_step = MiniMaxH3VideoDecodeStep()
        _, state = video_decode_step(pipe, state)
        audio_decode_step = MiniMaxH3AudioDecodeStep()
        _, state = audio_decode_step(pipe, state)
        decode_time = time.time() - t_decode

        videos = state.get("videos")
        audio = state.get("audio")
        sampling_rate = state.get("sampling_rate")

        video_tensor = videos[0] if isinstance(videos, list) else videos
        if video_tensor.dim() == 5:
            video_tensor = video_tensor[0]
        frames_uint8 = (
            (video_tensor.permute(0, 2, 3, 1).float().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        )
        audio_np = audio[0].float().cpu().numpy()
        rms = float(np.sqrt(np.mean(audio_np**2)))
        peak = float(np.max(np.abs(audio_np)))

        peak_vram = torch.cuda.max_memory_allocated() / 1e9

        # free the big activation buffers before muxing (CPU-bound, no need to hold onto GPU tensors)
        del video_tensor, videos, audio
        gc.collect()
        torch.cuda.empty_cache()

        # bnb-4bit mode: decode is done and frames/audio are already off-GPU (numpy
        # above) -- park the VAEs back on CPU, then reload the transformer that was
        # dropped for the decode window, restoring the transformer+TE-nf4 steady state
        # this mode keeps between requests. No-op in `none` mode (nothing was dropped
        # for decode in that mode).
        self._vae_to_cpu()
        if TE_QUANT == "bnb-4bit" and not H3_LOWVRAM:
            with self._load_lock:
                self._ensure_transformer(progress)
                if force_free_te:
                    # Restore the bnb-4bit steady state (transformer + TE-nf4 both
                    # resident) for the *next* request -- this request force-freed TE-nf4
                    # after encoding to make room for pass 2's activations (see above).
                    # Reloaded after the transformer so the transformer's own reload above
                    # (which needs headroom too, right after decode's own VAE trip) is not
                    # competing with a simultaneous TE reload for VRAM.
                    self._load_text_encoder(progress)
        # H3_LOWVRAM: deliberately do NOT reload the transformer here. This mode's
        # steady state between requests is "nothing big resident" (see the H3_LOWVRAM
        # module comment) -- the *next* request needs TE first, not transformer, so
        # preloading it now would just be evicted again at that request's own encode
        # phase for no benefit, and would leave a 34GB resident model sitting idle
        # between requests on a card that cannot spare it.

        if progress:
            progress.update(phase="muxing", message="mp4へmux中...")
        mode = "fl2va" if (image is not None or last_image is not None) else "t2va"
        job_stub = f"{mode}_{int(t_start)}"
        mp4_path = self.output_dir / f"{job_stub}.mp4"
        _mux_mp4(frames_uint8, audio_np, sampling_rate, FPS, mp4_path)

        result = {
            "prompt": prompt,
            "height": out_height,
            "width": out_width,
            "num_frames_requested_seconds": seconds,
            "num_frames": actual_num_frames,
            "duration_s": actual_num_frames / FPS,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "denoise_time_s": round(denoise_time, 2),
            "decode_time_s": round(decode_time, 2),
            "avg_step_time_s": round(sum(step_times) / len(step_times), 3) if step_times else None,
            "peak_vram_gb": round(peak_vram, 2),
            "ram": ram_gb(),
            "audio_rms": rms,
            "audio_peak": peak,
            "audio_sampling_rate": sampling_rate,
            "mp4_path": str(mp4_path),
            "mp4_filename": mp4_path.name,
            "total_elapsed_s": round(time.time() - t_start, 2),
            "mode": mode,
            "te_quant": TE_QUANT,
            "transformer_quant": H3_TRANSFORMER_QUANT,
            "lowvram": H3_LOWVRAM,
            "attn_backend": H3_ATTN_BACKEND or "default",
            "cache_mode": H3_CACHE,
            "cache_threshold": H3_CACHE_THRESHOLD if H3_CACHE == "fbc" else None,
            "upscale": int(do_upscale),
            "hires_denoise": H3_HIRES_DENOISE if do_upscale else None,
            "pass1_steps": n1 if do_upscale else None,
            "pass2_steps": (actual_steps - n1) if do_upscale else None,
            "pass1_time_s": round(pass1_time, 2) if pass1_time is not None else None,
            "interpolate_time_s": round(interpolate_time, 3) if interpolate_time is not None else None,
            "pass2_time_s": round(pass2_time, 2) if pass2_time is not None else None,
            # Number of denoise steps where FBC skipped the tail blocks (cache hit).
            # Always 0 in `none` mode (the counter never increments there).
            "cache_skipped_steps": cache_skips[0] if H3_CACHE == "fbc" else None,
        }
        if progress:
            progress.update(phase="done", message="完了", result_path=str(mp4_path))
        logger.info("generation done: %s", json.dumps({k: v for k, v in result.items() if k != "ram"}, ensure_ascii=False))
        return result

    # ------------------------------------------------------------------
    # ref2va (omni-reference) generation
    # ------------------------------------------------------------------
    def generate_ref2va(
        self,
        prompt: str,
        references: list,
        height: int | None = None,
        width: int | None = None,
        seconds: float | None = None,
        num_inference_steps: int = 30,
        seed: int | None = None,
        progress: ProgressState | None = None,
    ) -> dict:
        """
        Runs ref2va: joint video+audio generation conditioned on an ordered list of
        `MiniMaxH3Reference` images/videos/audio clips (up to 9/3/3, 12 total).

        `seconds=None` is only valid when `references` carries exactly one audio-bearing
        reference (a lone audio reference, or a video reference with a soundtrack) -- the
        generated duration is then that reference's own, per
        `MiniMaxH3Ref2VASetupStep.prepare_references`. `height`/`width` default to
        MiniMax-H3's own 16:9 canvas when left out (references never bind the target
        geometry -- each is prepared at its own resolution, see packing_ref2va.py's
        module docstring).

        Mirrors `generate()`'s structure closely (same FBC instrumentation, same
        bnb-4bit-mode decode-window transformer drop/reload pattern), but against
        `self._pipe_ref` / `transformer_ref` and the ref2va block set. Does not support
        `upscale` (hires-fix) -- out of scope for this task, and `_upscale_block_state_2x`
        assumes t2va's `num_condition_video_rows == 0`, which is never true here (a
        reference always adds condition rows).

        Returns a dict with mp4_path, frame counts, timing and VRAM/RAM stats, in the
        same shape `generate()` returns (plus `references_summary`).
        """
        from diffusers.modular_pipelines.minimax_h3.before_denoise import (
            MiniMaxH3PrepareLatentsStep,
            MiniMaxH3Ref2VAPrepareLayoutStep,
            MiniMaxH3SetTimestepsStep,
        )
        from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
        from diffusers.modular_pipelines.minimax_h3.decoders import (
            MiniMaxH3AudioDecodeStep,
            MiniMaxH3VideoDecodeStep,
        )
        from diffusers.modular_pipelines.minimax_h3.denoise import (
            MiniMaxH3Ref2VADenoiseStep,
            MiniMaxH3Ref2VALoopDenoiser,
            MiniMaxH3LoopSchedulerStep,
        )
        from diffusers.modular_pipelines.minimax_h3.encoders import (
            MiniMaxH3Ref2VAReferenceEncoderStep,
            MiniMaxH3Ref2VATextEncoderStep,
        )
        from diffusers.modular_pipelines.minimax_h3.packing_ref2va import reference_kind
        from diffusers.modular_pipelines.modular_pipeline import PipelineState

        t_start = time.time()
        if not references:
            raise ValueError("ref2va needs at least one reference; use generate() for text-only requests.")
        kinds = [reference_kind(index, entry) for index, entry in enumerate(references)]
        if set(kinds) == {"audio"}:
            raise ValueError(
                "An audio reference has to be paired with at least one image or video reference and cannot be "
                "used on its own."
            )
        num_frames = None if seconds is None else seconds_to_num_frames(seconds)

        with self._load_lock:
            # Free `transformer` (t2va's, if resident) now, but do NOT load
            # `transformer_ref` yet -- unlike generate()'s t2va entry, ref2va's own
            # reference-encoder step (below) needs `vae`/`audio_vae` on GPU *before*
            # transformer_ref is loaded: transformer_ref(66.3) + TE-nf4(21.0) + vae
            # pair(11.0) already exceeds this card's ~95.6GB (identical three-way
            # conflict to fl2va's own keyframe-encode-vs-transformer-load ordering, see
            # generate()'s comment on it -- reproduced here on the very first ref2va
            # request tried during this task: transformer_ref loaded eagerly at this
            # point OOM'd 8s into `vae._encode_clip()` with "Tried to allocate 98.00
            # MiB" at 93GB already in use). `transformer_ref` is loaded further down,
            # after the reference encoder step and (in bnb-4bit mode) after the vae
            # pair is parked back on CPU -- the same ordering `generate()` uses for
            # fl2va's keyframe step vs. transformer, just against the ref2va pair.
            #
            # int8 both-resident mode (`H3_TRANSFORMER_BOTH_RESIDENT`): this is a no-op
            # (see `_free_other_variant_transformer`'s docstring) -- `transformer` stays
            # resident (~34GB) through the reference-encode step below too. Even so, the
            # VAE-pair headroom conflict this comment describes still applies with BOTH
            # transformers resident: transformer(34) + transformer_ref(34, if resident
            # from steady state) + TE-nf4(21) + vae pair(11) = ~100GB, over this card's
            # ~95.6GB. See the `H3_TRANSFORMER_BOTH_RESIDENT` branch just below, which
            # frees only `transformer` (t2va's, the variant NOT being served by this
            # request) for this step instead of `transformer_ref` -- keeping ref2va's own
            # transformer_ref resident across the whole request (and across repeated
            # ref2va requests), which is the actual switch-elimination this mode exists
            # for. `transformer` is reloaded later, in the decode section below (see its
            # comment there for why the reload is deferred that far rather than done
            # right after the reference-encode step).
            if H3_TRANSFORMER_BOTH_RESIDENT:
                self._free_transformer()
            else:
                self._free_other_variant_transformer("ref2va")
                # Also free transformer_ref itself unconditionally, even though this is
                # the ref2va variant's *own* transformer: unlike the very first ref2va
                # request (where it is never loaded yet), a *second* (or later) ref2va
                # request in a row finds it already GPU-resident -- `_ensure_transformer_ref`
                # at the end of the *previous* request's decode section restores the
                # transformer_ref+TE-nf4 steady state between requests, the same way
                # `generate()`'s own `transformer` stays resident between t2va requests.
                # Reproduced during this task's own verification: a second ref2va
                # request's `_vae_to_gpu()` (below, via the reference encoder step)
                # logged `allocated_gb: 98.81` (transformer_ref 66.3 + TE 21-ish + vae
                # pair 11.0 all at once) and OOM'd on the first VAE conv. It is reloaded
                # fresh, later, after the reference encoder step -- same as the first-
                # request path. No-op (cheap) when it was not resident.
                self._free_transformer_ref()
            self._ensure_vaes(progress)
            self._load_text_encoder(progress)
            # H3_LOWVRAM bug found and fixed by this task's own verification: syncing
            # shared components (text_encoder among them) onto `self._pipe_ref` must
            # happen AFTER `_load_text_encoder` above, not before. `_sync_shared_
            # components_to_ref()` copies whatever `self._pipe.text_encoder` *currently*
            # is at the moment it runs (`ModularPipeline.components` is a live
            # attribute read, not a promise) -- in every non-lowvram mode this was
            # always safe because TE is already resident (permanently, or reloaded from
            # a previous request's steady state) by the time `generate_ref2va()` is
            # entered, so syncing before vs. after `_load_text_encoder` made no
            # observable difference. H3_LOWVRAM never preloads TE (see H3_LOWVRAM's
            # module comment), so the old ordering synced `self._pipe.text_encoder ==
            # None` onto `self._pipe_ref`, and the freshly loaded TE a few lines below
            # was never propagated -- reproduced as `AttributeError: 'NoneType' object
            # has no attribute 'config'` inside `MiniMaxH3Ref2VATextEncoderStep.
            # encode_prompt` (`components.text_encoder.config...`) on this task's first
            # ref2va-under-lowvram attempt. Calling this again on every request is
            # cheap and always safe (plain attribute re-assignment of already-loaded
            # modules, see the field comment on `_pipe_ref` in `__init__`).
            self._sync_shared_components_to_ref()

        # Reset peak stats after loading so the reported peak reflects this generation's
        # encode+denoise+decode, not the (much larger, one-time) model loading peak.
        torch.cuda.reset_peak_memory_stats()

        pipe = self._pipe_ref

        state = PipelineState()
        state.set("prompt", prompt)
        state.set("references", references)
        state.set("height", height)
        state.set("width", width)
        state.set("num_frames", num_frames)
        state.set("generator", torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None)
        state.set("num_inference_steps", num_inference_steps)
        state.set("output_type", "pt")
        state.set("attention_kwargs", None)
        state.set("latents", None)
        state.set("audio_latents", None)

        # --- setup (canvas / frame count / reference prep) ---
        # Reference images/videos/audio are decoded and resized here (each at its own
        # resolution -- see packing_ref2va.py's module docstring), and the frame count is
        # resolved from a lone audio-bearing reference when `num_frames` was left None.
        setup_step = MiniMaxH3Ref2VASetupStep()
        _, state = setup_step(pipe, state)
        actual_num_frames = state.get("num_frames")

        # --- text encode (references' vision blocks + prompt; still has TE on GPU) ---
        if progress:
            progress.update(phase="encoding", message="プロンプト+参照をエンコード中...")
        with torch.no_grad():
            prompt_embeds, text_token_tags = MiniMaxH3Ref2VATextEncoderStep.encode_prompt(
                pipe, prompt, state.get("prepared_references"), device=DEVICE, dtype=torch.bfloat16
            )
        state.set("prompt_embeds", prompt_embeds)
        state.set("text_token_tags", text_token_tags)

        # --- reference VAE encoding (image/video refs through vae, soundtracks through
        # audio_vae) -- this is ref2va's analogue of fl2va's keyframe step, and needs the
        # same "vae on GPU before transformer_ref is loaded" ordering in bnb-4bit mode:
        # transformer_ref(66.3) + TE-nf4(21.0) + vae pair(11.0) would be ~98.3GB resident
        # at once otherwise, over this card's ~95.6GB (identical three-way conflict to
        # fl2va's, see generate()'s own comment on this). transformer_ref was already
        # unconditionally freed above (before `_sync_shared_components_to_ref`/
        # `_ensure_vaes`/`_load_text_encoder`), including the "already resident from a
        # previous ref2va request's steady state" case -- see that comment for the bug
        # this closes. Nothing more to free here; just bring vae onto GPU.
        self._vae_to_gpu()
        if H3_LOWVRAM:
            # Same `_execution_device` resolution trap as generate()'s own H3_LOWVRAM
            # branch (see its long comment): `vae` sits between `text_encoder` and
            # `transformer_ref` in `MiniMaxH3Ref2VABlocks`' component order, and stays a
            # resident (if CPU-placed) `nn.Module` even outside its active phase -- so
            # freeing TE before transformer_ref is loaded is only safe once every step
            # that resolves its device via `_execution_device` has already run and
            # materialized its tensors. Unlike the non-lowvram int8 branch below (which
            # tolerates TE-nf4(21) + transformer_ref-int8(34) = 55GB coexisting briefly
            # during the transformer_ref load, then frees TE right after via the
            # deferred `force_free_te` further down), 55GB already exceeds a
            # 48GB-class card -- so here the reference-encoder step AND
            # layout_step/latents_step/timesteps_step all run first, while TE is still
            # the GPU-resident model `_execution_device` resolves to, and only then is
            # TE freed and transformer_ref loaded.
            reference_encoder_step = MiniMaxH3Ref2VAReferenceEncoderStep()
            _, state = reference_encoder_step(pipe, state)
            self._vae_to_cpu()

            layout_step = MiniMaxH3Ref2VAPrepareLayoutStep()
            _, state = layout_step(pipe, state)
            latents_step = MiniMaxH3PrepareLatentsStep()
            _, state = latents_step(pipe, state)
            timesteps_step = MiniMaxH3SetTimestepsStep()
            _, state = timesteps_step(pipe, state)

            with self._load_lock:
                self._free_text_encoder(force=True)
                self._ensure_transformer_ref(progress)
        elif TE_QUANT == "bnb-4bit":
            reference_encoder_step = MiniMaxH3Ref2VAReferenceEncoderStep()
            _, state = reference_encoder_step(pipe, state)
            self._vae_to_cpu()
            with self._load_lock:
                self._ensure_transformer_ref(progress)
                # NOTE: `transformer` (t2va's, freed at this method's entry in
                # H3_TRANSFORMER_BOTH_RESIDENT mode) is deliberately NOT reloaded here.
                # ref2va's denoise loop already runs a longer packed sequence than t2va's
                # (reference condition rows are prepended ahead of the generated ones --
                # see `force_free_te`'s comment below), so transformer_ref(34) +
                # TE-nf4(21) = 55GB steady state is kept as the *only* budget carried
                # into denoise, leaving the same headroom this task measured safe for
                # ref2va's own activation footprint. Reloading `transformer` back is
                # deferred to the decode section below (after denoise has finished
                # needing headroom), the same "restore steady state right before the
                # next request needs it, not a moment sooner than necessary" shape
                # `generate()`'s own force_free_te reload already uses.

            # --- layout / latents / timesteps ---
            layout_step = MiniMaxH3Ref2VAPrepareLayoutStep()
            _, state = layout_step(pipe, state)
            latents_step = MiniMaxH3PrepareLatentsStep()
            _, state = latents_step(pipe, state)
            timesteps_step = MiniMaxH3SetTimestepsStep()
            _, state = timesteps_step(pipe, state)
        else:
            # `none` mode: TE's job is done -- free it and bring in transformer_ref
            # (vae is already permanently resident in this mode, so the reference
            # encoder step can run either before or after; doing it here mirrors
            # generate()'s own `none`-mode ordering for keyframes).
            with self._load_lock:
                self._free_text_encoder()
                self._ensure_transformer_ref(progress)
            reference_encoder_step = MiniMaxH3Ref2VAReferenceEncoderStep()
            _, state = reference_encoder_step(pipe, state)

            # --- layout / latents / timesteps ---
            layout_step = MiniMaxH3Ref2VAPrepareLayoutStep()
            _, state = layout_step(pipe, state)
            latents_step = MiniMaxH3PrepareLatentsStep()
            _, state = latents_step(pipe, state)
            timesteps_step = MiniMaxH3SetTimestepsStep()
            _, state = timesteps_step(pipe, state)

        # bnb-4bit mode (bf16 transformer_ref): force-free TE-nf4 (~21GB) before denoise,
        # unconditionally (unlike generate()'s hires-fix-only `force_free_te` -- a
        # reference always adds condition rows ahead of the generated ones, so ref2va's
        # packed sequence is longer than plain t2va's even at the same target
        # resolution/duration, and this task's own first real request reproduced the
        # consequence: transformer_ref(66.3) + TE-nf4(21.0) = 87.5GB steady state left
        # only ~8GB of headroom, and the very first denoise step OOM'd inside attention
        # ("Tried to allocate 1.23 GiB" with 92.4GB already in use) with just one
        # 2048px-short-edge image reference at 768x768/5s). Reloaded after decode,
        # below -- same "restore the steady state for the next request" shape
        # generate()'s own force_free_te reload uses.
        #
        # int8 mode (`H3_TRANSFORMER_BOTH_RESIDENT`): transformer_ref is only ~34GB, so
        # transformer_ref(34) + TE-nf4(21) = 55GB leaves ~40GB of headroom for denoise
        # activations -- comfortably more than the ~5GB t2va's own activations measured
        # at 768x768 (see H3_INT8_MODULES_TO_NOT_CONVERT-adjacent log excerpt in this
        # task's verification), so TE does not need to be force-freed here at all in
        # this mode. (`transformer`, t2va's, was already freed at this method's entry in
        # this mode and stays freed through denoise -- see that comment -- so the actual
        # resident set during ref2va's denoise here is just transformer_ref + TE-nf4,
        # identical in shape to bf16 mode's own post-force-free state, just without
        # needing the force-free step to get there.)
        #
        # IMPORTANT (same reasoning as generate()'s force_free_te comment): this free is
        # deliberately deferred until after layout_step/latents_step/timesteps_step above,
        # not fused into the reference-encoder section further up. `_execution_device`
        # resolves to the device of the *first* `nn.Module` still set on `self._pipe_ref`,
        # in `MiniMaxH3Ref2VABlocks`' component order -- `text_encoder` first, then `vae`.
        # Freeing text_encoder before those three steps run would make `vae` (parked on
        # CPU in bnb-4bit mode outside its active phase, which ended when
        # `_vae_to_cpu()` ran above) the new first hit, silently resolving
        # `_execution_device` to `cpu` -- the identical device-mismatch trap generate()'s
        # own comment documents finding for its layout_step. Freeing TE only once those
        # position_ids/layout tensors already exist on the correct device (set once here,
        # and never touched again for the rest of the request) sidesteps it entirely.
        # H3_LOWVRAM: always False here -- TE was already force-freed above, before
        # transformer_ref was even loaded (see the H3_LOWVRAM branch above).
        force_free_te = TE_QUANT == "bnb-4bit" and not H3_TRANSFORMER_BOTH_RESIDENT and not H3_LOWVRAM
        if force_free_te:
            with self._load_lock:
                self._free_text_encoder(force=True)

        # --- denoise loop, instrumented for progress polling (mirrors generate()'s
        # non-upscale path exactly, against transformer_ref instead of transformer) ---
        if progress:
            progress.update(phase="denoising", step=0, total_steps=num_inference_steps, message="デノイズ中...")
        t_denoise = time.time()
        step_times = []
        cache_skips = [0]
        out_height, out_width = state.get("height"), state.get("width")

        def _fbc_reset_and_context():
            self._pipe_ref.transformer_ref._reset_stateful_cache()
            return self._pipe_ref.transformer_ref.cache_context("h3")

        denoise_step = MiniMaxH3Ref2VADenoiseStep()
        orig_loop_step = denoise_step.loop_step

        def timed_loop_step(components, bstate, i, t):
            ts = time.time()
            result = orig_loop_step(components, bstate, i=i, t=t)
            step_times.append(time.time() - ts)
            if H3_CACHE == "fbc":
                cache_skips[0] += self._fbc_last_step_was_skip_ref()
            if progress:
                progress.update(step=i + 1, message=f"デノイズ中 {i + 1}/{num_inference_steps}")
            return result

        denoise_step.loop_step = timed_loop_step
        if H3_CACHE == "fbc":
            # Per-request reset -- see generate()'s matching comment for why this is
            # required (a stale head-block residual from a previous call could otherwise
            # make step 0 wrongly skip).
            self._pipe_ref.transformer_ref._reset_stateful_cache()
            with self._pipe_ref.transformer_ref.cache_context("h3"):
                _, state = denoise_step(pipe, state)
        else:
            _, state = denoise_step(pipe, state)
        denoise_time = time.time() - t_denoise

        # --- decode (shared MiniMaxH3VideoDecodeStep/MiniMaxH3AudioDecodeStep -- no
        # ref2va-specific decode step exists; num_condition_video_rows/
        # num_condition_audio_rows on `state`, set by the layout step above from
        # build_ref2va_packed_sequence's reference row counts, is what makes these drop
        # the reference rows and decode only the generated ones) ---
        if progress:
            progress.update(phase="decoding", message="動画/音声をデコード中...")
        # bf16 mode: transformer_ref(66.3) + TE-nf4(21.0) + vae pair(11.0) would exceed
        # this card's ~95.6GB (same three-way conflict as everywhere else in this
        # file), so transformer_ref is dropped for this short decode window and
        # reloaded right after (see below).
        # int8 both-resident mode: `transformer` (t2va's) was already freed at this
        # method's entry and never reloaded before now (see the entry-section and
        # force_free_te comments above) -- resident set going into decode is just
        # transformer_ref(34) + TE-nf4(21) = 55GB, and adding the vae pair(11) is only
        # 66GB, comfortably under budget. So transformer_ref does NOT need to be
        # dropped here in this mode; it is left alone (stays resident straight through
        # decode and into the next request, which is the whole point of int8 mode for
        # ref2va<->ref2va requests specifically).
        if TE_QUANT == "bnb-4bit" and not H3_TRANSFORMER_BOTH_RESIDENT:
            self._free_transformer_ref()
        self._vae_to_gpu()
        t_decode = time.time()
        video_decode_step = MiniMaxH3VideoDecodeStep()
        _, state = video_decode_step(pipe, state)
        audio_decode_step = MiniMaxH3AudioDecodeStep()
        _, state = audio_decode_step(pipe, state)
        decode_time = time.time() - t_decode

        videos = state.get("videos")
        audio = state.get("audio")
        sampling_rate = state.get("sampling_rate")

        video_tensor = videos[0] if isinstance(videos, list) else videos
        if video_tensor.dim() == 5:
            video_tensor = video_tensor[0]
        frames_uint8 = (
            (video_tensor.permute(0, 2, 3, 1).float().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        )
        audio_np = audio[0].float().cpu().numpy()
        rms = float(np.sqrt(np.mean(audio_np**2)))
        peak = float(np.max(np.abs(audio_np)))

        peak_vram = torch.cuda.max_memory_allocated() / 1e9

        del video_tensor, videos, audio
        gc.collect()
        torch.cuda.empty_cache()

        self._vae_to_cpu()
        if TE_QUANT == "bnb-4bit" and not H3_LOWVRAM:
            with self._load_lock:
                self._ensure_transformer_ref(progress)
                if force_free_te:
                    # Restore the bnb-4bit steady state (transformer_ref + TE-nf4 both
                    # resident) for the *next* request -- this request force-freed TE-nf4
                    # before denoise to make room for the reference-lengthened sequence's
                    # attention activations (see above). Reloaded after transformer_ref so
                    # the two big reloads are not competing for VRAM at the same time,
                    # mirroring generate()'s own force_free_te reload ordering.
                    self._load_text_encoder(progress)
                if H3_TRANSFORMER_BOTH_RESIDENT:
                    # Restore the int8 both-resident steady state (`transformer` +
                    # `transformer_ref` + TE-nf4 all resident) for the *next* request.
                    # `transformer` (t2va's) was freed at this method's entry to make
                    # room for the reference VAE-encode step and has stayed freed
                    # through denoise/decode since (see the entry-section comment).
                    # Now that decode's own vae-pair trip is done (`_vae_to_cpu()` just
                    # above), there is headroom again: transformer_ref(34) + TE-nf4(21)
                    # = 55GB resident, +34GB for this reload = 89GB, the same steady
                    # state `generate()`'s own t2va path settles into. Reloaded last
                    # (after transformer_ref/TE, whichever of those needed restoring)
                    # so it is not competing with them for VRAM during their own
                    # reloads.
                    self._ensure_transformer(progress)
        # H3_LOWVRAM: deliberately do NOT reload transformer_ref/TE here -- same
        # "nothing big resident between requests" reasoning as generate()'s own
        # lowvram decode tail.

        if progress:
            progress.update(phase="muxing", message="mp4へmux中...")
        job_stub = f"ref2va_{int(t_start)}"
        mp4_path = self.output_dir / f"{job_stub}.mp4"
        _mux_mp4(frames_uint8, audio_np, sampling_rate, FPS, mp4_path)

        result = {
            "prompt": prompt,
            "height": out_height,
            "width": out_width,
            "num_frames_requested_seconds": seconds,
            "num_frames": actual_num_frames,
            "duration_s": actual_num_frames / FPS,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "denoise_time_s": round(denoise_time, 2),
            "decode_time_s": round(decode_time, 2),
            "avg_step_time_s": round(sum(step_times) / len(step_times), 3) if step_times else None,
            "peak_vram_gb": round(peak_vram, 2),
            "ram": ram_gb(),
            "audio_rms": rms,
            "audio_peak": peak,
            "audio_sampling_rate": sampling_rate,
            "mp4_path": str(mp4_path),
            "mp4_filename": mp4_path.name,
            "total_elapsed_s": round(time.time() - t_start, 2),
            "mode": "ref2va",
            "te_quant": TE_QUANT,
            "transformer_quant": H3_TRANSFORMER_QUANT,
            "lowvram": H3_LOWVRAM,
            "attn_backend": H3_ATTN_BACKEND or "default",
            "cache_mode": H3_CACHE,
            "cache_threshold": H3_CACHE_THRESHOLD if H3_CACHE == "fbc" else None,
            "cache_skipped_steps": cache_skips[0] if H3_CACHE == "fbc" else None,
            "references_summary": [
                {"index": index, "kind": kind, "has_audio": bool(references[index].has_audio)}
                for index, kind in enumerate(kinds)
            ],
        }
        if progress:
            progress.update(phase="done", message="完了", result_path=str(mp4_path))
        logger.info("ref2va generation done: %s",
                     json.dumps({k: v for k, v in result.items() if k != "ram"}, ensure_ascii=False))
        return result


def _mux_mp4(frames_uint8: np.ndarray, audio_np: np.ndarray, sampling_rate: int, fps: int, mp4_path: Path):
    import av

    container = av.open(str(mp4_path), mode="w")
    vstream = container.add_stream("libx264", rate=fps)
    vstream.width = frames_uint8.shape[2]
    vstream.height = frames_uint8.shape[1]
    vstream.pix_fmt = "yuv420p"

    astream = container.add_stream("aac", rate=sampling_rate)
    astream.layout = "stereo"

    for frame in frames_uint8:
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in vstream.encode(av_frame):
            container.mux(packet)
    for packet in vstream.encode():
        container.mux(packet)

    audio_i16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)  # (2, N)
    # av's packed s16 stereo format wants interleaved L,R,L,R,... in a (1, 2N) array, not
    # a (2, N) per-channel block layout (verified against a manual roundtrip probe).
    audio_interleaved = audio_i16.T.reshape(1, -1)
    audio_frame = av.AudioFrame.from_ndarray(audio_interleaved, format="s16", layout="stereo")
    audio_frame.sample_rate = sampling_rate
    for packet in astream.encode(audio_frame):
        container.mux(packet)
    for packet in astream.encode():
        container.mux(packet)

    container.close()

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
import hashlib
import io
import itertools
import json
import logging
import os
import threading
import time
import weakref
from contextlib import contextmanager
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

# Re-exported so callers (app.py) can do
# `from core.runner import MiniMaxH3ImageReference` (etc.) without reaching into
# diffusers' modular_pipelines package themselves. Cheap import (no model loading, just
# dataclass/PyAV/numpy/torch glue) -- safe at module level, unlike the actual big-model
# loading calls in this file, which all stay lazy/on-demand.
#
# PR #14355 (merged 2026-08-05, f37ab93) note: the old `MiniMaxH3Reference(image=...)` /
# `(video=...)` / `(audio=...)` single-class construction is gone -- `MiniMaxH3Reference`
# is now only the *base* class of a hierarchy (references.py): `MiniMaxH3ImageReference`,
# `MiniMaxH3VideoReference`, `MiniMaxH3AudioReference`, each a `@dataclass` with its own
# field (`image=`/`frames=`/`audio=`) and a `from_file(path_or_url)` classmethod that
# decodes a path itself (PIL for an image, PyAV for video/audio) -- the direct
# replacement for the old single-class path construction app.py used. `kind`
# ("image"/"video"/"audio") and `has_audio` are attributes on every instance (class
# attrs on `MiniMaxH3ImageReference`/`MiniMaxH3AudioReference`, a property on
# `MiniMaxH3VideoReference`), so call sites that used to call the old free function
# `packing_ref2va.reference_kind(index, entry)` now just read `entry.kind`/
# `entry.has_audio` directly -- confirmed by reading before_encoder.py's
# `MiniMaxH3Ref2VASetupStep.__call__`, which does exactly that.
from diffusers.modular_pipelines.minimax_h3 import (
    MiniMaxH3AudioReference,
    MiniMaxH3ImageReference,
    MiniMaxH3Reference,
    MiniMaxH3VideoReference,
)

# The single source of truth for which hidden_states index MiniMax-H3 conditions on
# (currently 50). PR #14355 turned the old `packing.MINIMAX_H3_TEXT_ENCODER_LAYER`
# module constant into the `components.text_encoder_layer` property of
# `MiniMaxH3ModularPipeline` (modular_pipeline.py) -- there is no longer a module-level
# constant to import, so this file keeps its own copy here (same value, 50) as the
# "single source of truth" the rest of this module reads, and H3_TE_PRUNE's layer-count
# math (`_text_encoder_config_kwargs`, below) uses it the same way it always did. Kept
# as a plain module constant rather than threaded through `self._pipe.text_encoder_layer`
# everywhere because several read sites (this constant's own module-docstring comments,
# `_text_encoder_config_kwargs`) run before a `self._pipe` necessarily exists.
MINIMAX_H3_TEXT_ENCODER_LAYER = 50

logger = logging.getLogger("minimax_h3.runner")

MODEL_ID = "MiniMaxAI/MiniMax-H3"
DEVICE = torch.device("cuda:0")
CPU = torch.device("cpu")

# ---------------------------------------------------------------------------
# H3_VRAM_LIMIT_GB: このプロセスが計算用GPU上で確保できる VRAM の上限 (GB, 10進)。
# 既定は空 = 無制限 (カード全部を使ってよい)。
#
# 用途1 **同居**: 同じGPUで ComfyUI や diffusers-server 等を並走させるとき、H3 が
#   カードを食い尽くさないように上限を切る。96GB のカードでも「H3 には半分だけ」と
#   いった運用ができる (例: H3_VRAM_LIMIT_GB=48)。上限を超える確保は PyTorch の
#   キャッシングアロケータが OOM として弾くので、**同居相手のメモリを奪わない**。
# 用途2 **低VRAM構成の検証**: `scripts/vram_ballast.py` はダミーテンソルを実際に確保
#   して空きを減らす方式だが、こちらは1行で済み、確保もしない。ただし
#   `torch.cuda.mem_get_info()` が返す空き容量やカード総容量の見え方は変わらない
#   (バラストは変わる) ので、**容量を読んで分岐するコード**
#   (`_te_external_usable_for()` の 24GB 判定など) の検証にはバラストを使うこと。
#
# 実装は `torch.cuda.set_per_process_memory_fraction()`。fraction はカード総容量に
# 対する比なので、GB 指定を総容量で割って渡す。指定が総容量以上なら無意味なので警告
# だけ出して素通しする。**予約ではなく上限**なので、使わない限りメモリは消費しない。
H3_VRAM_LIMIT_GB = os.environ.get("H3_VRAM_LIMIT_GB", "").strip()
if H3_VRAM_LIMIT_GB:
    _limit_gb = float(H3_VRAM_LIMIT_GB)
    if _limit_gb <= 0:
        raise ValueError(f"H3_VRAM_LIMIT_GB must be positive, got {H3_VRAM_LIMIT_GB!r}")
    if torch.cuda.is_available():
        _total_gb = torch.cuda.get_device_properties(DEVICE.index or 0).total_memory / 1e9
        if _limit_gb >= _total_gb:
            logger.warning(
                "H3_VRAM_LIMIT_GB=%.1f is at or above the card's %.1fGB -- no cap applied.",
                _limit_gb, _total_gb,
            )
        else:
            torch.cuda.set_per_process_memory_fraction(_limit_gb / _total_gb, DEVICE.index or 0)
            logger.info(
                "VRAM cap: this process may allocate at most %.1fGB of %.1fGB on %s "
                "(fraction %.3f, H3_VRAM_LIMIT_GB). Exceeding it raises OOM instead of "
                "taking memory from co-tenant processes.",
                _limit_gb, _total_gb, DEVICE, _limit_gb / _total_gb,
            )
    else:
        logger.warning("H3_VRAM_LIMIT_GB set but CUDA is unavailable -- ignored.")

# "none" (default) = current per-request TE<->transformer GPU swap.
# "bnb-4bit" = TE quantized NF4, TE+transformer both resident permanently, VAEs cycle
# through GPU per-phase instead. See module docstring above.
TE_QUANT = os.environ.get("H3_TE_QUANT", "bnb-4bit").strip().lower()
if TE_QUANT not in ("none", "bnb-4bit"):
    raise ValueError(f"H3_TE_QUANT must be 'none' or 'bnb-4bit', got {TE_QUANT!r}")

# EXPERIMENTAL, opt-in, diagnostic-only (2026-08-27 ref2va encode-phase profiling task).
# "0" (default) = zero overhead, byte-for-byte identical to pre-this-flag behaviour --
# every `_PhaseTimer.mark()`/`.report()` call below is a single `if not H3_PHASE_TIMING:
# return` early-out, no timing, no CUDA sync, no logging. "1" = emits one INFO log line
# per named checkpoint inside `generate_ref2va()` (and the handful of shared helpers it
# calls -- `_ensure_vaes`/`_load_text_encoder`/`_vae_to_gpu`/`_vae_to_cpu`/
# `_ensure_transformer_ref` already have their own unconditional timing logs; this flag
# only adds NEW checkpoints inside the encode phase those don't cover: reference
# setup/normalize (PIL resize), vision-tower feature gathering, presentation tokenize,
# and the 32B/4B conditioner forward itself) with a `torch.cuda.synchronize()`
# immediately before each timestamp so GPU-async work (the conditioner forward, VAE
# encode) is attributed to the checkpoint that actually did it rather than bleeding into
# the next (CPU-only) one. Never changes control flow, return values, or what is
# computed -- purely additive logging for profiling `docs/h3-adaln-precompute-
# 20260826.md`'s open question ("~100s of fixed cost, TE-size and vision-tower FLOP
# estimates both failed to explain it -- where does the time actually go?").
H3_PHASE_TIMING = os.environ.get("H3_PHASE_TIMING", "0").strip() == "1"
H3_TIMELINE = os.environ.get("H3_TIMELINE", "0").strip() == "1"  # probe 2026-10-05: 一時的な壁時計タイムライン


# H3_DENOISE_CUDAGRAPH=1 (probe 2026-10-05, 既定 0): transformer_ref.forward を CUDA Graph 化する (core/h3_cudagraph.py)。
import core.h3_cudagraph as _h3_graph  # noqa: E402

H3_DENOISE_CUDAGRAPH = os.environ.get("H3_DENOISE_CUDAGRAPH", "0").strip() == "1"


def _tl(msg):
    if H3_TIMELINE:
        logger.info("[TL] %s t=%.3f", msg, time.time() % 1000)


class _PhaseTimer:
    """Tiny opt-in wall-clock breakdown, one instance per `generate_ref2va()` call.

    `mark(label)` records the elapsed time since the previous `mark()` (or since
    construction, for the first call) under `label`, CUDA-synced first so GPU-async work
    lands on the checkpoint that issued it rather than the next one. `report()` logs the
    full breakdown plus a sum-check against the caller-supplied total. A fresh instance
    per call (rather than a module-level singleton) means concurrent requests -- were
    this ever called from more than one thread, which today's single `_load_lock`/
    `generation_lock` structure prevents -- can never interleave their marks; not
    exercised by this task (H3 serializes generation), but cheap to get right.

    No-op-shaped when `H3_PHASE_TIMING=0`: `mark()`/`report()` both check the flag first
    and return immediately, so the only cost on the default path is one attribute read
    per call site plus this object's own (never accessed) construction.
    """

    __slots__ = ("_t0", "_last", "_marks", "_label")

    def __init__(self, label: str = "ref2va"):
        self._label = label
        self._marks: list[tuple[str, float]] = []
        if not H3_PHASE_TIMING:
            return
        torch.cuda.synchronize()
        self._t0 = self._last = time.time()

    def mark(self, name: str) -> None:
        if not H3_PHASE_TIMING:
            return
        torch.cuda.synchronize()
        now = time.time()
        self._marks.append((name, now - self._last))
        self._last = now

    def report(self, total_hint: float | None = None) -> None:
        if not H3_PHASE_TIMING:
            return
        total = self._last - self._t0
        breakdown = ", ".join(f"{name}={dt:.2f}s" for name, dt in self._marks)
        hint = f" (caller total={total_hint:.2f}s)" if total_hint is not None else ""
        logger.info("[H3_PHASE_TIMING] %s breakdown: %s -- sum=%.2fs%s", self._label, breakdown, total, hint)


def _install_qwen3vl_submodule_timing(text_encoder) -> None:
    """H3_PHASE_TIMING drill-down (2026-08-27 encode-phase profiling task): split
    `get_qwen3vl_prompt_embeds()`'s single `text_encoder.model(...)` forward call --
    which `_encode_ref2va_prompt`'s own `conditioner_forward` checkpoint already times as
    one ~90s black box -- into "vision tower" (`Qwen3VLModel.get_image_features()`, which
    the model forward calls once per request when `pixel_values is not None`) vs.
    "language_model / text decoder" (`Qwen3VLModel.language_model(...)`, the 64
    Qwen3-VL-32B decoder layers run over the full presentation sequence, images-as-tokens
    included) time. No-op when `H3_PHASE_TIMING=0` (both wrapped methods early-out to the
    original bound method before doing anything else -- a single attribute check, no
    timing, no sync).

    Called once per fresh `text_encoder` load (from `_load_text_encoder()`, mirroring
    `enable_adaln_precompute()`'s own "re-arm on every fresh load" pattern a few call
    sites away in this file) -- `text_encoder.model` is a new `Qwen3VLModel` instance
    every time the bnb-4bit TE is (re)quantized from disk, so the patch has to be
    reapplied, not applied once at import time. Idempotent (`_h3_phase_timing_patched`
    guard) so calling it again on an already-patched instance is a harmless no-op --
    relevant because this project's `H3_TE_PROJ` path loads a *different* model class
    (`AutoModelForImageTextToText` resolving to Qwen3-VL-4B, whose `.model` is a plain
    `Qwen3VLModel` too, so this patch generalizes to that path for free, in case a future
    profiling task wants the same drill-down for the 4B TE_PROJ config).
    """
    if not H3_PHASE_TIMING:
        return
    model = getattr(text_encoder, "model", None)
    if model is None or getattr(model, "_h3_phase_timing_patched", False):
        return

    orig_get_image_features = model.get_image_features
    orig_language_model_forward = model.language_model.forward

    def _timed_get_image_features(pixel_values, image_grid_thw=None, **kwargs):
        torch.cuda.synchronize()
        t0 = time.time()
        result = orig_get_image_features(pixel_values, image_grid_thw=image_grid_thw, **kwargs)
        torch.cuda.synchronize()
        grid_str = image_grid_thw.tolist() if image_grid_thw is not None else None
        logger.info(
            "[H3_PHASE_TIMING] Qwen3VLModel.get_image_features (vision tower): %.2fs "
            "(pixel_values=%s, image_grid_thw=%s)",
            time.time() - t0, tuple(pixel_values.shape), grid_str,
        )
        return result

    def _timed_language_model_forward(*args, **kwargs):
        torch.cuda.synchronize()
        t0 = time.time()
        result = orig_language_model_forward(*args, **kwargs)
        torch.cuda.synchronize()
        logger.info("[H3_PHASE_TIMING] Qwen3VLModel.language_model (text decoder, 64 layers): %.2fs", time.time() - t0)
        return result

    model.get_image_features = _timed_get_image_features
    model.language_model.forward = _timed_language_model_forward

    # One level deeper (2026-08-27, same task): `get_image_features` -> `self.visual(...)`
    # is `Qwen3VLVisionModel.forward()` -- patch_embed (a `kernel_size==stride` patchify
    # `nn.Conv3d`, `modeling_qwen3_vl.py`'s `Qwen3VLVisionPatchEmbed.forward`, called once
    # with a huge batch dim = num_patches and a tiny spatial extent per patch) vs. the 27
    # `Qwen3VLVisionBlock` self-attention/MLP layers vs. the final `merger`. This exact
    # "huge-batch x tiny-spatial, kernel==stride Conv3d" shape is the one flagged in the
    # sibling diffusers-server repo's CLAUDE.md #46 as falling into a pathologically slow
    # cuDNN kernel on this machine's sm_120 (Blackwell) GPUs -- Qwen3-VL's vision
    # patch_embed uses the identical pattern, so this drill-down exists to confirm or rule
    # out the same root cause here. `visual` is `model.visual` (`Qwen3VLVisionModel`);
    # `blocks` timing is accumulated across all 27 sequential block calls into one number
    # (individually wrapping each would be noisy for no extra insight -- the hypothesis
    # under test is "patch_embed alone" vs. "everything else", not per-block variance).
    visual = getattr(model, "visual", None)
    if visual is not None:
        orig_patch_embed_forward = visual.patch_embed.forward
        orig_merger_forward = visual.merger.forward
        orig_block_forwards = [blk.forward for blk in visual.blocks]
        _blocks_total = {"s": 0.0}

        def _timed_patch_embed_forward(hidden_states, *args, **kwargs):
            torch.cuda.synchronize()
            t0 = time.time()
            result = orig_patch_embed_forward(hidden_states, *args, **kwargs)
            torch.cuda.synchronize()
            logger.info(
                "[H3_PHASE_TIMING]   visual.patch_embed (Conv3d, kernel==stride): %.2fs (input=%s)",
                time.time() - t0, tuple(hidden_states.shape),
            )
            return result

        def _timed_merger_forward(*args, **kwargs):
            torch.cuda.synchronize()
            t0 = time.time()
            result = orig_merger_forward(*args, **kwargs)
            torch.cuda.synchronize()
            logger.info("[H3_PHASE_TIMING]   visual.merger: %.2fs", time.time() - t0)
            return result

        def _make_timed_block_forward(orig_fwd):
            def _timed_block_forward(*args, **kwargs):
                torch.cuda.synchronize()
                t0 = time.time()
                result = orig_fwd(*args, **kwargs)
                torch.cuda.synchronize()
                _blocks_total["s"] += time.time() - t0
                return result

            return _timed_block_forward

        visual.patch_embed.forward = _timed_patch_embed_forward
        visual.merger.forward = _timed_merger_forward
        for blk, orig_fwd in zip(visual.blocks, orig_block_forwards):
            blk.forward = _make_timed_block_forward(orig_fwd)

        orig_visual_forward = visual.forward

        def _timed_visual_forward(*args, **kwargs):
            _blocks_total["s"] = 0.0
            result = orig_visual_forward(*args, **kwargs)
            logger.info(
                "[H3_PHASE_TIMING]   visual.blocks (all %d Qwen3VLVisionBlock, sum): %.2fs",
                len(visual.blocks), _blocks_total["s"],
            )
            return result

        visual.forward = _timed_visual_forward

    model._h3_phase_timing_patched = True
    logger.info("[H3_PHASE_TIMING] installed vision-tower/language-model sub-timing on text_encoder.model")


# "1" (default) = replace `Qwen3VLVisionPatchEmbed.proj` (a `kernel_size==stride`
# `nn.Conv3d(in_channels=3, embed_dim=1152, kernel_size=[2,16,16], stride=[2,16,16],
# bias=True)`) with a mathematically-equivalent `nn.functional.linear` on every fresh
# `text_encoder` load. "0" = stock Conv3d, unchanged behaviour (escape hatch).
#
# Why: on this box's sm_120 (Blackwell) GPUs, cuDNN picks a pathologically slow kernel
# for this exact shape -- huge batch dim (num_patches, 28,160 for the profiled
# 768x448/8s ref2va request logged in `_install_qwen3vl_submodule_timing`'s own
# `visual.patch_embed` checkpoint) x tiny spatial extent (2x16x16 = one "patch" worth of
# voxels, matching kernel_size exactly so there is only ever one output position per
# input). This is the identical pathology diffusers-server's CLAUDE.md #46 (JoyAI
# integration) already found and fixed for JoyImageEditPlusTransformer3DModel's
# `img_in` (same class of Conv3d) AND for this exact `Qwen3VLVisionPatchEmbed` in that
# repo's `families/joyai/pipeline.py` (`PatchifyLinear`) -- ~146x speedup there. Isolated
# microbench with THIS process's exact real shapes (`scratchpad/probe_patchify_conv3d.py`,
# run on this box 2026-08-27): Conv3d call = 87-91s *per call* (not a one-time cuDNN
# algo-search cost -- 3 repeated calls at the same shape averaged 90.8s/call), the
# reshape+linear form = 0.0007-0.005s, mean abs diff 6.4e-07 / max abs diff 1.56e-2
# (bf16 rounding-level; consistent with CLAUDE.md #46's own 6.18e-07 for JoyAI).
#
# For `kernel_size == stride` and no padding, a Conv3d degenerates to exactly one output
# position per input window, so `conv(x).view(N, out_ch)` is bit-identical (up to
# floating-point summation order, which bf16 rounds away) to
# `F.linear(x.flatten(1), conv.weight.reshape(out_ch, -1), conv.bias)` -- verified against
# THIS repo's pinned transformers `Qwen3VLVisionPatchEmbed.forward()`
# (`transformers/models/qwen3_vl/modeling_qwen3_vl.py`), which reshapes its input to
# exactly `(-1, in_channels, temporal_patch_size, patch_size, patch_size)` before calling
# `self.proj(...)` -- i.e. batch dim first, no other axis reordering, so `x.flatten(1)`
# on that same 5D tensor lines up with `conv.weight.reshape(out_ch, -1)`'s
# `(in_ch, kt, kh, kw)` -> flat layout with no extra permute needed (this project's
# layout was independently re-derived and confirmed here, not assumed to match JoyAI's
# transformer -- the two use different upstream input tensor orderings in general, but
# for THIS class they happen to already agree since `forward()` does the `.view()`
# itself right before the conv call, batch-dim-first).
H3_PATCH_EMBED_LINEAR = os.environ.get("H3_PATCH_EMBED_LINEAR", "1").strip() == "1"


class _PatchEmbedLinear(torch.nn.Module):
    """Drop-in replacement for a `kernel_size==stride` `nn.Conv3d` patchify layer.

    Ported from diffusers-server's `families/joyai/pipeline.py::PatchifyLinear`
    (CLAUDE.md #46) with the mechanism kept verbatim -- only the class name and this
    module's own call site are new. `weight`/`bias` properties are required because
    `Qwen3VLVisionPatchEmbed.forward()` reads `self.proj.weight.dtype` (to cast its
    input) before calling `self.proj(...)` -- a plain buffer without the alias would
    make that line raise `AttributeError` (the exact bug CLAUDE.md #46 records hitting
    and fixing on its first attempt at this same pattern for JoyAI's `img_in`).
    """

    def __init__(self, conv: torch.nn.Conv3d):
        super().__init__()
        self.out_channels = conv.out_channels
        self.register_buffer("_weight_flat", conv.weight.reshape(conv.out_channels, -1))
        self.register_buffer("_bias", conv.bias if conv.bias is not None else None)

    @property
    def weight(self):
        return self._weight_flat

    @property
    def bias(self):
        return self._bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `Qwen3VLVisionPatchEmbed.forward()` passes in the already-5D
        # `(N, in_channels, temporal_patch_size, patch_size, patch_size)` tensor and
        # itself does `.view(-1, embed_dim)` on this call's return value, so returning
        # the plain 2D `(N, out_channels)` linear output (rather than reshaping back to
        # 5D the way `PatchifyLinear.forward()` does for JoyAI's `img_in`, whose caller
        # expects a 5D shape back) is correct here -- confirmed by reading
        # `Qwen3VLVisionPatchEmbed.forward()` itself, not assumed from the JoyAI
        # precedent.
        return torch.nn.functional.linear(x.flatten(1), self._weight_flat, self._bias)


def _install_patch_embed_linear(text_encoder) -> None:
    """Replace `text_encoder.model.visual.patch_embed.proj` with `_PatchEmbedLinear` on
    a freshly-loaded `text_encoder` (no-op if `H3_PATCH_EMBED_LINEAR=0`, or if this
    instance's vision tower is missing/already patched).

    Must be re-armed on every fresh `text_encoder` load, same reasoning as
    `_install_qwen3vl_submodule_timing` a few call sites away in this file (each
    (re)quantization/reload of the bnb-4bit TE, or each `H3_TE_PROJ` 4B load, produces a
    brand new `Qwen3VLModel`/`Qwen3VLVisionModel` instance with its own fresh
    `nn.Conv3d`) -- called from the same 4 `_load_text_encoder*` call sites, right next
    to the existing `_install_qwen3vl_submodule_timing(...)` calls. Idempotent per
    instance (`isinstance(..., _PatchEmbedLinear)` guard), so calling it again on an
    already-patched instance is a harmless no-op.

    Covers both the 32B TE path (`H3_TE_QUANT=none`/`bnb-4bit`, `.model.visual` is
    `Qwen3VLModel.visual`) and the `H3_TE_PROJ` 4B path (`AutoModelForImageTextToText`
    resolving to Qwen3-VL-4B-Instruct, same `Qwen3VLVisionPatchEmbed` class, same
    pathological shape family just with fewer patches) for free -- both are plain
    `Qwen3VLModel` instances exposing `.model.visual.patch_embed.proj`, no branching on
    which load path called this needed.
    """
    if not H3_PATCH_EMBED_LINEAR:
        return
    model = getattr(text_encoder, "model", None)
    visual = getattr(model, "visual", None) if model is not None else None
    patch_embed = getattr(visual, "patch_embed", None) if visual is not None else None
    if patch_embed is None:
        return
    proj = getattr(patch_embed, "proj", None)
    if proj is None or isinstance(proj, _PatchEmbedLinear):
        return  # already patched, or not the Conv3d-based patchify this targets

    device = next(proj.parameters()).device
    n_params_before = sum(p.numel() for p in proj.parameters())
    replacement = _PatchEmbedLinear(proj).to(device)
    patch_embed.proj = replacement
    logger.info(
        "[H3_PATCH_EMBED_LINEAR] replaced Qwen3VLVisionPatchEmbed.proj (Conv3d, "
        "kernel==stride) with reshape+linear on %s (weight=%s, params=%d) -- "
        "sm_120 pathological-kernel avoidance, CLAUDE.md #46 lineage. "
        "Set H3_PATCH_EMBED_LINEAR=0 to revert to stock Conv3d.",
        device, tuple(replacement.weight.shape), n_params_before,
    )


# EXPERIMENTAL, opt-in. "0" (default) = text_encoder is built with its checkpoint's
# native 64 decoder layers, byte-for-byte identical to pre-this-flag behaviour. "1" =
# the text_encoder is built with only its first 51 decoder layers (the checkpoint's
# layers.51-63, ~14 layers, plus the final `norm`/`lm_head`, are never constructed at
# all -- their weights show up as "UNEXPECTED" in transformers' from_pretrained load
# report and are simply skipped).
#
# MiniMax-H3 conditions on `hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER]` (=50) of its
# Qwen3-VL-32B conditioner and never touches the LM head (see diffusers'
# minimax_h3/encoders.py and packing.py) -- confirmed by reading
# transformers/models/qwen3_vl/modeling_qwen3_vl.py's `Qwen3VLTextModel.forward`
# together with `_can_record_outputs = {"hidden_states": Qwen3VLTextDecoderLayer}`
# and `install_output_capuring_hook`'s `capture_initial_hidden_state` semantics
# (transformers/utils/output_capturing.py): `hidden_states[0]` is the embedding
# output (captured as the hook firing on `layers[0]`'s own *input*, `args[0]`), and
# `hidden_states[k]` for k=1..num_hidden_layers is the *output* of `layers[k-1]` (the
# hook fires as a forward hook on that layer). So `hidden_states[50]` = the output of
# `layers[49]` -- only `layers[0..49]` (50 layers) are ever executed before that value
# is read; everything from `layers[50]` onward, plus the model's final `norm` and the
# LM head, is dead weight for MiniMax-H3's own use of this checkpoint (confirmed this
# is not merely an unused *module*, but genuinely never touched at forward time: the
# decoder loop only runs `range(config.num_hidden_layers)` iterations in the first
# place, so truncating `num_hidden_layers` means those layers are literally never
# constructed nor executed, not just constructed-and-ignored).
#
# `num_hidden_layers` is set to 51 (MINIMAX_H3_TEXT_ENCODER_LAYER + 1), NOT 50 -- found
# by this task's own verification (scripts/probe_te_prune*.py), not assumed: pruning to
# exactly 50 layers makes `hidden_states[50]` the *last* entry of the captured tuple.
# `Qwen3VLTextModel.forward` is wrapped in `@capture_outputs` (transformers'
# output_capturing.py) with its default `tie_last_hidden_states=True`, which
# unconditionally overwrites the *last* captured hidden_states entry with
# `outputs.last_hidden_state` -- the value AFTER the model's final `self.norm(...)`
# call. In the real 64-layer checkpoint, index 50 is nowhere near the last entry (index
# 64), so this substitution never touches it and `hidden_states[50]` is genuinely the
# raw (pre-norm) output of `layers[49]`, matching what MiniMax-H3 was trained to
# condition on. Truncating to exactly 50 layers makes index 50 the *only* (and thus
# last) entry, silently swapping it for the post-norm value instead -- reproduced by
# this task's own probe (max abs diff ~1.5e4 against the real 64-layer model's
# `hidden_states[50]`, i.e. not quantization noise, a genuinely different number) and
# fixed by keeping one extra layer (51 built, `layers[50]` executes but its output is
# simply never read) so index 50 sits mid-stack again. Verified bit-identical
# (`torch.equal`, both bf16 and bnb-4bit nf4) against the unpruned model's
# `hidden_states[50]` after this fix. This is exactly the failure mode
# `get_qwen3vl_prompt_embeds`'s own guard (`if num_layers <= text_encoder_layer: raise
# ValueError(...)`, see encoders.py -- this used to be `MiniMaxH3TextEncoderStep.
# encode_prompt`'s guard pre-PR#14355, same check, moved to the new module function) was
# written to catch -- pruning to 50 would have raised there; 51 is the smallest value
# that both passes that guard and sidesteps the tie_last_hidden_states substitution.
#
# Applied in `_load_text_encoder()` via `config=` passed straight into
# `load_components()` (same per-component-kwarg dict shape already used for TE's
# `BitsAndBytesConfig`/`device_map`) -- `ComponentSpec.load()` forwards any kwarg that
# is not one of its own loading fields (pretrained_model_name_or_path/subfolder/
# variant/revision) straight to `from_pretrained(..., **kwargs)`, and
# `PreTrainedModel.from_pretrained` skips its own config auto-load entirely when
# `isinstance(config, PreTrainedConfig)` is already true, using the object passed in
# verbatim instead (confirmed by reading modeling_utils.py's `from_pretrained`).
# Composes with `H3_TE_QUANT` (bnb-4bit or none) and every `H3_LOWVRAM`/
# `H3_LOWVRAM_GROUP` mode without any choreography changes: this only shrinks the
# text_encoder's own footprint (measured ~3.6GB smaller bnb-4bit nf4, ~13.6GB smaller
# bf16 -- nf4 already compresses each pruned bf16 layer ~4x, so the absolute nf4 saving
# is proportionally smaller), it does not change *when* or *whether* TE is resident.
H3_TE_PRUNE = os.environ.get("H3_TE_PRUNE", "0").strip() == "1"

# 量子化済み text_encoder のディスクキャッシュ (既定ON)。
#
# 動機: `H3_LOWVRAM=1` は毎リクエストで TE を再ロードするため、その時間がそのまま
# 固定費になる。この時間の大半は「元の bf16 重みを読む + その場で bnb-4bit へ量子化
# する」処理であり、**量子化後の重みを一度保存しておけば次回以降は読むだけで済む**。
#
# 実測 (2026-08-08、scripts/probe_prequantized_ckpt.py と probe_prequant_equivalence.py):
#   ロード 66.9s -> 2.6s (25.7倍) / 別実行では 41.5s -> 4.0s (10.3倍)、保存物 17.44GB
#   (H3_TE_PRUNE=1 の場合)。**出力の等価性はビット一致で確認済み** --
#   `hidden_states[50]` が現行経路と `torch.equal` で完全一致 (max_abs_diff 0.0、
#   日英2プロンプト)、`text_token_tags` も一致。bnb-4bit の量子化は決定的なので当然の
#   結果だが、速いだけで採用しないのが本プロジェクトの流儀なので実測で確かめてある。
#
# キャッシュは **設定ごとに別ディレクトリ**へ置く (`te_<quant>_<prune>`)。TE_QUANT や
# TE_PRUNE を変えると中身が別物になるため、同じ場所を使い回すと設定を切り替えたときに
# 古い重みを読んでしまう。ディレクトリ名に設定を含めることでこの事故を構造的に防ぐ。
#
# ディスクを 17-21GB 消費する。空きが足りない環境のために "0" で完全に無効化できる
# (無効時は従来どおり毎回その場で量子化する = このフラグ導入前と同一挙動)。保存に
# 失敗した場合も**生成は続行する** (キャッシュはあくまで高速化であり、機能ではない)。
# EXPERIMENTAL, opt-in。"" (既定) は従来どおり全モデルが同じ GPU を使う。
# "cuda:1" 等を指定すると **text_encoder だけを別GPUへ常駐**させ、リクエスト間も解放しない。
#
# 動機: `H3_LOWVRAM=1` が毎リクエストで TE を再ロードする根本原因は、デノイズ中に TE の
# 置き場所が無いこと (48GB では transformer-int8 34GB + 活性化 5GB = 39GB で、残り 9GB に
# TE の 17.45GB は入らない)。TE を別カードへ逃がせばこの再ロード (実測 29.5-53s) が消える。
#
# 実測 (2026-08-08、scripts/probe_te_on_second_gpu.py、RTX 4000 SFF Ada 20GB / PCIe Gen4 x4):
#   t2va は成立 (peak 17.76GB、余裕 3.23GB、エンコード 0.5-0.7s) / **ref2va は OOM**
#   (2048px 短辺の参照を vision tower に通すため 1-2GB 不足)。よって 20GB 級では
#   t2va/fl2va/t2i 系のみ対象とし、**ref2va は従来経路へフォールバックする** (下記
#   `_te_external_usable_for()`)。ref2va まで含めるには 24GB 級の TE 用GPUが要る。
#   PCIe 幅は問題にならない: TE は起動時に一度載せるだけで、毎リクエストの転送は
#   prompt_embeds の 42MB のみ (x4 でも約6ms)。
#
# **最大の罠**: `_execution_device` は `components` の順で最初に見つかった nn.Module の
# デバイスを返す (実装を読んで確認: modular_pipeline.py)。順序は
# `text_encoder, tokenizer, processor, vae, scheduler, audio_scheduler, transformer, ...`
# なので、TE を cuda:1 に置くと **layout/latents/timesteps が cuda:1 上にテンソルを作り**、
# cuda:0 の transformer との device mismatch になる。対策は
# `_pin_execution_device_to_compute()` -- その窓の間だけ text_encoder と vae をパイプから
# 外し、transformer (cuda:0) が最初に見つかるようにする (モジュール自体は解放しない)。
H3_TE_DEVICE = os.environ.get("H3_TE_DEVICE", "").strip()

H3_TE_PREQUANT = os.environ.get("H3_TE_PREQUANT", "1").strip() == "1"
H3_TE_PREQUANT_DIR = Path(
    os.environ.get("H3_TE_PREQUANT_DIR", str(Path(__file__).resolve().parent.parent / "models" / "prequant"))
)
# 保存前に確認する空きディスクの下限 (GB)。TE-nf4 は削除版 17.44GB / 未削除 ~21GB
# なので、保存物 + 余裕を見て 25GB を既定とする。下回る場合は保存をスキップし、
# 従来経路で動作を続ける (ディスクを埋めてシステムを巻き添えにしないため)。
H3_TE_PREQUANT_MIN_FREE_GB = float(os.environ.get("H3_TE_PREQUANT_MIN_FREE_GB", "25"))

# EXPERIMENTAL, opt-in. "" (既定) は無効 -- 以下のブロックは1バイトも既存挙動に触れない。
#
# 動機: TE (Qwen3-VL-32B, NF4 で 21.02GB / H3_TE_PRUNE 併用で 17.45GB) と transformer
# (int8 34GB 級) が 48GB 機で同時常駐できないため、毎リクエストで載せ替えが起きており
# それが速度の律速になっている (README/RESIDENCY.md 参照)。Qwen3-VL-4B (bf16 で
# 約5.2GB 見込み) + 学習済み線形投影行列で 32B TE を置き換えれば、TE と transformer が
# 同時常駐できるようになる。
#
# 投影行列は HuggingFace `NicoLab28/ClipProj-MiniMax-H3` の
# `h3_qwen3vl_4b_tap24.safetensors` (実測確認済み、2026-08-10: W (2560, 5120) fp32,
# mean_in/std_in (2560,), mean_out/std_out/sink_out (5120,), metadata tap="24") --
# 4B の `hidden_states[24]` (36層中24層目、post-norm 混入の懸念なし -- 24 は最終層36と
# 十分離れているため `capture_outputs` の `tie_last_hidden_states` 置換の対象にならない。
# H3_TE_PRUNE がまさにこの罠を避けるために +1 していたのと同種の確認、ここでは該当しない)
# を学習済みの `W`/`mean_in`/`std_in`/`mean_out`/`std_out` で 5120 次元 (32B TE と同じ
# 出力次元) へ写す。適用式・sink_out の扱いは参照実装
# (https://github.com/nicolab28/ComfyUI-ClipProj の clipproj_projection.py) と同一にする
# ことが必須 (自前流に変えると学習済み統計とズレて劣化する)。
#
# `H3_TE_PROJ` に投影 safetensors のローカルパス、または HF リポジトリID
# (`NicoLab28/ClipProj-MiniMax-H3` のように) を指定する。パスとして存在すればローカル
# ファイル扱い、そうでなければ `hf_hub_download(H3_TE_PROJ, H3_TE_PROJ_FILE)` として
# 解決する (H3_TURBO_LORA の `_download_turbo_lora_if_needed` と同じパターン)。
#
# UI の「再ロード設定」パネルから te_proj を ON にしたとき (H3_TE_PROJ が env で未設定の
# 場合) に使う既定リポジトリID。`core/settings.py` の `apply_reload_settings()` が
# te_proj=True かつ `runner.H3_TE_PROJ` が空文字のときだけこれを書き込む -- env で
# 明示的にローカルパス等を指定している場合はそちらを優先し、上書きしない。
H3_TE_PROJ_DEFAULT_REPO = "NicoLab28/ClipProj-MiniMax-H3"
H3_TE_PROJ = os.environ.get("H3_TE_PROJ", "").strip()
# `H3_TE_PROJ` が HF リポジトリIDのときに読むファイル名。ローカルパス指定時は無視される。
# 既定ファイル名の変遷 (2026-08-12): 配布元が `h3_qwen3vl_4b_tap24.safetensors` を
# `obsolete/` へ移動し、**再校正版** `mmh3-4b-ClipProj.safetensors` に置き換えた
# (学習 1,666→5,664 プロンプト / 289K→1.14M トークン、cos_test 0.711→0.717。
# W の cosine は旧比 0.9596 = 実質別の関数)。旧名のままでは新規取得が 404 になるため
# 既定を新名へ更新。**注意: 2026-08-10 の品質実測 (PSNR 22.49dB 等) は旧行列での値**。
# 旧行列はローカル HF キャッシュ (snapshot 3f762f19) に残っており、必要なら
# H3_TE_PROJ にその絶対パスを渡せば再現できる。
H3_TE_PROJ_FILE = os.environ.get("H3_TE_PROJ_FILE", "mmh3-4b-ClipProj.safetensors").strip()
# 投影に使う小型 TE 本体。既定は投影行列が学習された対象 (safetensors メタデータの
# `source_model` = qwen3vl_4b_bf16) と同系列の Instruct 版。
H3_TE_PROJ_MODEL = os.environ.get("H3_TE_PROJ_MODEL", "Qwen/Qwen3-VL-4B-Instruct").strip()

# 投影TE (4B) 自身の量子化。**32B 用の `H3_TE_QUANT` とは別フラグ**にしてある。
# 下の排他ガードが `H3_TE_QUANT` 等を弾くのは「32B 向けの設定を 4B に流用させない」ため
# であって、4B を量子化してはいけないという意味ではない -- 混同しないこと。
#
#   "bnb-4bit"  (既定、2026-08-10 実測に基づき変更) NF4。常駐 3.11GB (bf16比 -65%)、
#               投影後の条件付けのズレは相対RMS 0.61〜0.96% (cosine 1.0000)。動画出力
#               でも劣化は確認されていない (README「追記(同日)」節参照)。
#   "bnb-8bit"  int8。常駐 4.84GB (-46%)、ズレは相対RMS 0.24〜0.53% (NF4よりさらに小さい)
#   "none"      bf16 のまま。常駐 8.88GB
#
# **なぜ量子化が既定か**: 48GB 機で TE と transformer(int8 34.03GB)を同時常駐させ、
# 位相ごとの載せ替え(このリポジトリの速度の律速)を消すため。bf16 の 8.88GB だと
# 8.88 + 34.03 + デノイズ活性化 6.6 = 49.5GB で実効予算 49.8GB に対し余裕 0.3GB しかなく、
# 20GB カードで「導出上は入るが実測 OOM」を踏んだ前例からして期待できない。NF4 なら
# 37.1 + 6.6 = 43.7GB で余裕 6.1GB。上記の実測(劣化ほぼ無し)から、H3_TE_PROJ 利用時は
# 既定で量子化する方が安全側に倒れていると判断した。
H3_TE_PROJ_QUANT = os.environ.get("H3_TE_PROJ_QUANT", "bnb-4bit").strip()
if H3_TE_PROJ_QUANT not in ("none", "bnb-4bit", "bnb-8bit"):
    raise RuntimeError(
        f"H3_TE_PROJ_QUANT must be one of none/bnb-4bit/bnb-8bit, got {H3_TE_PROJ_QUANT!r}"
    )
# 既定を "bnb-4bit" に変えたため、素朴に「H3_TE_PROJ_QUANT != 'none' かつ H3_TE_PROJ
# 未設定」で弾くと **H3_TE_PROJ を使わない全ユーザー**がこのガードに引っかかって即死する
# (投影OFFのまま H3_TE_PROJ_QUANT の既定値だけが有効値になっているだけなので、実際には
# 何も壊れていない)。「明示指定」の判定は `"H3_TE_PROJ_QUANT" in os.environ` -- 他の
# 排他ガード (`_proj_conflicts` 下記、H3_LOWVRAM の `_explicit_transformer_quant`) と
# 同じイディオム。オペレーターが実際に H3_TE_PROJ_QUANT を触っていて、なおかつ
# H3_TE_PROJ が未設定のときだけ矛盾として落とす。既定値のまま (未指定) なら投影OFF時は
# 黙って無視する (量子化対象の4B自体がロードされないので、値があっても単に使われない)。
if H3_TE_PROJ_QUANT != "none" and not H3_TE_PROJ and "H3_TE_PROJ_QUANT" in os.environ:
    raise RuntimeError(
        "H3_TE_PROJ_QUANT quantizes the projected 4B text encoder, but H3_TE_PROJ is not "
        "set, so there is no 4B to quantize. Set H3_TE_PROJ (or drop H3_TE_PROJ_QUANT)."
    )

if H3_TE_PROJ:
    # 小型モデルを別途量子化・層削除する意味がない (そもそも 4B は 32B より遥かに軽い上、
    # 投影行列は特定の tap 層・特定の重み分布に対して学習されているため、量子化や層削除で
    # 数値がずれると学習済み統計 (mean_in/std_in 等) との整合が崩れる恐れがある) ので、
    # 既存の TE 量子化/層削除/事前量子化キャッシュ系フラグとは排他とする。「明示指定」の
    # 判定は `"X" in os.environ` (H3_LOWVRAM の `_explicit_transformer_quant` と同じ書き方)
    # -- デフォルト値のまま (何も指定していない) なら黙って無視し、オペレーターが実際に
    # 何かを指定していたときだけ矛盾として落とす。
    _proj_conflicts = [
        f"{name}={os.environ[name]!r}"
        for name in ("H3_TE_QUANT", "H3_TE_PRUNE", "H3_TE_PREQUANT")
        if name in os.environ
    ]
    if _proj_conflicts:
        raise RuntimeError(
            "H3_TE_PROJ is set (Qwen3-VL-4B + learned projection replaces the 32B TE "
            "entirely) and is mutually exclusive with 32B-TE-specific quantization/pruning "
            f"flags, but these were also explicitly set: {', '.join(_proj_conflicts)}. "
            "Quantizing or layer-pruning a small model that is not what the projection "
            "matrix was trained against makes no sense and risks silently drifting from "
            "the trained mean_in/std_in statistics. Drop these flags (or unset H3_TE_PROJ)."
        )

# EXPERIMENTAL, opt-in, probe for 16GB-class support. "0" (default) = video VAE loads
# float32, byte-for-byte identical to pre-this-flag behaviour. "1" = the video VAE
# (`vae`, NOT `audio_vae` -- audio_vae must stay float32 end-to-end, see module
# docstring) is halved to float16 in place after loading, roughly halving its resident
# size (measured ~9.70GB fp32 -> ~4.85GB fp16 for the bare `nn.Module`, i.e. before
# accounting for the ~11GB `vae+audio_vae` pair's actual runner-measured decode-phase
# peak of ~16.29GB, which includes activation buffers on top of the weights).
#
# IMPORTANT: passing `dtype=torch.float16` straight into `ModularPipeline.load_components`
# (the naive approach) is a NO-OP for this VAE -- verified empirically, not assumed.
# `AutoencoderKLMiniMaxH3._keep_in_fp32_modules = ["encoder", "decoder", "quant_conv",
# "post_quant_conv"]` (autoencoder_kl_minimax_h3.py) covers essentially the entire
# module tree, and diffusers' own `from_pretrained` -> `load_model_dict_into_meta`
# (model_loading_utils.py) force-casts any parameter matching one of those names to
# float32 regardless of the `dtype=` kwarg (confirmed by loading with
# `torch_dtype=torch.float16` directly and finding every parameter still float32,
# 9.70GB). The only way to actually shrink the weights is a manual `.to(torch.float16)`
# call *after* `from_pretrained` returns (diffusers itself warns this "can lead to
# inconsistent results" -- exactly why this flag defaults off and is meant to be A/B'd
# against the fp32 baseline via PSNR before being trusted, see README).
#
# Decode already runs under `torch.autocast(dtype=torch.float16)` in diffusers' own
# `MiniMaxH3VideoDecodeStep` (decoders.py) even when the weights are float32 -- so
# halving the weights to float16 up front changes what precision the *weights*
# themselves are stored/read at, but the compute dtype inside the autocast region was
# already float16 either way. audio_vae is untouched by this flag; the module
# docstring's "cast to bf16 causes ~20dB volume loss" warning is specifically about
# audio_vae and does not apply here.
H3_VIDEO_VAE_FP16 = os.environ.get("H3_VIDEO_VAE_FP16", "0").strip() == "1"

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

# Overrides diffusers' `ConfigSpec("reference_image_short_edge", 2048)`
# (modular_pipelines/minimax_h3/before_encoder.py), which ref2va's prefix-encode step
# uses to normalize every reference image's short edge before it goes into Qwen3-VL-32B:
# `scale = reference_image_short_edge / min(width, height)` -- this is applied even when
# it means *upscaling* the reference, and there is no area cap on top of it. Measured on
# this box: a 1280x720 reference gets scaled up to 3648x2048, producing a 7,309-token
# prefix that alone takes ~92.9s to encode (of a ~140s scene; denoise is only 20-27s) --
# see the "single ref-prefix cache MISS: encoded 7309 prefix tokens in 92.9s" log line.
# Default 2048 matches diffusers' own default exactly, so leaving this unset reproduces
# current behaviour byte-for-byte (`_ensure_pipe_shell` below skips the
# `register_to_config` call entirely in that case). Lowering it shrinks the prefix (fewer
# tokens -> faster encode) but also shrinks how much detail of the reference survives
# into the prefix, which can affect character consistency -- this has NOT been quality
# A/B'd, so treat any non-default value as experimental and eyeball the output.
H3_REF_IMAGE_SHORT_EDGE_RAW = os.environ.get("H3_REF_IMAGE_SHORT_EDGE", "").strip()
H3_REF_IMAGE_SHORT_EDGE_DEFAULT = 2048
if H3_REF_IMAGE_SHORT_EDGE_RAW:
    try:
        H3_REF_IMAGE_SHORT_EDGE = int(H3_REF_IMAGE_SHORT_EDGE_RAW)
        if H3_REF_IMAGE_SHORT_EDGE <= 0:
            raise ValueError
    except ValueError:
        logger.warning(
            "H3_REF_IMAGE_SHORT_EDGE=%r is not a positive integer -- ignoring, using "
            "diffusers' default of %d.",
            H3_REF_IMAGE_SHORT_EDGE_RAW, H3_REF_IMAGE_SHORT_EDGE_DEFAULT,
        )
        H3_REF_IMAGE_SHORT_EDGE = H3_REF_IMAGE_SHORT_EDGE_DEFAULT
else:
    H3_REF_IMAGE_SHORT_EDGE = H3_REF_IMAGE_SHORT_EDGE_DEFAULT


def _resolve_ref_image_short_edge(override: int | None) -> int:
    """Per-request counterpart of `H3_REF_IMAGE_SHORT_EDGE` (see that env var's own
    comment above for the mechanism and the measured encode/denoise-time tradeoff).

    `override is None` (the only case app.py's `/api/ref2va` etc. hit when the caller
    omits the field entirely) returns `H3_REF_IMAGE_SHORT_EDGE` unchanged -- i.e. this
    process's env-var-resolved default, byte-for-byte the same value `_ensure_pipe_shell`
    would already have applied. Explicit values are validated the same way the env var
    is (positive int), but raise `ValueError` instead of warning-and-falling-back, since
    this is a live request the caller can correct and resubmit (unlike an env var typo
    baked in at process start).
    """
    if override is None:
        return H3_REF_IMAGE_SHORT_EDGE
    if not isinstance(override, int) or isinstance(override, bool) or override <= 0:
        raise ValueError(
            f"reference_image_short_edge は正の整数で指定してください: {override!r}"
        )
    return override


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
if H3_TRANSFORMER_QUANT not in ("none", "int8", "ck-w4a8"):
    raise ValueError(
        f"H3_TRANSFORMER_QUANT must be 'none', 'int8' or 'ck-w4a8', got {H3_TRANSFORMER_QUANT!r}")

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
# H3_TRANSFORMER_PREQUANT のメタデータ用の「定義時点のスナップショット」。
# diffusers の `TorchAoHfQuantizer._process_model_before_weight_loading()` は
# `quantization_config.modules_to_not_convert`(= 上のリストそのもの)を **in-place で
# extend する**(keep_in_fp32_modules の "rope" 追加や、複数回ロードでの重複追加。
# 2026-08-27 の実機 meta.json で確認)。そのため上のリストはプロセス内のロード回数に
# 応じて中身が変わってしまい、キャッシュ無効化メタデータの比較キーには使えない。
# 量子化の実効レシピ(「どの層を変換しないか」)は重複や "rope" の有無で変わらない
# (マッチ判定は any() なので同値)ため、pristine なスナップショットを比較キーにする。
_H3_INT8_MODULES_TO_NOT_CONVERT_PRISTINE = tuple(H3_INT8_MODULES_TO_NOT_CONVERT)

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
H3_TRANSFORMER_BOTH_RESIDENT = H3_TRANSFORMER_QUANT in ("int8", "ck-w4a8")

# 量子化済み transformer/transformer_ref のディスクキャッシュ (既定ON、H3_TE_PREQUANT と
# 同じ設計思想)。
#
# 動機: `H3_TRANSFORMER_QUANT=int8` は起動のたびに 66GB の bf16 重みを読み、
# `quantize_()` (torchao `Int8WeightOnlyConfig(version=2)`) でその場から int8 へ量子化
# する。この量子化はモデルが変わらない限り決定的な処理なので、**量子化後の重みを
# 一度保存しておけば次回以降は読むだけで済む**という H3_TE_PREQUANT と全く同じ理屈が
# そのまま当てはまる。
#
# 実現可能性は本タスクで実機検証済み (2026-08-27、Phase 1 probe):
# diffusers の `save_pretrained()`/`from_pretrained()` の既定 (`safe_serialization=True`)
# が、torchao>=0.16.0 の `flatten_tensor_state_dict`/`unflatten_tensor_state_dict`
# (safetensors ネイティブ経路、`torchao.prototype.safetensors.safetensors_support`) を
# 経由して `Int8Tensor` をそのまま安全に直列化できることを確認した。小型ダミー
# ModelMixin (Linear層 + modules_to_not_convert 相当の除外層) で
# 量子化 → save_pretrained → 別インスタンスへ from_pretrained → 固定入力での
# forward 出力を比較し、`torch.equal` で完全一致 (max_abs_diff 0.0)。
# H3_TE_PREQUANT の bnb-4bit 直列化 (transformers 側の実装) とは別の直列化機構
# (diffusers+torchao 側) だが、対応する版が両方ともビット一致で動くことを確認済み。
#
# TE 側との違い: TE は「設定ごとに別ディレクトリ」だけで衝突を防いでいたが、
# transformer は 2 インスタンス (`transformer` / `transformer_ref`) が別々の
# チェックポイント "スロット" (同一モデルクラス/config だが `_ensure_transformer` と
# `_ensure_transformer_ref` で個別にロード・保存される) を持つため、キャッシュも
# 個別ディレクトリにする (`models/prequant/transformer_int8/` /
# `models/prequant/transformer_ref_int8/`)。
#
# **保存する瞬間の注意**: `H3_TURBO_LORA` の構造的な Linear wrap や
# `_apply_turbo_setting` の遅延 turbo LoRA 適用より**前** (量子化直後、
# `_transformer_loaded = True` を立てる直前) に保存する。turbo LoRA は Linear モジュール
# 自体を差し替えるため、保存済みキャッシュに焼き込んでしまうと「turbo無効のはずの
# リクエストでも turbo 適用済みの重みしか手に入らない」事故になる。attention backend
# 切替 (属性代入のみ)・FBC (HookRegistry フック)・AdaLN precompute
# (`adaln_proj` サブモジュール差し替えは遅延実行、ロード時点ではまだ発生しない) は
# いずれも `state_dict()` に影響しないため、この保存点より後で構わない
# (本タスクで各実装のソースを確認して裏付け済み)。
#
# ディスク消費は各インスタンドあたり int8 で ~34GB (bf16 66GB の約半分)、2インスタンス
# で ~68GB。空きが `H3_TRANSFORMER_PREQUANT_MIN_FREE_GB` を下回る場合は保存をスキップし、
# 警告だけ出して**生成は続行する** (H3_TE_PREQUANT と同じ fail-open 方針、キャッシュは
# あくまで高速化であって機能ではない)。"0" で完全無効化できる。
H3_TRANSFORMER_PREQUANT = os.environ.get("H3_TRANSFORMER_PREQUANT", "1").strip() == "1"
H3_TRANSFORMER_PREQUANT_DIR = Path(
    os.environ.get("H3_TRANSFORMER_PREQUANT_DIR", str(H3_TE_PREQUANT_DIR))
)
# TE (17-21GB) よりシャードがはるかに大きい (~34GB/インスタンス) ため、TE の既定 25GB
# より大きい下限を既定にする。
H3_TRANSFORMER_PREQUANT_MIN_FREE_GB = float(
    os.environ.get("H3_TRANSFORMER_PREQUANT_MIN_FREE_GB", "40")
)
# 保存前に確認する空きホスト RAM の下限 (GB)。`save_pretrained` は
# `max_shard_size="10GB"` のシャード単位で `state.dict()` (GPU上のテンソルのまま) を
# safetensors へ直列化するため、34GB 全体を一度に CPU へコピーするわけではない
# (本タスクでソースを確認: `modeling_utils.py` の `save_pretrained` はシャードごとに
# `safetensors.torch.save_file(shard, ...)` を呼ぶだけで、GPU テンソルの CPU コピーは
# safetensors 内部がシャード単位で行う) が、念のため CLAUDE.md #33 と同じ流儀で
# 事前ガードを掛ける。
H3_TRANSFORMER_PREQUANT_MIN_RAM_GB = float(
    os.environ.get("H3_TRANSFORMER_PREQUANT_MIN_RAM_GB", "15")
)

# ref2va リクエストの終わりに、入口で解放した t2va 用 `transformer` を**その場で**
# 積み直すか (2026-08-12 に既定を「積み直さない」へ変更)。
#
# 旧既定 (eager) の問題: 復元は**そのリクエストの所要時間に含まれる**ため、ユーザーは
# 自分の動画とは無関係なロードを待たされる (実測 12.5s、初回は 36.9s)。しかも次も
# ref2va なら入口でまた解放されるので**完全な無駄**であり、復元した瞬間に両 transformer
# が載って VRAM 高水位が 49GB → 74.3GB へ跳ね上がる (48GB 級で運用する場合に致命的)。
# 元のコメントが根拠にしていた収支は 32B TE (21GB) 前提のもので、投影TE (3.11GB) を
# 使う現在の既定構成には当てはまらない。
#
# 遅延 (既定) にしても壊れない理由: t2va リクエストは入口で `_switch_to_variant("t2va")`
# → `_ensure_transformer()` を必ず通り、未ロードならそこでロードする (冪等)。つまり
# コストは**消える**のではなく、**それを必要とするリクエスト側へ移る**。連続 ref2va では
# まるごと消え、ref2va→t2va と切り替えたときだけ t2va 側が払う。
# 旧挙動に戻すには H3_EAGER_VARIANT_RESTORE=1。
H3_EAGER_VARIANT_RESTORE = os.environ.get("H3_EAGER_VARIANT_RESTORE", "0").strip() == "1"

# EXPERIMENTAL, opt-in. "0" (default) = every mode above is untouched -- this flag is
# read nowhere else unless it is "1" or "group". "1" = 48GB-class low-VRAM mode: TE
# (bnb-4bit nf4, ~21GB) and the big transformer (int8, ~34GB) are never allowed to be
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
# "group" = 24-32GB-class low-VRAM mode (see the H3_LOWVRAM_GROUP module comment
# further down for the full design, verified by scripts/probe_group_offload.py before
# being wired in here): instead of a full ~34GB int8 transformer ever being
# GPU-resident, the transformer is loaded once (CPU-resident, quantized in place via
# `device_map="cpu"` + torchao's `Int8WeightOnlyConfig` -- confirmed this does NOT hit
# torchao's cpu-offload skip-quantize path, since a plain string device_map becomes
# `{"": torch.device("cpu")}`, not a per-module dict with the *string* "cpu" as a
# value, which is the only thing `TorchAoHfQuantizer.validate_environment` checks for)
# and kept resident in host RAM for the life of the process via
# `enable_group_offload(block_level, num_blocks_per_group=1, use_stream=True)`, which
# streams ~1-2 of its 50 blocks (~0.68GB each) onto GPU at a time during denoise. This
# is diffusers' own decorator-based hook mechanism, not a CLAUDE.md-banned whole-module
# CPU<->GPU swap: the "resident" location for a group-offloaded module IS the CPU side,
# and the hooks manage small per-block GPU visits automatically.
#
# Requires H3_TRANSFORMER_QUANT=int8 (bf16's 66.3GB transformer alone is already
# larger than a 48GB card with headroom for anything else) -- if the transformer quant
# was left at its own default ("none") while H3_LOWVRAM is set, this is auto-upgraded
# to "int8" below (rather than silently running an unfittable bf16 config) UNLESS the
# operator *explicitly* set H3_TRANSFORMER_QUANT=none, in which case this raises at
# import time instead of silently overriding an explicit choice.
# H3_TRANSFORMER_BOTH_RESIDENT (both transformer AND transformer_ref resident at once,
# 34+34=68GB) is incompatible with either low-VRAM mode and is force-disabled below
# regardless of H3_TRANSFORMER_QUANT.
# upscale=1 (hires-fix) is rejected with a 400-mapped ValueError in both low-VRAM modes
# (see `generate()`) -- pass 2's ~4x-longer packed sequence was not verified to fit in
# the limited headroom either mode's steady state leaves at 24-48GB-class VRAM.
H3_LOWVRAM_RAW = os.environ.get("H3_LOWVRAM", "0").strip().lower()
if H3_LOWVRAM_RAW not in ("0", "1", "group"):
    raise ValueError(f"H3_LOWVRAM must be '0', '1' or 'group', got {H3_LOWVRAM_RAW!r}")
H3_LOWVRAM = H3_LOWVRAM_RAW == "1"
H3_LOWVRAM_GROUP = H3_LOWVRAM_RAW == "group"
H3_LOWVRAM_ANY = H3_LOWVRAM or H3_LOWVRAM_GROUP
if H3_LOWVRAM_ANY:
    _explicit_transformer_quant = "H3_TRANSFORMER_QUANT" in os.environ
    if _explicit_transformer_quant and H3_TRANSFORMER_QUANT == "none":
        raise RuntimeError(
            f"H3_LOWVRAM={H3_LOWVRAM_RAW!r} requires an int8 transformer (bf16's 66.3GB "
            "does not fit a 48GB-class card even alone, and group offloading a bf16 "
            "module would need ~66GB of host RAM just for the weights) but "
            "H3_TRANSFORMER_QUANT=none was explicitly set. Drop H3_TRANSFORMER_QUANT "
            "(it will default to int8 under H3_LOWVRAM) or set "
            "H3_TRANSFORMER_QUANT=int8 explicitly."
        )
    if H3_TRANSFORMER_QUANT == "none":
        H3_TRANSFORMER_QUANT = "int8"
    H3_TRANSFORMER_BOTH_RESIDENT = False
    # Every `H3_LOWVRAM`/`H3_LOWVRAM_GROUP` branch further down in this file
    # (generate()/generate_ref2va()) is written assuming TE_QUANT == "bnb-4bit" (it is
    # the only TE loading strategy that produces a small-enough, movable-only-by-full-
    # reload TE that these modes' choreography can work with -- `none` mode's 66.3GB
    # bf16-native TE would not fit alongside anything else on a 24-48GB-class card even
    # on its own). Reject the combination explicitly rather than silently
    # mis-choreograph an unfittable 66.3GB TE.
    if TE_QUANT != "bnb-4bit":
        raise RuntimeError(
            f"H3_LOWVRAM={H3_LOWVRAM_RAW!r} requires H3_TE_QUANT=bnb-4bit (default), "
            f"got H3_TE_QUANT={TE_QUANT!r}. bf16 TE (~66.3GB) cannot coexist with "
            "anything else on a 24-48GB-class card."
        )

# ---- AdaLN-pruned transformer_ref (2026-10-01、core/pruned.py 参照) -------------
# multimodalart/MiniMax-H3-Pruned: AdaLN 入力射影の構造リファクタ(13.03B 削減、
# ほぼロスレス)。ref2va 用 transformer_ref を pruned + int8 weight-only で読む
# (既定 int8wo: 4.60 s/step・peak 28GB 実測。常駐 ~21GB)。現状 ref2va(transformer_ref)専用 -- t2va 側の
# pruned 化は未実装(transformer/ サブフォルダを取得していない)。
H3_PRUNED = os.environ.get("H3_PRUNED", "0").strip() == "1"
H3_PRUNED_REPO = os.environ.get("H3_PRUNED_REPO", "multimodalart/MiniMax-H3-Pruned").strip()
H3_PRUNED_CONVROT_GROUP = int(os.environ.get("H3_PRUNED_CONVROT_GROUP", "256"))
# pruned transformer_ref の量子化方式(core/pruned.py の PRUNED_QUANT_SPECS)。既定は
# `int8wo`(torchao Int8WeightOnlyConfig(version=2, PerRow)、ConvRot なし。2026-10-01 に
# int8dyn-convrot から変更: 実測 4.60 vs 7.60 s/step、peak 28GB、品質同等。速度差の主因は
# torchao int8 動的量子化の eager 経路)。`int8dyn-convrot`(旧既定)・`int8wo-convrot`・
# `fp8`・`fp8-convrot`・`bf16` も選べる。キャッシュ dir 名・meta.json は方式ごとに別
# (旧既定 int8dyn-convrot のキャッシュも従来のまま使える)。
# 未知の値・torchao に無い config は初回リクエストまで持ち越さず、起動時にここで落とす。
if H3_TRANSFORMER_QUANT == "ck-w4a8" and not H3_PRUNED:
    raise RuntimeError(
        "H3_TRANSFORMER_QUANT=ck-w4a8 は現状 H3_PRUNED=1 と併用してください"
        "(非 pruned の transformer_ref への ck-w4a8 適用は未実装。base(t2va/fl2va)側のみ対応)。"
    )
H3_PRUNED_QUANT = os.environ.get("H3_PRUNED_QUANT", "int8wo").strip().lower()
if H3_PRUNED:
    from core import pruned as _pruned_mod

    H3_PRUNED_QUANT_SPEC = _pruned_mod.get_quant_spec(H3_PRUNED_QUANT)
    H3_PRUNED_QUANT_SPEC.check_available()
    H3_PRUNED_QUANT = H3_PRUNED_QUANT_SPEC.name
else:
    H3_PRUNED_QUANT_SPEC = None
# H3_PRUNED_COMPILE=1: ConvRot 系の方式(int8dyn-convrot 等)のとき、ロード後に 300 の ConvRot
# Linear(回転 + 量子化 GEMM)へ torch.compile(dynamic=True)を適用する(core/pruned.py の
# compile_convrot_layers)。torchao int8 動的量子化 eager 経路の未融合 per-token 量子化を
# 融合して GEMM を ~3 倍速くする。既定 0(完全に従来どおり)。ConvRot 無しの方式では無効。
_h3_pc = os.environ.get("H3_PRUNED_COMPILE", "0").strip()
# 1 = ConvRot Linear のみ compile。2 = さらに turbo LoRA ラッパー(base + LoRA 加算)も compile
# (実験: LoRA の mul/add を融合して elementwise の素通しを減らす)。
H3_PRUNED_COMPILE_LEVEL = int(_h3_pc) if _h3_pc in ("1", "2", "3") else 0  # 3 = probe: 2 + transformer ブロック単位 compile (attention は graph break)
H3_PRUNED_COMPILE = H3_PRUNED_COMPILE_LEVEL >= 1
if H3_PRUNED_COMPILE and not (H3_PRUNED and H3_PRUNED_QUANT_SPEC is not None and H3_PRUNED_QUANT_SPEC.convrot and H3_PRUNED_QUANT_SPEC.quantized):
    logging.getLogger("minimax_h3").warning(
        "H3_PRUNED_COMPILE=1 は H3_PRUNED=1 かつ ConvRot 系の量子化方式(H3_PRUNED_QUANT=int8dyn-convrot 等)"
        "でのみ有効です。現在の設定では無視します。"
    )
    H3_PRUNED_COMPILE = False
    H3_PRUNED_COMPILE_LEVEL = 0
if H3_PRUNED and H3_TRANSFORMER_QUANT not in ("int8", "ck-w4a8"):
    # pruned は ref2va の transformer_ref 専用で、その量子化方式は H3_PRUNED_QUANT が決める。
    # H3_TRANSFORMER_QUANT は t2va 側の transformer と両常駐(H3_TRANSFORMER_BOTH_RESIDENT)
    # / LOWVRAM 経路を決めるため別軸だが、pruned との組み合わせは int8 でしか検証して
    # いないので従来どおり int8 を要求する。
    raise RuntimeError(
        "H3_PRUNED=1 は H3_TRANSFORMER_QUANT=int8 と併用してください(pruned 側の量子化"
        "方式は H3_PRUNED_QUANT で選ぶ。H3_TRANSFORMER_QUANT は t2va 側/両常駐/LOWVRAM の"
        "経路を決める別軸で、pruned との組み合わせは int8 のみ検証済み)。"
    )
if H3_PRUNED and H3_LOWVRAM_GROUP:
    raise RuntimeError(
        "H3_PRUNED=1 と H3_LOWVRAM=group は併用できません(group offload 経路の "
        "pruned 対応は未実装。pruned は常駐 19.6GB なので group が必要な場面も薄い)。"
    )

# EXPERIMENTAL, opt-in. `H3_LOWVRAM=1` は毎リクエスト、デコード直前に transformer を
# 解放し次リクエストで再ロードする (実測 14.8-32.7s の固定費、RESIDENCY.md §5.5)。
# `H3_KEEP_TRANSFORMER=1` はその解放をスキップし、transformer をリクエスト間も
# GPU に常駐させたままにする -- 48GB級 (実効予算 ~49.8GB) では
# transformer(int8) 34.3GB + デコードピーク **fp16** 11.4GB = 45.7GB で入る見込み
# (未検証、余裕 ~4GB)。fp32 デコードピーク 16.29GB では 34.3+16.29=50.6GB で
# 入らないため、H3_VIDEO_VAE_FP16=1 が必須 (RESIDENCY.md §5.6)。
#
# **plain モード (H3_LOWVRAM=0) にも適用 (2026-08-12 に条件1を緩和)**: plain モードは
# transformer をリクエスト間は常駐させているが、**デコード窓だけは解放して直後に
# 再ロードする** (実測 11.9-12.3s/リクエスト)。この解放は「TE-nf4 21GB +
# transformer bf16 66.3GB + VAE fp32 11GB = 98.5GB > 96GB」という 32B TE 前提の収支から
# 来たもので、**TE が GPU0 に居ない構成 (条件2) では前提が成立しない**:
# 66.3 + fp16 デコード 11.4 = 77.7GB (投影TE を同居させても +3.11 で 80.8GB)。
# つまり条件2・3 がそのまま plain モードの成立条件でもあるので、条件1を
# 「group でないこと」に緩めるだけでよい (解放をスキップする分岐は共通、復元側の
# `_ensure_transformer` は冪等なので no-op になる)。bf16 のデノイズは int8 より
# 5-14% 速い (t2i 2.07s vs 2.40s) ため、96GB 級ではこちらが最速になりうる。
# **注意**: 66.3+11.4=77.7GB なので実質 80GB 級以上が必要 (48GB 級では bf16
# transformer 自体が載らないので自動的に対象外)。溢れた場合もデコードの try/except が
# steady state を復元してから re-raise する。
#
# 成立条件 (3つとも必須、欠けたら import 時に RuntimeError):
#   1) H3_LOWVRAM が "group" でないこと ("1" = 毎リクエストの再ロード固定費を削減、
#      "0"/plain = デコード窓の解放/再ロードを削減。"group" だけは対象外 --
#      そもそも transformer を常駐させたまま CPU/GPU 間を group offload する
#      別設計なので無関係)
#   2) H3_TE_DEVICE が設定済み (TE が別GPU)。そうでないと **デコード位相ではなく
#      エンコード位相が先に破綻する**: TE(bnb-4bit) 17.45GB + transformer(int8)
#      34.3GB = 51.75GB > 実効予算49.8GB で、prompt encode の時点で入らない
#      (transformer を常駐させたまま TE を同じGPUにロードしようとするため)
#   3) H3_VIDEO_VAE_FP16 == True (上記のとおり fp32 では 50.6GB で入らない)
# 既定 (H3_KEEP_TRANSFORMER=0) は 1バイトも挙動が変わらない -- 既存の
# H3_LOWVRAM=1 の「リクエスト間は何も常駐させない」定常状態のまま。
H3_KEEP_TRANSFORMER = os.environ.get("H3_KEEP_TRANSFORMER", "0").strip() == "1"
if H3_KEEP_TRANSFORMER:
    _keep_transformer_missing = []
    if H3_LOWVRAM_GROUP:
        _keep_transformer_missing.append(
            f"H3_LOWVRAM must be '1' or '0' (got {H3_LOWVRAM_RAW!r}) -- 'group' mode "
            "already keeps its transformer resident via a different (CPU+block-offload) "
            "design and is unrelated"
        )
    if not H3_TE_DEVICE and not H3_TE_PROJ:
        # この条件は **32B TE を前提にした収支** から来ている: TE-nf4 17.45GB +
        # 常駐 transformer-int8 34.3GB = 51.75GB で、実効予算 ~49.8GB を超えるため
        # デコードより先に**エンコード位相**が破綻する。だから「TE は別GPUへ」が必須だった。
        #
        # **投影TE (H3_TE_PROJ) はこの前提を満たさない**: NF4 で常駐 3.11GB (実測) なので
        # 3.11 + 34.03 = 37.1GB、デノイズ活性化 6.6GB を足しても 43.7GB で予算内に収まる。
        # つまり同一GPU上で TE と transformer を同時常駐させられる — H3_TE_DEVICE を
        # 要求する理由がない。投影TEのときはこのガードを免除する。
        _keep_transformer_missing.append(
            "H3_TE_DEVICE must be set (TE on a separate GPU) -- otherwise the *encode* "
            "phase (not decode) breaks first: TE-nf4 17.45GB + resident transformer-int8 "
            "34.3GB = 51.75GB, over the ~49.8GB effective budget. "
            "(Not required when H3_TE_PROJ is set: the projected TE is 3.11GB at NF4, so "
            "it fits on the same GPU alongside the transformer.)"
        )
    if not H3_VIDEO_VAE_FP16:
        _keep_transformer_missing.append(
            "H3_VIDEO_VAE_FP16 must be '1' -- fp32 decode peak 16.29GB + resident "
            "transformer-int8 34.3GB = 50.6GB, over the ~49.8GB effective budget "
            "(fp16 decode peak ~11.4GB fits: 34.3+11.4=45.7GB)"
        )
    if _keep_transformer_missing:
        raise RuntimeError(
            "H3_KEEP_TRANSFORMER=1 requires H3_LOWVRAM != 'group' AND (H3_TE_DEVICE set "
            "OR H3_TE_PROJ set) AND H3_VIDEO_VAE_FP16=1 (see this flag's module comment "
            "for the VRAM budget derivation). Missing: " + "; ".join(_keep_transformer_missing)
        )

# EXPERIMENTAL, opt-in (2026-10-01). `H3_KEEP_REF2VA=1`: ref2va の transformer_ref と
# TE をリクエスト間で GPU に常駐させる (`H3_LOWVRAM=1` 専用)。
#
# 背景: `H3_LOWVRAM=1` の ref2va は毎リクエスト「TE ロード → 参照エンコード → TE 解放 →
# transformer_ref ロード → denoise → transformer_ref 解放 → decode」を回すため、
# denoise 以外に ~43s の固定費がある (うち TE ロード 12.8s + transformer_ref ロード 13.2s)。
# H3_PRUNED=1 (AdaLN-pruned + int8wo) なら transformer_ref の常駐は 21.6GB、TE
# (nf4, prune) は 17.5GB なので、同一GPUへ両方載せたまま回せる見込みがある
# (denoise 活性化 +6.4GB)。このフラグは次の 4 箇所の解放だけを gate する:
#   (1) generate_ref2va() 入口の `_free_transformer_ref()`
#   (2) H3_LOWVRAM 分岐の「参照エンコード後の TE 解放」
#   (3) decode 窓の `_free_transformer_ref()`
#   (4) `H3_KEEP_REF2VA_VAE=1` のときだけ: VAE pair の CPU 退避 (既定 0 = 従来どおり
#       参照エンコード後と decode 後に CPU へ退避。VAE を GPU に置いたままにすると
#       denoise の予算が +5.8GB 増えるため、収支の取れる GPU だけで明示的に使う)
# 上記以外は一切触らない。既定 (H3_KEEP_REF2VA=0) は 1バイトも挙動が変わらない。
#
# 制約と設計判断:
#   - H3_LOWVRAM=1 以外は起動時エラー。LOWVRAM=0 (int8) は元々 transformer_ref+TE を
#     常駐させる設計 (H3_TRANSFORMER_BOTH_RESIDENT)、group は CPU 常駐で別設計のため。
#   - TE を別GPUへ置く構成 (H3_TE_DEVICE) や投影TE (H3_TE_PROJ) でも害はない
#     (`_free_text_encoder` が元々 no-op / 小さい)。KEEP_TRANSFORMER のような
#     「TE 別GPU必須」ガードは要らない: pruned の transformer_ref は 21.6GB で、
#     TE 17.5 + 21.6 = 39.1GB が 48GB 級の予算内に収まるため (KEEP_TRANSFORMER の
#     ガードは TE 17.45 + int8 transformer 34.3 = 51.75GB > 49.8GB が根拠だった)。
#   - **非 pruned** (transformer_ref int8 34.3GB) では TE 17.5 + 34.3 + 活性化 6.4 =
#     58.2GB 必要になり、48GB 級には載らない。起動時に GPU 総容量と突き合わせ、
#     足りなければ RuntimeError、足りるなら (96GB 級) 警告ログのみとする。
#   - VRAM が足りないときは素直に CUDA OOM にする。transformer を丸ごと CPU へ
#     スワップして凌ぐ実装は入れない (diffusers-server CLAUDE.md #33 の事故パターン)。
#   - 他リクエスト (t2va/fl2va/t2i/ref バッチ) は従来どおり `_free_transformer_ref()` で
#     transformer_ref を落とし、TE も force 解放するので、常駐は自然に解消される。
#     次の ref2va が `_load_text_encoder`/`_ensure_transformer_ref` (冪等) で再構築する。
#   - H3_REF_PREFIX_CACHE_SINGLE=1 のプレフィックス KV (~0.84GiB) は TE と寿命を共にする
#     (`_free_text_encoder` で捨てる) ので、TE が常駐するこのモードでは同一参照の
#     リクエスト間で HIT し続ける (VRAM +0.84GiB を常駐させる点に注意)。
H3_KEEP_REF2VA = os.environ.get("H3_KEEP_REF2VA", "0").strip() == "1"
H3_KEEP_REF2VA_VAE = os.environ.get("H3_KEEP_REF2VA_VAE", "0").strip() == "1"
if H3_KEEP_REF2VA:
    if H3_LOWVRAM_RAW != "1":
        raise RuntimeError(
            f"H3_KEEP_REF2VA=1 requires H3_LOWVRAM=1 (got {H3_LOWVRAM_RAW!r}): LOWVRAM=0 "
            "(int8) already keeps transformer_ref+TE resident by design, and 'group' keeps "
            "its transformer CPU-resident via a different design -- neither needs this flag."
        )
    _keep_ref2va_need_gb = 17.5 + (21.6 if H3_PRUNED else 34.3) + 6.4
    _keep_ref2va_total_gb = None
    try:
        if torch.cuda.is_available():
            _keep_ref2va_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:  # pragma: no cover - 診断用なので落とさない
        pass
    if _keep_ref2va_total_gb is not None and _keep_ref2va_need_gb > _keep_ref2va_total_gb * 0.98:
        raise RuntimeError(
            f"H3_KEEP_REF2VA=1 needs ~{_keep_ref2va_need_gb:.1f}GB resident+denoise "
            f"(TE 17.5 + transformer_ref {'pruned 21.6' if H3_PRUNED else 'int8 34.3'} + "
            f"activations 6.4) but this GPU has {_keep_ref2va_total_gb:.1f}GB total. "
            "Use H3_PRUNED=1 or drop H3_KEEP_REF2VA."
        )
    if not H3_PRUNED:
        logger.warning(
            "H3_KEEP_REF2VA=1 without H3_PRUNED=1: resident set ~%.1fGB (TE 17.5 + int8 "
            "transformer_ref 34.3 + denoise activations 6.4) -- only fits 80GB-class "
            "cards, will OOM on 48GB-class.", _keep_ref2va_need_gb,
        )
    if not H3_VIDEO_VAE_FP16:
        logger.warning(
            "H3_KEEP_REF2VA=1 without H3_VIDEO_VAE_FP16=1: the decode window must fit "
            "TE + transformer_ref + the fp32 VAE decode peak (16.3GB) -- set H3_VIDEO_VAE_FP16=1."
        )
    logger.info(
        "H3_KEEP_REF2VA=1: ref2va keeps text_encoder + transformer_ref resident across "
        "requests (pruned=%s, vae_resident=%s, prefix_cache_single=%s)",
        H3_PRUNED, H3_KEEP_REF2VA_VAE, os.environ.get("H3_REF_PREFIX_CACHE_SINGLE", "0"),
    )


def _keep_ref2va_active() -> bool:
    """H3_KEEP_REF2VA が今も有効か。`core.settings.apply_reload_settings()` が lowvram を
    実行時に書き換えうる (そのとき unload_all() で常駐は消える) ので、import 時の定数
    ではなく呼び出しのたびに LOWVRAM=1 かどうかも併せて見る。"""
    return H3_KEEP_REF2VA and H3_LOWVRAM


# `H3_KEEP_REF2VA=1` 専用 (2026-10-06)。**ref2va スタック (TE + transformer_ref) を常駐させたまま
# t2va/fl2va/t2i の base transformer を同居ロードする** (両常駐)。
#
# 動機 (実測): r-n-v の H3 運用では、会話ターンの合間に待機プール補充 (fl2va = base transformer)
# が走る。従来は fl2va の入口が `_free_transformer_ref()` で transformer_ref を、さらに encode 後に
# `_free_text_encoder(force=True)` で TE を解放していたため、次の会話ターンの先頭 ref2va が TE +
# transformer_ref の再ロードを払い、6秒 → 22〜40秒に劣化した。
# 収支 (RTX PRO 6000 96GB): TE 17.4 + transformer_ref(pruned) 22.2 + VAE ほか ~6 = ~46〜50GB 常駐
# + base transformer(int8) ~34.3 + denoise 活性化 ~6.4 = ~85〜90GB。96GB 級では収まるが、
# 32GB/48GB 級 (低VRAM構成) では収まらないので、**実行時に空きVRAMを測って判定**し、足りなければ
# 従来どおり解放する (黙って OOM させない)。判定根拠は `_keep_ref2va_coexist()` が INFO ログ1行で出す。
#   H3_KEEP_REF2VA_COEXIST=0                : 両常駐を使わない (従来挙動: 常に解放)
#   H3_KEEP_REF2VA_COEXIST_MIN_FREE_GB=42.7 : 同居に必要な空き (base 34.3 + 活性化 6.4 + 余裕 2.0)。
#                                             TE 未ロードなら +17.5GB を自動加算する。
# KEEP_REF2VA なし (既定) ではどちらも参照されず、挙動は 1 バイトも変わらない。
H3_KEEP_REF2VA_COEXIST = os.environ.get("H3_KEEP_REF2VA_COEXIST", "1").strip() != "0"
H3_KEEP_REF2VA_COEXIST_MIN_FREE_GB = float(os.environ.get("H3_KEEP_REF2VA_COEXIST_MIN_FREE_GB", "42.7"))


# EXPERIMENTAL, opt-in (2026-10-01). `H3_TE_DIET=1`: bnb-4bit text_encoder の VRAM ダイエット。
# LTX-2.5 の `LTX25_TE_DIET` (backends/ltx2_5/app/tediet.py) と同じ発想を H3 の TE
# (Qwen3-VL-32B, H3_TE_PRUNE=1 の 51 層) に適用する。H3 が読むのは
# `text_encoder.model(...)` の `hidden_states[50]` だけなので:
#   (1) `lm_head` (151936x5120 bf16 = 1.449GiB) は一度も呼ばれない (encoders.py も本ファイルの
#       `_encode_ref2va_prompt*` も `text_encoder.model` を直接呼ぶ。`tie_word_embeddings=false`
#       なので embed と共有でもない) -> ロード直後に重みを解放する。
#   (2) `embed_tokens` (151936x5120 bf16 = 1.449GiB, bnb は Linear しか量子化しないので bf16 の
#       まま GPU 常駐していた) -> モジュールごと CPU へ置き、forward を「input_ids を CPU へ ->
#       CPU で gather -> 呼び出し元デバイスへ戻す」ブリッジに差し替える (往復は数千トークン x
#       10KB = 数十MB 以下)。TE 全体を CPU へ動かすのではなく、埋め込みテーブル 1 枚だけの
#       配置換え (粒度が小さく、diffusers-server CLAUDE.md #33 の禁止パターンには当たらない)。
# 合計 常駐 -2.9GiB。**どちらも計算経路の外か値の等価な配置換えのみなので出力はビット一致**
# (CPU gather は同じ bf16 値を返し、lm_head は計算に参加しない)。既定 (0) は挙動不変。
# 適用対象は bnb-4bit の TE のみ (H3_TE_QUANT=none / H3_TE_PROJ は対象外)。
H3_TE_DIET = os.environ.get("H3_TE_DIET", "0").strip() == "1"

# EXPERIMENTAL, opt-in (2026-10-01). `H3_TE_STREAM=1`: bnb-4bit text_encoder の LM 層
# (H3_TE_PRUNE=1 の 51 層、~13GiB) を pinned host に置き、エンコード中だけ窓付きで層単位に
# GPU へ流す (LTX-2.5 の `LTX25_TE_STREAM`、`core/testream.py`)。TE が呼ばれるのは参照
# プレフィックスのエンコード (cache MISS 時) と各リクエストのプロンプト継続 forward だけで、
# denoise / decode の間は LM 層は使われない -> その間の GPU 常駐を ~13GiB 減らす。
# 層単位の移動 (モジュール丸ごとのスワップではない)・値は不変なので出力はビット一致。
# 適用は `_apply_te_diet` の直後 (同じ2箇所)。TE が計算用 GPU に載る bnb-4bit 構成専用
# (H3_TE_DEVICE 外部常駐・H3_TE_QUANT=none・H3_TE_PROJ では無視)。ホスト RAM は pinned
# 確保分 (~14GiB) + 余裕を MemAvailable で確認し、足りなければ RuntimeError。
# `H3_TE_STREAM_WINDOW` (既定 2 = LTX-2.5 と同じ): 先読みする層数。既定 (0) は挙動不変。
H3_TE_STREAM = os.environ.get("H3_TE_STREAM", "0").strip() == "1"
H3_TE_STREAM_WINDOW = max(1, int(os.environ.get("H3_TE_STREAM_WINDOW", "2").strip() or "2"))
H3_TE_STREAM_MIN_FREE_RAM_GB = float(os.environ.get("H3_TE_STREAM_MIN_FREE_RAM_GB", "10").strip() or "10")


def _apply_te_stream(text_encoder) -> float:
    """`H3_TE_STREAM`: 適用して pinned GiB を返す。冪等。呼び出し側が `not self._te_external` を保証する。"""
    from core.testream import apply_te_stream

    pinned = apply_te_stream(
        text_encoder, window=H3_TE_STREAM_WINDOW, min_free_ram_gb=H3_TE_STREAM_MIN_FREE_RAM_GB
    )
    if pinned > 0.0:
        gc.collect()
        torch.cuda.empty_cache()
    return pinned


# EXPERIMENTAL, opt-in (2026-10-01). `H3_REF_PREFIX_PARK=1`: `H3_REF_PREFIX_CACHE_SINGLE=1` の
# プレフィックス KV (~0.84GiB) を、エンコードしていない間は CPU (pinned) に置く。KV を使うのは
# 次のリクエストの `_encode_ref2va_prompt_prefix_cached()` の継続 forward だけで、denoise/decode
# の間は不要。HIT 時にその直前で GPU へ戻す (~100 テンソル / ~1GiB の PCIe 転送、同一値の
# 移動なので出力はビット一致)。`H3_KEEP_REF2VA=1` では TE (=キャッシュの寿命) が常駐するため
# キャッシュも常駐し続け、decode 窓の VRAM を 0.84GiB 食う -- それを返すためのフラグ。
# 既定 (0) は挙動不変。
H3_REF_PREFIX_PARK = os.environ.get("H3_REF_PREFIX_PARK", "0").strip() == "1"

# 診断用 (既定 0): ref2va の decode 窓の割り当て量とピークをログに出す。ピーク統計を窓の前で
# リセットするため (結果の peak_vram_gb は窓前のピークと合成して保つ) ふつうは使わない。
H3_DEBUG_DECODE_MEM = os.environ.get("H3_DEBUG_DECODE_MEM", "0").strip() == "1"

# EXPERIMENTAL, opt-in (2026-10-01). `H3_VAE_SPLIT=1` (`H3_KEEP_REF2VA=1` 専用): ref2va で VAE を
# GPU へ送る 2 つの窓のうち、参照エンコード窓には encode 側 (video encoder 0.34GiB + audio
# encoder 系 ~0.3GiB)、デコード窓には decode 側 (video decoder 4.5GiB + audio decoder ~0.25GiB)
# だけを送る。従来は両方の窓で pair 全体 (6.3GiB) を送っており、KEEP_REF2VA のように TE +
# transformer_ref が常駐している間は **この 2 窓が VRAM のピークを決めていた** (実測: 窓前の
# 割り当て 41.1GB -> 窓内 46.9GB、denoise 自体は窓より低い)。配置換えのみなので出力はビット一致。
# 既定 (0) は挙動不変。
H3_VAE_SPLIT = os.environ.get("H3_VAE_SPLIT", "0").strip() == "1"

# `H3_VAE_PINNED=1`: VAE の GPU<->CPU 退避 (`_vae_to_gpu`/`_vae_to_cpu`) を pinned CPU
# マスター方式にする。VAE の重みは推論中に変化しないので、初回に pinned CPU コピー
# (マスター) を作って以後は (1) ->CPU は `p.data` をマスターへ付け替えるだけ
# (D2H コピー自体を省略)、(2) ->GPU は pinned からの高速 H2D、にできる。
# 動機 = 単機 32GB 構成 (KEEP_REF2VA_VAE=0 で毎チャンク pageable 往復 2.4s が固定費、
# 2026-10-07 実測: ->GPU 0.83s + ->CPU 1.61s @ 141f)。ホスト RAM を VAE サイズぶん
# (~5-6GiB) pinned で常駐消費する。既定 0 = 従来どおり `.to()` で挙動不変。
H3_VAE_PINNED = os.environ.get("H3_VAE_PINNED", "0").strip() == "1"

# `H3_BASE_PINNED=1`: base transformer (t2va/fl2va) の解放・再ロードを pinned CPU
# マスター方式の退避に置き換える (H3_VAE_PINNED と同型のポインタ付け替え)。
# H3_LOWVRAM の振り付けは待機 (fl2va) のたびに base を `_free_transformer()` ->
# `_ensure_transformer()` で作り直すが、ck-w4a8 base はこの再ロードが毎回 7.7〜8.4s
# (12.5GB キャッシュの torch.load + H2D。int8 時代の 4.2s より悪化) かかる。重みは
# 推論中に不変なので、モジュールを壊さず保持し、重み実体だけを pinned CPU マスターと
# GPU の間で付け替える: 退避は `p.data = master` のみ (D2H 無し・一時二重化無し =
# CLAUDE.md #33 の丸ごと `.to()` スワップとは異なり RAM スパイクしない)、復帰は
# pinned H2D (~12.5GB で実測 0.5s 級)。turbo LoRA ラップ・attention backend 設定も
# モジュールごと保持されるので再適用も不要になる。
# ck-w4a8 (pruned base ~12.5GB) 専用: int8 base (~33GB) を pinned 常駐させる設計は
# 未検証のため、quant が ck-w4a8 以外 (または group mode) では無効化して警告する。
# ホスト RAM を ~12.5GiB pinned で常駐消費する。既定 0 = 従来どおり。
H3_BASE_PINNED = os.environ.get("H3_BASE_PINNED", "0").strip() == "1"
if H3_BASE_PINNED and (H3_TRANSFORMER_QUANT != "ck-w4a8" or H3_LOWVRAM_GROUP):
    logger.warning(
        "H3_BASE_PINNED=1 は H3_TRANSFORMER_QUANT=ck-w4a8 (非 group) 専用です "
        "(quant=%r, group=%s) -- 無効化して従来のロード経路を使います",
        H3_TRANSFORMER_QUANT, H3_LOWVRAM_GROUP,
    )
    H3_BASE_PINNED = False

# `H3_REF_PINNED=1`: transformer_ref (ref2va) にも H3_BASE_PINNED と同じ pinned CPU
# マスター退避を適用する。H3_LOWVRAM の低VRAM構成 (TE_STREAM/VAE_SPLIT 系) では
# decode 窓と fl2va 入口の `_free_transformer_ref()` が走るため、**連続する会話ターン
# ですら毎回 ck キャッシュの再ロード ~9.5s を払っていた** (2026-10-08 の 32GB/24GB 窓
# サイクル実測)。pinned 退避で park 0.3s / restore ~2s に縮む。
# ck-w4a8 pruned ref (~13GB) 専用 (int8wo pruned 20GB の pinned 常駐は未検証)。
# ホスト RAM を ~13.6GiB pinned で常駐消費する。既定 0 = 従来どおり。
H3_REF_PINNED = os.environ.get("H3_REF_PINNED", "0").strip() == "1"
if H3_REF_PINNED and (not H3_PRUNED or H3_PRUNED_QUANT != "ck-w4a8" or H3_LOWVRAM_GROUP):
    logger.warning(
        "H3_REF_PINNED=1 は H3_PRUNED=1 + H3_PRUNED_QUANT=ck-w4a8 (非 group) 専用です "
        "(pruned=%s, pruned_quant=%r, group=%s) -- 無効化して従来のロード経路を使います",
        H3_PRUNED, H3_PRUNED_QUANT, H3_LOWVRAM_GROUP,
    )
    H3_REF_PINNED = False

# `H3_FL2VA_KEEP_TE=1`: H3_LOWVRAM の t2va/fl2va が encode 後に行う
# `_free_text_encoder(force=True)` をスキップし、TE (bnb-4bit、TE_STREAM 後の GPU
# 常駐 ~4GB) を待機生成の間も残す。TE を解放すると次の会話ターンが TE 再ロード
# (8s) + TE_DIET/TE_STREAM 再適用 (7s) + ref-prefix 再エンコード (3s、prefix cache
# は TE free で消される) を払う (2026-10-08 実測、speech 定常 43s の主因)。
# int8 base 33GB 時代は TE と base の同居余地が無く解放が必須だったが、ck-w4a8
# base (13GB) なら fl 窓 peak +~4.3GB で済む (24GB 窓でも収支が合う)。
# 既定 0 = 従来どおり。KEEP_REF2VA 両常駐 (coexist) が成立する環境では不要
# (coexist が TE ごと残すため、このフラグは coexist 不成立時の部分的 keep)。
H3_FL2VA_KEEP_TE = os.environ.get("H3_FL2VA_KEEP_TE", "0").strip() == "1"


def _apply_te_diet(text_encoder) -> float:
    """`H3_TE_DIET`: lm_head の重みを解放し、embed_tokens を CPU ブリッジ化する。

    Returns: GPU から解放された GiB。冪等 (2回目以降は 0.0)。
    """
    inner = getattr(text_encoder, "model", None)
    lang = getattr(inner, "language_model", None) if inner is not None else None
    embed = getattr(lang, "embed_tokens", None) if lang is not None else None
    if embed is None or getattr(embed, "_h3_te_diet", False):
        return 0.0
    freed = 0
    # (1) lm_head: 使われないので、1 要素のダミー重みに差し替えて実体を解放する
    #     (属性ごと消すと `PreTrainedModel` の tie/重み管理が参照して壊れうるため、形だけ残す)。
    lm_head = getattr(text_encoder, "lm_head", None)
    w = getattr(lm_head, "weight", None)
    if w is not None and not getattr(w, "is_meta", False):
        freed += w.numel() * w.element_size()
        lm_head.weight = torch.nn.Parameter(
            torch.empty(1, 1, dtype=w.dtype, device=w.device), requires_grad=False
        )
    # (2) embed_tokens: CPU ブリッジ。`embed.weight.device` が cpu でも `text_encoder.device`
    #     (最初のパラメータ = vision tower) は変わらない。
    orig_forward = embed.forward
    freed += embed.weight.numel() * embed.weight.element_size()

    def _bridged_embed_forward(input_ids: torch.Tensor) -> torch.Tensor:
        return orig_forward(input_ids.to("cpu")).to(input_ids.device)

    embed.to("cpu")
    embed.forward = _bridged_embed_forward
    embed._h3_te_diet = True
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
    freed_gib = freed / 1024**3
    logger.info(
        "[H3_TE_DIET] lm_head released + embed_tokens -> CPU bridge: %.2fGiB freed from GPU", freed_gib
    )
    return freed_gib


# "group" mode's own RAM guard (see H3_LOWVRAM_GROUP's design comment further down):
# the int8 transformer (~34GB) is loaded once and stays resident in host RAM for the
# life of the process (unlike H3_LOWVRAM=1's per-request from-scratch reload) --
# refuse to even attempt that load if host RAM is already tight, rather than silently
# risking the swap-storm/OOM-killer incident CLAUDE.md #33 (this project's sibling
# diffusers-server repo) documents from a past project loading a large module the
# wrong way. Checked once, right before the first group-offload transformer load
# (`_ensure_transformer_group`), not at import time (RAM usage can shift between
# process start and first request).
# 40GB (default) covers the *unpinned* CPU load (~34GB, `H3_GROUP_OFFLOAD_LOW_CPU_MEM=1`)
# with a thin margin, but the *default* path (`H3_GROUP_OFFLOAD_LOW_CPU_MEM=0`, see that
# var's own comment for why it is the default despite the name) eagerly pins the whole
# transformer at `enable_group_offload()` time right after, measured to cost an
# additional ~14-16GB of available RAM on top of the ~32GB the plain CPU load itself
# used (avail_gb dropped 61.0->58.8 during load, then 58.8->45.3 during the pin step, in
# this task's own probe against the real transformer) -- so the *actual* peak
# requirement for the default configuration is closer to ~48GB than 40GB. 40GB is kept
# as the floor (matches this var's literal meaning: do not even start the CPU load
# below this) rather than raised to 48GB by default, since `H3_GROUP_OFFLOAD_LOW_CPU_MEM=1`
# remains available as an explicit lower-RAM (but slower-denoise) opt-out for boxes
# between 40-48GB of RAM -- raise this explicitly (e.g. to 48) if running the default
# (pinned) configuration on such a box.
H3_GROUP_OFFLOAD_MIN_RAM_GB = float(os.environ.get("H3_GROUP_OFFLOAD_MIN_RAM_GB", "40"))

# "group" mode's `enable_group_offload()` knobs (see `_ensure_transformer_group`'s
# docstring for the full design). `num_blocks_per_group=1` is diffusers-server's
# (this project's sibling repo, CLAUDE.md #33/#34/#37) own verified default for
# transformer group offloading -- the finest-grained onload unit, minimizing the
# resident-on-GPU footprint at the cost of more (smaller) PCIe round trips per step.
# `use_stream=True` overlaps the *next* group's H2D copy with the *current* group's
# compute via a dedicated CUDA stream (double-buffered prefetch), trading ~1 extra
# block's worth of GPU memory for less stalling on the copy.
H3_GROUP_OFFLOAD_BLOCKS = int(os.environ.get("H3_GROUP_OFFLOAD_BLOCKS", "1"))
H3_GROUP_OFFLOAD_USE_STREAM = os.environ.get("H3_GROUP_OFFLOAD_USE_STREAM", "1").strip() == "1"

# `low_cpu_mem_usage` for `enable_group_offload()` -- default "0" (i.e. `False`), the
# OPPOSITE of what its name suggests is the safe default, for a reason found and
# verified empirically during this task (scripts/probe_group_offload_forward.py /
# scripts/probe_group_offload_fix.py), not assumed from the diffusers docs:
# `low_cpu_mem_usage=True` (diffusers' own default) skips eagerly pinning
# `cpu_param_dict`'s tensors at `enable_group_offload()` time (`_to_cpu()`,
# hooks/group_offloading.py), deferring pinning to every single onload instead
# (`_pinned_memory_tensors()`, called from `_onload_from_memory()` whenever
# `use_stream=True`). For torchao's `Int8Tensor` (this mode's transformer weight type),
# that deferred pin path is broken: `Int8Tensor.qdata.pin_memory()` raises `RuntimeError:
# cannot pin 'torch.cuda.CharTensor' only dense CPU tensors can be pinned` on every
# single denoise step's block onload -- reproduced first against the real server (a
# t2va request failing inside the FIRST transformer_blocks forward) and then isolated
# down to a minimal dummy int8-quantized nn.Linear stack, confirming
# `use_stream=True + low_cpu_mem_usage=True` is the unconditional trigger (both
# `use_stream=False` and `low_cpu_mem_usage=False` independently avoid it -- see the
# probe scripts' own output for the full traceback and A/B). `low_cpu_mem_usage=False`
# was chosen over `use_stream=False` as this mode's actual default because it also
# measured ~4-5x faster per-block onload against the real transformer (pinned-memory
# H2D copies do not need to wait on a pageable-memory staging copy first): 0.04-0.07s
# vs 0.1-0.26s onload, and offload dropped to ~0s (pinned `cpu_param_dict` tensors are
# reused directly instead of a fresh `.to(cpu)` copy each time). The cost is paid once,
# up front, at `enable_group_offload()` time instead of amortized per-step: pinning the
# full ~34GB int8 transformer took an extra ~22s and reduced available host RAM by
# ~15.7GB in that same measurement (page-locked memory cannot be swapped out, unlike the
# `low_cpu_mem_usage=True` path's plain pageable CPU tensors) -- `H3_GROUP_OFFLOAD_MIN_RAM_GB`'s
# guard (checked before this load starts) accounts for this. Exposed as an env var
# rather than hardcoded so an operator on a truly RAM-starved box can opt back into the
# slower-but-lower-RAM `low_cpu_mem_usage=True` path if needed -- but note doing so
# still requires `H3_GROUP_OFFLOAD_USE_STREAM=0` as well (set automatically below,
# since `low_cpu_mem_usage=True` + `use_stream=True` together are exactly the broken
# combination) or the pin_memory() crash returns.
H3_GROUP_OFFLOAD_LOW_CPU_MEM = os.environ.get("H3_GROUP_OFFLOAD_LOW_CPU_MEM", "0").strip() == "1"
if H3_GROUP_OFFLOAD_LOW_CPU_MEM and "H3_GROUP_OFFLOAD_USE_STREAM" not in os.environ:
    H3_GROUP_OFFLOAD_USE_STREAM = False

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

# Opt-in. "0" (default) = the transformer is exactly the base bf16/int8 checkpoint,
# byte-for-byte identical to pre-this-flag behaviour (the LoRA is never downloaded or
# applied). "1" = `larryvrh/MiniMax-H3-Turbo-Lora`'s `minimax_h3_turbo_4step.safetensors`
# (the trained, non-EMA variant -- the author's own README calls the EMA sibling "less
# mature") is downloaded and applied to `transformer` as an unfused, run-time low-rank
# delta (`base(x) + B(A(x))`, alpha == rank so no extra scale factor -- matches the LoRA
# author's own reference `generate.py`, which this project's `_apply_turbo_lora()`
# mirrors module-for-module, see that function's docstring for the full key-mapping
# derivation) right after the transformer's normal load. NOT fused into the base
# weight (`core/loaders.py`-style fuse+cast would round most of a rank-64 delta away
# against a 66GB bf16 base, per the LoRA author's own `LoRALinear` docstring). This
# flag lets the *default* transformer path (bf16, `H3_TRANSFORMER_QUANT=none`) opt in
# without touching `transformer_ref` (ref2va is out of scope, see task brief) or the
# int8/lowvram branches -- CONFIRMED incompatible with those (not just unverified),
# by a follow-up task's own A/B run (2026-08-06): turbo=1 combined with any path where
# `H3_TRANSFORMER_QUANT=int8` (i.e. `H3_LOWVRAM_ANY` or `H3_TRANSFORMER_BOTH_RESIDENT`)
# reproducibly raises `NotImplementedError: Int8Tensor dispatch: ... aten.cat ...`
# inside `apply_turbo_lora()`'s `fuse_projections()` call -- torchao's `Int8Tensor` (the
# transformer's `to_q`/`to_k`/`to_v` under int8 quant; `H3_INT8_MODULES_TO_NOT_CONVERT`
# does not skip them) has no registered `aten.cat` kernel, so `torch.cat([to_q.weight,
# to_k.weight, to_v.weight])` fails outright, before any group-offload hook is even
# consulted (so this is not the "wrap LoRA before enable_group_offload()" ordering fix
# that helped a sibling project, diffusers-server CLAUDE.md #44 -- reordering cannot fix
# a missing kernel). Confirmed identical for `H3_LOWVRAM=1` and `H3_LOWVRAM=group` (both
# force int8), each failing loudly with a 500 and no VRAM leak or lasting corruption
# (a follow-up plain, non-turbo generation succeeded right after on the same server) --
# rejected below with a loud error rather than silently risking a wrong quantize-then-
# adapt order, the failure mode CLAUDE.md #47's "LoRA loaded after fp8 cast raises
# NotImplementedError" entry warns about for a sibling project's own fp8 base.
#
# turbo=1 + upscale=1 (hires-fix), by contrast, IS verified to work (same task, see
# `core/settings.py`'s `validate_instant_settings_for_upscale()` docstring for the full
# numbers) -- no structural conflict: `apply_instant_settings()`'s turbo wrap runs once
# the transformer is confirmed resident and well before hires-fix's own two-pass split,
# and hires-fix's FBC bookkeeping calls are already no-ops whenever turbo forces
# `effective_cache` to "none".
#
# Turbo changes three more things, all gated on this same flag (see `generate()`):
# the default `num_inference_steps` becomes 8 (matches the LoRA author's community-
# verified "8 steps works, 4-7 does not" finding, itself hedged in the README as
# possibly a ComfyUI-sampler artifact rather than a LoRA limit -- this project's own
# verification found no audio breakage even at 4 steps, see README); FirstBlockCache
# (`H3_CACHE=fbc`) is force-disabled regardless of its own env var (a handful of steps
# leaves no redundant-computation window for FBC's residual-similarity skip to safely
# exploit, and caching on top of an already-4-8-step trajectory risks compounding drift
# for no measured benefit); the video/audio schedulers' `shift` is left completely
# untouched (both already default to 12.0/3.0 -- `scheduler_config.json` on disk and
# `MiniMaxH3SetTimestepsStep`'s own docstring both confirm this, and this task's own
# verification found the LoRA author's reference sampler uses the identical two
# constants -- so there is nothing to reconfigure here, unlike a naive port from a
# scheduler that defaults elsewhere).
H3_TURBO_LORA = os.environ.get("H3_TURBO_LORA", "0").strip() == "1"
# 既定 LoRA は 2026-08-08 に lightx2v/Minimax-h3-Turbo (DMD蒸留、Apache 2.0) へ切替。
# キーが diffusers ネイティブ (`transformer_blocks.N.attn.to_q.lora_A.default.weight`
# 形式、to_q/to_k/to_v 分離) なので `fuse_projections()` が不要で、**int8 量子化
# transformer (H3_LOWVRAM/両常駐) にもそのまま適用できる** -- Ostris 版 (comfy 融合QKV
# 形式) を int8 で阻んでいた `Int8Tensor` の `aten.cat` 非互換を踏まない
# (README「Turbo LoRA 完成版のリリース待ち → lightx2v 版」節のスパイク実測参照)。
# 旧 Ostris 版に戻すには REPO/FILE を larryvrh/MiniMax-H3-Turbo-Lora /
# minimax_h3_turbo_4step.safetensors にする (bf16 経路専用のまま)。
# 既定ファイルは 2026-08-12 に v0.1 -> 4step v1.0 768p へ切替 (README「2026-08-12」節の
# スパイクで確認済み: 768p 版のほうが同じ4stepsで品質が上、scale/shift は下の
# resolve_turbo_lora_scale()/H3_TURBO_VIDEO_SHIFT が metadata/ファイル名から自動導出する
# ので追加の env 指定は不要)。旧 v0.1 に戻すには
# H3_TURBO_LORA_FILE=minimax_h3_fl2v_turbo_4step_v0.1.safetensors を明示すればよい
# (scale はその場で metadata フォールバック=0.094、shift は自動切替なしに戻る)。
H3_TURBO_LORA_REPO = os.environ.get("H3_TURBO_LORA_REPO", "lightx2v/Minimax-h3-Turbo")
H3_TURBO_LORA_FILE = os.environ.get(
    "H3_TURBO_LORA_FILE", "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
)
# 【2026-10-07 追加】base transformer (t2va/fl2va) 専用の turbo LoRA 指定。未指定 (既定) は
# 従来どおり上の REPO/FILE が base/ref 両 transformer に共通適用される (挙動不変)。
# 動機: 両常駐運用 (run.sh 既定 = ref2v 8step) だと待機 (fl2va = base transformer) にも
# ref2v 用 LoRA が転用され、かつファイル名が `_fl2v_` でないため video shift 6 の自動切替も
# 効かない。`H3_TURBO_LORA_FILE_BASE=minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors`
# なら base だけ fl2v 4step LoRA + shift 6、ref2va 側は従来どおり。
H3_TURBO_LORA_REPO_BASE = os.environ.get("H3_TURBO_LORA_REPO_BASE", "").strip() or H3_TURBO_LORA_REPO
H3_TURBO_LORA_FILE_BASE = os.environ.get("H3_TURBO_LORA_FILE_BASE", "").strip() or H3_TURBO_LORA_FILE

# 既知の comfy (融合QKV) 形式リポジトリ。int8 との組み合わせ拒否 (import 時と
# リクエスト時の両方) はこの形式のときだけ必要 -- diffusers ネイティブ形式は
# fuse_projections を呼ばないため int8 でも適用できる (スパイク実測済み)。
# 形式の確定判定はファイルのキーを見る `detect_turbo_lora_format()` (apply 時) で行い、
# ここではダウンロード前でも判定できるようリポジトリ名で予備判定する。
_TURBO_COMFY_REPOS = ("larryvrh/MiniMax-H3-Turbo-Lora",)

# 未指定時はリポジトリの形式に連動: lightx2v 版 (diffusers ネイティブ) は 4step 蒸留
# なので 4、Ostris 版 (comfy) はコミュニティ検証どおり 8 (「4-7 steps はダメ」)。
# REPO だけ Ostris に戻したデプロイが黙って 4steps に落ちる事故を防ぐ (レビュー指摘)。
H3_TURBO_STEPS_DEFAULT = int(
    os.environ.get("H3_TURBO_STEPS_DEFAULT", "").strip()
    or (8 if H3_TURBO_LORA_REPO in _TURBO_COMFY_REPOS else 4)
)
# LoRA デルタの適用係数。空 (既定) はチェックポイント形式ごとの実測既定に解決する:
# comfy (Ostris) = 1.0 (alpha==rank で scale 1 が作者実装どおり)、diffusers ネイティブ
# (lightx2v) = 0.094。lightx2v 版の罠 (スパイクで実測): Kijai のカードにある
# 「strength 0.75」は ComfyUI が alpha を折り込んで適用する前提の値で、生の B・A に
# 直接掛ける本実装では 0.75 × (alpha/rank) = 0.75 × 16/128 ≈ 0.094 が対応値。
# 0.75 をそのまま掛けると **30 steps でも出力が完全にノイズ化する** (強度スイープの
# 実測表は README 参照。0.094 が最良、0.10-0.15 も可)。
H3_TURBO_LORA_SCALE_RAW = os.environ.get("H3_TURBO_LORA_SCALE", "").strip()

# group offload (H3_LOWVRAM=group) と turbo は形式を問わず併用禁止: diffusers の
# `enable_group_offload()` は有効化時点の parameters/buffers を pinned CPU 辞書
# (`cpu_param_dict`) に固定登録するため、その後から `_TurboLoRALinear` で lora_a/lora_b
# バッファを追加すると offload/onload サイクルで辞書に無いバッファを引いて壊れる
# (KeyError または GPU 残留) 可能性が高い -- 未実測のまま解禁しない (レビュー指摘。
# hooks/group_offloading.py の `_init_cpu_param_dict()` が構築時1回きりであることを確認)。
if H3_TURBO_LORA and H3_LOWVRAM_GROUP:
    raise RuntimeError(
        "H3_TURBO_LORA=1 と H3_LOWVRAM=group は併用できません (enable_group_offload の "
        "cpu_param_dict は有効化時点で固定されるため、後から追加される LoRA バッファが "
        "offload サイクルから欠落するリスクがある -- 未検証)。H3_LOWVRAM=1 を使ってください。"
    )

# スケジューラの exponential shift の上書き (既定は空 = 触らない)。H3 の既定は
# video 12.0 / audio 3.0 (scheduler_config.json)。lightx2v turbo LoRA の訓練 shift は
# **fl2v 系の `_768p` バリアントだけが video shift 6** で、それ以外 (fl2v 非768p、
# ref2v 系は 768p 表記を含め全て) は基底と同じ 12 / 3
# (上流 ModelTC/Minimax-H3-Turbo の Model specs 表 + ref2v 8step v1.0 768p は
# HF discussions/51 の公式推奨 "video shift 12 / audio shift 3" で確認。2026-09-09)。
# fl2v 768p 系を使うときは `H3_VIDEO_SHIFT=6` を併せて指定しないとサンプリング格子が
# 蒸留時とずれる。適用箇所は `_ensure_vaes` のスケジューラロード直後 (プロセスに1回。
# scheduler は _pipe/_pipe_ref で同一オブジェクトを共有するため1箇所で足りる)。
H3_VIDEO_SHIFT = os.environ.get("H3_VIDEO_SHIFT", "").strip()
H3_AUDIO_SHIFT = os.environ.get("H3_AUDIO_SHIFT", "").strip()

# turbo LoRA が有効なリクエストにだけ適用する video shift の上書き (既定は空 =
# ファイル名から自動判定)。H3_VIDEO_SHIFT (上) はプロセス全体・全リクエスト共通の
# 上書きなのに対し、こちらは「turbo=1 のリクエストのときだけ shift を切り替え、
# turbo=0 のリクエストでは配布既定 (またはH3_VIDEO_SHIFTがあればそれ) に戻す」
# リクエスト単位の切替 -- turbo は `cache`/`attn` と同じ instant-apply (リクエスト
# ごとに on/off できる) なので、shift もそれに追従する必要がある (固定してしまうと
# turbo=0 のリクエストが誤った格子で走る)。
# 解決規則: 明示指定があればその値を最優先。空なら
# H3_TURBO_LORA_FILE のファイル名が「fl2v 系かつ `_768p`」のときだけ 6 に切り替える
# (video shift 6 で蒸留されているのは fl2v の 768p バリアントだけ -- 上の
# H3_VIDEO_SHIFT のコメント参照)。該当しなければ「切替なし」(turbo=1 でも
# 配布既定/H3_VIDEO_SHIFT のまま)。
# 【2026-09-09 修正】旧規則は `_768p` だけを見ていたため、ref2v 8step v1.0 768p
# (訓練/推奨 shift 12) に shift 6 を誤適用していた (gateway ログ 2026-09-03 21:52:11
# で実害を確認 -- mv_studio_V3 の balance tier がこの誤 shift で走っていた)。
# A/B 実測は outputs/ab_8step4_20260909/README.md 参照。
H3_TURBO_VIDEO_SHIFT_RAW = os.environ.get("H3_TURBO_VIDEO_SHIFT", "").strip()
if H3_TURBO_VIDEO_SHIFT_RAW:
    H3_TURBO_VIDEO_SHIFT: float | None = float(H3_TURBO_VIDEO_SHIFT_RAW)
elif "_fl2v_" in H3_TURBO_LORA_FILE and "_768p" in H3_TURBO_LORA_FILE:
    H3_TURBO_VIDEO_SHIFT = 6.0
else:
    H3_TURBO_VIDEO_SHIFT = None
# base transformer (generate/generate_still_batch) 用の shift。明示指定 (RAW) は base/ref 共通、
# なければ H3_TURBO_LORA_FILE_BASE のファイル名から判定 (BASE 未指定なら上と同一 = 不変)。
if H3_TURBO_VIDEO_SHIFT_RAW:
    H3_TURBO_VIDEO_SHIFT_BASE: float | None = H3_TURBO_VIDEO_SHIFT
elif "_fl2v_" in H3_TURBO_LORA_FILE_BASE and "_768p" in H3_TURBO_LORA_FILE_BASE:
    H3_TURBO_VIDEO_SHIFT_BASE = 6.0
else:
    H3_TURBO_VIDEO_SHIFT_BASE = None
if H3_TURBO_LORA and H3_TURBO_LORA_REPO in _TURBO_COMFY_REPOS and (H3_LOWVRAM_ANY or H3_TRANSFORMER_BOTH_RESIDENT):
    raise RuntimeError(
        "H3_TURBO_LORA=1 with the comfy-format (fused-QKV) LoRA "
        f"({H3_TURBO_LORA_REPO}) is only supported against the default transformer "
        "path (H3_TRANSFORMER_QUANT=none, H3_LOWVRAM=0): apply_turbo_lora()'s "
        "fuse_projections() call does torch.cat() on the transformer's to_q/to_k/to_v "
        "weights, and those are torchao Int8Tensor under transformer_quant=int8 -- "
        "Int8Tensor has no aten.cat kernel, so this reproducibly raises "
        "NotImplementedError. Use the default diffusers-native LoRA "
        "(lightx2v/Minimax-h3-Turbo) instead, or drop the other flag."
    )

# HyperFlow (Video Rebirth の 8-step flow-map 蒸留 LoRA、2026-09-26 統合、A/B 検証用)。
# rank256/alpha256(scale=1.0)の PEFT LoRA + TwoTimeEmbedder(現在時刻 t と step 終端 r の
# 2時刻条件付け、gate=0.25)+ 焼き込み9点σグリッド(8 NFE、shift はスケジューラ設定値を
# 実行時に適用 -- 既定 12/3 は蒸留時と同一)。**ref2va 専用**(t2va/fl2va は generate() 側で
# 明確に拒否する。LoRA 自体は3タスク対応だが transformer(非ref)側への配線は未実装)。
# 適用機構は hyperflow_h3 パッケージ(公式、pip 導入済み):
#   - runner 名前空間の MiniMaxH3SetTimestepsStep を HyperFlowSetTimestepsStep へ丸ごと
#     差し替える(下)。未ロード時の _hyperflow_is_disabled() は False(=有効扱い)を
#     返す実装のため、transformer_ref のロードが set_timesteps より後でも正しく
#     HyperFlow グリッドが組まれる(実装確認済み)。
#   - denoise 側は _maybe_hyperflowify_denoise_step() が LoopDenoiser サブブロックを
#     HyperFlowLoopDenoiser へ差し替える(endpoint_context の配線はそちらが担う)。
#   - LoRA 本体のロードは apply_instant_settings(is_ref=True) 直前の
#     _ensure_hyperflow_ref()(単一チョークポイント、ref2va と ref バッチ両方を通る)。
# フェーズA probe 実測(2026-09-26): int8 prequant transformer_ref に PEFT が
# TorchaoLoraLinear でそのまま適用可、LoRA 常駐 +2.87GB、ロード 6.5s。
H3_HYPERFLOW = os.environ.get("H3_HYPERFLOW", "0").strip() == "1"
H3_HYPERFLOW_LORA = os.environ.get("H3_HYPERFLOW_LORA", "videorebirth/hyperflow").strip()
if H3_HYPERFLOW and H3_TURBO_LORA:
    raise RuntimeError(
        "H3_HYPERFLOW=1 と H3_TURBO_LORA=1 は併用できません (turbo の _TurboLoRALinear と "
        "HyperFlow の PEFT アダプタが同じ Linear 群へ二重適用になる)。どちらか一方にしてください。"
    )
if H3_HYPERFLOW and H3_PRUNED:
    raise RuntimeError(
        "H3_HYPERFLOW=1 と H3_PRUNED=1 は併用できません (pruned は time_proj/"
        "time_embedder MLP を補間テーブルへ置換しており、HyperFlow の TwoTimeEmbedder が "
        "ラップする対象が存在しない)。HyperFlow 品質 tier は非 pruned のまま使うこと。"
    )
def _make_set_timesteps_step():
    """ref2va 経路の set_timesteps ブロックを返す。H3_HYPERFLOW 時は HyperFlow 版。

    グローバル名の import 差し替えではなくインスタンス化ヘルパーにしたのは、
    generate_ref2va() が関数ローカルで公式 `MiniMaxH3SetTimestepsStep` を import して
    おり、モジュールグローバルの上書きが効かない(= denoise 側だけ差し替わって
    set_timesteps 側が公式のまま 2-tuple を作りエラーになる)ため。両ブロックを
    必ず対で HyperFlow 版にするための単一ソース。
    """
    if H3_HYPERFLOW:
        from hyperflow_h3 import HyperFlowSetTimestepsStep
        return HyperFlowSetTimestepsStep()
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3SetTimestepsStep
    return MiniMaxH3SetTimestepsStep()


def _maybe_hyperflowify_denoise_step(step):
    """H3_HYPERFLOW 時、denoise 複合ブロック内の LoopDenoiser を HyperFlow 版へ差し替える。

    HyperFlowLoopDenoiser は step ごとに row_timestep_plan の (t, r, indices) 3-tuple を
    読み、transformer 呼び出しを TwoTimeEmbedder.endpoint_context(r) で包む(公式実装)。
    transformer_name は元ブロックの値(ref2va なら "transformer_ref")を引き継ぐ。
    """
    if not H3_HYPERFLOW:
        return step
    from hyperflow_h3 import HyperFlowLoopDenoiser
    from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3LoopDenoiser
    for key, block in list(step.sub_blocks.items()):
        if isinstance(block, MiniMaxH3LoopDenoiser) and not isinstance(block, HyperFlowLoopDenoiser):
            step.sub_blocks[key] = HyperFlowLoopDenoiser(transformer_name=block.transformer_name)
    return step


# MINIMAX_H3_MIN_DURATION..MAX_DURATION = 5..15s at 24fps, aligned to 17*n+5.
# H3_MIN_SECONDS: 既定 5.0(従来どおり)。r-n-v の会話初回チャンク短縮 probe 用に
# 下限を下げられるようにした(2026-10-06)。73f=3.04s 等の短尺はモデルの学習分布外の
# 可能性があるため、品質確認なしで本番の既定を下げないこと。
MIN_SECONDS = float(os.environ.get("H3_MIN_SECONDS", "5.0"))
MAX_SECONDS = 15.0
if MIN_SECONDS < 5.0:
    # diffusers 側にも min_duration=5.0 のバリデーションがある(before_encoder の
    # Ref2VASetupStep と before_denoise の PrepareLayoutStep)。H3_MIN_SECONDS で
    # 下限を下げた場合はプロセス全体でプロパティを合わせる(_relaxed_min_duration()
    # と同じクラスプロパティ差し替え。既定 5.0 のときは一切触らない)。
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
        MiniMaxH3ModularPipeline as _MMP,
    )
    _MMP.min_duration = property(lambda self, _v=MIN_SECONDS: _v)
FPS = 24

# 静止画モード (`generate(still=True)`) のフレーム数の選択肢。値は align_num_frames の
# 17n+5 制約を満たす最小の2つ: 22 (0.917s) は 2026-08-07 の超短尺プローブ
# (scripts/probe_short_frames*.py) で品質が5秒基準と遜色ないことを目視確認済みの既定値。
# 5 (0.208s) は学習分布からさらに外れる実験値で、デコードには下の
# `_patch_vae_smallclip_decode()` (H3_VAE_SMALLCLIP_FIX) が必須(潜在2フレームは
# 上流の `AutoencoderKLMiniMaxH3._decode()` のチャンク境界処理で num_chunks=0 になり
# `torch.cat([])` で落ちる)。
STILL_FRAME_CHOICES = (22, 5)

# Opt-in ("0" default = 完全に無効、既存の ref2va 経路と無変更)。"1" で
# `generate_ref2va()` の Vocal Lock を有効化する: 生成対象の音声行 (packed layout の
# `[text | 参照ブロック | 生成音声行 | 生成映像行]` のうち「生成音声行」) を、
# `references` に含まれる最初の `MiniMaxH3AudioReference` の波形から encode した
# latent で置き換え、denoise 中ずっとクリーン (t=1.0 固定・無変更) に固定する。
# 詳細は `generate_ref2va()` 内の該当コメント、および `_build_vocal_lock_latents()` /
# `_apply_vocal_lock_condition_rows()` の docstring 参照。
#
# 旧名 `H3_AUDIO_DRIVE` の後方互換: `H3_VOCAL_LOCK` が未設定で `H3_AUDIO_DRIVE` が
# 設定されている場合のみその値を使う(新名を優先)。2026-08 の改名(audio_drive ->
# vocal_lock、出所の ComfyUI-H3-NativeAudioLock は "audio lock" と呼んでおり、本実装は
# 駆動源が分離ボーカルである点が固有なので vocal_lock とした。「Audio Drive」は本
# プロジェクトの造語で、機構の実体(駆動ではなく凍結)を誤って示唆していた)。
if "H3_VOCAL_LOCK" not in os.environ and "H3_AUDIO_DRIVE" in os.environ:
    logger.warning("環境変数 H3_AUDIO_DRIVE は旧名です。H3_VOCAL_LOCK へ移行してください。")
    H3_VOCAL_LOCK = os.environ.get("H3_AUDIO_DRIVE", "0").strip() == "1"
else:
    H3_VOCAL_LOCK = os.environ.get("H3_VOCAL_LOCK", "0").strip() == "1"

# Opt-out ("1" default). "1" = `AutoencoderKLMiniMaxH3._decode()` に「潜在フレームが
# 1チャンク未満 (num_chunks==0、潜在1-2フレーム)なら全トークンを単一の `_decode_clip()`
# で復号する」分岐を monkeypatch で追加する(下の `_patch_vae_smallclip_decode()`)。
# 通常の動画 (num_chunks>=1) はパッチ後も従来の実装へそのまま委譲されるため
# byte-for-byte 影響なし。"0" は静止画モードの frames=5 が上流バグそのままで落ちる
# 状態に戻すためのトグル(例外時クリーンアップの実機検証にも使った)。
H3_VAE_SMALLCLIP_FIX = os.environ.get("H3_VAE_SMALLCLIP_FIX", "1").strip() == "1"

# EXPERIMENTAL, opt-in ("0" default = completely inert, zero behaviour change --
# `core/adaln_precompute.py` is never imported/called unless this is "1"). Ported from
# NVIDIA Sol-Engine's `sana-sol-engine` repo (`models/minimax_h3/GB200/adaln.py`,
# Apache-2.0) -- see that module's own docstring for the full mechanism/rationale.
# In short: precomputes the whole trajectory's AdaLN modulation table (~1.5GB) once, up
# front, from the request's fixed sampling schedule, and drops the ~26GB of
# `adaln_proj` weights that would otherwise sit GPU-resident recomputing the exact same
# values on every one of the transformer's 50 blocks x every denoising step. Bitwise
# identical to the uncached path (one GEMM per (block, step) at the reference's own
# shape -- no batching, no approximation), so this needs no quality gate, only an
# `ffmpeg framemd5` exact-match check (see docs/h3-adaln-precompute-20260826.md).
# Expected saving: bf16 transformer ~66.3GB -> roughly ~42GB resident.
#
# v2 (coexistence, see core/adaln_precompute.py's module docstring for the full
# verification): bf16 transformer only, still rejected at import time (this block)
# against H3_TRANSFORMER_QUANT=int8 and any H3_LOWVRAM mode -- both remain real
# structural conflicts, unrelated to turbo, unchanged from v1:
#
#   - int8: torchao's Int8Tensor-backed `adaln_proj.linear` was not verified against
#     `precompute()`'s own GEMM-per-step + `del`-the-weights shape (the whole point of
#     H3_TRANSFORMER_QUANT=int8 is to shrink `adaln_proj` along with everything else
#     via quantization, not to also precompute-and-drop it -- the two techniques target
#     the exact same weights for the exact same reason, combining them is redundant at
#     best and unverified at worst).
#   - H3_LOWVRAM/H3_LOWVRAM_GROUP: both require H3_TRANSFORMER_QUANT=int8 already (see
#     that block's own guard above), so this is really the same restriction stated
#     twice for a clearer error message at whichever flag combination an operator
#     actually sets.
#
# turbo is NO LONGER blanket-rejected here (v1's H3_TURBO_LORA check is gone): checked
# key-by-key against every non-comfyui-mirror checkpoint file the default
# H3_TURBO_LORA_REPO (lightx2v/Minimax-h3-Turbo, diffusers-native/DMD format) ships,
# none of them touch `adaln_proj`/`norm_out` at all (0 of 312 wrapped Linears each, see
# core/adaln_precompute.py's module docstring for the full per-file verification) -- so
# there is no structural conflict for that format, and this project's actual
# production config (`H3_TURBO_LORA_FILE=minimax_h3_ref2v_turbo_4step_v0.1_bf16.
# safetensors`, one of the checked files) can run turbo and AdaLN precompute together.
# Only the comfy-format LoRA (`_TURBO_COMFY_REPOS`, e.g. larryvrh/MiniMax-H3-Turbo-Lora)
# genuinely conflicts -- its checkpoint DOES carry an `adaln_proj`/`norm_out` delta (51
# of 259 wrapped Linears, verified against all 3 cached snapshots) -- so THAT specific
# repo is still rejected here, using the same lightweight repo-name heuristic
# `turbo_lora_expected_format()` uses elsewhere in this file (that function is defined
# later in this module and cannot be called yet at this point in module-body
# execution, hence the inline check rather than a call to it). This is a heuristic on
# the *configured* repo, checked again for real by `core/adaln_precompute.py`'s
# `_reject_turbo_wrapped_adaln()` once the actual checkpoint's keys are known (an
# operator could point H3_TURBO_LORA_REPO/H3_TURBO_LORA_FILE at some other,
# not-yet-known comfy-format checkpoint under a different repo name, which this
# heuristic would miss but that later, key-level check would still catch).
H3_ADALN_PRECOMP = os.environ.get("H3_ADALN_PRECOMP", "0").strip() == "1"
if H3_ADALN_PRECOMP:
    if H3_TRANSFORMER_QUANT != "none":
        raise RuntimeError(
            "H3_ADALN_PRECOMP=1 と H3_TRANSFORMER_QUANT=int8 は併用できません "
            "(int8 は adaln_proj も含め全 Linear を torchao Int8Tensor へ量子化するため、"
            "precompute() の GEMM-per-step + 重み del という前提が torchao の量子化テンソル "
            "に対して未検証です。両者は同じ重み集合を同じ理由で削減対象にしており、"
            "組み合わせる意味自体が薄いため明示的に拒否します)。"
            "H3_TRANSFORMER_QUANT=none (既定) で使ってください。"
        )
    if H3_LOWVRAM_ANY:
        raise RuntimeError(
            f"H3_ADALN_PRECOMP=1 と H3_LOWVRAM={H3_LOWVRAM_RAW!r} は併用できません "
            "(H3_LOWVRAM/H3_LOWVRAM_GROUP はどちらも H3_TRANSFORMER_QUANT=int8 を要求 "
            "しており、上の int8 拒否と同じ理由で未検証です)。H3_LOWVRAM=0 (既定) で "
            "使ってください。"
        )
    if H3_TURBO_LORA and H3_TURBO_LORA_REPO in _TURBO_COMFY_REPOS:
        raise RuntimeError(
            f"H3_ADALN_PRECOMP=1 と H3_TURBO_LORA_REPO={H3_TURBO_LORA_REPO!r} (comfy形式) "
            "は併用できません (このチェックポイントは adaln_proj.linear/norm_out.linear "
            "にも LoRA デルタを持つため、precompute() が一度だけ焼くテーブルではこの "
            "デルタを表現できません。詳細は core/adaln_precompute.py のモジュール "
            "docstring 参照)。既定の diffusers ネイティブ形式 "
            "(H3_TURBO_LORA_REPO=lightx2v/Minimax-h3-Turbo、または "
            "H3_TURBO_LORA_REPO/H3_TURBO_LORA_FILE を未設定のまま) は adaln_proj に "
            "触れないため AdaLN precompute と併用できます。"
        )
    logger.info(
        "H3_ADALN_PRECOMP=1: AdaLN modulation will be precomputed and adaln_proj "
        "weights freed on every fresh transformer/transformer_ref load (bf16 only, "
        "~66.3GB -> ~42GB expected resident). turbo=%s (repo=%s) -- coexistence "
        "verified for the diffusers-native LoRA format, see core/adaln_precompute.py.",
        H3_TURBO_LORA, H3_TURBO_LORA_REPO,
    )


def _patch_vae_smallclip_decode() -> None:
    """`AutoencoderKLMiniMaxH3._decode()` の潜在1-2フレーム境界バグを runner 側から直す。

    上流 (PR #14355 abc5e9b) の `_decode()` はチャンク数を
    `num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)`
    で計算する。5ピクセルフレーム動画の潜在は2フレーム (= `tokens_chunk_size(5)` から
    `token_drop(3)` を引いた値ぴったり) なので num_chunks が 0 になり、チャンクループが
    一度も回らず `torch.cat([])` が ValueError を投げる -- 2026-08-07 の超短尺プローブで
    実機再現した既知バグ(README「超短尺生成プローブ」参照)。

    この関数はクラスメソッドを wrap し、num_chunks>=1 の通常経路は元実装へそのまま
    委譲、num_chunks==0 のときだけ「(必要ならパディングした)全トークンを単一の
    `_decode_clip()` で復号し、`frame_pre_padding` とパディング由来の末尾フレームを
    切り落とす」経路を通す。これは元実装のチャンク0本目の処理 (`clip[:, :,
    frame_start:...]` -> `chunk[:, :, self.frame_pre_padding:]`) をチャンク分割なしに
    そのまま適用したもの: 潜在2フレーム x temporal_ratio(4) = 8フレーム -
    frame_pre_padding(3) = 5ピクセルフレーム、で幾何が一致する。パディングした
    潜在フレームは全て非チャンク末尾トークン (パディング後も2トークンしかなく、
    チャンク末尾 = index 4 に届かない) なので、末尾トリムは一律
    `pad_tokens * temporal_ratio` でよい(元実装の intra_tail 分岐が効く条件に入らない)。

    venv の diffusers 本体は変更しない(このプロジェクトの決まり)。idempotent:
    パッチ済みならフラグを見て何もしない。`_decode` の `@apply_forward_hook` は元の
    束縛関数越しに通常経路では従来どおり効く。小クリップ経路は accelerate フックを
    通らないが、この runner は VAE を手動で移動しており accelerate フックを付けない
    ため実害はない。
    """
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3

    orig_decode = AutoencoderKLMiniMaxH3._decode
    if getattr(orig_decode, "_h3_smallclip_patched", False):
        return

    def _decode_with_smallclip_fix(self, z: torch.Tensor) -> torch.Tensor:
        tokens_chunk_size = self.tokens_chunk_size
        token_drop = self.config.token_drop
        num_tokens = z.shape[2] + token_drop
        pad_tokens = (-num_tokens) % tokens_chunk_size
        num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
        if num_chunks >= 1:
            return orig_decode(self, z)

        temporal_ratio = self.temporal_compression_ratio
        if pad_tokens > 0:
            z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
        dec = self._decode_clip(z)
        dec = dec[:, :, self.frame_pre_padding :]
        if pad_tokens > 0:
            dec = dec[:, :, : -(pad_tokens * temporal_ratio)]
        return dec

    _decode_with_smallclip_fix._h3_smallclip_patched = True
    AutoencoderKLMiniMaxH3._decode = _decode_with_smallclip_fix
    logger.info("patched AutoencoderKLMiniMaxH3._decode with small-clip (num_chunks==0) fix")


if H3_VAE_SMALLCLIP_FIX:
    _patch_vae_smallclip_decode()


# 既定 ON: ref2va バッチ (`generate_ref_batch`) の参照プレフィックス KV キャッシュ共有。
# バッチは「参照共通・プロンプト違い」が入力仕様なので、参照ラベル+ビジョン
# (~4104トークン、~65s/場面) の Qwen3-VL エンコードを場面ごとに繰り返すのは純粋な重複 --
# プレフィックスを1回だけ `use_cache=True` で通し、場面ごとにプロンプト末尾
# (14-33トークン、~0.2s) だけをキャッシュ継続する。"0" でいつでも従来経路 (場面ごとの
# _encode_ref2va_prompt フル計算) に戻せる。
#
# 精度 (scripts/probe_ref_prefix_cache.py で実測済み、PR #14355 前の実装に対して):
# - プレフィックス部分の hidden_states[50] はフル計算と**ビット一致** (因果LMで参照が
#   前置・プロンプトは末尾 verbatim のため。旧 packing_ref2va.build_ref2va_presentation で
#   確認 -- 後継の MiniMaxH3Ref2VATextEncoderStep._build_presentation も同じ構造を保つ)
# - プロンプト末尾部分は相対RMS ~1.5% の丸め差が残る (カーネル/GEMM のタイル経路が
#   系列長で変わるため。eager 固定でも同水準 = sdpa 固有ではなく実行経路差そのもの)。
#   これがロジックバグでないことはネガティブコントロールで確定: わざと位置オフセットを
#   壊した継続は相対RMS 27-30% (20倍) に跳ねる -- 1.5% は「正しい計算の丸めノイズ」の水準
# - つまりバッチ出力は従来経路とビット一致しない (sage/FBC と同種の epsilon 級ドリフト)。
#   ビット再現が要る対照実験では H3_REF_PREFIX_CACHE=0 を使うこと
H3_REF_PREFIX_CACHE = os.environ.get("H3_REF_PREFIX_CACHE", "1").strip() == "1"


# 既定 OFF: 上の共有プレフィックスを **単発 `/api/ref2va` のリクエスト間** でも持ち越す。
# バッチ (`H3_REF_PREFIX_CACHE`) は1回の呼び出しの中で完結するので安全側の既定が ON だが、
# こちらはプロセス寿命の KV キャッシュ (実測 ~1.0GiB VRAM) をリクエストをまたいで持つため、
# 「既存挙動を変えない」を優先して既定 OFF にしてある (効果と安全性が実測で固まってから
# 既定を変えること)。
#
# 何が節約されるか: 同じ参照 (画像) で音声/プロンプトだけ違うリクエストを連投するとき、
# 参照ラベル+ビジョンの Qwen3-VL 前方計算 (768x448/8s の実測で ~55s) が2回目以降まるごと
# 消える。単発リクエストごとに参照の再エンコードが起きていたのはドキュメント既知の穴
# (README「単発リクエストの繰り返しでは共有されない」)。
#
# 成立条件 (実測で確認、`_single_ref_prefix_cache_key` がこの条件を機械的に判定する):
# - **画像参照のみ** (+ 音声参照はいくつでも可)。音声は `_build_presentation` が
#   `"<Audio j>: "` のラベルしか出さず波形は conditioner に届かないので、音声ファイルの
#   中身も長さも変わってプレフィックスのトークン列は**ビット単位で不変** (プローブ実測:
#   別内容・別長さの wav 4本で prefix_ids_sha / pixel_values_sha が完全一致)。
# - **動画参照が1つでもあると使えない** (動画はピクセルがプレフィックスに入るため、
#   中身が違えばプレフィックスも違う -- プローブ実測 max_abs_diff=102.0)。この場合は
#   従来のフル計算経路へ黙って落ちる (キャッシュは破棄する)。
#
# 精度: プレフィックス部分はフル計算と**ビット一致**、プロンプト末尾は相対RMS ~1.5% の
# 丸め差 (`H3_REF_PREFIX_CACHE` のコメントと同じ理由・同じ水準)。ビット再現が要る対照
# 実験では 0 のままにすること。
#
# **TE を解放しない構成では自動的に無効化する** (2026-08-24 レビュー指摘):
# キャッシュ (KV 約1.04GiB) は `_free_text_encoder()` に相乗りして捨てている。しかし
#   - `H3_TE_DEVICE` 指定時 (48gb-dual): `_free_text_encoder` が early return するため
#     TE が解放されず、キャッシュが **TE 側 GPU に恒久常駐**する。48gb-dual の TE 側は
#     「ref2va に約24GB 必要 (24GBカードは境界)」とプリセット自身が書いている GPU で、
#     そこへ 1.04GiB を返さないまま載せるのは危険。
#   - `H3_TE_PROJ` 指定時 (16gb-proj): 同じく TE 恒久常駐。16GB カード (peak ~11.4GB) に
#     +1.04GiB。しかも投影TE × 参照経路は元々 UNVERIFIED。
# どちらも実測していないので、明示的に無効化して理由をログに出す。
# 必要になったら「TE を解放しない構成でもキャッシュを個別に解放する」実装を足すこと。
_H3_REF_PREFIX_CACHE_SINGLE_REQUESTED = (
    os.environ.get("H3_REF_PREFIX_CACHE_SINGLE", "0").strip() == "1"
)
if _H3_REF_PREFIX_CACHE_SINGLE_REQUESTED and (H3_TE_DEVICE or H3_TE_PROJ):
    logger.warning(
        "H3_REF_PREFIX_CACHE_SINGLE=1 is ignored: the cache is freed together with the "
        "text_encoder, but H3_TE_DEVICE=%r / H3_TE_PROJ=%r keep the TE resident, so the "
        "~1.04GiB KV cache would never be released on that GPU (untested). "
        "Drop H3_TE_DEVICE / H3_TE_PROJ to use the single-request prefix cache.",
        H3_TE_DEVICE, H3_TE_PROJ,
    )
H3_REF_PREFIX_CACHE_SINGLE = _H3_REF_PREFIX_CACHE_SINGLE_REQUESTED and not (
    H3_TE_DEVICE or H3_TE_PROJ
)


# EXPERIMENTAL, opt-in (2026-10-05). `H3_REF_LATENT_CACHE=1`: ref2va の**参照画像 VAE エンコード
# 結果 (condition latent)** を、プロセス内に1エントリだけキャッシュする。
#
# 動機: 単機リアルタイム連続生成 (同じ衣装の参照画像 + 台詞=音声/プロンプトだけが毎回変わる)
# では、参照画像の VAE エンコード (実測 ~1.0s) を毎回同じ入力で計算し直している。
# `H3_REF_PREFIX_CACHE_SINGLE` が Qwen 画像特徴 (prefix KV) を使い回すのと同じ思想で、
# こちらは VAE 側の latent を使い回す。音声参照の audio_vae エンコードは毎回違うので
# キャッシュしない (hit 時も実行する)。
#
# キー: 正規化後 (setup step 通過後=短辺リサイズ済み) の画像参照すべての
#   (md5(ピクセル), shape) の列 + keyframe_encode_seed + pixel_mean/std + VAE dtype +
#   短辺設定。エンコード結果に影響しうる値をすべて含めるので、衣装切替・解像度変更・
#   短辺変更はどれも自然に別キー (= miss で再計算して置き換え) になる。
# 対象: 参照が「画像 (1枚以上) + 音声のみ」の場合。動画参照を含む場合はキャッシュせず
#   従来経路 (`ref_latent_cache="bypass"`)。キャッシュされるのは CPU 上の小さな float32
#   テンソル (数 MB 級) で、VRAM は消費しない。
# 出力は非キャッシュ時と bit 一致する (エンコードは `keyframe_encode_seed` 固定の決定論的
#   サンプリングで、hit 時はその結果をそのまま返すだけ。検証手順は README/報告参照)。
# 既定 OFF: OFF のとき生成経路は従来と完全に同一 (関数の先頭でフラグを見て元のステップを
#   そのまま呼ぶだけ)。
H3_REF_LATENT_CACHE = os.environ.get("H3_REF_LATENT_CACHE", "0").strip() == "1"

# EXPERIMENTAL, opt-in (2026-10-05). decode の分離 (単機リアルタイム連続生成の cadence 短縮):
#
#   `H3_DECODE_STREAM=1`  denoise 完了時点で app の生成ロックを解放し、video/audio VAE decode
#                         + uint8 変換 + mux を**そのリクエストのスレッド上で** (デフォルト
#                         ストリームのまま) 行う。次のリクエスト (エンコード/denoise) が decode と
#                         重なって走れる。`=2` は専用 CUDA ストリーム版 (下記の注意参照)。
#   `H3_DECODE_DEVICE=cuda:1`  video VAE (と audio VAE) の**デコード専用コピー**を別 GPU に常駐
#                         させ、denoise 後の latent (数 MB) をそのGPUへ送って decode する。
#                         transformer/TE/参照エンコード用 VAE は従来どおり cuda:0 (DEVICE) のまま。
#
# どちらも既定なし (現行どおり)。安全性 (なぜ重ねて良いか) は `_decode_ref2va_isolated()` と
# `generate_ref2va()` の `on_denoise_done` 周辺のコメント参照。成立条件は
# 「VAE 常駐 (H3_KEEP_REF2VA=1 + H3_KEEP_REF2VA_VAE=1 + H3_LOWVRAM=1)」。満たさない場合は
# 警告を出して従来のインライン decode にフォールバックする (黙って壊れた重なり方をしない)。
_H3_DECODE_STREAM_RAW = os.environ.get("H3_DECODE_STREAM", "0").strip()
H3_DECODE_STREAM = _H3_DECODE_STREAM_RAW in ("1", "2")
# "2" = 専用 CUDA ストリーム (decode 専用 VAE コピーを native attention で使う)。"1" はデフォルト
# ストリームのまま別スレッドで decode する。**専用ストリームで共有 VAE を使ってはいけない**:
# `H3_ATTN_BACKEND=sage` (既定) は diffusers の attention backend をプロセス全体で解決するため
# video VAE の attention も sage になり、sage のカーネルは current stream を尊重せず legacy
# default stream に投げる → side stream 上の前後の演算と順序が壊れ、decode 結果が全面 NaN
# (真っ黒な映像) になる。2026-10-05 に実機で再現 (H3_ATTN_BACKEND=default なら NaN にならない
# ことで原因を特定)。そのため "2" は native attention に固定した別コピーの VAE を使う。
H3_DECODE_STREAM_SIDE = _H3_DECODE_STREAM_RAW == "2"
H3_DECODE_DEVICE = os.environ.get("H3_DECODE_DEVICE", "").strip()
# 別 GPU の decode 専用 VAE コピーを native attention に固定するか (既定 0 = 本体と同じ backend)。
H3_DECODE_NATIVE_ATTN = os.environ.get("H3_DECODE_NATIVE_ATTN", "0").strip() == "1"
# H3_DECODE_VAE=light (2026-10-05, probe, 既定なし=標準 VAE): decode 専用コピーの video VAE を
# LynnReal の蒸留版 light VAE (`stdstu123/LynnReal-Onmi-light-vae`: encoder は H3 公式と同一、
# decoder だけ 36→26 層。latent の channel/mean/std/圧縮率は公式と同一) に差し替える。
# **encode 側・標準の `pipe.vae` は触らない**。`H3_DECODE_STREAM`/`H3_DECODE_DEVICE` の decode
# 分離経路 (`_decode_ref2va_deferred`) だけが対象で、インライン decode には効かない。
# 値は "light" (既定 repo) か、HF repo id / ローカルディレクトリ。再現 (同一 latent でも RGB は
# 標準 VAE と一致しない: LICENSE は minimax-h3-community = H3 本体と同じ地域制限あり)。
H3_DECODE_VAE = os.environ.get("H3_DECODE_VAE", "").strip()
H3_DECODE_VAE_REPO = (
    "stdstu123/LynnReal-Onmi-light-vae" if H3_DECODE_VAE.lower() == "light" else H3_DECODE_VAE
)


# H3_TE_PROJ 有効時、H3 トークナイザ固有の特殊トークン (`<d>`=151669 / `</d>`=151670)
# はここから拒否する。理由: これらは H3 の 32B TE 用チェックポイントの語彙にだけ追加
# されたトークンで、Qwen3-VL-4B-Instruct の埋め込み表 (vocab_size=151669、有効IDは
# 0..151668) には存在しない -- 実測確認済み (2026-08-10)。素通しすると埋め込みテーブル
# の範囲外アクセスになるか、たまたま無関係な埋め込みを引いて黙って壊れた条件付けに
# なる。台詞 (`<d>...</d>`) は音声参照 (fully_copy) 側で入れるか、H3_TE_PROJ を無効化
# した通常経路 (32B TE) を使うこと。
H3_TE_PROJ_UNSUPPORTED_TOKEN_ID_START = 151669


def _reject_unsupported_proj_tokens(token_ids: list[int]) -> None:
    """H3_TE_PROJ 有効時、4B の語彙に無いトークン (id >= 151669、`<d>`/`</d>` 等) を
    含むプロンプトを明示的に拒否する。呼び出し側 (`_encode_h3_prompt` /
    `_encode_ref2va_prompt` / `_encode_ref_prompts_shared_prefix`) はトークン化
    (H3 トークナイザ、通常語彙は 4B とID完全一致) の直後にこれを呼ぶ。"""
    bad = sorted({t for t in token_ids if t >= H3_TE_PROJ_UNSUPPORTED_TOKEN_ID_START})
    if bad:
        # 文面は**そのまま UI のエラー表示に出る**ので、原因だけでなく「次に何をすれば
        # よいか」まで書く (2026-08-12 に改稿)。自動でのTE切替はしない方針 -- 32B TE は
        # 常駐 21GB (投影TE は 3.11GB) で速度も落ちるため、切り替えるかどうかは
        # オペレーターの判断に委ねる。UI の再ロード設定パネルなら再起動なしで切替可能。
        raise ValueError(
            "台詞タグ <d>…</d> は現在の設定では使えません。"
            "いま有効な投影TE (Qwen3-VL-4B) の語彙に H3 固有の台詞タグが無いためです"
            f"(該当トークンID: {bad})。\n"
            "台詞を喋らせるには、次のいずれかにしてください:\n"
            "(1) 【推奨】UI 右側の「再ロードが必要な設定」パネルで **投影TE を OFF** にして"
            "適用する → 32B TE に切り替わり <d> が使えます"
            "(再起動不要。ただし TE 常駐が 3.11GB → 21GB に増え、生成も遅くなります)。\n"
            "(2) 台詞を音声参照 (fully_copy) で入れる → 投影TE のままで使えます。\n"
            "(3) <d> タグを外し、地の文で「〜と挨拶する」のように書く → 発話らしい音は出ますが、"
            "台詞の内容は保証されません。"
        )


class _TeProjection:
    """`H3_TE_PROJ` の学習済み線形投影 (Qwen3-VL-4B の `hidden_states[tap]`, 2560次元
    を 32B TE と同じ 5120次元へ写す) を1度だけロードしてキャッシュする。

    適用式は参照実装 (https://github.com/nicolab28/ComfyUI-ClipProj の
    clipproj_projection.py / clipproj_nodes.py) と同一にする必要がある:

        cond = ((h - mean_in) / std_in) @ W * std_out + mean_out
        cond[:, 0] = sink_out

    token 0 (先頭トークン) だけ実測値 `sink_out` で置き換えるのは、この位置がアテン
    ション・シンク (Qwen3-VL の因果アテンションで常に強く参照される先頭トークン) で、
    そのノルム/分布が他トークンと桁違いなため -- 投影行列は他の (シンクでない) トーク
    ンの統計だけで学習されており、token 0 に同じ写像を適用すると学習範囲外の外挿になる
    (参照実装が明示的に `sink_out` で上書きしているのはこのため、自前の統計に置き換え
    てはいけない)。
    """

    def __init__(self, path: str, device: torch.device):
        from safetensors import safe_open

        with safe_open(path, framework="pt") as f:
            meta = f.metadata() or {}
            tensors = {key: f.get_tensor(key) for key in f.keys()}

        required = {"W", "mean_in", "std_in", "mean_out", "std_out", "sink_out"}
        missing = required - tensors.keys()
        if missing:
            raise RuntimeError(f"H3_TE_PROJ checkpoint {path!r} is missing tensor(s): {sorted(missing)}")

        self.tap = int(meta.get("tap", MINIMAX_H3_TEXT_ENCODER_LAYER))
        # 演算精度は fp32 で保持する (チェックポイント自体も fp32 -- normalize/matmul を
        # bf16 に落とすと、学習時の統計とずれた丸め誤差が入るため)。呼び出し側の最終
        # dtype への変換は `project()` の戻り値で行う。
        self.device = device
        self.W = tensors["W"].to(device=device, dtype=torch.float32)
        self.mean_in = tensors["mean_in"].to(device=device, dtype=torch.float32)
        self.std_in = tensors["std_in"].to(device=device, dtype=torch.float32)
        self.mean_out = tensors["mean_out"].to(device=device, dtype=torch.float32)
        self.std_out = tensors["std_out"].to(device=device, dtype=torch.float32)
        self.sink_out = tensors["sink_out"].to(device=device, dtype=torch.float32)
        logger.info(
            "H3_TE_PROJ: loaded projection %s (tap=%d, d_in=%d, d_out=%d) to %s",
            path, self.tap, self.W.shape[0], self.W.shape[1], device,
        )

    def _project_raw(self, hidden: torch.Tensor) -> torch.Tensor:
        """token 0 の sink_out 置換を行わない素の投影。KVキャッシュ継続 (`_encode_ref_
        prompts_shared_prefix`) の suffix セグメントのように、渡された `hidden` の
        位置0がシーケンス全体の先頭 (アテンションシンク) ではない場合に使う。"""
        h = hidden.to(device=self.device, dtype=torch.float32)
        return ((h - self.mean_in) / self.std_in) @ self.W * self.std_out + self.mean_out

    def project(self, hidden: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """`hidden`: `(1, num_tokens, d_in)`, 4B の `hidden_states[self.tap]` で、
        位置0がシーケンス全体の先頭であるもの。戻り値: `(1, num_tokens, d_out)`,
        32B TE と同じ形/dtype 契約。"""
        cond = self._project_raw(hidden)
        # token 0 (先頭、アテンションシンク) は投影が学習していない -- クラスdocstring参照。
        cond[:, 0] = self.sink_out
        return cond.to(dtype=dtype)

    def project_continuation(self, hidden: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """KVキャッシュ継続の続き部分 (シーケンス先頭を含まない断片) を写す。
        sink_out 置換をしない点だけが `project()` と異なる -- 続き部分の位置0は
        シーケンス全体で見れば先頭トークンではないため、置換すると誤り。"""
        return self._project_raw(hidden).to(dtype=dtype)


def _te_projection_for(components) -> "_TeProjection | None":
    """有効な `H3_TE_PROJ` 投影インスタンスを返す (`_load_text_encoder_proj` がロード時に
    一度だけ `components._te_projection` へセットする)。無効なら None -- 呼び出し側は
    None のとき従来経路 (32B TE, `components.text_encoder_layer`=50) をそのまま使う。"""
    return getattr(components, "_te_projection", None)


def _te_encoder_layer_for(components) -> int:
    """`get_qwen3vl_prompt_embeds` / 直接呼び出しへ渡す `text_encoder_layer`。投影が
    有効なら 4B 側の tap (既定24)、無効なら従来どおり `components.text_encoder_layer`
    (32B TE の50)。"""
    proj = _te_projection_for(components)
    return proj.tap if proj is not None else components.text_encoder_layer


def _encode_ref_prompts_shared_prefix(
    pipe,
    prompts: list[str],
    normalized_references,
    device: torch.device,
    dtype: torch.dtype,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    r"""参照プレフィックスを1回だけ KV キャッシュ化し、場面 (プロンプト) ごとにプロンプト
    末尾だけを継続エンコードする (`MiniMaxH3Ref2VATextEncoderStep.__call__` のバッチ特化版。
    H3_REF_PREFIX_CACHE のモジュールコメントに実測精度、scripts/probe_ref_prefix_cache.py
    に検証手順 -- どちらも PR #14355 以前の `encode_prompt` 静的メソッドに対して取られた
    ものだが、プレフィックス/継続分割点の設計自体は新実装でも同じ形のまま成り立つ、
    このdocstringが説明する通り)。

    PR #14355 (f37ab93) 後: `packing_ref2va.build_ref2va_presentation` /
    `sample_reference_video_frames` は削除され、同等のロジックは
    `MiniMaxH3Ref2VATextEncoderStep` のインスタンスメソッドに吸収された --
    `_gather_vision_features` (画像/動画参照をプロセッサへ通し、ビジョントークン数と
    動画ブロックのタイムスタンプを求める。動画のフレームサンプリングは
    `_sample_video_condition_frames` staticmethod、旧 `sample_reference_video_frames` の
    後継)と `_build_presentation` (staticmethod、旧 `build_ref2va_presentation` の後継 --
    ラベル付け・ビジョンブロック挿入・プロンプト末尾追加のトークン化)。この関数は
    `MiniMaxH3Ref2VATextEncoderStep()` のインスタンスを1つ作り、その2メソッドを
    `__call__` 相当の手順でプレフィックス (プロンプト空文字) に対して1回だけ呼び、
    以降は `get_qwen3vl_prompt_embeds` を経由せず旧実装同様 `text_encoder.model(...)` を
    直接 `use_cache=True` で叩く (KVキャッシュ継続には `get_qwen3vl_prompt_embeds` の
    `use_cache=False` 固定呼び出しでは足りないため)。

    設計はプローブをそのまま踏襲する:
      1. `_build_presentation(tokenizer, "", ...)` (プロンプト空文字) のトークン列は、
         任意のプロンプト付きフル系列の先頭と完全一致する (プロンプトは常に最後に
         `emit(text(prompt))` されるだけ -- encoders.py で確認、トークン単位の一致も
         プローブで実測)。これをプレフィックスとして1回だけ `use_cache=True` で通す
      2. 場面ごとにプロンプト末尾のみを `past_key_values=cache` で継続する。
         `attention_mask`/`mm_token_type_ids`/`pixel_values` 系は全て None
         (`image_grid_thw` を渡すと `model.rope_deltas` が再計算・上書きされる罠がある)
      3. 継続のたびに `DynamicCache.crop(prefix_len)` でプレフィックスに切り戻す (直列運用)

    重要な制約: プレフィックス呼び出しは `model.rope_deltas` (Qwen3VLModel の
    **インスタンス状態**) を書き換え、継続呼び出しはそれを読む。この関数は
    「プレフィックス→全場面の継続」を1回の呼び出し内で完結させるので安全だが、
    呼び出し側はこの関数の実行中に他の text_encoder 呼び出しを挟んではならない。

    H3_TE_PRUNE (51層 TE) でも成立する (`get_qwen3vl_prompt_embeds` と同じ num_layers
    ガードを行い、DynamicCache はレイヤー数に自動追従する)。返り値は `prompts` と同順の
    `(prompt_embeds, text_token_tags)` で、`MiniMaxH3Ref2VATextEncoderStep.__call__` と
    同じ形・dtype 規約。
    """
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep
    from transformers.cache_utils import DynamicCache

    components = pipe
    step = MiniMaxH3Ref2VATextEncoderStep()

    te_proj = _te_projection_for(components)
    te_layer = _te_encoder_layer_for(components)
    num_layers = components.text_encoder.config.text_config.num_hidden_layers
    if num_layers <= te_layer:
        raise ValueError(
            f"MiniMax-H3 conditions on `hidden_states[{te_layer}]` of its Qwen3-VL "
            f"conditioner, which needs more than {te_layer} decoder layers, but "
            f"`text_encoder` has {num_layers}."
        )
    if te_proj is not None:
        # 投影TE + 参照経路 (ref2va) は 4B の vision tower の特徴が同じ行列で正しく
        # 写るか未検証 -- 一度だけ警告する (H3_TE_PROJ のモジュールコメント参照)。
        logger.warning(
            "H3_TE_PROJ + ref2va (reference) path is UNVERIFIED -- the projection matrix "
            "was only checked against 4B text hidden states, not vision tower features."
        )

    # --- MiniMaxH3Ref2VATextEncoderStep.__call__ と同一の画像/動画参照の前処理 ---
    vision_inputs, image_token_counts, video_token_counts, video_timestamps = step._gather_vision_features(
        components.processor, normalized_references, components.fps
    )
    pixel_values = vision_inputs.get("pixel_values")
    image_grid_thw = vision_inputs.get("image_grid_thw")
    pixel_values_videos = vision_inputs.get("pixel_values_videos")
    video_grid_thw = vision_inputs.get("video_grid_thw")

    prefix_ids, prefix_tags = step._build_presentation(
        components.tokenizer,
        "",
        normalized_references,
        image_token_counts,
        video_token_counts,
        video_timestamps,
        text_tag=components.text_tag,
        video_tag=components.video_tag,
    )
    if te_proj is not None:
        _reject_unsupported_proj_tokens(prefix_ids)
    prefix_input = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    mm_token_type_ids = torch.tensor(
        components.processor.create_mm_token_type_ids([prefix_ids]), dtype=torch.long, device=device
    )

    # get_qwen3vl_prompt_embeds と同じ CPU オフロードフックの手動発火 (`.model(...)` を
    # 直接呼ぶため -- KVキャッシュ継続にはそのヘルパーの `use_cache=False` 固定では
    # 足りないので、ここは自前で呼ぶ)。
    model = components.text_encoder.model
    hook = getattr(components.text_encoder, "_hf_hook", None)
    if hook is not None and hasattr(hook, "pre_forward"):
        hook.pre_forward(components.text_encoder)

    t_prefix = time.time()
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        prefix_out = model(
            input_ids=prefix_input,
            attention_mask=torch.ones_like(prefix_input),
            mm_token_type_ids=mm_token_type_ids,
            pixel_values=None if pixel_values is None else pixel_values.to(device, components.text_encoder.dtype),
            image_grid_thw=None if image_grid_thw is None else image_grid_thw.to(device),
            pixel_values_videos=(
                None if pixel_values_videos is None else pixel_values_videos.to(device, components.text_encoder.dtype)
            ),
            video_grid_thw=None if video_grid_thw is None else video_grid_thw.to(device),
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
        )
    prefix_hidden = prefix_out.hidden_states[te_layer].to(device=device, dtype=dtype)
    if te_proj is not None:
        prefix_hidden = te_proj.project(prefix_hidden, dtype=dtype)
    prefix_len = cache.get_seq_length()
    logger.info(
        "shared-prefix encode: prefix %d tokens in %.1fs, continuing %d scene prompt(s)",
        prefix_len, time.time() - t_prefix, len(prompts),
    )

    results: list[tuple[torch.Tensor, torch.Tensor]] = []
    for prompt in prompts:
        suffix_ids = components.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if te_proj is not None:
            _reject_unsupported_proj_tokens(suffix_ids)
        suffix_input = torch.tensor([suffix_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            suffix_out = model(
                input_ids=suffix_input,
                attention_mask=None,
                mm_token_type_ids=None,
                pixel_values=None,
                image_grid_thw=None,
                pixel_values_videos=None,
                video_grid_thw=None,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
            )
        suffix_hidden = suffix_out.hidden_states[te_layer].to(device=device, dtype=dtype)
        if te_proj is not None:
            # suffix はキャッシュ継続で得た「シーケンス先頭を含まない」断片なので
            # sink_out 置換をしない `project_continuation` を使う (`project()` は常に
            # 位置0を置換するため、そのまま使うと suffix の先頭トークンを誤って
            # sink_out に差し替えてしまう -- `_TeProjection.project_continuation` の
            # docstring参照)。
            suffix_hidden = te_proj.project_continuation(suffix_out.hidden_states[te_layer], dtype=dtype)
        cache.crop(prefix_len)

        prompt_embeds = torch.cat([prefix_hidden, suffix_hidden], dim=1)
        # プロンプトはビジョンブロックを含まない純テキストの末尾セグメントなので、
        # suffix のタグは全て components.text_tag。
        text_token_tags = torch.tensor(
            list(prefix_tags) + [components.text_tag] * len(suffix_ids), dtype=torch.long
        )
        results.append((prompt_embeds, text_token_tags))

    return results


@dataclass
class _SingleRefPrefixEntry:
    """`H3_REF_PREFIX_CACHE_SINGLE` の1エントリ (同時に1つだけ保持する)。

    `cache`/`prefix_hidden` は GPU 常駐 (実測 ~1.0GiB + ~40MiB)。`rope_deltas` を
    一緒に持つのが要点 -- 下の `_encode_ref2va_prompt_prefix_cached` の docstring 参照。
    """

    key: str
    cache: object            # transformers DynamicCache (プレフィックス長に crop 済み)
    prefix_hidden: torch.Tensor
    prefix_tags: list[int]
    prefix_len: int
    rope_deltas: torch.Tensor | None
    te_ref: object           # weakref.ref(text_encoder) -- 解放/再ロードの検出用
    device: torch.device
    dtype: torch.dtype
    te_layer: int
    te_proj_id: int | None
    nbytes: int
    # `H3_REF_PREFIX_PARK=1` のとき、KV (cache.layers[*].keys/values) が今 CPU にあるか。
    parked: bool = False


def _park_prefix_entry(entry: "_SingleRefPrefixEntry") -> None:
    """KV を CPU (pinned) へ退避する (`H3_REF_PREFIX_PARK`)。値は不変、置き場所だけ変える。
    小さいテンソル (~100 個 x ~10MiB) の移動で、モジュール丸ごとのスワップではない。"""
    if entry.parked:
        return
    t0 = time.time()
    for layer in entry.cache.layers:
        for name in ("keys", "values"):
            t = getattr(layer, name, None)
            if t is not None and t.is_cuda:
                setattr(layer, name, t.detach().to("cpu").pin_memory())
    entry.parked = True
    # decode を別ストリーム/別 GPU で重ねている間 (H3_DECODE_STREAM / H3_DECODE_DEVICE) は
    # empty_cache を呼ばない: 全デバイスのキャッシュを解放するため、前リクエストの decode と
    # 衝突して "illegal memory access" になる (2026-10-05 probe で再現: REF_PREFIX_PARK + DECODE_DEVICE)。
    # 解放した 0.2GiB はアロケータのキャッシュに残り、次の denoise で再利用される。
    if not (H3_DECODE_STREAM or H3_DECODE_DEVICE):
        gc.collect()
        torch.cuda.empty_cache()
    logger.info("single ref-prefix cache parked on CPU (%.2fGiB) in %.2fs. gpu=%s",
                entry.nbytes / 1024**3, time.time() - t0, gpu_mem_gb())


def _unpark_prefix_entry(entry: "_SingleRefPrefixEntry") -> None:
    """`_park_prefix_entry` の逆。継続 forward の直前に呼ぶ。"""
    if not entry.parked:
        return
    t0 = time.time()
    for layer in entry.cache.layers:
        for name in ("keys", "values"):
            t = getattr(layer, name, None)
            if t is not None and not t.is_cuda:
                setattr(layer, name, t.to(entry.device, non_blocking=True))
    torch.cuda.synchronize()
    entry.parked = False
    logger.info("single ref-prefix cache unparked to %s in %.2fs", entry.device, time.time() - t0)


# プロセス内に高々1エントリ。`generate_ref2va()` は app.py の generation_lock で直列化
# されているので追加のロックは要らない (キャッシュを触るのは生成経路だけ)。
_single_ref_prefix_entry: "_SingleRefPrefixEntry | None" = None


def _clear_single_ref_prefix_cache(reason: str) -> None:
    """単発プレフィックスキャッシュを捨てる (VRAM も返す)。

    呼ばれる場所は3つ: (1) キー不一致・条件外 (動画参照など) で作り直すとき、
    (2) `_free_text_encoder()` が実際に TE を落とすとき (KV は TE と同じ計算グラフの
    産物なので、TE が入れ替わったら必ず捨てる)、(3) `unload()`。
    """
    global _single_ref_prefix_entry
    if _single_ref_prefix_entry is None:
        return
    freed = _single_ref_prefix_entry.nbytes / 1024**3
    _single_ref_prefix_entry = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("single ref-prefix cache cleared (%.2fGiB): %s", freed, reason)


# H3_REF_LATENT_CACHE の1エントリ: (key, [condition_latent(CPU float32), ...])。
# 生成経路 (generate_ref2va) からしか触らない。連続生成で次リクエストの参照エンコードが
# 走るのは前リクエストの denoise 完了後 (H3_DECODE_STREAM でもエンコード同士は重ならない:
# 生成ロックは1件ずつ取られる) なので追加のロックは要らない。
_ref_latent_cache_entry: "tuple[str, list[torch.Tensor]] | None" = None
_ref_latent_cache_stats = {"hit": 0, "miss": 0, "bypass": 0}


def _clear_ref_latent_cache(reason: str) -> None:
    global _ref_latent_cache_entry
    if _ref_latent_cache_entry is None:
        return
    _ref_latent_cache_entry = None
    logger.info("ref latent cache cleared: %s", reason)


def _ref_latent_cache_key(pipe, image_refs: list, short_edge) -> str:
    """画像参照 (正規化後) の実ピクセルとエンコード条件から、キャッシュキーを作る。"""
    h = hashlib.md5()
    for ref in image_refs:
        arr = np.ascontiguousarray(np.array(ref.image))
        h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        h.update(hashlib.md5(arr.tobytes()).digest())
    vae = pipe.vae
    h.update(repr((
        int(getattr(pipe, "keyframe_encode_seed", -1)),
        tuple(pipe.pixel_mean), tuple(pipe.pixel_std),
        str(next(vae.parameters()).dtype), short_edge, len(image_refs),
    )).encode())
    return h.hexdigest()


def _ref2va_encode_references(pipe, state):
    """`MiniMaxH3Ref2VAReferenceEncoderStep` の呼び出しラッパー (`H3_REF_LATENT_CACHE` 対応)。

    戻り値: `(state, status)`。status は None (フラグ OFF) / "hit" / "miss" / "bypass"。
    フラグ OFF なら元のステップを呼ぶだけで従来と完全に同一。

    ON のとき:
      * hit : 画像参照のエンコードをスキップし、保存済み latent の clone を
        `condition_latents` に入れる。音声参照 (毎回違う) は元のステップを「音声参照のみ」の
        参照リストで走らせて `audio_condition_latents` を得る (画像のエンコードは走らない)。
      * miss: 元のステップをそのまま走らせ、出来た `condition_latents` を保存する。
    ステップの出力リストは「画像/動画参照 (パック順)」と「音声持ち参照 (パック順)」で
    別々なので、画像と音声だけの構成なら両者を独立に差し替えられる (順序の取り違えが無い)。
    """
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VAReferenceEncoderStep

    global _ref_latent_cache_entry
    step = MiniMaxH3Ref2VAReferenceEncoderStep()
    if not H3_REF_LATENT_CACHE:
        _, state = step(pipe, state)
        return state, None

    refs = state.get("normalized_references")
    images = [r for r in refs if r.kind == "image"]
    others = [r for r in refs if r.kind != "image"]
    if not images or any(r.kind != "audio" for r in others):
        _ref_latent_cache_stats["bypass"] += 1
        _, state = step(pipe, state)
        return state, "bypass"

    key = _ref_latent_cache_key(pipe, images, pipe.config.reference_image_short_edge)
    entry = _ref_latent_cache_entry
    if entry is not None and entry[0] == key:
        if others:
            state.set("normalized_references", others)
            try:
                _, state = step(pipe, state)
            finally:
                state.set("normalized_references", refs)
        else:
            state.set("audio_condition_latents", [])
        # clone: 後段が in-place で触っても保存分が汚れないようにする (数 MB なので安い)。
        state.set("condition_latents", [t.clone() for t in entry[1]])
        _ref_latent_cache_stats["hit"] += 1
        return state, "hit"

    _, state = step(pipe, state)
    _ref_latent_cache_entry = (key, [t.clone() for t in state.get("condition_latents")])
    _ref_latent_cache_stats["miss"] += 1
    return state, "miss"


def _single_ref_prefix_cache_key(
    prefix_ids: list[int],
    vision_inputs: dict,
) -> str | None:
    """プレフィックスの実体からキーを作る。使えない構成なら `None`。

    キーはファイル名やリクエストIDではなく **実際にプレフィックス forward へ入る値**
    (トークン列 + 画像テンソルのバイト列 + grid) から作る。これで参照の枚数・順序
    (ラベル番号が変わる)・解像度・ピクセルの中身、どれが変わっても必ず別キーになる。
    音声は `_build_presentation` がラベルしか出さないので `prefix_ids` に自然に
    含まれ、波形そのものはキーに入らない (= 音声だけ差し替えた連投でヒットする、
    これがこの機能の狙い)。

    `None` を返す条件:
      - 動画参照がある (`pixel_values_videos`)。動画はピクセルがプレフィックスに
        入るのでリクエスト間で同一とみなせない (プローブ実測 max_abs_diff=102.0)。
      - 画像参照が1枚も無い (キャッシュする価値が無く、`pixel_values` も無い)。
    """
    if vision_inputs.get("pixel_values_videos") is not None:
        return None
    pixel_values = vision_inputs.get("pixel_values")
    if pixel_values is None:
        return None
    h = hashlib.sha256()
    h.update(np.asarray(prefix_ids, dtype=np.int64).tobytes())
    grid = vision_inputs.get("image_grid_thw")
    if grid is not None:
        h.update(grid.detach().to("cpu").contiguous().numpy().tobytes())
    pv = pixel_values.detach().to("cpu").contiguous()
    h.update(str(pv.dtype).encode())
    h.update(np.asarray(pv.shape, dtype=np.int64).tobytes())
    try:
        raw = pv.numpy()
    except TypeError:
        # bf16 等 numpy に無い dtype はバイト列として見る。
        raw = pv.view(torch.uint8).numpy()
    h.update(raw.tobytes())
    return h.hexdigest()


def _encode_ref2va_prompt_prefix_cached(
    components,
    prompt: str,
    normalized_references,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    r"""単発 `/api/ref2va` 用: 参照プレフィックスの KV キャッシュを**リクエスト間で**
    使い回す (`H3_REF_PREFIX_CACHE_SINGLE=1` のときだけ呼ばれる)。使えない構成なら
    `None` を返し、呼び出し側は従来の `_encode_ref2va_prompt()` へ落ちる。

    仕組みは `_encode_ref_prompts_shared_prefix` (バッチ版) と同一で、違いは
    「プレフィックスを1回の呼び出しの中で使い切らず、モジュール変数に残す」ことだけ。
    そのぶんバッチ版には無い3つの安全装置が要る:

    1. **`model.rope_deltas` の保存/復元**。これは `Qwen3VLModel` の *インスタンス状態*
       で、プレフィックス forward が書き、継続 forward が読む
       (`compute_3d_position_ids`: `past_key_values_length > 0` の枝で
       `position_ids = arange(past_len, past_len+seq) + self.rope_deltas`)。バッチ版は
       「プレフィックス→全継続」を1呼び出しで閉じるので放置できたが、リクエストを
       またぐと間に t2va のプロンプトエンコードや `/api/prompt/enhance` が挟まって
       `rope_deltas` が上書きされうる。そこでプレフィックス時の値を clone して持ち、
       **継続の直前に必ず書き戻す**。これで間に何が挟まっても継続は同じ位置IDで走る。
    2. **text_encoder の同一性チェック**。KV は特定の TE インスタンスの重みで作った
       ものなので、TE が解放/再ロード/量子化変更されたら無効。`weakref` で保持して
       「今の `components.text_encoder` と同一オブジェクトか」を毎回見る
       (`_free_text_encoder()` 側でも能動的に捨てている -- 二重の安全側)。
    3. **キーは参照の実体から**。`_single_ref_prefix_cache_key()` 参照。

    VRAM: プレフィックス 4109 トークンの `DynamicCache` は実測 ~1.0GiB
    (64層 x kv_heads 8 x head_dim 128 x K/V 2 x bf16) + `prefix_hidden` ~40MiB。
    """
    global _single_ref_prefix_entry
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep
    from transformers.cache_utils import DynamicCache

    step = MiniMaxH3Ref2VATextEncoderStep()
    te_proj = _te_projection_for(components)
    te_layer = _te_encoder_layer_for(components)
    num_layers = components.text_encoder.config.text_config.num_hidden_layers
    if num_layers <= te_layer:
        raise ValueError(
            f"MiniMax-H3 conditions on `hidden_states[{te_layer}]` of its Qwen3-VL "
            f"conditioner, which needs more than {te_layer} decoder layers, but "
            f"`text_encoder` has {num_layers}."
        )

    t_key = time.time()
    vision_inputs, image_token_counts, video_token_counts, video_timestamps = step._gather_vision_features(
        components.processor, normalized_references, components.fps
    )
    prefix_ids, prefix_tags = step._build_presentation(
        components.tokenizer,
        "",
        normalized_references,
        image_token_counts,
        video_token_counts,
        video_timestamps,
        text_tag=components.text_tag,
        video_tag=components.video_tag,
    )
    key = _single_ref_prefix_cache_key(prefix_ids, vision_inputs)
    if key is None:
        # 動画参照あり / 画像参照なし -- この構成では持ち越せない。持っていた分は
        # VRAM を無駄に占めるだけなので捨てて、従来経路へ落とす。
        _clear_single_ref_prefix_cache("unsupported references (video or no image)")
        return None
    if te_proj is not None:
        _reject_unsupported_proj_tokens(prefix_ids)

    model = components.text_encoder.model
    hook = getattr(components.text_encoder, "_hf_hook", None)
    if hook is not None and hasattr(hook, "pre_forward"):
        hook.pre_forward(components.text_encoder)

    entry = _single_ref_prefix_entry
    hit = (
        entry is not None
        and entry.key == key
        and entry.te_ref() is components.text_encoder
        and entry.device == device
        and entry.dtype == dtype
        and entry.te_layer == te_layer
        and entry.te_proj_id == (id(te_proj) if te_proj is not None else None)
    )
    if not hit:
        if entry is not None:
            # ローカル参照を先に落としてから捨てる -- 残したままだと refcount が
            # 下がらず、新しいプレフィックスを確保する前に VRAM が返らない。
            entry = None
            _clear_single_ref_prefix_cache("reference/text-encoder changed")
        t_prefix = time.time()
        prefix_input = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        mm_token_type_ids = torch.tensor(
            components.processor.create_mm_token_type_ids([prefix_ids]), dtype=torch.long, device=device
        )
        pixel_values = vision_inputs.get("pixel_values")
        image_grid_thw = vision_inputs.get("image_grid_thw")
        cache = DynamicCache(config=model.config)
        with torch.no_grad():
            prefix_out = model(
                input_ids=prefix_input,
                attention_mask=torch.ones_like(prefix_input),
                mm_token_type_ids=mm_token_type_ids,
                pixel_values=(
                    None if pixel_values is None else pixel_values.to(device, components.text_encoder.dtype)
                ),
                image_grid_thw=None if image_grid_thw is None else image_grid_thw.to(device),
                pixel_values_videos=None,
                video_grid_thw=None,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
            )
        prefix_hidden = prefix_out.hidden_states[te_layer].to(device=device, dtype=dtype)
        if te_proj is not None:
            prefix_hidden = te_proj.project(prefix_hidden, dtype=dtype)
        prefix_len = cache.get_seq_length()
        rope_deltas = getattr(model, "rope_deltas", None)
        nbytes = sum(
            t.numel() * t.element_size()
            for layer in cache.layers
            for t in (layer.keys, layer.values)
            if t is not None
        ) + prefix_hidden.numel() * prefix_hidden.element_size()
        entry = _SingleRefPrefixEntry(
            key=key,
            cache=cache,
            prefix_hidden=prefix_hidden,
            prefix_tags=list(prefix_tags),
            prefix_len=prefix_len,
            rope_deltas=None if rope_deltas is None else rope_deltas.clone(),
            te_ref=weakref.ref(components.text_encoder),
            device=device,
            dtype=dtype,
            te_layer=te_layer,
            te_proj_id=id(te_proj) if te_proj is not None else None,
            nbytes=nbytes,
        )
        _single_ref_prefix_entry = entry
        logger.info(
            "single ref-prefix cache MISS: encoded %d prefix tokens in %.1fs (key prep %.2fs), "
            "kept %.2fGiB resident (key=%s)",
            prefix_len, time.time() - t_prefix, t_prefix - t_key, nbytes / 1024**3, key[:12],
        )
    else:
        logger.info(
            "single ref-prefix cache HIT: reusing %d prefix tokens (%.2fGiB, key prep %.2fs, key=%s)",
            entry.prefix_len, entry.nbytes / 1024**3, time.time() - t_key, key[:12],
        )

    # --- 継続 (プロンプト末尾のみ) ---
    # H3_REF_PREFIX_PARK: CPU に退避していた KV をここで GPU へ戻す (MISS 直後は元々 GPU)。
    _unpark_prefix_entry(entry)
    # rope_deltas は毎回書き戻す (MISS 直後でも安いので無条件に -- 上の docstring 1.)。
    if entry.rope_deltas is not None:
        model.rope_deltas = entry.rope_deltas.clone()
    # 直前の継続で伸びたままになっていないことを保証する (正常系では継続の直後に
    # crop 済みだが、例外で抜けた場合の保険)。
    if entry.cache.get_seq_length() != entry.prefix_len:
        entry.cache.crop(entry.prefix_len)

    suffix_ids = components.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if te_proj is not None:
        _reject_unsupported_proj_tokens(suffix_ids)
    suffix_input = torch.tensor([suffix_ids], dtype=torch.long, device=device)
    try:
        with torch.no_grad():
            suffix_out = model(
                input_ids=suffix_input,
                attention_mask=None,
                mm_token_type_ids=None,
                pixel_values=None,
                image_grid_thw=None,
                pixel_values_videos=None,
                video_grid_thw=None,
                past_key_values=entry.cache,
                use_cache=True,
                output_hidden_states=True,
            )
        if te_proj is not None:
            # 継続断片の位置0はシーケンス先頭ではないので sink_out 置換をしない
            # (`_TeProjection.project_continuation` の docstring 参照)。
            suffix_hidden = te_proj.project_continuation(suffix_out.hidden_states[te_layer], dtype=dtype)
        else:
            suffix_hidden = suffix_out.hidden_states[te_layer].to(device=device, dtype=dtype)
    finally:
        entry.cache.crop(entry.prefix_len)

    if H3_REF_PREFIX_PARK:
        # 次のリクエストまで KV は要らない (denoise/decode の VRAM を返す)。
        _park_prefix_entry(entry)
    prompt_embeds = torch.cat([entry.prefix_hidden, suffix_hidden], dim=1)
    text_token_tags = torch.tensor(
        entry.prefix_tags + [components.text_tag] * len(suffix_ids), dtype=torch.long
    )
    return prompt_embeds, text_token_tags


@contextmanager
def _relaxed_min_duration():
    """静止画モードの間だけ diffusers 側の最小尺バリデーション (5.0s) を緩和する。

    PR #14355 (f37ab93) 後: `MINIMAX_H3_MIN_DURATION` というモジュール定数はもう存在
    しない。`min_duration` は `MiniMaxH3ModularPipeline` (modular_pipeline.py) の
    `@property` になり(既定 5.0)、消費箇所も `before_encoder.MiniMaxH3Ref2VASetupStep`
    (ref2va) と `before_denoise.MiniMaxH3PrepareLayoutStep`(t2va/fl2va -- 静止画
    モードが通る経路はこちら)の2箇所に分かれた。モジュール定数の monkeypatch は
    もう効かないので、`MiniMaxH3ModularPipeline` クラスの `min_duration` プロパティ
    自体を一時的に差し替える -- インスタンス単位ではなくクラス単位なのは、
    `components.min_duration` を読むブロックが受け取る `components` がこのクラスの
    インスタンスだから(プロパティはインスタンス属性の代入では上書きできない)。
    生成は app.py の generation_lock で直列化されているため、この一時的な書き換えが
    並行リクエストへ漏れることはない。scope は `MiniMaxH3PrepareLayoutStep` の呼び出し
    1回分だけに絞る(それ以外のブロックはこのプロパティを読まない)。"""
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline

    saved = MiniMaxH3ModularPipeline.min_duration
    MiniMaxH3ModularPipeline.min_duration = property(lambda self: 0.01)
    try:
        yield
    finally:
        MiniMaxH3ModularPipeline.min_duration = saved


def _unpatchify_video_tokens(
    rows: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    channels: int,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    r"""Unpack transformer rows back into a 5D video latent tensor. The inverse of
    `diffusers.modular_pipelines.minimax_h3.before_denoise.patchify_video_latents`.

    PR #14355 (f37ab93) deleted `packing.py`, which used to export this as
    `unpatchify_video_tokens` -- there is no replacement upstream at all (the equivalent
    logic is now inlined directly inside `MiniMaxH3AfterDenoiseStep.__call__`,
    decoders.py, as part of unpacking a *denoised* sequence back into `latents`/
    `audio_latents`, not exposed as a standalone function). This project's own policy is
    to bring back a small self-implementation rather than vendor a private diffusers
    module (see the "自前実装を好む" project note), so this is a verbatim port of this
    repo's own copy of the pre-PR#14355 `packing.unpatchify_video_tokens` -- confirmed
    byte-for-byte identical to the reshape/permute `MiniMaxH3AfterDenoiseStep.__call__`
    performs on its own `latents` rows (decoders.py), which is the ground truth this port
    is checked against.

    Only used by hires-fix (`_upscale_block_state_2x`, below), which needs the pass-1 x0
    estimate as a 5D tensor mid-loop -- before the real `MiniMaxH3AfterDenoiseStep` ever
    runs (that only happens once, after the whole denoise loop, on the final denoised
    rows).

    Args:
        rows (`torch.Tensor` of shape `(num_patches, channels * prod(patch_size))`): The
            packed rows.
        num_latent_frames (`int`): Number of latent frames.
        latent_height (`int`): Latent height.
        latent_width (`int`): Latent width.
        channels (`int`): Number of latent channels.
        patch_size (`tuple[int, int, int]`): The `(t, h, w)` patch.

    Returns:
        `torch.Tensor` of shape `(batch_size, channels, num_latent_frames, latent_height, latent_width)`.
    """
    patch_t, patch_h, patch_w = patch_size
    rows = rows.reshape(
        -1,
        num_latent_frames // patch_t,
        latent_height // patch_h,
        latent_width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return rows.reshape(-1, channels, num_latent_frames, latent_height, latent_width).contiguous()


def _encode_h3_prompt(
    components,
    prompt: str,
    keyframes: list | None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Build MiniMax-H3's t2va/fl2va presentation of a request and encode it -- the
    `@torch.no_grad()`-free replacement for the retired
    `MiniMaxH3TextEncoderStep.encode_prompt` bare staticmethod this file used to call
    directly (see the call sites' own comments for why bypassing the block's own
    `__call__` matters here: without an explicit `no_grad()` around this, the autograd
    graph pins ~50GB of TE weights on GPU past the free that follows).

    PR #14355 (f37ab93) removed `encode_prompt` entirely -- there is no longer a single
    function that takes a raw prompt string plus optional keyframe images. Its role split
    across two block `__call__` methods (`MiniMaxH3TextEncoderStep` for t2va,
    `MiniMaxH3FL2VATextEncoderStep` for fl2va, both in encoders.py) built on top of the
    new `get_qwen3vl_prompt_embeds` module function, which itself only takes an
    already-tokenized presentation (`token_ids`/`vision_inputs`), not a prompt string.
    This function ports the presentation-building half of both `__call__` methods back
    into one place (read line-for-line from encoders.py as part of this migration; the
    tokenization shape -- `"<Picture i>: "` label + vision block per keyframe, prompt
    verbatim, no chat template, no special tokens -- is unchanged from the old
    `encode_prompt`), then calls `get_qwen3vl_prompt_embeds` for the actual conditioner
    forward, exactly like both new blocks do internally.

    Args:
        components: The pipe shell (`MiniMaxH3ModularPipeline` instance) -- `text_encoder`/
            `tokenizer`/`processor` are read off it, matching every other call site in
            this file's own naming (`components` is what the modular blocks call this
            argument too).
        prompt (`str`): The prompt to encode.
        keyframes (`list[PIL.Image.Image]` or `None`):
            The keyframes already prepared onto the target canvas (`MiniMaxH3ResizeStep`'s
            output), in packed order, or `None`/empty for a t2va (text-only) request.
        device (`torch.device`, *optional*): The device to run the conditioner on.
        dtype (`torch.dtype`, *optional*): The dtype of the returned embeddings.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: the `(1, num_text_tokens, 5120)` hidden
        states and the `(num_text_tokens,)` per-row modality tags, same shape/dtype
        contract `encode_prompt` used to return.
    """
    from diffusers.modular_pipelines.minimax_h3.encoders import get_qwen3vl_prompt_embeds

    tokenizer, processor = components.tokenizer, components.processor
    text_tag, video_tag = components.text_tag, components.video_tag

    vision_inputs: dict = {}
    token_ids: list[int] = []
    token_tags: list[int] = []
    if keyframes:
        # Mirrors `MiniMaxH3FL2VATextEncoderStep.__call__` exactly: a `"<Picture i>: "`
        # label plus one vision block (`<|vision_start|>`, one `<|image_pad|>` per merged
        # vision patch, `<|vision_end|>`) per keyframe, batched through the image
        # processor once. The label rows are tagged text, the vision block rows video --
        # what the transformer's AdaLN modulation keys off.
        vision = processor.image_processor(images=keyframes, return_tensors="pt")
        image_grid_thw = vision["image_grid_thw"]
        vision_inputs = {"pixel_values": vision["pixel_values"], "image_grid_thw": image_grid_thw}
        merge_size = processor.image_processor.merge_size**2
        for index in range(len(keyframes)):
            num_image_tokens = int(image_grid_thw[index].prod()) // merge_size
            label_ids = tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
            vision_ids = (
                [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * num_image_tokens
                + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
            )
            token_ids += label_ids + vision_ids
            token_tags += [text_tag] * len(label_ids) + [video_tag] * len(vision_ids)

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    token_ids += prompt_ids
    token_tags += [text_tag] * len(prompt_ids)

    te_proj = _te_projection_for(components)
    if te_proj is not None:
        # H3 固有の特殊トークン (`<d>`/`</d>`) は 4B の語彙に無い -- `_te_projection_for`
        # のモジュールコメント/`_reject_unsupported_proj_tokens` docstring参照。
        # トークナイザ自体は H3 のものを使い続ける (通常語彙は 4B とID完全一致、
        # H3_TE_PROJ のモジュールコメント参照) -- `tokenizer`/`processor` はどちらも
        # `components` (=H3 の pipe shell) 由来のまま、変更なし。
        _reject_unsupported_proj_tokens(token_ids)

    prompt_embeds = get_qwen3vl_prompt_embeds(
        components.text_encoder,
        processor,
        token_ids,
        vision_inputs,
        text_encoder_layer=_te_encoder_layer_for(components),
        device=device,
        dtype=dtype,
    )
    if te_proj is not None:
        # `get_qwen3vl_prompt_embeds` はここでは 4B の `hidden_states[tap]` (2560次元、
        # 生の hidden state) を返す -- それを学習済み線形投影で 5120次元 (32B TE と同じ
        # 出力次元) へ写す。この呼び出しは常に1つの自己完結したシーケンスを渡すので
        # (KVキャッシュ継続はしない)、位置0は本当にシーケンス先頭 = sink_out 置換が
        # 正しい `project()` (継続専用の `project_continuation()` ではない)。
        prompt_embeds = te_proj.project(prompt_embeds, dtype=dtype or prompt_embeds.dtype)
    return prompt_embeds, torch.tensor(token_tags, dtype=torch.long)


def _encode_ref2va_prompt(
    components,
    prompt: str,
    normalized_references: list,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Build MiniMax-H3's `ref2va` presentation of a request and encode it -- the single
    -request (non-batch) analogue of `_encode_ref_prompts_shared_prefix`, and the
    `@torch.no_grad()`-free replacement for the retired
    `MiniMaxH3Ref2VATextEncoderStep.encode_prompt` bare staticmethod this file used to
    call directly (see `_encode_h3_prompt`'s own docstring for why bypassing the block's
    `__call__` matters here -- same reasoning, ref2va's own text encoder step).

    PR #14355 (f37ab93) removed the `encode_prompt` staticmethod entirely; its role is now
    `MiniMaxH3Ref2VATextEncoderStep.__call__` (encoders.py), built out of two of that same
    class's own instance/static methods -- `_gather_vision_features` (runs the reference
    images/videos through the processor, batched per modality) and `_build_presentation`
    (tokenizes the labelled presentation: `"<Picture i>: "` / `"<Audio j>: "` /
    `"<Video k>: "` labels plus vision blocks, then the prompt verbatim) -- followed by one
    `get_qwen3vl_prompt_embeds` call. This function creates one throwaway
    `MiniMaxH3Ref2VATextEncoderStep()` instance to reuse those two methods (matching
    `__call__`'s own sequence exactly) and calls `get_qwen3vl_prompt_embeds` itself, the
    same shape `_encode_h3_prompt` already uses for t2va/fl2va.

    Args:
        components: The pipe shell (`MiniMaxH3ModularPipeline` instance).
        prompt (`str`): The prompt to encode.
        normalized_references (`list[MiniMaxH3Reference]`):
            The references already normalized by `MiniMaxH3Ref2VASetupStep` (rates/
            resolutions resolved), in packed order.
        device (`torch.device`, *optional*): The device to run the conditioner on.
        dtype (`torch.dtype`, *optional*): The dtype of the returned embeddings.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: the `(1, num_text_tokens, 5120)` hidden
        states and the `(num_text_tokens,)` per-row modality tags.
    """
    from diffusers.modular_pipelines.minimax_h3.encoders import (
        MiniMaxH3Ref2VATextEncoderStep,
        get_qwen3vl_prompt_embeds,
    )

    # H3_PHASE_TIMING (2026-08-27 encode-phase profiling task): this is the
    # `H3_REF_PREFIX_CACHE_SINGLE=0` / no-prefix-cache path -- the one
    # `_encode_ref2va_prompt_prefix_cached()` falls back to, and the one this task's own
    # measurement config uses. That sibling function already logs a `t_key`/`t_prefix`
    # split (see its "single ref-prefix cache MISS" log line); this mirrors the same
    # split here so the no-cache path gets equivalent visibility. No-op tuple/dict when
    # the flag is off -- `_PhaseTimer.mark()` is the only thing touched below, and it is
    # a single early-return in that case.
    _pt = _PhaseTimer("encode_ref2va_prompt")

    step = MiniMaxH3Ref2VATextEncoderStep()
    te_proj = _te_projection_for(components)
    if te_proj is not None:
        # 投影TE + 参照経路 (ref2va) は 4B の vision tower の特徴が同じ行列で正しく
        # 写るか未検証 -- 一度だけ警告する (H3_TE_PROJ のモジュールコメント参照)。
        logger.warning(
            "H3_TE_PROJ + ref2va (reference) path is UNVERIFIED -- the projection matrix "
            "was only checked against 4B text hidden states, not vision tower features."
        )
    vision_inputs, image_token_counts, video_token_counts, video_timestamps = step._gather_vision_features(
        components.processor, normalized_references, components.fps
    )
    _pt.mark("gather_vision_features")  # processor: PIL/np image -> patchified pixel_values (CPU)
    token_ids, token_tags = step._build_presentation(
        components.tokenizer,
        prompt,
        normalized_references,
        image_token_counts,
        video_token_counts,
        video_timestamps,
        text_tag=components.text_tag,
        video_tag=components.video_tag,
    )
    _pt.mark("build_presentation")  # tokenize (CPU, expected tiny)
    if te_proj is not None:
        # H3 固有の特殊トークン (`<d>`/`</d>`) は 4B の語彙に無い -- トークナイザ自体は
        # H3 のものを使い続ける (`components.tokenizer`、通常語彙は 4B とID完全一致)。
        _reject_unsupported_proj_tokens(token_ids)
    prompt_embeds = get_qwen3vl_prompt_embeds(
        components.text_encoder,
        components.processor,
        token_ids,
        vision_inputs,
        text_encoder_layer=_te_encoder_layer_for(components),
        device=device,
        dtype=dtype,
    )
    _pt.mark("conditioner_forward")  # the 32B/4B forward itself (vision tower + text decoder, GPU)
    if te_proj is not None:
        # 常に1つの自己完結したシーケンス (KVキャッシュ継続なし) なので位置0は本当に
        # シーケンス先頭 -- sink_out 置換込みの `project()` が正しい
        # (`_encode_h3_prompt` の同箇所コメント参照)。
        prompt_embeds = te_proj.project(prompt_embeds, dtype=dtype or prompt_embeds.dtype)
    _pt.report()
    return prompt_embeds, torch.tensor(token_tags, dtype=torch.long)


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


class _TurboLoRALinear(torch.nn.Module):
    """`y = base(x) + B(A(x))`, applied at run time (never fused into `base`'s weight).

    Verbatim port of the LoRA author's own `LoRALinear` (`generate.py` in
    `larryvrh/MiniMax-H3-Turbo-Lora`, fetched and read as part of this task, per the
    task brief's instruction to cross-check the reference sampler line-for-line): "Folding
    it into the (bf16) base weight instead would round most of the update away when it is
    small relative to the weight". `a`/`b` are the checkpoint's own `lora_A.weight`
    (`[rank, in_features]`) / `lora_B.weight` (`[out_features, rank]`) tensors, registered as
    buffers (not parameters -- this project only ever runs inference, no autograd needed,
    and a buffer follows `.to(device)`/`.to(dtype)` calls on the parent module the same way a
    parameter would). No extra scale factor: every rank in this checkpoint (64 for
    attn/mlp, 16 for adaln_proj) is used with alpha == rank in the author's own code, i.e.
    scale == 1 always -- reproduced by this task's own inspection of the checkpoint (no
    `alpha` key anywhere in the safetensors file, and the reference's own `LoRALinear.forward`
    has no scale multiply either).
    """

    def __init__(self, base: torch.nn.Linear, lora_a: torch.Tensor, lora_b: torch.Tensor, scale: float = 1.0):
        super().__init__()
        self.base = base
        self.register_buffer("lora_a", lora_a, persistent=False)
        self.register_buffer("lora_b", lora_b, persistent=False)
        # LoRA デルタの適用係数。Ostris 版 (comfy) は alpha==rank で常に 1.0 (既定値の
        # まま = 従来と bit 同一の挙動)。lightx2v 版 (diffusers ネイティブ) は
        # H3_TURBO_LORA_SCALE のモジュールコメントのとおり 0.094 が実測既定。
        self.scale = float(scale)
        # Instant on/off toggle for the *instant-apply* settings group (turbo LoRA is
        # request-scoped, not reload-scoped -- see the task brief / core/settings.py):
        # the wrapper module itself is only ever installed once (lazily, on first
        # request with turbo=1), then left in place permanently and just flipped on/off
        # per request via this flag. Cheap (`if` + early return, no tensor op) and safe
        # to flip from the request thread while `_load_lock` is held, matching how every
        # other per-request knob (FBC threshold, attention backend) in this file is
        # applied: no reload, no module replacement, just a stored setting the next
        # forward call reads.
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return self.base(x)
        # scale=1.0 のとき `1.0 * delta` は IEEE 的に恒等なので、Ostris 版の従来挙動と
        # bit 単位で同一 (回帰性を保つ)。
        return self.base(x) + self.scale * torch.nn.functional.linear(
            torch.nn.functional.linear(x, self.lora_a), self.lora_b
        )

    # `MiniMaxH3AdaLayerNormModulation.forward()`/`MiniMaxH3AdaLayerNormOut.forward()`
    # (transformer_minimax_h3.py) both read `self.linear.weight.dtype` directly to decide
    # what dtype to cast their SiLU'd input to, bypassing this wrapper's own `forward()`
    # entirely -- reproduced by this task's own verification
    # (`scripts/probe_turbo_lora_apply.py`): `AttributeError: '_TurboLoRALinear' object
    # has no attribute 'weight'`, the identical pitfall diffusers-server's sibling
    # project hit with JoyAI's `PatchifyLinear` (per that project's memory notes on this
    # exact failure mode). `.weight`/`.bias` here alias straight through to `base` so any
    # code elsewhere that introspects a wrapped Linear's weight tensor directly (instead
    # of calling it) keeps working unmodified.
    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features


def _turbo_lora_key_map(transformer) -> dict[str, str]:
    """Map every ComfyUI-native LoRA key prefix (e.g. `"blocks.3.attn.qkv_proj"`) this
    checkpoint uses to the dotted attribute path of the diffusers module it targets.

    Derived (not guessed) from reading `transformer_minimax_h3.py` module-by-module
    against the actual checkpoint keys (`scripts/probe_turbo_lora_keys.py` recorded the
    verification this task did before writing this function -- shapes, block/refiner
    counts and the `hidden_size`/`time_embed_dim` config values were all cross-checked,
    not assumed):

    - `blocks.N.attn.qkv_proj`  -> `transformer_blocks.N.attn.to_qkv` (only valid AFTER
      `attn.fuse_projections()` has concatenated `to_q`/`to_k`/`to_v` into `to_qkv`, in
      that exact q,k,v order -- `AttentionModuleMixin.fuse_projections()`, read in
      `diffusers/models/attention.py`, does `torch.cat([to_q.weight, to_k.weight,
      to_v.weight])`, the same order the LoRA author's own `_attn()` unpacks with
      `q, k, v = attn.qkv_proj(x).split(heads * hd, dim=-1)`).
    - `blocks.N.attn.out_proj`  -> `transformer_blocks.N.attn.to_out.0`
    - `blocks.N.mlp.fc1`        -> `transformer_blocks.N.ff.net.0.proj` (diffusers' own
      `SwiGLU.proj`, the un-chunked `Linear(dim, 2*inner_dim)` -- which half of that
      output diffusers labels "hidden"/"gate" internally does not matter for a LoRA
      delta added to the *whole* pre-chunk output, see this task's own derivation:
      the delta is additive in the shared pre-chunk activation space, not inside the
      gating itself).
    - `blocks.N.mlp.fc2`        -> `transformer_blocks.N.ff.net.2`
    - `blocks.N.adaln_proj.linear` -> `transformer_blocks.N.adaln_proj.linear` (name is
      unchanged -- `MiniMaxH3AdaLayerNormModulation`'s own docstring says it is "named
      after the checkpoint's `adaln_proj`, with the modulation projection under the
      `linear` name diffusers uses inside every AdaLN module").
    - `token_refiner.blocks.N.*` -> `token_refiner.refiner_blocks.N.*` (same
      attn/mlp sub-mapping as above; only blocks 0-1 exist, matching
      `num_refiner_layers=2`).
    - `final_layer.adaln_proj.linear` -> `norm_out.linear` (`MiniMaxH3AdaLayerNormOut`'s
      own docstring: "Same module layout and checkpoint keys as
      `AdaLayerNormContinuous`" -- diffusers' equivalent of the checkpoint's
      "final_layer", exposed under the `norm_out` attribute name.
      `final_layer.adaln_proj.linear.lora_B.weight`'s shape (10752 = 2*5376) confirmed
      against `norm_out.linear`'s `2 * hidden_size` output width, not the per-block
      `6 * hidden_size * 3` width).

    Returns `{comfy_prefix: diffusers_dotted_path}`, one entry per checkpoint-adapted
    Linear (518 lora_A/lora_B pairs -> 259 entries: 50 blocks x 5 + 2 token_refiner
    blocks x 4 + 1 final_layer, i.e. 250 + 8 + 1).
    """
    num_layers = transformer.config.num_layers
    num_refiner_layers = transformer.config.num_refiner_layers
    key_map: dict[str, str] = {}
    for i in range(num_layers):
        key_map[f"blocks.{i}.attn.qkv_proj"] = f"transformer_blocks.{i}.attn.to_qkv"
        key_map[f"blocks.{i}.attn.out_proj"] = f"transformer_blocks.{i}.attn.to_out.0"
        key_map[f"blocks.{i}.mlp.fc1"] = f"transformer_blocks.{i}.ff.net.0.proj"
        key_map[f"blocks.{i}.mlp.fc2"] = f"transformer_blocks.{i}.ff.net.2"
        key_map[f"blocks.{i}.adaln_proj.linear"] = f"transformer_blocks.{i}.adaln_proj.linear"
    for i in range(num_refiner_layers):
        key_map[f"token_refiner.blocks.{i}.attn.qkv_proj"] = f"token_refiner.refiner_blocks.{i}.attn.to_qkv"
        key_map[f"token_refiner.blocks.{i}.attn.out_proj"] = f"token_refiner.refiner_blocks.{i}.attn.to_out.0"
        key_map[f"token_refiner.blocks.{i}.mlp.fc1"] = f"token_refiner.refiner_blocks.{i}.ff.net.0.proj"
        key_map[f"token_refiner.blocks.{i}.mlp.fc2"] = f"token_refiner.refiner_blocks.{i}.ff.net.2"
    key_map["final_layer.adaln_proj.linear"] = "norm_out.linear"
    return key_map


def _get_module_by_dotted_path(root: torch.nn.Module, path: str) -> torch.nn.Module:
    obj = root
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _set_module_by_dotted_path(root: torch.nn.Module, path: str, value: torch.nn.Module) -> None:
    *parent_parts, leaf = path.split(".")
    parent = root
    for part in parent_parts:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    if leaf.isdigit():
        parent[int(leaf)] = value
    else:
        setattr(parent, leaf, value)


def apply_turbo_lora(transformer, lora_path: str) -> int:
    """Apply `larryvrh/MiniMax-H3-Turbo-Lora`'s checkpoint to `transformer` as an unfused,
    run-time low-rank delta (see `_TurboLoRALinear`), wrapping every Linear the checkpoint
    adapts (see `_turbo_lora_key_map`'s docstring for the full derivation).

    Fuses every `MiniMaxH3Attention` submodule's Q/K/V projections into `to_qkv` first
    (`attn.fuse_projections()`, a diffusers-native, weight-preserving op -- see
    `AttentionModuleMixin.fuse_projections`) since the checkpoint's `qkv_proj` LoRA targets
    the fused projection, not the three separate ones diffusers builds by default. This is
    required for the LoRA's `attn.qkv_proj` delta to land on the same activation the
    checkpoint's own training run saw; diffusers' unfused `to_q`/`to_k`/`to_v` triple has no
    single matching Linear for a fused-QKV LoRA to attach to.

    IMPORTANT (found by this task's own verification, not assumed from the diffusers
    docstring): `fuse_projections()` COPIES `to_q`/`to_k`/`to_v`'s weights into a new
    `to_qkv` Linear but does NOT delete the three originals (`unfuse_projections()` is the
    only place that ever removes an attribute, and it only ever removes `to_qkv`, never
    puts `to_q`/`to_k`/`to_v` back -- read in `diffusers/models/attention.py`). Left alone,
    this leaves every attention module holding BOTH the fused and unfused weights at once
    -- reproduced on this task's first server-integration attempt: the transformer's own
    resident size grew from the expected ~66.3GB to 79.08GB after this function ran (an
    extra ~12.8GB, consistent with Q+K+V's combined size roughly matching to_qkv's), which
    then OOM'd the very next component load (TE-nf4, ~21GB) with the card almost full.
    `del module.to_q / to_k / to_v` right after fusing reclaims that duplicate memory --
    safe because `to_qkv`'s weight is an independent concatenated copy (`torch.cat(...)`
    inside `fuse_projections()`, not a view), not an alias into the three originals.

    Idempotent is NOT guaranteed by design: this is meant to run exactly once, right after
    a fresh (un-adapted) transformer load -- see the `H3_TURBO_LORA` module comment and its
    call site in `_ensure_transformer`. Calling it twice on the same module would wrap an
    already-`_TurboLoRALinear`-wrapped module a second time (harmless numerically -- the
    LoRA delta would just apply twice -- but wasteful and not the intended usage) and the
    second `fuse_projections()` call would be a no-op (`fused_projections` is already True).

    Returns the number of Linear layers wrapped (259 for this checkpoint: 50 blocks x 5 +
    2 token_refiner blocks x 4 + 1 final_layer, see `_turbo_lora_key_map`'s docstring),
    which the caller logs and can sanity-check against the checkpoint's own key count.
    """
    from safetensors.torch import load_file

    t_fuse = time.time()
    n_fused = 0
    for module in transformer.modules():
        from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention

        if isinstance(module, MiniMaxH3Attention):
            module.fuse_projections()
            # Reclaim the ~12.8GB/transformer this task's own verification found `fuse_
            # projections()` otherwise leaves orphaned -- see the docstring's IMPORTANT
            # paragraph. `hasattr` guards the (never expected, but cheap to check) case of
            # a second call on an already-fused module, where these attributes are already
            # gone.
            for attr in ("to_q", "to_k", "to_v"):
                if hasattr(module, attr):
                    delattr(module, attr)
            n_fused += 1
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("turbo LoRA: fused Q/K/V on %d attention modules in %.2fs", n_fused, time.time() - t_fuse)

    t_load = time.time()
    lora_sd = load_file(lora_path)
    key_map = _turbo_lora_key_map(transformer)
    device = next(transformer.parameters()).device
    n_wrapped = 0
    seen_prefixes = {k.rsplit(".lora_", 1)[0] for k in lora_sd if ".lora_" in k}
    for comfy_prefix, dotted_path in key_map.items():
        if comfy_prefix not in seen_prefixes:
            raise RuntimeError(
                f"turbo LoRA checkpoint is missing expected key prefix {comfy_prefix!r} "
                f"(mapped to transformer.{dotted_path}) -- checkpoint layout may have "
                "changed upstream. Refusing to partially apply the adapter."
            )
        lora_a = lora_sd[f"{comfy_prefix}.lora_A.weight"].to(device=device, dtype=torch.bfloat16)
        lora_b = lora_sd[f"{comfy_prefix}.lora_B.weight"].to(device=device, dtype=torch.bfloat16)
        base_linear = _get_module_by_dotted_path(transformer, dotted_path)
        if not isinstance(base_linear, torch.nn.Linear):
            raise RuntimeError(
                f"expected transformer.{dotted_path} to be an nn.Linear, got "
                f"{type(base_linear).__name__} (turbo LoRA key {comfy_prefix!r})"
            )
        if lora_a.shape[1] != base_linear.in_features or lora_b.shape[0] != base_linear.out_features:
            raise RuntimeError(
                f"turbo LoRA shape mismatch for {comfy_prefix!r} -> transformer.{dotted_path}: "
                f"lora_A={tuple(lora_a.shape)} lora_B={tuple(lora_b.shape)} vs "
                f"base in_features={base_linear.in_features} out_features={base_linear.out_features}"
            )
        _set_module_by_dotted_path(transformer, dotted_path, _TurboLoRALinear(base_linear, lora_a, lora_b))
        n_wrapped += 1
        seen_prefixes.discard(comfy_prefix)
    if seen_prefixes:
        # Checkpoint has adapter keys this function does not know how to place -- fail
        # loudly rather than silently apply a partial LoRA (e.g. missing the token
        # refiner or final_layer would leave the model in an unverified state).
        raise RuntimeError(
            f"turbo LoRA checkpoint has {len(seen_prefixes)} key prefix(es) with no mapping "
            f"in _turbo_lora_key_map(): {sorted(seen_prefixes)[:5]}{'...' if len(seen_prefixes) > 5 else ''}"
        )
    logger.info(
        "turbo LoRA applied: %d Linear layers wrapped from %s in %.2fs",
        n_wrapped, lora_path, time.time() - t_load,
    )
    return n_wrapped


def detect_turbo_lora_format(lora_path: str) -> str:
    """turbo LoRA チェックポイントのキー形式を判定する: "comfy" | "diffusers"。

    - comfy (Ostris 版): `blocks.N.attn.qkv_proj.lora_A.weight` -- 融合QKVを対象と
      するため適用に `fuse_projections()` (= int8 では `aten.cat` 非互換) が要る
    - diffusers (lightx2v 版): `transformer_blocks.N.attn.to_q.lora_A.default.weight`
      -- パスがそのままモジュールパスで、融合不要 (int8 でも適用可)

    ヘッダのキーだけ読む (テンソル本体はロードしない) ので軽い。未知の形式は
    ValueError -- 黙って誤った適用関数に流さない。
    """
    from safetensors import safe_open

    with safe_open(lora_path, framework="pt") as f:
        keys = list(f.keys())
    # comfy 署名 (`qkv_proj`) を先に見ること: Ostris 版も `token_refiner.blocks.*` の
    # キーを持つため、プレフィックスだけで diffusers 判定すると誤検出する
    # (実ファイルで再現して修正済み)。
    if any(".qkv_proj.lora_A." in k for k in keys):
        return "comfy"
    if any(".attn.to_q.lora_A." in k for k in keys):
        return "diffusers"
    raise ValueError(
        f"unrecognized turbo LoRA key format in {lora_path} "
        f"(sample keys: {keys[:3]}) -- expected comfy (qkv_proj) or diffusers-native "
        "(transformer_blocks.*.to_q) layout."
    )


def turbo_lora_expected_format() -> str:
    """設定中の turbo LoRA (H3_TURBO_LORA_REPO/FILE) の想定キー形式: "comfy"|"diffusers"。

    リクエスト時バリデーション (core/settings.py) と UI のグレーアウト判定用。
    ファイルが HF キャッシュに既にあれば実物のキーで判定し、未ダウンロードなら既知
    リポジトリ名 (`_TURBO_COMFY_REPOS`) で予備判定する。未知リポジトリは diffusers
    ネイティブと仮定する -- 仮定が外れて実は comfy 形式だった場合も、適用時の
    `_apply_turbo_lora_checkpoint()` の実ファイル判定が int8 との組み合わせを 400 で
    拒否するので、黙って壊れることはない。"""
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(H3_TURBO_LORA_REPO, H3_TURBO_LORA_FILE)
        if isinstance(cached, str):
            return detect_turbo_lora_format(cached)
    except Exception:
        logger.debug("turbo_lora_expected_format: cache probe failed", exc_info=True)
    return "comfy" if H3_TURBO_LORA_REPO in _TURBO_COMFY_REPOS else "diffusers"


def resolve_turbo_lora_scale(lora_format: str, lora_path: str | None = None) -> float:
    """適用係数を解決する。優先順位:

    1. `H3_TURBO_LORA_SCALE` が明示されていればそれを最優先。
    2. 未指定なら、`lora_path` のチェックポイント自身の safetensors metadata に
       `alpha` があれば `scale = float(alpha) / rank` を使う (rank はソート済み
       最初の `lora_A` キーの shape[0] -- lightx2v/Minimax-h3-Turbo の3ファイルは
       いずれも128)。2026-08-12 に追加された v1.0 系 (8step/768p) はどちらも
       metadata に `alpha` を持つ (8step: alpha='8' -> scale=0.0625, 768p:
       alpha='128' -> scale=1.0 -- 実測でスパイク済み、README「2026-08-12」節参照)
       ので、ここで自動導出できる。
    3. metadata に `alpha` が無ければ (例: 旧 v0.1 ファイル) 形式ごとの実測既定へ
       フォールバック: comfy=1.0 / diffusers=0.094 (導出と強度スイープの実測は
       H3_TURBO_LORA_SCALE のモジュールコメントと README を参照)。

    解決した scale とその出典を1行 logger.info する。
    """
    if H3_TURBO_LORA_SCALE_RAW:
        scale = float(H3_TURBO_LORA_SCALE_RAW)
        logger.info("turbo LoRA scale resolved: %.4f (source=env H3_TURBO_LORA_SCALE)", scale)
        return scale

    if lora_path is not None:
        try:
            from safetensors import safe_open

            with safe_open(lora_path, framework="pt") as f:
                metadata = f.metadata() or {}
                # `lora_alpha` は PDMD 版 (pdmd2026/*) の metadata キー名
                alpha_raw = metadata.get("alpha") or metadata.get("lora_alpha")
                if alpha_raw is not None:
                    lora_a_keys = sorted(k for k in f.keys() if ".lora_A." in k)
                    rank = f.get_slice(lora_a_keys[0]).get_shape()[0]
                    scale = float(alpha_raw) / rank
                    logger.info(
                        "turbo LoRA scale resolved: %.4f (source=metadata alpha=%s, rank=%d, file=%s)",
                        scale, alpha_raw, rank, lora_path,
                    )
                    return scale
        except Exception:
            logger.warning(
                "turbo LoRA metadata alpha probe failed for %s, falling back to format default",
                lora_path, exc_info=True,
            )

    scale = 1.0 if lora_format == "comfy" else 0.094
    logger.info("turbo LoRA scale resolved: %.4f (source=format-default, format=%s)", scale, lora_format)
    return scale


def apply_diffusers_turbo_lora(transformer, lora_path: str, scale: float) -> int:
    """diffusers ネイティブキーの turbo LoRA (lightx2v 版) を融合なしで巻く。

    `apply_turbo_lora()` (comfy 版) との違い: `fuse_projections()` も
    `delattr(to_q/to_k/to_v)` も**呼ばない** -- キーのドット付きパスがそのまま
    モジュールパスなのでキーマップも不要。`torch.cat` を一切呼ばないため、torchao
    int8 量子化済み transformer にもそのまま適用できる (`Int8Tensor` の base(x) は
    weight-only 量子化なので出力 dtype は入力活性化 (bf16) に従い、bf16 の LoRA
    デルタとの加算に dtype 不整合は無い -- torchao 0.17.0 の int8_tensor.py で確認)。
    2026-08-08 のスパイク (`scripts/probe_lightx2v_turbo.py`) で int8 への適用と
    4steps の品質を実測済み。

    注意: `H3_INT8_MODULES_TO_NOT_CONVERT` に `token_refiner` が入っているため、int8
    モードでは transformer_blocks 側 (300モジュール) が int8 ベース + bf16 デルタ、
    token_refiner 側 (12モジュール) が bf16 ベース + bf16 デルタの混在になる --
    スパイクで生成まで通ることを確認済み (実害は観測されていない)。

    既に comfy 版が適用済み (= `fused_projections` が真で to_q が無い) の transformer
    には適用できないので、事前に検出して明確に拒否する。返り値は巻いた Linear 数
    (このチェックポイントでは 312 = 50ブロック×6 + refiner 2ブロック×6)。
    """
    from safetensors.torch import load_file

    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention

    for module in transformer.modules():
        if isinstance(module, MiniMaxH3Attention) and getattr(module, "fused_projections", False):
            raise RuntimeError(
                "transformer の QKV が既に融合されています (comfy 版 turbo LoRA を適用済み?)。"
                "diffusers ネイティブ LoRA は未融合の to_q/to_k/to_v を対象とするため適用できません。"
            )

    t_load = time.time()
    lora_sd = load_file(lora_path)
    # キー形式の正規化: lightx2v 版は `<path>.lora_A.default.weight`、PDMD 版
    # (pdmd2026/pdmd_{2,4}NFE_lora) は `transformer.<path>.lora_A.weight`。どちらも
    # `<path>.lora_A.default.weight` へ揃える (既存の lightx2v ファイルは無変更で通る)。
    _norm: dict[str, torch.Tensor] = {}
    for k, v in lora_sd.items():
        nk = k[len("transformer."):] if k.startswith("transformer.") else k
        nk = nk.replace(".lora_A.weight", ".lora_A.default.weight").replace(
            ".lora_B.weight", ".lora_B.default.weight"
        )
        _norm[nk] = v
    lora_sd = _norm
    paths = sorted({k.rsplit(".lora_", 1)[0] for k in lora_sd if ".lora_" in k})
    device = next(transformer.parameters()).device
    n_wrapped = 0
    for path in paths:
        lora_a = lora_sd[f"{path}.lora_A.default.weight"].to(device=device, dtype=torch.bfloat16)
        lora_b = lora_sd[f"{path}.lora_B.default.weight"].to(device=device, dtype=torch.bfloat16)
        base_linear = _get_module_by_dotted_path(transformer, path)
        if not isinstance(base_linear, torch.nn.Linear):
            raise RuntimeError(
                f"expected transformer.{path} to be an nn.Linear, got {type(base_linear).__name__}"
            )
        if lora_a.shape[1] != base_linear.in_features or lora_b.shape[0] != base_linear.out_features:
            raise RuntimeError(
                f"turbo LoRA shape mismatch for {path!r}: lora_A={tuple(lora_a.shape)} "
                f"lora_B={tuple(lora_b.shape)} vs in_features={base_linear.in_features} "
                f"out_features={base_linear.out_features}"
            )
        _set_module_by_dotted_path(transformer, path, _TurboLoRALinear(base_linear, lora_a, lora_b, scale=scale))
        n_wrapped += 1
    logger.info(
        "diffusers-native turbo LoRA applied: %d Linear layers wrapped (scale=%.3f) from %s in %.2fs",
        n_wrapped, scale, lora_path, time.time() - t_load,
    )
    return n_wrapped


def set_turbo_lora_enabled(transformer, enabled: bool) -> int:
    """Flip every already-wrapped `_TurboLoRALinear` module's `enabled` flag in place.

    Instant, no reload: `apply_turbo_lora()` only ever needs to run once per transformer
    (wrapping is structural -- replacing `nn.Linear` modules with `_TurboLoRALinear`
    ones); a *disabled* wrapper's `forward()` degrades to exactly `self.base(x)` (see
    `_TurboLoRALinear.forward`), so toggling this flag on/off between requests is
    numerically equivalent to the LoRA never having been applied at all, without paying
    the ~780MB download + fuse_projections()/wrap cost again. Returns the number of
    wrapper modules found (0 if the turbo LoRA was never applied to this transformer --
    callers use this to distinguish "toggled" from "nothing to toggle, apply it first").
    """
    n = 0
    for module in transformer.modules():
        if isinstance(module, _TurboLoRALinear):
            module.enabled = enabled
            n += 1
    return n


def align_num_frames(num_frames: int) -> int:
    while num_frames % 17 != 5:
        num_frames += 1
    return num_frames


def seconds_to_num_frames(seconds: float) -> int:
    seconds = max(MIN_SECONDS, min(MAX_SECONDS, seconds))
    return align_num_frames(round(seconds * FPS))


# デコード結果 (fp32 の全長テンソル) を uint8 の numpy 配列にするときの、一度に処理する
# フレーム数。8 は「削減がほぼ頭打ちになる最小」を実測で選んだ値 (下記)。
_FRAMES_TO_UINT8_CHUNK = 8


def frames_to_uint8(video_tensor: torch.Tensor, chunk: int = _FRAMES_TO_UINT8_CHUNK) -> np.ndarray:
    """`(F, C, H, W)` の fp32 デコード結果を `(F, H, W, C)` の uint8 numpy 配列にする。

    **なぜチャンクに分けるのか**: 素直に書くと
    `(v.permute(...).float().clamp(0,1) * 255).round().to(torch.uint8).cpu().numpy()` になるが、
    これは**全長ぶんの中間テンソルを何本も GPU 上に作る** (`float()` / `clamp` / `*255` /
    `round()` が各々新しいテンソルを返し、最後に uint8 版も作られる)。768x1344・107フレームで
    **+2.65GB** を積んでいた (2026-08-10 実測)。デコード位相のピーク 16.29GB のうち 15% が
    ここだった、という内訳の分解から見つけたもの。

    フレームを少しずつ変換して CPU 側の出力配列へ直接書き込めば、GPU に同時に存在するのは
    `chunk` フレームぶんだけになる。**実測 +2.65GB → +0.03GB (-99%)、出力は
    `np.array_equal` でバイト完全一致**。演算順序は現行と同一 (permute → float → clamp →
    ×255 → round → uint8) なので丸めも変わらない。

    `chunk` を大きくしても速度はほぼ変わらず (転送はどのみち全長ぶん)、小さくしすぎると
    Python ループのオーバーヘッドが出る。8 で削減はほぼ飽和する。
    """
    if video_tensor.dim() == 5:
        video_tensor = video_tensor[0]
    # チャネル数は実テンソルから取る (H3 は常に RGB=3 だが、置き換え前のコードは
    # チャネル数に依存しない書き方だったので、その一般性を保つ)。
    num_frames, channels, height, width = video_tensor.shape
    out = np.empty((num_frames, height, width, channels), dtype=np.uint8)
    for i in range(0, num_frames, chunk):
        part = (
            video_tensor[i : i + chunk]
            .permute(0, 2, 3, 1)
            .float()
            .clamp_(0, 1)
            .mul_(255)
            .round_()
            .to(torch.uint8)
        )
        out[i : i + chunk] = part.cpu().numpy()
        del part
    return out


# `MiniMaxH3VideoDecodeStep.__call__` (decoders.py, f37ab93) の置き換え。venv の diffusers は
# 無改変のまま、サブクラスで __call__ だけを差し替える (frames_to_uint8 と同族の対策)。
#
# 上流の最終行 `video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)` は、VAE が
# fp16 で出した全長テンソルを **GPU 上で一括 fp32 化**する。768²・124フレームで
# 124x768x768x3x4B = 838MiB の一時確保が `float()` / mul / add / clamp の各段で発生し、
# 8GB カード検証 (2026-08-11、ゴールB) では**デノイズは完走したのにこの1行で OOM** した。
# ここでは fp16 のまま CPU へ移してから逆正規化する。要素毎の fp32 mul/add/clamp は
# CPU/GPU で IEEE754 の丸めが一致する (縮約も FMA 融合もない) ので**出力はビット単位で
# 同一** -- 適用直後に同一 seed の PNG MD5 一致で実証済み (README 2026-08-11 の節)。
# 追加コストは fp16 全長 (~420MiB) の PCIe 転送1回と CPU 演算のみ。後段の
# postprocess_video / frames_to_uint8 はデバイス非依存で、CPU テンソルのまま処理できる。
_CPU_NORM_DECODE_STEP_CLS = None


def _cpu_norm_video_decode_step():
    global _CPU_NORM_DECODE_STEP_CLS
    if _CPU_NORM_DECODE_STEP_CLS is None:
        from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3VideoDecodeStep

        class _CpuNormVideoDecodeStep(MiniMaxH3VideoDecodeStep):
            @torch.no_grad()
            def __call__(self, components, state):
                block_state = self.get_block_state(state)
                device = components._execution_device

                if block_state.output_type not in ("pil", "np", "pt"):
                    raise ValueError(
                        f"`output_type` must be one of 'pil', 'np' or 'pt', got {block_state.output_type!r}. To keep the "
                        "latents instead of decoding them, run a pipeline that does not include the decode blocks."
                    )

                latents_mean = torch.tensor(components.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
                latents_std = torch.tensor(components.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
                latents = block_state.latents * latents_std + latents_mean

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    video = components.vae.decode(latents, return_dict=False)[0]
                # ここからが上流との差分: fp16 のまま CPU へ降ろし、逆正規化を CPU で行う
                # (上流は GPU 上で video.float() から一括生成する)。
                video = video.cpu()
                pixel_mean = torch.tensor(components.pixel_mean).view(1, -1, 1, 1, 1)
                pixel_std = torch.tensor(components.pixel_std).view(1, -1, 1, 1, 1)
                video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
                block_state.videos = components.video_processor.postprocess_video(
                    video, output_type=block_state.output_type
                )

                self.set_block_state(state, block_state)
                return components, state

        _CPU_NORM_DECODE_STEP_CLS = _CpuNormVideoDecodeStep
    return _CPU_NORM_DECODE_STEP_CLS()


def _num_frames_from_audio_reference(references: list, fps: int) -> int:
    r"""ref2va の `seconds=None` を、ちょうど1本の音声を持つ参照 (単体の
    `MiniMaxH3AudioReference`、または音声付きの `MiniMaxH3VideoReference`) の長さから
    `num_frames` へ変換する。

    PR #14355 (f37ab93) 前は `MiniMaxH3Ref2VASetupStep.prepare_references` がこの導出を
    内部で行っていたが、後継の `MiniMaxH3Ref2VASetupStep.__call__` は `num_frames` を
    必須入力にし (before_encoder.py)、この導出そのものを削除した -- そのブロックの
    `num_frames` の docstring 自身が代わりのレシピを明記している:
    `round(samples / sample_rate * 24)`。この関数はそのレシピをそのまま適用し、旧実装の
    「音声を持つ参照がちょうど1本のときだけ許可、それ以外は ValueError」という制約を
    维持する (呼び出し側のドキュメント/バリデーション文言と一致させるため)。

    参照はまだ正規化前 (`MiniMaxH3Ref2VASetupStep` を通す前) の生の入力なので、
    `sample_rate` が None のときは記録された値をそのまま秒数計算に使う (正規化後の
    audio_sampling_rate ではなく、参照自身が運んできたレート -- 正規化はここでは
    まだ行われていないので、無指定なら「そのまま」を意味する `MiniMaxH3VideoReference`/
    `MiniMaxH3AudioReference` の宣言どおりに扱う)。
    """
    audio_bearing = [entry for entry in references if entry.has_audio]
    if len(audio_bearing) != 1:
        raise ValueError(
            "`seconds` を省略できるのは references に音声を持つ参照 (単体の "
            "MiniMaxH3AudioReference、または音声付きの MiniMaxH3VideoReference) が "
            f"ちょうど1本のときだけです。見つかった数: {len(audio_bearing)}。"
        )
    reference = audio_bearing[0]
    waveform = reference.audio
    sample_rate = reference.sample_rate
    if sample_rate is None:
        raise ValueError(
            "音声参照の sample_rate が不明なため、seconds を自動導出できません。"
            "`from_file()` で読み込んだ参照は必ず sample_rate を持つはずです。"
        )
    num_samples = waveform.shape[-1]
    return align_num_frames(round(num_samples / sample_rate * fps))


def _build_vocal_lock_latents(pipe, references: list, actual_num_frames: int) -> torch.Tensor | None:
    r"""Vocal Lock (`H3_VOCAL_LOCK=1`) 用: 駆動音声を encode して `(2, audio_latent_channels,
    N)` の audio_latents テンソルにして返す。`references` に `MiniMaxH3AudioReference` が
    1つも無ければ `None` を返す (呼び出し側はこれを「発動しない」合図として扱うこと)。

    最初に見つかった `MiniMaxH3AudioReference` の波形を流用する (専用の API 引数は
    追加しない -- 参照としての音声はそのまま `references` に残り、ref2va の通常の
    参照エンコード経路 (`MiniMaxH3Ref2VAReferenceEncoderStep`) が別途エンコードする)。

    正規化・エンコード手順は `MiniMaxH3Ref2VAReferenceEncoderStep.__call__`
    (diffusers `modular_pipelines/minimax_h3/encoders.py:740-760` 付近) と完全に同一
    にする (独自の正規化を発明しない): `audio_vae.encode(...)` -> `posterior.mode()`
    -> `transpose` -> `(x - latents_mean) / latents_std`。ここでは行化
    (`.reshape(-1, C)`) はせず、`MiniMaxH3PrepareLatentsStep` がそのまま受け付ける
    `(2, C, N)` 形状で返す (行化はそのステップ自身が
    `.permute(0, 2, 1).reshape(-1, C)` で行う -- 参照エンコーダ出力とビット一致する
    ことは `probe_layout.py`/`check_layout.py`/`pack_roundtrip.py` で検証済み)。

    `N = audio_latent_num_frames(actual_num_frames)` に厳密に合わせる: 波形を
    `N * 800` サンプル (audio_vae は 32000Hz・800 samples/latent) へ trim/pad してから
    encode し、encode 結果が N と1行ずれる場合に備えて encode 後にも latent 側で
    trim/pad し、`assert` で N 一致を確認する。
    """
    from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
        MINIMAX_H3_AUDIO_LATENTS_PER_SECOND,
        audio_latent_num_frames,
    )

    audio_ref = next((entry for entry in references if isinstance(entry, MiniMaxH3AudioReference)), None)
    if audio_ref is None:
        logger.warning(
            "H3_VOCAL_LOCK=1 ですが references に MiniMaxH3AudioReference がありません -- "
            "Vocal Lock は発動せず、通常の ref2va 経路にフォールバックします。"
        )
        return None

    device = pipe._execution_device
    audio_channels = pipe.audio_channels
    audio_latent_channels = pipe.audio_latent_channels
    target_sample_rate = pipe.audio_sampling_rate
    N = audio_latent_num_frames(actual_num_frames)
    samples_per_latent = target_sample_rate // MINIMAX_H3_AUDIO_LATENTS_PER_SECOND
    target_num_samples = N * samples_per_latent

    sample_rate = audio_ref.sample_rate
    if sample_rate is None:
        sample_rate = target_sample_rate
    orig_num_samples = int(audio_ref.audio.shape[-1])

    # `_normalize_audio_condition` は truncate のみ (max_duration 経由) で pad はしない。
    # そのため先に truncate+resample+mono->stereo をこのステップに任せ、その後 pad/trim を
    # このヘルパー側で厳密に行う (target_sample_rate に揃った後のサンプル数基準で行う
    # 必要があるため -- max_duration は resample 前の sample_rate 基準の秒数)。
    waveform = MiniMaxH3Ref2VASetupStep._normalize_audio_condition(
        audio_ref.audio,
        sample_rate,
        target_sample_rate,
        max_duration=max(orig_num_samples / sample_rate, target_num_samples / target_sample_rate),
    )
    resampled_num_samples = int(waveform.shape[-1])
    if resampled_num_samples < target_num_samples:
        waveform = torch.nn.functional.pad(waveform, (0, target_num_samples - resampled_num_samples))
    elif resampled_num_samples > target_num_samples:
        waveform = waveform[:, :target_num_samples]
    trimmed_num_samples = int(waveform.shape[-1])

    logger.info(
        "Vocal Lock: driving audio orig_samples=%d (sr=%s) -> trimmed/padded=%d samples "
        "(target N=%d, samples/latent=%d)",
        orig_num_samples, sample_rate, trimmed_num_samples, N, samples_per_latent,
    )

    audio_latents_mean = torch.tensor(pipe.audio_vae.config.latents_mean).view(1, 1, -1)
    audio_latents_std = torch.tensor(pipe.audio_vae.config.latents_std).view(1, 1, -1)

    with torch.no_grad():
        posterior = pipe.audio_vae.encode(waveform.to(device)[:, None], return_dict=False)[0]
        latents = posterior.mode().float().cpu().transpose(1, 2)  # (2, T, C)
        normalized = (latents - audio_latents_mean) / audio_latents_std  # (2, T, C)

    encoded_num_frames = normalized.shape[1]
    if encoded_num_frames != N:
        # audio_vae のチャンク境界の丸めで N と1行ずれることがある -- latent 側でも
        # 厳密に N へ trim/pad する (task の指示どおり)。
        if encoded_num_frames < N:
            pad = N - encoded_num_frames
            normalized = torch.nn.functional.pad(normalized, (0, 0, 0, pad))
        else:
            normalized = normalized[:, :N, :]

    audio_latents = normalized.transpose(1, 2).contiguous()  # (2, C, N) -- what PrepareLatentsStep expects
    assert audio_latents.shape == (audio_channels, audio_latent_channels, N), (
        f"Vocal Lock: injected audio_latents shape {tuple(audio_latents.shape)} != "
        f"expected {(audio_channels, audio_latent_channels, N)}"
    )
    logger.info(
        "Vocal Lock: encoded latent shape=%s (posterior.mode) -> injected shape=%s",
        tuple(latents.shape), tuple(audio_latents.shape),
    )
    return audio_latents


def _inflate_vocal_lock_condition_rows(state, num_generated_audio_latents: int, audio_channels: int) -> int:
    r"""Vocal Lock (`H3_VOCAL_LOCK=1`) 用: `state["num_condition_audio_rows"]` を
    「生成音声行も含めてすべて条件行」に見せかけ、denoise 中はクリーン (t=1.0固定・
    無変更) に凍結する。書き換え前の (参照行数のみの) 元の値を返す -- 呼び出し側は
    これを保持しておき、`_restore_vocal_lock_condition_rows()` に渡して必ず復元する
    こと。

    **重要**: `MiniMaxH3Ref2VADenoiseStep.__call__`
    (`MiniMaxH3DenoiseLoopWrapper.__call__`, diffusers `denoise.py:260-266`) は
    `state` から `block_state` を**ループの前に1回だけ** `get_block_state()` で
    読み出し、以後 denoise ループの全ステップはその `block_state` (ローカルな
    スナップショット) を使い回す (`state` への書き戻しはループ終了後の
    `set_block_state()` のみ)。そのため、この関数で膨らませた
    `num_condition_audio_rows` は **`denoise_step(pipe, state)` の呼び出しが
    終わるまで `state` 上で膨らんだままにしておく必要がある** (`timesteps_step` の
    呼び出しが終わった直後に戻してしまうと、後で呼ばれる denoise ループが
    元の値でスナップショットを取ってしまい、生成音声行が凍結されない)。

    呼び出し側 (`generate_ref2va()` の4分岐、将来的には `generate_ref_batch()` も) は
    「`ref2va_latents_step` の後・`timesteps_step` の前」でこれを呼び (`ref2va_latents_step`
    自身が参照行数との一致を検証するため、それより前に書き換えてはいけない)、
    「`MiniMaxH3AfterDenoiseStep` の直前」(denoise ループの後) で
    `_restore_vocal_lock_condition_rows()` を呼んで元へ戻すこと。
    """
    original = state.get("num_condition_audio_rows")
    inflated = original + num_generated_audio_latents * audio_channels
    state.set("num_condition_audio_rows", inflated)
    logger.info(
        "Vocal Lock: num_condition_audio_rows %s -> %s (freezing generated audio rows)",
        original, inflated,
    )
    return original


def _restore_vocal_lock_condition_rows(state, original: int) -> None:
    r"""`_inflate_vocal_lock_condition_rows()` が返した元の値へ `state`
    (`num_condition_audio_rows`) を戻す。`MiniMaxH3AfterDenoiseStep` の呼び出し直前に
    必ず呼ぶこと -- 戻し忘れると、生成行だけを切り出す `decoders.py` の
    `audio_rows.reshape(components.audio_channels, block_state.num_audio_latents, ...)`
    が 0 要素の reshape になり確実に落ちる。
    """
    state.set("num_condition_audio_rows", original)
    logger.info("Vocal Lock: num_condition_audio_rows restored -> %s", original)


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


def _adaln_precompute_status(self: "MiniMaxH3Runner") -> dict:
    """`status()` helper: whether each currently-resident transformer instance actually
    has its AdaLN precompute table built yet (see H3_ADALN_PRECOMP's own module comment
    and `core/adaln_precompute.py`'s `enable_adaln_precompute()` docstring for why this
    can lag the `H3_ADALN_PRECOMP` flag itself -- precompute is armed at load time but
    only fires on the first denoise step of the next request against that instance).
    `False` (not `None`) when H3_ADALN_PRECOMP is off or the transformer is not loaded,
    so callers never need a three-way None/True/False check.
    """
    if not H3_ADALN_PRECOMP:
        return {"transformer": False, "transformer_ref": False}
    from core.adaln_precompute import is_precomputed

    return {
        "transformer": bool(self._transformer_loaded and is_precomputed(self._pipe.transformer)),
        "transformer_ref": bool(
            self._transformer_ref_loaded and is_precomputed(self._pipe_ref.transformer_ref)
        ),
    }


def _adaln_precompute_built_with_turbo(self: "MiniMaxH3Runner") -> dict:
    """`status()` helper (v2 coexistence): which turbo state each currently-installed
    precompute table was built while observing -- `None` per transformer that has no
    table installed yet (mirrors `_adaln_precompute_status()`'s own shape/gating, see
    that function's docstring). Purely informational (`core/adaln_precompute.py`'s
    `built_with_turbo()` docstring: both turbo states are served correctly off the same
    table for the default LoRA format), exposed so an operator watching `/api/status`
    can confirm which state the table was actually built against without needing to
    correlate it against request logs by hand.
    """
    if not H3_ADALN_PRECOMP:
        return {"transformer": None, "transformer_ref": None}
    from core.adaln_precompute import built_with_turbo

    return {
        "transformer": built_with_turbo(self._pipe.transformer) if self._transformer_loaded else None,
        "transformer_ref": (
            built_with_turbo(self._pipe_ref.transformer_ref) if self._transformer_ref_loaded else None
        ),
    }


def _log_gpu_tensor_diag(label: str, top_n: int = 20):
    """TEMPORARY diagnostic (opt-in via H3_DEBUG_MEM_DIAG=1): walks `gc.get_objects()` for
    live CUDA tensors and logs the largest ones by byte size, to find what is actually
    holding VRAM at a given point (as opposed to `torch.cuda.memory_allocated()`'s
    aggregate total, which does not say *what*). Used once during this task's own
    32GB-ballast investigation of decode's ~16GB-on-top-of-denoise's-~30GB peak. Not
    wired into any code path unless the env var is set -- safe to leave in place.
    """
    import gc as _gc

    seen = set()
    entries = []
    for obj in _gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                key = obj.data_ptr()
                if key in seen:
                    continue
                seen.add(key)
                nbytes = obj.numel() * obj.element_size()
                entries.append((nbytes, tuple(obj.shape), str(obj.dtype)))
        except Exception:
            continue
    entries.sort(key=lambda x: -x[0])
    total = sum(e[0] for e in entries) / 1e9
    logger.info("[mem-diag] %s: %d live cuda tensors, %.2fGB total (dedup by data_ptr)", label, len(entries), total)
    for nbytes, shape, dtype in entries[:top_n]:
        logger.info("[mem-diag]   %.3fGB shape=%s dtype=%s", nbytes / 1e9, shape, dtype)


class _NullContext:
    """A no-op context manager, used where FBC's `cache_context` is conditionally absent
    (H3_CACHE == "none") but the calling code wants one `with` statement either way."""

    def __enter__(self):
        return None

    def __exit__(self, *exc_info):
        return False


class GenerationInterrupted(Exception):
    """生成中断API(`POST /api/interrupt`)による中断を示す例外。

    denoise ループのステップ境界(`timed_loop_step` 系ラッパー、および hires-fix の
    `run_steps()`)でのみチェックされるため、反応は最大1ステップぶん遅れる
    (H3の1ステップは実測6〜9秒 -- 中断要求からその1ステップが終わるまで待つ形になる。
    現状の「シーン完走(約145秒)まで止まらない」よりは大幅に改善するが、即座には
    止まらない仕様として受け入れる)。

    ステップ途中のCUDA演算そのものを中断するものではない(そんな機構は無い) --
    直前のステップが正常に完了した直後、次のステップに入る前に例外を送出するだけ。
    そのため呼び出し元(`generate()` 等)の既存の `finally`/`except` によるVRAM解放・
    状態復元パスをそのまま通り、CUDA的に不整合な状態でVRAMが残ることはない
    (decode失敗時の `except BaseException: ... _restore_decode_steady_state()` と
    同じく、denoiseループの外側は無傷)。
    """


class _InterruptController:
    """プロセス内1件だけの生成を前提とした、シンプルな中断フラグ。

    どのバックエンドも同時1生成(app.py の `_generation_lock` が保証)なので、
    フラグは1個で足りる。複数スレッド(HTTPハンドラ用スレッドと、生成を実行している
    スレッド)から読み書きされるため `threading.Lock` で保護する。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requested = False
        self._job_id: str | None = None

    def begin(self, job_id: str | None) -> None:
        """生成開始時に必ず呼ぶこと。前回の要求が残っていて次の生成が即死するのを防ぐ。"""
        with self._lock:
            self._requested = False
            self._job_id = job_id

    def request(self, job_id: str | None) -> bool:
        """中断を要求する。

        `job_id` が指定されていて現在実行中のジョブと一致しない場合は何もせず False を
        返す(直後に始まった別のジョブを巻き添えにしないため)。現在ジョブ不明
        (アイドル中)の場合も False。実際に中断フラグを立てたら True。
        """
        with self._lock:
            if self._job_id is None:
                return False
            if job_id is not None and job_id != self._job_id:
                return False
            self._requested = True
            return True

    def check(self) -> None:
        """denoise ループのステップ境界から呼ぶ。要求が立っていれば例外を送出する。"""
        with self._lock:
            requested = self._requested
        if requested:
            raise GenerationInterrupted("生成が中断要求により停止しました")

    def current_job_id(self) -> str | None:
        with self._lock:
            return self._job_id

    def end(self) -> None:
        """生成終了時(成功・失敗・中断いずれでも)に呼ぶ。次の生成に影響しないようにする。"""
        with self._lock:
            self._job_id = None
            self._requested = False


interrupt_controller = _InterruptController()


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
        # H3_VAE_SPLIT: VAE のうち今 GPU にいる部分 ("encode"/"decode" の部分集合)。
        # `_vae_on_gpu` は「どれかが GPU にいる」の意味 (全部とは限らない)。
        self._vae_gpu_parts: set[str] = set()
        # H3_VAE_PINNED: param/buffer -> pinned CPU マスター。初回の退避時に遅延構築。
        self._vae_pin_map: dict = {}
        self._vae_pin_disabled = False
        # H3_BASE_PINNED / H3_REF_PINNED: transformer をモジュールごと保持し、重み実体
        # だけ pinned CPU マスター <-> GPU で付け替える (初回の park で遅延構築)。
        # slot = {"module", "pairs" [(tensor, pinned_master)], "wrapped" (park 時点の
        # turbo LoRA 適用状態)}。slot が存在する間、ラップ等の構造はモジュールに
        # そのまま生きている。キーは "base" / "ref"。
        self._pin_slots: dict[str, dict] = {}
        self._pin_disabled: set[str] = set()
        # H3_DECODE_DEVICE: decode 専用 VAE コピー (別 GPU 常駐) と、decode の直列化ロック・
        # 専用ストリーム。フラグ OFF の既定では一切使われない (None のまま)。
        self._decode_vae = None
        self._decode_audio_vae = None
        self._decode_lock = threading.Lock()
        self._decode_stream = None
        self._decode_warned = False
        self._load_lock = threading.Lock()

        # --- ref2va (omni-reference) additions ---
        # PR #14355 (f37ab93) note: there is only ONE ModularPipeline shell now, not two.
        # Pre-merge, `MiniMaxH3Ref2VABlocks` was a separate public blocks class, and the
        # default `ModularPipeline.from_pretrained(MODEL_ID)` shell (built from the t2va/
        # fl2va-only `MiniMaxH3Blocks` of that era) had no `transformer_ref` entry in its
        # `_component_specs` at all -- hence the old second `_pipe_ref` shell. Post-merge,
        # `MiniMaxH3Blocks` (default_blocks_name, what `_ensure_pipe_shell` below builds
        # with no `workflow=` argument) itself unions t2va+fl2va+ref2va: its
        # `expected_components` includes BOTH `transformer` and `transformer_ref` --
        # confirmed both by reading `SequentialPipelineBlocks.expected_components`
        # (modular_pipeline.py: unions every sub-block's own, and `MiniMaxH3AutoDenoiseStep`
        # -- one of `MiniMaxH3Blocks`' five sub-blocks -- lists `MiniMaxH3Ref2VACoreDenoiseStep`
        # as one of ITS three sub-blocks, which is what pulls `transformer_ref` in) and by
        # this project's own first-migration-stage boot log, which already showed
        # `self._pipe.component_names` containing `transformer_ref` right alongside
        # `transformer`. So `self._pipe.load_components(names=["transformer_ref"], ...)`
        # (see `_ensure_transformer_ref` below) works directly on the ONE shell -- no
        # second `init_pipeline()` call, no second spec table.
        # `self._pipe_ref` is kept as a plain alias for `self._pipe` (not a separate
        # object) purely so every `self._pipe_ref.transformer_ref` / `self._pipe_ref.*`
        # call site elsewhere in this file (there are dozens) keeps working unchanged --
        # it is always the exact same `ModularPipeline` instance as `self._pipe`, never a
        # distinct one. `transformer`/`transformer_ref` are each ~66.3GB bf16 and cannot
        # coexist in this card's ~96GB (same constraint as TE vs transformer above), so
        # only one of `self._pipe.transformer` / `self._pipe.transformer_ref` is ever
        # GPU-resident at a time (except `H3_TRANSFORMER_BOTH_RESIDENT`'s int8 mode, where
        # both fit) -- tracked by `self._active_variant`.
        self._pipe_ref = None
        self._transformer_ref_loaded = False
        # "t2va" | "ref2va" | None (nothing loaded yet). Only one of `transformer` /
        # `transformer_ref` may be GPU-resident at a time; this is the single source of
        # truth callers check before a cross-variant swap.
        self._active_variant: str | None = None
        # H3_TURBO_LORA only: cached local path of the downloaded turbo LoRA safetensors,
        # resolved once per process by `_download_turbo_lora_if_needed()`.
        self._turbo_lora_path: str | None = None
        # H3_TURBO_LORA_FILE_BASE (base transformer 専用) を別指定したときだけ使う。
        self._turbo_lora_path_base: str | None = None
        # H3_TE_PROJ only: cached local path of the resolved projection safetensors,
        # resolved once per process by `_resolve_te_proj_path()`. The projection
        # instance itself is cached on `self._pipe._te_projection` (not here), since
        # encode-side helpers (`_encode_h3_prompt` etc.) only receive `components`
        # (the pipe shell), never `self`.
        self._te_proj_path: str | None = None
        # TE 外部常駐 (H3_TE_DEVICE) のときの TE 実体。パイプからは普段外しておき
        # (`_te_attached()` 参照)、この属性が唯一の強参照になる。
        self._te_module = None
        # Instant-apply turbo toggle (see core/settings.py / _TurboLoRALinear.enabled):
        # whether `apply_turbo_lora()` has structurally wrapped `transformer`'s Linear
        # modules yet. Wrapping only ever happens once per transformer instance (lazily,
        # on the first request that asks for turbo=1) -- once wrapped, every later
        # request just flips `_TurboLoRALinear.enabled` via `set_turbo_lora_enabled()`,
        # which is instant (no reload). Reset to False whenever `transformer` itself is
        # freed/reloaded (a fresh module has no wrapping yet).
        self._turbo_lora_wrapped = False
        self._turbo_lora_wrapped_ref = False

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

        # H3_REF_IMAGE_SHORT_EDGE override (see its definition above for the "why").
        # Only touch the config when it actually differs from diffusers' own default, so
        # the default path calls `register_to_config` exactly as often as before this
        # change (zero times) and stays byte-for-byte identical.
        if H3_REF_IMAGE_SHORT_EDGE != H3_REF_IMAGE_SHORT_EDGE_DEFAULT:
            self._pipe.register_to_config(reference_image_short_edge=H3_REF_IMAGE_SHORT_EDGE)
            logger.info(
                "reference_image_short_edge overridden: %d -> %d (H3_REF_IMAGE_SHORT_EDGE). "
                "Affects ref2va's reference-prefix encode step: shorter edge -> fewer "
                "prefix tokens -> faster Qwen3-VL-32B prefix encode, but less reference "
                "detail reaches the prefix (character-consistency impact not A/B'd).",
                H3_REF_IMAGE_SHORT_EDGE_DEFAULT, H3_REF_IMAGE_SHORT_EDGE,
            )

    def _ensure_pipe_ref_shell(self):
        """PR #14355 (f37ab93): no second shell to build any more (see the `_pipe_ref`
        field comment in `__init__` for why) -- `self._pipe_ref` is just made to point at
        the same single `self._pipe` shell, which already has `transformer_ref` in its own
        `_component_specs`. Idempotent, and does not load any component weights. Kept as a
        method (rather than inlining the alias in `__init__`) so every existing call site
        that calls this before touching `self._pipe_ref` keeps working unchanged.
        """
        self._ensure_pipe_shell()
        self._pipe_ref = self._pipe

    def _sync_shared_components_to_ref(self):
        """PR #14355 (f37ab93): a no-op now that `self._pipe_ref is self._pipe` (see
        `_ensure_pipe_ref_shell`) -- there is nothing to mirror between two shells because
        there is only one. Kept (rather than deleted) so every existing call site that
        calls this before running a ref2va block keeps working unchanged; it still ensures
        the alias itself is set up.
        """
        self._ensure_pipe_ref_shell()

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
        # audio_vae の attention は「native」に固定する。
        #
        # `MiniMaxH3AudioAttnProcessor` は `backend=self._attention_backend`(既定 None)で
        # `dispatch_attention_fn` を呼ぶため、**バックエンドがグローバルに解決される**。
        # このアプリは transformer / transformer_ref にだけ `set_attention_backend()` を
        # 呼んでいるが、`H3_ATTN_BACKEND=sage` で起動すると audio_vae の attention まで
        # sage に流れる。ところが audio_vae は上のとおり**設計上 fp32 固定**(bf16 にすると
        # 音量が約20dB落ちる)で、sage は fp16/bf16 しか受け付けない:
        #
        #   sageattention/core.py: assert dtype in [torch.float16, torch.bfloat16]
        #   -> AssertionError: Input tensors must be in dtype of torch.float16 or torch.bfloat16
        #
        # このモジュールだけ明示的に native へ固定すれば、fp32 のまま矛盾なく動く。
        # 精度も native(SDPA)のほうが素直で、音声 VAE は計算量が小さく sage の利得もない。
        #
        # **踏んだ経緯**: 音声を含む参照 (`fully_copy` のリップシンク検証, 2026-08-10) で
        # 初めて発火した。この経路は「音声つき参照」でしか通らないため、それまでの
        # ref2va 回帰(画像参照のみ)を全てすり抜けていた。リクエストの `attn=` 上書きも
        # transformer 系にしか効かず回避できない、という点も含めて記録しておく。
        self._pipe.audio_vae.set_attention_backend("native")
        if H3_VIDEO_VAE_FP16:
            # `dtype=torch.float16` on the load_components call above would be a no-op
            # for this VAE (see H3_VIDEO_VAE_FP16's module comment) -- has to be a
            # manual cast after the fp32 load. audio_vae is deliberately excluded.
            t_cast = time.time()
            self._pipe.vae = self._pipe.vae.to(torch.float16)
            logger.info("video vae cast to float16 in %.2fs (H3_VIDEO_VAE_FP16=1)",
                        time.time() - t_cast)
            # デコードは上流ステップ自身が fp16 autocast を張る (decoders.py) が、
            # **エンコード**側 (encoders.py の encode_vae_condition -- ref2va の参照と
            # fl2va のキーフレーム条件付けが使う) は autocast なしで、内部で明示的に
            # `pixels.to(torch.float32)` してから `vae.encode()` を呼ぶ。fp16 化した VAE
            # では conv_in の bias (Half) と入力 (float) の不一致で必ず落ちる --
            # `H3_VIDEO_VAE_FP16=1 × 参照あり` は 2026-08-11 の 8GB×2 ref2va 検証で
            # 初めて併用され、そこで発覚した (それまでの ref2va 回帰は fp32 VAE 構成)。
            # デコード側と対称の fp16 autocast を encode だけに被せて吸収する。精度は
            # 設計内: encode_vae_condition は結果を `latents.to(torch.float16).float()` と
            # **自分で fp16 に丸めてから返す**ので、fp16 計算はその丸めと同格。
            _orig_vae_encode = self._pipe.vae.encode

            def _fp16_autocast_encode(sample, *args, **kwargs):
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=sample.is_cuda):
                    return _orig_vae_encode(sample, *args, **kwargs)

            self._pipe.vae.encode = _fp16_autocast_encode
        if TE_QUANT == "bnb-4bit":
            # Parked on CPU by default in this mode -- moved to GPU only for the phase
            # that needs them (keyframe encode / decode). See module docstring.
            self._pipe.vae.to(CPU)
            self._pipe.audio_vae.to(CPU)
            self._vae_on_gpu = False
            self._vae_gpu_parts = set()
        else:
            self._pipe.vae.to(DEVICE)
            self._pipe.audio_vae.to(DEVICE)
            self._vae_on_gpu = True
        self._pipe.load_components(names=["scheduler", "audio_scheduler"])
        # H3_VIDEO_SHIFT / H3_AUDIO_SHIFT (既定は空 = 配布 config の 12.0 / 3.0 のまま)。
        # 用途と根拠はモジュール定数のコメント参照 (turbo 4step v1.0 768p が video shift 6)。
        if H3_VIDEO_SHIFT:
            self._pipe.scheduler.set_shift(float(H3_VIDEO_SHIFT))
            logger.info("video scheduler shift overridden: %s (H3_VIDEO_SHIFT)", H3_VIDEO_SHIFT)
        if H3_AUDIO_SHIFT:
            self._pipe.audio_scheduler.set_shift(float(H3_AUDIO_SHIFT))
            logger.info("audio scheduler shift overridden: %s (H3_AUDIO_SHIFT)", H3_AUDIO_SHIFT)
        # turbo=1 のリクエストだけ video shift を切り替える `_apply_turbo_video_shift()`
        # の「元に戻す」先。H3_VIDEO_SHIFT の上書きが上で既に適用された*後*に読むことが
        # 重要 -- プロセス全体の base はこの値 (H3_VIDEO_SHIFT 指定時はそれ、未指定なら
        # 配布 config の 12.0) であって、常に 12.0 ではない。
        self._base_video_shift = self._pipe.scheduler.shift
        self._vae_loaded = True
        logger.info("vae/audio_vae loaded (%s) in %.1fs. gpu=%s ram=%s",
                     "GPU" if self._vae_on_gpu else "CPU", time.time() - t1, gpu_mem_gb(), ram_gb())
        # フェーズ境界での中断チェック(loading_vae): 固定費区間でも中断に反応できるように
        # する追加。denoise ループのステップ境界チェック(`interrupt_controller`
        # docstring 参照)と同じ「直前の重い処理が完全に終わった直後」の位置。
        interrupt_controller.check()

        from diffusers.video_processor import VideoProcessor

        if getattr(self._pipe, "video_processor", None) is None:
            self._pipe.video_processor = VideoProcessor(vae_scale_factor=16, do_normalize=False)

        # PR #14355 (f37ab93) note: `MiniMaxH3ResizeStep` (before_encoder.py, replacing the
        # old `MiniMaxH3SetupStep`) is a genuinely new consumer of an `image_processor`
        # component (`ComponentSpec("image_processor", VaeImageProcessor, config=
        # FrozenDict({"vae_scale_factor": 16}), default_creation_method="from_config")`) --
        # the pre-PR#14355 `MiniMaxH3SetupStep`/`MiniMaxH3Ref2VASetupStep` this project has
        # run until now used plain PIL/numpy resize helpers instead and declared zero
        # `expected_components` (confirmed by reading both in the pinned venv). Bootstrapped
        # by hand here, mirroring `video_processor`'s own pattern immediately above, for the
        # same reason: this project calls individual step classes directly rather than
        # running the full `MiniMaxH3Blocks` auto-pipeline, and it is not verified whether
        # that hand-assembled call style reliably triggers a `default_creation_method=
        # "from_config"` component's lazy auto-instantiation the way running the full block
        # tree would -- `video_processor`'s own manual bootstrap right above is this
        # project's existing precedent for not relying on that path. LOW CONFIDENCE: this
        # is new to this migration and has not been exercised against the real f37ab93
        # venv (see this task's own completion report for the full caveat).
        from diffusers.image_processor import VaeImageProcessor

        if getattr(self._pipe, "image_processor", None) is None:
            self._pipe.image_processor = VaeImageProcessor(vae_scale_factor=16)
        # H3_DECODE_DEVICE: 別 GPU 上の decode 専用コピーをここで一緒に作る (フラグ OFF なら no-op)。
        self._ensure_decode_vaes()

    # H3_VAE_SPLIT 用: VAE の子モジュールを encode 側 / decode 側に分ける。video VAE は
    # decoder が 4.5GiB (fp16)、encoder は 0.34GiB しかない (safetensors ヘッダの実測)。
    # audio VAE は encoder+pre_block+mean/logs_proj (encode) と dec_in_proj+decoder (decode)。
    # 各 VAE の encode()/decode() が触る子モジュールは autoencoder_kl_minimax_h3(.._audio).py
    # のソースで確認済み。
    _VAE_PARTS = {
        "vae": {
            "encode": ("encoder", "quant_conv"),
            "decode": ("post_quant_conv", "decoder"),
        },
        "audio_vae": {
            "encode": ("encoder", "pre_block", "mean_proj", "logs_proj"),
            "decode": ("dec_in_proj", "decoder"),
        },
    }

    def _vae_split_active(self) -> bool:
        """H3_VAE_SPLIT が効く条件。TE が常駐する KEEP_REF2VA 限定: VAE を部分配置すると
        `vae.device` (=最初のパラメータの置き場所) が CPU になる窓ができるが、KEEP 中は
        `_execution_device` が先頭コンポーネントの text_encoder (GPU) で決まるので影響しない。"""
        return H3_VAE_SPLIT and _keep_ref2va_active()

    def _move_vae_part(self, part: str, device) -> bool:
        """`part` ("encode"/"decode") の子モジュールだけを `device` へ移す。未知の子モジュール
        が見つかったら (将来の diffusers 更新対策) False を返し、呼び出し側が全体移動へ退避する。"""
        for attr, table in (("vae", self._VAE_PARTS["vae"]), ("audio_vae", self._VAE_PARTS["audio_vae"])):
            module = getattr(self._pipe, attr)
            known = set(table["encode"]) | set(table["decode"])
            if any(name not in known for name, _ in module.named_children()) or \
                    any(True for _ in module.named_parameters(recurse=False)) or \
                    any(True for _ in module.named_buffers(recurse=False)):
                return False
        for attr in ("vae", "audio_vae"):
            module = getattr(self._pipe, attr)
            for name in self._VAE_PARTS[attr][part]:
                getattr(module, name).to(device)
        return True

    # ---- H3_VAE_PINNED (pinned CPU マスター方式の VAE 退避) ----------------------

    def _vae_pin_masters(self) -> dict | None:
        """pinned CPU マスターを遅延構築して返す。失敗したら以後無効化して None
        (呼び出し側は従来の `.to()` 経路へ退避する)。重みが変化しない前提のため、
        もし将来 VAE を finetune/再キャストする経路を足すならマスターを破棄すること。"""
        if self._vae_pin_disabled:
            return None
        if self._vae_pin_map:
            return self._vae_pin_map
        try:
            t0 = time.time()
            m: dict = {}
            for attr in ("vae", "audio_vae"):
                module = getattr(self._pipe, attr)
                for t in itertools.chain(module.parameters(), module.buffers()):
                    m[t] = t.data.detach().to(CPU, copy=True).pin_memory()
                    t.data = m[t]  # 以後 CPU 側の実体は常にこのマスター
            gib = sum(x.numel() * x.element_size() for x in m.values()) / 2**30
            logger.info("H3_VAE_PINNED: pinned CPU masters built (%.2fGiB, %d tensors, %.2fs)",
                        gib, len(m), time.time() - t0)
            self._vae_pin_map = m
            return m
        except Exception:
            logger.exception("H3_VAE_PINNED: pinned マスター作成に失敗、従来の .to() 経路へ退避")
            self._vae_pin_map = {}
            self._vae_pin_disabled = True
            return None

    def _vae_pin_tensor_list(self, parts: set[str] | None):
        """移動対象の param/buffer を列挙する。`parts` None = vae/audio_vae 全体。
        parts 指定時は `_move_vae_part` と同じ「未知の子モジュールがあれば None」
        (呼び出し側が全体移動へ退避する) の保守則に従う。"""
        out = []
        for attr, table in (("vae", self._VAE_PARTS["vae"]), ("audio_vae", self._VAE_PARTS["audio_vae"])):
            module = getattr(self._pipe, attr)
            if parts is None:
                out.extend(module.parameters())
                out.extend(module.buffers())
                continue
            known = set(table["encode"]) | set(table["decode"])
            if any(name not in known for name, _ in module.named_children()) or \
                    any(True for _ in module.named_parameters(recurse=False)) or \
                    any(True for _ in module.named_buffers(recurse=False)):
                return None
            for part in sorted(parts):
                for name in table[part]:
                    child = getattr(module, name)
                    out.extend(child.parameters())
                    out.extend(child.buffers())
        return out

    def _vae_pinned_move(self, tensors, device) -> None:
        """pinned マスター方式の移動。->CPU は data の付け替えのみ (D2H 無し)、
        ->GPU は pinned からの non_blocking H2D (同一ストリーム順序なので後続カーネルと
        の整合は保たれるが、所要時間ログの正確さのため最後に同期する)。"""
        pin = self._vae_pin_map
        to_cpu = torch.device(device).type == "cpu"
        for t in tensors:
            master = pin.get(t)
            if to_cpu:
                t.data = master if master is not None else t.data.to(CPU)
            else:
                src = master if master is not None else t.data
                t.data = src.to(device, non_blocking=master is not None)
        if not to_cpu:
            torch.cuda.synchronize()

    def _vae_to_gpu(self, part: str = "all"):
        """bnb-4bit mode only: move the (small, fp32, ~11GB) VAEs onto GPU for their active
        phase. A single short one-way trip, not a standing swap -- see module docstring.

        `part` ("encode"/"decode") は `H3_VAE_SPLIT=1` かつ KEEP_REF2VA のときだけ効き、その
        窓で使う子モジュールだけを GPU へ送る (参照エンコード窓ではデコーダ 4.5GiB を、
        デコード窓ではエンコーダを置き去りにする)。それ以外は従来どおり全体を移す。
        配置換えだけなので出力はビット一致。
        """
        if TE_QUANT != "bnb-4bit":
            return
        split = part in ("encode", "decode") and self._vae_split_active()
        want = {part} if split else {"encode", "decode"}
        need = want - self._vae_gpu_parts
        if not need:
            return
        t0 = time.time()
        moved = False
        if H3_VAE_PINNED and self._vae_pin_masters() is not None:
            tensors = self._vae_pin_tensor_list(need if split else None)
            if tensors is None and split:
                tensors = self._vae_pin_tensor_list(None)  # 未知の子モジュール: 全体移動へ退避
                need = {"encode", "decode"}
            if tensors is not None:
                self._vae_pinned_move(tensors, DEVICE)
                self._vae_gpu_parts |= need if split else {"encode", "decode"}
                moved = True
        if not moved:
            if split and all(self._move_vae_part(p, DEVICE) for p in sorted(need)):
                self._vae_gpu_parts |= need
            else:
                # 従来経路 (または未知の子モジュールがあるときの退避): 全体を移す。
                self._pipe.vae.to(DEVICE)
                self._pipe.audio_vae.to(DEVICE)
                self._vae_gpu_parts = {"encode", "decode"}
        self._vae_on_gpu = True
        logger.info("vae/audio_vae -> GPU (%s) in %.2fs. gpu=%s",
                    "+".join(sorted(self._vae_gpu_parts)), time.time() - t0, gpu_mem_gb())

    def _vae_to_cpu(self):
        """bnb-4bit mode only: move the VAEs back off GPU once their phase is done, to make
        room for the permanently-resident transformer + TE-nf4 during denoise.
        """
        if TE_QUANT != "bnb-4bit" or not self._vae_on_gpu:
            return
        t0 = time.time()
        if H3_VAE_PINNED and self._vae_pin_masters() is not None:
            self._vae_pinned_move(self._vae_pin_tensor_list(None), CPU)
        else:
            self._pipe.vae.to(CPU)
            self._pipe.audio_vae.to(CPU)
        self._vae_on_gpu = False
        self._vae_gpu_parts = set()
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

        `H3_LOWVRAM_GROUP` ("group" mode): delegates entirely to `_ensure_transformer_group`
        (see its docstring for the CPU-resident + block-level-group-offload design) --
        this is a different enough loading shape (device_map="cpu", not "cuda", and the
        module is never actually freed between requests) that it is not worth threading
        through the branches below.
        """
        self._ensure_pipe_shell()
        if H3_LOWVRAM_GROUP:
            self._ensure_transformer_group(progress)
            return
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
        if "base" in self._pin_slots:
            # H3_BASE_PINNED: 退避中のモジュールを pinned H2D で復帰させる。構造
            # (turbo LoRA ラップ・attention backend) はモジュールに保持済みのため、
            # 下の新規ロード経路の再適用は一切通らない。
            module, wrapped = self._pinned_restore("base")
            self._pipe.transformer = module
            self._transformer_loaded = True
            self._turbo_lora_wrapped = wrapped
            self._active_variant = "t2va"
            return
        t0 = time.time()
        loaded_from_prequant = False
        if H3_TRANSFORMER_QUANT == "int8":
            # 量子化済みキャッシュがあればそこから読む (H3_TRANSFORMER_PREQUANT の
            # module docstring 参照)。読めた場合は bf16ロード+量子化を丸ごと省略する。
            if H3_TRANSFORMER_PREQUANT and self._load_transformer_from_prequant(
                self._transformer_prequant_dir(is_ref=False), is_ref=False, progress=progress
            ):
                loaded_from_prequant = True
            else:
                from diffusers import TorchAoConfig
                from torchao.quantization import Int8WeightOnlyConfig

                quant_config = TorchAoConfig(
                    Int8WeightOnlyConfig(version=2, set_inductor_config=False),
                    modules_to_not_convert=H3_INT8_MODULES_TO_NOT_CONVERT,
                )
                self._pipe.load_components(
                    names=["transformer"],
                    dtype=torch.bfloat16,
                    quantization_config={"transformer": quant_config},
                    device_map={"transformer": "cuda"},
                )
        elif H3_TRANSFORMER_QUANT == "ck-w4a8":
            # base = Kijai の AdaLN-pruned W4A8(fl2va_pruned_w4a8_mixed)を
            # scripts/convert_kijai_w4a8.py で変換したキャッシュからロードする。
            # アーキは pruned ref と同一(remote code クラス)なので pruned の
            # ロード機構をそのまま使う。fresh 量子化経路は存在しない(ソースが
            # Kijai の量子化済み配布物のため。フル 33B の直 W4 化は adaln_proj の
            # K=2688 が回転カーネル 256 固定に非対応で不成立 — commit c56ed36 参照)。
            from core import pruned as pruned_mod

            cache_dir = H3_TRANSFORMER_PREQUANT_DIR / "transformer_base_pruned_ck_w4a8"
            if not (cache_dir / "pruned_ck_w4a8_state.pt").exists():
                raise RuntimeError(
                    f"ck-w4a8 base のキャッシュがありません: {cache_dir}。"
                    "scripts/convert_kijai_w4a8.py で "
                    "Kijai/MiniMax-H3-experimental の minimax_h3_fl2va_pruned_w4a8_mixed."
                    "safetensors を変換してください。"
                )
            snapshot = pruned_mod.pruned_snapshot_dir(H3_PRUNED_REPO, weights=False)
            self._pipe.transformer = pruned_mod.load_pruned_ref_from_cache(
                cache_dir, snapshot, DEVICE, H3_PRUNED_CONVROT_GROUP, quant="ck-w4a8"
            )
            loaded_from_prequant = True
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
        # 初回のみ (量子化を実際にその場で行ったときだけ): 量子化済みの重みを保存して
        # おき、次回以降のロードを短縮する。turbo LoRA の構造的 wrap や attention
        # backend/FBC/AdaLN precompute の設定 (いずれも下記) より**前**、量子化直後の
        # まっさらな状態で保存する (H3_TRANSFORMER_PREQUANT の module docstring 参照)。
        if H3_TRANSFORMER_QUANT == "int8" and H3_TRANSFORMER_PREQUANT and not loaded_from_prequant:
            self._save_transformer_prequant(
                self._transformer_prequant_dir(is_ref=False), self._pipe.transformer, is_ref=False
            )

        self._transformer_loaded = True
        self._active_variant = "t2va"
        if H3_TURBO_LORA:
            # Applied right after load, before `set_attention_backend`/FBC: LoRA wrapping
            # only replaces Linear *modules* (`attn.to_qkv`, `ff.net.0.proj`, etc, see
            # `apply_turbo_lora`'s docstring), it does not touch attention dispatch or
            # cache hooks, so ordering against those two calls does not matter -- done
            # first here only because it is the more fundamental structural change of the
            # three. `H3_CACHE == "fbc"` is force-skipped below (not just "left at its
            # default") regardless of the env var's own value -- see `H3_TURBO_LORA`'s
            # module comment for why a handful of turbo steps leaves FBC no safe window.
            n = self._apply_turbo_lora_checkpoint(self._pipe.transformer, is_ref=False)
            self._turbo_lora_wrapped = True
            logger.info("H3_TURBO_LORA=1: applied turbo LoRA (%d layers wrapped), FBC force-disabled", n)
        if H3_ATTN_BACKEND:
            self._pipe.transformer.set_attention_backend(H3_ATTN_BACKEND)
            logger.info("transformer attention backend set to %r", H3_ATTN_BACKEND)
        if H3_CACHE == "fbc" and not H3_TURBO_LORA:
            self._enable_fbc()
        if H3_ADALN_PRECOMP:
            # Arms this fresh transformer instance for AdaLN precompute -- the actual
            # table build happens lazily, on the first denoise step of whichever request
            # drives this instance next (see core/adaln_precompute.py's
            # `enable_adaln_precompute()` docstring). Must re-arm on every fresh load,
            # not just once at process start: this project's default H3_TE_QUANT=
            # bnb-4bit steady state fully frees and reloads `transformer` around every
            # request's decode window (`_free_transformer()`/`_restore_decode_steady_
            # state()` below), which drops `_h3opt_adaln_cursor` along with the rest of
            # the module -- the freshly-reloaded instance has no precompute state yet.
            # Ordering against FBC (`_enable_fbc()`, just above) and turbo (already
            # rejected at import time when this flag is on, see H3_ADALN_PRECOMP's own
            # guard block) does not matter here: FBC wraps `block.forward` itself via a
            # HookRegistry hook, precompute only ever replaces the `block.adaln_proj`
            # submodule attribute that `block.forward` looks up dynamically on every
            # call -- the two never touch the same callable.
            from core.adaln_precompute import enable_adaln_precompute

            enable_adaln_precompute(self._pipe.transformer)
        logger.info(
            "transformer loaded to GPU in %.1fs (quant=%s, turbo_lora=%s, adaln_precomp=%s). gpu=%s ram=%s",
            time.time() - t0, H3_TRANSFORMER_QUANT, H3_TURBO_LORA, H3_ADALN_PRECOMP, gpu_mem_gb(), ram_gb(),
        )
        # フェーズ境界での中断チェック(loading_transformer)。
        interrupt_controller.check()

    def _apply_turbo_lora_checkpoint(self, transformer, is_ref: bool = False) -> int:
        """設定済みの turbo LoRA をダウンロード → キー形式を判定 → 形式に応じた適用
        関数へディスパッチする (`_ensure_transformer` の起動時適用と
        `_apply_turbo_setting` の遅延適用、両呼び出し元の共通化)。

        comfy 形式 (Ostris 版) × int8 はここで明確に拒否する -- import 時のガードは
        リポジトリ名の予備判定 (`_TURBO_COMFY_REPOS`) しかできないため、未知リポジトリの
        comfy 形式チェックポイントはこの実ファイル判定が最後の砦。"""
        lora_path = self._download_turbo_lora_if_needed(is_ref=is_ref)
        lora_format = detect_turbo_lora_format(lora_path)
        if lora_format == "comfy":
            if H3_TRANSFORMER_QUANT == "int8":
                raise ValueError(
                    "comfy形式 (融合QKV) の turbo LoRA は int8 transformer に適用できません "
                    "(fuse_projections() の torch.cat が Int8Tensor 非対応)。既定の "
                    "diffusers ネイティブ形式 (lightx2v/Minimax-h3-Turbo) を使うか、"
                    "int8/低VRAM を無効にしてください。"
                )
            return apply_turbo_lora(transformer, lora_path)
        return apply_diffusers_turbo_lora(
            transformer, lora_path,
            resolve_turbo_lora_scale(lora_format, lora_path),
        )

    def _download_turbo_lora_if_needed(self, is_ref: bool = True) -> str:
        """Resolve (downloading if necessary, via the normal HF cache) the turbo LoRA
        safetensors path once per process, caching it on `self._turbo_lora_path`. Split
        out from `_ensure_transformer` only so the download (network I/O, ~780MB) is not
        interleaved with that method's own docstring-documented load-order reasoning.
        """
        # is_ref=False (base transformer) は H3_TURBO_LORA_{REPO,FILE}_BASE を使う
        # (未指定なら共通値 = 従来挙動)。is_ref=True (既定) は従来の共通 REPO/FILE。
        # 結果は is_ref ごとに別キャッシュする。BASE が共通と同一なら同じパスを共有する。
        if is_ref:
            repo, fname, attr = H3_TURBO_LORA_REPO, H3_TURBO_LORA_FILE, "_turbo_lora_path"
        else:
            repo, fname, attr = H3_TURBO_LORA_REPO_BASE, H3_TURBO_LORA_FILE_BASE, "_turbo_lora_path_base"
            if (repo, fname) == (H3_TURBO_LORA_REPO, H3_TURBO_LORA_FILE):
                attr = "_turbo_lora_path"
        cached = getattr(self, attr, None)
        if cached is not None:
            return cached
        # H3_TE_PROJ と同じ流儀: H3_TURBO_LORA_FILE が実在する絶対パスならローカル
        # ファイルとして直接使う (HF に無い自作/フィルタ済み LoRA の A/B 用。
        # 2026-09-10、FastH3 dense LoRA の adaln 除外版検証で追加)。
        if os.path.isabs(fname) and os.path.isfile(fname):
            setattr(self, attr, fname)
            logger.info("turbo LoRA checkpoint resolved (%s): %s (local file)", "ref" if is_ref else "base", fname)
            return fname
        from huggingface_hub import hf_hub_download

        t0 = time.time()
        path = hf_hub_download(repo, fname)
        setattr(self, attr, path)
        logger.info(
            "turbo LoRA checkpoint resolved (%s): %s (%.1fs, repo=%s file=%s)",
            "ref" if is_ref else "base", path, time.time() - t0, repo, fname,
        )
        return path

    def _ensure_hyperflow_ref(self, progress: ProgressState | None = None) -> None:
        """H3_HYPERFLOW 時、transformer_ref に HyperFlow LoRA + TwoTimeEmbedder を適用する。

        冪等: 適用済みかは transformer_ref インスタンスの time_embedder が
        TwoTimeEmbedder かどうかで判定する(bool フラグではなくインスタンス基準 --
        lowvram のパーティション切替等で transformer_ref が作り直された場合にも
        自動で再適用される)。呼び出し点は apply_instant_settings(is_ref=True) の直前
        (ref2va 本体と ref バッチの両経路が通る単一チョークポイント。denoise より
        前であれば set_timesteps との順序は問わない -- モジュールコメント参照)。
        """
        if not H3_HYPERFLOW:
            return
        tr = getattr(self._pipe_ref, "transformer_ref", None)
        if tr is None:
            raise RuntimeError("H3_HYPERFLOW: transformer_ref が未ロードのまま適用点に到達しました")
        from hyperflow_h3.embedder import TwoTimeEmbedder

        if isinstance(tr.time_embedder, TwoTimeEmbedder):
            return
        if progress:
            progress.update(phase="loading_transformer", message="HyperFlow LoRA を適用中...")
        import types as _types

        from hyperflow_h3 import load_hyperflow_lora

        t0 = time.time()
        load_hyperflow_lora(_types.SimpleNamespace(transformer_ref=tr), H3_HYPERFLOW_LORA)
        logger.info(
            "HyperFlow LoRA applied to transformer_ref in %.1fs (source=%s). gpu=%s",
            time.time() - t0, H3_HYPERFLOW_LORA, gpu_mem_gb(),
        )

    def _check_group_offload_ram_guard(self):
        """Refuse to start a group-offload transformer load if host RAM is already tight.

        See `H3_GROUP_OFFLOAD_MIN_RAM_GB`'s module-level comment: the whole point of this
        check is to fail loudly with a clear error *before* attempting a ~34GB CPU load,
        rather than risk the swap-storm/OOM-killer failure mode CLAUDE.md #33 (diffusers-
        server, this project's sibling repo) documents from a similarly-shaped mistake in
        a different project. Cheap (`/proc/meminfo` read only), safe to call defensively.
        """
        avail = ram_gb()["avail_gb"]
        if avail < H3_GROUP_OFFLOAD_MIN_RAM_GB:
            raise RuntimeError(
                f"H3_LOWVRAM=group requires at least {H3_GROUP_OFFLOAD_MIN_RAM_GB}GB of "
                f"available host RAM before loading the (~34GB, permanently CPU-resident) "
                f"int8 transformer, but only {avail}GB is available right now. Refusing to "
                "start the load rather than risk a swap storm (see CLAUDE.md #33 in the "
                "sibling diffusers-server repo for the incident this guards against). Free "
                "up host RAM, or lower H3_GROUP_OFFLOAD_MIN_RAM_GB if you have verified "
                "this box's actual headroom."
            )

    def _ensure_transformer_group(self, progress: ProgressState | None = None):
        """`H3_LOWVRAM_GROUP` ("group" mode) transformer loading: CPU-resident int8 +
        diffusers block-level group offload, for 24-32GB-class cards.

        Unlike every other mode in this file (including `H3_LOWVRAM=1`, which frees the
        transformer completely between requests), this mode's transformer is loaded
        *once* and stays resident -- in host RAM, not VRAM -- for the life of the
        process, exactly like `bnb-4bit` TE's own "load once, keep forever" shape (see
        `_load_text_encoder`'s docstring: bnb 4bit modules cannot be `.to()`-moved
        between devices either, so reload-from-scratch is the only alternative there;
        here it is simply unnecessary, since a group-offloaded module's GPU visits are
        already small and self-managed by diffusers' own hooks).

        Design, verified against this file's actual constraints by
        `scripts/probe_group_offload.py` before being wired in here (see that script's
        own docstring and this task's write-up for the full reasoning):

        1. `device_map={"transformer": "cpu"}` + `TorchAoConfig(Int8WeightOnlyConfig)`.
           A plain string device_map value becomes `{"": torch.device("cpu")}` inside
           `from_pretrained` (see `modeling_utils.py`'s device_map normalization) -- a
           single-entry dict whose *value* is a `torch.device` object, not the string
           `"cpu"`. `TorchAoHfQuantizer.validate_environment` only sets its
           offload-skip-quantize flag (`self.offload = True`, which makes
           `check_if_quantized_param` skip quantizing anything placed on `"cpu"`) when
           `"cpu" in device_map.values()` -- a `torch.device("cpu") == "cpu"` comparison
           is `False` in Python, so that flag is never set here and every eligible
           linear layer DOES get quantized to torchao's `Int8Tensor`, even though every
           weight lands on CPU. Confirmed empirically: probe scan found 370/370 eligible
           linear layers as `Int8Tensor` (none fell back to plain bf16 `Tensor`), and RAM
           dropped by ~32GB during the load (consistent with the already-measured ~34GB
           int8 size measured on GPU in `H3_TRANSFORMER_QUANT=int8` mode).
        2. `enable_group_offload(onload_device=cuda, offload_device=cpu,
           offload_type="block_level", num_blocks_per_group=1, use_stream=True,
           low_cpu_mem_usage=H3_GROUP_OFFLOAD_LOW_CPU_MEM)` -- the CPU-then-offload
           ordering (module loaded CPU-side *first*, group offloading layered on top
           after) is the same shape diffusers-server's sibling project (CLAUDE.md
           #33/#34/#37) already established as correct: the hooks are in place before
           anything ever tries to move the whole ~34GB module onto GPU at once (which is
           the failure mode "block-level group offload" exists to avoid in the first
           place). 50 transformer_blocks at ~0.68GB (int8) each means
           `num_blocks_per_group=1` with `use_stream=True`'s double-buffered prefetch
           keeps only ~1-2 blocks (~1.4GB) GPU-resident at any moment during denoise, not
           the full 34GB. `low_cpu_mem_usage` defaults to `False` here -- see
           `H3_GROUP_OFFLOAD_LOW_CPU_MEM`'s own module-level comment for why: diffusers'
           own default (`True`) combined with `use_stream=True` hits a real bug for
           torchao's `Int8Tensor` (`RuntimeError: cannot pin 'torch.cuda.CharTensor'
           only dense CPU tensors can be pinned`, reproduced against both a minimal
           dummy int8 stack and the real transformer, isolated to exactly this
           combination), and the fix (`low_cpu_mem_usage=False`, which eagerly pins
           `cpu_param_dict` once at this call instead of once per onload) also measured
           ~4-5x faster per-block onload as a side benefit.
        3. FBC (`H3_CACHE=fbc`) and the attention backend (`H3_ATTN_BACKEND=sage`) are
           applied exactly as in every other mode: both are independent hook layers
           (FBC decides whether to skip a block's compute at all; group offloading
           decides whether that block's weights are already GPU-resident; the attention
           backend only changes what happens inside a block's own attention call once it
           does run) -- diffusers' `HookRegistry` supports multiple hooks per module by
           design (see `hooks/hooks.py`), and this is exactly what the task's own
           empirical verification (this run) is checking end-to-end, not just asserting.

        No RAM-vs-VRAM cycling for the transformer itself is needed once this call
        returns -- `_free_transformer`/decode-window drops elsewhere in this file are
        skipped for `H3_LOWVRAM_GROUP` (see the `generate()`/`generate_ref2va()` call
        sites), since group offloading already keeps VRAM usage low without a full
        drop+reload.
        """
        if self._transformer_loaded:
            self._active_variant = "t2va"
            return
        self._check_group_offload_ram_guard()
        if progress:
            progress.update(phase="loading_transformer", message="transformer (group offload) をロード中...")
        t0 = time.time()
        from diffusers import TorchAoConfig
        from torchao.quantization import Int8WeightOnlyConfig

        quant_config = TorchAoConfig(
            Int8WeightOnlyConfig(version=2, set_inductor_config=False),
            modules_to_not_convert=H3_INT8_MODULES_TO_NOT_CONVERT,
        )
        self._pipe.load_components(
            names=["transformer"],
            dtype=torch.bfloat16,
            quantization_config={"transformer": quant_config},
            device_map={"transformer": "cpu"},
        )
        if getattr(self._pipe, "transformer", None) is None:
            raise RuntimeError(
                "transformer load failed (see the diffusers 'Failed to create component "
                "transformer' warning above for the underlying error) -- "
                "self._pipe.transformer is still None after load_components()."
            )
        t1 = time.time()
        logger.info(
            "transformer loaded to CPU (int8, group-offload target) in %.1fs. ram=%s",
            t1 - t0, ram_gb(),
        )
        self._pipe.transformer.enable_group_offload(
            onload_device=DEVICE,
            offload_device=CPU,
            offload_type="block_level",
            num_blocks_per_group=H3_GROUP_OFFLOAD_BLOCKS,
            non_blocking=True,
            use_stream=H3_GROUP_OFFLOAD_USE_STREAM,
            record_stream=False,
            low_cpu_mem_usage=H3_GROUP_OFFLOAD_LOW_CPU_MEM,
        )
        self._transformer_loaded = True
        self._active_variant = "t2va"
        if H3_ATTN_BACKEND:
            self._pipe.transformer.set_attention_backend(H3_ATTN_BACKEND)
            logger.info("transformer attention backend set to %r", H3_ATTN_BACKEND)
        if H3_CACHE == "fbc":
            self._enable_fbc()
        logger.info(
            "transformer group offload enabled in %.1fs (total load %.1fs). gpu=%s ram=%s",
            time.time() - t1, time.time() - t0, gpu_mem_gb(), ram_gb(),
        )
        # フェーズ境界での中断チェック(loading_transformer)。
        interrupt_controller.check()

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

    def _pinned_park(self, name: str, module, wrapped: bool) -> None:
        """H3_BASE_PINNED / H3_REF_PINNED: transformer をモジュールごと保持したまま、
        GPU 上の重み実体だけを pinned CPU マスターへ付け替えて VRAM を解放する。
        初回のみマスター構築 (D2H コピー)、2回目以降は `p.data = master` のポインタ
        付け替えのみ (コピー無し・一時二重化無し)。復帰は `_pinned_restore`。
        turbo LoRA の lora_a/b は buffer 登録 (persistent=False) のため
        `module.buffers()` で漏れなく対象になる。`wrapped` = park 時点で turbo LoRA
        が構造適用済みか (restore 時にフラグへ戻す)。"""
        slot = self._pin_slots.get(name)
        if slot is None:
            t0 = time.time()
            seen: set[int] = set()
            pairs: list = []
            for t in itertools.chain(module.parameters(), module.buffers()):
                if id(t) in seen or t.device.type != "cuda":
                    continue
                seen.add(id(t))
                master = t.data.detach().to(CPU, copy=True).pin_memory()
                pairs.append((t, master))
                t.data = master
            self._pin_slots[name] = {"module": module, "pairs": pairs, "wrapped": wrapped}
            torch.cuda.empty_cache()
            gib = sum(m.numel() * m.element_size() for _, m in pairs) / 2**30
            logger.info(
                "H3_%s_PINNED: pinned masters built + transformer parked "
                "(%.2fGiB, %d tensors, %.2fs). gpu=%s ram=%s",
                name.upper(), gib, len(pairs), time.time() - t0, gpu_mem_gb(), ram_gb(),
            )
        else:
            t0 = time.time()
            for t, master in slot["pairs"]:
                t.data = master
            slot["wrapped"] = wrapped
            torch.cuda.empty_cache()
            logger.info(
                "H3_%s_PINNED: transformer parked (pointer swap, %.3fs). gpu=%s",
                name.upper(), time.time() - t0, gpu_mem_gb(),
            )

    def _pinned_restore(self, name: str):
        """pinned マスターから GPU へ復帰させ、(module, wrapped) を返す。
        構造 (turbo LoRA ラップ・attention backend 設定) はモジュールに保持済み。"""
        slot = self._pin_slots[name]
        t0 = time.time()
        for t, master in slot["pairs"]:
            t.data = master.to(DEVICE, non_blocking=True)
        torch.cuda.synchronize()
        logger.info(
            "H3_%s_PINNED: transformer restored from pinned masters in %.2fs. gpu=%s",
            name.upper(), time.time() - t0, gpu_mem_gb(),
        )
        return slot["module"], slot["wrapped"]

    def _free_transformer(self):
        if not self._transformer_loaded:
            return
        if H3_BASE_PINNED and "base" not in self._pin_disabled and not H3_LOWVRAM_GROUP:
            try:
                self._pinned_park("base", self._pipe.transformer, self._turbo_lora_wrapped)
                self._pipe.transformer = None
                self._transformer_loaded = False
                self._turbo_lora_wrapped = False
                return
            except Exception:
                logger.exception(
                    "H3_BASE_PINNED: pinned 退避に失敗、従来の解放経路へ退避 (以後無効化)")
                self._pin_disabled.add("base")
                self._pin_slots.pop("base", None)
        # Drop in place, no CPU staging (same reasoning as _free_text_encoder). In
        # H3_LOWVRAM_GROUP mode the module's parameters mostly live on CPU already (only
        # ~1-2 group-offloaded blocks are ever GPU-resident at a time), so this call
        # mainly reclaims ~34GB of *host RAM*, not VRAM -- still exactly the same "drop
        # in place, no staging trip" shape, just freeing the other kind of memory this
        # mode keeps the model resident in. Used when switching t2va<->ref2va under
        # H3_LOWVRAM_GROUP (see `_free_other_variant_transformer`): unlike every other
        # mode's transformer/transformer_ref pair, group mode never keeps both resident
        # at once (not verified to fit two ~34GB CPU-resident copies alongside TE-nf4
        # reload headroom within this mode's RAM guard -- see H3_GROUP_OFFLOAD_MIN_RAM_GB).
        del self._pipe.transformer
        self._pipe.transformer = None
        self._transformer_loaded = False
        self._turbo_lora_wrapped = False
        gc.collect()
        torch.cuda.empty_cache()
        if H3_LOWVRAM_GROUP:
            # group offload (use_stream=True, low_cpu_mem_usage=False) pins the whole
            # ~34GB CPU weight copy. del+gc returns those pages to torch's *host*
            # caching allocator, and `torch.cuda.empty_cache()` (device-side only)
            # never releases them to the OS -- so MemAvailable stays ~34GB short and
            # the RAM guard in `_ensure_transformer_ref_group` refuses the
            # t2va->ref2va switch. Found in the 2026-08-11 8GB×2 ref2va verification:
            # after "transformer freed" avail stayed ~38GB (RssShmem still held the
            # full pinned copy) instead of recovering to ~72GB. `_host_emptyCache`
            # releases the *unused* cached pinned blocks back to the OS; the next
            # group-offload load simply re-pins (a re-registration cost, not a
            # correctness issue). Private API (verified on this venv's torch 2.9) --
            # guarded so a future torch that drops it degrades to the old behaviour
            # (RAM stays cached, mode switch may hit the RAM guard) instead of dying.
            _host_empty_cache = getattr(torch._C, "_host_emptyCache", None)
            if _host_empty_cache is not None:
                _host_empty_cache()
            else:
                logger.warning("torch._C._host_emptyCache missing -- pinned host cache not released")
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

        `H3_LOWVRAM_GROUP`: delegates to `_ensure_transformer_ref_group` (CPU-resident
        int8 + block-level group offload, mirroring `_ensure_transformer_group` exactly
        but against `transformer_ref`/`self._pipe_ref`).
        """
        self._ensure_pipe_ref_shell()
        if H3_LOWVRAM_GROUP:
            self._ensure_transformer_ref_group(progress)
            return
        if self._transformer_ref_loaded:
            # See `_ensure_transformer`'s matching comment: must update
            # `_active_variant` even on the cached-return path, for int8 both-resident
            # mode's `_switch_to_variant` early-return check.
            self._active_variant = "ref2va"
            return
        if "ref" in self._pin_slots:
            # H3_REF_PINNED: 退避中のモジュールを pinned H2D で復帰 (base 側と同型)。
            # turbo LoRA ラップ・attention backend・FBC はモジュールに保持済み。
            module, wrapped = self._pinned_restore("ref")
            self._pipe_ref.transformer_ref = module
            self._transformer_ref_loaded = True
            self._turbo_lora_wrapped_ref = wrapped
            self._active_variant = "ref2va"
            interrupt_controller.check()
            return
        if progress:
            progress.update(phase="loading_transformer", message="transformer_ref (ref2va) をロード中...")
        t0 = time.time()
        loaded_from_prequant = False
        if H3_PRUNED:
            # AdaLN-pruned 経路(core/pruned.py)。方式は H3_PRUNED_QUANT(既定 int8wo)。キャッシュ優先、無ければ bf16(CPU)-> 量子化 ->
            # GPU 常駐で作って保存する(bf16 はキャッシュ対象外)。
            spec = H3_PRUNED_QUANT_SPEC
            logger.info(
                "pruned transformer_ref: H3_PRUNED_QUANT=%s -> %s",
                spec.name, spec.describe(H3_PRUNED_CONVROT_GROUP),
            )
            if spec.cacheable and H3_TRANSFORMER_PREQUANT and self._load_pruned_ref_from_prequant(
                self._transformer_prequant_dir(is_ref=True), progress=progress
            ):
                loaded_from_prequant = True
            else:
                from core import pruned as pruned_mod

                avail_ram = ram_gb()["avail_gb"]
                if avail_ram < 45.0:
                    raise RuntimeError(
                        f"H3_PRUNED の初回ロードには bf16 重み ~40GB を CPU に置く必要が"
                        f"ありますが、空きホストRAMが {avail_ram:.1f}GB しかありません "
                        "(量子化方式ならキャッシュ作成後は不要になります)。"
                    )
                snapshot = pruned_mod.pruned_snapshot_dir(H3_PRUNED_REPO)
                self._pipe_ref.transformer_ref = pruned_mod.load_pruned_ref_fresh(
                    snapshot, DEVICE, H3_PRUNED_CONVROT_GROUP, quant=spec.name
                )
        elif H3_TRANSFORMER_QUANT == "int8":
            # 量子化済みキャッシュがあればそこから読む (`_ensure_transformer` と同じ、
            # H3_TRANSFORMER_PREQUANT の module docstring 参照)。
            if H3_TRANSFORMER_PREQUANT and self._load_transformer_from_prequant(
                self._transformer_prequant_dir(is_ref=True), is_ref=True, progress=progress
            ):
                loaded_from_prequant = True
            else:
                from diffusers import TorchAoConfig
                from torchao.quantization import Int8WeightOnlyConfig

                quant_config = TorchAoConfig(
                    Int8WeightOnlyConfig(version=2, set_inductor_config=False),
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
        # See `_ensure_transformer`'s matching save call: only on a fresh in-place
        # quantize, and before turbo LoRA/attn backend/FBC/AdaLN precompute setup below.
        # pruned は方式がキャッシュ対象のときだけ(bf16 はスナップショットがそのまま
        # キャッシュなので保存しない)。
        if H3_PRUNED:
            cache_this_ref = H3_PRUNED_QUANT_SPEC.cacheable
        else:
            cache_this_ref = H3_TRANSFORMER_QUANT == "int8"
        if cache_this_ref and H3_TRANSFORMER_PREQUANT and not loaded_from_prequant:
            self._save_transformer_prequant(
                self._transformer_prequant_dir(is_ref=True), self._pipe_ref.transformer_ref, is_ref=True
            )
        self._transformer_ref_loaded = True
        self._active_variant = "ref2va"
        if H3_PRUNED_COMPILE:
            # キャッシュ保存より後・turbo LoRA wrap(遅延、初回 turbo リクエスト)より前。
            import sys as _sys

            from core import pruned as pruned_mod

            pruned_mod.compile_convrot_layers(
                self._pipe_ref.transformer_ref, _sys.modules["modeling_minimax_h3_pruned"]
            )
        if H3_ATTN_BACKEND:
            self._pipe_ref.transformer_ref.set_attention_backend(H3_ATTN_BACKEND)
            logger.info("transformer_ref attention backend set to %r", H3_ATTN_BACKEND)
        if H3_CACHE == "fbc":
            self._enable_fbc_ref()
        if H3_ADALN_PRECOMP and H3_PRUNED:
            logger.info(
                "H3_ADALN_PRECOMP は H3_PRUNED=1 ではスキップします"
                "(pruned は AdaLN 構造自体が別物で、精計算の対象が存在しない)"
            )
        if H3_ADALN_PRECOMP and not H3_PRUNED:
            # Mirrors `_ensure_transformer`'s own arming call -- see that method's
            # comment for the full rationale (re-arm on every fresh load, ordering vs
            # FBC does not matter). `enable_adaln_precompute()`'s class-level monkeypatch
            # of `MiniMaxH3LoopDenoiser.__call__` covers `MiniMaxH3Ref2VALoopDenoiser`
            # too (same bound method by inheritance, neither subclass overrides
            # `__call__` -- verified against this project's pinned diffusers commit), so
            # calling `enable_adaln_precompute()` a second time here (already called once
            # from `_ensure_transformer`, or will be from a future call) is safe: the
            # class-patch half is a no-op on the second call (`_h3opt_patched` guard),
            # only the per-instance `_h3opt_adaln_wanted = True` arming actually happens
            # against this transformer_ref instance.
            from core.adaln_precompute import enable_adaln_precompute

            enable_adaln_precompute(self._pipe_ref.transformer_ref)
        logger.info(
            "transformer_ref loaded to GPU in %.1fs (quant=%s, adaln_precomp=%s). gpu=%s ram=%s",
            time.time() - t0,
            f"pruned/{H3_PRUNED_QUANT}" if H3_PRUNED else H3_TRANSFORMER_QUANT,
            H3_ADALN_PRECOMP, gpu_mem_gb(), ram_gb(),
        )
        # フェーズ境界での中断チェック(loading_transformer)。
        interrupt_controller.check()

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

    def _ensure_transformer_ref_group(self, progress: ProgressState | None = None):
        """`H3_LOWVRAM_GROUP` transformer_ref loading -- mirrors `_ensure_transformer_group`
        exactly (see its docstring for the full design/verification), just against
        `transformer_ref`/`self._pipe_ref`. Not called directly by outside code; reached
        through `_ensure_transformer_ref`'s own `H3_LOWVRAM_GROUP` branch.
        """
        if self._transformer_ref_loaded:
            self._active_variant = "ref2va"
            return
        self._check_group_offload_ram_guard()
        if progress:
            progress.update(phase="loading_transformer", message="transformer_ref (group offload) をロード中...")
        t0 = time.time()
        from diffusers import TorchAoConfig
        from torchao.quantization import Int8WeightOnlyConfig

        quant_config = TorchAoConfig(
            Int8WeightOnlyConfig(version=2, set_inductor_config=False),
            modules_to_not_convert=H3_INT8_MODULES_TO_NOT_CONVERT,
        )
        self._pipe_ref.load_components(
            names=["transformer_ref"],
            dtype=torch.bfloat16,
            quantization_config={"transformer_ref": quant_config},
            device_map={"transformer_ref": "cpu"},
        )
        if getattr(self._pipe_ref, "transformer_ref", None) is None:
            raise RuntimeError(
                "transformer_ref load failed (see the diffusers 'Failed to create "
                "component transformer_ref' warning above for the underlying error) -- "
                "self._pipe_ref.transformer_ref is still None after load_components()."
            )
        t1 = time.time()
        logger.info(
            "transformer_ref loaded to CPU (int8, group-offload target) in %.1fs. ram=%s",
            t1 - t0, ram_gb(),
        )
        self._pipe_ref.transformer_ref.enable_group_offload(
            onload_device=DEVICE,
            offload_device=CPU,
            offload_type="block_level",
            num_blocks_per_group=H3_GROUP_OFFLOAD_BLOCKS,
            non_blocking=True,
            use_stream=H3_GROUP_OFFLOAD_USE_STREAM,
            record_stream=False,
            low_cpu_mem_usage=H3_GROUP_OFFLOAD_LOW_CPU_MEM,
        )
        self._transformer_ref_loaded = True
        self._active_variant = "ref2va"
        if H3_ATTN_BACKEND:
            self._pipe_ref.transformer_ref.set_attention_backend(H3_ATTN_BACKEND)
            logger.info("transformer_ref attention backend set to %r", H3_ATTN_BACKEND)
        if H3_CACHE == "fbc":
            self._enable_fbc_ref()
        logger.info(
            "transformer_ref group offload enabled in %.1fs (total load %.1fs). gpu=%s ram=%s",
            time.time() - t1, time.time() - t0, gpu_mem_gb(), ram_gb(),
        )
        # フェーズ境界での中断チェック(loading_transformer)。
        interrupt_controller.check()

    def _free_transformer_ref(self):
        if not self._transformer_ref_loaded:
            return
        if H3_REF_PINNED and "ref" not in self._pin_disabled:
            try:
                self._pinned_park("ref", self._pipe_ref.transformer_ref, self._turbo_lora_wrapped_ref)
                self._pipe_ref.transformer_ref = None
                self._transformer_ref_loaded = False
                self._turbo_lora_wrapped_ref = False
                return
            except Exception:
                logger.exception(
                    "H3_REF_PINNED: pinned 退避に失敗、従来の解放経路へ退避 (以後無効化)")
                self._pin_disabled.add("ref")
                self._pin_slots.pop("ref", None)
        # Drop in place, no CPU staging -- same reasoning as _free_transformer /
        # _free_text_encoder (CLAUDE.md #33: no whole-module CPU-staging trips for
        # 60GB+ modules on this box). In H3_LOWVRAM_GROUP mode this reclaims host RAM
        # (see _free_transformer's matching comment), not VRAM.
        del self._pipe_ref.transformer_ref
        self._pipe_ref.transformer_ref = None
        self._transformer_ref_loaded = False
        self._turbo_lora_wrapped_ref = False
        gc.collect()
        torch.cuda.empty_cache()
        if H3_LOWVRAM_GROUP:
            # Same pinned-host-cache release as _free_transformer (see its comment):
            # without this the ref2va->t2va switch strands ~34GB in torch's host
            # caching allocator and the t2va side's own RAM guard hits the same wall.
            _host_empty_cache = getattr(torch._C, "_host_emptyCache", None)
            if _host_empty_cache is not None:
                _host_empty_cache()
            else:
                logger.warning("torch._C._host_emptyCache missing -- pinned host cache not released")
        logger.info("transformer_ref freed. gpu=%s ram=%s", gpu_mem_gb(), ram_gb())

    def _drain_deferred_decode(self, caller: str, timeout_s: float = 30.0) -> None:
        """base transformer のロード/解放を始める前に、実行中の deferred decode を待ち切る。

        この機では「cuda:0 への大量 H2D(base 34GB のシャードロード)と cuda:1 の
        decode(cuBLASLt)を同時に走らせる」と CUBLAS_STATUS_INTERNAL_ERROR ->
        illegal memory access でプロセスの CUDA コンテキストごと壊れる(2026-10-06
        実運用+ストレスで再現。P2P 全ゼロ・クロスデバイス empty_cache と並ぶ
        このマシン固有のクロスデバイス罠)。decode は ~4s で終わるので待つのが最小対処。
        生成ロックは呼び出し側が保持しており、待っている間に新しい decode が
        生まれることはない(新 decode は ref2va の denoise 完了からしか始まらない)。
        """
        if not (self._decode_overlap_configured() if hasattr(self, "_decode_overlap_configured")
                else (H3_DECODE_STREAM or H3_DECODE_DEVICE)):
            return
        if not self._decode_lock.locked():
            return
        t0 = time.time()
        logger.info("%s: deferred decode の完了を待ってから base をロードします", caller)
        while self._decode_lock.locked() and time.time() - t0 < timeout_s:
            time.sleep(0.05)
        logger.info("%s: deferred decode 完了 (%.2fs 待ち)", caller, time.time() - t0)

    def _keep_ref2va_coexist(self, caller: str) -> bool:
        """`H3_KEEP_REF2VA=1` で常駐している ref2va スタック (transformer_ref [+ TE]) を、base
        transformer を使うリクエスト (t2va/fl2va/t2i) の間も残してよいか (= 両常駐が成立するか)。

        True なら呼び出し側は `_free_transformer_ref()` と encode 後の `_free_text_encoder(force=True)`
        をスキップする。False なら従来どおり解放する。判定は実行時の空きVRAMで行う:
        空き >= `H3_KEEP_REF2VA_COEXIST_MIN_FREE_GB` (+ TE 未ロードなら 17.5GB)。
        `H3_VRAM_LIMIT_GB` が設定されていれば (上限 - このプロセスの reserved) との小さい方を空きとする。
        判定結果と根拠は INFO ログ 1 行。KEEP_REF2VA でない / transformer_ref が未ロード (守るものが
        無い) ときは無言で False (従来挙動)。呼び出しは `_load_lock` の中で行うこと。
        """
        if not (_keep_ref2va_active() and H3_KEEP_REF2VA_COEXIST):
            return False
        if not self._transformer_ref_loaded:
            return False
        if H3_TE_STREAM or H3_REF_PREFIX_PARK or H3_VAE_SPLIT:
            # 32GB/48GB 級向けの低VRAM構成 (TE を CPU から流す・prefix を退避・VAE を分割配置)。
            # 常駐の意味が通常構成と異なり、base 同居は未検証かつそもそも収支が合わないので常に解放。
            return False
        # ここで empty_cache() を呼んではいけない: H3_DECODE_STREAM/H3_DECODE_DEVICE の
        # 重ね実行中は前リクエストの decode が cuda:1 で走っており、empty_cache は全デバイス
        # 対象のため "illegal memory access" で CUDA コンテキストごと壊す (2026-10-06 実運用で
        # 発生。547f7e1 の park 側と同じ罠を本判定自身が踏んでいた)。代わりに、この
        # プロセスのアロケータが抱える「reserved だが未割当」のキャッシュ分を空きに足し込む
        # (empty_cache が返すのはこの分なので、測定精度は同等)。
        free_gb = (torch.cuda.mem_get_info(DEVICE)[0]
                   + torch.cuda.memory_reserved(DEVICE) - torch.cuda.memory_allocated(DEVICE)
                   ) / 1024**3
        if H3_VRAM_LIMIT_GB:
            free_gb = min(free_gb, float(H3_VRAM_LIMIT_GB) * 1e9 / 1024**3 - torch.cuda.memory_reserved(DEVICE) / 1024**3)
        te_resident = self._text_encoder_loaded or self._te_external
        need_gb = H3_KEEP_REF2VA_COEXIST_MIN_FREE_GB + (0.0 if te_resident else 17.5)
        ok = free_gb >= need_gb
        logger.info(
            "%s: H3_KEEP_REF2VA 両常駐判定 -> %s (空きVRAM %.1fGB %s 必要 %.1fGB = base 34.3 + 活性化 6.4 "
            "+ 余裕 2.0%s)。%s",
            caller, "ref2va スタック (transformer_ref + TE) を常駐のまま base を同居ロード" if ok else "解放して従来どおり",
            free_gb, ">=" if ok else "<", need_gb, "" if te_resident else " + TE再ロード 17.5",
            "" if ok else "VRAM不足 (低VRAM構成) のため transformer_ref を解放する",
        )
        return ok

    # ------------------------------------------------------------------
    # Instant-apply per-request settings (cache/attn/turbo) -- see core/settings.py's
    # module docstring for the full instant-vs-reload split. Every method here is meant
    # to run in well under a second (no reload, no big tensor copy) and is applied fresh
    # at the top of every generate()/generate_ref2va() call, right after that request's
    # transformer is confirmed GPU-resident and before denoise starts -- so a client
    # that never passes these fields sees byte-for-byte the same behaviour as before
    # they existed (whatever the process-wide H3_CACHE/H3_ATTN_BACKEND/H3_TURBO_LORA
    # env vars resolved to at transformer-load time), and a client that does pass them
    # gets an immediate per-request override with no server restart or model reload.
    # ------------------------------------------------------------------
    def _apply_cache_setting(self, transformer, cache: str, threshold: float, label: str = "transformer"):
        """Enable/disable FirstBlockCache on an already-loaded transformer, or just
        change its threshold, without any reload.

        `cache`: "fbc" or "none". `threshold`: only meaningful when cache == "fbc".
        Idempotent per (cache, threshold) pair -- re-applying the same combination is a
        cheap no-op (checked via `_cache_config` introspection) rather than an
        unnecessary disable+re-enable, which would also needlessly reset the stateful
        cache's leftover residual from mid-request (harmless here since this is only
        ever called between requests, before the loop, but kept minimal anyway).
        `enable_cache()` raises `ValueError` if a cache is already enabled (see
        `MiniMaxH3Transformer3DModel.enable_cache`'s source, inherited from
        `CacheMixin`) -- so an existing FBC hook must always be `disable_cache()`d
        first before installing a new one, even just to change the threshold.
        """
        if transformer is None:
            return
        from diffusers.hooks import FirstBlockCacheConfig

        currently_enabled = bool(getattr(transformer, "is_cache_enabled", False))
        current_threshold = None
        if currently_enabled:
            current_config = getattr(transformer, "_cache_config", None)
            current_threshold = getattr(current_config, "threshold", None)

        if cache == "fbc":
            if currently_enabled and current_threshold == threshold:
                return  # already exactly this configuration
            if currently_enabled:
                transformer.disable_cache()
            _register_minimax_h3_block_for_fbc()
            transformer.enable_cache(FirstBlockCacheConfig(threshold=threshold))
            logger.info("%s: FirstBlockCache enabled (threshold=%s)", label, threshold)
        else:
            if currently_enabled:
                transformer.disable_cache()
                logger.info("%s: FirstBlockCache disabled", label)

    def _apply_attn_setting(self, transformer, attn: str, label: str = "transformer"):
        """Set the attention backend on an already-loaded transformer, no reload.

        `attn`: "default" (diffusers' own native/SDPA dispatch -- `set_attention_backend`
        is simply never called, byte-for-byte the same as this project's own
        `H3_ATTN_BACKEND=""` behaviour) or any `AttentionBackendName` value (e.g. "sage",
        "native"). `set_attention_backend()` itself is a pure attribute-set over every
        attention submodule (see its source: no tensor copy, no reload) -- safe and cheap
        to call on every request even when the value has not changed.
        """
        if transformer is None:
            return
        if attn and attn != "default":
            transformer.set_attention_backend(attn)
            logger.info("%s: attention backend set to %r", label, attn)
        # attn == "default"/"" : leave whatever backend is already active alone. There is
        # no diffusers API to "unset" a backend back to native once another has been
        # selected other than explicitly requesting "native" -- but this project's own
        # process-wide default is native/SDPA (H3_ATTN_BACKEND=""'s behaviour, i.e.
        # set_attention_backend is simply never called at load time), so a request that
        # wants that same default explicitly should pass attn="native", not "default".
        # "default" here specifically means "do not touch whatever the server's current
        # backend is" -- distinguishing "no opinion" from "force native" matters once a
        # previous request in this same process instant-applied a non-default backend.

    def _apply_turbo_setting(self, transformer, turbo: bool, is_ref: bool = False, progress: ProgressState | None = None):
        """Enable/disable the turbo LoRA on an already-loaded transformer for this
        request only, no reload.

        First call with `turbo=True` against a given transformer instance structurally
        wraps its Linear modules via `apply_turbo_lora()` (downloading the ~780MB
        checkpoint once per process if not already cached, and paying the one-time
        `fuse_projections()` + wrap cost, a few seconds) -- every later call, on either
        transformer instance, just flips `_TurboLoRALinear.enabled` via
        `set_turbo_lora_enabled()`, which is instant. Wrapping state is tracked per
        transformer instance (`self._turbo_lora_wrapped` / `_wrapped_ref`) and reset to
        False whenever that transformer is freed/reloaded (see `_free_transformer(_ref)`)
        since a fresh module has no wrapping yet.
        """
        if transformer is None:
            return
        wrapped_attr = "_turbo_lora_wrapped_ref" if is_ref else "_turbo_lora_wrapped"
        label = "transformer_ref" if is_ref else "transformer"
        if turbo and not getattr(self, wrapped_attr):
            if progress:
                progress.update(message=f"turbo LoRA を {label} へ適用中...")
            n = self._apply_turbo_lora_checkpoint(transformer, is_ref=is_ref)
            setattr(self, wrapped_attr, True)
            if is_ref and H3_PRUNED_COMPILE_LEVEL >= 2:
                from core import pruned as pruned_mod

                pruned_mod.compile_turbo_wrappers(transformer, _TurboLoRALinear)
            if is_ref and H3_PRUNED_COMPILE_LEVEL >= 3:
                from core import pruned as pruned_mod

                pruned_mod.compile_transformer_blocks(transformer)
            logger.info("turbo LoRA lazily applied to %s (%d layers wrapped)", label, n)
        elif getattr(self, wrapped_attr):
            n = set_turbo_lora_enabled(transformer, turbo)
            logger.debug("%s: turbo LoRA %s (%d wrapper modules)", label, "enabled" if turbo else "disabled", n)
        # else: turbo requested False and it was never wrapped in the first place --
        # nothing to do, the transformer's Linears are still the plain unwrapped ones.

    def _apply_turbo_video_shift(self, turbo_effective: bool, is_ref: bool = True) -> None:
        """Switch the (process-wide, shared) video scheduler's shift for this request,
        based on the request's *resolved* turbo state (`instant["turbo"]`) -- turbo is a
        per-request instant-apply setting (like `cache`/`attn`), so the shift a
        turbo-distilled checkpoint needs has to follow it per-request too, not get
        fixed at process start the way `H3_VIDEO_SHIFT` alone would.

        No-op entirely when `H3_TURBO_VIDEO_SHIFT` (module-level, resolved from either
        the env var or the configured turbo LoRA file's name -- see its own module
        comment) is `None`: this happens for every LoRA file that is not an fl2v
        `_768p` variant (fl2v non-768p, and all ref2v files including the `_768p`-named
        ref2v 8step v1.0), which were all distilled at the same shift the base model
        already defaults to, so there is nothing to switch between turbo=1 and turbo=0.
        Also effectively unreachable when `H3_TURBO_LORA=0` (LoRA disabled for this
        process): callers only invoke this after `settings.resolve_instant_settings()`,
        and `turbo_effective` can only be True if `H3_TURBO_LORA=1` unless a request
        explicitly opts in via `turbo=True` -- but the request-level override is itself
        rejected before reaching a transformer that never got the turbo LoRA structurally
        wrapped (`_apply_turbo_setting` above only wraps lazily on `turbo=True`, and
        every call site of this helper runs on the same request whose turbo flag it
        checks). Must run before the request's `MiniMaxH3SetTimestepsStep` call(s) --
        the scheduler bakes `shift` into the sigma schedule at `set_timesteps()` time.

        `self._pipe.scheduler` and `self._pipe_ref.scheduler` are the same object
        (single ModularPipeline shell, see `_ensure_pipe_ref_shell`'s docstring), so one
        `set_shift()` call here covers both `generate()`/`generate_still_batch()` (t2va)
        and `generate_ref2va()`/`generate_ref_batch()` (ref2va) call sites. Audio is
        deliberately left untouched -- no turbo LoRA variant distilled at a different
        audio shift has been observed (see `H3_VIDEO_SHIFT`'s module comment: video
        12.0/audio 3.0 is the shared baseline, and the 768p checkpoint's own upstream
        spec table only lists a different *video* training shift).
        """
        # is_ref=False (generate/generate_still_batch = base transformer) は
        # H3_TURBO_LORA_FILE_BASE 由来の shift (未指定なら ref と同一 = 従来挙動)。
        turbo_shift = H3_TURBO_VIDEO_SHIFT if is_ref else H3_TURBO_VIDEO_SHIFT_BASE
        if turbo_shift is None:
            # このバリアントの LoRA は shift 切替なし。ただし scheduler は base/ref 共有なので、
            # 直前の別バリアントのリクエストが shift を切り替えたままなら元に戻す。
            if (H3_TURBO_VIDEO_SHIFT is not None or H3_TURBO_VIDEO_SHIFT_BASE is not None) \
                    and self._pipe.scheduler.shift != self._base_video_shift:
                self._pipe.scheduler.set_shift(self._base_video_shift)
                logger.info("turbo video scheduler shift restored: %.3f (variant has no turbo shift)", self._base_video_shift)
            return
        desired = turbo_shift if turbo_effective else self._base_video_shift
        scheduler = self._pipe.scheduler
        if scheduler.shift == desired:
            return
        scheduler.set_shift(desired)
        logger.info(
            "turbo video scheduler shift %s: %.3f (turbo=%s, H3_TURBO_VIDEO_SHIFT=%.3f, base=%.3f)",
            "applied" if turbo_effective else "restored", desired, turbo_effective,
            turbo_shift, self._base_video_shift,
        )

    def apply_instant_settings(
        self,
        transformer,
        resolved: dict,
        is_ref: bool = False,
        progress: ProgressState | None = None,
    ) -> None:
        """Apply the full instant-apply settings group (cache/attn/turbo) to one
        transformer instance, in the order that matters least-to-most structural:
        attention backend (pure attribute set) -> cache (hook install/remove) -> turbo
        (Linear module wrap, only on first use). See each `_apply_*_setting` method's
        own docstring. Called from `generate()`/`generate_ref2va()` right after the
        request's transformer is confirmed resident, before the denoise loop.

        `resolved` is the dict `core.settings.resolve_instant_settings()` returns --
        uses `resolved["effective_cache"]` (not `resolved["cache"]`) so a turbo=True
        request's FBC force-off (see that function's docstring) actually takes effect
        on the transformer, not just in the reported settings.
        """
        label = "transformer_ref" if is_ref else "transformer"
        self._apply_attn_setting(transformer, resolved["attn"], label=label)
        self._apply_cache_setting(transformer, resolved["effective_cache"], resolved["cache_threshold"], label=label)
        self._apply_turbo_setting(transformer, resolved["turbo"], is_ref=is_ref, progress=progress)

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

    def _text_encoder_config_kwargs(self) -> dict:
        """Build the `config=` per-component kwarg for `load_components(names=["text_encoder", ...])`.

        `{}` (H3_TE_PRUNE=0, default): no `config` override passed at all -- `spec.load()`
        falls back to its own auto-load-config-from-checkpoint path, byte-for-byte the
        pre-H3_TE_PRUNE behaviour.

        H3_TE_PRUNE=1: loads the checkpoint's own `Qwen3VLConfig` (same
        pretrained_model_name_or_path/subfolder the text_encoder `ComponentSpec` itself
        uses -- `MiniMaxAI/MiniMax-H3`, subfolder `text_encoder`, confirmed by reading
        `self._pipe._component_specs["text_encoder"]` at task-verification time), then
        truncates `text_config.num_hidden_layers` to `MINIMAX_H3_TEXT_ENCODER_LAYER + 1`
        (see `H3_TE_PRUNE`'s module-level comment for why +1, not the read index itself)
        before handing it back. `PreTrainedModel.from_pretrained` skips its own config
        auto-load whenever `config` is already a `PreTrainedConfig` instance (verified by
        reading modeling_utils.py), so passing this object through is enough to make
        `Qwen3VLForConditionalGeneration.__init__` build only `layers[0:51]` -- the
        checkpoint's `layers.51-63.*` and `lm_head.*` become "UNEXPECTED" keys in the
        load report and are simply skipped, never materialized on GPU/CPU at all. The
        text model's own final `norm` (unlike the layers, its construction does not
        depend on `num_hidden_layers`) is still built and its checkpoint weight still
        loads normally, but it is functionally dead: nothing downstream ever reads
        `last_hidden_state` (the only thing `norm` feeds), only `hidden_states[50]`
        (`layers[49]`'s raw output, captured before `norm` ever runs on it).
        """
        if not H3_TE_PRUNE:
            return {}
        from transformers import Qwen3VLConfig

        spec = self._pipe._component_specs["text_encoder"]
        cfg = Qwen3VLConfig.from_pretrained(
            spec.pretrained_model_name_or_path, subfolder=spec.subfolder or None
        )
        cfg.text_config.num_hidden_layers = MINIMAX_H3_TEXT_ENCODER_LAYER + 1
        return {"text_encoder": cfg}

    @property
    def _te_external(self) -> bool:
        """TE を計算用GPUとは別のデバイスへ常駐させる構成か(`H3_TE_DEVICE` 参照)。"""
        return bool(H3_TE_DEVICE) and H3_TE_DEVICE != str(DEVICE)

    # ref2va で TE 外部常駐を許すのに必要な、TE用GPUの総容量 (10進GB)。
    # **32B TE**: 24GB。2048px 短辺の参照を 32B の vision tower に通す活性化が大きく、
    #   TE-nf4 17.45GB + 参照エンコード 3.22GB 以上 = 20.67GB 以上を要するため
    #   (20GB 級カードで 19.25GB 使用中に 204MB 不足する OOM を実測済み)。
    # **投影TE (H3_TE_PROJ)**: 8GB。こちらの vision tower は Qwen3-VL-4B で、重みが
    #   NF4 3.11GB。2026-08-11 の 8GiB×2 検証では TE 側の実測ピークが約 5.7GB で
    #   ref2i/ref2va が完走している。24GB を課すのは 32B 前提の値の流用で、16GB や
    #   20GB の2枚目GPUを不当に弾いてしまう (2026-08-13 に分離)。
    # ref2va の参照ビジョンエンコードに要る「TE用GPU上の**空き** VRAM」(10進GB)。
    # 2026-08-14 に「総容量」判定から「実測空き」判定へ変更 (回帰監査ケース9の教訓:
    # 総容量だけ見ると、同じGPUに別プロセスが同居しているとき判定を通過してから
    # 実行時に OOM する。空きで判定すれば同居環境でも安全側に倒れる)。
    #
    # 値の根拠 (判定時点で TE は常駐済みなので、常駐分を**含まない**追加要求量):
    # - 32B: 旧ルール「総容量 24GB 以上」から TE-nf4 常駐 17.45GB を引いた 6.5GB。
    #   (参照1枚の vision 活性化 3.22GB 以上 + 余裕。20GB カードで 204MB 不足の OOM を
    #    実測した経緯は H3_TE_DEVICE のコメント参照)
    # - 投影 (4B): 4.0GB。8GiB×2 検証 (2026-08-11) では参照1枚で TE 側ピーク 5.7GB
    #   (= 常駐 3.11 + 活性化 ~2.6GB)。回帰監査 (08-13) では空き 2.3GB で1枚は成功・
    #   2枚は OOM だったので、2〜3枚分の余裕を見て 4.0GB とする。専有 GPU なら
    #   16GB カードでも空き ~12GB で余裕で通る。同居で足りないときは 400 で
    #   「H3_TE_DEVICE を外せ」と案内される (黙って OOM するよりよい)。
    _TE_EXTERNAL_MIN_FREE_GB_REF2VA_32B = 6.5
    _TE_EXTERNAL_MIN_FREE_GB_REF2VA_PROJ = 4.0

    def _te_external_usable_for(self, mode: str) -> bool:
        """このモードで TE 外部常駐を使ってよいか。

        ref2va は参照画像を vision tower に通すぶん TE 側の活性化が大きいので、TE用GPUの
        **実測空き VRAM** が足りなければ拒否する (呼び出し側が 400 で案内する) --
        「動くはず」で走らせて OOM させるより、理由を添えて明確に断る。必要量は**どちらの
        TE を使っているか**で大きく違うので、閾値は上の2定数で分けている。

        空きは `torch.cuda.mem_get_info()` で読む。これは**他プロセスの使用分も反映**する
        ので、同じ GPU にほかのサービスが同居していても正しく安全側に倒れる
        (`H3_VRAM_LIMIT_GB` の上限はここには映らない -- あれは自プロセスの確保上限で、
        mem_get_info が返す物理空きとは別勘定。同居運用で TE 側にも上限を掛けたい場合は
        物理空きがそのまま判定に効くのでこのままでよい)。
        """
        if not self._te_external:
            return False
        if mode != "ref2va":
            return True
        try:
            device_index = torch.device(H3_TE_DEVICE).index
            free_bytes, _total_bytes = torch.cuda.mem_get_info(device_index)
        except Exception:
            return False
        free_gb = free_bytes / 1e9
        need = (self._TE_EXTERNAL_MIN_FREE_GB_REF2VA_PROJ if H3_TE_PROJ
                else self._TE_EXTERNAL_MIN_FREE_GB_REF2VA_32B)
        if free_gb < need:
            logger.warning(
                "ref2va with external TE refused: %s has %.2fGB free but %.1fGB is "
                "needed for the reference vision encode (%s TE). Another process may "
                "be sharing this GPU.",
                H3_TE_DEVICE, free_gb, need, "projected 4B" if H3_TE_PROJ else "32B",
            )
        return free_gb >= need

    @property
    def _encode_device(self) -> torch.device:
        """プロンプトエンコードを実行するデバイス。TE 外部常駐なら TE のある側。"""
        return torch.device(H3_TE_DEVICE) if self._te_external else DEVICE

    def _to_compute_device(self, prompt_embeds, text_token_tags):
        """エンコード結果を計算用GPUへ移す(TE 外部常駐のときのみ実体のコピーが起きる)。

        運ぶのは prompt_embeds だけ(4,104トークン×5,120×bf16 で約42MB)なので、
        PCIe Gen4 x4 でも約6ms。`text_token_tags` は CPU 上の小さな整数テンソル。
        """
        if not self._te_external:
            return prompt_embeds, text_token_tags
        return prompt_embeds.to(DEVICE), text_token_tags

    # PR #14355 マージ版 (f37ab93) 対応: バッチ経路のレイアウト段が作る state テンソルを
    # デノイズ開始前に計算用GPUへ移すためのキー一覧。新しい before_denoise 系ステップは
    # 出力テンソルを `components._execution_device` に置くが、バッチの位相並べ替え
    # (エンコード位相で layout/latents/timesteps まで済ませ、transformer ロードは
    # デノイズ位相まで遅延)では、その時点の `_execution_device` が解決不能
    # (TE 外部常駐だと transformer 未ロード・TE デタッチ済みで、CPU 常駐の audio_vae か
    # フォールバックの cpu に落ちる)なので、テンソルは CPU に生まれる。値は正しく
    # デバイスだけが違うため、デノイズ直前に明示的に運べばよい。generate() 単発経路は
    # transformer ロード後に `_pin_execution_device_to_compute()` の窓で回すので不要。
    _SCENE_STATE_TENSOR_KEYS = (
        "latents", "audio_latents", "prompt_embeds",
        "position_ids", "token_tags", "video_indices", "audio_indices", "text_indices",
        "timesteps", "audio_timesteps", "row_timestep_plan",
        "condition_latents", "audio_condition_latents",
    )

    def _scene_state_to_compute(self, state) -> None:
        """バッチ場面の state テンソル群を計算用GPU (`DEVICE`) へ移す(上のキー一覧の
        コメント参照)。tensor / list / tuple の入れ子(`row_timestep_plan` は
        `(unique_timesteps, timestep_indices)` タプルのリスト)を再帰的に運ぶ。
        既に GPU 上なら `.to()` は no-op なので常に呼んで安全。"""
        def move(v):
            if isinstance(v, torch.Tensor):
                return v.to(DEVICE)
            if isinstance(v, (list, tuple)):
                return type(v)(move(x) for x in v)
            return v

        for key in self._SCENE_STATE_TENSOR_KEYS:
            v = state.get(key)
            if v is not None:
                state.set(key, move(v))

    def _detach_te_if_external(self):
        """ロード直後に呼ぶ: 外部常駐 TE の実体を `self._te_module` へ移し、パイプからは
        外す。以後は `_te_attached()` の窓の中でだけ繋がる(理由はそちらの docstring)。"""
        if not self._te_external:
            return
        self._te_module = self._pipe.text_encoder
        self._pipe.text_encoder = None
        if self._pipe_ref is not None:
            self._pipe_ref.text_encoder = None

    @contextmanager
    def _te_attached(self):
        """TE 外部常駐時、**この窓の間だけ** TE をパイプへ繋ぐ。

        窓の外では外しておくのが要点。`_execution_device` は components 順で最初の
        nn.Module を拾うので、別GPU上の TE を繋ぎっぱなしにすると layout だけでなく
        **decode も別GPUにテンソルを作る** -- 実機で
        `latents = latents * latents_std + latents_mean` が
        `Expected all tensors to be on the same device, cuda:0 and cuda:1` で落ちるのを
        再現した。窓を個別に塞ぐのではなく「既定は外れている」方が安全なので、
        エンコードのときだけ繋ぐ設計にしている。

        モジュール自体は `self._te_module` が保持し続けるので、外しても解放されない
        (別GPUに常駐させたままにするのがこの構成の目的)。
        """
        if not self._te_external:
            yield
            return
        pipe, pipe_ref = self._pipe, self._pipe_ref
        pipe.text_encoder = self._te_module
        if pipe_ref is not None:
            pipe_ref.text_encoder = self._te_module
        try:
            yield
        finally:
            pipe.text_encoder = None
            if pipe_ref is not None:
                pipe_ref.text_encoder = None

    @contextmanager
    def _pin_execution_device_to_compute(self):
        """この窓の間だけ `_execution_device` が計算用GPU(`DEVICE`)を返すようにする。

        `_execution_device` は components 順で最初の nn.Module のデバイスを返すため、
        TE を別GPUへ置くと layout/latents/timesteps がそちらにテンソルを作ってしまう
        (`H3_TE_DEVICE` のコメントの「最大の罠」)。PR #14355 マージ版 (f37ab93) の順序は
        `image_processor, text_encoder, tokenizer, processor, vae, audio_vae, scheduler,
        audio_scheduler, transformer_ref, transformer, video_processor` -- **`audio_vae` が
        `transformer` より前に移動した**(旧版は transformer の後)ので、text_encoder と
        vae に加えて **audio_vae も一時的にパイプから外さないと、CPU 常駐の audio_vae が
        最初の nn.Module として拾われて `_execution_device` が cpu に化ける**(マージ版
        での最初の E2E で position_ids が CPU に作られ、rope() 内の device mismatch として
        実測再現)。3つ外すと次の nn.Module が `transformer` (計算用GPU上) になる
        (`transformer_ref` は未ロード時 None なので isinstance スキャンに掛からない)。
        モジュールは runner 側で参照を保持したまま外すだけなので、解放も再ロードも
        発生しない。

        前提: この窓に入る時点で transformer が計算用GPUにロード済みであること
        (呼び出し側がそう並べる)。
        """
        pipe = self._pipe
        saved_te, saved_vae, saved_audio_vae = pipe.text_encoder, pipe.vae, pipe.audio_vae
        pipe.text_encoder = None
        pipe.vae = None
        pipe.audio_vae = None
        try:
            yield
        finally:
            pipe.text_encoder = saved_te
            pipe.vae = saved_vae
            pipe.audio_vae = saved_audio_vae

    def _te_prequant_dir(self) -> Path:
        """量子化済み TE のキャッシュ先。**設定ごとに別ディレクトリ**にする
        (TE_QUANT / TE_PRUNE を変えると重みの中身が別物になるため、同じ場所を使い回すと
        設定切替時に古い重みを読む事故が起きる。名前に設定を埋めて構造的に防ぐ)。"""
        return H3_TE_PREQUANT_DIR / f"te_{TE_QUANT}_prune{int(H3_TE_PRUNE)}"

    def _load_te_from_prequant(self, cache_dir: Path, progress: ProgressState | None = None) -> bool:
        """量子化済みキャッシュから TE を読む。成功したら True。

        読めなかった場合(壊れている・transformers のバージョン差など)は**例外を投げず
        False を返す** -- キャッシュは高速化であって機能ではないので、失敗したら通常の
        ロード経路へ黙って落ちるのが正しい。壊れたキャッシュは呼び出し側が作り直す。
        """
        if not (cache_dir / "config.json").exists():
            return False
        if progress:
            progress.update(
                phase="loading_text_encoder",
                message="text_encoder (量子化済みキャッシュ) をロード中...",
            )
        t0 = time.time()
        try:
            from transformers import AutoModelForImageTextToText

            te = AutoModelForImageTextToText.from_pretrained(
                str(cache_dir), dtype=torch.bfloat16,
                device_map=H3_TE_DEVICE if self._te_external else "cuda",
            )
        except Exception:
            logger.exception("量子化済み TE キャッシュの読み込みに失敗、通常経路へフォールバック: %s", cache_dir)
            return False
        # tokenizer/processor はキャッシュ対象外(小さく、量子化とも無関係)なので
        # 通常どおりロードする。text_encoder だけを差し替える。
        self._pipe.load_components(names=["tokenizer", "processor"])
        self._pipe.text_encoder = te
        self._text_encoder_loaded = True
        logger.info(
            "text_encoder loaded from prequantized cache in %.1fs (%s). gpu=%s",
            time.time() - t0, cache_dir, gpu_mem_gb(),
        )
        _install_patch_embed_linear(self._pipe.text_encoder)  # no-op unless H3_PATCH_EMBED_LINEAR=0
        _install_qwen3vl_submodule_timing(self._pipe.text_encoder)  # no-op unless H3_PHASE_TIMING=1
        if H3_TE_DIET and not self._te_external:
            _apply_te_diet(self._pipe.text_encoder)  # no-op unless H3_TE_DIET=1 (bnb-4bit TE only)
            logger.info("text_encoder after H3_TE_DIET. gpu=%s", gpu_mem_gb())
        if H3_TE_STREAM and not self._te_external:
            t_stream = time.time()
            pinned = _apply_te_stream(self._pipe.text_encoder)  # after diet (same order as LTX2.5)
            logger.info(
                "text_encoder after H3_TE_STREAM (window=%d, pinned %.2fGiB, %.1fs). gpu=%s ram=%s",
                H3_TE_STREAM_WINDOW, pinned, time.time() - t_stream, gpu_mem_gb(), ram_gb(),
            )
        self._detach_te_if_external()
        # フェーズ境界での中断チェック(loading_text_encoder)。
        interrupt_controller.check()
        return True

    def _save_te_prequant(self, cache_dir: Path):
        """ロード済み TE を量子化済みのまま保存する。失敗しても生成は続行する。

        ディスクを 17-21GB 消費するため、空きが `H3_TE_PREQUANT_MIN_FREE_GB` を下回る
        場合は保存せずに警告だけ出す(ディスクを埋めてシステムを巻き添えにしないため)。
        """
        import shutil

        try:
            free_gb = shutil.disk_usage(H3_TE_PREQUANT_DIR.parent if H3_TE_PREQUANT_DIR.exists()
                                        else Path.cwd()).free / 1e9
        except Exception:
            free_gb = float("inf")
        if free_gb < H3_TE_PREQUANT_MIN_FREE_GB:
            logger.warning(
                "量子化済み TE の保存をスキップ: 空きディスクが %.1fGB で下限 %.1fGB を"
                "下回る (H3_TE_PREQUANT_MIN_FREE_GB で調整可、H3_TE_PREQUANT=0 で無効化可)",
                free_gb, H3_TE_PREQUANT_MIN_FREE_GB,
            )
            return
        tmp_dir = cache_dir.with_name(cache_dir.name + ".tmp")
        try:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            self._pipe.text_encoder.save_pretrained(str(tmp_dir))
            # 一時ディレクトリへ書いてから rename する: 保存中にプロセスが落ちても
            # 中途半端なキャッシュが「有効」に見えてしまうのを防ぐ (config.json の
            # 存在でキャッシュ有無を判定しているため)。
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            tmp_dir.rename(cache_dir)
            size_gb = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()) / 1e9
            logger.info(
                "量子化済み TE を保存: %s (%.2fGB, %.1fs)。次回以降のロードが高速になる",
                cache_dir, size_gb, time.time() - t0,
            )
        except Exception:
            logger.exception("量子化済み TE の保存に失敗(生成は続行): %s", cache_dir)
            try:
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # transformer/transformer_ref 量子化済みキャッシュ (H3_TRANSFORMER_PREQUANT)
    # ------------------------------------------------------------------
    def _transformer_prequant_dir(self, is_ref: bool) -> Path:
        """量子化済み transformer(_ref) のキャッシュ先。TE と同じく**設定ごとに別
        ディレクトリ**にする (`transformer_int8` / `transformer_ref_int8` -- 2つの
        インスタンスは同一モデルクラス/config だが個別にロード・保存されるため、
        混同しないよう名前でも分ける)。H3_TRANSFORMER_QUANT が int8 以外のときは
        呼び出し側がそもそもこのキャッシュに触れないので、ディレクトリ名に量子化方式は
        含めていない (現状 int8 のみが対象)。"""
        if H3_PRUNED and is_ref:
            # pruned は重み・構造とも別物なので専用ディレクトリ(非 pruned の
            # transformer_ref_int8 と取り違えない)。さらに方式ごとに分ける(ConvRot の
            # 有無はキャッシュファイルから判別できないため、名前で分けるしかない)。
            # int8dyn-convrot は従来名 transformer_ref_pruned_int8convrot のまま。
            name = H3_PRUNED_QUANT_SPEC.cache_name
            if name is None:
                raise RuntimeError(
                    f"H3_PRUNED_QUANT={H3_PRUNED_QUANT!r} はキャッシュ対象外です(呼び出し側のバグ)"
                )
        else:
            name = "transformer_ref_int8" if is_ref else "transformer_int8"
        return H3_TRANSFORMER_PREQUANT_DIR / name

    def _transformer_prequant_metadata(self, pruned: bool = False) -> dict:
        """キャッシュ無効化用のメタデータ。保存時に `meta.json` として書き込み、
        ロード時にこれと一致するかを確認する。一致しなければキャッシュは無効
        (作り直す) 扱い -- ソースチェックポイントが更新された、torchao がバージョン
        アップされた、量子化レシピ (modules_to_not_convert) が変わった、のいずれかで
        古い重みを黙って読んでしまう事故を防ぐ。

        ソースチェックポイントの識別には HF ローカルキャッシュのスナップショットパス
        (コミットハッシュを含む、`try_to_load_from_cache` で安価に取得できる) を使う --
        ファイル自体のハッシュ化は 66GB を読み直すことになり本末転倒なので行わない。
        """
        import importlib.metadata

        from huggingface_hub import try_to_load_from_cache

        try:
            cached = try_to_load_from_cache(MODEL_ID, "transformer/config.json")
            source_snapshot = str(cached) if cached else None
        except Exception:
            source_snapshot = None
        meta = {
            "model_id": MODEL_ID,
            "source_snapshot": source_snapshot,
            "torchao_version": importlib.metadata.version("torchao"),
            "torch_version": torch.__version__,
            "quant_config": "Int8WeightOnlyConfig(version=2)",
            # プロセス内で mutate されうる H3_INT8_MODULES_TO_NOT_CONVERT ではなく、
            # 定義時点の pristine スナップショットを使う(定義箇所のコメント参照)。
            "modules_to_not_convert": sorted(_H3_INT8_MODULES_TO_NOT_CONVERT_PRISTINE),
        }
        if pruned:
            # 【2026-10-07 修正】pruned キーを足すのは「pruned キャッシュの検証/保存」の
            # ときだけ(引数 pruned)。旧実装は H3_PRUNED(プロセス全体のフラグ)で分岐して
            # いたため、H3_PRUNED=1 運用では base 用 transformer_int8 の meta にまで
            # pruned の quant_config が混入し、H3_PRUNED_QUANT を切り替えただけで無関係な
            # base キャッシュが無効化 -> 削除済み bf16 シャード 62GB の再DLが走った(実機)。
            try:
                from huggingface_hub import try_to_load_from_cache as _ttlfc

                cached = _ttlfc(H3_PRUNED_REPO, "transformer_ref/config.json")
                pruned_snapshot = str(cached) if cached else None
            except Exception:
                pruned_snapshot = None
            # quant_config は方式ごとの文字列(既定は従来の
            # "Int8DynamicActivationInt8WeightConfig+convrot" のまま)。group size は
            # ConvRot を掛ける方式でだけレシピの一部になる(回転なしの方式では値を
            # 変えてもキャッシュが無効にならないよう含めない)。
            meta.update({
                "pruned_repo": H3_PRUNED_REPO,
                "pruned_snapshot": pruned_snapshot,
                "quant_config": H3_PRUNED_QUANT_SPEC.meta_quant_config,
            })
            if H3_PRUNED_QUANT_SPEC.convrot:
                meta["convrot_group_size"] = H3_PRUNED_CONVROT_GROUP
        return meta

    def _load_transformer_from_prequant(
        self, cache_dir: Path, is_ref: bool, progress: ProgressState | None = None
    ) -> bool:
        """量子化済みキャッシュから transformer(_ref) を読む。成功したら True。

        `_load_te_from_prequant` と同じ fail-open 方針: 読めなかった場合 (キャッシュが
        存在しない、壊れている、メタデータが現在の設定と食い違う) は**例外を投げず
        False を返す** -- 呼び出し側は黙って通常の bf16ロード+量子化経路へ落ちる。
        """
        meta_path = cache_dir / "meta.json"
        config_path = cache_dir / "config.json"
        if not (meta_path.exists() and config_path.exists()):
            return False
        try:
            saved_meta = json.loads(meta_path.read_text())
        except Exception:
            logger.warning("transformer 量子化済みキャッシュの meta.json が壊れています、"
                            "通常経路へフォールバック: %s", cache_dir)
            return False
        current_meta = self._transformer_prequant_metadata(pruned=False)
        if saved_meta != current_meta:
            logger.info(
                "transformer 量子化済みキャッシュのメタデータが現在の設定と不一致のため無効"
                "扱いにします (通常経路で作り直します): %s\n保存済み=%s\n現在=%s",
                cache_dir, saved_meta, current_meta,
            )
            return False
        label = "transformer_ref" if is_ref else "transformer"
        if progress:
            progress.update(
                phase="loading_transformer",
                message=f"{label} (量子化済みキャッシュ) をロード中...",
            )
        t0 = time.time()
        try:
            from diffusers import MiniMaxH3Transformer3DModel

            tr = MiniMaxH3Transformer3DModel.from_pretrained(str(cache_dir), torch_dtype=torch.bfloat16)
            tr = tr.to(DEVICE)
        except Exception:
            logger.exception(
                "%s 量子化済みキャッシュの読み込みに失敗、通常経路へフォールバック: %s",
                label, cache_dir,
            )
            return False
        if is_ref:
            self._pipe_ref.transformer_ref = tr
        else:
            self._pipe.transformer = tr
        logger.info(
            "%s loaded from prequantized cache in %.1fs (%s). gpu=%s",
            label, time.time() - t0, cache_dir, gpu_mem_gb(),
        )
        return True

    def _load_pruned_ref_from_prequant(
        self, cache_dir: Path, progress: ProgressState | None = None
    ) -> bool:
        """pruned(H3_PRUNED_QUANT の方式)の量子化済みキャッシュから transformer_ref を読む。

        `_load_transformer_from_prequant` と同じ fail-open 方針(読めなければ False を
        返し、呼び出し側が通常の bf16 ロード+量子化経路で作り直す)。違いは
        ①モデルクラスが remote code の MiniMaxH3PrunedTransformer3DModel であること
        ②ConvRot 系の方式ではオンライン入力回転がクラス差し替えで実現されており直列化
        されないため、ロード後に `mark_convrot()` でクラスを付け直すこと(ConvRot 無しの
        方式では呼ばない。どちらを呼ぶかは core/pruned.py が方式から機械的に決める)。
        """
        meta_path = cache_dir / "meta.json"
        config_path = cache_dir / "config.json"
        if not (meta_path.exists() and config_path.exists()):
            return False
        try:
            saved_meta = json.loads(meta_path.read_text())
        except Exception:
            logger.warning("pruned 量子化済みキャッシュの meta.json が壊れています、"
                           "通常経路へフォールバック: %s", cache_dir)
            return False
        current_meta = self._transformer_prequant_metadata(pruned=True)
        if saved_meta != current_meta:
            logger.info(
                "pruned 量子化済みキャッシュのメタデータが現在の設定と不一致のため無効"
                "扱いにします: %s\n保存済み=%s\n現在=%s",
                cache_dir, saved_meta, current_meta,
            )
            return False
        if progress:
            progress.update(
                phase="loading_transformer",
                message="transformer_ref (pruned 量子化済みキャッシュ) をロード中...",
            )
        try:
            from core import pruned as pruned_mod

            # weights=False: キャッシュロードに bf16 シャードは不要(remote code と
            # config だけ解決する。シャード削除運用で 38GB の再DLを誘発しないため)。
            snapshot = pruned_mod.pruned_snapshot_dir(H3_PRUNED_REPO, weights=False)
            tr = pruned_mod.load_pruned_ref_from_cache(
                cache_dir, snapshot, DEVICE, H3_PRUNED_CONVROT_GROUP, quant=H3_PRUNED_QUANT
            )
        except Exception:
            logger.exception(
                "pruned 量子化済みキャッシュの読み込みに失敗、通常経路へフォールバック: %s",
                cache_dir,
            )
            return False
        self._pipe_ref.transformer_ref = tr
        return True

    def _save_transformer_prequant(self, cache_dir: Path, transformer, is_ref: bool):
        """ロード済み (量子化直後、turbo LoRA 等でまだ手を加えていない) transformer(_ref)
        を量子化済みのまま保存する。失敗しても生成は続行する (`_save_te_prequant` と
        同じ fail-open 方針)。

        呼び出し側 (`_ensure_transformer`/`_ensure_transformer_ref`) は、turbo LoRA の
        構造的な Linear wrap や attention backend 設定より**前**、量子化直後にこれを
        呼ぶこと -- 保存点の妥当性は本タスクの Phase 1/2 で確認済み (H3_TRANSFORMER_
        PREQUANT の module docstring 参照)。
        """
        import shutil

        label = "transformer_ref" if is_ref else "transformer"
        try:
            free_gb = shutil.disk_usage(
                H3_TRANSFORMER_PREQUANT_DIR.parent if H3_TRANSFORMER_PREQUANT_DIR.exists()
                else Path.cwd()
            ).free / 1e9
        except Exception:
            free_gb = float("inf")
        if free_gb < H3_TRANSFORMER_PREQUANT_MIN_FREE_GB:
            logger.warning(
                "%s 量子化済みキャッシュの保存をスキップ: 空きディスクが %.1fGB で下限 %.1fGB"
                "を下回る (H3_TRANSFORMER_PREQUANT_MIN_FREE_GB で調整可、"
                "H3_TRANSFORMER_PREQUANT=0 で無効化可)",
                label, free_gb, H3_TRANSFORMER_PREQUANT_MIN_FREE_GB,
            )
            return
        avail_ram = ram_gb()["avail_gb"]
        if avail_ram < H3_TRANSFORMER_PREQUANT_MIN_RAM_GB:
            logger.warning(
                "%s 量子化済みキャッシュの保存をスキップ: 空きホストRAMが %.1fGBで下限 %.1fGB"
                "を下回る (CLAUDE.md #33 と同じ理由でホストRAM枯渇を避けるため。"
                "H3_TRANSFORMER_PREQUANT_MIN_RAM_GB で調整可)",
                label, avail_ram, H3_TRANSFORMER_PREQUANT_MIN_RAM_GB,
            )
            return
        tmp_dir = cache_dir.with_name(cache_dir.name + ".tmp")
        try:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            # `save_pretrained` の既定 (`safe_serialization=True`, `max_shard_size="10GB"`)
            # はシャード単位で `safetensors.torch.save_file()` を呼ぶ (GPU上のテンソルの
            # ままの `state_dict()` をシャードに分けて直列化するため、34GB 全体を一度に
            # CPU へ複製することはない -- モジュール冒頭の H3_TRANSFORMER_PREQUANT_MIN_RAM_GB
            # のコメント参照)。
            if H3_PRUNED and is_ref:
                # pruned は hf_quantizer を介さない生 torchao 量子化のため safetensors
                # 保存が不可(core/pruned.py の save_pruned_ref_cache 参照)。
                from core import pruned as pruned_mod

                pruned_mod.save_pruned_ref_cache(transformer, tmp_dir, quant=H3_PRUNED_QUANT)
            else:
                transformer.save_pretrained(str(tmp_dir))
            (tmp_dir / "meta.json").write_text(
                json.dumps(self._transformer_prequant_metadata(pruned=H3_PRUNED and is_ref),
                           indent=2, ensure_ascii=False)
            )
            # 一時ディレクトリへ書いてから rename する: 保存中にプロセスが落ちても
            # 中途半端なキャッシュが「有効」に見えてしまうのを防ぐ (config.json/meta.json
            # の存在でキャッシュ有無を判定しているため)。
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            tmp_dir.rename(cache_dir)
            size_gb = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()) / 1e9
            logger.info(
                "%s 量子化済みキャッシュを保存: %s (%.2fGB, %.1fs)。次回以降のロードが高速になる",
                label, cache_dir, size_gb, time.time() - t0,
            )
        except Exception:
            logger.exception("%s 量子化済みキャッシュの保存に失敗(生成は続行): %s", label, cache_dir)
            try:
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
            except Exception:
                pass

    def _resolve_te_proj_path(self) -> str:
        """`H3_TE_PROJ` の実ファイルパスを解決する (キャッシュ、初回のみ)。

        ローカルパスとして存在すればそのまま使う。存在しなければ HF リポジトリID
        として扱い、`hf_hub_download(H3_TE_PROJ, H3_TE_PROJ_FILE)` で取得する --
        `_download_turbo_lora_if_needed` と同じパターン (通常の HF キャッシュ経由、
        2回目以降はローカルヒット)。
        """
        if getattr(self, "_te_proj_path", None) is not None:
            return self._te_proj_path
        if Path(H3_TE_PROJ).is_file():
            self._te_proj_path = H3_TE_PROJ
            logger.info("H3_TE_PROJ: using local projection file %s", self._te_proj_path)
            return self._te_proj_path
        from huggingface_hub import hf_hub_download

        t0 = time.time()
        self._te_proj_path = hf_hub_download(H3_TE_PROJ, H3_TE_PROJ_FILE)
        logger.info(
            "H3_TE_PROJ: projection checkpoint resolved: %s (%.1fs, repo=%s file=%s)",
            self._te_proj_path, time.time() - t0, H3_TE_PROJ, H3_TE_PROJ_FILE,
        )
        return self._te_proj_path

    def _load_text_encoder_proj(self, progress: ProgressState | None = None):
        """`H3_TE_PROJ` 有効時の text_encoder ロード経路: 32B TE の代わりに
        `H3_TE_PROJ_MODEL` (既定 Qwen3-VL-4B-Instruct) を bf16 でロードし、学習済み
        投影行列を一度だけロードしてキャッシュする。トークナイザ/プロセッサは H3 の
        ものを使い続ける (通常語彙は 4B とID完全一致 -- モジュール冒頭コメント参照)ので
        通常経路と同じく `self._pipe.load_components(names=["tokenizer", "processor"])`
        で取得し、`text_encoder` だけこの経路で個別にロードして差し替える
        (`_load_te_from_prequant` が量子化済みキャッシュの `text_encoder` を差し替える
        のと同じ形)。

        `H3_TE_DEVICE` (TE 別GPU常駐) とは併用可能: 4B も 32B 同様、指定があれば
        そちらへ直接ロードし、`_detach_te_if_external`/`_te_attached` の窓開閉に
        そのまま乗る (この2つはモデルの中身を問わない汎用ロジックのため)。
        """
        if progress:
            progress.update(
                phase="loading_text_encoder",
                message=f"text_encoder ({H3_TE_PROJ_MODEL}, 投影TE) をロード中...",
            )
        t0 = time.time()
        from transformers import AutoModelForImageTextToText

        load_kwargs = dict(
            dtype=torch.bfloat16,
            device_map=H3_TE_DEVICE if self._te_external else "cuda",
        )
        if H3_TE_PROJ_QUANT != "none":
            # 4B 自体の量子化。**投影行列は bf16 用のものをそのまま使う**のが正しい
            # (2026-08-10 実測、下記)。
            #
            # 投影は 4B (bf16) の隠れ状態の統計 (mean_in/std_in) に合わせて校正されて
            # いるので、「量子化すると統計がずれて行列が使えないのでは」という懸念は
            # 当然ある。3プロンプト (英語/公式記法/日本語) で投影後の条件付けを実測した
            # 結果、**ズレは 1% 未満**で cosine は 1.0000 だった:
            #
            #   NF4  : 相対RMS 0.61〜0.96%   常駐 3.11GB (bf16 8.88GB から -65%)
            #   int8 : 相対RMS 0.24〜0.53%   常駐 4.84GB
            #
            # 配布元は `h3_qwen3vl_4b_int8convrot_tap24.safetensors` という量子化版
            # 専用の行列も出しているが、**あれを使うとかえってズレが大きくなる**
            # (相対RMS 1.02〜2.97%)。ComfyUI の `int8_convrot` 方式に合わせて校正された
            # もので、bitsandbytes の量子化とは別物だから。**bf16 用行列を使うこと。**
            from transformers import BitsAndBytesConfig

            if H3_TE_PROJ_QUANT == "bnb-4bit":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:  # "bnb-8bit"
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        te = AutoModelForImageTextToText.from_pretrained(H3_TE_PROJ_MODEL, **load_kwargs)
        # tokenizer/processor は H3 のもの (通常ロード経路と同一) -- text_encoder だけ
        # 4B に差し替える。
        self._pipe.load_components(names=["tokenizer", "processor"])
        self._pipe.text_encoder = te
        self._text_encoder_loaded = True
        logger.info(
            "text_encoder (%s, H3_TE_PROJ) loaded to GPU%s in %.1fs. gpu=%s ram=%s",
            H3_TE_PROJ_MODEL, f" ({H3_TE_DEVICE})" if self._te_external else "",
            time.time() - t0, gpu_mem_gb(), ram_gb(),
        )
        _install_patch_embed_linear(self._pipe.text_encoder)  # no-op unless H3_PATCH_EMBED_LINEAR=0
        _install_qwen3vl_submodule_timing(self._pipe.text_encoder)  # no-op unless H3_PHASE_TIMING=1
        self._detach_te_if_external()

        # 投影行列は一度だけロードしてキャッシュする (`self._pipe._te_projection`,
        # `_te_projection_for()` が読む場所)。ロード先デバイスは TE と同じ
        # (`_encode_device` -- 外部常駐なら TE 側GPU) にして、エンコード時に
        # デバイスまたぎのコピーが発生しないようにする。
        if getattr(self._pipe, "_te_projection", None) is None:
            proj_path = self._resolve_te_proj_path()
            self._pipe._te_projection = _TeProjection(proj_path, device=self._encode_device)
        # フェーズ境界での中断チェック(loading_text_encoder)。
        interrupt_controller.check()

    def _load_text_encoder(self, progress: ProgressState | None = None):
        """Load the text_encoder to GPU.

        `none` mode: ~66GB bf16-native TE, loaded/freed per request, frees the
        transformer first (they cannot coexist).
        `bnb-4bit` mode: ~18GB NF4-quantized TE, loaded once at startup and kept
        resident forever (bnb 4bit models cannot be moved between devices, so
        `device_map="cuda"` places it directly and there is nothing to cycle).

        `H3_TE_PRUNE=1` (either TE_QUANT mode): the text_encoder is built with only its
        first 51 (of 64) decoder layers -- see `H3_TE_PRUNE`'s module-level comment and
        `_text_encoder_config_kwargs()`'s docstring for why 51 and why this is exact,
        not an approximation. Measured savings: ~3.6GB (bnb-4bit nf4, 21.0GB -> 17.4GB)
        or ~13.6GB (bf16, 66.7GB -> 53.1GB).

        `H3_TE_PROJ` (opt-in, mutually exclusive with `H3_TE_QUANT`/`H3_TE_PRUNE`/
        `H3_TE_PREQUANT` -- import-time guard, see the module comment): the 32B TE is not
        loaded at all. Instead `H3_TE_PROJ_MODEL` (Qwen3-VL-4B-Instruct, bf16, ~5.2GB) is
        loaded onto `self._pipe.text_encoder`, and the learned projection matrix
        (`H3_TE_PROJ`) is loaded once and cached on `self._pipe._te_projection`. The
        tokenizer/processor stay H3's own (loaded normally below) -- only the conditioner
        model itself is swapped.
        """
        self._ensure_pipe_shell()
        if self._text_encoder_loaded:
            return
        if H3_TE_PROJ:
            self._load_text_encoder_proj(progress)
            return
        # 量子化済みキャッシュがあればそこから読む (実測 66.9s -> 2.6s、出力はビット一致。
        # H3_TE_PREQUANT のモジュールコメント参照)。bnb-4bit のときだけ意味がある --
        # `none` モードは量子化しないので保存しても得が無い。
        cache_dir = self._te_prequant_dir()
        if H3_TE_PREQUANT and TE_QUANT == "bnb-4bit" and self._load_te_from_prequant(cache_dir, progress):
            return
        config_kwargs = self._text_encoder_config_kwargs()
        prune_suffix = ", pruned to 51 layers" if H3_TE_PRUNE else ""
        if TE_QUANT == "bnb-4bit":
            if progress:
                progress.update(
                    phase="loading_text_encoder",
                    message=f"text_encoder (Qwen3-VL-32B, NF4{prune_suffix}) をロード中...",
                )
            t0 = time.time()
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            # Per-component kwargs: `load_components` broadcasts a plain (non-dict) kwarg
            # value to every named component, but tokenizer/processor do not accept
            # `quantization_config` / `device_map` / (pruned) `config`. Use the dict form
            # (component name -> value) so only `text_encoder` gets them -- same shape
            # `config_kwargs` (from `_text_encoder_config_kwargs`) already uses, `{}` when
            # H3_TE_PRUNE=0 so this is a pure no-op addition to the kwargs dict in that case.
            self._pipe.load_components(
                names=["text_encoder", "tokenizer", "processor"],
                dtype=torch.bfloat16,
                quantization_config={"text_encoder": quant_config},
                # TE 外部常駐 (H3_TE_DEVICE) のときはそのデバイスへ直接置く。
                # bnb-4bit はデバイス間移動ができないので、置き場所はロード時に決める。
                device_map={"text_encoder": H3_TE_DEVICE if self._te_external else "cuda"},
                config=config_kwargs,
            )
            self._text_encoder_loaded = True
            logger.info(
                "text_encoder (NF4%s) loaded to GPU%s in %.1fs. gpu=%s ram=%s",
                prune_suffix, f" ({H3_TE_DEVICE})" if self._te_external else "",
                time.time() - t0, gpu_mem_gb(), ram_gb(),
            )
            # 初回のみ: 量子化済みの重みを保存しておき、次回以降のロードを短縮する。
            # **必ず `_detach_te_if_external()` より前に呼ぶこと**: detach は
            # `self._pipe.text_encoder = None` にするため、後に回すと TE 外部常駐
            # (H3_TE_DEVICE) 構成で保存が毎回 AttributeError になり、prequant
            # キャッシュが永遠に作られない (2026-08-20 実機で発症・修正)。
            if H3_TE_PREQUANT:
                self._save_te_prequant(cache_dir)
            _install_patch_embed_linear(self._pipe.text_encoder)  # no-op unless H3_PATCH_EMBED_LINEAR=0
            _install_qwen3vl_submodule_timing(self._pipe.text_encoder)  # no-op unless H3_PHASE_TIMING=1
            if H3_TE_DIET and not self._te_external:
                # prequant 保存(上)は完全な重みで済ませた後に適用する (保存物を痩せさせない)。
                _apply_te_diet(self._pipe.text_encoder)
            if H3_TE_STREAM and not self._te_external:
                t_stream = time.time()
                pinned = _apply_te_stream(self._pipe.text_encoder)  # after diet (same order as LTX2.5)
                logger.info(
                    "text_encoder after H3_TE_STREAM (window=%d, pinned %.2fGiB, %.1fs). gpu=%s ram=%s",
                    H3_TE_STREAM_WINDOW, pinned, time.time() - t_stream, gpu_mem_gb(), ram_gb(),
                )
            self._detach_te_if_external()
            # フェーズ境界での中断チェック(loading_text_encoder)。
            interrupt_controller.check()
            return
        # TE (66GB, or ~53GB pruned) + transformer (66GB) cannot coexist in 96GB VRAM:
        # measured 66.73GB for the unpruned TE alone (the checkpoint shards are
        # bf16-native, not fp32). The two big models therefore cycle: TE on GPU only
        # during prompt encoding.
        self._free_transformer()
        if progress:
            progress.update(phase="loading_text_encoder", message=f"text_encoder (Qwen3-VL-32B{prune_suffix}) をロード中...")
        t0 = time.time()
        self._pipe.load_components(
            names=["text_encoder", "tokenizer", "processor"], dtype=torch.bfloat16, config=config_kwargs
        )
        self._pipe.text_encoder.to(DEVICE)
        self._text_encoder_loaded = True
        logger.info(
            "text_encoder%s loaded to GPU in %.1fs. gpu=%s ram=%s",
            prune_suffix, time.time() - t0, gpu_mem_gb(), ram_gb(),
        )
        _install_patch_embed_linear(self._pipe.text_encoder)  # no-op unless H3_PATCH_EMBED_LINEAR=0
        _install_qwen3vl_submodule_timing(self._pipe.text_encoder)  # no-op unless H3_PHASE_TIMING=1
        # フェーズ境界での中断チェック(loading_text_encoder)。
        interrupt_controller.check()

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
        # TE 外部常駐 (H3_TE_DEVICE): 別GPUに置いてある TE は解放しない -- 解放しないこと
        # 自体がこの構成の目的 (毎リクエストの再ロード 29.5-53s を消すため)。計算用GPUの
        # ヘッドルームには一切影響しないので、force=True の呼び出しも無視してよい。
        if self._te_external:
            logger.debug("text_encoder は %s に常駐しているため解放しない", H3_TE_DEVICE)
            return
        if (H3_TE_PROJ or TE_QUANT == "bnb-4bit") and not force:
            # Permanently resident in this mode -- never freed mid-run (see
            # _load_text_encoder docstring). Guard so a stray call is a harmless no-op
            # rather than silently dropping the model. `H3_TE_PROJ` (4B TE) is small
            # enough that it is meant to stay resident alongside the transformer just
            # like bnb-4bit's nf4 TE -- same "load once, keep forever" steady state,
            # just with a much smaller model.
            logger.debug(
                "text_encoder (%s) is permanently resident; ignoring free request",
                "H3_TE_PROJ" if H3_TE_PROJ else "bnb-4bit",
            )
            return
        # Drop the CUDA model directly: releasing the last reference frees the VRAM in
        # place. Do NOT stage through .to("cpu") first -- the text_encoder is ~21-66GB
        # depending on quantization, and a host-RAM transit would both waste time and
        # evict the page-cached model shards that make the next per-request reload fast.
        #
        # BUG FOUND DURING THIS TASK'S FIRST MIGRATION STAGE (pre-PR#14355-ref2va-port,
        # two-shell design): `self._pipe_ref.text_encoder` (set by
        # `_sync_shared_components_to_ref` via plain attribute assignment onto a
        # *separate* shell object, so it held its own strong reference to the same
        # module) also had to be cleared here, or it kept the refcount above zero and
        # `del self._pipe.text_encoder` freed nothing. PR #14355's ref2va port made
        # `self._pipe_ref` a plain alias for `self._pipe` (`_ensure_pipe_ref_shell`'s
        # docstring) -- there is only one shell now, so the block below is a redundant
        # no-op (`self._pipe_ref.text_encoder` is already `None`, set by the line right
        # above, since they are the same object) rather than a fix for a live bug. Left
        # in rather than deleted: harmless, and it stays correct if this alias
        # relationship were ever to change again.
        # 単発プレフィックス KV キャッシュ (`H3_REF_PREFIX_CACHE_SINGLE`) は、これから
        # 落とす TE インスタンスの重みで作ったもの。TE を落とす以上は必ず一緒に捨てる
        # (捨てないと ~1.0GiB の VRAM が「もう二度と使えない KV」として残る。
        # `_encode_ref2va_prompt_prefix_cached` 側の weakref チェックでも救えるが、
        # VRAM を即座に返すためここで能動的に捨てる)。既定 OFF のときは常に no-op。
        _clear_single_ref_prefix_cache("text_encoder freed")
        # H3_TE_STREAM: LM 層の pinned host 実体 (~14GiB) は PyTorch の host キャッシュに残って
        # しまう (del + gc + empty_cache では OS へ返らない) ので、TE を落とすときに明示的に返す。
        _te_was_streamed = bool(getattr(self._pipe.text_encoder, "_te_stream_applied", False))
        del self._pipe.text_encoder
        self._pipe.text_encoder = None
        if self._pipe_ref is not None and getattr(self._pipe_ref, "text_encoder", None) is not None:
            del self._pipe_ref.text_encoder
            self._pipe_ref.text_encoder = None
        # H3_TE_PROJ の投影行列キャッシュ (`_load_text_encoder_proj` が
        # `self._pipe._te_projection` へセットしたもの) もここで捨てる。`self._pipe`
        # シェル自体は unload_all()/preload_all() の間も生き続ける (_ensure_pipe_shell
        # 参照) ので、消さないと te_proj を OFF にした後も古い投影行列が
        # `_te_projection_for()` から見え続け、32B TE に戻ったはずの経路が投影を使う
        # 「静かな残留」になる。逆に ON にする再ロードでも、直前の TE 設定 (bnb-4bit 等)
        # と混同しないよう常に作り直させるのが安全 (`_load_text_encoder_proj` は
        # `getattr(..., None) is None` のときだけ作るため、消しておかないと使い回されて
        # しまう)。
        if getattr(self._pipe, "_te_projection", None) is not None:
            self._pipe._te_projection = None
        self._text_encoder_loaded = False
        gc.collect()
        torch.cuda.empty_cache()
        if _te_was_streamed and hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()
        logger.info("text_encoder freed (force=%s). gpu=%s ram=%s", force, gpu_mem_gb(), ram_gb())

    def _free_vaes(self):
        """Drop vae/audio_vae entirely (not just move to CPU -- see `_vae_to_cpu` for the
        short-window move used mid-request; this is the full free, used only by
        `unload_all()`). Needed because `_ensure_vaes()` early-returns when
        `self._vae_loaded` is already True, so a reload-group change to
        `H3_VIDEO_VAE_FP16` (whether the video VAE is cast to fp16 after load) or
        `TE_QUANT` (whether the VAE pair defaults to parked-on-CPU or
        permanently-on-GPU) would otherwise silently keep serving the *old* VAE
        placement/dtype after `apply_reload_settings()` flips the underlying env-var
        equivalents. Same "drop in place, no CPU staging" shape as
        `_free_transformer`/`_free_text_encoder` (CLAUDE.md #33) -- the VAE pair is only
        ~11GB either way, small enough that staging concerns do not really apply, but
        consistency with the rest of this file's unload methods is kept anyway.
        """
        if not self._vae_loaded:
            return
        if getattr(self._pipe, "vae", None) is not None:
            del self._pipe.vae
            self._pipe.vae = None
        if getattr(self._pipe, "audio_vae", None) is not None:
            del self._pipe.audio_vae
            self._pipe.audio_vae = None
        # H3_VAE_PINNED: モジュールを捨てるのでマスターも破棄 (pinned ホスト RAM を返す)。
        self._vae_pin_map = {}
        self._vae_loaded = False
        self._vae_on_gpu = False
        self._vae_gpu_parts = set()
        self._free_decode_vaes()
        _clear_ref_latent_cache("vae freed")
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("vae/audio_vae freed. gpu=%s ram=%s", gpu_mem_gb(), ram_gb())

    def unload_all(self):
        """Free every big model this runner may be holding (both transformers, both
        text_encoder references, the VAE pair) -- used by
        `core.settings.apply_reload_settings()` before reloading under a new
        configuration, and safe to call any time nothing is mid-request (callers must
        hold the same lock `generate()`/`generate_ref2va()` use -- `apply_reload_settings`
        gets this for free via the app-level generation lock, see app.py).

        Deliberately does NOT drop `self._pipe`/`self._pipe_ref` (the ModularPipeline
        shells themselves) -- rebuilding those is cheap and stateless (see
        `_ensure_pipe_shell`/`_ensure_pipe_ref_shell`'s own docstrings: no component
        weights, just the block-spec wiring), so there is no reason to pay that cost
        again. `_active_variant` is reset to None since neither transformer is resident
        any more after this call.
        """
        t0 = time.time()
        self._free_transformer()
        self._free_transformer_ref()
        if self._pin_slots:
            # H3_BASE_PINNED/H3_REF_PINNED: 上の free は park (モジュール保持) になる
            # が、unload_all は設定変更リロードの前段 = 完全破棄が必要 (park したまま
            # だと新しい turbo/attn 設定が restore 経路で再適用されない)。スロットごと
            # 捨てて pinned ホスト RAM も返す。
            self._pin_slots.clear()
            gc.collect()
            _host_empty_cache = getattr(torch._C, "_host_emptyCache", None)
            if _host_empty_cache is not None:
                _host_empty_cache()
            logger.info("unload_all: pinned park slots purged")
        self._free_text_encoder(force=True)
        # `_free_text_encoder` は TE 外部常駐 (`H3_TE_DEVICE`) では早期 return するので、
        # そちらの構成でも確実に捨てるためここでも呼ぶ (既定 OFF なら no-op)。
        _clear_single_ref_prefix_cache("unload_all")
        self._free_vaes()
        self._active_variant = None
        logger.info("unload_all: done in %.1fs. gpu=%s ram=%s", time.time() - t0, gpu_mem_gb(), ram_gb())

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
        `H3_LOWVRAM_GROUP`: unlike `H3_LOWVRAM=1`, the (group-offloaded) transformer
        IS preloaded here and stays resident for the life of the process -- it lives in
        host RAM, not VRAM, so there is no reason to pay its ~34GB CPU load + quantize
        cost on every request the way `H3_LOWVRAM=1` pays a ~34GB *GPU* load each time.
        TE still cycles per-request (not preloaded), matching `H3_LOWVRAM=1`'s choice to
        keep the steady-state VRAM footprint minimal between requests.
        """
        with self._load_lock:
            if H3_PRUNED:
                # import 時の logger.info は uvicorn のロギング設定前で消えるので、起動時の
                # 構成はここで出す(H3_LOWVRAM=1 では transformer_ref は初回リクエストまで
                # ロードされず、`_ensure_transformer_ref` のログだけだと起動直後に分からない)。
                logger.info(
                    "H3_PRUNED=1: transformer_ref は pruned、H3_PRUNED_QUANT=%s -> %s",
                    H3_PRUNED_QUANT, H3_PRUNED_QUANT_SPEC.describe(H3_PRUNED_CONVROT_GROUP),
                )
            if _keep_ref2va_active():
                # import 時のログは uvicorn のロギング設定前で消えるので、起動時の構成表明はここで出す。
                logger.info(
                    "H3_KEEP_REF2VA=1: ref2va は text_encoder + transformer_ref(%s) をリクエスト間で "
                    "GPU 常駐させる (初回 ref2va でロード、以降スキップ)。vae_resident=%s",
                    "pruned" if H3_PRUNED else "int8", H3_KEEP_REF2VA_VAE,
                )
                logger.info(
                    "H3_KEEP_REF2VA low-VRAM options: te_diet=%s ref_prefix_park=%s vae_split=%s "
                    "te_stream=%s (window=%d)",
                    H3_TE_DIET, H3_REF_PREFIX_PARK and H3_REF_PREFIX_CACHE_SINGLE, H3_VAE_SPLIT,
                    H3_TE_STREAM, H3_TE_STREAM_WINDOW,
                )
                if H3_REF_PREFIX_PARK and not H3_REF_PREFIX_CACHE_SINGLE:
                    logger.warning("H3_REF_PREFIX_PARK=1 has no effect without H3_REF_PREFIX_CACHE_SINGLE=1")
            elif H3_VAE_SPLIT:
                logger.warning("H3_VAE_SPLIT=1 is ignored without H3_KEEP_REF2VA=1 (+H3_LOWVRAM=1)")
            self._ensure_vaes()
            if H3_LOWVRAM_GROUP:
                self._ensure_transformer()
            elif not H3_LOWVRAM:
                self._ensure_transformer()
                # H3_TE_PROJ (4B+投影) は bnb-4bit の 32B NF4 TE と同じ「一度ロードして
                # 常駐させ続ける」対象 (`_free_text_encoder` の force=False no-op ガード
                # 参照)。preload しないと最初のリクエストまで4Bロードが遅延するだけで
                # 壊れはしないが、bnb-4bit と扱いを揃えて起動時に済ませておく (UI から
                # te_proj を ON にする apply_reload_settings() の直後もこの preload_all()
                # を通るので、この分岐がないと ON 切替の「再ロード」がTEをロードしない
                # まま終わってしまう)。
                if H3_TE_PROJ:
                    self._load_text_encoder()
                elif TE_QUANT == "bnb-4bit":
                    self._load_text_encoder()

    # ------------------------------------------------------------------
    # H3_DECODE_STREAM / H3_DECODE_DEVICE (2026-10-05, 既定 OFF)
    # ------------------------------------------------------------------
    def _decode_target_device(self) -> torch.device:
        return torch.device(H3_DECODE_DEVICE) if H3_DECODE_DEVICE else DEVICE

    def decode_deferred_active(self) -> bool:
        """decode を denoise から分離する経路 (`_decode_ref2va_deferred`) が今のリクエストで
        使えるか。フラグ OFF なら常に False (= 従来のインライン decode)。

        成立条件は「VAE が GPU に常駐していて、リクエストの出入りで動かされない」こと
        (`H3_KEEP_REF2VA=1` + `H3_KEEP_REF2VA_VAE=1` + `H3_LOWVRAM=1`)。そうでない構成では
        次リクエストのエンコードが `_vae_to_cpu()` 等で VAE を動かし、decode と衝突しうる
        ので、警告を1回出して従来経路にフォールバックする。
        """
        if not (H3_DECODE_STREAM or H3_DECODE_DEVICE):
            return False
        reason = None
        if not (_keep_ref2va_active() and H3_KEEP_REF2VA_VAE):
            reason = "H3_KEEP_REF2VA=1 + H3_KEEP_REF2VA_VAE=1 + H3_LOWVRAM=1 (VAE 常駐) が必要"
        elif (H3_VAE_SPLIT and not H3_DECODE_VAE
              and self._decode_target_device() == DEVICE):
            # H3_VAE_SPLIT では標準 VAE のデコーダはデコード窓でだけ GPU に載る (常駐しない)。
            # deferred decode はロックを手放した後に走り、次リクエストのエンコードが並行して
            # VAE を出し入れしうるので、ここでデコーダを GPU へ戻すことはできない。light VAE
            # (H3_DECODE_VAE) と別 GPU (H3_DECODE_DEVICE) は専用コピーを使うので影響しない。
            reason = ("H3_VAE_SPLIT=1 では標準 VAE のデコーダが常駐しないため、同じ GPU での "
                      "deferred decode には H3_DECODE_VAE(light)か別 GPU の H3_DECODE_DEVICE が必要")
        elif H3_DECODE_DEVICE:
            try:
                d = torch.device(H3_DECODE_DEVICE)
                if d.type != "cuda" or (d.index or 0) >= torch.cuda.device_count():
                    reason = f"H3_DECODE_DEVICE={H3_DECODE_DEVICE!r} は存在する CUDA デバイスではない"
            except Exception as e:  # noqa: BLE001
                reason = f"H3_DECODE_DEVICE={H3_DECODE_DEVICE!r} を解釈できない ({e})"
        if reason:
            if not self._decode_warned:
                self._decode_warned = True
                logger.warning("H3_DECODE_STREAM/H3_DECODE_DEVICE は無効化されインライン decode を使う: %s", reason)
            return False
        return True

    def decode_overlap_active(self) -> bool:
        """denoise 完了時点で生成ロックを手放す (= 次リクエストと decode が重なる) か。"""
        return H3_DECODE_STREAM and self.decode_deferred_active()

    def _ensure_decode_vaes(self) -> None:
        """`H3_DECODE_DEVICE` が cuda:0 以外のとき、video/audio VAE の decode 専用コピーを
        そのデバイスへ作る (冪等)。transformer/TE/参照エンコード用の VAE (cuda:0) には触らない。
        """
        if self._decode_vae is not None:
            return
        if not self.decode_deferred_active():
            return
        separate = self._decode_target_device() != DEVICE
        if not (separate or H3_DECODE_STREAM_SIDE or H3_DECODE_VAE):
            return
        import copy

        dev = self._decode_target_device()
        t0 = time.time()
        if H3_DECODE_VAE:
            # light VAE (probe)。標準 VAE と同じ手順 (fp32 ロード→fp16 キャスト) で作る。
            from diffusers import AutoencoderKLMiniMaxH3

            vae = AutoencoderKLMiniMaxH3.from_pretrained(H3_DECODE_VAE_REPO, torch_dtype=torch.float32)
            vae = vae.to(torch.float16)
            vae.eval()
            vae.set_attention_backend("native")
            logger.info("decode-only video VAE = %s (H3_DECODE_VAE, decoder_num_layers=%s) loaded in %.2fs",
                        H3_DECODE_VAE_REPO, vae.config.decoder_num_layers, time.time() - t0)
        else:
            vae = copy.deepcopy(self._pipe.vae)
            # `_fp16_autocast_encode` (encode のパッチ) は元の VAE にひも付くクロージャ。decode 専用
            # コピーでは使わないので外して、取り違えようがないようにする。
            vae.__dict__.pop("encode", None)
            if H3_DECODE_NATIVE_ATTN or not separate:
                # 専用ストリーム版: sage は side stream で壊れる (H3_DECODE_STREAM のコメント) ので native 固定。
                vae.set_attention_backend("native")
        self._decode_vae = vae.to(dev)
        # audio VAE は deepcopy できない (legacy weight_norm が計算済みの非リーフ `weight` を属性に持つ:
        # "Only Tensors created explicitly by the user support the deepcopy protocol")。config から
        # 組み立て直して state_dict を流し込む。
        from diffusers import AutoencoderKLMiniMaxH3Audio

        src_audio = self._pipe.audio_vae
        audio_copy = AutoencoderKLMiniMaxH3Audio.from_config(src_audio.config)
        audio_copy.load_state_dict({k: v.detach().cpu() for k, v in src_audio.state_dict().items()})
        audio_copy.eval()
        audio_copy.set_attention_backend("native")  # 本体の audio_vae と同じ (fp32 固定なので sage 不可)
        self._decode_audio_vae = audio_copy.to(dev)
        torch.cuda.synchronize(dev)
        logger.info("decode-only vae/audio_vae copies placed on %s in %.2fs (H3_DECODE_DEVICE)", dev, time.time() - t0)

    def _free_decode_vaes(self) -> None:
        if self._decode_vae is None and self._decode_audio_vae is None:
            return
        dev = self._decode_target_device()
        self._decode_vae = None
        self._decode_audio_vae = None
        gc.collect()
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()
        logger.info("decode-only vae copies freed (%s)", dev)

    def _decode_ref2va_deferred(self, pipe, holder: list, progress, on_denoise_done):
        r"""ref2va の decode (video VAE + audio VAE) + uint8 変換を、denoise から分離して実行する。

        `holder` は `[video_latents, audio_latents]` (呼び出し側がローカル参照を残さないための
        受け渡し用リスト。ここで pop して、decode 完了時に解放する)。

        **なぜ生成ロック解放後に decode して安全か** (`H3_DECODE_STREAM=1` のとき):
          1. この時点で transformer の仕事は終わっており、残りの入力は latent 2本だけ。
             以降 decode が触るのは VAE の重みと自分の中間テンソルだけで、`state` /
             scheduler / transformer / TE / KV キャッシュには一切触れない。
          2. VAE モジュールはステートレス (`autoencoder_kl_minimax_h3.py` に feature cache 等の
             インスタンス状態なし。tiling フラグは常に True のまま誰も書き換えない)。次の
             リクエストの参照エンコード (同じ video VAE の encode) と同じ重みを読むだけ。
             VAE の GPU⇔CPU 往復 (`_vae_to_cpu`) は KEEP_REF2VA_VAE=1 では起きない
             (`decode_deferred_active` がこの構成を成立条件にしている)。
          3. torch の current stream / autocast / no_grad はスレッドローカル。`=1` では
             デフォルト (legacy) ストリームに両スレッドの演算が投入され、GPU 上では投入順に
             処理される (カーネルの真の同時実行は無いが、飢餓も起きない)。`=2` は専用ストリーム
             (PyTorch の side stream は non-blocking) で真に重なるが、sage attention 等
             「current stream を尊重しないカーネル」が共有 VAE に混ざると壊れるため、native
             attention に固定した decode 専用コピーを使う (`_ensure_decode_vaes`)。
          4. 入力 latent は denoise と同じスレッドで `synchronize` 済み、decode 完了
             (`.cpu()` で同期) まで本スレッドが保持するので、use-after-free は起きない。
          5. decode 同士は `_decode_lock` で直列化 (次々リクエストの decode が重なって VRAM を
             二重に積まない)。mux はロック外 (CPU のみ)。
          6. 注意: 重なっている間は (a) `torch.cuda.reset_peak_memory_stats()` を次リクエストが
             呼ぶので同一 GPU 上のピーク統計は信頼できない、(b) 同一 GPU の SM を取り合うので
             denoise/decode とも単体より遅くなる、(c) VRAM は両者の和が要る (96GB 機の
             resident 構成では余裕があるが、48GB 以下では `H3_DECODE_DEVICE` で別 GPU に逃がす)。
        `H3_DECODE_DEVICE` のみ (STREAM なし) の場合はロックを手放さない: 純粋に decode を
        別 GPU へオフロードするだけ (VRAM を cuda:0 から外す効果。レイテンシは同等)。
        """
        from contextlib import nullcontext

        dev = self._decode_target_device()
        separate = dev != DEVICE
        lat, alat = holder.pop(0), holder.pop(0)
        _tl("pre_sync")
        torch.cuda.synchronize(DEVICE)  # denoise + unpatchify の GPU 作業が完了していること
        _tl("post_sync")
        _dump_dir = os.environ.get("H3_DEBUG_DUMP_LATENT", "").strip()  # probe 用 (既定 OFF): 標準/light の同一 latent 比較
        if _dump_dir:
            os.makedirs(_dump_dir, exist_ok=True)
            torch.save({"video": lat.detach().cpu(), "audio": alat.detach().cpu()},
                       os.path.join(_dump_dir, f"lat_{int(time.time() * 1000)}_{lat.shape[2]}f.pt"))
        peak_at_release = torch.cuda.max_memory_allocated() / 1e9
        released = False
        if H3_DECODE_STREAM and on_denoise_done is not None:
            on_denoise_done()
            released = True
            _tl("lock_released")
        t_wait0 = time.time()
        if progress:
            progress.update(phase="decoding", message="動画/音声をデコード中...")
        info: dict = {
            "device": str(dev), "separate_gpu": separate, "stream": bool(H3_DECODE_STREAM),
            "side_stream": bool(H3_DECODE_STREAM_SIDE),
            "lock_released_before_decode": released,
            "decode_vae": H3_DECODE_VAE_REPO or "standard",
        }
        with self._decode_lock:
            info["decode_lock_wait_s"] = round(time.time() - t_wait0, 3)
            if separate or H3_DECODE_STREAM_SIDE or H3_DECODE_VAE:
                self._ensure_decode_vaes()
                vae, audio_vae = self._decode_vae, self._decode_audio_vae
            else:
                vae, audio_vae = pipe.vae, pipe.audio_vae
            if separate:
                torch.cuda.reset_peak_memory_stats(dev)
            if H3_DECODE_STREAM_SIDE and self._decode_stream is None:
                with torch.cuda.device(dev):
                    self._decode_stream = torch.cuda.Stream(device=dev)
            stream = self._decode_stream if H3_DECODE_STREAM_SIDE else None
            t_dec = time.time()
            with torch.cuda.device(dev), (torch.cuda.stream(stream) if stream is not None else nullcontext()), \
                    torch.no_grad():
                t_x = time.time()
                if separate:
                    # **GPU 間の直接コピー (P2P) は使わない**: この機 (RTX PRO 6000 + RTX PRO 5000,
                    # PCIe NODE 接続) では `tensor.to("cuda:1")` が can_device_access_peer=True
                    # にもかかわらず**無警告で全要素 0 を書き込む** (2026-10-05 実機、float 100M 要素
                    # まで全サイズで再現)。最初の A/B で GPU1 decode が「ゼロ latent の decode」に
                    # なり PSNR 11dB の別物映像になって発覚した。latent は数 MB なので必ず
                    # ホスト経由でステージングする (実測 ~2ms)。
                    lat_d = lat.detach().cpu().to(dev)
                    alat_d = alat.detach().cpu().to(dev)
                    torch.cuda.synchronize(dev)
                else:
                    lat_d, alat_d = lat, alat
                info["latent_transfer_s"] = round(time.time() - t_x, 4)
                info["latent_transfer_mb"] = round((lat.numel() * lat.element_size()
                                                    + alat.numel() * alat.element_size()) / 1e6, 2)
                mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
                std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)
                x = lat_d * std + mean
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    video = vae.decode(x, return_dict=False)[0]
                video = video.cpu()  # 同期。fp16 のまま CPU へ降ろして CPU 側で逆正規化 (既定経路と同じ)
                a_mean = torch.tensor(audio_vae.config.latents_mean, device=dev).view(1, -1, 1)
                a_std = torch.tensor(audio_vae.config.latents_std, device=dev).view(1, -1, 1)
                audio = audio_vae.decode(alat_d * a_std + a_mean, return_dict=False)[0]
                audio = audio.float().permute(1, 0, 2)
                audio_np = audio[0].float().cpu().numpy()
                sampling_rate = pipe.audio_sampling_rate
                if separate:
                    info["decode_peak_vram_gb"] = round(torch.cuda.max_memory_allocated(dev) / 1e9, 2)
            pixel_mean = torch.tensor(pipe.pixel_mean).view(1, -1, 1, 1, 1)
            pixel_std = torch.tensor(pipe.pixel_std).view(1, -1, 1, 1, 1)
            video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
            videos = pipe.video_processor.postprocess_video(video, output_type="pt")
            decode_time = time.time() - t_dec
            video_tensor = videos[0] if isinstance(videos, list) else videos
            t_u8 = time.time()
            frames_uint8 = frames_to_uint8(video_tensor)
            info["uint8_s"] = round(time.time() - t_u8, 3)
            rms = float(np.sqrt(np.mean(audio_np**2)))
            peak = float(np.max(np.abs(audio_np)))
            peak_vram = peak_at_release
            del lat, alat, lat_d, alat_d, x, video, videos, video_tensor, audio
            if not released:
                # 重なりのない構成では従来どおり掃除する。生成ロックを手放して重ねている間は
                # `torch.cuda.empty_cache()` を呼ばない: これは**全デバイス**のキャッシュを解放する
                # (current device だけではない) ため、別 GPU 上で走行中の次リクエストの割り当て
                # キャッシュまで cudaFree してしまう。実際に cuda:1 側で呼んだ empty_cache が
                # cuda:0 で denoise 中の次リクエストを "illegal memory access" で落とした
                # (2026-10-05 実機)。gc.collect() も同様にやらない。
                gc.collect()
                with torch.cuda.device(dev):
                    torch.cuda.empty_cache()
        return frames_uint8, audio_np, sampling_rate, rms, peak, peak_vram, decode_time, info

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
            "te_prune": H3_TE_PRUNE,
            "te_proj": bool(H3_TE_PROJ),
            "te_proj_model": H3_TE_PROJ_MODEL if H3_TE_PROJ else None,
            "te_proj_quant": H3_TE_PROJ_QUANT if H3_TE_PROJ else None,
            "te_proj_tap": (
                getattr(self._pipe, "_te_projection", None).tap
                if self._pipe is not None and getattr(self._pipe, "_te_projection", None) is not None
                else None
            ),
            "video_vae_fp16": H3_VIDEO_VAE_FP16,
            "transformer_quant": H3_TRANSFORMER_QUANT,
            # ref2va の transformer_ref を AdaLN-pruned で読むか、その量子化方式
            # (H3_PRUNED_QUANT、core/pruned.py)。実行中の構成を logs と突合するため。
            "pruned": H3_PRUNED,
            "pruned_quant": H3_PRUNED_QUANT if H3_PRUNED else None,
            "pruned_compile": H3_PRUNED_COMPILE_LEVEL,
            # fp32 matmul の精度設定(torch.get_float32_matmul_precision())。torchao の
            # quantize_() は set_inductor_config=True(既定)だと "high"(TF32)へ変えて
            # しまうので、量子化経路の副作用で変わっていないこと("highest")を観測するため。
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            # ref2va's reference-image normalization short edge (H3_REF_IMAGE_SHORT_EDGE,
            # see its definition above). Reported here (rather than only logged once at
            # pipe-shell build time) so a run's logs can be cross-checked against which
            # setting was actually in effect. 2048 = diffusers' own default / unset.
            "ref_image_short_edge": H3_REF_IMAGE_SHORT_EDGE,
            "lowvram": H3_LOWVRAM_RAW,
            "keep_ref2va": _keep_ref2va_active(),
            "keep_ref2va_vae": _keep_ref2va_active() and H3_KEEP_REF2VA_VAE,
            "te_diet": H3_TE_DIET,
            "te_stream": H3_TE_STREAM,
            "te_stream_window": H3_TE_STREAM_WINDOW if H3_TE_STREAM else None,
            "ref_prefix_park": H3_REF_PREFIX_PARK and H3_REF_PREFIX_CACHE_SINGLE,
            # 単機リアルタイム連続生成用の追加機能 (2026-10-05、いずれも既定 OFF)
            "ref_latent_cache": H3_REF_LATENT_CACHE,
            "ref_latent_cache_stats": dict(_ref_latent_cache_stats) if H3_REF_LATENT_CACHE else None,
            "decode_stream": _H3_DECODE_STREAM_RAW if H3_DECODE_STREAM else None,
            "decode_device": H3_DECODE_DEVICE or None,
            "decode_vae": H3_DECODE_VAE_REPO or None,
            "decode_deferred_active": self.decode_deferred_active(),
            "vae_split": H3_VAE_SPLIT and _keep_ref2va_active(),
            "lowvram_group": H3_LOWVRAM_GROUP,
            # import 時の logger.info は uvicorn がロギングを設定する前に走って消えるので、
            # 上限が効いているかは status 経由で確認できるようにしておく (None = 無制限)。
            "vram_limit_gb": float(H3_VRAM_LIMIT_GB) if H3_VRAM_LIMIT_GB else None,
            "group_offload_blocks": H3_GROUP_OFFLOAD_BLOCKS if H3_LOWVRAM_GROUP else None,
            "group_offload_use_stream": H3_GROUP_OFFLOAD_USE_STREAM if H3_LOWVRAM_GROUP else None,
            "group_offload_low_cpu_mem": H3_GROUP_OFFLOAD_LOW_CPU_MEM if H3_LOWVRAM_GROUP else None,
            "attn_backend": H3_ATTN_BACKEND or "default",
            "cache_mode": H3_CACHE if not H3_TURBO_LORA else "none (force-disabled by H3_TURBO_LORA)",
            "cache_threshold": H3_CACHE_THRESHOLD if (H3_CACHE == "fbc" and not H3_TURBO_LORA) else None,
            "turbo_lora": H3_TURBO_LORA,
            "turbo_lora_repo": H3_TURBO_LORA_REPO if H3_TURBO_LORA else None,
            "turbo_lora_path": self._turbo_lora_path,
            "turbo_lora_path_base": self._turbo_lora_path_base,
            "turbo_lora_file_base": H3_TURBO_LORA_FILE_BASE if H3_TURBO_LORA else None,
            # **`H3_TURBO_LORA` で出し分けしないこと** (2026-08-20 修正): turbo は
            # リクエスト単位の即反映設定 (`_apply_turbo_setting`、LoRA は初回 ON 時に
            # 遅延ロードされる) なので、起動時 `H3_TURBO_LORA=0` でも UI のチェック
            # ボックスから有効にできる。推奨ステップ数はそのとき初めて必要になる値で、
            # 「起動時に既定 ON だったか」とは無関係。旧実装は None を返しており、
            # UI の `syncStepsToTurbo()` が `TURBO_STEPS_DEFAULT === null` で
            # 黙って no-op になり、turbo を ON にしてもステップが 30 のままだった。
            "turbo_steps_default": H3_TURBO_STEPS_DEFAULT,
            # H3_ADALN_PRECOMP: the env flag itself, plus whether the table has actually
            # been *built yet* on each currently-resident transformer -- precompute is
            # armed at load time but only fires lazily on the first denoise step of the
            # next request that uses it (see core/adaln_precompute.py's
            # `enable_adaln_precompute()` docstring), so right after a fresh load these
            # can be `True`/`False`/`False` (flag on, nothing precomputed yet) until a
            # request actually runs a denoise step against that instance.
            "adaln_precomp": H3_ADALN_PRECOMP,
            "adaln_precomp_built": _adaln_precompute_status(self),
            # v2 coexistence: which turbo state each installed table was built while
            # observing (informational only -- see `_adaln_precompute_built_with_turbo()`
            # docstring, both turbo states are served correctly off the same table).
            "adaln_precomp_built_with_turbo": _adaln_precompute_built_with_turbo(self),
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

        PR #14355 (f37ab93) note: `packing.py` is gone. `patchify_video_latents` survives
        as a plain module function in before_denoise.py (imported below, unchanged
        signature/behaviour). `build_packed_sequence` and `build_row_timesteps` moved onto
        their respective step classes as `@staticmethod`s and both grew new required
        arguments (`audio_channels`/`audio_tag`/`video_tag` for the former, an explicit
        `video_indices`/`audio_indices`/... row-index shape for the latter, replacing the
        old `MiniMaxH3PackedSequence` namedtuple-style return with a plain positional
        tuple) -- both call sites below are updated for the new signatures.
        `unpatchify_video_tokens` has **no replacement upstream at all** (deliberately, per
        this project's "prefer self-implementation over vendoring" policy) -- it is ported
        verbatim below as a private module-level helper (`_unpatchify_video_tokens`),
        copied from this repo's own vendored copy of the pre-PR#14355 packing.py (the
        algorithm is also, byte-for-byte, what `MiniMaxH3AfterDenoiseStep.__call__` now
        inlines in decoders.py -- confirmed by reading both).
        """
        from diffusers.modular_pipelines.minimax_h3.before_denoise import (
            MiniMaxH3PrepareLayoutStep,
            MiniMaxH3SetTimestepsStep,
            patchify_video_latents,
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
        video_latent = _unpatchify_video_tokens(
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
        # indices all key off latent_height/latent_width -- see
        # MiniMaxH3PrepareLayoutStep.build_packed_sequence). text_token_tags/
        # num_audio_latents are unchanged (audio + text are untouched by the spatial
        # upscale), only the video row count and its rotary grid change. Calls the
        # staticmethod directly (the same one `MiniMaxH3PrepareLayoutStep.__call__` calls
        # internally) instead of going through the block, so `device` can be passed
        # explicitly instead of resolved via `components._execution_device` (unsafe here --
        # see the NOTE at the top of this function). PR #14355 made `audio_channels`/
        # `audio_tag`/`video_tag` required positional arguments -- supplied here from the
        # same `components` properties `MiniMaxH3PrepareLayoutStep.__call__` itself reads
        # (`components.audio_channels`/`.audio_tag`/`.video_tag`). The return is now a
        # plain positional tuple (`position_ids, token_tags, video_indices, audio_indices,
        # text_indices, num_condition_video_rows, num_condition_audio_rows`), not the old
        # `MiniMaxH3PackedSequence` namedtuple-style object with `sequence_length` as an
        # extra field -- unpacked by position below instead of by attribute.
        (
            new_position_ids,
            new_token_tags,
            new_video_indices,
            new_audio_indices,
            new_text_indices,
            _new_num_condition_video_rows,
            _new_num_condition_audio_rows,
        ) = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            state.get("text_token_tags"),
            num_latent_frames,
            new_h,
            new_w,
            num_audio_latents,
            patch_size,
            components.audio_channels,
            components.audio_tag,
            components.video_tag,
            (),  # keyframe_anchors: t2va only, enforced by the caller.
        )

        block_state.token_tags = new_token_tags.to(device)
        block_state.position_ids = new_position_ids.to(device)
        block_state.video_indices = new_video_indices.to(device)
        block_state.audio_indices = new_audio_indices.to(device)
        block_state.text_indices = new_text_indices.to(device)

        # PR #14355 (f37ab93) 対応 -- **これが無いとパス2が必ず落ちる** (2026-08-13 修正)。
        # マージ版の `MiniMaxH3LoopDenoiser` はレイアウト値を個々の属性からではなく
        # `block_state.denoiser_input_fields`(レイアウト段の OutputParam に付いた
        # `kwargs_type="denoiser_input_fields"` でまとめられた dict)から読む:
        #
        #     layout_kwargs = {name: value
        #                      for name, value in block_state.denoiser_input_fields.items()
        #                      if name in inspect.signature(transformer.forward).parameters}
        #
        # この dict はパス1のレイアウトを作った時点の**古いテンソルを参照したまま**なので、
        # 上の属性代入だけでは反映されない。結果、`state.set()` で書いている
        # row_timestep_plan だけがパス2の長さになり、token_tags/position_ids はパス1のまま
        # 残って `token_tags と timestep_indices の seq_len 不一致` で落ちていた
        # (この経路はマージ追従の回帰セット (t2i/t2va/バッチ/ref2va) に入っておらず、
        #  移行以降ずっと壊れていた)。
        # 既にあるキーだけを差し替える (新しいフィールドを勝手に生やさない)。
        _fields = getattr(block_state, "denoiser_input_fields", None)
        if isinstance(_fields, dict):
            for _name, _value in (
                ("token_tags", block_state.token_tags),
                ("position_ids", block_state.position_ids),
                ("video_indices", block_state.video_indices),
                ("audio_indices", block_state.audio_indices),
                ("text_indices", block_state.text_indices),
            ):
                if _name in _fields:
                    _fields[_name] = _value
        # t2va only (enforced by the caller): no conditioning rows, so both stay 0 --
        # matches what `build_packed_sequence` itself returns for an empty
        # `keyframe_anchors` (`_new_num_condition_video_rows`/`_new_num_condition_audio_rows`
        # above are always 0 too; assigned explicitly rather than read back for clarity).
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
        # onward (pass 2's own steps) with plans built against the new layout, while the
        # earlier entries (never read again -- pass 2 only iterates i >= pass1_steps) are
        # left as-is, just to keep the list the same full length the denoiser indexes into.
        #
        # PR #14355 note: `build_row_timesteps` moved onto `MiniMaxH3SetTimestepsStep` as a
        # `@staticmethod` and dropped the old `MiniMaxH3PackedSequence` `layout` object in
        # favour of the individual row-index tensors and counts it actually reads
        # (`video_indices`/`audio_indices`/`num_condition_video_rows`/
        # `num_condition_audio_rows`/`num_text_tokens`) -- all already in hand from step 5
        # above (`new_video_indices`/`new_audio_indices`, and the two condition-row counts,
        # which are always 0 here). `keyframe_noise_aug` (0.999) is now
        # `components.keyframe_noise_aug`, a property, replacing the old
        # `MINIMAX_H3_KEYFRAME_NOISE_AUG` module constant -- same value, same role (the
        # fixed noise level a conditioning anchor is held at), read the same way
        # `MiniMaxH3SetTimestepsStep.__call__` itself reads it.
        video_timesteps = state.get("timesteps")
        audio_timesteps = block_state.audio_timesteps
        num_text_tokens = state.get("text_token_tags").shape[0]
        old_plan = block_state.row_timestep_plan
        new_plan = list(old_plan)
        for i in range(pass1_steps, len(video_timesteps)):
            new_plan[i] = tuple(
                tensor.to(device)
                for tensor in MiniMaxH3SetTimestepsStep.build_row_timesteps(
                    new_video_indices,
                    new_audio_indices,
                    0,  # num_condition_video_rows: t2va only, enforced by the caller.
                    0,  # num_condition_audio_rows: t2va only, enforced by the caller.
                    num_text_tokens,
                    float(video_timesteps[i]),
                    float(audio_timesteps[i]),
                    max(float(video_timesteps[i]), components.keyframe_noise_aug),
                    1.0,
                )
            )
        block_state.row_timestep_plan = new_plan

        # Update state's own latent_height/latent_width too, in case anything reads them
        # again downstream (decode step reads latent_height/latent_width off `state`, not
        # `block_state` -- see the caller in generate()). The old `state.set("layout", ...)`
        # is dropped: it stored the retired `MiniMaxH3PackedSequence` object, which nothing
        # in this file ever read back via `state.get("layout")` (confirmed by grep) -- it
        # was write-only bookkeeping, not a real dependency.
        state.set("latent_height", new_h)
        state.set("latent_width", new_w)
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
        cache: str | None = None,
        cache_threshold: float | None = None,
        attn: str | None = None,
        turbo: bool | None = None,
        # 出力 mp4 に音声ストリームを入れない (生成そのものは止まらない --
        # `_mux_mp4` の docstring 参照)。
        mute: bool = False,
        still: bool = False,
        still_frames: int = 22,
    ) -> dict:
        if H3_HYPERFLOW:
            raise RuntimeError(
                "H3_HYPERFLOW=1 は ref2va 専用です (t2va/fl2va/静止画への配線は未実装)。"
                "generate_ref2va を使うか、H3_HYPERFLOW を外して起動し直してください。"
            )
        """
        Runs T2VA (image=None, last_image=None) or FL2VA (either/both given).

        `still=True` は静止画モード (t2i): `seconds` を無視して `still_frames`
        (STILL_FRAME_CHOICES のいずれか) の超短尺動画を生成し、中央フレームを PNG
        (`t2i_<ts>.png`) として書き出す。超短尺 mp4 も従来どおり保存する。diffusers 側の
        最小尺 5 秒バリデーションは `MiniMaxH3PrepareLayoutStep` の呼び出しの間だけ
        `_relaxed_min_duration()` で緩和する(PR #14355 後、このバリデーションは
        setup step ではなく layout step が行う -- 詳細はそちらの docstring 参照)。
        fl2va (image/last_image) および upscale との併用は未検証のため拒否。

        `upscale=1` enables two-pass hires-fix: pass 1 denoises `round(num_inference_steps
        * (1 - H3_HIRES_DENOISE))` steps at the requested (height, width), the video latent's
        x0 estimate is then spatially upscaled 2x with `F.interpolate` and re-noised (audio
        latent is left untouched -- it has no spatial axes, see `_upscale_block_state_2x`
        docstring), and pass 2 continues the same sigma trajectory for the remaining steps
        at 2x resolution. The returned `height`/`width` reflect the actual (2x) output
        resolution in that case.

        `cache`/`cache_threshold`/`attn`/`turbo`: instant-apply per-request overrides
        (see core/settings.py) for FirstBlockCache, the attention backend, and the
        turbo LoRA -- each defaults to whatever this process's own H3_CACHE/
        H3_CACHE_THRESHOLD/H3_ATTN_BACKEND/H3_TURBO_LORA env var resolved to when left
        `None`, so an existing caller that never passes these sees unchanged behaviour.
        Applied in-place to the already-resident transformer, no reload.

        Returns a dict with mp4_path, frame counts, timing and VRAM/RAM stats.
        """
        import core.settings as settings

        instant = settings.resolve_instant_settings(cache, cache_threshold, attn, turbo)
        # PR #14355 (f37ab93) import updates (see README "今後の外部イベント待ち" §1 for
        # the full audit this follows):
        #   - `MiniMaxH3SetupStep` is gone. Its t2va/fl2va-shared role (canvas defaulting +
        #     `17*n+5`/duration validation) moved into `MiniMaxH3PrepareLayoutStep.__call__`
        #     itself (before_denoise.py); its fl2va-only role (putting keyframes onto the
        #     canvas) is now the separate `MiniMaxH3ResizeStep` (before_encoder.py), run
        #     only for `is_fl2va` below -- there is no longer a step to run unconditionally
        #     for both branches the way `MiniMaxH3SetupStep()` used to be.
        #   - `MiniMaxH3AutoKeyframeVaeEncoderStep` (the conditional auto-block) is gone;
        #     this file already knows `is_fl2va` itself, so it calls the concrete
        #     `MiniMaxH3KeyframeVaeEncoderStep` (encoders.py) directly instead of a
        #     `references`/`image`-sniffing conditional wrapper.
        #   - `MiniMaxH3TextEncoderStep.encode_prompt` (the bare staticmethod this file used
        #     to call to get an un-@torch.no_grad()-wrapped encode) is gone, replaced by
        #     this file's own `_encode_h3_prompt` module helper (defined near the top of
        #     this file, next to `_unpatchify_video_tokens`), which builds the same
        #     presentation (tokenize prompt +, for fl2va, prepend a `"<Picture i>: "` label
        #     and vision block per keyframe -- matching `MiniMaxH3TextEncoderStep`/
        #     `MiniMaxH3FL2VATextEncoderStep.__call__` line-for-line, both read in full as
        #     part of this migration) and calls the new module function
        #     `get_qwen3vl_prompt_embeds` (encoders.py) for the actual conditioner forward.
        #   - fl2va's keyframe-conditioning noise+pack moved OUT of
        #     `MiniMaxH3PrepareLatentsStep` (found by cross-checking the old venv's
        #     `MiniMaxH3PrepareLatentsStep.__call__`, which used to fold in
        #     `condition_latents`/`audio_condition_latents` from state directly, against the
        #     new `MiniMaxH3PrepareLatentsStep.__call__`, which no longer reads either --
        #     this is a real behavioural contract change, not just a rename). Two new steps
        #     now carry that role, run only for `is_fl2va` and in this exact order relative
        #     to `MiniMaxH3PrepareLatentsStep` (matches `MiniMaxH3FL2VACoreDenoiseStep`'s own
        #     block order in modular_blocks_minimax_h3.py -- prepare_layout,
        #     prepare_condition_latents, prepare_latents, prepare_latents_fl2va):
        #     `MiniMaxH3PrepareConditionLatentsStep` noises+packs the keyframe VAE-encode
        #     step's raw `condition_latents` into `condition_rows` *before*
        #     `MiniMaxH3PrepareLatentsStep` draws the generated rows' own noise (draw order
        #     is part of what the request's generator reproduces), and
        #     `MiniMaxH3FL2VAPrepareLatentsStep` prepends `condition_rows` onto `latents`
        #     *after*.
        from diffusers.modular_pipelines.minimax_h3.before_denoise import (
            MiniMaxH3FL2VAPrepareLatentsStep,
            MiniMaxH3NoKeyframeAnchorsStep,
            MiniMaxH3PrepareConditionLatentsStep,
            MiniMaxH3PrepareLatentsStep,
            MiniMaxH3PrepareLayoutStep,
            MiniMaxH3SetTimestepsStep,
        )
        from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3ResizeStep
        from diffusers.modular_pipelines.minimax_h3.decoders import (
            MiniMaxH3AfterDenoiseStep,
            MiniMaxH3AudioDecodeStep,
            MiniMaxH3VideoDecodeStep,
        )
        from diffusers.modular_pipelines.minimax_h3.denoise import (
            MiniMaxH3DenoiseStep,
            MiniMaxH3LoopDenoiser,
            MiniMaxH3LoopSchedulerStep,
        )
        from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3KeyframeVaeEncoderStep
        from diffusers.modular_pipelines.modular_pipeline import PipelineState

        t_start = time.time()
        if still:
            if still_frames not in STILL_FRAME_CHOICES:
                raise ValueError(f"still_frames must be one of {STILL_FRAME_CHOICES}, got {still_frames}")
            if image is not None or last_image is not None:
                raise ValueError("still=True (t2i) はテキストからの生成専用です (image/last_image は併用不可)。")
            if upscale:
                raise ValueError("still=True (t2i) と upscale=1 (hires-fix) の併用は未検証のため拒否します。")
            if still_frames == 5 and not H3_VAE_SMALLCLIP_FIX:
                raise ValueError(
                    "still_frames=5 には H3_VAE_SMALLCLIP_FIX=1 (既定) が必要です "
                    "(潜在2フレームのデコードは上流のチャンク境界バグで落ちるため)。"
                )
            num_frames = still_frames
        else:
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
        if do_upscale:
            # 2026-08-13 のステップ掃引 (README 当日節) で確定した品質特性:
            #   パス2 = 1 step (turbo 既定4): 動く輪郭が激しく二重化
            #   パス2 = 3 step (8steps)     : 半透明のゴーストが残る
            #   パス2 = 4 step (12steps)    : クリーン (推奨)
            # sage は無関係 (SDPA でも同一の破綻を確認)。パッキング往復・x0/再ノイズ式は
            # マージ版スケジューラと一致済みで、律速は純粋にパス2のステップ数。
            # turbo なし 30steps (パス2=10) では別種の格子モザイクが出る (int8/投影TE との
            # 相互作用を疑うが未切り分け -- bf16+32B 時代の実測では出ていなかった)。
            n2_projected = num_inference_steps - max(
                1, min(num_inference_steps - 1, round(num_inference_steps * (1.0 - H3_HIRES_DENOISE)))
            )
            if n2_projected < 4:
                logger.warning(
                    "upscale=1: このステップ数 (%d) ではパス2が %d ステップしか取れず、"
                    "動く輪郭の二重化/ゴーストが出ます (実測)。turbo なら "
                    "num_inference_steps=12 (パス2=4) を推奨します。",
                    num_inference_steps, n2_projected,
                )
            if not instant["turbo"]:
                logger.warning(
                    "upscale=1 + turbo なしの組合せは、現在の既定構成 (int8/投影TE) で"
                    "格子状モザイクが出ることを実測済みです (原因未切り分け、2026-08-13)。"
                    "turbo=1 + 12steps を推奨します。"
                )
        if do_upscale and H3_LOWVRAM_ANY:
            # Not verified to fit: pass 2 runs full self-attention over a ~4x longer
            # packed sequence (~16x pass 1's attention activation cost), and neither
            # low-VRAM mode's steady state was sized with that much extra headroom in
            # mind (see H3_LOWVRAM's module comment). Fail loudly rather than risk an
            # OOM mid-request.
            raise ValueError(f"upscale=1 (hires-fix) is not supported with H3_LOWVRAM={H3_LOWVRAM_RAW!r}.")
        # Not part of this task's A/B scope (5/8/16/30-step single-pass only) and the
        # hires-fix branch's own FBC bookkeeping (`_fbc_last_step_was_skip()` calls
        # further down) is not guarded against turbo the way the single-pass path's is
        # -- rather than silently skip that bookkeeping too, reject the combination
        # until it is actually verified. Checked against the *resolved* (possibly
        # request-overridden) turbo value, not the raw H3_TURBO_LORA env-var default.
        settings.validate_instant_settings_for_upscale(instant, do_upscale)

        coexist_ref2va = False  # H3_KEEP_REF2VA 両常駐が成立したか (LOWVRAM=1 の分岐でだけ True になりうる)
        with self._load_lock:
            if H3_LOWVRAM:
                # This mode's whole point is TE (21GB) and transformer (34GB) are never
                # GPU-resident together (55GB already exceeds a 48GB-class card) -- so
                # unlike the branches below, do NOT call `_switch_to_variant`/
                # `_ensure_transformer` here: that would load the (int8) transformer
                # *before* TE, and TE has not even encoded the prompt yet. Just free
                # whichever big transformer happens to be resident (leftover from a
                # previous request -- lowvram's own steady state never leaves one
                # resident, but a mode-flag flip mid-process or a request that errored
                # out mid-denoise could) without loading a replacement; the transformer
                # is loaded further down, after TE has already finished encoding and
                # been freed again.
                #
                # H3_KEEP_TRANSFORMER=1: skip freeing `transformer` here so it stays
                # GPU-resident across requests (the whole point of the flag -- see its
                # module comment for the VRAM budget derivation). Only safe because the
                # flag's own import-time guard requires H3_TE_DEVICE to be set, i.e. TE
                # loads onto a *different* GPU than transformer, so the encode phase
                # below (`_load_text_encoder`) does not have to share this GPU's budget
                # with TE at all. `transformer_ref` is still freed unconditionally (ref2va
                # is out of scope for this flag -- it uses a different transformer_ref
                # residency path entirely, see `generate_ref2va`). `_active_variant` is
                # deliberately left alone (not reset to None): if `transformer` is still
                # resident from the previous request, it is still the t2va one (this
                # branch never loads transformer_ref), so "t2va" remains correct; if
                # nothing is resident yet (first request), `_ensure_transformer` below
                # sets it to "t2va" itself once it loads.
                if not H3_KEEP_TRANSFORMER:
                    self._free_transformer()
                    self._active_variant = None
                # H3_KEEP_REF2VA=1 + 空きVRAM十分: ref2va スタックを解放せず base を同居させる
                # (`_keep_ref2va_coexist` の docstring / H3_KEEP_REF2VA_COEXIST のモジュールコメント)。
                self._drain_deferred_decode("generate")
                coexist_ref2va = self._keep_ref2va_coexist("generate")
                if not coexist_ref2va:
                    self._free_transformer_ref()
                self._ensure_vaes(progress)
                self._load_text_encoder(progress)
            elif H3_LOWVRAM_GROUP:
                # Unlike `H3_LOWVRAM=1`, this mode's (group-offloaded) transformer is
                # cheap to have GPU-adjacent -- it lives on CPU and only ~1-2 blocks
                # (~1.4GB) ever visit GPU at a time, so TE-nf4 (21GB) + a resident
                # group-offloaded transformer do not compete for VRAM the way TE(21GB) +
                # a *fully* GPU-resident int8 transformer(34GB) would. So, same shape as
                # the plain `bnb-4bit`/`none` branch below: `_switch_to_variant` first
                # (frees transformer_ref if that was the last-used variant, then loads/
                # confirms `transformer` resident -- a cheap no-op via
                # `_ensure_transformer_group`'s early-return if it already is), then TE.
                self._switch_to_variant("t2va", progress)
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
        # Must run before this request's `MiniMaxH3SetTimestepsStep` call (further down,
        # inside the mode-specific branches below) -- see `_apply_turbo_video_shift`'s
        # own docstring for why per-request rather than process-wide.
        self._apply_turbo_video_shift(instant["turbo"], is_ref=False)

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

        # --- setup (canvas / keyframe prep) ---
        # PR #14355 note: there is no longer one `MiniMaxH3SetupStep` shared by both
        # branches. `MiniMaxH3ResizeStep` (before_encoder.py) now owns fl2va's own half --
        # putting the keyframes onto the target canvas and resolving `keyframe_anchors`
        # (`"first"`/`"last"`) -- while `MiniMaxH3NoKeyframeAnchorsStep` (before_denoise.py)
        # is t2va's declaration that it anchors none (`MiniMaxH3CoreDenoiseStep`'s own
        # first block, per modular_blocks_minimax_h3.py). Neither step validates duration
        # or defaults an unset canvas any more -- both moved into
        # `MiniMaxH3PrepareLayoutStep.__call__` itself (before_denoise.py), which now runs
        # later in this method regardless of branch, so `_relaxed_min_duration()`'s scope
        # moves there too (see that call site below) instead of wrapping a setup step here.
        if is_fl2va:
            resize_step = MiniMaxH3ResizeStep()
            _, state = resize_step(pipe, state)
        else:
            no_anchors_step = MiniMaxH3NoKeyframeAnchorsStep()
            _, state = no_anchors_step(pipe, state)
        keyframes = state.get("keyframes")

        # --- text encode (still has text_encoder on GPU at this point) ---
        if progress:
            progress.update(phase="encoding", message="プロンプトをエンコード中...")
        # `_encode_h3_prompt` (this file's own helper, replacing the retired
        # `MiniMaxH3TextEncoderStep.encode_prompt` bare staticmethod) is not wrapped in
        # `@torch.no_grad()` internally, matching the old staticmethod's own contract --
        # the `@torch.no_grad()` lives on the *block*'s `__call__`, which both this call
        # and the old one bypass. Without no_grad here the autograd graph pins ~50GB of TE
        # weights on GPU past the free below (observed on the first probe run).
        with self._te_attached(), torch.no_grad():
            prompt_embeds, text_token_tags = _encode_h3_prompt(
                pipe, prompt, keyframes or None, device=self._encode_device, dtype=torch.bfloat16
            )
        # TE 外部常駐のときはここで計算用GPUへ運ぶ(約42MB)。既定構成では no-op。
        prompt_embeds, text_token_tags = self._to_compute_device(prompt_embeds, text_token_tags)
        state.set("prompt_embeds", prompt_embeds)
        state.set("text_token_tags", text_token_tags)
        # フェーズ境界での中断チェック(encoding): テキストエンコード自体が終わった直後。
        # この先は layout/latents/timesteps の準備と transformer(_ref) のロードが続き、
        # それぞれ個別のフェーズ境界チェック(loading_transformer 側、denoising 直前)で
        # 別途カバーされる。
        interrupt_controller.check()

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
        force_free_te = TE_QUANT == "bnb-4bit" and not H3_LOWVRAM_ANY and (
            do_upscale or (H3_TRANSFORMER_BOTH_RESIDENT and self._transformer_ref_loaded)
        )

        # H3_LOWVRAM_GROUP's normal design (see the branch below and its own module
        # comment) keeps TE-nf4 resident straight through denoise -- correct at its
        # original ~21GB TE size, where 32GB-class ballast testing measured this fitting
        # with room to spare (28.67GB peak, see README). H3_TE_PRUNE shrinks TE to
        # ~17.45GB, which sounded like it should only make this mode's headroom bigger,
        # but this task's own 22GB/24GB ballast testing found the OPPOSITE is what
        # matters at the 24GB-class floor this mode is meant to reach: pruned TE
        # (17.45GB) + the group-offloaded transformer's own resident blocks + denoise
        # activations still leaves only ~2GB of slack at a 24GB budget, and reproducibly
        # OOMs 1 step into denoise ("Tried to allocate 1.16 GiB" with 23.12GB already in
        # use against a 24GB ballast). Pruning alone was not enough to cross the 24GB
        # line this mode's docs already draw at 32GB -- so H3_TE_PRUNE=1 additionally
        # borrows H3_LOWVRAM=1's own choreography (force-free TE for the denoise loop,
        # reload it after) for this mode specifically, verified by this task's own
        # 24GB-ballast retest to fix the OOM (see H3_TE_PRUNE's own module comment for
        # the full measurement table). Unpruned H3_LOWVRAM_GROUP (H3_TE_PRUNE=0, the
        # default) is completely unaffected -- this flag is only ever True when both
        # H3_LOWVRAM_GROUP and H3_TE_PRUNE are set.
        group_free_te_for_denoise = H3_LOWVRAM_GROUP and H3_TE_PRUNE

        if H3_LOWVRAM_GROUP:
            # Transformer is already resident (loaded/confirmed in the entry section
            # above, via `_switch_to_variant` -> `_ensure_transformer` ->
            # `_ensure_transformer_group`) -- unlike `H3_LOWVRAM=1`, group mode's
            # transformer does not need to be deferred behind TE/vae headroom concerns,
            # since its GPU footprint during any of these steps is tiny (no big matmuls
            # run yet, and even once denoise starts only ~1-2 blocks are ever
            # GPU-resident at once). So this can run keyframe/layout/latents/timesteps
            # exactly like plain `bnb-4bit` t2va mode's own steady state (transformer +
            # TE both already resident), for both t2va AND fl2va (unlike plain
            # `bnb-4bit`, which routes fl2va through its own `is_fl2va` branch above
            # specifically to defer the transformer's ~66GB/34GB load past the keyframe
            # vae-encode step -- unnecessary here since the transformer was never a big
            # *GPU* load in the first place).
            # PR #14355 note: `MiniMaxH3AutoKeyframeVaeEncoderStep` (the conditional
            # auto-block that used to skip itself for t2va) is gone -- this file already
            # knows `is_fl2va`, so it gates the concrete `MiniMaxH3KeyframeVaeEncoderStep`
            # itself. Required here (not just a redundant optimization): that step's
            # `keyframes` input is `required=True` with no default, so calling it
            # unconditionally would raise `ValueError` for every t2va request.
            if is_fl2va:
                keyframe_step = MiniMaxH3KeyframeVaeEncoderStep()
                _, state = keyframe_step(pipe, state)
                self._vae_to_cpu()

            with _relaxed_min_duration() if still else _NullContext():
                layout_step = MiniMaxH3PrepareLayoutStep()
                _, state = layout_step(pipe, state)
            # PR #14355 note: fl2va's keyframe-conditioning noise+pack is no longer folded
            # into `MiniMaxH3PrepareLatentsStep` itself (that step now only ever draws/packs
            # the *generated* rows' noise, for every task) -- it moved to two new steps that
            # only apply when there is conditioning to noise: `MiniMaxH3PrepareConditionLatentsStep`
            # (noises + packs the raw `condition_latents` the keyframe VAE-encode step above
            # produced, *before* `MiniMaxH3PrepareLatentsStep` draws the generated rows' own
            # noise -- draw order is part of what the request's generator reproduces) and
            # `MiniMaxH3FL2VAPrepareLatentsStep` (prepends the now-noised `condition_rows` to
            # `latents` *after* -- mirrors `MiniMaxH3FL2VACoreDenoiseStep`'s own block order
            # in modular_blocks_minimax_h3.py: prepare_layout, prepare_condition_latents,
            # prepare_latents, prepare_latents_fl2va).
            if is_fl2va:
                condition_latents_step = MiniMaxH3PrepareConditionLatentsStep()
                _, state = condition_latents_step(pipe, state)
            latents_step = MiniMaxH3PrepareLatentsStep()
            _, state = latents_step(pipe, state)
            if is_fl2va:
                fl2va_latents_step = MiniMaxH3FL2VAPrepareLatentsStep()
                _, state = fl2va_latents_step(pipe, state)
            timesteps_step = MiniMaxH3SetTimestepsStep()
            _, state = timesteps_step(pipe, state)
        elif H3_LOWVRAM:
            # fl2va's keyframe step (if any) runs here too, while TE is still resident
            # (harmless -- it only touches vae/scheduler, not TE) and `vae` is already
            # on GPU from the `is_fl2va` block above this function's setup section.
            if is_fl2va:
                keyframe_step = MiniMaxH3KeyframeVaeEncoderStep()
                _, state = keyframe_step(pipe, state)
                self._vae_to_cpu()

            if self._te_external:
                # TE 外部常駐 (H3_TE_DEVICE): TE は別GPUにあるので、計算用GPUのヘッドルームを
                # 空けるための解放は不要 -- 先に transformer をロードしてしまう。ただし
                # `_execution_device` は components 順で最初の nn.Module (= 別GPU上の TE) を
                # 拾ってしまうため、layout/latents/timesteps は
                # `_pin_execution_device_to_compute()` の窓の中で回す (その間だけ
                # text_encoder と vae をパイプから外し、transformer が最初に見つかるようにする)。
                with self._load_lock:
                    self._ensure_transformer(progress)
                with self._pin_execution_device_to_compute():
                    with _relaxed_min_duration() if still else _NullContext():
                        layout_step = MiniMaxH3PrepareLayoutStep()
                        _, state = layout_step(pipe, state)
                    # See the `H3_LOWVRAM_GROUP` branch above for why fl2va needs these two
                    # extra steps now (condition-noise moved out of
                    # `MiniMaxH3PrepareLatentsStep`).
                    if is_fl2va:
                        condition_latents_step = MiniMaxH3PrepareConditionLatentsStep()
                        _, state = condition_latents_step(pipe, state)
                    latents_step = MiniMaxH3PrepareLatentsStep()
                    _, state = latents_step(pipe, state)
                    if is_fl2va:
                        fl2va_latents_step = MiniMaxH3FL2VAPrepareLatentsStep()
                        _, state = fl2va_latents_step(pipe, state)
                    timesteps_step = MiniMaxH3SetTimestepsStep()
                    _, state = timesteps_step(pipe, state)
            else:
                # --- layout / latents / timesteps, run NOW (TE still GPU-resident) ---
                # `_execution_device` resolves via `text_encoder` (still resident on GPU)
                # here, exactly like every non-lowvram bnb-4bit branch's own
                # `force_free_te`-deferred ordering achieves -- see the long comment above.
                with _relaxed_min_duration() if still else _NullContext():
                    layout_step = MiniMaxH3PrepareLayoutStep()
                    _, state = layout_step(pipe, state)
                # See the `H3_LOWVRAM_GROUP` branch above for why fl2va needs these two extra
                # steps now (condition-noise moved out of `MiniMaxH3PrepareLatentsStep`).
                if is_fl2va:
                    condition_latents_step = MiniMaxH3PrepareConditionLatentsStep()
                    _, state = condition_latents_step(pipe, state)
                latents_step = MiniMaxH3PrepareLatentsStep()
                _, state = latents_step(pipe, state)
                if is_fl2va:
                    fl2va_latents_step = MiniMaxH3FL2VAPrepareLatentsStep()
                    _, state = fl2va_latents_step(pipe, state)
                timesteps_step = MiniMaxH3SetTimestepsStep()
                _, state = timesteps_step(pipe, state)

                # Only now is it safe to free TE and load the (int8) transformer: every
                # tensor that would have needed `_execution_device` to resolve correctly
                # already exists, materialized on the right device, on `state`.
                # transformer ロード (prequant キャッシュでも ~4.2s、途中中断不可) に入る前に
                # 中断を拾う。r-n-v の会話ターンが待機 (fl2va) の「VAE 退避 + base ロード」
                # 窓 (~6.5s、2026-10-07 実測 turn@8) と衝突したときの待ちを縮めるため。
                interrupt_controller.check()
                with self._load_lock:
                    if coexist_ref2va:
                        # TE も ref2va スタックの一部 (次の ref2va が再ロードを払わないよう残す)。
                        # encode は済んでおり、収支は _keep_ref2va_coexist で確認済み。
                        logger.info("generate: text_encoder を解放せず常駐のまま base transformer をロード (H3_KEEP_REF2VA 両常駐)")
                    elif H3_FL2VA_KEEP_TE:
                        # 部分的 keep (H3_FL2VA_KEEP_TE のモジュールコメント参照):
                        # coexist 不成立の低VRAM構成でも、TE (TE_STREAM 後 ~4GB) だけは
                        # 残して次の会話ターンの TE 再ロード (~15s) と ref-prefix 再
                        # エンコードを省く。ref/transformer_ref は従来どおり解放済み。
                        logger.info("generate: H3_FL2VA_KEEP_TE=1 -- text_encoder を解放せず base transformer をロード")
                    else:
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
            # (`is_fl2va` is always True on this branch, so `MiniMaxH3KeyframeVaeEncoderStep`
            # runs unconditionally here -- see the `H3_LOWVRAM_GROUP`/`H3_LOWVRAM` branches
            # above for why other branches gate this on `is_fl2va` instead.)
            keyframe_step = MiniMaxH3KeyframeVaeEncoderStep()
            _, state = keyframe_step(pipe, state)
            self._vae_to_cpu()
            with self._load_lock:
                self._ensure_transformer(progress)

            # --- layout / latents / timesteps ---
            # `still` is always False here (fl2va + still=True is rejected earlier in this
            # function), so `_relaxed_min_duration()` never actually triggers on this
            # branch -- wrapped anyway for the same reason every other branch is, so the
            # scoping rule ("min_duration only relaxed around MiniMaxH3PrepareLayoutStep")
            # holds uniformly across all branches rather than as a special case.
            # `_te_external` のときは catch-all 分岐と同じ理由でピン窓が要る(そちらの
            # コメント参照)。transformer は直上でロード済みなので前提を満たす。
            with self._pin_execution_device_to_compute() if self._te_external else _NullContext():
                with _relaxed_min_duration() if still else _NullContext():
                    layout_step = MiniMaxH3PrepareLayoutStep()
                    _, state = layout_step(pipe, state)
                # `is_fl2va` is always True on this branch (see the comment above), so these
                # two run unconditionally -- see the `H3_LOWVRAM_GROUP` branch's own comment
                # for why fl2va needs them now.
                condition_latents_step = MiniMaxH3PrepareConditionLatentsStep()
                _, state = condition_latents_step(pipe, state)
                latents_step = MiniMaxH3PrepareLatentsStep()
                _, state = latents_step(pipe, state)
                fl2va_latents_step = MiniMaxH3FL2VAPrepareLatentsStep()
                _, state = fl2va_latents_step(pipe, state)
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

            # --- keyframe VAE conditioning (fl2va only; vae already permanently resident
            # in `none` mode, already brought to GPU by the `is_fl2va` block earlier in
            # this method for `bnb-4bit` t2va/fl2va) --- this branch serves both t2va and
            # fl2va (it is the catch-all "everything else" arm), so -- same reasoning as
            # the `H3_LOWVRAM`/`H3_LOWVRAM_GROUP` branches above -- the now-required
            # `keyframes` input of `MiniMaxH3KeyframeVaeEncoderStep` means this has to be
            # gated on `is_fl2va` rather than called unconditionally.
            if is_fl2va:
                keyframe_step = MiniMaxH3KeyframeVaeEncoderStep()
                _, state = keyframe_step(pipe, state)

            # --- layout / latents / timesteps ---
            # `_te_external` (H3_TE_DEVICE): この分岐の前提「TE が常駐しているので
            # `_execution_device` は text_encoder 経由で正しく解決する」が崩れる --
            # 外部常駐の TE はパイプからデタッチされているため、スキャンが CPU 常駐の
            # audio_vae に落ち、レイアウトの position_ids が CPU に作られて rope() 内で
            # device mismatch になる(2026-08-12、96GB機の plain モード + H3_TE_DEVICE で
            # 初めてこの組み合わせが叩かれ実機再現。従来 H3_TE_DEVICE は H3_LOWVRAM=1 と
            # のみ併用されており、この分岐は未カバーだった)。transformer は直上の
            # `_ensure_transformer` でロード済みなので、lowvram=1 の te_external 分岐と
            # 同じピン窓がそのまま前提を満たす。通常構成(TE 同居)では従来どおり
            # 窓なし -- 検証済み経路をバイト単位で変えないため。
            with self._pin_execution_device_to_compute() if self._te_external else _NullContext():
                with _relaxed_min_duration() if still else _NullContext():
                    layout_step = MiniMaxH3PrepareLayoutStep()
                    _, state = layout_step(pipe, state)
                # See the `H3_LOWVRAM_GROUP` branch above for why fl2va needs these two extra
                # steps now (condition-noise moved out of `MiniMaxH3PrepareLatentsStep`).
                if is_fl2va:
                    condition_latents_step = MiniMaxH3PrepareConditionLatentsStep()
                    _, state = condition_latents_step(pipe, state)
                latents_step = MiniMaxH3PrepareLatentsStep()
                _, state = latents_step(pipe, state)
                if is_fl2va:
                    fl2va_latents_step = MiniMaxH3FL2VAPrepareLatentsStep()
                    _, state = fl2va_latents_step(pipe, state)
                timesteps_step = MiniMaxH3SetTimestepsStep()
                _, state = timesteps_step(pipe, state)

        # `num_frames` is only resolved (aligned to `17*n+5`) by `MiniMaxH3PrepareLayoutStep`
        # now, which runs inside the branch above -- moved here (out of every branch) so it
        # reads the post-layout_step value regardless of which branch ran. PR #14355 note:
        # this used to be read right after the old shared `MiniMaxH3SetupStep()` call,
        # which no longer exists (see the "setup (canvas / keyframe prep)" comment above).
        actual_num_frames = state.get("num_frames")

        # --- denoise loop, instrumented for progress polling ---
        # denoise 開始直前にも1回チェックしておく(ループ内は timed_loop_step 側で
        # 毎ステップ境界チェック済み -- ここは「まだ1歩も進んでいない」状態での保険)。
        interrupt_controller.check()
        if progress:
            progress.update(phase="denoising", step=0, total_steps=num_inference_steps, message="デノイズ中...")
        t_denoise = time.time()
        step_times = []
        cache_skips = [0]
        pass1_time = None
        interpolate_time = None
        pass2_time = None
        # Canonical (post-layout) resolution -- `MiniMaxH3PrepareLayoutStep` (t2va) /
        # `MiniMaxH3ResizeStep` (fl2va, via keyframes' own aspect ratio) resolve `None` and
        # snap to the canvas rules, so this is not necessarily identical to the raw
        # `height`/`width` args.
        out_height, out_width = state.get("height"), state.get("width")

        # Instant-apply this request's cache/attn/turbo settings now that `transformer`
        # is confirmed resident in every branch above (this is the first point after the
        # if/elif/else above where that is unconditionally true) -- no reload, see
        # `apply_instant_settings()`'s own docstring. Every gate from here on in this
        # method reads `instant["cache"]`/`instant["turbo"]` (the *resolved*,
        # possibly-request-overridden values), not the raw H3_CACHE/H3_TURBO_LORA
        # globals, so a request that left these fields unset still gets exactly the
        # same behaviour as before this feature existed (resolve_instant_settings()
        # already folded the current globals in as the default).
        self.apply_instant_settings(self._pipe.transformer, instant, is_ref=False, progress=progress)

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

        if group_free_te_for_denoise:
            # H3_LOWVRAM_GROUP + H3_TE_PRUNE only (see `group_free_te_for_denoise`'s own
            # comment above for why this is needed at the 24GB-class floor). Safe here
            # for the identical reason `force_free_te`'s own free (just above) is safe at
            # this exact point: layout_step/latents_step/timesteps_step have already run,
            # so every tensor whose creation needed `_execution_device` to resolve
            # correctly already exists, materialized on the right device -- freeing
            # text_encoder from here on cannot change what `_execution_device` resolves
            # to for the remainder of this request. `vae` -- normally the very next
            # `nn.Module` `_execution_device` would fall through to once text_encoder is
            # gone (see the long comment earlier in this method on `text_encoder,
            # tokenizer, processor, vae, ...` insertion order) -- does not create a
            # device-mismatch risk here the way it did for the (rejected-much-earlier,
            # `H3_LOWVRAM=1`-only) upscale/fl2va orderings that comment warns about: t2va
            # never touches `vae` again until the decode section below, well after the
            # denoise loop's own device usage is already locked in by the transformer's
            # own `.device`, not `_execution_device`, inside `MiniMaxH3DenoiseStep`.
            # Reloaded around the decode window exactly like every other
            # H3_LOWVRAM_GROUP request already does unconditionally (see the decode
            # section's own `elif H3_LOWVRAM_GROUP:` branch) -- that reload is not
            # gated on this flag, so it fires regardless and restores the
            # transformer(group-offloaded)+TE-nf4(pruned) steady state for the next
            # request either way.
            with self._load_lock:
                self._free_text_encoder(force=True)

        if not do_upscale:
            denoise_step = MiniMaxH3DenoiseStep()
            orig_loop_step = denoise_step.loop_step

            def timed_loop_step(components, bstate, i, t):
                # ステップ境界での中断チェック(_InterruptController の docstring 参照)。
                # 前のステップが完全に終わった直後・次のステップの計算に入る前。
                interrupt_controller.check()
                ts = time.time()
                result = orig_loop_step(components, bstate, i=i, t=t)
                step_times.append(time.time() - ts)
                if instant["effective_cache"] == "fbc":
                    cache_skips[0] += self._fbc_last_step_was_skip()
                if progress:
                    progress.update(step=i + 1, message=f"デノイズ中 {i + 1}/{num_inference_steps}")
                return result

            denoise_step.loop_step = timed_loop_step
            if instant["effective_cache"] == "fbc":
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
                fbc_cm = _fbc_reset_and_context() if instant["effective_cache"] == "fbc" else None
                cm = fbc_cm if fbc_cm is not None else _NullContext()
                with cm:
                    for i in range(i_start, i_end):
                        # ステップ境界での中断チェック(_InterruptController の docstring
                        # 参照)。hires-fix はループを自前で回す唯一の経路のため、他の
                        # timed_loop_step 系サイトと同じ規約でここにも入れる。
                        interrupt_controller.check()
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
                        if instant["effective_cache"] == "fbc":
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

        # PR #14355 note: the old `MiniMaxH3VideoDecodeStep` used to unpatchify the
        # denoised video rows internally before decoding. The new decode contract splits
        # that out into its own step, `MiniMaxH3AfterDenoiseStep` (decoders.py): it drops
        # the leading conditioning rows the loop never wrote (`num_condition_video_rows`/
        # `num_condition_audio_rows`, both 0 for every path this function drives -- fl2va's
        # keyframe rows and hires-fix are both t2va/fl2va-only, never carrying ref2va-style
        # conditioning rows past this point) and reshapes `latents`/`audio_latents` from
        # packed rows back into the 5D video tensor / channel-major audio tensor
        # `MiniMaxH3VideoDecodeStep`/`MiniMaxH3AudioDecodeStep` now expect as input. This
        # has to run once, after the whole denoise loop (single-pass or the hires-fix
        # pass1+pass2 splice, both converge on `state` by this point), and before either
        # decode step below -- matches where `after_denoise` sits in
        # `MiniMaxH3CoreDenoiseStep`/`MiniMaxH3FL2VACoreDenoiseStep`'s own block list
        # (modular_blocks_minimax_h3.py), right before the separate `MiniMaxH3DecodeStep`.
        after_denoise_step = MiniMaxH3AfterDenoiseStep()
        _, state = after_denoise_step(pipe, state)

        if os.environ.get("H3_DEBUG_MEM_DIAG") == "1":
            _log_gpu_tensor_diag("post-denoise, pre-decode (t2va)")

        # denoise 後〜mux までにチェックが無いと、中断要求が届いても decode+vocoder+mux
        # (~4s)を完走するまで GPU ロックを握り続ける。r-n-v の会話ターンが待機クリップ
        # (fl2va)の追い生成と衝突したとき、初回チャンクが丸ごとこの分だけ遅れることを
        # 実測で確認した(2026-10-07: 衝突ターンは 9.6〜9.9s wall、非衝突は 5.0〜6.5s)。
        interrupt_controller.check()

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
        # `H3_LOWVRAM_GROUP`: the transformer itself is left alone here (unlike every
        # other bnb-4bit branch) -- the group-offloaded transformer's *actual* GPU
        # footprint is already tiny (~1-2 blocks, ~1.4GB) regardless of decode's own
        # VAE-pair trip, so there is no transformer-vs-vae headroom conflict to resolve
        # here in the first place, and freeing it would mean paying its ~34GB CPU load +
        # int8 quantize cost (~35-70s, see README) on every single request instead of
        # once at process start -- exactly the per-request churn this mode's "load once,
        # keep forever" design (see `_ensure_transformer_group`'s docstring) exists to
        # avoid.
        #
        # TE-nf4 (~21GB) is a DIFFERENT story and DOES need to be freed here, force=True,
        # even though `force_free_te` (computed above) is False for this mode: this was
        # found, not assumed, via this task's own 32GB-ballast investigation using
        # `_log_gpu_tensor_diag()` (H3_DEBUG_MEM_DIAG=1) -- the initial guess that a
        # plain `empty_cache()` would be enough (reserved-but-idle allocator cache) was
        # WRONG. The diagnostic showed only ~22.25GB of genuinely *live* (referenced)
        # CUDA tensors at this point, dominated by two 1.556GB `(151936, 5120)` bf16
        # tensors -- TE-nf4's own embedding table / tied lm_head weight (151936 = Qwen3
        # tokenizer vocab size, 5120 = text_dim) -- i.e. TE-nf4's own ~21GB footprint
        # (kept resident throughout group mode's t2va path, since `force_free_te` is
        # False here) is the actual culprit, not fragmentation. TE-nf4(21GB) +
        # decode-only peak(~16.3GB, measured directly via
        # scripts/probe_vae_tile_size.py, and found NOT to shrink with a smaller VAE
        # tile size -- the decode buffer's size is independent of spatial tiling) = 37GB,
        # already over a 32GB-class card's budget before the group-offloaded
        # transformer's own tiny footprint is even counted. Freeing TE for this decode
        # window (and reloading it right after, mirroring the "restore steady state
        # right before the next request needs it" shape `force_free_te`'s own reload
        # already uses elsewhere in this file) is the fix -- same bounded "short window"
        # pattern as every other TE/transformer cycle in this file, not a new pattern.
        def _restore_decode_steady_state():
            # bnb-4bit mode: park the VAEs back on CPU, then reload the transformer that
            # was dropped for the decode window, restoring the transformer+TE-nf4 steady
            # state this mode keeps between requests. No-op in `none` mode (nothing was
            # dropped for decode in that mode). 正常系だけでなく decode 例外時にも呼ぶ
            # (下の try/except): 超短尺プローブ (2026-08-07) で「decode 例外 →
            # transformer drop 済み・復元未実行のまま残留 → 後続リクエストが連鎖 OOM」を
            # 実機再現したため、復元は例外経路でも必須(README「超短尺生成プローブ」)。
            self._vae_to_cpu()
            if TE_QUANT == "bnb-4bit" and not H3_LOWVRAM_ANY:
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
            elif H3_LOWVRAM_GROUP:
                # The transformer was never touched around decode in this mode (see the
                # decode section's own comment), only TE-nf4 was force-freed there to make
                # room for the vae pair -- reload it now to restore the
                # transformer(group-offloaded)+TE-nf4 steady state this mode keeps between
                # requests (unlike `H3_LOWVRAM=1` just below, this mode's transformer is
                # cheap enough to always keep ready, so there is no reason to leave TE
                # unloaded between requests either -- the *next* request needs TE first
                # regardless of which big model "waits", and reloading it now means the next
                # request does not pay TE's ~15-40s reload cost on its own critical path).
                with self._load_lock:
                    self._load_text_encoder(progress)
            # H3_LOWVRAM: deliberately do NOT reload the transformer here. This mode's
            # steady state between requests is "nothing big resident" (see the H3_LOWVRAM
            # module comment) -- the *next* request needs TE first, not transformer, so
            # preloading it now would just be evicted again at that request's own encode
            # phase for no benefit, and would leave a 34GB resident model sitting idle
            # between requests on a card that cannot spare it.
            #
            # H3_KEEP_TRANSFORMER=1 also falls through this same H3_LOWVRAM branch (it
            # does nothing here) and that is correct *by construction*: this flag never
            # freed `transformer` in the first place (see the decode-phase skip just
            # below), so there is nothing to restore -- it was never dropped. Confirmed by
            # reading this closure: the only other branches that touch `transformer` here
            # are the `bnb-4bit and not H3_LOWVRAM_ANY` branch (`none`/plain `bnb-4bit`
            # modes, not lowvram) and `H3_LOWVRAM_GROUP`'s TE-only branch -- neither
            # applies when H3_LOWVRAM=1, which this flag requires.

        # H3_KEEP_TRANSFORMER=1: skip freeing `transformer` for the decode window too --
        # this is the flag's actual payoff (H3_LOWVRAM=1 otherwise pays the ~14.8-32.7s
        # reload cost on *every* request just to make room for the VAE pair's decode
        # peak). Only safe because the import-time guard already forced
        # H3_VIDEO_VAE_FP16=1: transformer-int8 34.3GB + fp16 decode peak ~11.4GB =
        # 45.7GB fits the ~49.8GB effective budget (RESIDENCY.md §5.5). With the fp32 VAE
        # decode peak (~16.29GB) this would be 50.6GB and would NOT fit -- which is
        # exactly why H3_VIDEO_VAE_FP16=1 is a hard requirement of this flag, rejected at
        # import time rather than left to fail here mid-request.
        if H3_KEEP_TRANSFORMER:
            pass
        elif TE_QUANT == "bnb-4bit" and not H3_LOWVRAM_GROUP:
            self._free_transformer()
        elif H3_LOWVRAM_GROUP:
            with self._load_lock:
                self._free_text_encoder(force=True)
        self._vae_to_gpu()
        t_decode = time.time()
        try:
            # デバッグ専用・一回限り: decode 失敗時のクリーンアップ経路 (下の except) を
            # 実機 E2E で通すための人為的な失敗注入。`pop` なので同一プロセスの次の
            # リクエストからは通常動作に戻る(= 「失敗 → 次リクエストが正常に通る」の
            # 復旧シナリオをサーバ再起動なしで検証できる)。通常運用では未設定。
            if os.environ.pop("H3_DEBUG_FAIL_DECODE", None) == "1":
                raise RuntimeError("H3_DEBUG_FAIL_DECODE=1: intentional decode failure (one-shot, cleanup-path test)")
            video_decode_step = _cpu_norm_video_decode_step()
            _, state = video_decode_step(pipe, state)
            audio_decode_step = MiniMaxH3AudioDecodeStep()
            _, state = audio_decode_step(pipe, state)
            decode_time = time.time() - t_decode

            videos = state.get("videos")
            audio = state.get("audio")
            sampling_rate = state.get("sampling_rate")

            video_tensor = videos[0] if isinstance(videos, list) else videos
            # 全長ぶんの中間テンソルを GPU に積まないよう、フレームを小分けにして
            # CPU の出力配列へ直接書き込む (frames_to_uint8 の docstring 参照)。
            frames_uint8 = frames_to_uint8(video_tensor)
            audio_np = audio[0].float().cpu().numpy()
            rms = float(np.sqrt(np.mean(audio_np**2)))
            peak = float(np.max(np.abs(audio_np)))

            peak_vram = torch.cuda.max_memory_allocated() / 1e9

            # free the big activation buffers before muxing (CPU-bound, no need to hold onto GPU tensors)
            del video_tensor, videos, audio
            gc.collect()
            torch.cuda.empty_cache()
        except BaseException:
            logger.exception(
                "decode failed -- freeing partial buffers and restoring the steady state "
                "before re-raising (so the next request does not inherit a corrupted resident set)"
            )
            gc.collect()
            torch.cuda.empty_cache()
            try:
                _restore_decode_steady_state()
            except Exception:
                # 復元自体の失敗で元の decode 例外を潰さない(原因情報は元例外側にある)。
                logger.exception("steady-state restore after decode failure also failed")
            raise

        _restore_decode_steady_state()

        # decode 完了直後にも中断を拾う(上の pre-decode チェックと同じ理由。
        # decode 中に届いた中断要求はここで初めて観測できる — mux は CPU 処理だが
        # 完走まで生成ロックを握るため、ここで打ち切れば次ジョブが即座に通る)。
        interrupt_controller.check()

        if progress:
            progress.update(phase="muxing", message="mp4へmux中...")
        if still:
            mode = "t2i"
        else:
            mode = "fl2va" if (image is not None or last_image is not None) else "t2va"
        job_stub = f"{mode}_{int(t_start)}"
        mp4_path = self.output_dir / f"{job_stub}.mp4"
        _mux_mp4(frames_uint8, audio_np, sampling_rate, FPS, mp4_path, mute=mute)

        # 静止画モード: 中央フレームを PNG として書き出す(超短尺 mp4 も上で保存済み。
        # 別フレームを選び直したいときは mp4 から取り出せる)。
        png_path = None
        still_frame_index = None
        if still:
            still_frame_index = len(frames_uint8) // 2
            png_path = self.output_dir / f"{job_stub}.png"
            Image.fromarray(frames_uint8[still_frame_index]).save(png_path)

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
            "still": int(still),
            "still_frames": still_frames if still else None,
            "still_frame_index": still_frame_index,
            "png_path": str(png_path) if png_path else None,
            "png_filename": png_path.name if png_path else None,
            "total_elapsed_s": round(time.time() - t_start, 2),
            "mode": mode,
            "te_quant": TE_QUANT,
            "transformer_quant": H3_TRANSFORMER_QUANT,
            "lowvram": H3_LOWVRAM_RAW,
            # Instant-apply settings actually used for this request (resolved --
            # reflects any per-request cache/cache_threshold/attn/turbo override, not
            # just the process-wide env-var defaults). "cache_mode" mirrors the old
            # field name/shape (a human-readable string noting the turbo force-off)
            # for backward compatibility with any existing caller parsing this field;
            # "cache"/"turbo"/"attn_backend" below are the new, directly-machine-
            # readable equivalents.
            "attn_backend": instant["attn"],
            "cache_mode": instant["cache"] if not instant["turbo"] else "none (force-disabled by turbo=1)",
            "cache": instant["effective_cache"],
            "cache_threshold": instant["cache_threshold"] if instant["effective_cache"] == "fbc" else None,
            "turbo_lora": instant["turbo"],
            "turbo": instant["turbo"],
            "mute": bool(mute),
            "upscale": int(do_upscale),
            "hires_denoise": H3_HIRES_DENOISE if do_upscale else None,
            "pass1_steps": n1 if do_upscale else None,
            "pass2_steps": (actual_steps - n1) if do_upscale else None,
            "pass1_time_s": round(pass1_time, 2) if pass1_time is not None else None,
            "interpolate_time_s": round(interpolate_time, 3) if interpolate_time is not None else None,
            "pass2_time_s": round(pass2_time, 2) if pass2_time is not None else None,
            # Number of denoise steps where FBC skipped the tail blocks (cache hit).
            # Always 0 in `none`/turbo mode (the counter never increments there).
            "cache_skipped_steps": cache_skips[0] if instant["effective_cache"] == "fbc" else None,
        }
        if progress:
            progress.update(phase="done", message="完了", result_path=str(png_path) if png_path else str(mp4_path))
        logger.info("generation done: %s", json.dumps({k: v for k, v in result.items() if k != "ram"}, ensure_ascii=False))
        return result

    # ------------------------------------------------------------------
    # 静止画バッチ生成 (t2i_batch、H3_LOWVRAM=1 専用の位相並べ替え)
    # ------------------------------------------------------------------
    def generate_still_batch(
        self,
        prompts: list[str],
        height: int = 768,
        width: int = 768,
        num_inference_steps: int = 30,
        seed: int | None = None,
        still_frames: int = 22,
        progress: ProgressState | None = None,
        cache: str | None = None,
        cache_threshold: float | None = None,
        attn: str | None = None,
        turbo: bool | None = None,
        # 出力 mp4 に音声ストリームを入れない (生成そのものは止まらない --
        # `_mux_mp4` の docstring 参照)。
        mute: bool = False,
    ) -> dict:
        """プロンプト違いの静止画 N 枚を、`H3_LOWVRAM=1` の固定費をバッチ全体で1回に
        償却して生成する(物語の場面画像の連番生成用)。

        `generate(still=True)` を N 回呼ぶと毎回 TE ロード(~75-97s) + transformer
        ロード(~35-40s)を払う(lowvram=1 は「リクエスト間は何も常駐させない」設計の
        ため)。この関数は同じ choreography を **位相順** に並べ替える:

            entry   : [nothing big resident]
            encode  : [TE-nf4 21GB]   全場面の setup/エンコード/layout/latents/timesteps
            (TE freed)
            denoise : [transformer-int8 34GB]   全場面を順にデノイズ
            (transformer freed)
            decode  : [vae pair]   全場面を順にデコード → PNG/mp4 保存
            (vae parked; 何も再ロードしない = lowvram=1 の定常状態そのまま)

        各位相の常駐セットは generate() の lowvram 分岐と同一なので、VRAM 予算は
        1枚生成と変わらない(場面ごとに増えるのは潜在とprompt_embedsのみ。22フレームの
        潜在は2フレーム相当で場面あたり数十MB)。実測の狙い: ~157s/枚 → ~35-40s/枚。

        場面間で共有される可変状態のリセット(このタスクで確認した2点):
        - スケジューラ: sigmas/timesteps の**値**は全場面で同一(同じ幾何・ステップ数)
          なので、encode 位相で場面ごとに timesteps_step を回した後は、デノイズ直前に
          `_step_index = None` に戻すだけでよい(`MiniMaxH3Scheduler.step()` は
          `_step_index is None` のとき timestep 値から index を再導出する --
          scheduling_minimax_h3.py L262-263 で確認)。video/audio 両方に適用
        - FirstBlockCache: generate() と同じく場面ごとに `_reset_stateful_cache()` +
          `cache_context("h3")`(前の場面の残差で新しい場面の step 0 が誤スキップ
          しないように)

        seed は全場面共通(= 手動で同一 seed を N 回叩くのと同じ挙動。場面ごとの
        再現性がバッチの構成に依存しない)。decode 位相は場面ごとに PNG/mp4 を保存
        しながら進むので、途中で失敗しても完了済み場面のファイルは outputs/ に残る
        (レスポンス自体は 500)。decode 例外時の steady state 復元は generate() と
        同じ(lowvram=1 では VAE を CPU に戻すだけ)。

        `H3_LOWVRAM=1` 以外のモードでは呼ばない(他モードは大モデルが常駐するため
        位相並べ替えの利得がなく、choreography も異なる)。app.py 側でモードを見て
        フォールバック(逐次 generate())する。
        """
        import core.settings as settings

        if not H3_LOWVRAM:
            raise RuntimeError(
                "generate_still_batch() は H3_LOWVRAM=1 専用です(他モードは大モデル常駐の"
                "ため位相並べ替えの利得がない)。呼び出し側で逐次 generate() にフォール"
                "バックしてください。"
            )
        if not prompts:
            raise ValueError("prompts が空です。")
        if still_frames not in STILL_FRAME_CHOICES:
            raise ValueError(f"still_frames must be one of {STILL_FRAME_CHOICES}, got {still_frames}")
        if still_frames == 5 and not H3_VAE_SMALLCLIP_FIX:
            raise ValueError(
                "still_frames=5 には H3_VAE_SMALLCLIP_FIX=1 (既定) が必要です "
                "(潜在2フレームのデコードは上流のチャンク境界バグで落ちるため)。"
            )

        instant = settings.resolve_instant_settings(cache, cache_threshold, attn, turbo)
        # PR #14355 (f37ab93) import updates -- same shape as `generate()`'s own (see that
        # method's matching comment for the full reasoning), simpler here since this method
        # is t2i-only (no keyframes, ever -- `state.set("image", None)` below): no
        # `MiniMaxH3ResizeStep` is needed, only `MiniMaxH3NoKeyframeAnchorsStep` (t2va's
        # "I anchor no keyframes" declaration, replacing the retired `MiniMaxH3SetupStep`)
        # and `MiniMaxH3AfterDenoiseStep` (unpatchify, now a separate step -- see
        # `generate()`'s own comment at its call site for the full contract change).
        from diffusers.modular_pipelines.minimax_h3.before_denoise import (
            MiniMaxH3NoKeyframeAnchorsStep,
            MiniMaxH3PrepareLatentsStep,
            MiniMaxH3PrepareLayoutStep,
            MiniMaxH3SetTimestepsStep,
        )
        from diffusers.modular_pipelines.minimax_h3.decoders import (
            MiniMaxH3AfterDenoiseStep,
            MiniMaxH3AudioDecodeStep,
            MiniMaxH3VideoDecodeStep,
        )
        from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3DenoiseStep
        from diffusers.modular_pipelines.modular_pipeline import PipelineState

        t_start = time.time()
        n_scenes = len(prompts)

        # --- entry: lowvram=1 の定常状態 (nothing big resident) から開始 ---
        # H3_KEEP_TRANSFORMER=1: generate() のエントリ分岐と同じ理由で `transformer` の
        # 解放をスキップし常駐させる (詳細はそちらのコメント/H3_KEEP_TRANSFORMER の
        # モジュールコメント参照)。このメソッドは H3_LOWVRAM=1 専用 (上のガード参照) な
        # ので、H3_KEEP_TRANSFORMER の import 時ガードが要求する H3_LOWVRAM=1 は既に
        # 満たされている。
        with self._load_lock:
            if not H3_KEEP_TRANSFORMER:
                self._free_transformer()
                self._active_variant = None
            self._drain_deferred_decode("generate_still_batch")
            coexist_ref2va = self._keep_ref2va_coexist("generate_still_batch")
            if not coexist_ref2va:
                self._free_transformer_ref()
            self._ensure_vaes(progress)
            self._load_text_encoder(progress)
        torch.cuda.reset_peak_memory_stats()
        # バッチは場面ループの外・リクエスト先頭で1回 (全場面共通の turbo 状態 -- この
        # メソッドはプロンプト以外のパラメータが全場面共通という前提そのもの)。以降の
        # 場面ループ内で回る `MiniMaxH3SetTimestepsStep` より前に必ず適用しておく。
        self._apply_turbo_video_shift(instant["turbo"], is_ref=False)
        pipe = self._pipe

        # --- encode 位相: TE 常駐のまま全場面を準備 ---
        # layout/latents/timesteps を TE 常駐中に回すのは generate() の lowvram 分岐と
        # 同じ理由 (`_execution_device` が text_encoder で解決される必要がある)。
        t_encode = time.time()
        scenes: list[dict] = []
        for idx, prompt in enumerate(prompts):
            # フェーズ境界での中断チェック(encoding): 場面ごとのループ境界。
            interrupt_controller.check()
            if progress:
                progress.update(phase="encoding", message=f"場面 {idx + 1}/{n_scenes} をエンコード中...")
            state = PipelineState()
            state.set("prompt", prompt)
            state.set("image", None)
            state.set("last_image", None)
            state.set("height", height)
            state.set("width", width)
            state.set("num_frames", still_frames)
            state.set("generator", torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None)
            state.set("num_inference_steps", num_inference_steps)
            state.set("output_type", "pt")
            state.set("attention_kwargs", None)
            state.set("latents", None)
            state.set("audio_latents", None)
            state.set("condition_latents", None)
            state.set("audio_condition_latents", None)

            # t2i is always t2va (no keyframes) -- `MiniMaxH3NoKeyframeAnchorsStep` is the
            # PR #14355 replacement for the retired `MiniMaxH3SetupStep()` call this used to
            # make (see `generate()`'s matching comment for the full contract change); the
            # canvas/duration validation `MiniMaxH3SetupStep` used to also perform now lives
            # in `MiniMaxH3PrepareLayoutStep` itself, so `_relaxed_min_duration()`'s scope
            # moves down to wrap that call instead, below.
            no_anchors_step = MiniMaxH3NoKeyframeAnchorsStep()
            _, state = no_anchors_step(pipe, state)

            with self._te_attached(), torch.no_grad():
                prompt_embeds, text_token_tags = _encode_h3_prompt(
                    pipe, prompt, None, device=self._encode_device, dtype=torch.bfloat16
                )
            prompt_embeds, text_token_tags = self._to_compute_device(prompt_embeds, text_token_tags)
            state.set("prompt_embeds", prompt_embeds)
            state.set("text_token_tags", text_token_tags)

            with _relaxed_min_duration():
                layout_step = MiniMaxH3PrepareLayoutStep()
                _, state = layout_step(pipe, state)
            latents_step = MiniMaxH3PrepareLatentsStep()
            _, state = latents_step(pipe, state)
            timesteps_step = MiniMaxH3SetTimestepsStep()
            _, state = timesteps_step(pipe, state)

            scenes.append({"prompt": prompt, "state": state})
        encode_time = time.time() - t_encode

        # --- TE を解放して transformer を1回だけロード ---
        interrupt_controller.check()  # ロード (~4.2s、途中中断不可) 前に中断を拾う (generate() と同じ理由)
        with self._load_lock:
            if coexist_ref2va:
                logger.info("generate_still_batch: text_encoder を解放せず常駐のまま base transformer をロード (H3_KEEP_REF2VA 両常駐)")
            else:
                self._free_text_encoder(force=True)
            self._ensure_transformer(progress)
        self.apply_instant_settings(self._pipe.transformer, instant, is_ref=False, progress=progress)

        # --- denoise 位相: 全場面を順に ---
        t_denoise = time.time()
        total_steps_all = num_inference_steps * n_scenes
        for idx, scene in enumerate(scenes):
            state = scene["state"]
            # PR #14355 対応: レイアウト段が CPU に作った state テンソルを計算用GPUへ
            # (`_scene_state_to_compute` の docstring 参照。既に GPU なら no-op)。
            self._scene_state_to_compute(state)
            # 場面間のスケジューラリセット (docstring 参照): 値は全場面同一なので
            # _step_index だけ初期化すれば step() が timestep から再導出する。
            pipe.scheduler._step_index = None
            pipe.audio_scheduler._step_index = None

            step_times: list[float] = []
            cache_skips = [0]
            denoise_step = MiniMaxH3DenoiseStep()
            orig_loop_step = denoise_step.loop_step

            def timed_loop_step(components, bstate, i, t, _idx=idx, _step_times=step_times, _skips=cache_skips):
                interrupt_controller.check()
                ts = time.time()
                result = orig_loop_step(components, bstate, i=i, t=t)
                _step_times.append(time.time() - ts)
                if instant["effective_cache"] == "fbc":
                    _skips[0] += self._fbc_last_step_was_skip()
                if progress:
                    progress.update(
                        phase="denoising",
                        step=_idx * num_inference_steps + i + 1,
                        total_steps=total_steps_all,
                        message=f"場面 {_idx + 1}/{n_scenes} をデノイズ中 {i + 1}/{num_inference_steps}",
                    )
                return result

            denoise_step.loop_step = timed_loop_step
            if instant["effective_cache"] == "fbc":
                # 場面ごとにリセット: 前の場面の最終ステップの残差が残っていると、
                # 新しい場面の step 0 が誤って skip 判定されうる (generate() の
                # per-request リセットと同じ理屈の per-scene 版)。
                self._pipe.transformer._reset_stateful_cache()
                with self._pipe.transformer.cache_context("h3"):
                    _, state = denoise_step(pipe, state)
            else:
                _, state = denoise_step(pipe, state)
            # PR #14355 note: unpatchify is a separate step now (`MiniMaxH3AfterDenoiseStep`,
            # decoders.py) -- see `generate()`'s matching comment for the full contract
            # change. Run once per scene, right after that scene's own denoise loop
            # finishes (while its `state` is still the active one in this per-scene loop),
            # rather than deferred to the decode-phase loop below -- it only reshapes
            # `state`'s own tensors (no vae/transformer dependency), so there is no
            # residency reason to defer it, and doing it here keeps every per-scene `state`
            # fully decode-ready by the time `scene["state"] = state` is stored.
            after_denoise_step = MiniMaxH3AfterDenoiseStep()
            _, state = after_denoise_step(pipe, state)
            scene["state"] = state
            scene["denoise_time_s"] = round(sum(step_times), 2)
            scene["avg_step_time_s"] = round(sum(step_times) / len(step_times), 3) if step_times else None
            scene["cache_skipped_steps"] = cache_skips[0] if instant["effective_cache"] == "fbc" else None
        denoise_time = time.time() - t_denoise

        # --- decode 位相: transformer を落として VAE で全場面をデコード ---
        if progress:
            progress.update(phase="decoding", message="全場面をデコード中...")
        # H3_KEEP_TRANSFORMER=1: generate() のデコード前分岐と同じ理由 (fp16 VAE 前提で
        # transformer 34.3 + デコード~11.4 = 45.7GB が実効予算49.8GBに収まる導出、
        # RESIDENCY.md §5.5) でスキップ。このメソッドはバッチ全体で1回しかこの位相を
        # 通らないため、常駐維持の効果は generate() の毎リクエスト分より小さいが、
        # 挙動は同じにしておく (二重の特別扱いを避ける)。
        if not H3_KEEP_TRANSFORMER:
            self._free_transformer()
        self._vae_to_gpu()
        t_decode = time.time()
        results: list[dict] = []
        try:
            for idx, scene in enumerate(scenes):
                if progress:
                    progress.update(phase="decoding", message=f"場面 {idx + 1}/{n_scenes} をデコード中...")
                state = scene["state"]
                video_decode_step = _cpu_norm_video_decode_step()
                _, state = video_decode_step(pipe, state)
                audio_decode_step = MiniMaxH3AudioDecodeStep()
                _, state = audio_decode_step(pipe, state)

                videos = state.get("videos")
                audio = state.get("audio")
                sampling_rate = state.get("sampling_rate")
                video_tensor = videos[0] if isinstance(videos, list) else videos
                # 全長ぶんの中間テンソルを GPU に積まないよう、フレームを小分けにして
                # CPU の出力配列へ直接書き込む (frames_to_uint8 の docstring 参照)。
                frames_uint8 = frames_to_uint8(video_tensor)
                audio_np = audio[0].float().cpu().numpy()
                del video_tensor, videos, audio
                gc.collect()
                torch.cuda.empty_cache()

                # 場面ごとに保存しながら進む (途中失敗でも完了済み場面は残る)
                job_stub = f"t2i_{int(t_start)}_s{idx + 1}"
                mp4_path = self.output_dir / f"{job_stub}.mp4"
                _mux_mp4(frames_uint8, audio_np, sampling_rate, FPS, mp4_path, mute=mute)
                still_frame_index = len(frames_uint8) // 2
                png_path = self.output_dir / f"{job_stub}.png"
                Image.fromarray(frames_uint8[still_frame_index]).save(png_path)

                results.append({
                    "prompt": scene["prompt"],
                    "png_path": str(png_path),
                    "png_filename": png_path.name,
                    "mp4_path": str(mp4_path),
                    "mp4_filename": mp4_path.name,
                    "still_frame_index": still_frame_index,
                    "denoise_time_s": scene["denoise_time_s"],
                    "avg_step_time_s": scene["avg_step_time_s"],
                    "cache_skipped_steps": scene["cache_skipped_steps"],
                })
        except BaseException:
            logger.exception(
                "t2i_batch decode failed (scene %d/%d) -- restoring steady state before re-raising",
                len(results) + 1, n_scenes,
            )
            gc.collect()
            torch.cuda.empty_cache()
            try:
                self._vae_to_cpu()
            except Exception:
                logger.exception("steady-state restore after t2i_batch decode failure also failed")
            raise
        decode_time = time.time() - t_decode
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        self._vae_to_cpu()
        # lowvram=1: 何も再ロードしない (generate() の decode 後と同じ定常状態)

        total_elapsed = time.time() - t_start
        result = {
            "mode": "t2i_batch",
            "num_scenes": n_scenes,
            "height": height,
            "width": width,
            "num_frames": still_frames,
            "still_frames": still_frames,
            "duration_s": still_frames / FPS,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "encode_time_s": round(encode_time, 2),
            "denoise_time_s": round(denoise_time, 2),
            "decode_time_s": round(decode_time, 2),
            "peak_vram_gb": round(peak_vram, 2),
            "ram": ram_gb(),
            "total_elapsed_s": round(total_elapsed, 2),
            "per_image_s": round(total_elapsed / n_scenes, 2),
            "te_quant": TE_QUANT,
            "transformer_quant": H3_TRANSFORMER_QUANT,
            "lowvram": H3_LOWVRAM_RAW,
            "attn_backend": instant["attn"],
            "cache": instant["effective_cache"],
            "cache_threshold": instant["cache_threshold"] if instant["effective_cache"] == "fbc" else None,
            "turbo": instant["turbo"],
            "mute": bool(mute),
            "scenes": results,
        }
        if progress:
            progress.update(phase="done", message=f"完了 ({n_scenes}場面)", result_path=results[-1]["png_path"] if results else None)
        logger.info("t2i_batch done: %s", json.dumps({k: v for k, v in result.items() if k not in ("ram", "scenes")}, ensure_ascii=False))
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
        cache: str | None = None,
        cache_threshold: float | None = None,
        attn: str | None = None,
        turbo: bool | None = None,
        # 出力 mp4 に音声ストリームを入れない (生成そのものは止まらない --
        # `_mux_mp4` の docstring 参照)。
        mute: bool = False,
        still: bool = False,
        still_frames: int = 22,
        # Per-request override of `reference_image_short_edge` (see
        # H3_REF_IMAGE_SHORT_EDGE's module-level comment for the "why"). `None` (default)
        # means "use whatever this process's H3_REF_IMAGE_SHORT_EDGE env var resolved to"
        # -- byte-for-byte identical to this parameter not existing at all. Validated and
        # applied just before the setup step below (see that call site's comment).
        reference_image_short_edge: int | None = None,
        # Per-request override of the module-level `H3_VOCAL_LOCK` env-var default.
        # `None` (default) means "use whatever this process's H3_VOCAL_LOCK env var
        # resolved to" -- byte-for-byte identical to this parameter not existing at all.
        # `True`/`False` take precedence over the env var for this one request. Resolved
        # once, right below, into a local `vocal_lock_effective` that every one of the
        # (former) `H3_VOCAL_LOCK` guard sites in this method now reads instead of the
        # module-level constant directly (the constant itself is untouched and still
        # backs the env-var default via this resolution).
        vocal_lock: bool | None = None,
        # H3_DECODE_STREAM / H3_DECODE_DEVICE 用: denoise 完了直後 (decode 開始前) に1回だけ
        # 呼ばれるコールバック。app が生成ロックの解放に使う。`decode_deferred_active()` が
        # False のとき (既定) は一切呼ばれない。
        on_denoise_done=None,
    ) -> dict:
        """
        Runs ref2va: joint video+audio generation conditioned on an ordered list of
        `MiniMaxH3ImageReference` / `MiniMaxH3VideoReference` / `MiniMaxH3AudioReference`
        instances (up to 9/3/3, 12 total).

        `still=True` は参照付き静止画モード (ref2i): `seconds` を無視して `still_frames`
        (STILL_FRAME_CHOICES) の超短尺を生成し、中央フレームを PNG に書き出す。
        キャラクター参照から場面ごとの一貫した静止画を作る用途
        (2026-08-07 のスパイク `scripts/probe_ref2va_short.py` で品質成立を実証済み。
        README「スパイク: Ref2VA×超短尺」参照)。generate() の still と同じ3点セット
        (setup step の間だけの尺ゲート緩和・VAE 小クリップ修正・既存の decode 例外
        クリーンアップ) で動く。

        `seconds=None` is only valid when `references` carries exactly one audio-bearing
        reference (a lone audio reference, or a video reference with a soundtrack) -- the
        generated duration is then that reference's own, per
        `MiniMaxH3Ref2VASetupStep.__call__`. `height`/`width` default to MiniMax-H3's own
        16:9 canvas when left out (references never bind the target geometry -- each is
        prepared at its own resolution, see references.py's module docstring).

        Mirrors `generate()`'s structure closely (same FBC instrumentation, same
        bnb-4bit-mode decode-window transformer drop/reload pattern), but against
        `self._pipe_ref` / `transformer_ref` and the ref2va block set. Does not support
        `upscale` (hires-fix) -- out of scope for this task, and `_upscale_block_state_2x`
        assumes t2va's `num_condition_video_rows == 0`, which is never true here (a
        reference always adds condition rows).

        `cache`/`cache_threshold`/`attn`/`turbo`: same instant-apply overrides as
        `generate()` (see core/settings.py), applied to `transformer_ref` instead of
        `transformer`. `upscale` is not a parameter here at all (see above), so there is
        no upscale-vs-turbo interaction to validate on this path.

        `vocal_lock`: per-request override of the module-level `H3_VOCAL_LOCK` env-var
        default (`None` = use the env var, unchanged behavior). See that constant's
        module comment and `_build_vocal_lock_latents()`'s docstring for what it does.

        Returns a dict with mp4_path, frame counts, timing and VRAM/RAM stats, in the
        same shape `generate()` returns (plus `references_summary`).

        PR #14355 (f37ab93) migration note: `self._pipe_ref` is now just an alias for
        `self._pipe` (see `_ensure_pipe_ref_shell`'s docstring) -- there is only ONE
        ModularPipeline shell, whose `_component_specs` already carries `transformer_ref`
        alongside `transformer`. `MiniMaxH3Ref2VACoreDenoiseStep`'s own block order
        (modular_blocks_minimax_h3.py) is `prepare_layout, prepare_condition_latents,
        prepare_latents, prepare_latents_ref2va, set_timesteps, denoise, after_denoise` --
        this method follows that order exactly (see `generate()`'s own fl2va comment on
        why `MiniMaxH3PrepareConditionLatentsStep` must run *before*
        `MiniMaxH3PrepareLatentsStep`, and `MiniMaxH3Ref2VAPrepareLatentsStep` after: the
        draw order is part of what the request's generator reproduces).
        `MiniMaxH3AfterDenoiseStep` (unpatchify) is inserted right before decode, same as
        `generate()`'s own t2va/fl2va path.
        """
        import core.settings as settings

        instant = settings.resolve_instant_settings(cache, cache_threshold, attn, turbo)

        from diffusers.modular_pipelines.minimax_h3.before_denoise import (
            MiniMaxH3PrepareConditionLatentsStep,
            MiniMaxH3PrepareLatentsStep,
            MiniMaxH3Ref2VAPrepareLatentsStep,
            MiniMaxH3Ref2VAPrepareLayoutStep,
            MiniMaxH3SetTimestepsStep,
        )
        from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
        from diffusers.modular_pipelines.minimax_h3.decoders import (
            MiniMaxH3AfterDenoiseStep,
            MiniMaxH3AudioDecodeStep,
            MiniMaxH3VideoDecodeStep,
        )
        from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3Ref2VADenoiseStep
        from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VAReferenceEncoderStep
        from diffusers.modular_pipelines.modular_pipeline import PipelineState

        _tl("gen_entry")
        t_start = time.time()
        # H3_PHASE_TIMING (2026-08-27 encode-phase profiling task, see `_PhaseTimer`'s
        # own docstring): one timer per request, marked at every named checkpoint
        # between here and the start of denoise. No-op (single flag check per `.mark()`
        # call) when `H3_PHASE_TIMING=0` (default).
        _pt = _PhaseTimer("generate_ref2va")
        ref_latent_cache_status = None  # H3_REF_LATENT_CACHE: None(OFF)/hit/miss/bypass
        # Per-request override of `H3_VOCAL_LOCK` (see this parameter's own docstring
        # paragraph above). `None` -> fall back to the module-level env-var default,
        # byte-for-byte identical to today's always-env-var behavior. Every
        # `H3_VOCAL_LOCK`-gated branch below reads this local instead of the module
        # constant directly.
        vocal_lock_effective = H3_VOCAL_LOCK if vocal_lock is None else vocal_lock
        if not references:
            raise ValueError("ref2va needs at least one reference; use generate() for text-only requests.")
        # TE 外部常駐 (H3_TE_DEVICE) の TE用GPUが 24GB 未満なら ref2va は動かない:
        # 2048px 短辺の参照を vision tower に通す活性化が入らず OOM することを実測済み
        # (20GB カードで 19.25GB 使用中に 204MB 不足、`H3_TE_DEVICE` のコメント参照)。
        # 「動くはず」で走らせて OOM させるより、理由を添えて明確に拒否する。
        if self._te_external and not self._te_external_usable_for("ref2va"):
            raise ValueError(
                f"ref2va は H3_TE_DEVICE={H3_TE_DEVICE!r} との併用ができません: "
                "参照画像を vision tower に通す活性化のぶん、TE用GPU に空き "
                f"{self._TE_EXTERNAL_MIN_FREE_GB_REF2VA_PROJ if H3_TE_PROJ else self._TE_EXTERNAL_MIN_FREE_GB_REF2VA_32B:.1f}GB "
                f"以上が必要です ({'投影TE(4B)' if H3_TE_PROJ else '32B TE'} 使用時の実測値。"
                "同じGPUを他のプロセスと共有している場合はそちらの使用分も空きを減らします)。"
                "そのGPUを空けるか、H3_TE_DEVICE を外して起動してください"
                "(t2va/fl2va/t2i は併用可能)。"
            )
        # PR #14355 後: `packing_ref2va.reference_kind(index, entry)` は削除され、
        # `kind`/`has_audio` は各 MiniMaxH3*Reference インスタンス自身の属性になった
        # (references.py -- `MiniMaxH3ImageReference.kind`/`has_audio` はクラス属性、
        # `MiniMaxH3VideoReference.has_audio` はプロパティ)。
        kinds = [entry.kind for entry in references]
        if set(kinds) == {"audio"}:
            raise ValueError(
                "An audio reference has to be paired with at least one image or video reference and cannot be "
                "used on its own."
            )
        if still:
            if still_frames not in STILL_FRAME_CHOICES:
                raise ValueError(f"still_frames must be one of {STILL_FRAME_CHOICES}, got {still_frames}")
            if still_frames == 5 and not H3_VAE_SMALLCLIP_FIX:
                raise ValueError(
                    "still_frames=5 には H3_VAE_SMALLCLIP_FIX=1 (既定) が必要です "
                    "(潜在2フレームのデコードは上流のチャンク境界バグで落ちるため)。"
                )
            # 音声参照からの尺自動導出 (seconds=None) と静止画の固定超短尺は両立しない --
            # 静止画では常に still_frames が尺を決める。
            num_frames = still_frames
        elif seconds is not None:
            num_frames = seconds_to_num_frames(seconds)
        else:
            # PR #14355 後: `num_frames` は `MiniMaxH3Ref2VASetupStep` の必須入力になり
            # (before_encoder.py, `InputParam(name="num_frames", required=True)`)、
            # 音声参照の長さからの尺自動導出は setup step 内ではもう行われない --
            # そのブロックの `num_frames` docstring 自身が「音声参照と同じ尺にするには
            # `round(samples / sample_rate * 24)` を渡せ」と明記している。旧実装の
            # `seconds=None` 挙動 (音声を持つ参照がちょうど1本なら、その音声長を尺にする)
            # はこの runner 側で肩代わりする。
            num_frames = _num_frames_from_audio_reference(references, FPS)

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
                if _keep_ref2va_active() and self._transformer_ref_loaded:
                    # H3_KEEP_REF2VA=1: 常駐のため解放をスキップ (上のコメントの OOM は
                    # 「TE + transformer_ref + VAE pair が同時に載る」収支が 96GB 超になる
                    # bf16/int8-both-resident 構成の話で、pruned の 21.6GB 常駐では成立しない)。
                    logger.info("ref2va: transformer_ref is resident (H3_KEEP_REF2VA=1) -- skip free/reload")
                else:
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
        _pt.mark("entry_lock(free_transformer+ensure_vaes+load_te+sync)")

        # Reset peak stats after loading so the reported peak reflects this generation's
        # encode+denoise+decode, not the (much larger, one-time) model loading peak.
        torch.cuda.reset_peak_memory_stats()
        # Must run before this request's `MiniMaxH3SetTimestepsStep` call (further down,
        # inside the mode-specific branches below) -- see `generate()`'s matching call
        # site and `_apply_turbo_video_shift`'s own docstring.
        self._apply_turbo_video_shift(instant["turbo"], is_ref=True)

        pipe = self._pipe_ref

        state = PipelineState()
        state.set("prompt", prompt)
        state.set("references", references)
        state.set("height", height)
        state.set("width", width)
        state.set("num_frames", num_frames)
        state.set("generator", torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None)
        if H3_HYPERFLOW and num_inference_steps != 8:
            logger.info(
                "H3_HYPERFLOW: num_inference_steps %s -> 8 (step数とσグリッドは重みに焼き込み)",
                num_inference_steps,
            )
            num_inference_steps = 8
        state.set("num_inference_steps", num_inference_steps)
        state.set("output_type", "pt")
        state.set("attention_kwargs", None)
        state.set("latents", None)
        state.set("audio_latents", None)

        # Per-request `reference_image_short_edge` override (see
        # `_resolve_ref_image_short_edge`'s docstring). Read by the setup step called
        # right below via `MiniMaxH3Ref2VASetupStep.__call__` ->
        # `before_encoder.py:490`, so it must be set on the shared pipe config *before*
        # that call, every request (not just when it differs from the current value --
        # unlike `_ensure_pipe_shell`'s one-time env-var setup, a later request with no
        # override must not silently keep an earlier request's explicit value, so this
        # always re-asserts the resolved value rather than skipping when unchanged).
        # Cache-safety note: ref2va's own prefix cache is keyed off the actual resized
        # reference pixels/shape (see H3_REF_PREFIX_CACHE's module comment), which
        # already differ whenever this value differs -- so there is no risk of one
        # request's cached prefix leaking into a request that asked for a different
        # short edge; a changed short edge is just a cache miss like any other input change.
        resolved_short_edge = _resolve_ref_image_short_edge(reference_image_short_edge)
        pipe.register_to_config(reference_image_short_edge=resolved_short_edge)
        if resolved_short_edge != H3_REF_IMAGE_SHORT_EDGE_DEFAULT:
            logger.info(
                "ref2va reference_image_short_edge=%d (default=%d)%s",
                resolved_short_edge, H3_REF_IMAGE_SHORT_EDGE_DEFAULT,
                " [per-request override]" if reference_image_short_edge is not None else " [H3_REF_IMAGE_SHORT_EDGE]",
            )

        # --- setup (canvas / frame count / reference prep) ---
        # Reference images/videos/audio are decoded and resized here (each at its own
        # resolution -- see references.py's module docstring). `num_frames` is already
        # resolved above (this block's own input is `required=True` now -- see the
        # `_num_frames_from_audio_reference` call site's comment).
        setup_step = MiniMaxH3Ref2VASetupStep()
        if still:
            # 超短尺 (5秒未満) は setup step の duration バリデーションが弾くため、
            # generate() の still と同じくこの1呼び出しの間だけ緩和する。
            with _relaxed_min_duration():
                _, state = setup_step(pipe, state)
        else:
            _, state = setup_step(pipe, state)
        actual_num_frames = state.get("num_frames")
        _pt.mark("setup_step(reference_normalize/resize)")  # PIL LANCZOS resize to ref_image_short_edge (CPU)

        # --- text encode (references' vision blocks + prompt; still has TE on GPU) ---
        if progress:
            progress.update(phase="encoding", message="プロンプト+参照をエンコード中...")
        with self._te_attached(), torch.no_grad():
            encoded = None
            if H3_REF_PREFIX_CACHE_SINGLE:
                # 参照が同一 (画像参照のみ、音声/プロンプトは違ってよい) ならプレフィックスの
                # KV キャッシュを前回リクエストから使い回す。使えない構成 (動画参照あり等)
                # なら `None` が返り、下の従来経路へそのまま落ちる。
                encoded = _encode_ref2va_prompt_prefix_cached(
                    pipe, prompt, state.get("normalized_references"),
                    device=self._encode_device, dtype=torch.bfloat16,
                )
            if encoded is None:
                encoded = _encode_ref2va_prompt(
                    pipe, prompt, state.get("normalized_references"),
                    device=self._encode_device, dtype=torch.bfloat16,
                )
            prompt_embeds, text_token_tags = encoded
        _pt.mark("text_encode(prefix_cache_or_direct)")  # whichever of the two _encode_ref2va_prompt* ran, as one unit -- see its own internal H3_PHASE_TIMING breakdown (gather_vision_features/build_presentation/conditioner_forward) for the split inside this
        prompt_embeds, text_token_tags = self._to_compute_device(prompt_embeds, text_token_tags)
        _pt.mark("to_compute_device")
        state.set("prompt_embeds", prompt_embeds)
        state.set("text_token_tags", text_token_tags)
        # フェーズ境界での中断チェック(encoding): プロンプト+参照のテキストエンコードが
        # 終わった直後(generate() の t2va/fl2va 版と同じ位置)。
        interrupt_controller.check()

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
        #
        # H3_VOCAL_LOCK (opt-in, "0" default): set inside whichever branch below runs,
        # while `audio_vae` is still GPU-resident (see `_build_vocal_lock_latents()`'s
        # docstring for why it cannot wait until after `_vae_to_cpu()`). Declared here,
        # ahead of the dispatch, so every branch (and the `H3_VOCAL_LOCK and
        # vocal_lock_latents is not None` checks around each branch's own
        # `timesteps_step` call) can rely on it always being defined, even for the
        # `H3_VOCAL_LOCK=0` (default, fully inert) path.
        vocal_lock_latents = None
        if H3_LOWVRAM_GROUP:
            # UPDATE (found via this task's own 32GB-ballast verification, after the
            # original version of this branch -- which called `self._vae_to_gpu()`
            # unconditionally above, before this `if`, mirroring the plain `bnb-4bit`
            # branch below -- OOM'd right at that call): TE-nf4(21GB, still resident
            # here) + vae pair(11GB) = 32GB already exceeds a 30GB-class card's budget
            # on its own, *before* transformer_ref is even loaded -- a genuine
            # TE-vs-vae conflict, unrelated to this mode's transformer choreography
            # (which is why the original comment here, reasoning only about
            # transformer_ref's tiny footprint, missed it). Prompt+reference text
            # encoding (`_encode_ref2va_prompt`, this file's own no_grad-wrapped
            # replacement for the retired `encode_prompt` staticmethod -- see its own
            # docstring -- already ran above and does not need `_execution_device`) is
            # TE's only job for this request, and it is already done by this point --
            # so TE can be freed here, before `vae` goes to GPU, same as generate()'s
            # own decode-window fix. Safe ordering for `_execution_device` (resolved by
            # `reference_encoder_step`/`layout_step` below, per the pipe's own
            # `_component_specs` order `text_encoder, ..., vae, ..., transformer_ref`):
            # free TE FIRST, then bring `vae` onto GPU -- by the time
            # `reference_encoder_step` runs, `text_encoder` is gone from the scan and
            # `vae` is already GPU-resident, so `_execution_device` resolves to `vae`'s
            # correct (GPU) location. The reverse order (`vae` to GPU while TE is still
            # resident, then free TE) would also resolve correctly per the scan order,
            # but freeing first avoids ever holding TE(21)+vae(11)=32GB at the same
            # time even transiently.
            with self._load_lock:
                self._free_text_encoder(force=True)
            self._vae_to_gpu()
            state, ref_latent_cache_status = _ref2va_encode_references(pipe, state)
            if vocal_lock_effective:
                # Must run inside this "audio_vae on GPU" window, before `_vae_to_cpu()`
                # parks it back -- see `_build_vocal_lock_latents()`'s docstring.
                vocal_lock_latents = _build_vocal_lock_latents(pipe, references, actual_num_frames)
                if vocal_lock_latents is not None:
                    state.set("audio_latents", vocal_lock_latents)
            self._vae_to_cpu()
            with self._load_lock:
                self._ensure_transformer_ref(progress)

            layout_step = MiniMaxH3Ref2VAPrepareLayoutStep()
            _, state = layout_step(pipe, state)
            condition_latents_step = MiniMaxH3PrepareConditionLatentsStep()
            _, state = condition_latents_step(pipe, state)
            latents_step = MiniMaxH3PrepareLatentsStep()
            _, state = latents_step(pipe, state)
            ref2va_latents_step = MiniMaxH3Ref2VAPrepareLatentsStep()
            _, state = ref2va_latents_step(pipe, state)
            timesteps_step = _make_set_timesteps_step()
            if vocal_lock_effective and vocal_lock_latents is not None:
                vocal_lock_original_num_condition_audio_rows = _inflate_vocal_lock_condition_rows(
                    state, state.get("num_audio_latents"), pipe.audio_channels
                )
            _, state = timesteps_step(pipe, state)
        elif H3_LOWVRAM:
            self._vae_to_gpu("encode")  # H3_VAE_SPLIT 時は encode 側だけ (それ以外は従来どおり全体)
            # Same `_execution_device` resolution trap as generate()'s own H3_LOWVRAM
            # branch (see its long comment): `vae` sits between `text_encoder` and
            # `transformer_ref` in the pipe's own component order, and stays a
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
            state, ref_latent_cache_status = _ref2va_encode_references(pipe, state)
            if vocal_lock_effective:
                # Must run inside this "audio_vae on GPU" window, before `_vae_to_cpu()`
                # parks it back -- see `_build_vocal_lock_latents()`'s docstring.
                vocal_lock_latents = _build_vocal_lock_latents(pipe, references, actual_num_frames)
                if vocal_lock_latents is not None:
                    state.set("audio_latents", vocal_lock_latents)
            if not (_keep_ref2va_active() and H3_KEEP_REF2VA_VAE):
                self._vae_to_cpu()
            else:
                logger.info("ref2va: vae/audio_vae stay on GPU (H3_KEEP_REF2VA_VAE=1) -- skip CPU park")

            layout_step = MiniMaxH3Ref2VAPrepareLayoutStep()
            _, state = layout_step(pipe, state)
            condition_latents_step = MiniMaxH3PrepareConditionLatentsStep()
            _, state = condition_latents_step(pipe, state)
            latents_step = MiniMaxH3PrepareLatentsStep()
            _, state = latents_step(pipe, state)
            ref2va_latents_step = MiniMaxH3Ref2VAPrepareLatentsStep()
            _, state = ref2va_latents_step(pipe, state)
            timesteps_step = _make_set_timesteps_step()
            if vocal_lock_effective and vocal_lock_latents is not None:
                vocal_lock_original_num_condition_audio_rows = _inflate_vocal_lock_condition_rows(
                    state, state.get("num_audio_latents"), pipe.audio_channels
                )
            _, state = timesteps_step(pipe, state)

            # TE 外部常駐 (`H3_TE_DEVICE`) のとき、この分岐の前提「layout 系ステップは
            # TE が GPU 常駐のうちに走るので `_execution_device` が正しく解決される」は
            # 成り立たない: TE は `_te_attached()` の窓の外では常にパイプから外れており
            # (`text_encoder=None`)、直前の `_vae_to_cpu()` の後は components スキャンが
            # CPU 常駐の vae/audio_vae に落ちて layout/latents/timesteps のテンソルが
            # CPU に作られる → denoise の transformer_ref forward で `cuda:0 and cpu` の
            # device 不一致になる (2026-08-20 実機で発症)。bnb-4bit/none 分岐のピン窓は
            # transformer_ref ロード済みが前提でここでは使えない (この分岐はロードが
            # layout の後) ため、バッチ経路 (`generate_ref_batch`) と同じ
            # `_scene_state_to_compute` で denoise 前に運ぶ。TE 同居時は全テンソルが
            # 既に GPU 上で no-op なので無条件に呼んでも安全だが、意図を明確にするため
            # 外部常駐時に限定する。
            if self._te_external:
                self._scene_state_to_compute(state)

            with self._load_lock:
                if _keep_ref2va_active():
                    # H3_KEEP_REF2VA=1: TE を解放せず、transformer_ref も常駐済みなら
                    # ロードしない (`_ensure_transformer_ref` は冪等、初回だけ実ロード)。
                    # layout 系ステップは既に走り終わっているので、TE が残っていても
                    # `_execution_device` の罠 (上のコメント) には影響しない。
                    logger.info(
                        "ref2va: text_encoder resident (H3_KEEP_REF2VA=1) -- skip free/reload; "
                        "transformer_ref %s",
                        "resident -- skip load" if self._transformer_ref_loaded else "loading (first request)",
                    )
                else:
                    self._free_text_encoder(force=True)
                self._ensure_transformer_ref(progress)
        elif TE_QUANT == "bnb-4bit":
            self._vae_to_gpu()  # already has its own unconditional timing log
            _pt.mark("vae_to_gpu")
            state, ref_latent_cache_status = _ref2va_encode_references(pipe, state)
            _pt.mark("reference_encoder_step(vae_encode_condition_latents)")
            if vocal_lock_effective:
                # Must run inside this "audio_vae on GPU" window, before `_vae_to_cpu()`
                # parks it back -- see `_build_vocal_lock_latents()`'s docstring.
                vocal_lock_latents = _build_vocal_lock_latents(pipe, references, actual_num_frames)
                if vocal_lock_latents is not None:
                    state.set("audio_latents", vocal_lock_latents)
            _pt.mark("vocal_lock_latents")  # no-op mark (0s) when H3_VOCAL_LOCK=0; else audio_vae encode of ref audio
            self._vae_to_cpu()  # already has its own unconditional timing log
            _pt.mark("vae_to_cpu")
            with self._load_lock:
                self._ensure_transformer_ref(progress)
                _pt.mark("ensure_transformer_ref")  # no-op mark in int8 both-resident steady state (already has its own timing log when it actually (re)loads)
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

            # --- layout / condition latents / latents / ref2va latents / timesteps ---
            # Order matches `MiniMaxH3Ref2VACoreDenoiseStep`'s own block_classes list
            # (modular_blocks_minimax_h3.py) exactly: the conditioning noise (image/video
            # references) has to be drawn from the request's generator BEFORE the
            # generated rows' own noise, and packed into `latents`/`audio_latents` AFTER.
            #
            # TE 外部常駐 (`H3_TE_DEVICE`) のときはピン窓の中で回す。TE を切り離すと
            # `_execution_device` が CPU 上の audio_vae に落ち、layout がテンソルを CPU に
            # 作ってしまい、デノイズの transformer forward で `cuda:0 and cpu` の
            # device 不一致になる (generate() の非 lowvram 分岐で 2026-08-12 に踏んだのと
            # 同型)。**この経路は 2026-08-13 に `_te_external_usable_for()` の 24GB 判定を
            # 投影TE向けに分離するまでガードで塞がれており、一度も走っていなかった** --
            # 緩和と同時に発火した。TE 同居時 (既定) は素通しで挙動はバイト単位で不変。
            with self._pin_execution_device_to_compute() if self._te_external else _NullContext():
                layout_step = MiniMaxH3Ref2VAPrepareLayoutStep()
                _, state = layout_step(pipe, state)
                condition_latents_step = MiniMaxH3PrepareConditionLatentsStep()
                _, state = condition_latents_step(pipe, state)
                latents_step = MiniMaxH3PrepareLatentsStep()
                _, state = latents_step(pipe, state)
                ref2va_latents_step = MiniMaxH3Ref2VAPrepareLatentsStep()
                _, state = ref2va_latents_step(pipe, state)
                timesteps_step = _make_set_timesteps_step()
                if vocal_lock_effective and vocal_lock_latents is not None:
                    vocal_lock_original_num_condition_audio_rows = _inflate_vocal_lock_condition_rows(
                        state, state.get("num_audio_latents"), pipe.audio_channels
                    )
                _, state = timesteps_step(pipe, state)
            _pt.mark("layout+condition_latents+latents+ref2va_latents+timesteps")
        else:
            # `none` mode: TE's job is done -- free it and bring in transformer_ref
            # (vae is already permanently resident in this mode, so `_vae_to_gpu()` is a
            # no-op here -- see its own guard -- kept for parity with the original
            # unconditional call this branch used to share with the others above; the
            # reference encoder step can run either before or after transformer_ref,
            # doing it here mirrors generate()'s own `none`-mode ordering for keyframes).
            self._vae_to_gpu()
            with self._load_lock:
                self._free_text_encoder()
                self._ensure_transformer_ref(progress)
            state, ref_latent_cache_status = _ref2va_encode_references(pipe, state)
            if vocal_lock_effective:
                # `none` mode keeps `vae`/`audio_vae` permanently GPU-resident (see the
                # branch comment above), so there is no "vae on GPU" window to miss here
                # -- but for consistency with the other three branches (and in case that
                # assumption ever changes) this still runs right after the reference
                # encoder step, before anything else touches `audio_vae`.
                vocal_lock_latents = _build_vocal_lock_latents(pipe, references, actual_num_frames)
                if vocal_lock_latents is not None:
                    state.set("audio_latents", vocal_lock_latents)

            # --- layout / condition latents / latents / ref2va latents / timesteps ---
            # TE 外部常駐 (`H3_TE_DEVICE`) のときはピン窓の中で回す。TE を切り離すと
            # `_execution_device` が CPU 上の audio_vae に落ち、layout がレイアウト用
            # テンソルを CPU に作ってしまい、デノイズで rope が
            # `cuda:0 and cpu` の device 不一致で落ちる (generate() の非 lowvram 分岐で
            # 2026-08-12 に踏んだのと同型)。**この経路は 2026-08-13 に
            # `_te_external_usable_for()` の 24GB 判定を投影TE向けに緩めるまで
            # ガードで塞がれており、一度も走っていなかった** -- 緩和と同時に発火した。
            # `_pin_execution_device_to_compute()` は text_encoder/vae/audio_vae を一時的に
            # 外して transformer(_ref) を先頭に見せる (transformer_ref は直前の
            # `_ensure_transformer_ref()` でロード済みなので前提を満たす)。
            # TE 同居時 (既定) は従来どおり素通しで、挙動はバイト単位で不変。
            with self._pin_execution_device_to_compute() if self._te_external else _NullContext():
                layout_step = MiniMaxH3Ref2VAPrepareLayoutStep()
                _, state = layout_step(pipe, state)
                condition_latents_step = MiniMaxH3PrepareConditionLatentsStep()
                _, state = condition_latents_step(pipe, state)
                latents_step = MiniMaxH3PrepareLatentsStep()
                _, state = latents_step(pipe, state)
                ref2va_latents_step = MiniMaxH3Ref2VAPrepareLatentsStep()
                _, state = ref2va_latents_step(pipe, state)
                timesteps_step = _make_set_timesteps_step()
                if vocal_lock_effective and vocal_lock_latents is not None:
                    vocal_lock_original_num_condition_audio_rows = _inflate_vocal_lock_condition_rows(
                        state, state.get("num_audio_latents"), pipe.audio_channels
                    )
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
        # resolves to the device of the *first* `nn.Module` still set on `self._pipe_ref`
        # (== `self._pipe`), in the pipe's own `_component_specs` order -- `text_encoder`
        # first, then `vae`. Freeing text_encoder before those three steps run would make
        # `vae` (parked on CPU in bnb-4bit mode outside its active phase, which ended when
        # `_vae_to_cpu()` ran above) the new first hit, silently resolving
        # `_execution_device` to `cpu` -- the identical device-mismatch trap generate()'s
        # own comment documents finding for its layout_step. Freeing TE only once those
        # position_ids/layout tensors already exist on the correct device (set once here,
        # and never touched again for the rest of the request) sidesteps it entirely.
        # H3_LOWVRAM: always False here -- TE was already force-freed above, before
        # transformer_ref was even loaded (see the H3_LOWVRAM branch above).
        # H3_LOWVRAM_GROUP: always False here too -- transformer_ref's tiny actual GPU
        # footprint never needed TE force-freed to make room for it in the first place
        # (see the H3_LOWVRAM_GROUP branch above).
        force_free_te = (
            TE_QUANT == "bnb-4bit" and not H3_TRANSFORMER_BOTH_RESIDENT and not H3_LOWVRAM_ANY
        )
        if force_free_te:
            with self._load_lock:
                self._free_text_encoder(force=True)
        _pt.mark("force_free_te")  # no-op mark (0s) in int8/H3_TRANSFORMER_BOTH_RESIDENT mode (force_free_te=False)
        _pt.report(total_hint=time.time() - t_start)  # full encode-phase breakdown; sum should equal t_denoise - t_start below

        # --- denoise loop, instrumented for progress polling (mirrors generate()'s
        # non-upscale path exactly, against transformer_ref instead of transformer) ---
        # denoise 開始直前の保険チェック(generate() と同じ位置、docstring 参照)。
        interrupt_controller.check()
        if progress:
            progress.update(phase="denoising", step=0, total_steps=num_inference_steps, message="デノイズ中...")
        t_denoise = time.time()
        _tl("denoise_start")
        step_times = []
        cache_skips = [0]
        out_height, out_width = state.get("height"), state.get("width")

        # Instant-apply this request's cache/attn/turbo settings -- see generate()'s
        # matching comment for the full reasoning. `transformer_ref` is confirmed
        # resident by every branch above this point.
        self._ensure_hyperflow_ref(progress=progress)
        self.apply_instant_settings(self._pipe_ref.transformer_ref, instant, is_ref=True, progress=progress)
        if H3_DENOISE_CUDAGRAPH and not isinstance(self._pipe_ref.transformer_ref.__dict__.get("forward"), _h3_graph.ForwardGraphRunner):
            _h3_graph.ForwardGraphRunner(self._pipe_ref.transformer_ref).install()
            logger.info("H3_DENOISE_CUDAGRAPH: ForwardGraphRunner installed on transformer_ref")

        def _fbc_reset_and_context():
            self._pipe_ref.transformer_ref._reset_stateful_cache()
            return self._pipe_ref.transformer_ref.cache_context("h3")

        denoise_step = _maybe_hyperflowify_denoise_step(MiniMaxH3Ref2VADenoiseStep())
        orig_loop_step = denoise_step.loop_step

        def timed_loop_step(components, bstate, i, t):
            interrupt_controller.check()
            ts = time.time()
            result = orig_loop_step(components, bstate, i=i, t=t)
            step_times.append(time.time() - ts)
            if instant["effective_cache"] == "fbc":
                cache_skips[0] += self._fbc_last_step_was_skip_ref()
            if progress:
                progress.update(step=i + 1, message=f"デノイズ中 {i + 1}/{num_inference_steps}")
            return result

        denoise_step.loop_step = timed_loop_step
        if instant["effective_cache"] == "fbc":
            # Per-request reset -- see generate()'s matching comment for why this is
            # required (a stale head-block residual from a previous call could otherwise
            # make step 0 wrongly skip).
            self._pipe_ref.transformer_ref._reset_stateful_cache()
            with self._pipe_ref.transformer_ref.cache_context("h3"):
                _, state = denoise_step(pipe, state)
        else:
            _, state = denoise_step(pipe, state)
        denoise_time = time.time() - t_denoise
        _tl("denoise_end(cpu)")

        # PR #14355 note: unpatchify is a separate step now (`MiniMaxH3AfterDenoiseStep`,
        # decoders.py) -- see `generate()`'s matching comment for the full contract. Has to
        # run once, right after the denoise loop and before either decode step below;
        # matches where `after_denoise` sits in `MiniMaxH3Ref2VACoreDenoiseStep`'s own
        # block list (modular_blocks_minimax_h3.py). Unlike t2va/fl2va,
        # `num_condition_video_rows`/`num_condition_audio_rows` on `state` (set by the
        # layout step above) are NOT zero here -- a reference always adds condition rows,
        # which is exactly what this step drops before reshaping the generated rows back
        # into a 5D video tensor / channel-major audio tensor.
        #
        # H3_VOCAL_LOCK: `num_condition_audio_rows` was inflated (by whichever branch
        # ran above, via `_inflate_vocal_lock_condition_rows()`) to freeze the
        # generated audio rows through the just-finished denoise loop -- see that
        # function's docstring for why it has to stay inflated in `state` all the way
        # through the `denoise_step(pipe, state)` call above. It MUST be restored here,
        # before `MiniMaxH3AfterDenoiseStep` runs: that step slices the generated rows
        # as `audio_latents[num_condition_audio_rows:]` and reshapes them assuming the
        # *un*-inflated (reference-rows-only) count -- left inflated, the slice would be
        # empty and the reshape would fail (`decoders.py`'s
        # `audio_rows.reshape(components.audio_channels, block_state.num_audio_latents, ...)`).
        if vocal_lock_effective and vocal_lock_latents is not None:
            _restore_vocal_lock_condition_rows(state, vocal_lock_original_num_condition_audio_rows)
        after_denoise_step = MiniMaxH3AfterDenoiseStep()
        _, state = after_denoise_step(pipe, state)
        _tl("after_denoise_step")

        # --- decode (shared MiniMaxH3VideoDecodeStep/MiniMaxH3AudioDecodeStep -- no
        # ref2va-specific decode step exists; `MiniMaxH3AfterDenoiseStep` just above
        # already dropped the reference condition rows, so these two only ever see the
        # generated rows) ---
        # --- H3_DECODE_STREAM / H3_DECODE_DEVICE (既定 OFF): decode を denoise から分離する ---
        # 条件を満たすときだけ、denoise 直後に生成ロックを手放し、decode + uint8 変換を専用
        # ストリーム/別 GPU で行う (`_decode_ref2va_deferred` の docstring に安全性の根拠)。
        # 満たさない/フラグ OFF なら下の従来経路 (インライン decode) がそのまま走る。
        _decode_deferred = self.decode_deferred_active()
        decode_info = None
        if _decode_deferred:
            _holder = [state.get("latents"), state.get("audio_latents")]
            # 重い参照を手放してから (= 次リクエストに VRAM を返してから) 生成ロックを解放する。
            state = prompt_embeds = text_token_tags = encoded = vocal_lock_latents = None
            (frames_uint8, audio_np, sampling_rate, rms, peak, peak_vram, decode_time,
             decode_info) = self._decode_ref2va_deferred(pipe, _holder, progress, on_denoise_done)
        else:
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
            # H3_LOWVRAM_GROUP: `transformer_ref` is left alone here -- same reasoning as
            # generate()'s own decode section (a group-offloaded transformer_ref's actual
            # GPU footprint never conflicted with the vae pair's headroom in the first
            # place). TE-nf4 DOES need force-freeing here though, for the same reason found
            # by this task's own 32GB-ballast diagnostic against generate()'s t2va path (see
            # that decode section's own comment for the full `_log_gpu_tensor_diag()`
            # investigation): TE-nf4's own ~21GB is real, live, referenced memory, not
            # reclaimable via `empty_cache()` alone, and it is not needed by either decode
            # step (MiniMaxH3VideoDecodeStep/MiniMaxH3AudioDecodeStep only touch
            # vae/audio_vae/video_processor). `force_free_te` was already True and did the
            # force-free earlier in this method (before denoise, per its own definition
            # above) in the non-group-mode branches, but H3_LOWVRAM_GROUP always has
            # `force_free_te=False` (transformer_ref's tiny footprint never needed it
            # before denoise) -- so it has to be freed here, at decode, instead.
            def _restore_decode_steady_state_ref():
                # generate() の `_restore_decode_steady_state()` と同じ役割の ref2va 版。
                # 正常系と decode 例外時の両方から呼ぶ(例外時に復元しないと後続リクエストが
                # 不整合な常駐セットを引き継いで連鎖 OOM する -- generate() 側の同名 closure の
                # コメント参照)。
                if not (_keep_ref2va_active() and H3_KEEP_REF2VA_VAE):
                    self._vae_to_cpu()
                if TE_QUANT == "bnb-4bit" and not H3_LOWVRAM_ANY:
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
                        if H3_TRANSFORMER_BOTH_RESIDENT and H3_EAGER_VARIANT_RESTORE:
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
                elif H3_LOWVRAM_GROUP:
                    # `transformer_ref` was force-freed unconditionally at this method's entry
                    # (see the entry-section comment) and is not reloaded here -- ref2va never
                    # keeps a cross-request transformer_ref steady state in this mode (matches
                    # plain bnb-4bit's own non-both-resident choice). TE-nf4 is reloaded though,
                    # for the same reasoning as generate()'s own t2va decode tail: the next
                    # request (t2va or ref2va) needs TE first regardless, so restoring it now
                    # avoids paying its reload cost on that request's own critical path.
                    with self._load_lock:
                        self._load_text_encoder(progress)
                # H3_LOWVRAM: deliberately do NOT reload transformer_ref/TE here -- same
                # "nothing big resident between requests" reasoning as generate()'s own
                # lowvram decode tail.

            if _keep_ref2va_active():
                # H3_KEEP_REF2VA=1: decode 窓でも transformer_ref/TE を落とさない (LOWVRAM=1 では
                # 元々 decode 後に再ロードしないので、ここで落とすと次リクエストが丸ごと再ロードになる)。
                logger.info("ref2va: transformer_ref/text_encoder stay resident through decode (H3_KEEP_REF2VA=1)")
            elif TE_QUANT == "bnb-4bit" and not H3_TRANSFORMER_BOTH_RESIDENT and not H3_LOWVRAM_GROUP:
                self._free_transformer_ref()
            elif H3_LOWVRAM_GROUP:
                with self._load_lock:
                    self._free_text_encoder(force=True)
            _dbg_pre_peak = 0.0
            if H3_DEBUG_DECODE_MEM:
                # 窓前までのピークを退避してからリセット (結果の peak_vram_gb は合成して保つ)。
                torch.cuda.synchronize()
                _dbg_pre_peak = torch.cuda.max_memory_allocated() / 1e9
                logger.info("[DECODE_MEM] before vae_to_gpu: pre-window peak=%.2fGB gpu=%s", _dbg_pre_peak, gpu_mem_gb())
                torch.cuda.reset_peak_memory_stats()
            self._vae_to_gpu("decode")  # H3_VAE_SPLIT 時は decode 側だけ (それ以外は従来どおり全体)
            if H3_DEBUG_DECODE_MEM:
                logger.info("[DECODE_MEM] after vae_to_gpu: gpu=%s", gpu_mem_gb())
            t_decode = time.time()
            try:
                video_decode_step = _cpu_norm_video_decode_step()
                _, state = video_decode_step(pipe, state)
                audio_decode_step = MiniMaxH3AudioDecodeStep()
                _, state = audio_decode_step(pipe, state)
                decode_time = time.time() - t_decode

                videos = state.get("videos")
                audio = state.get("audio")
                sampling_rate = state.get("sampling_rate")

                video_tensor = videos[0] if isinstance(videos, list) else videos
                # 全長ぶんの中間テンソルを GPU に積まないよう、フレームを小分けにして
                # CPU の出力配列へ直接書き込む (frames_to_uint8 の docstring 参照)。
                frames_uint8 = frames_to_uint8(video_tensor)
                audio_np = audio[0].float().cpu().numpy()
                rms = float(np.sqrt(np.mean(audio_np**2)))
                peak = float(np.max(np.abs(audio_np)))

                peak_vram = torch.cuda.max_memory_allocated() / 1e9
                if H3_DEBUG_DECODE_MEM:
                    logger.info("[DECODE_MEM] decode window peak=%.2fGB gpu=%s", peak_vram, gpu_mem_gb())
                    peak_vram = max(peak_vram, _dbg_pre_peak)

                del video_tensor, videos, audio
                gc.collect()
                torch.cuda.empty_cache()
            except BaseException:
                logger.exception(
                    "ref2va decode failed -- freeing partial buffers and restoring the steady "
                    "state before re-raising (so the next request does not inherit a corrupted "
                    "resident set)"
                )
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    _restore_decode_steady_state_ref()
                except Exception:
                    # 復元自体の失敗で元の decode 例外を潰さない(原因情報は元例外側にある)。
                    logger.exception("steady-state restore after ref2va decode failure also failed")
                raise

            _restore_decode_steady_state_ref()

        if progress:
            progress.update(phase="muxing", message="mp4へmux中...")
        ref_mode = "ref2i" if still else "ref2va"
        job_stub = f"{ref_mode}_{int(t_start)}"
        mp4_path = self.output_dir / f"{job_stub}.mp4"
        _t_mux0 = time.time()
        _mux_mp4(frames_uint8, audio_np, sampling_rate, FPS, mp4_path, mute=mute)
        mux_time = time.time() - _t_mux0

        # 参照付き静止画モード: 中央フレームを PNG として書き出す (generate() の still と同じ)
        png_path = None
        still_frame_index = None
        if still:
            still_frame_index = len(frames_uint8) // 2
            png_path = self.output_dir / f"{job_stub}.png"
            Image.fromarray(frames_uint8[still_frame_index]).save(png_path)

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
            "still": int(still),
            "still_frames": still_frames if still else None,
            "still_frame_index": still_frame_index,
            "png_path": str(png_path) if png_path else None,
            "png_filename": png_path.name if png_path else None,
            "total_elapsed_s": round(time.time() - t_start, 2),
            "mode": ref_mode,
            "te_quant": TE_QUANT,
            "transformer_quant": H3_TRANSFORMER_QUANT,
            "lowvram": H3_LOWVRAM_RAW,
            "attn_backend": instant["attn"],
            "cache_mode": instant["cache"] if not instant["turbo"] else "none (force-disabled by turbo=1)",
            "cache": instant["effective_cache"],
            "cache_threshold": instant["cache_threshold"] if instant["effective_cache"] == "fbc" else None,
            "turbo_lora": instant["turbo"],
            "turbo": instant["turbo"],
            "mute": bool(mute),
            "cache_skipped_steps": cache_skips[0] if instant["effective_cache"] == "fbc" else None,
            "reference_image_short_edge": resolved_short_edge,
            "vocal_lock": bool(vocal_lock_effective),
            "references_summary": [
                {"index": index, "kind": kind, "has_audio": bool(references[index].has_audio)}
                for index, kind in enumerate(kinds)
            ],
        }
        # --- 以下は新フラグ (H3_REF_LATENT_CACHE / H3_DECODE_*) が有効なときだけ付く追加キー ---
        # (フラグ OFF の既定では従来と同一のキー集合を返す)
        if H3_REF_LATENT_CACHE:
            result["ref_latent_cache"] = ref_latent_cache_status
        if decode_info is not None:
            result["decode_mode"] = decode_info
            result["mux_time_s"] = round(mux_time, 2)
        if progress:
            progress.update(phase="done", message="完了", result_path=str(png_path) if png_path else str(mp4_path))
        logger.info("ref2va generation done: %s",
                     json.dumps({k: v for k, v in result.items() if k != "ram"}, ensure_ascii=False))
        return result

    def generate_ref_batch(
        self,
        prompts: list[str],
        references: list,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 30,
        seed: int | None = None,
        still: bool = False,
        still_frames: int = 22,
        seconds: float | None = None,
        progress: ProgressState | None = None,
        cache: str | None = None,
        cache_threshold: float | None = None,
        attn: str | None = None,
        turbo: bool | None = None,
        # 出力 mp4 に音声ストリームを入れない (生成そのものは止まらない --
        # `_mux_mp4` の docstring 参照)。
        mute: bool = False,
        # 単発 generate_ref2va() と同じ per-request override。全場面共通
        # (references 自体が全場面共通なのと同じ理由、バッチ内で場面ごとに変える
        # ユースケースが無いため)。
        reference_image_short_edge: int | None = None,
    ) -> dict:
        """参照共通・プロンプト違いの ref2va 生成 N 本を、`H3_LOWVRAM=1` の固定費を
        バッチ全体で1回に償却して回す (`generate_still_batch()` の ref2va 版)。
        `still=True` なら超短尺→中央フレーム PNG (ref2i、キャラ一貫の場面静止画)、
        `still=False` なら通常尺の動画 (ref2va、`seconds` 必須・全場面共通)。

        位相並べ替え (generate_ref2va() の lowvram=1 choreography を位相順に再構成):

            entry   : [nothing big resident]  (transformer/transformer_ref とも解放)
            encode  : [TE-nf4]        全場面の setup + テキスト/参照ビジョンエンコード
            ref-enc : [TE-nf4 + vae]  VAE を1回だけ GPU に上げ、全場面の参照VAEエンコード
            layout  : [TE-nf4]        全場面の layout/latents/timesteps (TE 常駐が
                                      `_execution_device` 解決の前提 -- generate_ref2va の
                                      同名コメント参照)
            denoise : [transformer_ref] 1回ロードして全場面を順に
            decode  : [vae pair]      全場面をデコード → mp4 (still なら PNG も) を
                                      場面ごとに保存

        場面間の共有状態リセットは generate_still_batch() と同一
        (スケジューラ `_step_index=None` ×2 + per-scene FBC リセット。等価性は t2i_batch で
        逐次生成との mp4/PNG md5 一致により実証済みの手法)。スケジューラの
        sigmas/timesteps の**値**が全場面同一であることが前提なので、尺 (still_frames /
        seconds)・解像度・ステップ数は全場面共通 -- 変えられるのはプロンプトのみ。
        音声参照からの尺自動導出 (単発 ref2va の seconds=None) は場面間で尺が揃う保証が
        ないためバッチでは使えない。参照 (references) は全場面で共通 -- 参照の
        デコード/リサイズ (setup step) と VAE エンコードは場面ごとに再実行される
        (テンソル共有より単純で、状態の別名参照バグの余地がない)。

        seed は全場面共通。t2i_batch と同じく decode は場面ごとに保存しながら進むので
        途中失敗でも完了分は残る。`H3_LOWVRAM=1` 以外では呼ばない (app.py 側で逐次
        generate_ref2va() にフォールバック)。

        PR #14355 (f37ab93) migration note: same single-shell/`self._pipe_ref is
        self._pipe` shape as `generate_ref2va()` (see its own migration-note docstring
        paragraph) -- `MiniMaxH3PrepareConditionLatentsStep`/`MiniMaxH3Ref2VAPrepareLatentsStep`
        are inserted into the per-scene layout/latents/timesteps phase below in the same
        order `MiniMaxH3Ref2VACoreDenoiseStep`'s block list uses, and
        `MiniMaxH3AfterDenoiseStep` runs right after each scene's own denoise loop, before
        that scene's decode.
        """
        import core.settings as settings

        if not H3_LOWVRAM:
            raise RuntimeError(
                "generate_ref_batch() は H3_LOWVRAM=1 専用です。呼び出し側で逐次 "
                "generate_ref2va() にフォールバックしてください。"
            )
        if not prompts:
            raise ValueError("prompts が空です。")
        if not references:
            raise ValueError("ref batch needs at least one reference.")
        # 単発 ref2va と同じ理由で、TE用GPUが 24GB 未満なら参照バッチも成立しない
        # (`generate_ref2va()` の同じガードのコメント参照)。
        if self._te_external and not self._te_external_usable_for("ref2va"):
            raise ValueError(
                f"参照バッチは H3_TE_DEVICE={H3_TE_DEVICE!r} との併用ができません: "
                "参照ビジョンエンコードのぶん、TE用GPU に空き "
                f"{self._TE_EXTERNAL_MIN_FREE_GB_REF2VA_PROJ if H3_TE_PROJ else self._TE_EXTERNAL_MIN_FREE_GB_REF2VA_32B:.1f}GB "
                "以上が必要です(同居プロセスの使用分も空きを減らします)。"
                "そのGPUを空けるか、H3_TE_DEVICE を外して起動してください"
                "(t2i バッチは併用可能)。"
            )
        if still:
            if still_frames not in STILL_FRAME_CHOICES:
                raise ValueError(f"still_frames must be one of {STILL_FRAME_CHOICES}, got {still_frames}")
            if still_frames == 5 and not H3_VAE_SMALLCLIP_FIX:
                raise ValueError("still_frames=5 には H3_VAE_SMALLCLIP_FIX=1 (既定) が必要です。")
            batch_num_frames = still_frames
        else:
            if seconds is None:
                raise ValueError(
                    "ref2va_batch では seconds が必須です (音声参照からの尺自動導出は、"
                    "場面間で尺が揃う保証がないためバッチでは使えません)。"
                )
            batch_num_frames = seconds_to_num_frames(seconds)
        # Fail fast, before acquiring `self._load_lock` / doing any loading work below
        # (same early-validation placement as the other `raise ValueError`s just above).
        resolved_short_edge = _resolve_ref_image_short_edge(reference_image_short_edge)

        from diffusers.modular_pipelines.minimax_h3.before_denoise import (
            MiniMaxH3PrepareConditionLatentsStep,
            MiniMaxH3PrepareLatentsStep,
            MiniMaxH3Ref2VAPrepareLatentsStep,
            MiniMaxH3Ref2VAPrepareLayoutStep,
            MiniMaxH3SetTimestepsStep,
        )
        from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
        from diffusers.modular_pipelines.minimax_h3.decoders import (
            MiniMaxH3AfterDenoiseStep,
            MiniMaxH3AudioDecodeStep,
            MiniMaxH3VideoDecodeStep,
        )
        from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3Ref2VADenoiseStep
        from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VAReferenceEncoderStep
        from diffusers.modular_pipelines.modular_pipeline import PipelineState

        # PR #14355 後: `packing_ref2va.reference_kind` は削除 -- `entry.kind` を直接読む
        # (`generate_ref2va()` と同じ、references.py 参照)。
        kinds = [entry.kind for entry in references]
        if set(kinds) == {"audio"}:
            raise ValueError("An audio reference has to be paired with at least one image or video reference.")

        instant = settings.resolve_instant_settings(cache, cache_threshold, attn, turbo)
        t_start = time.time()
        n_scenes = len(prompts)

        # --- entry: generate_ref2va() の lowvram entry と同一 (何も常駐させない) ---
        with self._load_lock:
            self._free_transformer()
            self._free_transformer_ref()
            self._ensure_vaes(progress)
            self._load_text_encoder(progress)
            # TE ロード後に sync すること (generate_ref2va の H3_LOWVRAM バグ修正コメント参照:
            # 先に sync すると _pipe_ref.text_encoder が None のまま取り残される)。
            self._sync_shared_components_to_ref()
        torch.cuda.reset_peak_memory_stats()
        # バッチは場面ループの外・リクエスト先頭で1回 (generate_still_batch() と同じ
        # 理由 -- 全場面共通の turbo 状態を、以降の場面ループ内で回る
        # `MiniMaxH3SetTimestepsStep` より前に適用しておく)。
        self._apply_turbo_video_shift(instant["turbo"], is_ref=True)
        pipe = self._pipe_ref

        # Same per-request override as generate_ref2va() (see that call site's comment
        # for the mechanism / cache-safety note). Applied once here, before the
        # per-scene setup-step loop below, since `references` (and hence this value) is
        # constant across all scenes in a batch and `pipe` is the same shell each
        # iteration reuses.
        pipe.register_to_config(reference_image_short_edge=resolved_short_edge)
        if resolved_short_edge != H3_REF_IMAGE_SHORT_EDGE_DEFAULT:
            logger.info(
                "ref batch reference_image_short_edge=%d (default=%d)%s",
                resolved_short_edge, H3_REF_IMAGE_SHORT_EDGE_DEFAULT,
                " [per-request override]" if reference_image_short_edge is not None else " [H3_REF_IMAGE_SHORT_EDGE]",
            )

        # --- encode 位相 (TE 常駐): 全場面の setup + テキスト/参照ビジョンエンコード ---
        t_encode = time.time()
        scenes: list[dict] = []
        for idx, prompt in enumerate(prompts):
            # フェーズ境界での中断チェック(encoding): 場面ごとの setup ループ境界。
            interrupt_controller.check()
            if progress:
                progress.update(phase="encoding", message=f"場面 {idx + 1}/{n_scenes} をエンコード中...")
            state = PipelineState()
            state.set("prompt", prompt)
            state.set("references", references)
            state.set("height", height)
            state.set("width", width)
            state.set("num_frames", batch_num_frames)
            state.set("generator", torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None)
            state.set("num_inference_steps", num_inference_steps)
            state.set("output_type", "pt")
            state.set("attention_kwargs", None)
            state.set("latents", None)
            state.set("audio_latents", None)

            setup_step = MiniMaxH3Ref2VASetupStep()
            if still:
                with _relaxed_min_duration():
                    _, state = setup_step(pipe, state)
            else:
                _, state = setup_step(pipe, state)
            scenes.append({"prompt": prompt, "state": state})

        # --- テキストエンコード: 共有プレフィックス (H3_REF_PREFIX_CACHE=1、既定) か
        # 従来の場面ごとフル計算か。共有方式は参照ラベル+ビジョン (~4104トークン、
        # ~65s/場面) の Qwen3-VL 前方計算を1回にまとめ、場面ごとにはプロンプト末尾
        # (14-33トークン、~0.2s) だけを KV キャッシュ継続する -- 実測精度と検証手順は
        # H3_REF_PREFIX_CACHE のモジュールコメントと scripts/probe_ref_prefix_cache.py。
        # setup は場面ごとに再実行済みだが normalized_references はプロンプト非依存
        # (デコード/リサイズのみ) なので、先頭場面のものを代表としてプレフィックスに使う。
        # TE 外部常駐 (`H3_TE_DEVICE`): 単発の `generate_ref2va()` と同じく、エンコードは
        # `_te_attached()` の窓の中で・`self._encode_device` (= TE のいる側) に対して行い、
        # 結果だけを `_to_compute_device()` で計算用GPUへ運ぶ。この3点セットが無いと、
        # 窓の外で `pipe.text_encoder` が None のままエンコードに入り
        # `AttributeError: 'NoneType' object has no attribute 'config'`
        # (`_encode_ref_prompts_shared_prefix` の num_layers 参照) で落ちる。
        # **バッチ経路にだけこの対応が無く、上の `_te_external_usable_for("ref2va")` の
        # ガードは通す**ため、`H3_TE_DEVICE` 指定時の /api/ref2i_batch・/api/ref2va_batch は
        # 一度も成功していなかった (2026-08-20 実機で再現・修正。単発 ref2va 側で
        # 2026-08-13 に踏んだ「ガードを緩めた瞬間に発火する」のと同型)。
        # TE 同居時 (既定) は3つとも素通しで、挙動はバイト単位で不変。
        with self._te_attached():
            if H3_REF_PREFIX_CACHE:
                if progress:
                    progress.update(phase="encoding", message="参照プレフィックスをエンコード中 (全場面で共有)...")
                encoded = _encode_ref_prompts_shared_prefix(
                    pipe, prompts, scenes[0]["state"].get("normalized_references"),
                    device=self._encode_device, dtype=torch.bfloat16,
                )
                for scene, (prompt_embeds, text_token_tags) in zip(scenes, encoded):
                    prompt_embeds, text_token_tags = self._to_compute_device(prompt_embeds, text_token_tags)
                    scene["state"].set("prompt_embeds", prompt_embeds)
                    scene["state"].set("text_token_tags", text_token_tags)
                # フェーズ境界での中断チェック(encoding): 共有プレフィックスの
                # 前方計算が終わった直後(こちらは単発の長い呼び出しなのでループ境界ではなく
                # 呼び出し直後に1回)。
                interrupt_controller.check()
            else:
                for idx, scene in enumerate(scenes):
                    # フェーズ境界での中断チェック(encoding): 場面ごとのフォールバック
                    # エンコードループ境界。
                    interrupt_controller.check()
                    if progress:
                        progress.update(phase="encoding", message=f"場面 {idx + 1}/{n_scenes} をエンコード中...")
                    with torch.no_grad():
                        prompt_embeds, text_token_tags = _encode_ref2va_prompt(
                            pipe, scene["prompt"], scene["state"].get("normalized_references"),
                            device=self._encode_device, dtype=torch.bfloat16,
                        )
                    prompt_embeds, text_token_tags = self._to_compute_device(prompt_embeds, text_token_tags)
                    scene["state"].set("prompt_embeds", prompt_embeds)
                    scene["state"].set("text_token_tags", text_token_tags)

        # --- 参照VAEエンコード位相: VAE を1回だけ GPU へ (TE は常駐のまま --
        # 単発 generate_ref2va の lowvram 分岐と同じ同居構成で、48GB 級なら
        # TE(21GB)+vae(11GB) は問題なく収まることを単発実測で確認済み) ---
        self._vae_to_gpu()
        for idx, scene in enumerate(scenes):
            # フェーズ境界での中断チェック(encoding): 場面ごとの参照VAEエンコード
            # ループ境界。
            interrupt_controller.check()
            if progress:
                progress.update(phase="encoding", message=f"場面 {idx + 1}/{n_scenes} の参照をVAEエンコード中...")
            reference_encoder_step = MiniMaxH3Ref2VAReferenceEncoderStep()
            _, scene["state"] = reference_encoder_step(pipe, scene["state"])
        self._vae_to_cpu()

        # --- layout/condition latents/latents/ref2va latents/timesteps 位相 (TE まだ
        # 常駐 = `_execution_device` が正しく解決)。ブロック順は
        # `MiniMaxH3Ref2VACoreDenoiseStep`'s own block_classes (generate_ref2va() の
        # 同名コメント参照) と一致させる。---
        for scene in scenes:
            state = scene["state"]
            layout_step = MiniMaxH3Ref2VAPrepareLayoutStep()
            _, state = layout_step(pipe, state)
            condition_latents_step = MiniMaxH3PrepareConditionLatentsStep()
            _, state = condition_latents_step(pipe, state)
            latents_step = MiniMaxH3PrepareLatentsStep()
            _, state = latents_step(pipe, state)
            ref2va_latents_step = MiniMaxH3Ref2VAPrepareLatentsStep()
            _, state = ref2va_latents_step(pipe, state)
            timesteps_step = _make_set_timesteps_step()
            _, state = timesteps_step(pipe, state)
            scene["state"] = state
        encode_time = time.time() - t_encode

        # --- TE を解放して transformer_ref を1回だけロード ---
        with self._load_lock:
            self._free_text_encoder(force=True)
            self._ensure_transformer_ref(progress)
        self._ensure_hyperflow_ref(progress=progress)
        self.apply_instant_settings(self._pipe_ref.transformer_ref, instant, is_ref=True, progress=progress)

        # --- denoise 位相: 全場面を順に ---
        t_denoise = time.time()
        total_steps_all = num_inference_steps * n_scenes
        for idx, scene in enumerate(scenes):
            state = scene["state"]
            # PR #14355 対応: レイアウト段が (TE 常駐時の `_execution_device` 解決の下で)
            # CPU に作った state テンソルを計算用GPUへ (`_scene_state_to_compute` の
            # docstring 参照。generate_still_batch の t2va 版と同じ罠 -- transformer_ref
            # がまだロードされていない encode 位相で layout/latents/timesteps を回すため。
            # 既に GPU なら no-op)。
            self._scene_state_to_compute(state)
            # 場面間のスケジューラリセット (generate_still_batch の docstring 参照)。
            # scheduler/audio_scheduler は _sync_shared_components_to_ref で _pipe と
            # 同一オブジェクトを共有しているため、pipe(_ref) 側から触れば足りる。
            pipe.scheduler._step_index = None
            pipe.audio_scheduler._step_index = None

            step_times: list[float] = []
            cache_skips = [0]
            denoise_step = _maybe_hyperflowify_denoise_step(MiniMaxH3Ref2VADenoiseStep())
            orig_loop_step = denoise_step.loop_step

            def timed_loop_step(components, bstate, i, t, _idx=idx, _step_times=step_times, _skips=cache_skips):
                interrupt_controller.check()
                ts = time.time()
                result = orig_loop_step(components, bstate, i=i, t=t)
                _step_times.append(time.time() - ts)
                if instant["effective_cache"] == "fbc":
                    _skips[0] += self._fbc_last_step_was_skip_ref()
                if progress:
                    progress.update(
                        phase="denoising",
                        step=_idx * num_inference_steps + i + 1,
                        total_steps=total_steps_all,
                        message=f"場面 {_idx + 1}/{n_scenes} をデノイズ中 {i + 1}/{num_inference_steps}",
                    )
                return result

            denoise_step.loop_step = timed_loop_step
            if instant["effective_cache"] == "fbc":
                self._pipe_ref.transformer_ref._reset_stateful_cache()
                with self._pipe_ref.transformer_ref.cache_context("h3"):
                    _, state = denoise_step(pipe, state)
            else:
                _, state = denoise_step(pipe, state)
            # PR #14355 note: unpatchify separated into its own step now
            # (`MiniMaxH3AfterDenoiseStep`) -- see generate_ref2va()'s matching comment.
            # Has to run once per scene, right after that scene's own denoise loop and
            # before its decode below.
            after_denoise_step = MiniMaxH3AfterDenoiseStep()
            _, state = after_denoise_step(pipe, state)
            scene["state"] = state
            scene["denoise_time_s"] = round(sum(step_times), 2)
            scene["avg_step_time_s"] = round(sum(step_times) / len(step_times), 3) if step_times else None
            scene["cache_skipped_steps"] = cache_skips[0] if instant["effective_cache"] == "fbc" else None

        denoise_time = time.time() - t_denoise

        # --- decode 位相: transformer_ref を落として VAE で全場面をデコード ---
        if progress:
            progress.update(phase="decoding", message="全場面をデコード中...")
        self._free_transformer_ref()
        self._vae_to_gpu()
        t_decode = time.time()
        results: list[dict] = []
        try:
            for idx, scene in enumerate(scenes):
                if progress:
                    progress.update(phase="decoding", message=f"場面 {idx + 1}/{n_scenes} をデコード中...")
                state = scene["state"]
                video_decode_step = _cpu_norm_video_decode_step()
                _, state = video_decode_step(pipe, state)
                audio_decode_step = MiniMaxH3AudioDecodeStep()
                _, state = audio_decode_step(pipe, state)

                videos = state.get("videos")
                audio = state.get("audio")
                sampling_rate = state.get("sampling_rate")
                video_tensor = videos[0] if isinstance(videos, list) else videos
                # 全長ぶんの中間テンソルを GPU に積まないよう、フレームを小分けにして
                # CPU の出力配列へ直接書き込む (frames_to_uint8 の docstring 参照)。
                frames_uint8 = frames_to_uint8(video_tensor)
                audio_np = audio[0].float().cpu().numpy()
                del video_tensor, videos, audio
                gc.collect()
                torch.cuda.empty_cache()

                job_stub = f"{'ref2i' if still else 'ref2va'}_{int(t_start)}_s{idx + 1}"
                mp4_path = self.output_dir / f"{job_stub}.mp4"
                _mux_mp4(frames_uint8, audio_np, sampling_rate, FPS, mp4_path, mute=mute)
                png_path = None
                still_frame_index = None
                if still:
                    still_frame_index = len(frames_uint8) // 2
                    png_path = self.output_dir / f"{job_stub}.png"
                    Image.fromarray(frames_uint8[still_frame_index]).save(png_path)

                results.append({
                    "prompt": scene["prompt"],
                    "png_path": str(png_path) if png_path else None,
                    "png_filename": png_path.name if png_path else None,
                    "mp4_path": str(mp4_path),
                    "mp4_filename": mp4_path.name,
                    "still_frame_index": still_frame_index,
                    "num_frames": len(frames_uint8),
                    "denoise_time_s": scene["denoise_time_s"],
                    "avg_step_time_s": scene["avg_step_time_s"],
                    "cache_skipped_steps": scene["cache_skipped_steps"],
                })
        except BaseException:
            logger.exception(
                "ref batch decode failed (scene %d/%d) -- restoring steady state before re-raising",
                len(results) + 1, n_scenes,
            )
            gc.collect()
            torch.cuda.empty_cache()
            try:
                self._vae_to_cpu()
            except Exception:
                logger.exception("steady-state restore after ref batch decode failure also failed")
            raise
        decode_time = time.time() - t_decode
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        self._vae_to_cpu()
        # lowvram=1: 何も再ロードしない (generate_ref2va の decode 後と同じ定常状態)

        total_elapsed = time.time() - t_start
        result = {
            "mode": "ref2i_batch" if still else "ref2va_batch",
            "num_scenes": n_scenes,
            "height": height,
            "width": width,
            "num_frames": batch_num_frames,
            "still_frames": still_frames if still else None,
            "duration_s": batch_num_frames / FPS,
            "seconds": seconds if not still else None,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "encode_time_s": round(encode_time, 2),
            "denoise_time_s": round(denoise_time, 2),
            "decode_time_s": round(decode_time, 2),
            "peak_vram_gb": round(peak_vram, 2),
            "ram": ram_gb(),
            "total_elapsed_s": round(total_elapsed, 2),
            "per_image_s": round(total_elapsed / n_scenes, 2),
            "te_quant": TE_QUANT,
            "transformer_quant": H3_TRANSFORMER_QUANT,
            "lowvram": H3_LOWVRAM_RAW,
            "attn_backend": instant["attn"],
            "cache": instant["effective_cache"],
            "cache_threshold": instant["cache_threshold"] if instant["effective_cache"] == "fbc" else None,
            "turbo": instant["turbo"],
            "mute": bool(mute),
            "reference_image_short_edge": resolved_short_edge,
            "references_summary": [
                {"index": index, "kind": kind, "has_audio": bool(references[index].has_audio)}
                for index, kind in enumerate(kinds)
            ],
            "scenes": results,
        }
        if progress:
            last_path = (results[-1]["png_path"] or results[-1]["mp4_path"]) if results else None
            progress.update(phase="done", message=f"完了 ({n_scenes}場面)", result_path=last_path)
        logger.info("%s done: %s", result["mode"],
                     json.dumps({k: v for k, v in result.items() if k not in ("ram", "scenes")}, ensure_ascii=False))
        return result


def _mux_mp4(frames_uint8: np.ndarray, audio_np: np.ndarray, sampling_rate: int, fps: int, mp4_path: Path,
             mute: bool = False):
    """デコード済みフレームと音声を mp4 に多重化する。

    `mute=True` のときは**音声ストリームを一切作らない**(映像のみの mp4)。
    H3 は動画と音声を同一の transformer forward で同時に生成する
    (`denoise.py` が `audio_hidden_states` を毎ステップ渡す)omni-modal モデルなので、
    **音声の生成そのものは止められない** -- このフラグができるのは「生成された音声を
    出力コンテナに入れない」ところまでで、**デノイズ時間は1秒も変わらない**。
    速度目的で使うものではなく、「確実に無音の成果物が欲しい」用途のためのもの。
    (プロンプト側で `overall_soundscape` に無音を指示する手もあるが、実測では
    audio_rms が 0 にはならず完全な無音は保証できない -- README 参照。)
    """
    import av

    container = av.open(str(mp4_path), mode="w")
    vstream = container.add_stream("libx264", rate=fps)
    vstream.width = frames_uint8.shape[2]
    vstream.height = frames_uint8.shape[1]
    vstream.pix_fmt = "yuv420p"

    astream = None
    if not mute:
        astream = container.add_stream("aac", rate=sampling_rate)
        astream.layout = "stereo"

    for frame in frames_uint8:
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in vstream.encode(av_frame):
            container.mux(packet)
    for packet in vstream.encode():
        container.mux(packet)

    if astream is not None:
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

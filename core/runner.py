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

# "none" (default) = current per-request TE<->transformer GPU swap.
# "bnb-4bit" = TE quantized NF4, TE+transformer both resident permanently, VAEs cycle
# through GPU per-phase instead. See module docstring above.
TE_QUANT = os.environ.get("H3_TE_QUANT", "bnb-4bit").strip().lower()
if TE_QUANT not in ("none", "bnb-4bit"):
    raise ValueError(f"H3_TE_QUANT must be 'none' or 'bnb-4bit', got {TE_QUANT!r}")

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

# EXPERIMENTAL, opt-in. `H3_LOWVRAM=1` は毎リクエスト、デコード直前に transformer を
# 解放し次リクエストで再ロードする (実測 14.8-32.7s の固定費、RESIDENCY.md §5.5)。
# `H3_KEEP_TRANSFORMER=1` はその解放をスキップし、transformer をリクエスト間も
# GPU に常駐させたままにする -- 48GB級 (実効予算 ~49.8GB) では
# transformer(int8) 34.3GB + デコードピーク **fp16** 11.4GB = 45.7GB で入る見込み
# (未検証、余裕 ~4GB)。fp32 デコードピーク 16.29GB では 34.3+16.29=50.6GB で
# 入らないため、H3_VIDEO_VAE_FP16=1 が必須 (RESIDENCY.md §5.6)。
# 成立条件 (3つとも必須、欠けたら import 時に RuntimeError):
#   1) H3_LOWVRAM == True (raw "1" のみ。"group" は対象外 -- このフラグの目的は
#      lowvram=1 の毎リクエスト再ロード固定費の削減であって、group モードは
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
    if not H3_LOWVRAM:
        _keep_transformer_missing.append(
            f"H3_LOWVRAM must be '1' (got {H3_LOWVRAM_RAW!r}) -- this flag only removes "
            "H3_LOWVRAM=1's per-request transformer reload cost; 'group' mode already "
            "keeps its transformer resident via a different (CPU+block-offload) design "
            "and is unrelated"
        )
    if not H3_TE_DEVICE:
        _keep_transformer_missing.append(
            "H3_TE_DEVICE must be set (TE on a separate GPU) -- otherwise the *encode* "
            "phase (not decode) breaks first: TE-nf4 17.45GB + resident transformer-int8 "
            "34.3GB = 51.75GB, over the ~49.8GB effective budget"
        )
    if not H3_VIDEO_VAE_FP16:
        _keep_transformer_missing.append(
            "H3_VIDEO_VAE_FP16 must be '1' -- fp32 decode peak 16.29GB + resident "
            "transformer-int8 34.3GB = 50.6GB, over the ~49.8GB effective budget "
            "(fp16 decode peak ~11.4GB fits: 34.3+11.4=45.7GB)"
        )
    if _keep_transformer_missing:
        raise RuntimeError(
            "H3_KEEP_TRANSFORMER=1 requires H3_LOWVRAM=1 AND H3_TE_DEVICE set AND "
            "H3_VIDEO_VAE_FP16=1 (see this flag's module comment for the VRAM budget "
            "derivation). Missing: " + "; ".join(_keep_transformer_missing)
        )

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
H3_TURBO_LORA_REPO = os.environ.get("H3_TURBO_LORA_REPO", "lightx2v/Minimax-h3-Turbo")
H3_TURBO_LORA_FILE = os.environ.get("H3_TURBO_LORA_FILE", "minimax_h3_fl2v_turbo_4step_v0.1.safetensors")

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

# MINIMAX_H3_MIN_DURATION..MAX_DURATION = 5..15s at 24fps, aligned to 17*n+5.
MIN_SECONDS = 5.0
MAX_SECONDS = 15.0
FPS = 24

# 静止画モード (`generate(still=True)`) のフレーム数の選択肢。値は align_num_frames の
# 17n+5 制約を満たす最小の2つ: 22 (0.917s) は 2026-08-07 の超短尺プローブ
# (scripts/probe_short_frames*.py) で品質が5秒基準と遜色ないことを目視確認済みの既定値。
# 5 (0.208s) は学習分布からさらに外れる実験値で、デコードには下の
# `_patch_vae_smallclip_decode()` (H3_VAE_SMALLCLIP_FIX) が必須(潜在2フレームは
# 上流の `AutoencoderKLMiniMaxH3._decode()` のチャンク境界処理で num_chunks=0 になり
# `torch.cat([])` で落ちる)。
STILL_FRAME_CHOICES = (22, 5)

# Opt-out ("1" default). "1" = `AutoencoderKLMiniMaxH3._decode()` に「潜在フレームが
# 1チャンク未満 (num_chunks==0、潜在1-2フレーム)なら全トークンを単一の `_decode_clip()`
# で復号する」分岐を monkeypatch で追加する(下の `_patch_vae_smallclip_decode()`)。
# 通常の動画 (num_chunks>=1) はパッチ後も従来の実装へそのまま委譲されるため
# byte-for-byte 影響なし。"0" は静止画モードの frames=5 が上流バグそのままで落ちる
# 状態に戻すためのトグル(例外時クリーンアップの実機検証にも使った)。
H3_VAE_SMALLCLIP_FIX = os.environ.get("H3_VAE_SMALLCLIP_FIX", "1").strip() == "1"


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

    num_layers = components.text_encoder.config.text_config.num_hidden_layers
    if num_layers <= MINIMAX_H3_TEXT_ENCODER_LAYER:
        raise ValueError(
            f"MiniMax-H3 conditions on `hidden_states[{MINIMAX_H3_TEXT_ENCODER_LAYER}]` of its Qwen3-VL "
            f"conditioner, which needs more than {MINIMAX_H3_TEXT_ENCODER_LAYER} decoder layers, but "
            f"`text_encoder` has {num_layers}."
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
    prefix_hidden = prefix_out.hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER].to(device=device, dtype=dtype)
    prefix_len = cache.get_seq_length()
    logger.info(
        "shared-prefix encode: prefix %d tokens in %.1fs, continuing %d scene prompt(s)",
        prefix_len, time.time() - t_prefix, len(prompts),
    )

    results: list[tuple[torch.Tensor, torch.Tensor]] = []
    for prompt in prompts:
        suffix_ids = components.tokenizer(prompt, add_special_tokens=False)["input_ids"]
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
        suffix_hidden = suffix_out.hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER].to(device=device, dtype=dtype)
        cache.crop(prefix_len)

        prompt_embeds = torch.cat([prefix_hidden, suffix_hidden], dim=1)
        # プロンプトはビジョンブロックを含まない純テキストの末尾セグメントなので、
        # suffix のタグは全て components.text_tag。
        text_token_tags = torch.tensor(
            list(prefix_tags) + [components.text_tag] * len(suffix_ids), dtype=torch.long
        )
        results.append((prompt_embeds, text_token_tags))

    return results


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

    prompt_embeds = get_qwen3vl_prompt_embeds(
        components.text_encoder,
        processor,
        token_ids,
        vision_inputs,
        text_encoder_layer=components.text_encoder_layer,
        device=device,
        dtype=dtype,
    )
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

    step = MiniMaxH3Ref2VATextEncoderStep()
    vision_inputs, image_token_counts, video_token_counts, video_timestamps = step._gather_vision_features(
        components.processor, normalized_references, components.fps
    )
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
    prompt_embeds = get_qwen3vl_prompt_embeds(
        components.text_encoder,
        components.processor,
        token_ids,
        vision_inputs,
        text_encoder_layer=components.text_encoder_layer,
        device=device,
        dtype=dtype,
    )
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


def resolve_turbo_lora_scale(lora_format: str) -> float:
    """適用係数を解決する: H3_TURBO_LORA_SCALE が明示されていればそれ、空なら
    形式ごとの実測既定 (comfy=1.0 / diffusers=0.094 -- 導出と強度スイープの実測は
    H3_TURBO_LORA_SCALE のモジュールコメントと README を参照)。"""
    if H3_TURBO_LORA_SCALE_RAW:
        return float(H3_TURBO_LORA_SCALE_RAW)
    return 1.0 if lora_format == "comfy" else 0.094


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
        if H3_VIDEO_VAE_FP16:
            # `dtype=torch.float16` on the load_components call above would be a no-op
            # for this VAE (see H3_VIDEO_VAE_FP16's module comment) -- has to be a
            # manual cast after the fp32 load. audio_vae is deliberately excluded.
            t_cast = time.time()
            self._pipe.vae = self._pipe.vae.to(torch.float16)
            logger.info("video vae cast to float16 in %.2fs (H3_VIDEO_VAE_FP16=1)",
                        time.time() - t_cast)
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
        if H3_TURBO_LORA:
            # Applied right after load, before `set_attention_backend`/FBC: LoRA wrapping
            # only replaces Linear *modules* (`attn.to_qkv`, `ff.net.0.proj`, etc, see
            # `apply_turbo_lora`'s docstring), it does not touch attention dispatch or
            # cache hooks, so ordering against those two calls does not matter -- done
            # first here only because it is the more fundamental structural change of the
            # three. `H3_CACHE == "fbc"` is force-skipped below (not just "left at its
            # default") regardless of the env var's own value -- see `H3_TURBO_LORA`'s
            # module comment for why a handful of turbo steps leaves FBC no safe window.
            n = self._apply_turbo_lora_checkpoint(self._pipe.transformer)
            self._turbo_lora_wrapped = True
            logger.info("H3_TURBO_LORA=1: applied turbo LoRA (%d layers wrapped), FBC force-disabled", n)
        if H3_ATTN_BACKEND:
            self._pipe.transformer.set_attention_backend(H3_ATTN_BACKEND)
            logger.info("transformer attention backend set to %r", H3_ATTN_BACKEND)
        if H3_CACHE == "fbc" and not H3_TURBO_LORA:
            self._enable_fbc()
        logger.info(
            "transformer loaded to GPU in %.1fs (quant=%s, turbo_lora=%s). gpu=%s ram=%s",
            time.time() - t0, H3_TRANSFORMER_QUANT, H3_TURBO_LORA, gpu_mem_gb(), ram_gb(),
        )

    def _apply_turbo_lora_checkpoint(self, transformer) -> int:
        """設定済みの turbo LoRA をダウンロード → キー形式を判定 → 形式に応じた適用
        関数へディスパッチする (`_ensure_transformer` の起動時適用と
        `_apply_turbo_setting` の遅延適用、両呼び出し元の共通化)。

        comfy 形式 (Ostris 版) × int8 はここで明確に拒否する -- import 時のガードは
        リポジトリ名の予備判定 (`_TURBO_COMFY_REPOS`) しかできないため、未知リポジトリの
        comfy 形式チェックポイントはこの実ファイル判定が最後の砦。"""
        self._download_turbo_lora_if_needed()
        lora_format = detect_turbo_lora_format(self._turbo_lora_path)
        if lora_format == "comfy":
            if H3_TRANSFORMER_QUANT == "int8":
                raise ValueError(
                    "comfy形式 (融合QKV) の turbo LoRA は int8 transformer に適用できません "
                    "(fuse_projections() の torch.cat が Int8Tensor 非対応)。既定の "
                    "diffusers ネイティブ形式 (lightx2v/Minimax-h3-Turbo) を使うか、"
                    "int8/低VRAM を無効にしてください。"
                )
            return apply_turbo_lora(transformer, self._turbo_lora_path)
        return apply_diffusers_turbo_lora(
            transformer, self._turbo_lora_path, resolve_turbo_lora_scale(lora_format)
        )

    def _download_turbo_lora_if_needed(self):
        """Resolve (downloading if necessary, via the normal HF cache) the turbo LoRA
        safetensors path once per process, caching it on `self._turbo_lora_path`. Split
        out from `_ensure_transformer` only so the download (network I/O, ~780MB) is not
        interleaved with that method's own docstring-documented load-order reasoning.
        """
        if getattr(self, "_turbo_lora_path", None) is not None:
            return
        from huggingface_hub import hf_hub_download

        t0 = time.time()
        self._turbo_lora_path = hf_hub_download(H3_TURBO_LORA_REPO, H3_TURBO_LORA_FILE)
        logger.info(
            "turbo LoRA checkpoint resolved: %s (%.1fs, repo=%s file=%s)",
            self._turbo_lora_path, time.time() - t0, H3_TURBO_LORA_REPO, H3_TURBO_LORA_FILE,
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
            Int8WeightOnlyConfig(version=2),
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
            Int8WeightOnlyConfig(version=2),
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

    def _free_transformer_ref(self):
        if not self._transformer_ref_loaded:
            return
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
        logger.info("transformer_ref freed. gpu=%s ram=%s", gpu_mem_gb(), ram_gb())

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
            n = self._apply_turbo_lora_checkpoint(transformer)
            setattr(self, wrapped_attr, True)
            logger.info("turbo LoRA lazily applied to %s (%d layers wrapped)", label, n)
        elif getattr(self, wrapped_attr):
            n = set_turbo_lora_enabled(transformer, turbo)
            logger.debug("%s: turbo LoRA %s (%d wrapper modules)", label, "enabled" if turbo else "disabled", n)
        # else: turbo requested False and it was never wrapped in the first place --
        # nothing to do, the transformer's Linears are still the plain unwrapped ones.

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

    def _te_external_usable_for(self, mode: str) -> bool:
        """このモードで TE 外部常駐を使ってよいか。

        ref2va は 2048px 短辺の参照を vision tower に通すため活性化が大きく、20GB 級では
        OOM することを実測済み(`H3_TE_DEVICE` のコメント参照)。TE 用GPUの総容量が
        24GB 未満なら ref2va は従来経路へフォールバックする — 「動くはず」で走らせて
        OOM させるより、確実に動く経路を選ぶ。
        """
        if not self._te_external:
            return False
        if mode != "ref2va":
            return True
        try:
            total_gb = torch.cuda.get_device_properties(torch.device(H3_TE_DEVICE).index).total_memory / 1e9
        except Exception:
            return False
        return total_gb >= 24.0

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
        self._detach_te_if_external()
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
        """
        self._ensure_pipe_shell()
        if self._text_encoder_loaded:
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
            self._detach_te_if_external()
            # 初回のみ: 量子化済みの重みを保存しておき、次回以降のロードを短縮する。
            if H3_TE_PREQUANT:
                self._save_te_prequant(cache_dir)
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
        del self._pipe.text_encoder
        self._pipe.text_encoder = None
        if self._pipe_ref is not None and getattr(self._pipe_ref, "text_encoder", None) is not None:
            del self._pipe_ref.text_encoder
            self._pipe_ref.text_encoder = None
        self._text_encoder_loaded = False
        gc.collect()
        torch.cuda.empty_cache()
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
        self._vae_loaded = False
        self._vae_on_gpu = False
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
        self._free_text_encoder(force=True)
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
            self._ensure_vaes()
            if H3_LOWVRAM_GROUP:
                self._ensure_transformer()
            elif not H3_LOWVRAM:
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
            "te_prune": H3_TE_PRUNE,
            "video_vae_fp16": H3_VIDEO_VAE_FP16,
            "transformer_quant": H3_TRANSFORMER_QUANT,
            "lowvram": H3_LOWVRAM_RAW,
            "lowvram_group": H3_LOWVRAM_GROUP,
            "group_offload_blocks": H3_GROUP_OFFLOAD_BLOCKS if H3_LOWVRAM_GROUP else None,
            "group_offload_use_stream": H3_GROUP_OFFLOAD_USE_STREAM if H3_LOWVRAM_GROUP else None,
            "group_offload_low_cpu_mem": H3_GROUP_OFFLOAD_LOW_CPU_MEM if H3_LOWVRAM_GROUP else None,
            "attn_backend": H3_ATTN_BACKEND or "default",
            "cache_mode": H3_CACHE if not H3_TURBO_LORA else "none (force-disabled by H3_TURBO_LORA)",
            "cache_threshold": H3_CACHE_THRESHOLD if (H3_CACHE == "fbc" and not H3_TURBO_LORA) else None,
            "turbo_lora": H3_TURBO_LORA,
            "turbo_lora_repo": H3_TURBO_LORA_REPO if H3_TURBO_LORA else None,
            "turbo_lora_path": self._turbo_lora_path,
            "turbo_steps_default": H3_TURBO_STEPS_DEFAULT if H3_TURBO_LORA else None,
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
        still: bool = False,
        still_frames: int = 22,
    ) -> dict:
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

        if progress:
            progress.update(phase="muxing", message="mp4へmux中...")
        if still:
            mode = "t2i"
        else:
            mode = "fl2va" if (image is not None or last_image is not None) else "t2va"
        job_stub = f"{mode}_{int(t_start)}"
        mp4_path = self.output_dir / f"{job_stub}.mp4"
        _mux_mp4(frames_uint8, audio_np, sampling_rate, FPS, mp4_path)

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
            self._free_transformer_ref()
            self._ensure_vaes(progress)
            self._load_text_encoder(progress)
        torch.cuda.reset_peak_memory_stats()
        pipe = self._pipe

        # --- encode 位相: TE 常駐のまま全場面を準備 ---
        # layout/latents/timesteps を TE 常駐中に回すのは generate() の lowvram 分岐と
        # 同じ理由 (`_execution_device` が text_encoder で解決される必要がある)。
        t_encode = time.time()
        scenes: list[dict] = []
        for idx, prompt in enumerate(prompts):
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
        with self._load_lock:
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
                video_decode_step = MiniMaxH3VideoDecodeStep()
                _, state = video_decode_step(pipe, state)
                audio_decode_step = MiniMaxH3AudioDecodeStep()
                _, state = audio_decode_step(pipe, state)

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
                del video_tensor, videos, audio
                gc.collect()
                torch.cuda.empty_cache()

                # 場面ごとに保存しながら進む (途中失敗でも完了済み場面は残る)
                job_stub = f"t2i_{int(t_start)}_s{idx + 1}"
                mp4_path = self.output_dir / f"{job_stub}.mp4"
                _mux_mp4(frames_uint8, audio_np, sampling_rate, FPS, mp4_path)
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
        still: bool = False,
        still_frames: int = 22,
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

        t_start = time.time()
        if not references:
            raise ValueError("ref2va needs at least one reference; use generate() for text-only requests.")
        # TE 外部常駐 (H3_TE_DEVICE) の TE用GPUが 24GB 未満なら ref2va は動かない:
        # 2048px 短辺の参照を vision tower に通す活性化が入らず OOM することを実測済み
        # (20GB カードで 19.25GB 使用中に 204MB 不足、`H3_TE_DEVICE` のコメント参照)。
        # 「動くはず」で走らせて OOM させるより、理由を添えて明確に拒否する。
        if self._te_external and not self._te_external_usable_for("ref2va"):
            raise ValueError(
                f"ref2va は H3_TE_DEVICE={H3_TE_DEVICE!r} との併用ができません "
                "(参照画像を vision tower に通す活性化が入らず OOM するため、TE用GPUには "
                "24GB 以上が必要 -- 20GB 級で実測確認済み)。ref2va を使うときは "
                "H3_TE_DEVICE を外して起動してください(t2va/fl2va/t2i は併用可能)。"
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

        # --- text encode (references' vision blocks + prompt; still has TE on GPU) ---
        if progress:
            progress.update(phase="encoding", message="プロンプト+参照をエンコード中...")
        with self._te_attached(), torch.no_grad():
            prompt_embeds, text_token_tags = _encode_ref2va_prompt(
                pipe, prompt, state.get("normalized_references"),
                device=self._encode_device, dtype=torch.bfloat16,
            )
        prompt_embeds, text_token_tags = self._to_compute_device(prompt_embeds, text_token_tags)
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
            reference_encoder_step = MiniMaxH3Ref2VAReferenceEncoderStep()
            _, state = reference_encoder_step(pipe, state)
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
            timesteps_step = MiniMaxH3SetTimestepsStep()
            _, state = timesteps_step(pipe, state)
        elif H3_LOWVRAM:
            self._vae_to_gpu()
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
            reference_encoder_step = MiniMaxH3Ref2VAReferenceEncoderStep()
            _, state = reference_encoder_step(pipe, state)
            self._vae_to_cpu()

            layout_step = MiniMaxH3Ref2VAPrepareLayoutStep()
            _, state = layout_step(pipe, state)
            condition_latents_step = MiniMaxH3PrepareConditionLatentsStep()
            _, state = condition_latents_step(pipe, state)
            latents_step = MiniMaxH3PrepareLatentsStep()
            _, state = latents_step(pipe, state)
            ref2va_latents_step = MiniMaxH3Ref2VAPrepareLatentsStep()
            _, state = ref2va_latents_step(pipe, state)
            timesteps_step = MiniMaxH3SetTimestepsStep()
            _, state = timesteps_step(pipe, state)

            with self._load_lock:
                self._free_text_encoder(force=True)
                self._ensure_transformer_ref(progress)
        elif TE_QUANT == "bnb-4bit":
            self._vae_to_gpu()
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

            # --- layout / condition latents / latents / ref2va latents / timesteps ---
            # Order matches `MiniMaxH3Ref2VACoreDenoiseStep`'s own block_classes list
            # (modular_blocks_minimax_h3.py) exactly: the conditioning noise (image/video
            # references) has to be drawn from the request's generator BEFORE the
            # generated rows' own noise, and packed into `latents`/`audio_latents` AFTER.
            layout_step = MiniMaxH3Ref2VAPrepareLayoutStep()
            _, state = layout_step(pipe, state)
            condition_latents_step = MiniMaxH3PrepareConditionLatentsStep()
            _, state = condition_latents_step(pipe, state)
            latents_step = MiniMaxH3PrepareLatentsStep()
            _, state = latents_step(pipe, state)
            ref2va_latents_step = MiniMaxH3Ref2VAPrepareLatentsStep()
            _, state = ref2va_latents_step(pipe, state)
            timesteps_step = MiniMaxH3SetTimestepsStep()
            _, state = timesteps_step(pipe, state)
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
            reference_encoder_step = MiniMaxH3Ref2VAReferenceEncoderStep()
            _, state = reference_encoder_step(pipe, state)

            # --- layout / condition latents / latents / ref2va latents / timesteps ---
            layout_step = MiniMaxH3Ref2VAPrepareLayoutStep()
            _, state = layout_step(pipe, state)
            condition_latents_step = MiniMaxH3PrepareConditionLatentsStep()
            _, state = condition_latents_step(pipe, state)
            latents_step = MiniMaxH3PrepareLatentsStep()
            _, state = latents_step(pipe, state)
            ref2va_latents_step = MiniMaxH3Ref2VAPrepareLatentsStep()
            _, state = ref2va_latents_step(pipe, state)
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

        # --- denoise loop, instrumented for progress polling (mirrors generate()'s
        # non-upscale path exactly, against transformer_ref instead of transformer) ---
        if progress:
            progress.update(phase="denoising", step=0, total_steps=num_inference_steps, message="デノイズ中...")
        t_denoise = time.time()
        step_times = []
        cache_skips = [0]
        out_height, out_width = state.get("height"), state.get("width")

        # Instant-apply this request's cache/attn/turbo settings -- see generate()'s
        # matching comment for the full reasoning. `transformer_ref` is confirmed
        # resident by every branch above this point.
        self.apply_instant_settings(self._pipe_ref.transformer_ref, instant, is_ref=True, progress=progress)

        def _fbc_reset_and_context():
            self._pipe_ref.transformer_ref._reset_stateful_cache()
            return self._pipe_ref.transformer_ref.cache_context("h3")

        denoise_step = MiniMaxH3Ref2VADenoiseStep()
        orig_loop_step = denoise_step.loop_step

        def timed_loop_step(components, bstate, i, t):
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

        # PR #14355 note: unpatchify is a separate step now (`MiniMaxH3AfterDenoiseStep`,
        # decoders.py) -- see `generate()`'s matching comment for the full contract. Has to
        # run once, right after the denoise loop and before either decode step below;
        # matches where `after_denoise` sits in `MiniMaxH3Ref2VACoreDenoiseStep`'s own
        # block list (modular_blocks_minimax_h3.py). Unlike t2va/fl2va,
        # `num_condition_video_rows`/`num_condition_audio_rows` on `state` (set by the
        # layout step above) are NOT zero here -- a reference always adds condition rows,
        # which is exactly what this step drops before reshaping the generated rows back
        # into a 5D video tensor / channel-major audio tensor.
        after_denoise_step = MiniMaxH3AfterDenoiseStep()
        _, state = after_denoise_step(pipe, state)

        # --- decode (shared MiniMaxH3VideoDecodeStep/MiniMaxH3AudioDecodeStep -- no
        # ref2va-specific decode step exists; `MiniMaxH3AfterDenoiseStep` just above
        # already dropped the reference condition rows, so these two only ever see the
        # generated rows) ---
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

        if TE_QUANT == "bnb-4bit" and not H3_TRANSFORMER_BOTH_RESIDENT and not H3_LOWVRAM_GROUP:
            self._free_transformer_ref()
        elif H3_LOWVRAM_GROUP:
            with self._load_lock:
                self._free_text_encoder(force=True)
        self._vae_to_gpu()
        t_decode = time.time()
        try:
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
        _mux_mp4(frames_uint8, audio_np, sampling_rate, FPS, mp4_path)

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
            "cache_skipped_steps": cache_skips[0] if instant["effective_cache"] == "fbc" else None,
            "references_summary": [
                {"index": index, "kind": kind, "has_audio": bool(references[index].has_audio)}
                for index, kind in enumerate(kinds)
            ],
        }
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
                f"参照バッチは H3_TE_DEVICE={H3_TE_DEVICE!r} との併用ができません "
                "(TE用GPUに 24GB 以上が必要 -- 20GB 級で OOM を実測確認済み)。"
                "H3_TE_DEVICE を外して起動してください(t2i バッチは併用可能)。"
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
        pipe = self._pipe_ref

        # --- encode 位相 (TE 常駐): 全場面の setup + テキスト/参照ビジョンエンコード ---
        t_encode = time.time()
        scenes: list[dict] = []
        for idx, prompt in enumerate(prompts):
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
        if H3_REF_PREFIX_CACHE:
            if progress:
                progress.update(phase="encoding", message="参照プレフィックスをエンコード中 (全場面で共有)...")
            encoded = _encode_ref_prompts_shared_prefix(
                pipe, prompts, scenes[0]["state"].get("normalized_references"),
                device=DEVICE, dtype=torch.bfloat16,
            )
            for scene, (prompt_embeds, text_token_tags) in zip(scenes, encoded):
                scene["state"].set("prompt_embeds", prompt_embeds)
                scene["state"].set("text_token_tags", text_token_tags)
        else:
            for idx, scene in enumerate(scenes):
                if progress:
                    progress.update(phase="encoding", message=f"場面 {idx + 1}/{n_scenes} をエンコード中...")
                with torch.no_grad():
                    prompt_embeds, text_token_tags = _encode_ref2va_prompt(
                        pipe, scene["prompt"], scene["state"].get("normalized_references"),
                        device=DEVICE, dtype=torch.bfloat16,
                    )
                scene["state"].set("prompt_embeds", prompt_embeds)
                scene["state"].set("text_token_tags", text_token_tags)

        # --- 参照VAEエンコード位相: VAE を1回だけ GPU へ (TE は常駐のまま --
        # 単発 generate_ref2va の lowvram 分岐と同じ同居構成で、48GB 級なら
        # TE(21GB)+vae(11GB) は問題なく収まることを単発実測で確認済み) ---
        self._vae_to_gpu()
        for idx, scene in enumerate(scenes):
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
            timesteps_step = MiniMaxH3SetTimestepsStep()
            _, state = timesteps_step(pipe, state)
            scene["state"] = state
        encode_time = time.time() - t_encode

        # --- TE を解放して transformer_ref を1回だけロード ---
        with self._load_lock:
            self._free_text_encoder(force=True)
            self._ensure_transformer_ref(progress)
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
            denoise_step = MiniMaxH3Ref2VADenoiseStep()
            orig_loop_step = denoise_step.loop_step

            def timed_loop_step(components, bstate, i, t, _idx=idx, _step_times=step_times, _skips=cache_skips):
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
                video_decode_step = MiniMaxH3VideoDecodeStep()
                _, state = video_decode_step(pipe, state)
                audio_decode_step = MiniMaxH3AudioDecodeStep()
                _, state = audio_decode_step(pipe, state)

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
                del video_tensor, videos, audio
                gc.collect()
                torch.cuda.empty_cache()

                job_stub = f"{'ref2i' if still else 'ref2va'}_{int(t_start)}_s{idx + 1}"
                mp4_path = self.output_dir / f"{job_stub}.mp4"
                _mux_mp4(frames_uint8, audio_np, sampling_rate, FPS, mp4_path)
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

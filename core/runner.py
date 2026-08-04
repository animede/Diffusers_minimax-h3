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

import numpy as np
import torch
from PIL import Image

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

# MINIMAX_H3_MIN_DURATION..MAX_DURATION = 5..15s at 24fps, aligned to 17*n+5.
MIN_SECONDS = 5.0
MAX_SECONDS = 15.0
FPS = 24


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
        """Load the 66GB bf16 transformer to GPU.

        `none` mode: frees the text_encoder first if resident (they cannot coexist).
        `bnb-4bit` mode: TE-nf4 is permanently resident, nothing to free here. Called at
        startup, and again after every request's decode phase (which drops the
        transformer for its ~9s window -- see the decode section of `generate()`) to
        restore the transformer+TE-nf4 steady state between requests.
        """
        self._ensure_pipe_shell()
        if self._transformer_loaded:
            return
        if TE_QUANT != "bnb-4bit":
            # TE (66GB) + transformer (66GB) cannot coexist in 96GB VRAM.
            self._free_text_encoder()
        if progress:
            progress.update(phase="loading_transformer", message="transformer をロード中...")
        t0 = time.time()
        self._pipe.load_components(names=["transformer"], dtype=torch.bfloat16)
        self._pipe.transformer.to(DEVICE)
        self._transformer_loaded = True
        logger.info("transformer loaded to GPU in %.1fs. gpu=%s ram=%s", time.time() - t0, gpu_mem_gb(), ram_gb())

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

    def _free_text_encoder(self):
        if not self._text_encoder_loaded:
            return
        if TE_QUANT == "bnb-4bit":
            # Permanently resident in this mode -- never freed mid-run (see
            # _load_text_encoder docstring). Guard so a stray call is a harmless no-op
            # rather than silently dropping the model.
            logger.debug("bnb-4bit text_encoder is permanently resident; ignoring free request")
            return
        # Drop the CUDA model directly: releasing the last reference frees the VRAM in
        # place. Do NOT stage through .to("cpu") first -- the text_encoder is ~66GB
        # (the checkpoint is bf16-native, not fp32), and a host-RAM transit would both
        # waste time and evict the page-cached model shards that make the next
        # per-request reload fast.
        del self._pipe.text_encoder
        self._pipe.text_encoder = None
        self._text_encoder_loaded = False
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("text_encoder freed. gpu=%s ram=%s", gpu_mem_gb(), ram_gb())

    def preload_all(self):
        """Load the steady-state residents once at startup.

        `none` mode: transformer + VAEs (the text_encoder cycles per request, so
        preloading it would only be churn).
        `bnb-4bit` mode: transformer + text_encoder(NF4) + VAEs are ALL loaded here --
        the VAEs' weights are loaded now (onto CPU, see _ensure_vaes) and the TE is
        loaded straight to GPU permanently, since nothing cycles anymore in this mode.
        """
        with self._load_lock:
            self._ensure_vaes()
            self._ensure_transformer()
            if TE_QUANT == "bnb-4bit":
                self._load_text_encoder()

    def status(self) -> dict:
        return {
            "pipe_built": self._pipe is not None,
            "transformer_loaded": self._transformer_loaded,
            "vae_loaded": self._vae_loaded,
            "vae_on_gpu": self._vae_on_gpu,
            "text_encoder_loaded": self._text_encoder_loaded,
            "te_quant": TE_QUANT,
            "gpu": gpu_mem_gb(),
            "ram": ram_gb(),
        }

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
    ) -> dict:
        """
        Runs T2VA (image=None, last_image=None) or FL2VA (either/both given).

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
        from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3DenoiseStep
        from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3TextEncoderStep
        from diffusers.modular_pipelines.modular_pipeline import PipelineState

        t_start = time.time()
        num_frames = seconds_to_num_frames(seconds)

        with self._load_lock:
            # `none` mode: VAEs (permanent residents) + text encoder. _load_text_encoder
            # frees the transformer internally if it is resident (TE 66GB + transformer
            # 66GB cannot coexist in 96GB VRAM).
            # `bnb-4bit` mode: everything is already resident from preload_all() except
            # the VAEs, which are parked on CPU -- nothing to do here, they get moved to
            # GPU right before the phase that needs them, below.
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

        if TE_QUANT == "bnb-4bit" and is_fl2va:
            # bnb-4bit + fl2va only: transformer(66.3) + TE-nf4(21.0) + vae pair(11.0)
            # already sums to ~98.3GB before any activation buffer, over this card's
            # ~95.6GB (the same three-way conflict measured for decode, see the decode
            # section below and the module docstring) -- so the keyframe VAE-encode step
            # (which needs `vae` on GPU, already brought in above) has to run *before*
            # the transformer is loaded, not after. TE stays resident throughout (it is
            # not involved in this step).
            keyframe_step = MiniMaxH3AutoKeyframeVaeEncoderStep()
            _, state = keyframe_step(pipe, state)
            self._vae_to_cpu()
            with self._load_lock:
                self._free_text_encoder()  # no-op in this mode; kept for symmetry
                self._ensure_transformer(progress)
        else:
            # `none` mode: TE's job is done for this request -- free it and bring in the
            # transformer (which stays resident until the next request's encode phase
            # kicks it out again).
            # `bnb-4bit` + t2va: TE is permanently resident (free is a no-op) and the
            # transformer is normally already resident too -- except right after a
            # previous request's decode phase dropped it (see the decode section below),
            # in which case this is the reload that restores it before denoise. No vae
            # conflict here since t2va's vae never went to GPU in the first place.
            with self._load_lock:
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
        denoise_step = MiniMaxH3DenoiseStep()
        orig_loop_step = denoise_step.loop_step
        step_times = []

        def timed_loop_step(components, bstate, i, t):
            ts = time.time()
            result = orig_loop_step(components, bstate, i=i, t=t)
            step_times.append(time.time() - ts)
            if progress:
                progress.update(step=i + 1, message=f"デノイズ中 {i + 1}/{num_inference_steps}")
            return result

        denoise_step.loop_step = timed_loop_step
        _, state = denoise_step(pipe, state)
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
        if TE_QUANT == "bnb-4bit":
            with self._load_lock:
                self._ensure_transformer(progress)

        if progress:
            progress.update(phase="muxing", message="mp4へmux中...")
        mode = "fl2va" if (image is not None or last_image is not None) else "t2va"
        job_stub = f"{mode}_{int(t_start)}"
        mp4_path = self.output_dir / f"{job_stub}.mp4"
        _mux_mp4(frames_uint8, audio_np, sampling_rate, FPS, mp4_path)

        result = {
            "prompt": prompt,
            "height": height,
            "width": width,
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
        }
        if progress:
            progress.update(phase="done", message="完了", result_path=str(mp4_path))
        logger.info("generation done: %s", json.dumps({k: v for k, v in result.items() if k != "ram"}, ensure_ascii=False))
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

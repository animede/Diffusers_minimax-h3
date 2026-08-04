"""
MiniMax-H3 T2VA/FL2VA runner.

Loading strategy (see dev_notes/handoff-minimax-h3.md and diffusers-server CLAUDE.md
#33/#46/#47 for the constraints this follows):

- This box has 96GB VRAM but only ~94GB host RAM. The four big components add up to
  ~110GB (text_encoder bf16 ~33GB, transformer bf16 ~66GB, vae+audio_vae fp32 ~11GB),
  which does not fit in host RAM at once. `ComponentsManager.enable_auto_cpu_offload()`
  keeps every component CPU-resident as its steady state (accelerate hooks only move the
  *active* one to GPU), so it would try to hold all ~110GB in RAM simultaneously -- not
  safe here.
- Instead: transformer + vae + audio_vae stay resident on GPU permanently after the
  first load (66 + 11 = 77GB, comfortably under 96GB). Only the text_encoder
  (~33GB bf16) is loaded for the prompt-encoding phase and then moved back to CPU / freed
  before the denoise loop starts. This is a single one-way trip per generation, not a
  standing swap -- the "short window" pattern CLAUDE.md sanctions, not the banned
  "swap the whole big module every step" pattern.
- video VAE decode runs under a float16 autocast internally (diffusers' own
  MiniMaxH3VideoDecodeStep) even though its weights are float32. audio_vae must stay
  float32 end-to-end: casting it to bf16 is a known upstream bug that makes generated
  audio ~20dB too quiet, so we never touch its dtype after loading fp32.
"""
from __future__ import annotations

import gc
import io
import json
import logging
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
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Component lifecycle
    # ------------------------------------------------------------------
    def _ensure_pipe_shell(self):
        if self._pipe is not None:
            return
        from diffusers import ModularPipeline

        logger.info("building ModularPipeline shell from %s", MODEL_ID)
        self._pipe = ModularPipeline.from_pretrained(MODEL_ID)
        logger.info("pipe shell built: blocks=%s components=%s",
                     self._pipe._blocks.__class__.__name__, self._pipe.component_names)

    def _ensure_transformer_and_vae(self, progress: ProgressState | None = None):
        """Load transformer + vae + audio_vae to GPU once; they then stay resident."""
        self._ensure_pipe_shell()
        if self._transformer_loaded and self._vae_loaded:
            return
        if progress:
            progress.update(phase="loading_transformer", message="transformer/vae をロード中...")
        t0 = time.time()
        if not self._transformer_loaded:
            self._pipe.load_components(names=["transformer"], dtype=torch.bfloat16)
            self._pipe.transformer.to(DEVICE)
            self._transformer_loaded = True
            logger.info("transformer loaded to GPU in %.1fs. gpu=%s ram=%s", time.time() - t0, gpu_mem_gb(), ram_gb())
        if not self._vae_loaded:
            t1 = time.time()
            # video VAE must stay fp32 (decode step applies its own fp16 autocast);
            # audio VAE must stay fp32 end-to-end (bf16 causes ~20dB volume loss, see
            # module docstring / handoff doc).
            self._pipe.load_components(names=["vae", "audio_vae"], dtype=torch.float32)
            self._pipe.vae.to(DEVICE)
            self._pipe.audio_vae.to(DEVICE)
            self._pipe.load_components(names=["scheduler", "audio_scheduler"])
            self._vae_loaded = True
            logger.info("vae/audio_vae loaded to GPU in %.1fs. gpu=%s ram=%s", time.time() - t1, gpu_mem_gb(), ram_gb())

        from diffusers.video_processor import VideoProcessor

        if getattr(self._pipe, "video_processor", None) is None:
            self._pipe.video_processor = VideoProcessor(vae_scale_factor=16, do_normalize=False)

    def _load_text_encoder(self, progress: ProgressState | None = None):
        self._ensure_pipe_shell()
        if self._text_encoder_loaded:
            return
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
        self._pipe.text_encoder.to("cpu")
        del self._pipe.text_encoder
        self._pipe.text_encoder = None
        self._text_encoder_loaded = False
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("text_encoder freed. gpu=%s ram=%s", gpu_mem_gb(), ram_gb())

    def preload_all(self):
        """Load everything once at startup so the first UI request doesn't pay the load cost."""
        with self._load_lock:
            self._ensure_transformer_and_vae()
            self._load_text_encoder()
            self._free_text_encoder()

    def status(self) -> dict:
        return {
            "pipe_built": self._pipe is not None,
            "transformer_loaded": self._transformer_loaded,
            "vae_loaded": self._vae_loaded,
            "text_encoder_loaded": self._text_encoder_loaded,
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
        from diffusers.modular_pipelines.minimax_h3.before_encoder import (
            MiniMaxH3AutoKeyframeVaeEncoderStep,
            MiniMaxH3SetupStep,
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
            self._ensure_transformer_and_vae(progress)
            self._load_text_encoder(progress)

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

        # --- setup (canvas / frame count / keyframe prep) ---
        setup_step = MiniMaxH3SetupStep()
        _, state = setup_step(pipe, state)
        block_state = setup_step.get_block_state(state)

        # --- text encode (still has text_encoder on GPU at this point) ---
        if progress:
            progress.update(phase="encoding", message="プロンプトをエンコード中...")
        prompt_embeds, text_token_tags = MiniMaxH3TextEncoderStep.encode_prompt(
            pipe, prompt, block_state.keyframes or None, device=DEVICE, dtype=torch.bfloat16
        )
        state.set("prompt_embeds", prompt_embeds)
        state.set("text_token_tags", text_token_tags)

        # text_encoder's job is done for this request -- free it before the transformer
        # denoise loop so we never need bf16 TE (33GB) + bf16 transformer (66GB) +
        # fp32 VAEs (11GB) resident together (110GB > 96GB VRAM).
        self._free_text_encoder()

        # --- keyframe VAE conditioning (fl2va only; needs `vae`, already resident) ---
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

        if progress:
            progress.update(phase="muxing", message="mp4へmux中...")
        job_stub = f"t2va_{int(t_start)}"
        mp4_path = self.output_dir / f"{job_stub}.mp4"
        _mux_mp4(frames_uint8, audio_np, sampling_rate, FPS, mp4_path)

        result = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_frames_requested_seconds": seconds,
            "num_frames": block_state.num_frames,
            "duration_s": block_state.num_frames / FPS,
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
            "mode": "fl2va" if (image is not None or last_image is not None) else "t2va",
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

"""
MiniMax-H3 T2VA regression probe.

Loads components sequentially and directly onto CUDA (never staging more than one
large model in host RAM or on GPU at a time), runs the MiniMaxH3Blocks (t2va) pipeline
by hand block-by-block so we can free the text_encoder before the transformer denoise
loop, and finally frees the transformer before VAE decode.

Why not `ComponentsManager.enable_auto_cpu_offload()`: that mechanism keeps every
component resident in host RAM as its steady state (models start on CPU, move to GPU
only for their forward). text_encoder(bf16 ~33GB) + transformer(bf16 ~66GB) +
vae+audio_vae(fp32 ~11GB) = ~110GB, which does not fit in this box's 94GB RAM. Instead
we load each big component directly to CUDA, run its phase, then move it back to CPU
(a single one-way trip) or drop it before loading the next -- following the
diffusers-server CLAUDE.md #33/#47 "short window" swap pattern, never a standing
CPU-resident swap of everything at once.

Run: venv/bin/python scripts/probe_t2va.py
"""
import gc
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOG = "/home/animede/minimax-h3/logs/probe_t2va.log"
OUT_DIR = "/home/animede/minimax-h3/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

t0 = time.time()


def log(msg):
    line = f"[{time.time() - t0:8.1f}s] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def gpu_mem():
    if not torch.cuda.is_available():
        return "no cuda"
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    return f"alloc={alloc:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB"


def ram_mem():
    with open("/proc/meminfo") as f:
        meminfo = {}
        for line in f:
            parts = line.split()
            meminfo[parts[0].rstrip(":")] = int(parts[1])
    total = meminfo["MemTotal"] / 1e6
    avail = meminfo["MemAvailable"] / 1e6
    swap_total = meminfo.get("SwapTotal", 0) / 1e6
    swap_free = meminfo.get("SwapFree", 0) / 1e6
    swap_used = swap_total - swap_free
    return f"avail={avail:.1f}GB/{total:.1f}GB swap_used={swap_used:.2f}GB/{swap_total:.1f}GB"


log(f"start. gpu={gpu_mem()} ram={ram_mem()}")

MODEL_ID = "MiniMaxAI/MiniMax-H3"
DEVICE = torch.device("cuda:0")

import PIL.Image  # noqa: E402
from diffusers import ModularPipeline  # noqa: E402
from diffusers.modular_pipelines.modular_pipeline import PipelineState  # noqa: E402

log("importing diffusers OK")

# ---------------------------------------------------------------------------
# Build the pipeline shell (no weights yet -- from_pretrained() on the blocks
# just resolves component specs from modular_model_index.json).
# ---------------------------------------------------------------------------
pipe = ModularPipeline.from_pretrained(MODEL_ID)
log(f"pipe shell built, blocks={pipe._blocks.__class__.__name__}, "
    f"component_names={pipe.component_names}")

# ---------------------------------------------------------------------------
# Phase 1: text encoder -- load straight to CUDA in bf16, encode, free.
# ---------------------------------------------------------------------------
log("loading text_encoder/tokenizer/processor to cuda (bf16) ...")
t_load = time.time()
pipe.load_components(
    names=["text_encoder", "tokenizer", "processor"],
    dtype=torch.bfloat16,
)
pipe.text_encoder.to(DEVICE)
log(f"text_encoder loaded in {time.time() - t_load:.1f}s. gpu={gpu_mem()} ram={ram_mem()}")

PROMPT = (
    "A golden retriever puppy runs across a sunlit backyard lawn, chasing a red ball. "
    "The camera follows in a smooth tracking shot at dog-eye level. Leaves rustle in the "
    "wind, birds chirp, and the puppy barks happily as it catches the ball."
)
HEIGHT, WIDTH = 768, 768
# MINIMAX_H3_MIN_DURATION = 5.0s is a hard floor (checked in MiniMaxH3SetupStep._check_inputs),
# so the shortest legal clip is 124 frames (17*7+5) at 24fps = 5.17s.
NUM_FRAMES = 124
NUM_INFERENCE_STEPS = 30
SEED = 12345

from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3TextEncoderStep  # noqa: E402

prompt_embeds, text_token_tags = MiniMaxH3TextEncoderStep.encode_prompt(
    pipe, PROMPT, images=None, device=DEVICE, dtype=torch.bfloat16
)
log(f"prompt encoded: prompt_embeds={tuple(prompt_embeds.shape)} "
    f"text_token_tags={tuple(text_token_tags.shape)}. gpu={gpu_mem()}")

log("moving text_encoder to cpu ...")
pipe.text_encoder.to("cpu")
del pipe.text_encoder
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
log(f"text_encoder freed. gpu={gpu_mem()} ram={ram_mem()}")

# ---------------------------------------------------------------------------
# Phase 2: transformer -- load straight to CUDA in bf16, run the layout /
# latents / timesteps / denoise blocks by hand (skipping the text_encoder
# block, whose outputs we already computed above), then free.
# ---------------------------------------------------------------------------
log("loading transformer to cuda (bf16) ...")
t_load = time.time()
pipe.load_components(names=["transformer"], dtype=torch.bfloat16)
pipe.transformer.to(DEVICE)
log(f"transformer loaded in {time.time() - t_load:.1f}s. gpu={gpu_mem()} ram={ram_mem()}")

from diffusers.modular_pipelines.minimax_h3.before_denoise import (  # noqa: E402
    MiniMaxH3PrepareLatentsStep,
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
)
from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3SetupStep  # noqa: E402
from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3DenoiseStep  # noqa: E402

# scheduler/audio_scheduler are tiny configs, cheap to load with the rest
pipe.load_components(names=["scheduler", "audio_scheduler"])

state = PipelineState()
state.set("prompt", PROMPT)
state.set("image", None)
state.set("last_image", None)
state.set("height", HEIGHT)
state.set("width", WIDTH)
state.set("num_frames", NUM_FRAMES)
state.set("generator", torch.Generator(device="cpu").manual_seed(SEED))
state.set("num_inference_steps", NUM_INFERENCE_STEPS)
state.set("output_type", "pt")
state.set("attention_kwargs", None)
state.set("latents", None)
state.set("audio_latents", None)
state.set("condition_latents", None)
state.set("audio_condition_latents", None)

setup_step = MiniMaxH3SetupStep()
_, state = setup_step(pipe, state)
block_state = setup_step.get_block_state(state)
log(f"setup done: height={block_state.height} width={block_state.width} "
    f"num_frames={block_state.num_frames} num_latent_frames={block_state.num_latent_frames} "
    f"latent_hw=({block_state.latent_height},{block_state.latent_width}) "
    f"num_audio_latents={block_state.num_audio_latents}")

# inject the prompt_embeds/text_token_tags we computed in phase 1
state.set("prompt_embeds", prompt_embeds)
state.set("text_token_tags", text_token_tags)
state.set("keyframes", [])
state.set("keyframe_anchors", ())

layout_step = MiniMaxH3PrepareLayoutStep()
_, state = layout_step(pipe, state)

latents_step = MiniMaxH3PrepareLatentsStep()
_, state = latents_step(pipe, state)

timesteps_step = MiniMaxH3SetTimestepsStep()
_, state = timesteps_step(pipe, state)
log(f"layout/latents/timesteps prepared. gpu={gpu_mem()}")

log(f"starting denoise loop: {NUM_INFERENCE_STEPS} steps ...")
t_denoise = time.time()
denoise_step = MiniMaxH3DenoiseStep()

# instrument per-step timing without editing library code: wrap loop_step
orig_loop_step = denoise_step.loop_step
step_times = []


def timed_loop_step(components, block_state, i, t):
    ts = time.time()
    result = orig_loop_step(components, block_state, i=i, t=t)
    dt = time.time() - ts
    step_times.append(dt)
    log(f"  step {i + 1}/{NUM_INFERENCE_STEPS} took {dt:.2f}s. gpu={gpu_mem()}")
    return result


denoise_step.loop_step = timed_loop_step
_, state = denoise_step(pipe, state)
denoise_time = time.time() - t_denoise
log(f"denoise loop done in {denoise_time:.1f}s "
    f"(avg {sum(step_times) / len(step_times):.2f}s/step). gpu={gpu_mem()} ram={ram_mem()}")

log("moving transformer to cpu ...")
pipe.transformer.to("cpu")
del pipe.transformer
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
log(f"transformer freed. gpu={gpu_mem()} ram={ram_mem()}")

# ---------------------------------------------------------------------------
# Phase 3: VAEs -- load fp32 (mandatory, see handoff doc: bf16 audio VAE loses
# ~20dB), decode video + audio, free.
# ---------------------------------------------------------------------------
log("loading vae + audio_vae to cuda (fp32) ...")
t_load = time.time()
pipe.load_components(names=["vae", "audio_vae"], dtype=torch.float32)
pipe.vae.to(DEVICE)
pipe.audio_vae.to(DEVICE)
log(f"VAEs loaded in {time.time() - t_load:.1f}s. gpu={gpu_mem()} ram={ram_mem()}")

from diffusers.modular_pipelines.minimax_h3.decoders import (  # noqa: E402
    MiniMaxH3AudioDecodeStep,
    MiniMaxH3VideoDecodeStep,
)
from diffusers.video_processor import VideoProcessor  # noqa: E402

pipe.video_processor = VideoProcessor(vae_scale_factor=16, do_normalize=False)

state.set("output_type", "pt")

t_decode = time.time()
video_decode_step = MiniMaxH3VideoDecodeStep()
_, state = video_decode_step(pipe, state)
log(f"video decoded in {time.time() - t_decode:.1f}s. gpu={gpu_mem()}")

t_decode = time.time()
audio_decode_step = MiniMaxH3AudioDecodeStep()
_, state = audio_decode_step(pipe, state)
log(f"audio decoded in {time.time() - t_decode:.1f}s. gpu={gpu_mem()}")

videos = state.get("videos")
audio = state.get("audio")
sampling_rate = state.get("sampling_rate")

log(f"videos type={type(videos)} audio shape={tuple(audio.shape)} sampling_rate={sampling_rate}")

peak_vram = torch.cuda.max_memory_allocated() / 1e9
log(f"phase3 peak vram={peak_vram:.2f}GB")

# ---------------------------------------------------------------------------
# Mux to mp4 with PyAV, save wav-equivalent stats, dump a report.
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402

video_tensor = videos[0] if isinstance(videos, list) else videos  # (T, C, H, W) or (1,T,C,H,W)
if video_tensor.dim() == 5:
    video_tensor = video_tensor[0]
log(f"video_tensor shape={tuple(video_tensor.shape)} dtype={video_tensor.dtype} "
    f"min={video_tensor.min().item():.3f} max={video_tensor.max().item():.3f}")

audio_np = audio[0].float().cpu().numpy()  # (2, num_samples)
rms = float(np.sqrt(np.mean(audio_np ** 2)))
peak = float(np.max(np.abs(audio_np)))
log(f"audio rms={rms:.6f} peak={peak:.6f} (non-silent if rms > ~1e-4)")

# save a frame as PNG for visual check
frames_uint8 = (video_tensor.permute(0, 2, 3, 1).float().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
mid_frame = frames_uint8[len(frames_uint8) // 2]
PIL.Image.fromarray(mid_frame).save(os.path.join(OUT_DIR, "probe_mid_frame.png"))
PIL.Image.fromarray(frames_uint8[0]).save(os.path.join(OUT_DIR, "probe_first_frame.png"))
PIL.Image.fromarray(frames_uint8[-1]).save(os.path.join(OUT_DIR, "probe_last_frame.png"))
log(f"saved frame PNGs to {OUT_DIR}")

# mux to mp4 via PyAV
import av  # noqa: E402

mp4_path = os.path.join(OUT_DIR, "probe_t2va.mp4")
container = av.open(mp4_path, mode="w")
fps = 24
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
# av's packed s16 stereo format wants interleaved L,R,L,R,... in a (1, 2N) array,
# not a (2, N) per-channel block layout -- verified by roundtrip probe.
audio_interleaved = audio_i16.T.reshape(1, -1)
audio_frame = av.AudioFrame.from_ndarray(audio_interleaved, format="s16", layout="stereo")
audio_frame.sample_rate = sampling_rate
for packet in astream.encode(audio_frame):
    container.mux(packet)
for packet in astream.encode():
    container.mux(packet)

container.close()
log(f"muxed mp4 -> {mp4_path}")

report = {
    "prompt": PROMPT,
    "height": HEIGHT,
    "width": WIDTH,
    "num_frames_requested": NUM_FRAMES,
    "num_frames_actual": block_state.num_frames,
    "num_inference_steps": NUM_INFERENCE_STEPS,
    "seed": SEED,
    "denoise_time_s": denoise_time,
    "avg_step_time_s": sum(step_times) / len(step_times),
    "peak_vram_gb": peak_vram,
    "audio_rms": rms,
    "audio_peak": peak,
    "audio_sampling_rate": sampling_rate,
    "mp4_path": mp4_path,
    "total_elapsed_s": time.time() - t0,
}
with open(os.path.join(OUT_DIR, "probe_report.json"), "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
log(f"report: {json.dumps(report, indent=2, ensure_ascii=False)}")
log("PROBE OK")

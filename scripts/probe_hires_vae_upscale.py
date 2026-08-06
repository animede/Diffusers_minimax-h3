"""Isolate whether AutoencoderKLMiniMaxH3 can decode a naively bilinear-upsampled latent
at all, independent of the two-pass runner plumbing. Loads vae + a completed pass-1
latent captured from a real generate() call is not convenient standalone, so instead this
re-derives a similar situation: run a short single-pass generation at 384x384 with
output_type="latent" to get a real converged latent tensor, then decode it directly at
native res (sanity) and after a manual 2x bilinear upsample (the actual test)."""
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("H3_CACHE", "none")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from diffusers import ModularPipeline  # noqa: E402
from diffusers.modular_pipelines.modular_pipeline import PipelineState  # noqa: E402
from diffusers.modular_pipelines.minimax_h3.before_denoise import (  # noqa: E402
    MiniMaxH3PrepareLatentsStep,
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
)
from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3SetupStep  # noqa: E402
from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3DenoiseStep  # noqa: E402
from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3TextEncoderStep  # noqa: E402
from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3VideoDecodeStep  # noqa: E402
from diffusers.modular_pipelines.minimax_h3.packing import patchify_video_latents, unpatchify_video_tokens  # noqa: E402
from diffusers.video_processor import VideoProcessor  # noqa: E402
from transformers import BitsAndBytesConfig  # noqa: E402

MODEL_ID = "MiniMaxAI/MiniMax-H3"
DEVICE = torch.device("cuda:0")
OUT_DIR = Path("/home/animede/minimax-h3/outputs/debug_hires")
OUT_DIR.mkdir(parents=True, exist_ok=True)

pipe = ModularPipeline.from_pretrained(MODEL_ID)

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
pipe.load_components(
    names=["text_encoder", "tokenizer", "processor"],
    dtype=torch.bfloat16,
    quantization_config={"text_encoder": quant_config},
    device_map={"text_encoder": "cuda"},
)

PROMPT = "A golden retriever puppy runs across a sunlit backyard lawn, chasing a red ball."
HEIGHT, WIDTH = 384, 384
NUM_FRAMES = 124
NUM_INFERENCE_STEPS = 20  # -> 19 model evals, fully converged
SEED = 12345

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
keyframes = state.get("keyframes")

with torch.no_grad():
    prompt_embeds, text_token_tags = MiniMaxH3TextEncoderStep.encode_prompt(
        pipe, PROMPT, keyframes or None, device=DEVICE, dtype=torch.bfloat16
    )
state.set("prompt_embeds", prompt_embeds)
state.set("text_token_tags", text_token_tags)

del pipe.text_encoder
pipe.text_encoder = None
import gc
gc.collect()
torch.cuda.empty_cache()

pipe.load_components(names=["transformer"], dtype=torch.bfloat16)
pipe.transformer.to(DEVICE)
pipe.load_components(names=["scheduler", "audio_scheduler"])

layout_step = MiniMaxH3PrepareLayoutStep()
_, state = layout_step(pipe, state)
latents_step = MiniMaxH3PrepareLatentsStep()
_, state = latents_step(pipe, state)
timesteps_step = MiniMaxH3SetTimestepsStep()
_, state = timesteps_step(pipe, state)

print("running denoise...", flush=True)
t0 = time.time()
denoise_step = MiniMaxH3DenoiseStep()
_, state = denoise_step(pipe, state)
print("denoise done in", time.time() - t0, flush=True)

# free transformer, load vae
del pipe.transformer
pipe.transformer = None
gc.collect()
torch.cuda.empty_cache()

pipe.load_components(names=["vae", "audio_vae"], dtype=torch.float32)
pipe.vae.to(DEVICE)
pipe.audio_vae.to(DEVICE)
pipe.video_processor = VideoProcessor(vae_scale_factor=16, do_normalize=False)

num_latent_frames = state.get("num_latent_frames")
latent_height = state.get("latent_height")
latent_width = state.get("latent_width")
patch_size = pipe.patch_size
vae_latent_channels = pipe.vae_latent_channels
print("geometry:", num_latent_frames, latent_height, latent_width, patch_size, vae_latent_channels, flush=True)

MINIMAX_H3_PIXEL_MEAN = (0.485, 0.456, 0.406)
MINIMAX_H3_PIXEL_STD = (0.229, 0.224, 0.225)


def decode_and_save(latents_2d_rows, num_latent_frames, latent_height, latent_width, out_name):
    """latents_2d_rows: patchified rows tensor (num_patches, channels*4). Mirrors
    MiniMaxH3VideoDecodeStep.__call__ exactly (core/runner.py's own decode path)."""
    latents = unpatchify_video_tokens(
        latents_2d_rows, num_latent_frames, latent_height, latent_width, vae_latent_channels, patch_size
    )
    latents_mean = torch.tensor(pipe.vae.config.latents_mean, device=DEVICE).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(pipe.vae.config.latents_std, device=DEVICE).view(1, -1, 1, 1, 1)
    latents = latents * latents_std + latents_mean
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        video = pipe.vae.decode(latents, return_dict=False)[0]
    pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=DEVICE).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=DEVICE).view(1, -1, 1, 1, 1)
    video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
    videos = pipe.video_processor.postprocess_video(video, output_type="pt")
    video_tensor = videos[0] if isinstance(videos, list) else videos
    if video_tensor.dim() == 5:
        video_tensor = video_tensor[0]
    frames_uint8 = (
        (video_tensor.permute(0, 2, 3, 1).float().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    )
    mid = frames_uint8[len(frames_uint8) // 2]
    Image.fromarray(mid).save(OUT_DIR / out_name)
    print("saved", out_name, "frame shape", mid.shape, flush=True)


# 1. Decode native resolution (sanity: should be clean, matches pass1only script result)
decode_and_save(state.get("latents"), num_latent_frames, latent_height, latent_width, "vaeprobe_native.png")

# 2. Manually 2x bilinear upsample the latent (same logic as _upscale_block_state_2x) and decode
video_latent = unpatchify_video_tokens(
    state.get("latents"), num_latent_frames, latent_height, latent_width, vae_latent_channels, patch_size
)
b, c, t_dim, h_dim, w_dim = video_latent.shape
video_latent_2d = video_latent.permute(0, 2, 1, 3, 4).reshape(b * t_dim, c, h_dim, w_dim)
video_latent_2d = F.interpolate(video_latent_2d.float(), scale_factor=2, mode="bilinear", align_corners=False)
new_h, new_w = video_latent_2d.shape[-2:]
video_latent_up = video_latent_2d.reshape(b, t_dim, c, new_h, new_w).permute(0, 2, 1, 3, 4).to(video_latent.dtype)
rows_up = patchify_video_latents(video_latent_up, patch_size).to(DEVICE)
decode_and_save(rows_up, num_latent_frames, new_h, new_w, "vaeprobe_upsampled.png")

# 3. Same upsampled latent, but with VAE tiling disabled -- isolates whether the
# checkerboard pattern is a tiled-decode seam artifact or already present in the latent
# content itself.
pipe.vae.disable_tiling()
decode_and_save(rows_up, num_latent_frames, new_h, new_w, "vaeprobe_upsampled_notiling.png")
pipe.vae.enable_tiling()

print("ALL DONE", flush=True)

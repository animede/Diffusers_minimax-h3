"""
Standalone VAE decode peak-VRAM probe, isolating decode from denoise entirely.

Goal: characterize whether reducing `tile_sample_min_height`/`tile_sample_min_width`
below the VAE's own default (256px) meaningfully lowers decode's peak VRAM for a
768x768 / 124-frame t2va video, since H3_LOWVRAM_GROUP's 32GB-ballast verification
found decode (not denoise, not the transformer) is the actual peak-VRAM phase (~38-39GB
peak measured even in H3_LOWVRAM=1 mode, which has ZERO transformer GPU footprint during
decode -- so this is a VAE-decode-buffer-size question, independent of the transformer
choreography question this task is otherwise about).

Builds a synthetic latent tensor of the right shape (skips text encode + denoise
entirely -- ~20-30 minutes saved per trial) and runs `vae.decode()` directly, at
several tile sizes, recording `torch.cuda.max_memory_allocated()` for each.

Run: venv/bin/python scripts/probe_vae_tile_size.py
"""
import gc
import time

import torch


def gpu_gb():
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
        "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }


def main():
    from diffusers import ModularPipeline

    print("[probe] building pipe shell + loading vae/audio_vae...", flush=True)
    pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3")
    pipe.load_components(names=["vae", "audio_vae"], dtype=torch.float32)
    vae = pipe.vae
    print(f"[probe] vae loaded. use_tiling={vae.use_tiling} tile_h={vae.tile_sample_min_height} "
          f"tile_w={vae.tile_sample_min_width}", flush=True)

    # Latent shape for 768x768, 124 frames: need latent_channels, spatial_compression_ratio,
    # temporal geometry from the vae's own config -- derive it the same way generate() does
    # (see MiniMaxH3PrepareLatentsStep / packing.py), but simplified: just build a
    # plausible-shaped random latent directly matching what _decode() expects
    # (B, C, T_latent, H_latent, W_latent).
    height, width, num_frames = 768, 768, 124
    scr = vae.spatial_compression_ratio
    latent_h = height // scr
    latent_w = width // scr
    # token count: from _decode's own chunking logic, num_latent_frames derived from
    # tokens_chunk_size * num_chunks roughly matches num_frames // temporal_compression_ratio
    # (token_drop-adjusted) -- use the encoder's own clip_length/temporal_compression_ratio
    # to compute a matching latent frame count via a real (cheap) encode of a zero video
    # instead of hand-deriving the token_drop arithmetic (safer, avoids an off-by-N bug in
    # this throwaway probe skewing the result).
    print(f"[probe] deriving latent frame count via a real (zero-valued, no grad) encode "
          f"(on GPU -- CPU fp32 encode of a full 768x768x124 tensor was too slow in practice)...", flush=True)
    vae.to("cuda")
    dummy_video = torch.zeros(1, vae.config.in_channels if hasattr(vae.config, "in_channels") else 3,
                               num_frames, height, width, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        posterior = vae.encode(dummy_video).latent_dist
        latent = posterior.mode().cpu()
    print(f"[probe] derived latent shape: {tuple(latent.shape)}", flush=True)
    del dummy_video, posterior
    vae.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    latent_gpu_template = latent.clone()

    tile_configs = [
        ("default_256", 256, 256, 64, 64),
        ("tile_192", 192, 192, 48, 48),
        ("tile_128", 128, 128, 32, 32),
        ("tile_96", 96, 96, 24, 24),
    ]

    results = {}
    for label, th, tw, oh, ow in tile_configs:
        print(f"\n=== {label}: tile={th}x{tw} overlap={oh}x{ow} ===", flush=True)
        vae.enable_tiling(
            tile_sample_min_height=th,
            tile_sample_min_width=tw,
            tile_sample_min_overlap_height=oh,
            tile_sample_min_overlap_width=ow,
        )
        vae.to("cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        z = latent_gpu_template.to("cuda")
        t0 = time.time()
        try:
            with torch.no_grad():
                # video decode step in the real pipeline uses fp16 autocast internally
                # (see runner.py's module docstring) -- match that here.
                with torch.autocast("cuda", dtype=torch.float16):
                    decoded = vae.decode(z).sample
            torch.cuda.synchronize()
            t1 = time.time()
            peak = gpu_gb()
            print(f"[{label}] decode OK in {t1-t0:.1f}s. peak={peak} out_shape={tuple(decoded.shape)}", flush=True)
            results[label] = {"ok": True, "time_s": round(t1 - t0, 1), "peak_gb": peak["peak_gb"]}
            del decoded
        except Exception as e:
            print(f"[{label}] FAILED: {e!r}", flush=True)
            results[label] = {"ok": False, "error": str(e)}
        del z
        gc.collect()
        torch.cuda.empty_cache()
        vae.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== SUMMARY ===", flush=True)
    for label, r in results.items():
        print(f"  {label}: {r}", flush=True)


if __name__ == "__main__":
    main()

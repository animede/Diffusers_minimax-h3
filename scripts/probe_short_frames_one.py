"""
Single-generation worker for probe_short_frames.py's driver. Runs exactly ONE
MiniMaxH3Runner.generate() call in its own fresh process and writes a JSON result
file. Split out from the original all-in-one probe_short_frames.py after the first
run showed the 5-frame case's VAE-decode ValueError (see below) leaves the runner's
transformer/TE residency bookkeeping in a state inconsistent with actual GPU
occupancy (the exception fires before generate()'s own post-decode "reload
transformer, restore steady state" code runs), which cascaded into CUDA OOM for the
next two (unrelated, otherwise-fine) requests in the same process. Isolating each
generation in its own process sidesteps that entirely -- each one starts from a
clean CUDA context matching a real single-request server restart, which is also a
more faithful measurement of each frame count in isolation anyway.

Usage: venv/bin/python scripts/probe_short_frames_one.py <label> <seconds> <extract> <out_json>
  label:   e.g. "5frames"
  seconds: float, e.g. 0.2083333
  extract: "all" | "ends" | "first"
  out_json: path to write the result dict to
"""
import json
import logging
import sys
import time
import traceback
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

OUT_DIR = BASE_DIR / "outputs" / "ab_short_frames"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = "A red fox walks through a snowy forest at dawn, birds chirping, wind blowing through pine trees"
SEED = 12345
HEIGHT = 768
WIDTH = 768
STEPS = 30  # turbo=off, quality-first condition per task brief


def patch_duration_gates():
    patched = []

    import diffusers.modular_pipelines.minimax_h3.before_encoder as before_encoder

    patched.append((before_encoder, "MINIMAX_H3_MIN_DURATION", before_encoder.MINIMAX_H3_MIN_DURATION))
    before_encoder.MINIMAX_H3_MIN_DURATION = 0.01

    import diffusers.modular_pipelines.minimax_h3.packing as packing

    patched.append((packing, "MINIMAX_H3_MIN_DURATION", packing.MINIMAX_H3_MIN_DURATION))
    packing.MINIMAX_H3_MIN_DURATION = 0.01

    import core.runner as runner_mod

    patched.append((runner_mod, "MIN_SECONDS", runner_mod.MIN_SECONDS))
    runner_mod.MIN_SECONDS = 0.01

    for mod, name, old in patched:
        new = getattr(mod, name)
        logging.info("[patch] %s.%s: %r -> %r", mod.__name__, name, old, new)
    return patched


def gpu_gb():
    import torch

    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
    }


def extract_frames_png(mp4_path: Path, out_prefix: Path, which: str):
    import av

    container = av.open(str(mp4_path))
    stream = container.streams.video[0]
    frames = [f.to_image() for f in container.decode(stream)]
    container.close()
    n = len(frames)
    written = []
    if which == "all":
        indices = list(range(n))
    elif which == "ends":
        indices = sorted(set([0, n // 2, n - 1]))
    elif which == "first":
        indices = [0]
    else:
        raise ValueError(which)
    for idx in indices:
        p = out_prefix.parent / f"{out_prefix.name}_frame{idx:03d}.png"
        frames[idx].save(p)
        written.append(str(p))
    return written, n


def main():
    label = sys.argv[1]
    seconds = float(sys.argv[2])
    extract = sys.argv[3]
    out_json = Path(sys.argv[4])

    logging.info("=== patching duration gates ===")
    patch_duration_gates()

    logging.info("=== building runner (label=%s seconds=%.4f) ===", label, seconds)
    from core.runner import MiniMaxH3Runner, ProgressState

    runner = MiniMaxH3Runner(OUT_DIR)
    progress = ProgressState()
    progress.update(job_id=label, started_at=time.time())

    import torch

    result = None
    tb = None
    t0 = time.time()
    try:
        result = runner.generate(
            prompt=PROMPT,
            height=HEIGHT,
            width=WIDTH,
            seconds=seconds,
            num_inference_steps=STEPS,
            seed=SEED,
            progress=progress,
        )
    except Exception:
        tb = traceback.format_exc()
        logging.error("[%s] EXCEPTION during generate():\n%s", label, tb)
    wall = time.time() - t0

    record = {
        "label": label,
        "requested_seconds": seconds,
        "wall_time_s": round(wall, 2),
        "gpu_after": gpu_gb(),
        "exception": tb,
    }
    if result is not None:
        record.update({
            "num_frames": result["num_frames"],
            "duration_s": result["duration_s"],
            "denoise_time_s": result["denoise_time_s"],
            "decode_time_s": result["decode_time_s"],
            "peak_vram_gb": result["peak_vram_gb"],
            "audio_rms": result["audio_rms"],
            "audio_peak": result["audio_peak"],
            "audio_sampling_rate": result["audio_sampling_rate"],
            "mp4_path": result["mp4_path"],
        })
        mp4_path = Path(result["mp4_path"])
        dest = OUT_DIR / f"{label}.mp4"
        dest.write_bytes(mp4_path.read_bytes())
        record["mp4_saved"] = str(dest)
        try:
            frame_paths, n_decoded = extract_frames_png(dest, OUT_DIR / label, extract)
            record["extracted_frames"] = frame_paths
            record["n_decoded_frames"] = n_decoded
        except Exception:
            record["frame_extract_exception"] = traceback.format_exc()
            logging.error("[%s] frame extraction failed:\n%s", label, record["frame_extract_exception"])

    out_json.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str))
    logging.info("=== wrote %s ===", out_json)


if __name__ == "__main__":
    main()

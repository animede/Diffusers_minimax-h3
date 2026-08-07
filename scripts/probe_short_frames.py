"""
Driver: runs the 5-frame / 22-frame / 124-frame(baseline) MiniMax-H3 quality probe
described in scripts/probe_short_frames_one.py, one fresh subprocess per frame count.

Why subprocesses, not three `generate()` calls in one process (the original design):
the 5-frame case's VAE decode raises `ValueError: torch.cat(): expected a non-empty
list of Tensors` inside `AutoencoderKLMiniMaxH3._decode()` (see below) -- with only 2
latent frames (5 pixel frames -> `video_latent_num_frames(5) == 2`, well under
`tokens_chunk_size`), the chunk loop's `range(num_chunks)` is empty and
`decoded_chunks` stays `[]`. Because that exception fires from inside
`generate()`'s decode phase, generate()'s own post-decode bookkeeping (reload the
transformer that decode's own bnb-4bit VAE-vs-transformer swap dropped, restore the
transformer+TE-nf4 steady state) never runs -- the first probe run's 22-frame and
124-frame requests then both CUDA-OOM'd immediately after, with ~98.5GB allocated
(TE-nf4 + transformer both still resident from a state generate() had no chance to
clean up). One subprocess per generation sidesteps this: each starts from a clean
CUDA context, which is also what "does 5-frame/22-frame break in isolation" (the
actual question this probe asks) should be measuring anyway, not "does it break
*after* an unrelated exception already corrupted this process's GPU state".

Run with the real server DOWN (`pgrep -f "[u]vicorn app:app.*8611"`, brackets
required).
"""
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "outputs" / "ab_short_frames"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PYTHON = BASE_DIR / "venv" / "bin" / "python"
WORKER = BASE_DIR / "scripts" / "probe_short_frames_one.py"

PLAN = [
    ("5frames", 5 / 24, "all"),
    ("22frames", 22 / 24, "ends"),
    ("124frames_baseline", 124 / 24, "first"),
]


def gpu_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return int(out.splitlines()[0])


def main():
    results = []
    for label, seconds, extract in PLAN:
        logging.info("=== launching subprocess for %s (seconds=%.4f) ===", label, seconds)
        out_json = OUT_DIR / f"{label}_result.json"
        log_path = OUT_DIR / f"{label}.subprocess.log"
        t0 = time.time()
        with open(log_path, "w") as logf:
            proc = subprocess.run(
                [str(PYTHON), "-u", str(WORKER), label, str(seconds), extract, str(out_json)],
                cwd=str(BASE_DIR), stdout=logf, stderr=subprocess.STDOUT,
            )
        wall = time.time() - t0
        logging.info("[%s] subprocess exited rc=%d in %.1fs (full log: %s)", label, proc.returncode, wall, log_path)

        if out_json.exists():
            record = json.loads(out_json.read_text())
        else:
            # Worker crashed before writing its own result file (e.g. hard OOM kill,
            # uncaught SIGABRT from a CUDA assert) -- capture what we can from the
            # subprocess return code and tail of its log instead of failing the whole
            # driver.
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-60:])
            record = {
                "label": label, "requested_seconds": seconds, "wall_time_s": round(wall, 2),
                "subprocess_returncode": proc.returncode, "exception": f"worker did not write {out_json}; log tail:\n{tail}",
            }
        record["subprocess_returncode"] = proc.returncode
        results.append(record)

        gpu_now = gpu_used_mib()
        logging.info("[%s] gpu after subprocess exit: %d MiB", label, gpu_now)
        if gpu_now > 3000:
            logging.warning("[%s] GPU still shows %d MiB used after subprocess exit -- waiting up to 30s for it to clear",
                             label, gpu_now)
            for _ in range(30):
                time.sleep(1)
                gpu_now = gpu_used_mib()
                if gpu_now <= 3000:
                    break
            logging.info("[%s] gpu after wait: %d MiB", label, gpu_now)

    out_all = OUT_DIR / "probe_results.json"
    out_all.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    logging.info("=== wrote %s ===", out_all)

    logging.info("=== summary ===")
    for rec in results:
        if rec.get("exception"):
            logging.info("%s: EXCEPTION/FAILED (rc=%s) -- see json/log for detail", rec["label"], rec.get("subprocess_returncode"))
        else:
            logging.info(
                "%s: frames=%s duration=%.3fs denoise=%.2fs decode=%.2fs peak_vram=%.2fGB audio_rms=%.5f wall=%.2fs",
                rec["label"], rec.get("num_frames"), rec.get("duration_s", float("nan")),
                rec.get("denoise_time_s", float("nan")), rec.get("decode_time_s", float("nan")),
                rec.get("peak_vram_gb", float("nan")), rec.get("audio_rms", float("nan")), rec["wall_time_s"],
            )


if __name__ == "__main__":
    main()

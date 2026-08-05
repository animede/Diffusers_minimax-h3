"""
Standalone VRAM ballast holder for H3_LOWVRAM verification.

Allocates a single dummy CUDA tensor sized so the *free* VRAM on cuda:0 drops to
approximately `--target-free-gb` (default 43.5, chosen to be stricter than a real
48GB card's ~47GB free -- see the task brief), then sleeps forever holding the
allocation, so a separate uvicorn server process (started after this one) sees a
48GB-class amount of free VRAM for its own CUDA context.

This never touches the diffusers-server-style "swap a resident model" pattern --
it is a completely independent process holding one dummy tensor, modeled on the same
technique used for the LTX-2.3 48GB verification in ../diffusers-server (see that
repo's CLAUDE.md #37/#45 and tools/joyai_regress.py's `allocate_ballast`).

Run in background, kill by PID when done:
    venv/bin/python scripts/vram_ballast.py --target-free-gb 43.5 &
    echo $! > /tmp/ballast.pid
    ...
    kill $(cat /tmp/ballast.pid)
"""
import argparse
import time

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-free-gb", type=float, default=43.5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / 1024**3
    total_gb = total_bytes / 1024**3
    to_consume_gb = free_gb - args.target_free_gb
    print(
        f"[ballast] free={free_gb:.2f}GB total={total_gb:.2f}GB "
        f"target_free={args.target_free_gb:.2f}GB to_consume={to_consume_gb:.2f}GB",
        flush=True,
    )
    if to_consume_gb <= 0:
        print("[ballast] free already <= target, not allocating anything", flush=True)
    else:
        n_elements = int(to_consume_gb * 1024**3 / 4)  # float32 = 4 bytes
        tensor = torch.empty(n_elements, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        free_bytes2, _ = torch.cuda.mem_get_info()
        print(f"[ballast] allocated. free now={free_bytes2 / 1024**3:.2f}GB", flush=True)

    print("[ballast] holding forever. kill this process to release.", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()

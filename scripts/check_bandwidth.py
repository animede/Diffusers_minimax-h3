#!/usr/bin/env python
"""ホスト ↔ GPU の転送速度と、量子化キャッシュのディスク読み出し速度を測る(低 VRAM 構成の速さはこの 2 つで決まる)。

  venv/bin/python scripts/check_bandwidth.py            # Linux
  .\\venv\\Scripts\\python.exe scripts\\check_bandwidth.py   # Windows

目安: PCIe 4.0 x8 ならピン留め H2D が 10GB/s 前後、x4 で 5GB/s 前後、PCIe 3.0 x4 で 3GB/s 前後。1GB/s を切っていたら
リンクが x1 / Gen1 に落ちているか、ライザーや BIOS 設定の問題(GPU-Z の Bus Interface で負荷中のリンク状態を確認)。
ディスクは NVMe で 2〜3GB/s、SATA SSD で 0.4〜0.5GB/s。
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

MB = 2**20
GB = 2**30


def bench_copy(src: torch.Tensor, dst: torch.Tensor, label: str, reps: int = 10) -> None:
    dst.copy_(src)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"{label:34s} {src.numel() * src.element_size() * reps / dt / GB:6.2f} GB/s")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA が見えません")
    dev = torch.device("cuda:0")
    p = torch.cuda.get_device_properties(dev)
    print(f"GPU: {p.name}  VRAM {p.total_memory / GB:.1f} GiB  torch {torch.__version__}")
    free, total = torch.cuda.mem_get_info(dev)
    print(f"mem_get_info: free {free / GB:.2f} / total {total / GB:.2f} GiB")
    n = 256 * MB
    gpu = torch.empty(n, dtype=torch.uint8, device=dev)
    pageable = torch.empty(n, dtype=torch.uint8)
    pinned = torch.empty(n, dtype=torch.uint8).pin_memory()
    reg = torch.empty(n, dtype=torch.uint8)
    err = torch.cuda.cudart().cudaHostRegister(reg.data_ptr(), reg.numel(), 0)
    print(f"cudaHostRegister: {'ok' if int(err) == 0 else err}")
    bench_copy(pageable, gpu, "H2D pageable (256MB x10)")
    bench_copy(pinned, gpu, "H2D pinned, pin_memory() (x10)")
    if int(err) == 0:
        bench_copy(reg, gpu, "H2D pinned, cudaHostRegister (x10)")
    bench_copy(gpu, pinned, "D2H pinned (x10)")
    gpu2 = torch.empty(n, dtype=torch.uint8, device=dev)
    bench_copy(gpu, gpu2, "D2D (x10)")
    if int(err) == 0:
        torch.cuda.cudart().cudaHostUnregister(reg.data_ptr())
    # ディスク: 量子化キャッシュの最初のシャードを 1GB だけ読む(ページキャッシュを避けるため大きめのオフセットから)
    root = Path(os.environ.get("H3_TE_PREQUANT_DIR", str(Path(__file__).resolve().parent.parent / "models" / "prequant")))
    shards = sorted((root / "transformer_int8").glob("*.safetensors")) if (root / "transformer_int8").exists() else []
    if shards:
        path = shards[-1]
        size = path.stat().st_size
        chunk, want = 32 * MB, min(1 * GB, size // 2)
        off = size - want
        flags = os.O_RDONLY | (getattr(os, "O_BINARY", 0) | getattr(os, "O_SEQUENTIAL", 0) if sys.platform == "win32" else 0)
        fd = os.open(str(path), flags)
        buf = bytearray(chunk)
        got = 0
        t0 = time.perf_counter()
        with os.fdopen(fd, "rb", buffering=0, closefd=True) as f:
            f.seek(off)
            while got < want:
                k = f.readinto(buf)
                if not k:
                    break
                got += k
        dt = time.perf_counter() - t0
        print(f"{'disk read ' + path.name[-26:]:34s} {got / dt / GB:6.2f} GB/s  (1GB 逐次読み。2 回目以降はページキャッシュの値になる)")
    else:
        print("disk read: models/prequant/transformer_int8 が無いので省略")


if __name__ == "__main__":
    main()

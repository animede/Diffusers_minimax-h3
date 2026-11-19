#!/usr/bin/env python
"""video VAE のデコードだけを新しいプロセスで回し、1 パス(1 タイル × 1 クリップ)の秒数を測る。

サーバの中でデコードが遅いとき、「この GPU でのデコード計算そのものが遅い」のか「サーバの VRAM 状態(torch のプール、
他の常駐)のせい」なのかを切り分ける(2026-11-19、RTX 3050 6GB で 1 ブロックあたり 0.25 秒という遅さが出た件)。
サーバと同じ経路(fp16 キャッシュからデコーダブロックを読み、入るぶんを GPU に置いて残りをファイルから流す)を、
プールが空のプロセスで再現する。モデルは既にあるキャッシュを使い、ダウンロードはしない。

  venv/bin/python scripts/probe_vae_decode.py                       # 512²・5 秒相当、常駐ブロック数は自動
  venv/bin/python scripts/probe_vae_decode.py --size 768 --resident 0   # 768²、常駐なし(全部ファイルから流す)
  .\\venv\\Scripts\\python.exe scripts\\probe_vae_decode.py --resident 15   # Windows、常駐 15 ブロック

出力の見方: 「pass N: X.XXs」が 1 パスの時間(サーバのログと同じ)。4060 Laptop で 0.5 秒前後、A4000 で 0.2 秒前後。
GPU 計算だけならカードの性能に比例する。これが 3〜5 秒以上なら、流すブロックの転送か VRAM の退避が疑わしい。
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core.hubenv  # noqa: E402,F401

import torch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("minimax_h3.probe_vae_decode")
GB = 2**30


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512, help="正方形の辺(px)。既定 512")
    ap.add_argument("--seconds", type=float, default=5.0, help="長さ(秒)。既定 5")
    ap.add_argument("--resident", type=int, default=-1, help="GPU に置くデコーダブロック数(0〜36)。-1 = サーバと同じ自動判定")
    ap.add_argument("--passes", type=int, default=0, help="計測するパス数の上限(0 = 1 クリップ分すべて)")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA が見えません")
    dev = torch.device("cuda:0")
    cap = os.environ.get("H3_VRAM_LIMIT_GB", "").strip()
    if cap:   # サーバと同じ上限の模擬(小さいカードの再現用)
        total = torch.cuda.get_device_properties(dev).total_memory / 1e9
        if float(cap) < total:
            torch.cuda.set_per_process_memory_fraction(float(cap) / total, 0)
            logger.info("VRAM cap %.1f GB of %.1f GB (H3_VRAM_LIMIT_GB)", float(cap), total)

    from diffusers import AutoencoderKLMiniMaxH3

    from core import filestream, vaecache

    root = Path(__file__).resolve().parent.parent
    prequant = Path(os.environ.get("H3_TE_PREQUANT_DIR", str(root / "models" / "prequant")))
    t0 = time.time()
    vae = AutoencoderKLMiniMaxH3.from_pretrained("MiniMaxAI/MiniMax-H3", subfolder="vae", torch_dtype=torch.float32)
    cache = vaecache.ensure_cache(vae, prequant / "vae_decoder_fp16")
    vaecache.detach_blocks(vae)
    vaecache.cast_except_blocks(vae, torch.float16)
    # ブロック以外(encoder、デコーダの入出力層)を GPU へ。ブロックはプレースホルダのまま
    for name, child in vae.named_children():
        if name != "decoder":
            child.to(dev)
    for name, child in vae.decoder.named_children():
        if name != "transformer_blocks":
            child.to(dev)
    for module in (vae, vae.decoder):
        for p in module.parameters(recurse=False):
            p.data = p.data.to(dev)
    torch.cuda.synchronize()
    logger.info("vae ready in %.1fs (fp16, decoder blocks from %s). gpu allocated %.2f / reserved %.2f GiB",
                time.time() - t0, cache, torch.cuda.memory_allocated(dev) / GB, torch.cuda.memory_reserved(dev) / GB)

    blocks = list(vae.decoder.transformer_blocks)
    free_drv = torch.cuda.mem_get_info(dev)[0] / GB
    if cap:
        free_drv = min(free_drv, float(cap) * 1e9 / GB - torch.cuda.memory_allocated(dev) / GB)
    phys = filestream.physical_free_gib(dev)
    frames = int(round(args.seconds * 24))
    out_gib = frames * args.size * args.size * 3 * 4 / GB
    logger.info("free VRAM: driver %.2f GiB, physical (nvidia-smi) %s, output ~%.2f GiB", free_drv,
                "n/a" if phys is None else f"{phys:.2f} GiB", out_gib)
    if args.resident < 0:
        usable = min(free_drv, phys) if phys is not None else free_drv
        k = max(0, min(len(blocks), int((usable - 0.5 - out_gib - 1.0) / (4.5 / len(blocks)))))
    else:
        k = max(0, min(len(blocks), args.resident))
    floor = 0.5 + out_gib + 1.0
    streamer = None
    if k > 0:
        resident = filestream.ResidentLoader(blocks[:k], dev, names_root=vae, files=[cache], label="vae.decoder[:%d]" % k,
                                             min_driver_free_gib=floor if args.resident < 0 else 0.0)
        k = resident.n_units_loaded
    if k < len(blocks):
        streamer = filestream.FileBlockStreamer(blocks[k:], dev, root=vae.decoder, names_root=vae, files=[cache],
                                                label="vae.decoder[%d:]" % k, drop_units=0, ring=3)
    logger.info("decoder: %d/%d blocks resident on the GPU, %d streamed from file. gpu allocated %.2f / reserved %.2f GiB",
                k, len(blocks), len(blocks) - k, torch.cuda.memory_allocated(dev) / GB, torch.cuda.memory_reserved(dev) / GB)

    # 潜在: 24ch、時間 (17n+5 フレーム → 5n+2)、空間 1/16
    n = max(1, (frames - 5) // 17)
    lat_t, lat_hw = 5 * n + 2, args.size // 16
    z = torch.randn(1, 24, lat_t, lat_hw, lat_hw, device=dev, dtype=torch.float32) * 0.5
    std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)
    mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
    z = z * std + mean
    ny = max(1, math.ceil((args.size - 64) / (256 - 64)))
    logger.info("latents %s -> about %d spatial tiles x %d clips = %d decoder passes", tuple(z.shape), ny * ny, n, ny * ny * n)

    # パスごとの時間: デコーダの最初のブロックの pre-hook で区切る
    marks: list[float] = []
    def _mark(_m, _args):           # 戻り値 None(値を返すと forward の引数が差し替わる)
        torch.cuda.synchronize()
        marks.append(time.perf_counter())

    h = blocks[0].register_forward_pre_hook(_mark)
    torch.cuda.reset_peak_memory_stats(dev)
    t1 = time.time()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        if args.passes:
            # 時間を切り詰めたいときは最初の 1 クリップだけ
            z_small = z[:, :, : min(lat_t, 7)]
            vae.decode(z_small, return_dict=False)
        else:
            vae.decode(z, return_dict=False)
    torch.cuda.synchronize()
    h.remove()
    total = time.time() - t1
    gaps = [b - a for a, b in zip(marks, marks[1:])]
    if gaps:
        gaps_sorted = sorted(gaps)
        logger.info("decode %.1fs for %d passes: per pass median %.2fs, min %.2fs, max %.2fs (first 3: %s)", total, len(marks),
                    gaps_sorted[len(gaps) // 2], gaps_sorted[0], gaps_sorted[-1], ", ".join(f"{g:.2f}" for g in gaps[:3]))
    logger.info("peak allocated %.2f GiB, reserved %.2f GiB; driver free now %.2f GiB%s",
                torch.cuda.max_memory_allocated(dev) / GB, torch.cuda.memory_reserved(dev) / GB,
                torch.cuda.mem_get_info(dev)[0] / GB,
                "" if phys is None else f", physical free now {filestream.physical_free_gib(dev):.2f} GiB")
    if streamer is not None:
        streamer.release()


if __name__ == "__main__":
    main()

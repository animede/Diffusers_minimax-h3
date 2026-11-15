"""video VAE デコーダブロックの fp16 キャッシュ(`H3_GROUP_STREAM_HOST=file` 用、2026-11-15)。

低 VRAM 構成では VAE デコーダの 36 ブロック(fp16 で 4.5GiB)をホストから流すが、原本は fp32 の safetensors なので
起動時に `vae.to(float16)` した写しが匿名メモリに 4.8GiB 残る。RAM 16GB の PC ではこれがデコード時の RAM 不足の
主因だった。ここでは fp16 の写しを一度だけ safetensors に書き出し(キー名は原本と同じ `decoder.transformer_blocks.N.*`)、
以後はブロックのパラメータを 0 要素のプレースホルダにして、`core/filestream.FileBlockStreamer` がそのファイルから
タイルごとに流す。値は `vae.to(float16)` と同じ丸めなので出力は変わらない。
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import torch

logger = logging.getLogger("minimax_h3.vaecache")

CACHE_FILE = "decoder_blocks_fp16.safetensors"


def _meta(vae) -> dict:
    blocks = vae.decoder.transformer_blocks
    return {
        "dtype": "float16",
        "blocks": len(blocks),
        "numel": int(sum(p.numel() for b in blocks for p in b.parameters())),
        "name_or_path": str(getattr(getattr(vae, "config", None), "_name_or_path", "")),
    }


def ensure_cache(vae, cache_dir: Path) -> Path:
    """キャッシュが無い(か合わない)ときだけ、fp32 の原本から fp16 に丸めて書く。パスを返す。"""
    from safetensors.torch import save_file

    cache_dir = Path(cache_dir)
    path = cache_dir / CACHE_FILE
    meta_path = cache_dir / "meta.json"
    meta = _meta(vae)
    if path.exists() and meta_path.exists():
        try:
            if json.loads(meta_path.read_text()) == meta:
                return path
        except (OSError, ValueError):
            pass
        logger.info("vae decoder fp16 cache does not match the loaded VAE, rebuilding: %s", cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tensors = {}
    for i, block in enumerate(vae.decoder.transformer_blocks):
        prefix = f"decoder.transformer_blocks.{i}."
        for name, p in block.named_parameters():
            tensors[prefix + name] = p.detach().to(torch.float16).contiguous()
        for name, b in block.named_buffers():
            tensors[prefix + name] = b.detach().to(torch.float16).contiguous()
    tmp = path.with_suffix(".tmp")
    save_file(tensors, str(tmp))
    os.replace(tmp, path)
    meta_path.write_text(json.dumps(meta, indent=1))
    gib = sum(t.numel() * t.element_size() for t in tensors.values()) / 2**30
    logger.info("vae decoder fp16 cache written: %s (%.2f GiB, %d tensors, %.1fs)", path, gib, len(tensors), time.time() - t0)
    return path


def detach_blocks(vae) -> float:
    """デコーダブロックのパラメータ/バッファを 0 要素の fp16 プレースホルダに差し替え、原本(mmap)への参照を切る。
    外した GiB(fp32 換算)を返す。以後の実体は FileBlockStreamer がファイルから届ける。"""
    freed = 0
    for block in vae.decoder.transformer_blocks:
        for p in block.parameters():
            freed += p.numel() * p.element_size()
            p.data = torch.empty(0, dtype=torch.float16)
        for owner_name, b in list(block.named_buffers()):
            owner = block
            parts = owner_name.split(".")
            for part in parts[:-1]:
                owner = getattr(owner, part)
            freed += b.numel() * b.element_size()
            setattr(owner, parts[-1], torch.empty(0, dtype=torch.float16))
    return freed / 2**30


def cast_except_blocks(vae, dtype: torch.dtype) -> None:
    """デコーダブロック以外(encoder、デコーダの入出力層、ルート直下のテンソル)だけを dtype にする。"""
    for name, child in vae.named_children():
        if name != "decoder":
            child.to(dtype)
    for name, child in vae.decoder.named_children():
        if name != "transformer_blocks":
            child.to(dtype)
    for module in (vae, vae.decoder):
        for p in module.parameters(recurse=False):
            p.data = p.data.to(dtype)
        for bname, b in list(module.named_buffers(recurse=False)):
            setattr(module, bname, b.to(dtype))

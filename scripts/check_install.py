#!/usr/bin/env python
"""インストール後の確認(scripts/install.sh と scripts/install.ps1 の最後に呼ばれる)。
torch / diffusers / transformers / torchao のバージョンと CUDA の有無、SageAttention と内蔵 LLM の状態を表示する。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

print("torch", torch.__version__, "| cuda", torch.version.cuda, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu:", p.name, "%.1f GB" % (p.total_memory / 1e9), "| sm_%d%d" % torch.cuda.get_device_capability(0))
import diffusers, transformers, torchao  # noqa: E402,E401

print("diffusers", diffusers.__version__, "| transformers", transformers.__version__, "| torchao", torchao.__version__)
try:
    import sageattention  # noqa: F401

    print("sageattention: installed (H3_ATTN_BACKEND=sage, sm_120 only)")
except ImportError:
    print("sageattention: not installed -> H3_ATTN_BACKEND=default (set automatically by H3_PROFILE=auto)")
import shutil  # noqa: E402

ff = shutil.which("ffmpeg") and shutil.which("ffprobe")
print("ffmpeg / ffprobe:", "found" if ff else "NOT found -> gallery length / resolution and video concatenation need it (generation works without). "
      "Windows: winget install --id Gyan.FFmpeg -e --source winget (then reopen PowerShell); Linux: sudo apt install ffmpeg")
from core import local_llm  # noqa: E402

exe = local_llm.find_binary()
print("built-in LLM (llama-server):", exe if exe else "not fetched yet -> fetched automatically on first fallback, or install with --local-llm / -LocalLLM")

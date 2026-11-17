#!/usr/bin/env python
"""内蔵 LLM(フォールバック)用のビルド済み llama-server と Gemma 4 E4B の GGUF を事前取得する。

  venv/bin/python scripts/fetch_llama_server.py            # llama-server(約 25MB)だけ
  venv/bin/python scripts/fetch_llama_server.py --model    # GGUF(約 4.4GB)も

取得しなくても、初めてフォールバックしたときにサーバが自動で取得する(core/local_llm.py)。
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from core import local_llm  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--model", action="store_true", help="GGUF(約 4.4GB)も取得する")
args = parser.parse_args()

exe = local_llm.find_binary() or local_llm.download_binary()
print(f"llama-server: {exe}")
if args.model:
    print(f"model: {local_llm.download_model()}")
else:
    print(f"model: {local_llm.model_path()} ({'あり' if local_llm.model_path().exists() else '未取得: 初回フォールバック時に自動取得、または --model'})")

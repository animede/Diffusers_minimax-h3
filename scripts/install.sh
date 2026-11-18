#!/usr/bin/env bash
# minimax-h3 のワンショットインストール。
#
#   scripts/install.sh            # venv を作って requirements.txt を入れる
#   scripts/install.sh --sage     # さらに SageAttention(sm_120 向け)を入れる
#   scripts/install.sh --local-llm # 内蔵 LLM(フォールバック)用に、llama.cpp のビルド済み llama-server(約 25MB、
#                                  # コンパイラ不要)を tools/llama/ へ取得する。GGUF(約 4.4GB)は初回フォールバック時に
#                                  # 自動取得(先に取るなら scripts/fetch_llama_server.py --model)
#
# - venv はリポジトリ直下の venv/ に作る(既にあればそのまま使う)。
# - torch は cu128 wheel を使う(requirements.txt の --extra-index-url)。
# - diffusers は検証済みコミット f37ab93 に固定(requirements.txt を参照)。
# - SageAttention は既定では入れない。sm_120 以外のカードでは使えないので、
#   入れないときは H3_ATTN_BACKEND=default で起動する。--sage を付けると
#   third_party/wheels/ にビルド済み wheel があればそれを、無ければ
#   scripts/build_sageattention.sh でソースビルドする(CUDA_HOME が必要)。
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3.12}"
WANT_SAGE=0
WANT_LOCAL_LLM=0
for a in "$@"; do
  case "$a" in
    --sage) WANT_SAGE=1 ;;
    --local-llm) WANT_LOCAL_LLM=1 ;;
    -h|--help) sed -n 2,14p "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: $PYTHON が見つかりません(Python 3.12 が必要。PYTHON=python3.12 のように指定可)" >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "error: git が見つかりません(diffusers をコミット固定で入れるのに必要)" >&2
  exit 1
fi

if [ ! -x venv/bin/python ]; then
  echo "==> venv を作成: $PYTHON -m venv venv"
  "$PYTHON" -m venv venv
fi
venv/bin/pip install --upgrade pip wheel setuptools

echo "==> requirements.txt をインストール"
venv/bin/pip install -r requirements.txt

if [ "$WANT_SAGE" = 1 ]; then
  wheel="$(ls third_party/wheels/sageattention-*.whl 2>/dev/null | head -1 || true)"
  if [ -n "$wheel" ]; then
    echo "==> SageAttention: ビルド済み wheel をインストール($wheel)"
    venv/bin/pip install "$wheel"
  else
    echo "==> SageAttention: ビルド済み wheel が無いのでソースビルド(scripts/build_sageattention.sh)"
    : "${CUDA_HOME:?CUDA_HOME を指定してください(例: CUDA_HOME=/usr/local/cuda-12.8)}"
    scripts/build_sageattention.sh
  fi
fi

if [ "$WANT_LOCAL_LLM" = 1 ]; then
  echo "==> 内蔵 LLM: ビルド済み llama-server を取得"
  venv/bin/python scripts/fetch_llama_server.py
fi

echo
echo "==> 確認"
venv/bin/python scripts/check_install.py

echo
echo "完了。次はモデルの準備:"
echo "  低 VRAM / 低 RAM の PC(32GB 以下・RAM 48GB 未満): venv/bin/python scripts/prepare_lowvram.py"
echo "  48GB 以上・RAM 64GB 以上:                         venv/bin/python scripts/download_t2va.py"
echo "起動:                    venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611"

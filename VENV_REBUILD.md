# minimax-h3 venv 完全独立再構築手順(Phase 5 用メモ)

現状(Phase 0)の `venv` は **/home/animede/minimax-h3/venv への symlink** であり、
再構築は行っていない。将来この backend を旧ディレクトリから完全に切り離す場合は
以下の手順で venv を独立再構築する。実際の旧 venv の完全なパッケージ一覧は
同ディレクトリの **`venv-freeze.txt`**(`pip freeze` 実測、343行)を参照。

## 前提・重要な固定バージョン

| パッケージ | バージョン | 備考 |
|---|---|---|
| Python | 3.12(旧 venv は cp312) | |
| torch | **2.9.0+cu128** | torchao 0.17 の対応上限。2.11 系に上げてはいけない(torchao/SageAttention wheel が壊れる) |
| torchvision / torchaudio | 0.24.0 / 2.9.0 | torch 2.9 対応版 |
| diffusers | **git PR #14355 コミット固定 `f37ab93e621d5ce206c9662e8291ca8b67d9c555`** | MiniMax-H3 対応 PR。マージ前のため必ずコミットハッシュ固定でインストールすること |
| transformers | 5.14.1 | |
| torchao | **0.17.0**(公式 wheel) | prequant キャッシュ(models/prequant、36GB)はこのバージョンで生成されたもの。バージョンを変えると再量子化が必要になる可能性あり |
| bitsandbytes | 0.49.0 | TE NF4(既定 `H3_TE_QUANT=bnb-4bit`) |
| accelerate | 1.12.0 | |
| fastapi / uvicorn | 0.104.1 / 0.24.0 | |
| sageattention | 2.2.0(**自前ビルド wheel**) | `third_party/wheels/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl` に保存済み |

## 手順

```bash
cd /home/animede/diffusers-movie-server/backends/minimax-h3
rm venv                              # symlink を外す(旧ディレクトリ側は無変更)
python3.12 -m venv venv
venv/bin/pip install --upgrade pip

# torch 2.9.0+cu128(cu128 wheel を明示)
venv/bin/pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu128

# diffusers は PR #14355 のコミットハッシュ固定
venv/bin/pip install \
  "git+https://github.com/huggingface/diffusers.git@f37ab93e621d5ce206c9662e8291ca8b67d9c555"

# 主要パッケージ(完全な一覧は venv-freeze.txt)
venv/bin/pip install transformers==5.14.1 torchao==0.17.0 bitsandbytes==0.49.0 \
  accelerate==1.12.0 fastapi==0.104.1 uvicorn==0.24.0 safetensors==0.8.0 \
  python-multipart pillow numpy==2.2.6

# SageAttention: まずビルド済み wheel を使う(再ビルド不要)
venv/bin/pip install third_party/wheels/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl
```

## SageAttention を再ビルドする場合の注意(wheel が使えない場合のみ)

**このマシンで CUDA 拡張をソースビルドする際は必ず並列数を制限すること**
(過去に MAX_JOBS 無制限の並列 nvcc がホスト RAM を食い潰し、OOM でシステム全体が
巻き添え強制終了した事故がある — diffusers-server/CLAUDE.md 50番):

```bash
cd third_party/SageAttention
MAX_JOBS=4 NVCC_THREADS=2 TORCH_CUDA_ARCH_LIST="12.0" \
  systemd-run --user --scope -p MemoryMax=45G -p MemorySwapMax=0 \
  ../../venv/bin/pip install --no-build-isolation .
```

- `TORCH_CUDA_ARCH_LIST="12.0"`: sm_120(Blackwell)向けに絞るとビルド時間短縮。
- 参考スクリプト: `scripts/build_sageattention.sh`(旧リポジトリ由来)。

## 再構築後の検証

1. `venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
   → `2.9.0+cu128 True`
2. `venv/bin/python -c "import diffusers; print(diffusers.__version__)"`(PR コミットの dev 版)
3. `bash run.sh` で起動し `GET /api/status` が 200・`/api/t2i`(768x768)が完走すること。
4. prequant キャッシュ(`models/prequant`、symlink 先 36GB)がそのまま読めること
   (torchao のバージョンを変えた場合はキャッシュ再生成が必要になる可能性がある。
   起動ログで prequant のロード成否を確認する)。

## 追加依存(2026-10-07)
- `pip install comfy-kitchen==0.2.37 --no-deps`(H3_PRUNED_QUANT=ck-w4a8 の W4A8 カーネル。Apache-2.0・依存なし・torch 2.9 で動作確認済み)

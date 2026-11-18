# minimax-h3

**日本語** | [English](README.en.md)

**最新版: 2026-11-18 19:25 JST** — 空き VRAM が足りればデコードの間だけ VAE デコーダを GPU に置き切る(`H3_VAE_DECODER_RESIDENT`、デコード 86 → 34 秒) / Windows: RAM に収まる重みはページキャッシュに残す / 6GB・8GB 実機の生成時間は公平な測定まで非掲載に / ffmpeg 未導入時はトレースバックではなく 1 度だけ導入案内を出す / turbo LoRA をホスト RAM から流してピーク VRAM を 4.19GB → 2.82GB に(6GB のカードで Windows が共有メモリへ退避して極端に遅くなる問題の対策)/ Windows 10 + RTX 3050 6GB で生成を確認 / Windows・低 VRAM 向けクイックスタートを冒頭に追加。変更の履歴は [docs/HISTORY.md](docs/HISTORY.md)。

**MiniMax H3 (Hailuo 3.0) — 動画+ステレオ音声を1回のデノイズで同時生成する 33B オムニモーダルモデルを、普及帯 GPU で動かす検証アプリ。**
テキスト / 画像参照 / 音声参照から **動画+ステレオ音声** を生成し、静止画のみ(T2I / Ref2I)も出せる。
基盤は diffusers の Modular Pipeline(PR #14355、コミット **f37ab93** にピン留め)。
**6GB 単体・8GB×2・実 RTX 4060 Ti 16GB 単体、Windows 11 + RTX 4060 8GB(RAM 16GB)で動くことを実測済み**で、32GB 単騎ならリアルタイム会話生成(実時間の 0.95 倍)まで到達している。

![GUI メイン画面](docs/images/ui_main.png)

*ブラウザから使う単一ページ UI。5 つのモードをタブで切替(動画: **T2VA**=テキスト / **FL2VA**=フレーム / **Ref2VA**=参照、静止画: **T2I** / **Ref2I**)。下段は生成物ギャラリーで、チェックした動画を連結して書き出せる。プロンプトはローカル LLM で H3 公式ガイドの形式へ強化でき、FirstBlockCache・Sage・Turbo・量子化・低 VRAM モードなどは**再起動なしで**このパネルから切替。日英切替あり。*

## クイックスタート(Windows・VRAM 6〜16GB の PC)

X などから来た方向けの最短手順。詳細は下の「[インストール](#インストール)」「[起動](#起動)」。

```powershell
winget install --id Git.Git -e --source winget
winget install --id Python.Python.3.12 -e --source winget
winget install --id Gyan.FFmpeg -e --source winget
git clone https://github.com/animede/Diffusers_minimax-h3.git
cd Diffusers_minimax-h3
.\scripts\install.bat
.\venv\Scripts\python.exe scripts\prepare_lowvram.py
.\run.bat
```

- winget のあとは PowerShell を一度開き直す。ffmpeg はギャラリーの長さ・解像度の表示と動画の連結に使う(無くても生成はできるが、ログに案内が出る)。`install.bat` は venv を作って依存を入れる(数分)。
- `prepare_lowvram.py` はモデルの準備。144GB を落とすのではなく、66GB の重みを 1 つずつ落として量子化しながら消すので、**ディスクは約 60GB**、RAM 16GB で足りる。所要時間はほぼ回線速度で決まる(数時間)。途中で止まっても再実行すれば続きから進む。Ref2VA / Ref2I(参照素材から動画)も使うなら `--ref` を付ける(+32GB)。
- `run.bat` でサーバが起動し、ブラウザで `http://127.0.0.1:8611` を開く。初回起動は投影 TE の量子化キャッシュを作るので数分かかる。GPU の大きさに合わせた設定(低 VRAM モード、turbo 4 ステップ)は自動で入る。
- 目安: 768²・5 秒の動画(turbo 4 ステップ)が、A4000 に 6GB の上限を切った模擬(NVMe 3.3GB/s)で 211 秒、ピーク VRAM 約 2.8GB(768²)・3.5GB(1024²)。RAM 16GB の PC は毎ステップ 30GB の重みを NVMe から読むので、SSD の速さがそのまま効く。6GB の GPU(RTX 3050 など)でも 768² が動く。6GB / 8GB の実機(RTX 3050 6GB、RTX 4060 Laptop 8GB、いずれも RAM 16GB の Windows)は動作確認の段階で、生成時間はまだ公平な条件(同じ NVMe、同じ設定)で測っていないため載せていない。測り次第追記する。
- うまく行かないときは「[Windows](#windows)」節のトラブル対処と [Issues](https://github.com/animede/Diffusers_minimax-h3/issues) へ。

## 特徴

- **普及帯 GPU で動く**: transformer int8 のブロック単位ストリーミング(`H3_LOWVRAM=group`)、投影 TE(Qwen3-VL-4B + 線形写像で 32B TE を代替)、video VAE の fp16 化、TE / token_refiner / VAE デコーダのストリーミングにより、768²×5 秒が **4.6GB**、1024²×5 秒が **6.4GB** で完走する。ホスト RAM が 32GB / 16GB でも量子化キャッシュから直接流せる。
- **固定費の削減**: 律速はデノイズではなく「モデルの load/free」だった。transformer 常駐・量子化済みキャッシュ・pinned 退避・参照 prefix の KV キャッシュで、48GB 級で t2i 9.7 秒 / t2va 5 秒 44 秒、96GB 級で t2i 7.4 秒。
- **高速化オプション**: FirstBlockCache、SageAttention(sm_120)、lightx2v turbo LoRA(4 / 8 ステップ蒸留)、ck-w4a8 量子化、AdaLN-pruned 版 transformer_ref。
- **5 つの生成モードとバッチ**: T2VA / FL2VA / Ref2VA(画像最大 9・動画最大 3・音声最大 3 の順序付き参照、音声参照でリップシンク)/ T2I / Ref2I。静止画・参照付き動画は最大 24 場面のバッチ生成に対応。**静止画(T2I / Ref2I)は画像専用モデルの MiniMax-H3-Image ではなく、この動画モデル(MiniMax-H3)で 22 フレームの超短尺動画を生成して中央フレームを PNG にしたもの**。専用モデルより遅いが、FL2VA の先頭フレームや Ref2VA の参照に使うと画風が完全に一致する。MiniMax-H3-Image は別リポジトリ [minimax-h3-image-server](https://github.com/animede/minimax-h3-image-server) で扱う。
- **LLM プロンプト強化**: 外部 LLM(OpenAI 互換 API、`H3_LLM_URL`)で入力を H3 公式スキル `h3-prompt-writing` の形式へ整形。生成前の検証 + 修復ループつき。外部 LLM に接続できないときは CPU 上の内蔵 LLM(Gemma 4 E4B、GGUF Q4_0、約 4.4GB、初回に自動 DL、thinking OFF)へ自動フォールバックする(ACE-Step-1.5-fast の easy-music と同じ考え方。実体は llama.cpp のビルド済み `llama-server` を子プロセスで起動するので、コンパイラ不要で Windows でもそのまま動く)。
- **品質の担保**: diffusers の更新や量子化キャッシュの導入は、同一 seed の MD5 一致で回帰確認している。

## サンプル

[![t2va 768² 5 秒、turbo 4 ステップ、RTX 4060 Laptop 8GB](docs/samples/t2va_768_turbo4_rtx4060laptop.gif)](docs/samples/t2va_768_turbo4_rtx4060laptop.mp4)

*利用者の Windows 11 + RTX 4060 Laptop 8GB / RAM 16GB で生成した t2va(768²・5 秒・turbo 4 ステップ)。GIF は縮小・減色したプレビューで、クリックすると音声つきの元の mp4(1MB)が開く。*

## 対応 GPU(早見表)

t2va 5 秒・768² の実測。詳細な行(80GB / 18GB / 12GB 級など)と測定条件は [docs/FEATURES.md](docs/FEATURES.md#vram-構成と常駐設計) を参照。

| GPU | 起動時の指定 | 結果 |
|---|---|---|
| 96GB | (指定なし = 既定) | peak 92GB / 約 160 秒 |
| 48GB 級 | `H3_LOWVRAM=1 H3_TE_PRUNE=1 H3_TE_DEVICE=cuda:1 H3_VIDEO_VAE_FP16=1 H3_KEEP_TRANSFORMER=1`(2 枚構成の推奨) | t2i 9.7 秒 / t2va 44.2 秒(turbo 4 ステップ) |
| 24 / 32GB 単騎 | ck-w4a8 + AdaLN-pruned + pinned 退避の一式([詳細](docs/FEATURES.md#2026-10-更新-24gb32gb-単騎でリアルタイム会話生成)) | 320×448・141f が **5.57 秒 = 実時間の 0.95 倍**(32GB)/ 7.1 秒(24GB) |
| 32GB 級 | `H3_LOWVRAM=group` | peak 28.7GB / 約 280 秒 |
| 16GB 単体(実 4060 Ti / A4000) | `H3_LOWVRAM=group H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_VIDEO_VAE_FP16=1` | 4060 Ti: peak 11.4GB(2026-08)。A4000(2026-11-17): 30 ステップ peak 8.0GB / 790 秒、turbo 4 ステップ(`H3_TURBO_LORA=1`)peak 7.0GB / 151 秒 ※ |
| 6GB / 8GB 単体 | 上の行 + `H3_TE_STREAM=1 H3_TE_DIET=1` | 30 ステップ: 768² 4.62GB / 1024² 6.41GB(48GB 機で 272 / 552 秒)。turbo 4 ステップ(A4000 に 8GB の上限、VAE デコーダをストリーミング): 768² 3.08GB / 209 秒、1024² 3.73GB / 376 秒。fl2va 3.37GB / 225 秒、ref2va 3.02GB / 299 秒(いずれも 768²・5 秒)。ホスト RAM 48GB 未満の `file` ストリーミング + turbo(A4000 に 6GB の上限、LoRA はホスト RAM から流す): 768² 2.82GB / 211 秒、1024² 3.48GB / 380 秒(ref2va は LoRA 常駐時の値で 768² 4.13GB / 276 秒)。実機(Windows の RTX 3050 6GB / RTX 4060 Laptop 8GB、RAM 16GB)は動作確認済み、時間は未計測(※ 参照) |
| 8GB×2 | 16GB 単体の指定 + `H3_TE_DEVICE=cuda:1` | peak 7.23GB ※ |

※ 時間は PCIe 帯域に強く依存する(低 VRAM 構成は毎ステップの重み転送が律速)。4060 Ti(Gen3 x4、sm_89、30 ステップ)では 16GB 単体・8GB×2 とも t2va 5 秒に約 25 分かかったが、同じ構成が A4000(Gen4 x4)では 790 秒、48GB 機の GPU0 では 272 秒で、Gen4 x16 なら 4〜5 分の見込み。**turbo LoRA(`H3_TURBO_LORA=1`、4 ステップ)が group でも使えるようになったため、転送律速の「動くが遅い」はほぼ解消できる**: x4 接続の A4000 でも t2va 5 秒が 151 秒(16GB 級)、8GB 級の指定(VAE デコーダのストリーミング込み)でも 768² 209 秒 / 1024² 376 秒で完走する(転送回数がステップ数に比例して減るため)。8GB 級の値は A4000 に 8GB の上限(`H3_VRAM_LIMIT_GB`)を切った模擬で、デノイズ中のピーク 3.1〜3.7GB は 6GB 級にも収まる。sm_120 以外のカードは `H3_ATTN_BACKEND=default` も必要(既定の sage は sm_120 専用ビルド。未指定なら `H3_PROFILE=auto` が自動で入れる)。構成をまたぐと同一 seed でも構図が変わる(劣化ではなく軌道の分岐)ことは [性能の概要](docs/FEATURES.md#性能の概要) に詳しい。 RAM 16GB の PC(file ストリーミング)では毎ステップ約 30GB の重みを SSD から読むため、SSD の読み出し速度も効く(A4000 + NVMe 3.3GB/s の 6GB 上限で 211 秒。実効 500MB/s 級の SSD では読みだけで 1 ステップ 1 分かかる)。6GB / 8GB の実機(RTX 3050 6GB、RTX 4060 Laptop 8GB、いずれも RAM 16GB の Windows)は動作確認の段階で、生成時間はまだ公平な条件(同じ NVMe、同じ設定)で測っていないため載せていない。測り次第追記する。

## インストール

### 必要なもの

| | 要件 |
|---|---|
| GPU | 最小: **6GB 単体**(768²)/ **8GB 単体**(1024²)/ **16GB 単体**(全機能)。快適動作は **48GB 級**を推奨 |
| ホスト RAM | 64GB 以上を推奨(`H3_LOWVRAM=group` は int8 重み ~34GB をピン留めするため 48GB 以上の空きが要る)。48GB 未満では量子化キャッシュから直接流す方式に自動で切り替わる(`H3_GROUP_STREAM_HOST=file`、32GB 機で 1K は同速・512² は約 2 倍) |
| ディスク | 約 145GB(T2VA / FL2VA のみ)/ 約 207GB(Ref2VA も使う場合)。低 VRAM / 低 RAM の準備経路(`scripts/prepare_lowvram.py`)は量子化キャッシュだけを置くので約 60GB(Ref2VA も使うなら `--ref` で +32GB) |
| CUDA | 12.8 系(torch 2.9.0+cu128 に合わせる)。SageAttention をビルドするなら `nvcc` も同系統 |
| Python | 3.12 |
| その他 | `ffmpeg`(ギャラリーの連結・情報取得に使用。無くても生成自体は動く) |

### 手順

```bash
git clone https://github.com/animede/Diffusers_minimax-h3.git
cd Diffusers_minimax-h3
scripts/install.sh
```

`scripts/install.sh` は `venv/` を作って [requirements.txt](requirements.txt) を入れ、最後に torch / diffusers / transformers のバージョンを表示する。必要なのは Python 3.12 と `git` だけで、CUDA のビルドは発生しない(数分で終わる)。`--sage` を付けると SageAttention も入れる(下記)。`--local-llm` を付けると内蔵 LLM(Gemma 4 E4B)用のビルド済み `llama-server` も先に取得する(下記。付けなくても初回のフォールバック時に自動取得される)。

<details>
<summary>手動で入れる場合(requirements.txt と同じ内容)</summary>

```bash
python3.12 -m venv venv

# PyTorch (cu128)
venv/bin/pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128

# ============================================================================
# 【重要】diffusers は検証済みコミット f37ab93(PR #14355 のマージ最終形)に固定する。
#        pip install diffusers(PyPI 版)や最新 main を入れないこと(下の注記を参照)
# ============================================================================
venv/bin/pip install "git+https://github.com/huggingface/diffusers.git@f37ab93e621d5ce206c9662e8291ca8b67d9c555"

# transformers は 5.14.1 以上が必須(5.1.0 には Qwen3VLProcessor.create_mm_token_type_ids が無い)
venv/bin/pip install "transformers==5.14.1" accelerate==1.12.0 safetensors huggingface_hub

# 動画/音声の多重化と Web API
venv/bin/pip install av==16.0.1 fastapi==0.104.1 "uvicorn==0.24.0" python-multipart pillow numpy

# text_encoder の 4bit 量子化(既定の H3_TE_QUANT=bnb-4bit に必要)
venv/bin/pip install bitsandbytes==0.49.0

# transformer の int8 量子化を使う場合のみ(H3_TRANSFORMER_QUANT=int8 / H3_LOWVRAM)
#   0.18 以降は torch>=2.11 を要求するため 0.17.0 を固定する
venv/bin/pip install torchao==0.17.0
```

</details>

> **diffusers の更新について**: インストール自体は上の 1 行で済む(純 Python なのでビルドは無く、`git` があれば数十秒で入る)。注意が要るのは**あとから diffusers を上げるとき**で、計算経路が変わって出力が変わりうるため、コミット固定のまま使うこと。f37ab93 への追従も、全経路(t2i / t2va / バッチ / ref2va / 参照バッチ)が旧ピン abc5e9b と同一 seed の MD5 で等価であることを確認してから行った。さらに上げる場合は同じ回帰確認(`scripts/run_regression.py` と docs/internal の基準値)を踏む。普通に使うだけなら不要。

### Windows

依存パッケージは Windows 用の wheel がそろっています(torch 2.9.0+cu128 は PyTorch の index から、torchao は PyPI の純 Python wheel、bitsandbytes / av / psutil は PyPI の Windows wheel)。コンパイラや Visual Studio Build Tools は不要です。SageAttention は Windows 向けのビルドが無いため使わず、起動時に `H3_PROFILE=auto` が `H3_ATTN_BACKEND=default`(SDPA)を自動で入れます。コード側の OS 依存(RAM の取得、ファイルの先読み指示、一時ファイルの置き場所、内蔵 LLM の子プロセス)は Windows 用に分岐済みです。Hugging Face のキャッシュは Windows では symlink を使わない設定(`HF_HUB_DISABLE_SYMLINKS=1`)を自動で入れます(開発者モードが OFF の PC で初回ダウンロード中に `WinError 1314` で止まる huggingface_hub の競合を避けるため。ディスク使用量は変わりません)。

必要なもの: Python 3.12(`py -3.12` で呼べること)、git、CUDA 12.8 系に対応した NVIDIA ドライバ。`ffmpeg` は任意(ギャラリーの連結用)。git や Python が無い場合は winget で入れる(入れたあと PowerShell を開き直す):

```powershell
winget install --id Git.Git -e --source winget
winget install --id Python.Python.3.12 -e --source winget
winget install --id Gyan.FFmpeg -e --source winget
```

```powershell
git clone https://github.com/animede/Diffusers_minimax-h3.git
cd Diffusers_minimax-h3
.\scripts\install.bat
```

`install.bat` は PowerShell の実行ポリシーに関係なく `scripts\install.ps1` を実行する(Windows の既定では未署名の `.ps1` を直接実行できないため)。venv を作り、torch(cu128)と `requirements.txt` を入れる。`-LocalLLM` を付けると内蔵 LLM のバイナリも取得する。続けてモデルを取得する:

```powershell
.\venv\Scripts\python.exe scripts\prepare_lowvram.py
```

(VRAM 48GB 以上・RAM 64GB 以上の PC なら `scripts\download_t2va.py` でもよい。下の「低 VRAM / 低 RAM の PC の準備」参照。)手動で入れる場合は次の順に実行する(PyPI の Windows 版 torch は CPU 専用なので、torch を先に PyTorch の index から入れる順序が必要):

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

起動方法(環境変数の渡し方、VRAM 級別の例)は「[起動](#起動)」の Windows 節を参照。`.ps1` を直接実行したい場合は `powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1` のように Bypass を付けるか、一度だけ `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` を実行しておく。

内蔵 LLM(外部 LLM に接続できないときのフォールバック)は、インストール時に `-LocalLLM` を付けなくても後から入れられる。初回のフォールバック時に自動取得されるが、先に取っておく場合は次を実行する(1 行目がビルド済み `llama-server.exe`(約 25MB)、2 行目が Gemma 4 E4B の GGUF(約 4.4GB)。GGUF は省略可):

```powershell
.\venv\Scripts\python.exe scripts\fetch_llama_server.py
.\venv\Scripts\python.exe scripts\fetch_llama_server.py --model
```

`.\scripts\install.bat -LocalLLM` を後から実行しても同じ(既に入っている依存はそのまま)。詳細は下の「[内蔵 LLM(任意)](#内蔵-llm任意)」。

> Windows の実機状況(2026-11-18): 利用者の Windows 11 + RTX 4060 8GB / RAM 16GB で、インストール → `prepare_lowvram.py` → 起動 → **動画生成まで確認済み(30 ステップと turbo 4 ステップの両方)**。生成中のタスクマネージャの表示は専用 GPU メモリ 6.3 / 8.0GB、共有 GPU メモリ 2.0GB(ノート PC の RTX 4060 Laptop)。専用側はサーバが報告するピーク(約 4.2GB)より大きく見えるが、これは torch が確保したまま使い回す領域とデスクトップの分を含むため。共有側はストリーミング用にピン留めしたホスト RAM(受け皿)で、Windows はこれを共有 GPU メモリとして数える。Windows 10 + RTX 3050 6GB / RAM 16GB でも turbo 768²・5 秒の生成を確認(その過程で見つかった HF キャッシュの symlink、原本シャードの削除、実行ポリシー、文字コードの各問題は修正済み)。この機では専用 GPU メモリ 5.8 / 6.0GB・共有 3.4GB と VRAM があふれ、Windows が共有メモリへ退避して極端に遅くなっていた(ハングもエラーも出ない)。turbo LoRA 約 1.2GB を GPU 常駐からホスト RAM 経由のストリーミングに変えてピークを 4.19GB → 2.82GB に下げたので(2026-11-18 15 時以降の版)、6GB でも収まる。どちらの実機も生成時間は公平な条件で測り直してから載せる。同じ方式の画像サーバ [minimax-h3-image-server](https://github.com/animede/minimax-h3-image-server) は Windows 11 + RTX 3050 で生成まで確認済み。問題があれば [Issues](https://github.com/animede/Diffusers_minimax-h3/issues) へ。

### SageAttention(任意、既定で有効。Linux のみ)

既定の `H3_ATTN_BACKEND=sage` を使うには sm_120 向けのビルドが要る(Linux 向けの事前ビルド wheel は無い)。**ビルドしない場合は `H3_ATTN_BACKEND=default` を指定して起動する**(デノイズが約 12% 遅くなるだけで機能に影響はない)。

```bash
CUDA_HOME=/usr/local/cuda-12.8 scripts/install.sh --sage     # または scripts/build_sageattention.sh
```

`--sage` は `third_party/wheels/` にビルド済み wheel があればそれを入れ、無ければソースビルドする。

### 内蔵 LLM(任意)

LLM 強化は既定で外部の OpenAI 互換 API(`H3_LLM_URL`)を使い、接続できないときは CPU 上の内蔵 LLM(Gemma 4 E4B、GGUF Q4_0、thinking OFF)へ自動フォールバックする。内蔵 LLM の実体は llama.cpp の [GitHub Releases](https://github.com/ggml-org/llama.cpp/releases) にあるビルド済み `llama-server`(CPU 版、約 25MB。固定タグ `b12475`、`H3_LLAMA_CPP_TAG` で変更可)で、サーバがこれを子プロセスとして起動し、OpenAI 互換 API(既定 `127.0.0.1:64661`、`H3_LLM_LOCAL_PORT`)で話す。**コンパイラは不要**で、Linux x64 / Windows x64 / macOS のバイナリが自動で選ばれる。

```bash
scripts/install.sh --local-llm            # llama-server を tools/llama/ へ先に取得(任意)
venv/bin/python scripts/fetch_llama_server.py --model   # GGUF(約 4.4GB)も先に取得(任意)
```

RAM の少ない PC(16GB 級)では動画生成と内蔵 LLM が RAM を奪い合うため、llama-server は最後の利用から 300 秒(`H3_LLM_LOCAL_IDLE_S`)で止まり、ホスト RAM 32GB 未満の PC では生成の開始時にも止まる(`H3_LLM_LOCAL_STOP_ON_GENERATE=auto`)。次に LLM 強化を押せば数秒で再起動する。

どちらも省略でき、後から実行しても同じ(インストール済みの環境に足せる)。初めてフォールバックしたときに `llama-server` は `tools/llama/` へ、モデルは `models/llm/` へ自動取得される。子プロセスはサーバ終了時に止まり、サーバが強制終了された場合も Windows は Job Object(KILL_ON_JOB_CLOSE)、Linux は `PR_SET_PDEATHSIG` で道連れにするので孤児にならない(Linux は SIGKILL で実測、Windows は未検証)。既にあるものを使うなら `H3_LLAMA_SERVER=<llama-server のパス>`、`H3_LLM_LOCAL_GGUF=<GGUF のパス>` で指定する(自分で起動した llama-server が `H3_LLM_LOCAL_PORT` で応答していればそれをそのまま使う)。


### モデルの取得

**VRAM 32GB 以下、またはホスト RAM 48GB 未満の PC は、次の「低 VRAM / 低 RAM の PC の準備」を使う**(こちらの手順は 144GB をそのまま落とすので、低 VRAM 級では無駄が多く、RAM の少ない PC では量子化キャッシュを作れず起動時に止まる)。

`MiniMaxAI/MiniMax-H3` は 498.6GB あるが、T2VA / FL2VA には `transformer/`(66.3GB)+ `text_encoder/`(66.7GB)+ `vae/`(10.4GB)+ `audio_vae/`(0.6GB)+ 設定類の**約 144GB** で足りる。`transformer_ref/`(Ref2VA 用、66.3GB)は必要なときだけ取得し、**丸ごと `snapshot_download` しないこと**。

```bash
venv/bin/python scripts/download_t2va.py
```

内部で `allow_patterns` を使って必要なサブフォルダのみを取得する。ダウンロード中は `logs/du_monitor.log` でキャッシュサイズを監視できる。bf16 の transformer / transformer_ref がキャッシュに無いときは、`H3_ALLOW_WEIGHT_DOWNLOAD=1` のときだけ自動取得する(既定はエラーで案内)。

### 低 VRAM / 低 RAM の PC の準備

`H3_LOWVRAM=group`(32GB 以下の級)で重みを量子化キャッシュのファイルから流す構成は、**int8 量子化済みキャッシュ(`models/prequant/transformer_int8`、約 32GB)が先にあること**が前提。無いと bf16(66GB)から量子化する従来経路に落ち、ホスト RAM 40GB のガードで止まる(RAM 16GB の PC で実際にこのエラーになる)。`scripts/prepare_lowvram.py` は、bf16 のシャード(14 個、各 4.7GB)を 1 つずつ落として量子化して消す方式でこのキャッシュを作る。RAM は数 GB、VRAM は不要(既定は CPU で量子化、`--device cuda` も可)、ディスクのピークはシャード 1 つ分 + 出力。出来上がりはサーバが自分で作るキャッシュと**ビット単位で同一**(確認済み)。あわせて VAE デコーダの fp16 キャッシュを作り、投影 TE(Qwen3-VL-4B + ClipProj)を先に取得する。32B の text_encoder は落とさない。

```bash
venv/bin/python scripts/prepare_lowvram.py
```

Windows:

```powershell
.\venv\Scripts\python.exe scripts\prepare_lowvram.py
```

所要時間はほぼ 66GB のダウンロード時間(量子化はシャードあたり数秒〜数十秒)。キャッシュは HF 側のスナップショットと照合されるが、bf16 の原本がローカルに無い PC では(作り直しができないので)照合を省略してそのまま使う。再実行は冪等で、既にあるキャッシュは検査だけして数秒で終わる。`--keep-source` で bf16 シャードを残せる(+66GB、再構築が速い)。投影 TE の NF4 キャッシュは初回起動時にサーバが作る(VRAM 8GB・RAM 16GB で可)。

**Ref2VA / Ref2I も使うなら `--ref` を付ける。** 参照モードは `transformer` とは別の重み `transformer_ref/`(66GB、全シャードの内容が異なる)を使うので、そのキャッシュ `models/prequant/transformer_ref_int8`(約 32GB)も同じ方式で作る。ホスト RAM 48GB 未満の PC ではこれが無いと参照モードは起動時に止まる(作り方を案内するエラーになる)。後から足してもよい(再実行は冪等で、既にある `transformer_int8` は数秒の検査だけ)。

```bash
venv/bin/python scripts/prepare_lowvram.py --ref
```

Windows:

```powershell
.\venv\Scripts\python.exe scripts\prepare_lowvram.py --ref
```

## 起動

### Linux

```bash
venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
```

何も指定しなければ **`H3_PROFILE=auto`** が計算用 GPU の総 VRAM から 96 / 80 / 48 / 32 / 16 / 8GB 級の既定値一式(下の起動例と同じ内容)を選ぶ。32GB 以下の級(group)では turbo LoRA(4 ステップ)も既定で ON になる(group では起動時に巻かないと GUI からも使えないため。GUI で turbo を外せば 30 ステップ、`H3_TURBO_LORA=0` で起動時から無効)。48GB 以上は GUI の turbo チェックでリクエストごとに使う。`H3_LOWVRAM` や `H3_TE_PROJ` など主プロファイル系の変数をひとつでも指定すると手動扱いになり、自動では何も入れない。sage が使えない GPU(sm_120 以外)や未インストールのときは `H3_ATTN_BACKEND=default` も自動で入る。選ばれた級は `/api/status` の `profile` と起動ログで確認できる。

ブラウザで `http://<host>:8611/` を開く。初回起動時はモデルのロード(既定では TE を NF4 量子化して GPU 常駐)に数分かかる。`run.sh` は diffusers-movie-server 統合用の起動スクリプトで、内部ポート 8631 と会話用途の既定値(turbo LoRA 等)を設定したうえで同じサーバを起動する。

VRAM 級別の起動例:

```bash
# 実16GB 単体(sm_89 の 4060 Ti など。sm_120 以外は H3_ATTN_BACKEND=default が必須)
H3_LOWVRAM=group H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_VIDEO_VAE_FP16=1 \
  H3_ATTN_BACKEND=default \
  venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611

# 6GB / 8GB 単体(16GB の指定に TE のストリーミングと削減を足す。さらに下げるなら H3_CACHE=none)
H3_LOWVRAM=group H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_VIDEO_VAE_FP16=1 H3_TE_STREAM=1 H3_TE_DIET=1 \
  H3_ATTN_BACKEND=default \
  venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611

# 8GB×2(TE を 2 枚目 GPU へ)
H3_LOWVRAM=group H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_VIDEO_VAE_FP16=1 \
  H3_ATTN_BACKEND=default H3_TE_DEVICE=cuda:1 \
  venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611

# 32GB 級
H3_LOWVRAM=group H3_TE_PRUNE=1 venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611

# 48GB 級(推奨: 高速化フル。ref2va を使うときは H3_TE_DEVICE を外す)
H3_LOWVRAM=1 H3_TE_PRUNE=1 H3_TE_DEVICE=cuda:1 H3_VIDEO_VAE_FP16=1 H3_KEEP_TRANSFORMER=1 \
  venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611

# 96GB 級(全モデル常駐、env なし)
venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
```

24 / 32GB 単騎でのリアルタイム会話生成(ck-w4a8 量子化 + pinned 退避)の環境変数一式と注意点は [docs/FEATURES.md](docs/FEATURES.md#2026-10-更新-24gb32gb-単騎でリアルタイム会話生成) を参照。

UI より先に動作確認したいときは回帰プローブを使う(`outputs/probe_t2va.mp4` と `outputs/probe_report.json` を出力):

```bash
venv/bin/python scripts/probe_t2va.py
```

### Windows

`run.bat` が同じサーバを起動する(中身は `run.ps1` を実行ポリシーに関係なく呼ぶだけ。既定は 127.0.0.1:8611。`H3_HOST` / `H3_PORT` で待ち受けアドレスとポートを変更):

```powershell
.\run.bat
```

環境変数は起動前に `$env:` で 1 行ずつ設定する。上の Linux の起動例は、`H3_X=値` をそれぞれ `$env:H3_X = "値"` に読み替えて `.\run.ps1` を呼べばそのまま使える。16GB 単体の例(sm_120 以外は `H3_ATTN_BACKEND=default` が必須だが、未指定なら `H3_PROFILE=auto` が自動で入れる):

```powershell
$env:H3_LOWVRAM = "group"
$env:H3_TE_PROJ = "NicoLab28/ClipProj-MiniMax-H3"
$env:H3_VIDEO_VAE_FP16 = "1"
$env:H3_ATTN_BACKEND = "default"
.\run.bat
```

何も設定しなければ `H3_PROFILE=auto` が GPU の総 VRAM から級を選ぶので、まずは `.\run.bat` だけで起動してよい(32GB 以下の級では turbo LoRA も自動で ON。48GB 以上で turbo を既定にしたいときだけ `$env:H3_TURBO_LORA = "1"` を足す)。設定した環境変数はその PowerShell を閉じるまで残る(別の構成で起動し直すときは `Remove-Item Env:H3_LOWVRAM` のように消すか、PowerShell を開き直す)。

`run.ps1` を直接実行して実行ポリシーで止まる場合は次のように起動する(`run.bat` はこれを内部で行う):

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

回帰プローブは次のとおり:

```powershell
.\venv\Scripts\python.exe scripts\probe_t2va.py
```

## 使い方

### UI

ブラウザの単一ページ UI。左が入力と設定、右が結果とギャラリー。右上の「日本語 / English」で表示言語を切り替えられる。

**モードのタブ**

- 上段が動画(**T2VA** = テキスト、**FL2VA** = 先頭 / 末尾フレーム、**Ref2VA** = 参照素材)、下段が静止画(**T2I**、**Ref2I**)。タブを切り替えると、そのモードに必要な入力欄だけが表示される。
- 静止画タブは MiniMax-H3-Image ではなく、動画モデルで 22 フレーム(または 5 フレーム)の超短尺動画を生成して中央フレームを PNG にする。mp4 も一緒に保存される。

**プロンプトと LLM 強化**

- プロンプトは日本語のままでも効く。「プロンプトガイド」を開くと、公式ブリーフ構造(シーン → 被写体 → アクション → カメラ → 音 → 終わり方)や `CUT n [X-Y秒]:` のカット記法、台詞タグ `<d>` の書き方が手書き用チートシートとして見られる。
- 「LLM強化」ボタンは、外部 LLM(無ければ内蔵の Gemma 4 E4B)でプロンプトを整形してプロンプト欄を置き換える。モードは 4 つ: `storyboard`(複数カットのタイムコード形式に展開)、`brief`(公式ブリーフ形式の単一ショット詳細化)、`h3-official`(MiniMax 公式スキル `h3-prompt-writing` の形式に厳密準拠。現在のタブを task として送り、出力言語は英語 / 日本語を選べる。生成前に検証 + 自動修復が走り、未解決の指摘があれば警告が出る)、`translate`(英訳のみ)。「元に戻す」で強化前の文に 1 回戻せる。
- 投影 TE(低 VRAM 級の既定)は台詞タグ `<d>` を扱えないため、`<d>` を含むプロンプトは生成前に警告される(台詞は音声参照で入れる)。

**モードごとの入力**

- FL2VA: 先頭フレーム / 末尾フレーム画像をドラッグ&ドロップ(どちらか一方でも可。両方指定すると先頭 → 末尾の間を生成)。
- Ref2VA / Ref2I: 参照ファイルを複数追加(画像最大 9・動画最大 3・音声最大 3、計 12 まで)。**追加した順序がそのまま参照順**になり、プロンプト内では `<Picture 1>` のようにタイルに表示されるラベルで参照できる。音声参照 1 本だけなら秒数は空欄でよく、音声の長さが尺になる。Ref2I はフレーム数(22 / 5)を選んで中央フレームを PNG にする。
- T2I: フレーム数(22 = 品質検証済みの既定、5 = 実験的)。
- バッチ連続生成(T2I / Ref2I / Ref2VA): チェックを入れるとプロンプト欄が「1 行 = 1 場面」(最大 24 場面)になり、参照・seed・解像度・ステップは全場面で共通。モデルのロードや参照エンコードを場面間で共有するので、低 VRAM モードや参照付きで特に効く。Ref2VA のバッチは秒数が必須。

**サイズ・秒数・ステップ・seed**

- 解像度は 768² / 1344×768(横長)/ 768×1344(縦長)のプリセットのほか、「任意」で幅 × 高さを直接指定(32 の倍数へ丸め、256〜2048。ネイティブ範囲を超えると警告)、「自動」でサーバ既定(Ref2VA は 16:9)。
- 秒数は 5〜15。実際のフレーム数は 17n + 5 に切り上げられ、「実際の尺」として表示される(例: 5 秒 → 124 フレーム 5.17 秒)。
- ステップ数は Turbo LoRA の ON / OFF に応じて推奨値(4 / 30)へ自動で切り替わる。手で変えた値はそのまま使われる。seed は空欄でランダム。

**即反映設定(再ロード不要、リクエストごとに送る)**

- FirstBlockCache(+ しきい値)、Sage Attention(sm_120 のみ)、Turbo LoRA(4 ステップ推奨)、無音で出力(音声トラックを付けない)。
- サーバの構成で使えない組み合わせはチェックが無効化され、理由がツールチップに出る(例: 起動時に turbo を巻いていない group モードでは turbo を選べない)。turbo を入れると FirstBlockCache は自動で外れる。

**生成と結果**

- 「生成」を押すと進捗バーに局面(TE エンコード / デノイズ n/N / デコード)と経過秒が出る。完了すると右側に動画(または PNG と mp4)と、モード・解像度・尺・ステップ・seed・デノイズ / デコード時間・ピーク VRAM・音声の有無・即反映設定・使用したプロンプト全文(折りたたみ)が表示される。
- LLM 強化のメッセージには、外部 LLM と内蔵 LLM のどちらで処理したかが出る。

**再ロードが必要な設定(ヘッダの折りたたみ)**

- transformer 量子化(none / int8。ck-w4a8 は起動時のみ)、text_encoder 量子化、TE レイヤー削除、投影 TE とその量子化、低 VRAM モード(0 / 1 / group)、video VAE の fp16 化。変更すると「適用(再ロード)」が有効になり、見込み時間が表示される。適用はモデルを解放して再ロードするだけで、プロセスは再起動しない。
- 投影 TE が ON のときは text_encoder 量子化と TE レイヤー削除が無効になる(32B TE 自体を読まないため)など、サーバ側の制約に合わせて項目が無効化される。

**サーバ状態**

- 入力欄の下に 4 秒ごとに更新される状態行がある: ロード状態(transformer / transformer_ref / VAE / TE)、GPU の使用量とピーク、ホスト RAM、`profile`(自動プロファイルの選択結果)、lowvram / 量子化 / attention / turbo / TE の構成、ストリーミングの原本の置き場所、LLM の接続先と内蔵 LLM の状態。

**ギャラリー**

- 生成済みの動画と静止画を新しい順に表示。サムネイルをクリックすると拡大表示(ライトボックス)、ダウンロードも可能。
- チェックして「選択を削除」「選択を連結」。連結は**チェックした順**につなぎ(静止画は対象外)、`outputs/concat_<時刻>.mp4` として保存される。

### API

| エンドポイント | 内容 |
|---|---|
| `POST /api/t2va` | テキスト→動画+音声。`prompt`, `resolution`(`768x768` / `768x1344` / `1344x768`), `seconds`(5〜15), `num_inference_steps`, `seed` |
| `POST /api/fl2va` | 上記に加え `image` / `last_image`(どちらか一方以上) |
| `POST /api/ref2va` | 参照付き生成。`references`(画像最大 9・動画最大 3・音声最大 3、順序が意味を持つ)。音声 1 本だけの参照なら `seconds=0` で音声長に合わせる。`still=1` + `frames` で参照付き静止画(ref2i) |
| `POST /api/t2i` | 静止画。`seconds` の代わりに `frames`(既定 22 / 5)。超短尺 mp4 と中央フレーム PNG を保存 |
| `POST /api/t2i_batch` / `POST /api/ref2i_batch` / `POST /api/ref2va_batch` | バッチ生成。`prompts` を場面数ぶん繰り返す(最大 24)。参照系は共通 `references`、`ref2va_batch` は共通 `seconds` が必須 |
| `POST /api/prompt/enhance` | LLM プロンプト強化。`{text, mode, seconds, task, lang}` |
| `GET /api/status` / `GET /api/progress` / `POST /api/interrupt` | ロード状態と VRAM / RAM、生成中の進捗、生成の中断 |
| `GET /api/settings` / `POST /api/settings/apply` | UI から切り替えられる設定の取得と適用(再ロード) |
| `POST /api/admin/unload` / `POST /api/admin/reload` | モデルの解放と再ロード |
| `GET /api/outputs` / `POST /api/outputs/delete` / `POST /api/outputs/concat` | ギャラリーの一覧・削除・連結 |

すべて multipart/form-data。各モードの詳細は [docs/FEATURES.md の生成モード](docs/FEATURES.md#生成モード) を参照。

## 主な環境変数

既定値を示す。多くは **UI からも切り替えられる**ので、恒久的に変えたいときだけ指定する。全変数と実測値は [docs/FEATURES.md](docs/FEATURES.md) を参照。

| 変数 | 既定 | 意味 |
|---|---|---|
| `H3_PROFILE` | `auto` | 主プロファイルの自動選択。`auto` は主プロファイル系の変数が未指定のときだけ総 VRAM から級を選ぶ(32GB 以下の級は turbo LoRA も ON)/ `96gb`〜`8gb` で級を強制(`H3_VRAM_LIMIT_GB` で小さいカードを模擬するときに使う)/ `off` |
| `H3_LOWVRAM` | `0` | `1` = 48GB 級のフェーズ循環 / `group` = 32GB 級以下の block offload(int8 重みをブロック単位でストリーム) |
| `H3_TRANSFORMER_QUANT` | `none` | `int8` で transformer を 66.3→34GB。`ck-w4a8` は ComfyUI の W4A8 カーネル(要 `comfy-kitchen`) |
| `H3_TE_QUANT` | `bnb-4bit` | text_encoder の量子化(`none` は bf16 で 66.7GB) |
| `H3_TE_PRUNE` | `0` | TE の未使用上位レイヤー削除(出力は不変、-3.6GB) |
| `H3_TE_PROJ` | (無効) | **投影 TE**: 32B TE を Qwen3-VL-4B + 線形写像で代替(repo id か .safetensors パス。品質は近似) |
| `H3_TE_PROJ_QUANT` | `bnb-4bit` | 投影 TE の量子化(NF4 で 3.11GB。`none` = bf16 8.88GB / `bnb-8bit`) |
| `H3_TE_DEVICE` | (無効) | TE を 2 枚目 GPU へ常駐(例 `cuda:1`。32B TE は 20GB 級、投影 TE なら 8GB 級) |
| `H3_TE_STREAM` / `H3_TE_DIET` | `0` / `0` | 投影 TE・token_refiner のストリーミングと削減(6GB / 8GB 構成の鍵) |
| `H3_GROUP_STREAM` / `H3_GROUP_STREAM_HOST` | `own` / `auto` | group モードの重み転送方式。`HOST` は搭載 RAM 48GB 未満なら量子化キャッシュのファイルから直接流す(`file`) |
| `H3_VIDEO_VAE_FP16` | `0` | video VAE を fp16 化(デコードのピークを削減。16GB 以下では必須) |
| `H3_KEEP_TRANSFORMER` | `0` | transformer 常駐で再ロード固定費を撤廃 |
| `H3_PRUNED` / `H3_PRUNED_QUANT` | `0` / `int8wo` | ref2va の transformer_ref を AdaLN-pruned 版で読む(`H3_TRANSFORMER_QUANT=int8` 併用必須)と、その量子化方式 |
| `H3_CACHE` / `H3_CACHE_THRESHOLD` | `fbc` / `0.05` | FirstBlockCache(デノイズ -25%。int8+SDPA 軌道では不発の実測あり) |
| `H3_ATTN_BACKEND` | `sage` | sage は sm_120 専用ビルド。**それ以外の GPU は `default`(SDPA)必須** |
| `H3_VAE_DECODER_RESIDENT` | `auto` | file ストリーミングのデコードで、空き VRAM が足りれば VAE デコーダ(4.5GB)をデコードの間だけ GPU に置き切る(読みは 1 回、出力は同じ。A4000 の 8GB 上限で 768² のデコードが 86 → 34 秒)。`1` で強制、`0` で無効。余裕は `H3_VAE_DECODER_RESIDENT_MARGIN_GB`(既定 0.5) |
| `H3_TURBO_LORA` | `0` | lightx2v の 4 / 8 ステップ蒸留 LoRA(初回に約 780MB を DL)。`group` でも使える(自作ストリーマ `H3_GROUP_STREAM=own` + 起動時指定が条件。`file` ストリーミングでは LoRA 約 1.2GB をホスト RAM からブロックごとに流す。`H3_FILE_STREAM_HOST_MIN_MB`(既定 1)以上のファイルに無いテンソルが対象) |
| `H3_ALLOW_WEIGHT_DOWNLOAD` | `0` | bf16 重み(各 ~66GB)がキャッシュに無いとき、`1` のときだけ HF から取得 |
| `H3_LLM_URL` / `H3_LLM_API_KEY` | `http://127.0.0.1:64660` / なし | プロンプト強化に使う外部 LLM と API キー(キーは git 管理外の `config.local.env` からも読む) |
| `H3_LLM_LOCAL_IDLE_S` / `H3_LLM_LOCAL_STOP_ON_GENERATE` | `300` / `auto` | 内蔵 LLM の省メモリ: 最後の利用から 300 秒で llama-server を止める(0 で止めない)/ 生成開始時に止める(`auto` = ホスト RAM 32GB 未満のときだけ、`1` / `0` で強制)。止まっても次の LLM 強化で自動再起動する |
| `H3_LLM_LOCAL` / `H3_LLM_LOCAL_GGUF` | `auto` / なし | 内蔵 LLM(Gemma 4 E4B、CPU、ビルド済み `llama-server` を子プロセスで起動)。`auto` = 外部 LLM に接続できないときだけ使う / `1` = 内蔵のみ / `0` = 使わない。GGUF は `models/llm/` へ自動 DL、既にあるファイルは `H3_LLM_LOCAL_GGUF` で指定。`H3_LLAMA_SERVER`(バイナリのパス)、`H3_LLAMA_CPP_TAG`(取得する llama.cpp のタグ)、`H3_LLM_LOCAL_PORT`(既定 64661)も可 |

## ドキュメント

| 文書 | 内容 |
|---|---|
| [docs/FEATURES.md](docs/FEATURES.md) | **機能と設定の詳細**: 各環境変数の仕組み・成立条件・実測値、VRAM 級ごとの構成、生成モード、UI、LLM 強化 |
| [docs/TECHNICAL_OVERVIEW.md](docs/TECHNICAL_OVERVIEW.md) | **技術概要**: アーキテクチャ、各種方式の統合、VRAM 容量別の扱い、性能、設定リファレンス |
| [docs/RESIDENCY.md](docs/RESIDENCY.md) | **常駐リファレンス**: どのモード・設定のどの局面で GPU に何が載っているか |
| [docs/COMMUNITY_IMPROVEMENTS.md](docs/COMMUNITY_IMPROVEMENTS.md) | ComfyUI コミュニティ等の改良の取り込み判断(取り込んだもの / 見送ったもの / 自前実装したもの) |
| [docs/HISTORY.md](docs/HISTORY.md) | **開発履歴**: 日付付きの作業記録、バグと修正、A/B 実測、積み残し |
| [docs/internal/TECHNICAL_REPORT.md](docs/internal/TECHNICAL_REPORT.md) | 内部資料: 2026-08-04〜08-08 の設計判断の背景と、実機で踏んだ罠 |
| [docs/internal/regression_baselines_dev96.json](docs/internal/regression_baselines_dev96.json) | diffusers を上げるときの回帰基準値(`scripts/run_regression.py` で実行) |

会話アプリとしての実例は [Realtime_Narration_Video](https://github.com/animede/Realtime_Narration_Video) と音声会話クライアント [Realtime_Conversation_Video](https://github.com/animede/Realtime_Conversation_Video)、本番統合先は [diffusers-movie-server](https://github.com/animede/diffusers-movie-server) を参照。

## 構成

```
minimax-h3/
├── app.py                 # FastAPI 本体(port 8611)
├── run.sh                 # diffusers-movie-server 統合用の起動スクリプト(port 8631)
├── run.ps1, run.bat       # Windows 用の起動スクリプト(port 8611。.bat は実行ポリシーを回避して .ps1 を呼ぶ)
├── core/
│   ├── runner.py          # ModularPipeline のロード / 生成ロジック本体
│   ├── blockstream.py     # transformer ブロックの窓付き先読みストリーマ(H3_GROUP_STREAM=own)
│   ├── filestream.py      # 量子化キャッシュのファイルから直接流す(H3_GROUP_STREAM_HOST=file)
│   ├── leanblock.py       # メモリ節約版のブロック forward
│   ├── attnchunk.py       # クエリ分割 / ヘッド分割 attention
│   ├── testream.py        # 投影 TE・token_refiner のストリーミング
│   ├── vaecache.py        # VAE デコーダの fp16 キャッシュ
│   ├── pruned.py          # AdaLN-pruned 版 transformer_ref と量子化キャッシュ
│   ├── adaln_precompute.py, h3_cudagraph.py
│   ├── llm.py, prompt_check.py   # LLM プロンプト強化と検証
│   ├── settings.py, gallery.py
├── static/index.html      # 単一ページ UI(日英)
├── scripts/               # install.sh / install.ps1 / install.bat、ダウンロード・回帰・各種プローブ
├── docs/                  # ドキュメント(上の表)。docs/samples/ にサンプル動画
├── models/                # 量子化済みキャッシュ等(.gitignore 対象)
├── outputs/, logs/        # 生成物・ログ(.gitignore 対象)
└── venv/                  # 専用 venv(.gitignore 対象)
```

## ライセンス

本リポジトリのソースコードの著作権者: © 2026 YUZUKI (animede)。著作権表示と [NOTICE](NOTICE) の内容は、Apache-2.0 第 4 条に基づき再配布時も保持してください。

**このリポジトリのコードは Apache License 2.0**([LICENSE](LICENSE))。モデル重みは含まれない。

使用するモデル / 重みのライセンスは別途それぞれに従うこと:

| 対象 | ライセンス |
|---|---|
| MiniMax-H3 本体の重み | MiniMax Community License(非商用無料、商用は年商 $20M 未満まで、要クレジット) |
| Turbo LoRA(`larryvrh/MiniMax-H3-Turbo-Lora`、`lightx2v/Minimax-h3-Turbo`) | Apache-2.0 |
| Qwen3-VL-32B(text_encoder)/ Qwen3-VL-4B(投影 TE) | Qwen 公式のライセンスに従う |
| diffusers / transformers / torchao / SageAttention 等 | 各パッケージのライセンス |

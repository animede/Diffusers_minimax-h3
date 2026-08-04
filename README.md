# minimax-h3

MiniMax H3 (Hailuo 3.0) の機能確認用スタンドアロンアプリ。動画+ステレオ音声を1回のデノイズで
同時生成するオムニモーダル33Bモデルを、diffusers の Modular Diffusers 経路 (PR #14355) で
動かす。将来 [diffusers-server](../diffusers-server) へ統合するための先行検証ワークスペース
(diffusers-server 本体には一切手を入れていない)。

詳細な調査メモは `../diffusers-server/dev_notes/handoff-minimax-h3.md` を参照。

## 構成

```
minimax-h3/
├── app.py               # FastAPI 本体 (port 8611)
├── core/
│   └── runner.py         # ModularPipeline のロード/生成ロジック本体
├── static/
│   └── index.html        # 単一ページUI (日本語)
├── scripts/
│   ├── download_t2va.py  # T2VA検証に必要なサブフォルダのみをDLするスクリプト
│   └── probe_t2va.py     # UIより先に動作確認する回帰スクリプト
├── outputs/               # 生成物 (.gitignore対象)
├── logs/                  # ダウンロード監視ログ等 (.gitignore対象)
└── venv/                  # 専用venv (.gitignore対象、下記参照)
```

## セットアップ済みの環境

- `venv/` — comfy-env 継承 (torch 2.9.0+cu128 / transformers 5.1.0 / PyAV 16.0.1) +
  PR #14355 版 diffusers 0.40.0.dev0。fastapi/uvicorn/python-multipart も既に揃っている
  (追加インストール不要)。
- ハード: RTX PRO 6000 Blackwell 96GB, RAM 94GB。

## モデルの取得

`MiniMaxAI/MiniMax-H3` は498.6GB(FL2VA/Ref2VAの2チェックポイント + 両方のtransformer分)
だが、T2VA/FL2VA検証には `transformer/`(FL2VA用、66.3GB)+ `text_encoder/`(66.7GB)+
`vae/`(10.4GB)+ `audio_vae/`(0.6GB)+ 設定類の**約144GBのみ**で足りる。
`transformer_ref/`(Ref2VA用、66.3GB)と `Ref2VA/` `FL2VA/` の別パッケージ(各144GB)は
今回不要なので **絶対に丸ごと `snapshot_download` しないこと**。

```bash
venv/bin/python scripts/download_t2va.py
```

内部で `allow_patterns` を使い、必要なサブフォルダのみを取得する。ダウンロード中は
`logs/du_monitor.log` でキャッシュサイズを監視できる(170GB超で警告)。

## VRAM/RAM 設計 (重要、実測に基づく)

このマシンは VRAM 96GB に対し RAM は 94GB。**text_encoder は bf16ネイティブ配布で
実測66.73GB**(fp32配布を仮定した「bf16化で半分の33GB」という当初推定は誤りだった)。
transformer bf16 66.3GB / vae+audio_vae fp32 計11GB と合わせると約144GBになり、
**VRAMにもRAMにも同時に載らない**。diffusers の
`ComponentsManager.enable_auto_cpu_offload()`(全コンポーネントを定常的にRAM常駐させ、
アクティブな1つだけをGPUへ出す方式)は RAM 94GB では成立しないため採用していない。

TEのロード方式は環境変数 `H3_TE_QUANT` で選択する(既定 `bnb-4bit`、2026-08-04 に
A/B検証して既定化)。

### `H3_TE_QUANT=bnb-4bit` (既定)

text_encoder を起動時に NF4(bitsandbytes、compute_dtype=bf16)へ量子化して
**GPU常駐のまま維持する**(bnb 4bitモデルはデバイス間移動不可のため常駐一択)。
実測サイズ **21.0GB**(当初推定の~17-18GBより大きい)。transformer(66.3GB)も常駐し、
リクエストごとの TE⇔transformer 入れ替えが消滅する。

- 定常常駐: transformer + TE-nf4 = **~87.5GB**。これに VAE 11GB を足すと~98.5GBで
  96GBを超えるため、**このモードでは VAE ペアは常駐しない**(CPUに置き、キーフレーム
  エンコード/デコードの当該フェーズだけGPUへ往復する。fp32 11GBのPCIe往復のみで
  ディスクI/Oなし)。
- デコード窓(~9s)だけは transformer を解放してから実行し、直後にリロードする
  (transformer+TE+VAE+デコードバッファは物理的に収まらないため。実測でOOM確認済み)。
  毎stepのスワップではなく単発の片道×2なので、diffusers-server CLAUDE.md 33番の
  禁止パターンには当たらない。
- **品質A/B(同一seed 12345)**: フレーム比較で構図・被写体・シャープさは同等
  (条件付け数値の変化による木立の配置等の微差のみ)、音声も rms 0.0080→0.0061 と
  同水準で -20dB 型の崩壊なし。**劣化なしと判定して既定化した**。

### `H3_TE_QUANT=none` (bf16 TE、旧方式)

**2つの66GBモデルをリクエストごとにGPU上で入れ替える**:

- `vae` + `audio_vae`(fp32 計11GB)は常時GPU常駐。
- エンコード段階: [VAE 11GB + text_encoder 66GB](transformerが常駐していれば先に解放)
- デノイズ/デコード段階: [VAE 11GB + transformer 66GB](エンコード直後にTEを解放)
- 解放は **CUDAモデルの参照を直接落とす**(`.to("cpu")` でRAMへ退避しない。66GBの
  RAM経由はスワップ突入の実測原因になった)。リロードはページキャッシュ/ディスクから
  11〜40秒/モデル。リクエスト間の定常状態は transformer+VAE 常駐(77.5GB)。

**実測のオーバーヘッド**: 1リクエストあたり TEロード ~37s + transformerリロード ~26s。
この入れ替えコストを解消したのが上記の `bnb-4bit`(既定)で、リクエスト合計は
245s → **185s** に短縮された(デノイズ157sは共通のモデル律速で不変)。

**2つの実装上の罠(実機で踏んで修正済み)**:
1. `MiniMaxH3TextEncoderStep.encode_prompt` は素の staticmethod で、`@torch.no_grad()`
   はブロックの `__call__` 側にしか付いていない。直接呼ぶ場合は必ず `torch.no_grad()`
   で包むこと。忘れると autograd グラフが TE の重み約50GB分をGPU上にピン留めし、
   モデルを解放してもVRAMが返ってこない(diffusers-server CLAUDE.md 39番と同型)。
2. ブロックの出力(`num_frames`・`keyframes`・latent形状等)は `PipelineState` に入る。
   `get_block_state()` は宣言された入力しかマップしないので、出力は `state.get(名前)`
   で読むこと。

video VAE の decode は diffusers 側の `MiniMaxH3VideoDecodeStep` が内部で
`torch.autocast(dtype=torch.float16)` を使うため、重み自体は fp32 のままでよい。
**audio VAE は fp32 のまま一切キャストしないこと**(bf16化すると生成音声の音量が
約20dB小さくなる既知の問題があるため、`runner.py` は `vae`/`audio_vae` のロードに
明示的に `dtype=torch.float32` を渡している)。

## 実測値 (RTX PRO 6000 Blackwell 96GB, 768×768, 124フレーム=5.17秒, 30steps)

| 項目 | 実測 |
|---|---|
| DLサイズ(T2VA必要分のみ) | 135GiB (HFキャッシュ実測) |
| text_encoder ロード | 37.6s (コールド) / 15.9s (ページキャッシュ温) |
| transformer ロード | 37.7s (コールド) / 10〜26s (温) |
| vae+audio_vae ロード | 10.0s |
| プロンプトエンコード | 0.7s |
| デノイズ (30steps) | 157〜159s (約5.4s/step, GPU 100%/600W) |
| VAEデコード (video+audio) | 6.5〜9s |
| ピークVRAM (生成中) | none: 83.4GB (デコード時。デノイズ中70.4GB) / bnb-4bit: 91.7GB |
| リクエスト合計 (サーバAPI経由、ロード込み) | none: 245s / **bnb-4bit(既定): 185s** |
| RAM | 使用~6.5GBで安定、スワップ増ゼロ |

bnb-4bit のピーク91.7GBは96GBカードに収まるが余裕は約4GB。ヘッドルームを優先したい
場合は `H3_TE_QUANT=none` で旧方式(ピーク83.4GB、+60s/リクエスト)に戻せる。

## FirstBlockCache によるデノイズ高速化 (`H3_CACHE`、既定 `fbc`)

ComfyUIコミュニティのEasyCache高速化に相当する、diffusers公式の step間キャッシュ
(FirstBlockCache)を `H3_CACHE=fbc`(既定)で有効化している。ステップ間で最初の
transformerブロックの残差変化が小さいとき、残りの計算をスキップする。

- `H3_CACHE_THRESHOLD`(既定 0.05): 実測でデノイズ 157s→118s(-25%、30step中7スキップ)、
  出力はキャッシュ無しと PSNR 31.8〜34.3dB・音声相関0.979 でほぼ同一(目視でも区別困難)。
- threshold 0.1 は 1.92x(デノイズ81.5s、14スキップ)だが構図が目視で分かるレベルで
  ドリフトするため既定にしていない(速度最優先の場合のみ)。
- `H3_CACHE=none` でキャッシュ無しの従来挙動に完全に戻る(バイト一致を回帰確認済み)。
- ピークVRAMは残差キャッシュ分 +0.7GB(91.4→92.1GB)。
- 実装メモ: `MiniMaxH3TransformerBlock` はPRブランチの `TransformerBlockRegistry` に
  未登録のため、runner側で `TransformerBlockMetadata` を登録してから `enable_cache()` を
  呼ぶ(venvのdiffusers本体は無改変)。リクエストごとに `_reset_stateful_cache()` +
  `cache_context("h3")` で包む(リセット漏れは前リクエストの残差による誤スキップを招く)。
  同一seed連続2本のmp4バイト完全一致でリセットの正しさを検証済み。

sm_120 (Blackwell) のパッチ化conv3d病的低速(diffusers-server CLAUDE.md 46番)は
**発症しない**(全ステップ均一に約5.4s、GPU 100%張り付きの健全なcompute-bound)。

音声: 32kHz ステレオ生成 → AAC で mux。プローブ実測 rms=0.10(犬・鳥・風の環境音)。
静かなシーン指定では rms~0.006 程度になるが非無音。

transformers は **venv 内に 5.14.1** を上書きインストールしてある(comfy-env の 5.1.0
には `Qwen3VLProcessor.create_mm_token_type_ids` が無く PR #14355 のエンコーダが
動かないため。comfy-env 自体は無変更)。

## 起動

```bash
venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
```

起動時に transformer と TE(既定では NF4量子化してGPU常駐)をプリロードする
(`H3_TE_QUANT=none` の場合は旧方式: transformer/VAEを常駐させ、TEは各リクエストの
たびにロード/解放)。ブラウザで `http://<host>:8611/` を開く。

## 回帰確認プローブ (UIより先に動作確認する場合)

```bash
venv/bin/python scripts/probe_t2va.py
```

`outputs/probe_t2va.mp4` と `outputs/probe_report.json` (ロード時間・生成時間・
ピークVRAM等)を出力する。

## API

- `POST /api/t2va` (multipart/form-data): `prompt`, `resolution`
  (`768x768`|`768x1344`|`1344x768`), `seconds`(5〜15), `num_inference_steps`, `seed`
- `POST /api/fl2va`: 上記に加え `image` / `last_image`(どちらか一方以上)
- `GET /api/status`: ロード状態・VRAM/RAM
- `GET /api/progress`: 生成中の進捗ポーリング用

## ライセンス注意

MiniMax Community License(非商用無料、商用は年商$20M未満まで、要クレジット)。
本ワークスペース自体はモデル重みを含まない。

## LLMプロンプト強化(2026-08-04追加)

ローカルLLM(gemma4-31B、OpenAI互換 `/v1/chat/completions`)で、入力プロンプトを
H3公式ガイドの形式へ整形する。クラウド版Hailuo AIの内部プロンプト整形層のローカル再現。

- 接続先: 環境変数 `H3_LLM_URL`(既定 `http://127.0.0.1:64650`)。接続不可時は502
  (生成機能には影響しない)
- `POST /api/prompt/enhance` {text, mode, seconds}
- モード(UIの「LLM強化」ボタン+モード選択):
  - `storyboard`(既定): マルチショットのCUTタイムコード形式へ展開(総尺=seconds、2〜3カット、
    ハードカット・被写体同一性維持・カット毎の音指示。焦点距離は35/50/65/100mmに制限)
  - `brief`: 公式ブリーフ形式(シーン→被写体→アクション→カメラ→音→終わり方)の単一ショット詳細化
  - `translate`: 過剰創作なしの英訳(CUT構造は保持)
- 強化結果はプロンプト欄を置き換え(編集可)、「元に戻す」で1世代復元。生成結果には
  使用プロンプト全文を折りたたみ表示(強化あり/なし・モード間の比較評価用)
- UIに手書き用「プロンプトガイド」チートシート(公式ブリーフ構造・CUT記法・実測カット精度±1秒)を同梱
- 実LLM検証済み(gemma4-31B Q4_K_M): 3モードとも形式に従うことを確認(storyboardは
  タイムコード合計・焦点距離制限も遵守、応答11〜17秒)

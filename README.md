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

## VRAM/RAM 設計 (重要)

このマシンは VRAM 96GB に対し RAM は 94GB しかない。4つの大きいコンポーネントを
足すと約110GB (text_encoder bf16 ~33GB + transformer bf16 ~66GB + vae/audio_vae fp32
~11GB) になり、**同時にRAMへ載らない**。そのため diffusers の
`ComponentsManager.enable_auto_cpu_offload()`(全コンポーネントを定常的にRAM常駐させ、
アクティブな1つだけをGPUへ出す方式)は採用していない。

代わりに `core/runner.py` は以下の設計を取る:

- `transformer` + `vae` + `audio_vae` は初回ロード後、**常時GPU常駐**のままにする
  (66+11=77GB、96GB VRAM に十分収まる)。
- `text_encoder`(bf16 ~33GB)だけは生成のたびに GPU へロードし、プロンプトエンコード
  完了後すぐ CPU へ退避 + 明示的に解放する(**単発の片道スワップ**であり、
  diffusers-server の CLAUDE.md が禁止する「巨大モジュールの毎ステップ往復」パターン
  ではない)。

video VAE の decode は diffusers 側の `MiniMaxH3VideoDecodeStep` が内部で
`torch.autocast(dtype=torch.float16)` を使うため、重み自体は fp32 のままでよい。
**audio VAE は fp32 のまま一切キャストしないこと**(bf16化すると生成音声の音量が
約20dB小さくなる既知の問題があるため、`runner.py` は `vae`/`audio_vae` のロードに
明示的に `dtype=torch.float32` を渡している)。

## 起動

```bash
venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
```

起動時に transformer/vae/audio_vae をプリロードする(text_encoderは各リクエストの
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

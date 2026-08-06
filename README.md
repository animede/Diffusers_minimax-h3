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

## 2段生成による2xアップスケール (`/api/t2va` の `upscale=1`、既定OFF)

ComfyUIコミュニティの MiniMaxH3_LatentUpscaler と同系の hires-fix。低解像度(768²)で
前半をデノイズ → **x0推定値**の映像latentだけを bilinear で空間2x → フレッシュノイズを
`scheduler.scale_noise()` で再注入 → 残りの低σステップを1536²で仕上げ → デコード。
`H3_HIRES_DENOISE`(既定0.35)がパス2の担当デノイズ強度。UIはT2VAタブのチェックボックス。

実測(768²→1536²・5秒・30steps・seed=12345、fbc+bnb-4bit):

| | 合計 | デノイズ | デコード | ピークVRAM | 出力 |
|---|---|---|---|---|---|
| upscale=0 | 181s | 125s | 6.5s | 92.1GB | 768² |
| upscale=1 | 645s | 533s (パス1 78s + パス2 455s) | 24.7s | 88.0GB | 1536² |

- 構図・被写体は upscale=0 と一致し、毛並み・芝などの実ディテールが乗る。背景の
  細部(フェンス等)はパス2の再デノイズで軽微にドリフトする(hires-fixの性質)。
- 音声: latentテンソル自体はアップスケール処理で無変更だが、映像と音声は1つの
  パックドシーケンスで自己注意を共有するため、パス2以降の音声出力は upscale=0 と
  bit一致しない(相関0.89、非無音・品質同等。アーキテクチャ上の制約で仕様)。
- VRAM: パス2はシーケンス長~4倍のため、upscale=1 のリクエストではエンコード直後に
  TE-nf4 を解放してからデノイズする(次リクエストのエンコード時に遅延再ロード)。
- **実装の要点(実機でバグを踏んで確定)**: 補間対象は**ノイズ付きlatentではなくx0推定値**
  であること(ノイズ付きを補間すると市松状ノイズが増幅されて全面ノイズ化する。ComfyUI
  参考実装も denoised_output を使っている)。解像度変更時は `build_packed_sequence()` で
  position_ids/token_tags/各indicesを再構築し、`row_timestep_plan` も残ステップ分を
  作り直す。ModularPipeline の `_execution_device` はコンポーネント登録順の先頭モジュール
  で決まるため、TE解放後は `components.transformer.device` を明示的に使う
  (diffusers-server CLAUDE.md 23番・47番と同型の罠)。
## Sage Attention (`H3_ATTN_BACKEND`、既定 `sage`)

sm_120(Blackwell)向けにソースビルドした SageAttention 2.2.0 を既定で使う
(ビルドは `scripts/build_sageattention.sh`、約2分。**必ず `MAX_JOBS=4 NVCC_THREADS=2` +
systemd-run のメモリ上限付きで実行**——無制限並列nvccはホストRAM枯渇でシステム巻き添え
事故歴あり。`CUDA_HOME=/usr/local/cuda-12.8` の明示が必要、既定のcuda-13.0はtorchの
cu128と不一致)。PyPI/コミュニティのLinux向けsm_120 wheelは存在しなかった(全てWindows)。

- 実測: デノイズ 118s→104s(**-12%**)。完全決定論(同一seed 2本バイト一致)。品質は
  目視同等(PSNR 21dBは int8-QK 近似による軌道ドリフトで劣化ではない)
- `H3_ATTN_BACKEND=default` で従来のSDPAに戻る
- FBCと独立に併用可: sage + `H3_CACHE_THRESHOLD=0.1` でデノイズ67s(-43%、リクエスト
  ~125s。FBC 0.1の構図ドリフト特性は既知どおり)
- hub系backend(`flash_hub`/`sage_hub`)は torch 2.9 向けビルドがHub側に存在せず不成立
  (2026-08-05時点。環境の問題ではない)

## ステップ数の指針(蒸留モデル、実測 2026-08-05)

`num_inference_steps` はAPI/UIパラメータ。30(検証既定)に対し **20で-15%、16で-31%** の
デノイズ短縮。16/20とも単フレーム品質・時間方向の安定性に破綻なし(ただし構図は
ステップ数で変わる)。ドラフト用途は16-20、本番は30が目安。ステップを減らすと
FBCのスキップ機会も減る(30stepsで7スキップ→16stepsで0)ため、効果は単純比例しない。

## transformer int8量子化 (`H3_TRANSFORMER_QUANT`、既定 `none`)

`H3_TRANSFORMER_QUANT=int8` で transformer / transformer_ref を torchao
(`Int8WeightOnlyConfig(version=2)`、PR #14355ドキュメントのレシピ、torchao 0.17.0)で
int8化する。66.3GB → 34.0GB。品質は目視同等(PSNR 19dBは軌道分岐であり劣化ではない。
int8同士は同一seedでmp4バイト一致の完全決定論)、デノイズは+5s程度(dequantコスト)。

**int8時は両transformer同時常駐**(34+34+TE-nf4 21=~89GB)になり、ref2va⇔t2vaの
変種切替時の66GB級再ロードが消える(初回のref2vaのみ~36sのコールドロード)。実測:

| | bf16(既定) | int8両常駐 |
|---|---|---|
| t2va | 175-185s / peak 92.1GB | 177-196s / peak 59.7-91.1GB |
| ref2va(2回目以降) | 523s / peak 87.6GB | **463-471s / peak 74.5GB** |
| 変種切替の再ロード | 毎回~26-40s | **なし** |

フェーズ制御(int8両常駐時): t2vaデノイズ前は transformer_ref 常駐時のみTE強制解放
(89GB定常+activationsでOOMするため。実機確認)、ref2vaはTE強制解放不要になり
デコード窓も transformer_ref 常駐のまま通る。`PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True` をrunnerが設定(int8ロード/解放サイクルの断片化で
「54GBしか使っていないのに15GB確保失敗」が実機再現したため。diffusers-serverでも
実績のある設定)。既定 `none` は従来とバイト一致(回帰確認済み)。

## 48GB級VRAM対応 (`H3_LOWVRAM`、既定 `0`)

TE-nf4(21GB)+ transformer int8(34GB)= 55GB は48GB級カードでは同時常駐できない
(96GB機の既定・int8両常駐モードいずれも成立しない)。`H3_LOWVRAM=1` は
「TEとtransformerを絶対に同時常駐させない」フェーズ循環方式で48GB級に対応する。

- **強制**: `H3_TRANSFORMER_QUANT` 未指定なら自動で `int8` に上書きする(bf16
  66.3GBは48GBに単体でも収まらないため)。明示的に `H3_TRANSFORMER_QUANT=none` を
  指定した場合は起動時に `RuntimeError` で拒否する。`H3_TE_QUANT` は `bnb-4bit`
  (既定)以外を指定するとやはり起動時に拒否する。`H3_TRANSFORMER_BOTH_RESIDENT`
  (int8両常駐)は無条件で無効化する。
- **定常状態**: リクエスト間は「何も大きいものが常駐しない」(VAEペアのみCPU常駐、
  他モードのような transformer/TE の常時居座りが無い)。
- **t2va/fl2vaのフェーズ**: [エントリ: 常駐transformer/transformer_refがあれば解放]
  → TEロード → エンコード(+ fl2vaならkeyframeエンコード)→ **layout/latents/
  timestepsをTE常駐のまま先に実行**(`_execution_device`解決の罠対策、下記参照)→
  TE解放 → transformer(int8)ロード → デノイズ(~34GB+活性化~5GB≒39GB)→
  transformer解放 → VAE→GPU → デコード(~11GB+バッファ)→ VAE→CPU
  (**transformerは次リクエストのために再ロードしない** — 次は encode が先に必要)。
- **ref2vaのフェーズ**: 同じ原則で、参照VAEエンコード → layout/latents/timesteps
  (ここもTE常駐のまま)→ TE解放 → transformer_ref(int8)ロード → デノイズ → 以下同様。
- **`_execution_device` 解決の罠(実装時に発見・修正)**: パイプラインのコンポーネント
  順は `text_encoder, tokenizer, processor, vae, scheduler, audio_scheduler,
  transformer, ...`。TEを先に解放してからtransformerをロードする素朴な実装だと、
  `vae`(CPUに退避されていても“存在する”nn.Module)が`text_encoder`の次に解決され、
  `_execution_device`が`cpu`に解決されてしまいデノイズの最初のtransformer forward
  で `RuntimeError: Expected all tensors to be on the same device` になる
  (実機で再現・特定)。対策として **layout/latents/timesteps は TE がまだ
  GPU常駐のうちに実行し、それらの出力テンソルが正しいデバイスに確定してから
  TEを解放する** 順序にした(他モードの `force_free_te` 遅延パターンと同じ発想)。
  ref2vaでは reference_encoder_step も同様にTE常駐のうちに実行する必要がある
  (layout_stepが参照の latent 形状に依存するため)。
- **upscale=1(hires-fix)は非対応**: パス2は約4倍の系列長(~16倍のattention活性化
  コスト)を要し、このモードの定常余裕(~9GB)では未検証のため、`ValueError`
  (400)で拒否する。
- **副次的に見つけた既存バグの修正**: `generate_ref2va()` の
  `_sync_shared_components_to_ref()` はTEロードより**前**に呼ばれていたため、
  TEが未ロードの状態(H3_LOWVRAM、または`none`系モードでref2vaが最初のリクエスト
  になるケース)では `self._pipe_ref.text_encoder` に `None` が同期され、
  `AttributeError: 'NoneType' object has no attribute 'config'` になっていた
  (実機で再現・特定)。TEロードの**後**に呼ぶよう順序を修正した(全モード共通の
  修正、H3_LOWVRAM専用ではない)。
- **正しさの検証**: フェーズ循環は計算内容を変えないはずなので、同一seed
  (768²・5秒・30steps・キツネのプロンプト、seed=12345)で `H3_LOWVRAM=1
  H3_TRANSFORMER_QUANT=int8` のt2va出力と通常int8モードのt2va出力を比較したところ
  **mp4がバイト完全一致**(md5一致)することを確認済み。
- **48GB相当の実機検証**(96GB機でダミーVRAM確保により空きを~43.5GBへ制限、実48GB
  カードの空き~47GBより厳しい条件):

  | | 完走 | ピークVRAM | 内訳(概算) |
  |---|---|---|---|
  | t2va 1本目 | ○ | 38.68GB | TEロード~52s + transformerロード~36s + デノイズ108s + デコード6.6s |
  | t2va 2本目(連続) | ○ | 38.94GB | 同様の固定費が毎回発生(定常状態を持たない設計どおり) |
  | ref2va(画像参照1枚) | ○ | 43.84GB | デノイズ283s(参照行分シーケンスが伸びるため39GB台では収まらずやや高め) |
  | upscale=1 | - | - | 実装時点でOOMリスクを判断し明示的に400で拒否(未実行) |

  いずれもホストRAMスワップの増加なし(`free -g` で作業前後とも Swap used ~6GB台で
  安定)。作業後は `H3_LOWVRAM` 未指定(完全デフォルト設定、bf16 transformer)で
  再起動して同一プロンプトのt2vaを1本実行し、`peak_vram_gb: 91.94GB` /
  `cache_skipped_steps: 6` など既存の実測値(本README上部の表・
  int8量子化セクション)と同水準であることを確認し、回帰なしと判定した。
- **量子化済みチェックポイントの事前保存によるロード時間短縮**: 未調査
  (bnb 4bitは`save_pretrained`直列化に対応している可能性が高いが未検証。torchao
  int8の直列化保存も同様に未検証。今回はリクエストごとの固定費 ~90-100s
  (TEロード+transformerロード)をそのまま許容する設計とした)。

## 任意サイズ・秒数の丸め (2026-08-06)

UIの解像度セレクトに **「任意 (32の倍数へ丸め)」** を追加し、幅/高さを自由に入力できる。
入力値は **エラーにせず H3 の規則へ丸める**:

- **キャンバス**: 32の倍数へ四捨五入し、256〜2048にクランプ(`app.py` の
  `round_canvas_value`)。H3のブロックは32の倍数でないと `ValueError` を出す仕様
  (`MINIMAX_H3_CANVAS_MULTIPLE`)なので、以前は端数を送ると400になっていた。
  ネイティブ範囲(短辺768・最大768×1344)を超える指定はUIが警告を出す(VRAM・品質は未検証)
- **秒数→フレーム数**: 5〜15秒にクランプ後、`17n+5` へ切り上げ(`align_num_frames`)。
  例: 6.3秒 → 158フレーム(6.58秒)。UIは送信前に実フレーム数と実尺をプレビューする
- API: `/api/t2va`・`/api/fl2va` に `height`/`width`(任意、指定時は `resolution`
  プリセットより優先)を追加。`/api/ref2va` は元から受け付けるが、こちらも丸めるように変更
- `/api/status` の `constraints` で丸め規則(canvas_multiple/min/max、fps、frame_step/offset)
  を公開し、UIは同じ規則でプレビューする(丸めの権威はサーバ側)

実測確認: `height=700 width=1000 seconds=6.3` で生成 → レスポンス `704×992 / 158フレーム`、
出力mp4も ffprobe で 992×704・158フレームと一致。

**UI実装の罠**: 幅/高さの `<input type=number>` に `min`/`max`/`step="32"` を付けると、
HTML5の入力検証(stepMismatch/rangeOverflow)が**丸める前の端数入力を不正扱いして
フォーム送信自体をブロック**する(1024×576のような偶然valid な値だけ通り、1000×700は
何も起きない、という分かりにくい症状になる)。丸める前提の欄には制約属性を付けないこと。

## Turbo LoRA (`H3_TURBO_LORA`、既定 `0`、2026-08-06)

`H3_TURBO_LORA=1` で Ostris氏学習中の4/8ステップ蒸留LoRA
(`larryvrh/MiniMax-H3-Turbo-Lora`、Apache 2.0、rank64・259 Linear対象)を適用し、
既定ステップ数を8にする。**プレビュー版LoRA(学習途上)のため既定OFF**。完成版が
出たら再評価する。

実測(768²/5秒/seed12345): 8steps **87.7s(-46%)** で基準30steps(163.5s)に迫る品質、
16steps 98.4sで基準同等、4steps 39.6sは柔らかめだが破綻なし。
**コミュニティの「4〜7stepはダメ」はComfyUI標準サンプラーがデュアルスケジュール
(video shift12/audio shift3)を扱えないことが原因の可能性が高い**——本実装(diffusers
PRのscheduler/audio_scheduler分離+手動ループ)では4stepsでも音声破損は起きなかった。
シフト配線は改修不要(12/3はH3基準スケジューラの既定値で、sigma格子が作者リファレンス
実装とビット一致することを確認済み)。

実装メモ: LoRAキーはComfyUI命名のfused-QKV形式のため、`attn.fuse_projections()` +
ランタイムデルタ(W_eff=W+BA、fuseしない)で適用。**罠: `fuse_projections()` は旧
to_q/k/vを削除せず+12.8GBリークする**(明示deleteで対処)。AdaLNが `linear.weight` を
直接読むためラッパーに weight/bias 等のパススルーが必要。turbo時はFBC自動無効化。

### turbo × 他機能の組み合わせ検証(2026-08-06)

デフォルトのtransformer経路(`transformer_quant=none`, `lowvram=0`)以外との組み合わせ
は当初「未検証のため予防的に拒否」だったが、実測でA/B検証した。

- **turbo × upscale(2xアップスケール hires-fix): 動作確認済み、解禁。**
  768²→1536²・seed12345で8/16stepsとも成功。8steps: 総所要210.3s(denoise 82.6s+
  decode 24.4s)、pass1=5/pass2=2steps(`H3_HIRES_DENOISE=0.35`既定分割)、
  peak VRAM 88.09GB——非turbo30stepsの基準645sから大幅短縮。16steps: 総所要331.8s、
  pass1=10/pass2=5steps、peak VRAM 88.35GB、8stepsより明確にシャープ。
  全フレーム(先頭/中間/末尾)を目視確認し色化け・チェッカーボード崩壊なし、音声も
  RMS/peakが正常値(無音・クリップなし)。turbo時はFBCが自動無効化されるため、
  hires-fixのFBCブックキーピング(`_fbc_last_step_was_skip()`)はtry/exceptで安全に
  no-op化される(懸念だった「turbo未対応のFBC呼び出し」は実害なしと確認)。
- **turbo × transformer int8(`H3_TRANSFORMER_QUANT=int8`、`transformer_both_resident`
  含む): 実測で不可と確定、拒否のまま維持。** `apply_turbo_lora()` の
  `attn.fuse_projections()` が `torch.cat([to_q.weight, to_k.weight, to_v.weight])`
  を実行するが、int8量子化された `to_q`/`to_k`/`to_v`(`H3_INT8_MODULES_TO_NOT_CONVERT`
  はこれらをスキップしない)は torchao の `Int8Tensor` であり、`aten.cat` カーネルが
  未実装のため `NotImplementedError: Int8Tensor dispatch: attempting to run
  unimplemented operator/function: func=<OpOverload(op='aten.cat', overload='default')>`
  で確実に失敗する(HTTP 500、リクエスト単位でクリーンに失敗しVRAMリークなし。
  直後の非turbo生成は正常動作を確認)。
- **turbo × lowvram=1: 実測で不可と確定、拒否のまま維持。** `lowvram=1` は
  `transformer_quant=int8` を強制するため、上記と全く同じ `Int8Tensor`/`aten.cat`
  エラーで失敗(同一エラーメッセージを実機確認)。
- **turbo × lowvram=group: 実測で不可と確定、拒否のまま維持。** 同じ理由
  (`lowvram=group` も `transformer_quant=int8` 前提)で同一エラー。**この失敗は
  group offloadフックの適用順序とは無関係**(sibling project の
  「LoRAをenable_group_offload()より前に適用する」という順序修正パターンは
  ここでは効かない——`fuse_projections()`自体がgroup offloadフックを一切介さず
  `torch.cat`だけで失敗するため、順序を入れ替えても直らないと判断し深追いしなかった)。
- **ref2va: 本タスクの検証対象外のまま**(元々タスクブリーフのスコープ外)。

## 16GB級の検証結果: 非対応(床は~18GB、2026-08-06確定)

16GBバラスト(空き15.5GB)では **TEロード(nf4量子化)の終盤でOOM**(15.37GB使用時点で
+250MiB要求に失敗)。削除済みTE-nf4の常駐17.45GB自体が床であり、video VAE fp16は
デコード段階の対策のためこの床を動かせない。**18GBバラストでは完走**
(peak 17.72GB、total 302s)——つまり現アーキの実質下限は**~18GB**
(24GB級構成 `H3_LOWVRAM=group H3_TE_PRUNE=1` がそのまま18GB級でも動く)。
16GB突破にはTEのストリーミング実行か4bit未満の量子化が必要(未着手の別課題)。

## video VAE の fp16 化 (`H3_VIDEO_VAE_FP16`、既定 `0`)

`H3_VIDEO_VAE_FP16=1` で video VAE の重みだけを fp16 化する(9.70→4.85GB、デコード
ピーク 16.29→~11.4GB)。**audio VAE は絶対に触らない**(bf16化で-20dBの既知問題)。
- 品質: 全124フレームの平均PSNR **39.97dB**(min 39.08)で目視区別不能。デコード計算は
  元々autocast fp16のため重みfp16化の影響が小さい
- **実装の罠**: `AutoencoderKLMiniMaxH3._keep_in_fp32_modules` が encoder/decoder等を
  強制fp32に戻すため、`from_pretrained(dtype=fp16)` は効かない(実機確認)。ロード後に
  `.to(torch.float16)` を明示的に呼ぶ必要がある
- 既定OFFでは既存基準とMD5一致(回帰ゼロ)

## 24〜32GB級VRAM対応 (`H3_LOWVRAM=group`、2026-08-05追加)

`H3_LOWVRAM=1`(48GB級)は transformer(34GB)を毎リクエスト GPU に丸ごとロードするため、
24〜32GB級カードでは transformer 単体でも収まらない。`H3_LOWVRAM=group` は
diffusers-server(姉妹プロジェクト)の CLAUDE.md #33/#34/#37 で確立された
「block-level group offload」パターンをこのプロジェクトに移植したもので、transformer を
**ホストRAMに常駐**させたまま、denoise の各ステップで必要なブロック(50層中1〜2層、
~0.68GB×1〜2)だけを都度 GPU へ出し入れする。transformer は**プロセス起動時に一度だけ
ロードされ、リクエストをまたいで常駐し続ける**(`H3_LOWVRAM=1` のような毎リクエスト
再ロードは発生しない)。

### PR側の「streamed offload時のload-time量子化」調査結果

タスク時点で読んだ `TorchAoHfQuantizer`(`quantizers/torchao/torchao_quantizer.py`)の
`validate_environment()` は、`device_map` に(accelerateの自動割当のような)**辞書**
形式で `"cpu"` という**文字列値**が含まれる場合にのみ `self.offload = True` を立て、
`check_if_quantized_param()` はこのフラグが立っていると CPU 配置のパラメータの量子化を
スキップする(=CPUオフロードするパラメータは意図的に非量子化のまま残す設計)。
一方、本実装が使う `device_map={"transformer": "cpu"}` は `load_components()` を経由して
最終的に `from_pretrained()` に**プレーン文字列** `"cpu"` として渡り、
`modeling_utils.py` の正規化コードにより `{"": torch.device("cpu")}` という
**単一キーの辞書**(値は `torch.device` オブジェクト、文字列ではない)に変換される。
`torch.device("cpu") == "cpu"` は Python 上で `False` になるため、
`"cpu" in device_map.values()` は False のまま保たれ、`self.offload` は立たない。
つまり **CPU上へロードしても量子化はスキップされず、torchaoのInt8Tensorとして
正しく量子化される**(`scripts/probe_group_offload.py` で370/370層がInt8Tensor化
されることを実機確認)。**CPU上でのint8量子化は問題なく可能**という結論。

### 実装の要点

- `_ensure_transformer_group()`(`core/runner.py`)が
  `device_map={"transformer": "cpu"}` + `TorchAoConfig(Int8WeightOnlyConfig)` で
  CPU上に量子化ロードしてから `enable_group_offload(offload_type="block_level",
  num_blocks_per_group=1, use_stream=..., low_cpu_mem_usage=...)` を呼ぶ
  (transformer_refも同型の `_ensure_transformer_ref_group()`)。
- TEは `H3_LOWVRAM=1` と同じくbnb-4bit必須(起動時強制)。t2vaの定常状態では
  TEはリクエストをまたいで常駐する(transformerがそもそも常駐するため、
  TEも常駐させておいた方がリクエストごとの再ロードコストを避けられる)。

### 【重大な発見】`use_stream=True` + `low_cpu_mem_usage=True`(diffusers既定)は
torchao Int8Tensorに対してバグがあり動かない

`scripts/probe_group_offload_forward.py` で実際にforwardを走らせたところ、
`RuntimeError: cannot pin 'torch.cuda.CharTensor' only dense CPU tensors can be
pinned` で denoise の最初のブロックで必ず失敗することを実機確認した。
`hooks/group_offloading.py` の `_pinned_memory_tensors()`(`use_stream=True`なら
`_onload_from_memory()` から毎ステップ無条件で呼ばれる)が
`low_cpu_mem_usage`の値に関わらず `.pin_memory()` を試みるのに対し、
`_init_cpu_param_dict()`(`enable_group_offload()`呼び出し時点で1回だけ実行)は
`low_cpu_mem_usage=True` なら pin をスキップする、という非対称な実装になっており、
両者の想定が食い違っている。torchaoの `Int8Tensor.qdata` はこの食い違いが起きると
壊れた状態(内部的に `torch.cuda.CharTensor` として認識される)でpin_memory()が
呼ばれてクラッシュする。`scripts/probe_group_offload_fix.py` で対照実験した結果:

| 設定 | 結果 | 1ブロックあたりonload/offload |
|---|---|---|
| `use_stream=True, low_cpu_mem_usage=True`(diffusers既定) | **クラッシュ** | - |
| `use_stream=False, low_cpu_mem_usage=True` | 動作OK | onload 0.1-0.26s / offload ~0.22s |
| `use_stream=True, low_cpu_mem_usage=False` | 動作OK | **onload 0.04-0.07s** / offload ~0s |

`low_cpu_mem_usage=False`(`enable_group_offload()`呼び出し時点で全パラメータを
eagerにpin)を新既定に採用した(`H3_GROUP_OFFLOAD_LOW_CPU_MEM`、既定`0`=False)。
理由: onloadが4-5倍速い(pinned memoryはページアウト不可でDMA転送が速いため)。
代償はロード時に追加で~14-16GBのホストRAMをpinする(page-lockedなのでスワップ
不可)ことと、`enable_group_offload()`自体が約22秒かかること(実機測定、
CPU上へのロード70秒 + pin化22秒 = 合計約90秒)。より少ないRAMを優先したい場合は
`H3_GROUP_OFFLOAD_LOW_CPU_MEM=1`(このとき`H3_GROUP_OFFLOAD_USE_STREAM`も
明示指定しない限り自動で`0`にフォールバックする、上記の壊れる組み合わせを
避けるため)を明示指定すればよい。

### choreography最終形(フェーズ×常駐物×ピーク)

| フェーズ | 常駐する大きいもの | 備考 |
|---|---|---|
| 起動時preload | transformer(int8, CPU常駐+groupoffloadフック) | 約90秒(CPUロード70s+pin化22s) |
| t2va encode | TE-nf4(GPU,21GB) + transformer(CPU) | |
| t2va denoise | TE-nf4(GPU,21GB) + transformerの1-2ブロック(GPU,~1.4GB) | |
| t2va decode | vaeペア(GPU,~11GB) + transformerの1-2ブロック | **TEはこの窓だけ強制解放**(下記参照)、decode後に再ロード |
| ref2va参照エンコード | vaeペア(GPU,11GB) | TEはこの窓だけ強制解放(下記参照) |
| ref2va denoise | TE-nf4(GPU,21GB) + transformer_refの1-2ブロック | |
| リクエスト間定常 | transformer(CPU) + TE-nf4(GPU) | ref2va後はtransformer_refが未ロードに戻る(t2va↔ref2va切替のたび再ロード) |

### 【実装中に発見・修正した2つ目のバグ】decode窓・参照エンコード窓でのTE強制解放が必要だった

当初「group offloadされたtransformerのGPU実消費は極小(~1.4GB)だから、decode時に
transformerを解放する必要は無い」と設計したが、32GBダミーVRAM検証で
`CUDA out of memory` を実機再現し、`_log_gpu_tensor_diag()`
(`H3_DEBUG_MEM_DIAG=1`で有効化する一時診断関数、`core/runner.py`に残置)で
実際に生存しているCUDAテンソルを列挙したところ、TE-nf4自身の埋め込みテーブル/
lm_head重み(shape `(151936, 5120)`、bf16、1.556GB×2 = 3.1GB強を含む合計22.25GB)が
デコード直前まで**常駐したまま**だったことが判明した(`empty_cache()`だけでは
解放されない、実際に参照されている生きたテンソルだったため)。つまり
TE-nf4(21GB)+ decode専用バッファ(~16.3GB、下記VAEタイル調査参照)=37GBが
真の必要量で、transformerのフットプリントとは無関係にTEとVAEの競合だった。
対策: **decode窓(と、ref2vaの参照VAEエンコード窓)でTEを強制解放し、窓を抜けたら
再ロードする**(`force_free_te`とは別枠の専用ロジック、`_execution_device`解決順序
はTE解放→vaeをGPUへ、の順で安全性を確保)。

### MD5一致チェックの結果

同一seed(768²・5秒・30steps・キツネのプロンプト、seed=12345)で、通常int8モード
(`H3_LOWVRAM`未指定、`H3_TRANSFORMER_QUANT=int8`、FBC `H3_CACHE=fbc`有効)の
出力と `H3_LOWVRAM=group` の出力を比較したところ、**FBCのキャッシュスキップ判定が
実行経路の違いで異なった**(`cache_skipped_steps`が6→0)ため素朴な比較ではmp4が
不一致だった。FBCは前ステップとの残差の類似度という数値的に鋭敏な判定のため、
数学的に等価な演算でも経路が変わればスキップ判定が変わりうる(劣化ではない)。
両モードとも `H3_CACHE=none` に揃えて再比較したところ、**mp4がバイト完全一致
(md5一致)**することを確認した。group offloadの計算内容は既存経路と数学的に
同一であることの裏付け。

### 計測表: 32GB制限・24GB制限プローブ

**32GB制限**(ダミーVRAM確保で空きを~30GBへ制限、`H3_CACHE=fbc`有効のまま):

| | 完走 | ピークVRAM | 所要時間 |
|---|---|---|---|
| t2va 1本目(768²・5秒・30steps) | ○ | **28.67GB** | denoise 220.79s / decode 6.31s / 総計337.19s(TE初回ロード込み) |
| t2va 2本目(連続) | ○ | **28.23GB** | denoise 220.85s / decode 6.01s / 総計278.83s(TE常駐のため短縮) |
| ref2va(画像参照1枚、768×1344) | **RAM不足で拒否**(下記参照) | - | - |

2本とも1本目と完全に同一mp4(md5一致、`be3f32a84de074990208ad0d30f31a63`)。
ホストRAM/スワップは各フェーズとも増加なし(`free -g`のSwap usedは作業前後とも
~7-8GB台で安定、この値は他プロセス由来の既存分)。

**24GB制限プローブ**(ダミーVRAM確保で空きを~22GBへ制限):

- 768²: denoise中(transformerブロックのonload)でOOM。実消費21.85GB、うち
  TE-nf4だけで21GB。**TE-nf4自体の固定サイズ(21GB)が22GB予算の大半を占め、
  transformerブロック1個分(~148MB)の追加onloadすら入らない**。
- 544×960(RESOLUTION_PRESETSへ一時追加して検証、検証後に削除済み): **同一箇所・
  同一21.85GBでOOM**。解像度を下げても失敗点も消費量も変わらず、**VAEタイル
  縮小と同じく「解像度に依存しない固定コストが律速」であることを確認**
  (下記VAEタイル調査参照)。
- **結論**: 現行アーキテクチャ(TEをbnb-4bit・常時ほぼ常駐という設計)では、
  TE-nf4単体の21GBが24GB級カードの実効予算(~22GB)の大半を占めてしまい、
  どんな解像度でも成立しない。24GB級に対応するには、TEをdenoise中は解放する
  (`H3_LOWVRAM=1`的な設計に戻す)か、TE自体をさらに軽量化する(GGUF等、ただし
  transformers系モデルへのGGUF適用は構造的に困難)必要がある。本タスクの
  スコープ外(48GB→24-32GB対応が目的で24GBは探索的プローブ)のため、
  現時点では未対応と結論する。

### VAE tiling調査結果

`scripts/probe_vae_tile_size.py` で768²・124フレームのVAE decodeを単体で
(denoiseを介さず合成潜在から)直接ベンチマークしたところ、
**tile_sample_min_height/width を256(既定)→192→128→96まで縮小しても
ピークVRAM(16.29GB)が全く変化しなかった**(所要時間はタイル数が増える分
5.9s→10.5sへ悪化)。この結果から、decodeのピークは空間タイルの合成バッファでは
なく、**時間チャンク(`tokens_chunk_size`単位)の1チャンク分をまるごとデコード
するバッファ**か、固定的なVAEアーキテクチャのオーバーヘッドに支配されていると
推測される(VAEクラスは時間方向のチャンクサイズを公開パラメータとして持たない
ため、これ以上の調整はコード変更が必要で本タスクの範囲外)。
**結論: 24-32GB級対応において空間タイルサイズの調整は無意味**(既定のままでよい)。

### FBC/sage共存の確認結果

全てのballast検証(32GB・24GB双方)を通じて `H3_CACHE=fbc`(既定)・
`H3_ATTN_BACKEND=sage`(既定)を有効にしたまま実行し、group offloadのフックと
競合するエラーは一切発生しなかった(FBCはブロック単位の計算スキップ判定、
group offloadはブロックのGPU常駐管理、sageは各ブロック内部のattention実装、と
三者は独立したレイヤーで動作するため)。MD5一致チェック(`H3_CACHE=none`で
実施)とは別に、既定のFBC有効設定での通し実行(32GB制限のt2va 2本)が完走した
ことも確認済み。

### Ref2VAのRAM制約(既知の制限、未解決)

`H3_LOWVRAM=group` でt2va実行後にref2vaを呼ぶと、VRAM予算に関わらず(96GB機で
ダミーVRAM確保無しでも再現)ホストRAMガードで拒否されることを実機確認した:

```
H3_LOWVRAM=group requires at least 40.0GB of available host RAM before loading
the (~34GB, permanently CPU-resident) int8 transformer, but only 33.0GB is
available right now.
```

原因の切り分け: transformer解放直後は`avail_gb`が正しく回復する(44.6GB前後)が、
その後のTE再ロード→参照VAEエンコード→レイアウト計算の過程で`avail_gb`が
33GB前後まで下がる(実機ログで確認)。`swap_used_gb`は一貫して増加しないため
実際のスワップ発生は無い ─ `MemAvailable`(Linuxのbuff/cache込みヒューリスティック
推定値)の変動が、真の空きRAMより保守的に振れている可能性が高い。ただし
`free -g`の`used`ベースで見ても94GB中62GB使用(32GB残)という状況で
追加34GBの確保は本質的にタイトであり、ガード自体が誤りとは断定できなかった
(96GB機でも「t2va transformerを常駐させたままref2va用transformer_refをさらに
CPUへpinしようとする」設計そのものが、94GB RAM機の物理容量に対してすでに
ギリギリ)。**安全側に倒し、ガードを緩めることはせず既知の制限として記録する**
(過去のスワップ暴走事故の教訓を優先)。将来の改善候補: `preload_all()`での
transformer即時ロードをやめてTEと同様に遅延ロード化する(初回リクエストの
レイテンシとのトレードオフ)、またはt2va⇔ref2va切替時によりRAMを消費しない
経路を設計する。**RAM 48GB以上のマシンでは(未検証だがRAM予算に余裕があるため)
この問題は起きない可能性が高い**(本タスクは94GB機でのみ検証、より多いRAM
搭載機での追試は未実施)。

## text_encoder の未使用上位レイヤー削除 (`H3_TE_PRUNE`、既定 `0`、2026-08-06追加)

MiniMax-H3 の text_encoder(Qwen3-VL-32B、64層)は
`hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER]`(=50)しか読まない
(`diffusers/modular_pipelines/minimax_h3/encoders.py`)。`H3_TE_PRUNE=1` は
text_encoder を **51層だけ**(0〜50、`MINIMAX_H3_TEXT_ENCODER_LAYER + 1`)で構築し、
未使用の52〜64層目 + 最終`norm` + `lm_head`(重み換算で14層分、bnb-4bit実測 ~3.6GB /
bf16実測 ~13.6GB)を一度もロードしない。既定 `0` は完全無変更(この分岐自体が
一切呼ばれない)。

### なぜ「50層」ちょうどではなく「51層」なのか(このタスクで発見・検証したtransformers側の罠)

`hidden_states[k]` の意味は transformers の `_can_record_outputs = {"hidden_states":
Qwen3VLTextDecoderLayer}` フック機構(`output_capturing.py`)により決まる:
`hidden_states[0]` = 埋め込み出力(layer 0 の入力を捕捉)、`hidden_states[k]`
(k=1..num_hidden_layers) = `layers[k-1]` の出力。つまり `hidden_states[50]` =
`layers[49]` の出力で、**layers[0..49](50層)を実行すれば十分**なはずだった。
しかし `num_hidden_layers` を**ちょうど50**に切り詰めると、`hidden_states[50]` が
捕捉タプルの**最後の要素**になってしまい、`Qwen3VLTextModel.forward` を包む
`@capture_outputs(tie_last_hidden_states=True)`(既定)が「最後の要素を
`outputs.last_hidden_state`(=最終`norm`適用後の値)で強制的に上書きする」という
挙動を発動させる。実機検証(`scripts/probe_te_prune*.py`)で、50層ちょうどに
切り詰めた場合の `hidden_states[50]` は本来の(64層モデルの)値と**桁違いに
異なる**(max abs diff ~1.5e4、量子化誤差の水準ではなく完全に別の値)ことを確認した。
これはまさに `encoders.py` 自身のガード
(`if num_layers <= MINIMAX_H3_TEXT_ENCODER_LAYER: raise ValueError(...)`)が
警告している「ちょうど50層に切り詰めた最終隠れ状態はpost-normであり、MiniMax-H3が
期待する値ではない」という事態そのもの(このガードのおかげで50層ちょうどの誤設定は
`encode_prompt()` 経由なら例外で弾かれる)。**51層**(`layers[50]`は実行されるが
その出力は読まれない、無駄な1層分の計算コストのみ)にすることで
`hidden_states[50]` が捕捉タプルの中間に位置するようになり、上書きを回避できる。
51層版で64層版の `hidden_states[50]` と**完全一致**(`torch.equal`、bf16・bnb-4bit
nf4とも)することを実機確認済み。

### 実装

`core/runner.py` の `_text_encoder_config_kwargs()` が、text_encoder の
`ComponentSpec`(`pretrained_model_name_or_path="MiniMaxAI/MiniMax-H3"`,
`subfolder="text_encoder"`)と同じ場所から `Qwen3VLConfig` を個別ロードし、
`text_config.num_hidden_layers = 51` に書き換えたオブジェクトを
`load_components(..., config={"text_encoder": pruned_config})` として渡す。
`PreTrainedModel.from_pretrained` は `config` が既に `PreTrainedConfig` インスタンス
なら自前のconfig自動ロードをスキップしてそのまま使う(`modeling_utils.py`で確認)。
チェックポイントの `layers.51-63.*` は `from_pretrained` のロードレポートに
`UNEXPECTED` として現れ、単純に無視される(構築されないため一切のVRAM/RAMを消費
しない)。vision tower(`model.visual`)は無変更(fl2va のキーフレーム/ref2va の
参照画像・動画のpixel_valuesがここを通るため、削除対象から明示的に除外)。

`H3_TE_QUANT`(bnb-4bit/none)・`H3_LOWVRAM`(0/1/group)のどの組み合わせとも合成可能。

### 削除後TEの実測サイズ

| 精度 | 削除前 | 削除後(51層) | 削減 |
|---|---|---|---|
| bnb-4bit nf4 | 21.02GB | **17.45GB** | -3.57GB (-17%) |
| bf16 | 66.71GB | **53.06GB** | -13.65GB (-20%) |

nf4は量子化で1層あたりのサイズがbf16の約1/4に圧縮されるため、削減の絶対量もbf16より
小さい(相対削減率はほぼ同じ)。

### MD5一致チェックの結果

同一seed(768²・5秒・30steps・キツネのプロンプト、seed=12345、`H3_CACHE=none`で
FBCの経路依存を排除)で、`H3_TE_PRUNE=0`(削除なし)と`H3_TE_PRUNE=1`(削除あり)の
出力を比較したところ、**t2va・ref2va(画像参照1枚、vision tower経由)とも
mp4がバイト完全一致(md5一致)**した。削除が数学的に無影響であることの実証。
`H3_LOWVRAM=1`・`H3_LOWVRAM=group`の各モードでも、削除の有無で出力が完全一致する
ことを確認済み(後述)。

### 24GB級対応: 削除だけでは不十分だった(実機で発見・`H3_LOWVRAM_GROUP`側に追加修正)

24〜32GB級対応の既存機構(`H3_LOWVRAM=group`)は、TE-nf4(削除前21GB)を
denoise中も常駐させたままにする設計だった(group offloadされたtransformerの
実消費が~1.4GBと小さいため、32GB級では問題にならなかった)。削除後のTE(17.45GB)は
それでもまだ大きく、**22GB制限で実機OOMを再現した**(denoise開始直後、
21.73GB使用中に224MB要求で失敗)。24GB制限でも同様にOOM(23.12GB使用中に
1.16GB要求で失敗、step 1で発生)。

対策として `H3_LOWVRAM_GROUP` かつ `H3_TE_PRUNE=1` の場合に限り、
`H3_LOWVRAM=1`と同じ「denoiseループの間だけTEを強制解放し、decode窓の前後で
リロードする」選択法を追加した(`core/runner.py`の`group_free_te_for_denoise`
フラグ)。解放位置はlayout_step/latents_step/timesteps_stepの**後**(既存の
`force_free_te`と同じ理由: これらのステップの出力は既にテンソルとして
`state`に載っているため、`_execution_device`解決には以後一切影響しない)。
`H3_TE_PRUNE=0`(既定)の`H3_LOWVRAM_GROUP`は完全無変更(このフラグは
`H3_LOWVRAM_GROUP and H3_TE_PRUNE`の両方が真のときのみ真になる)。

修正後、22GB/24GB/20GBいずれのVRAM制限でも実機で完走を確認した:

| VRAM制限 | 結果 | ピークVRAM(reset後の計測) | 総所要時間 |
|---|---|---|---|
| 22GB(修正前、削除のみ) | **OOM**(denoise開始直後、21.73GB使用中に224MB要求) | - | - |
| 24GB(修正前、削除のみ) | **OOM**(step 1、23.12GB使用中に1.16GB要求) | - | - |
| 24GB(修正後、1本目) | ○ | 17.72GB | 321.7s(TE初回ロード込み) |
| 24GB(修正後、2本目・連続) | ○ | 18.68GB | 277.7s(TE常駐のため短縮) |
| 20GB(修正後) | ○ | 17.72GB | 320.3s |

24GB×2本・20GB×1本の出力mp4は**すべてバイト完全一致**(md5一致、通常int8モード
(`H3_LOWVRAM`未指定)の出力とも一致)。group offloadの計算内容が
VRAM予算に関わらず数学的に同一であることの裏付け(既存の32GB/24GB検証結果と同じ
結論)。ホストRAM/スワップは各テストの前後で増加なし(`free -h`のSwap usedは
一貫して~7.9GB台、既存分のまま)。

### `H3_LOWVRAM=1`(48GB級)でのTEロード時間短縮

削除により、`H3_LOWVRAM=1`が毎リクエスト支払うTEロード固定費が短縮される
(実機、ダミーVRAM確保で空きを~43.5GBに制限):

| | TEロード時間 | TEサイズ |
|---|---|---|
| 削除なし | 42.3s | 21.01GB |
| 削除あり | **35.0s**(-17%) | 17.44GB |

出力mp4は削除の有無でバイト完全一致(md5一致)。

### 回帰確認

`H3_TE_PRUNE`未指定(既定`0`)の状態で同一条件のt2vaを実行し、この機能追加前に
取得していた基準mp4とバイト完全一致(md5一致)することを確認した。既定動作は
完全無変更。

## Ref2VA (オムニ参照生成、`/api/ref2va`)

順序付きの参照素材(**画像最大9・動画最大3・音声最大3、合計12**。音声単独は不可)から
動画+音声を生成する。参照の順序はプロンプト内ラベル(`<Picture i>` 等)とrotary配置に
対応するため意味を持つ。動画参照はサウンドトラックも条件付けに使われる。
**参照が音声ちょうど1本のときは秒数省略可**(音声の長さが生成尺になる。APIでは
`seconds=0`)。出力キャンバスは参照に縛られず、未指定なら16:9(1344×768)。

- **専用チェックポイント `transformer_ref/`(61.7GB、クラス/configは`transformer`と同一で
  重みのみ別)** を使う。t2va/fl2va用transformerとは同時常駐不可のため、runnerは
  変種切替(アクティブな片方だけ常駐、解放→再ロード)で管理する(`/api/status` の
  `active_variant`)。TE-nf4・VAE類・processorは両変種で共有。
- VRAM対策(実機OOM 3件を踏んで確定): 参照VAEエンコード完了後に transformer_ref を
  ロード(逆順は98.5GBでOOM)、デノイズ前にTE-nf4を強制解放(参照行でシーケンスが
  伸びるため。hires-fixと同じパターン)。共有text_encoderの解放は両パイプライン
  シェルの参照を両方消すこと(片方だけではrefcountが残りVRAMが返らない)。
- 実測(768x768指定→1344×768出力・30steps・seed=12345): 画像1枚参照 523s/87.6GB、
  画像+音声(尺は音声由来7.3s) 753s/88.1GB、画像2枚 635s/88.1GB。参照人物の同一性・
  複数参照の合成(人物が参照シーンのカフェに座る)を目視確認済み。ref2va⇔t2vaの
  往復切替も正常(切替込みt2va 188s)。
- UIは「Ref2VA (参照→動画)」タブ(複数ファイル選択、選択順=参照順)。

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

## コミュニティ改良の取り込み一覧

ComfyUI コミュニティ等で出た改良を本アプリ(diffusers 経路)へ取り込んだ作業の記録は
**[docs/COMMUNITY_IMPROVEMENTS.md](docs/COMMUNITY_IMPROVEMENTS.md)** にまとめてある
(取り込んだもの / 調査の結果取り込まなかったもの / 着想を得て自前実装したもの、
それぞれの出典・実測値・判定・踏んだ罠)。

## UIからの設定切替 (2026-08-06)

環境変数でしか変えられなかったオプトイン設定を、性質で2つに分けてUIから操作できる。

### 即反映(生成ボタン直上のチェックボックス、再ロード不要)

FirstBlockCache(+しきい値)・Sage Attention・Turbo LoRA。リクエストごとのパラメータ
(`cache` / `cache_threshold` / `attn` / `turbo`)として送り、`MiniMaxH3Runner.
apply_instant_settings()` が生成ロック取得後・denoise前に常駐transformerへ適用する
(`disable_cache()`/`enable_cache()`、`set_attention_backend()`、`_TurboLoRALinear.enabled`)。
**未指定なら従来どおりプロセス既定**なので既存のcurl/スクリプトは無変更で動く。
turbo有効時はFBCを自動的に無効化する(元の安全規則を踏襲)。

実測(同一seed、再起動なしで切替): FBC on 100.8s(6スキップ)/ off 129.3s(0スキップ)、
Sage on 129.3s / off(native) 158.5s、Turbo on(8steps) 38.9s。

### 再ロードが必要(ヘッダの折りたたみ + 「適用(再ロード)」ボタン)

transformer int8・TE量子化・TEレイヤー削除・低VRAMモード・video VAE fp16。
`POST /api/settings/apply` が `core/settings.py` の `apply_reload_settings()` を呼び、
**プロセスは再起動せず** runner内で全モデルを解放→新設定でロードし直す
(自プロセスをkillすると誰も起動し直せずUIごと復帰不能になるため、
os.execv/self-kill の類は実装しない)。生成中は409、未検証の組み合わせは400。

実測: transformer_quant none→int8→none が 56.0s / 55.0s(GPU 87.5→55.0→87.3GB)、
lowvram 0→1→0 も往復動作。`GET /api/settings` が現在値と選択肢を返し、UIはこれで初期化する。

**UI実装の注意**: チェックボックスOFFは空文字ではなく明示的に `turbo=0` を送ること
(空文字は「未指定=サーバ既定」と解釈されうるため、`H3_TURBO_LORA=1` で起動した
サーバではチェックを外しても無効化されない恐れがある)。turboとupscaleは相互排他で、
片方を選ぶともう片方が自動的に解除・無効化される(サーバ側の400と整合)。

## 生成済み動画ギャラリー (2026-08-06)

結果表示の下に `outputs/` 直下の mp4 をタイル表示する。サムネイルはサーバで生成せず
`<video preload="metadata" src="....mp4#t=0.1">` に先頭フレームを描かせる(依存追加なし。
Ref2VAの参照タイルと同じ手法)。

- `GET /api/outputs`: **直下の *.mp4 のみ**(`outputs/ab_*` 等の検証資料は対象外)。
  尺/解像度は ffprobe で取得し mtime+size をキーにメモリキャッシュ
- `POST /api/outputs/delete`: **パストラバーサル対策**(`/`・`\`・`..` を拒否し、
  resolve 後に `outputs/` 直下であることを検証=シンボリックリンク経由の脱出も遮断)。
  UIは `confirm()` 必須。実機で `../app.py` `/etc/passwd` `ab_*/...` 等が400になることを確認済み
- `POST /api/outputs/concat`: **連結順は「チェックした順」**(表示順は新しい順)。
  全入力のパラメータが一致すれば `concat demuxer + -c copy`(**再エンコードなし=劣化ゼロ**)、
  不一致なら `filter_complex` で先頭動画の解像度へ揃えて再エンコード(無音入力には
  `anullsrc` を合成)。2本未満は400、連結同士の同時実行は409(GPUを使わないので
  generation_lock は取らない)

**依存に関する注意**: このアプリの**生成物のmuxは PyAV**(`av.open()` で libx264+aac を
直接書き出す、`core/runner.py` の `_mux_mp4()`)で行っており、**ffmpeg コマンドは使って
いなかった**。ギャラリーの ffprobe/ffmpeg 呼び出しが**このアプリで初めての外部コマンド
依存**になる(この環境には `/usr/bin/ffmpeg` があり、不在時は `FileNotFoundError` を
捕捉して明示的にエラーを返す)。外部コマンド依存を無くしたい場合は「同一パラメータ限定で
PyAVによるパケット再多重化(=`-c copy` 相当)にし、混在時はエラーにする」のが現実的な代替
(パラメータ混在の再エンコードまでPyAVで実装するのは実質ffmpegの再実装になるため非推奨)。

**UI実装の注意**: 選択のたびに全タイルを作り直すと `<video>` が全数再生成されて
ちらつく(95枚で顕著)。選択状態はバッジ/チェックの**差分更新**にすること
(`updateGallerySelectionUI()`。一覧の再構築は `/api/outputs` 取得時のみ)。

## 今後の外部イベント待ち(積み残し、2026-08-06時点)

### 1. diffusers PR #14355 のマージ待ち — **安易に上げないこと**

本アプリは PR #14355(`Add MiniMax-H3`)のブランチに依存する。2026-08-06時点で
**open / draft / 未マージ**(mergeable_state: unstable)。venvは
`refs/pull/14355/head` の **abc5e9b(2026-08-02)にピン留め**されている
(`pip` の `direct_url.json` で確認可)。

**上げると壊れる可能性が高い**: 8/4〜8/5にPRへ4コミット追加されており、特に
`8ab3662`(Minimax h3 follow up: review & refactor、#14371)と
`99ced1b`(Fix the H3 fast tests against **the refactored state contract**)が入っている。
本アプリの `core/runner.py` は Modular のブロック(`MiniMaxH3SetTimestepsStep`、
`MiniMaxH3LoopDenoiser` 等)を**自前で呼び**、`get_block_state()`/`set_block_state()`・
`state.get(...)`・`row_timestep_plan` といった state 契約に強く依存しているため、
この refactor に追従するには相応の改修が要る。**現構成は全機能をA/B実測済みなので、
マージされるまでは据え置きが安全**。追従する際は「まず t2va の同一seed MD5一致」で
回帰を確認すること(本リポジトリの各機能はこの手法で等価性を検証してある)。

なお int8 レシピ(`TorchAoConfig` + `Int8WeightOnlyConfig`)は abc5e9b に既に含まれて
おり、**マージを待たずに使える**(`H3_TRANSFORMER_QUANT=int8` として実装済み)。

### 2. Turbo LoRA 完成版のリリース待ち

`H3_TURBO_LORA=1` の配線は完成済み(上記セクション参照)。現行LoRAは作者(Ostris氏)が
「デモ/プレビュー、学習途上」と明記しているため既定OFF。**完成版が出たら、
LoRAファイルの差し替えと同一seed A/Bだけで既定化を判断できる**状態にしてある。

### 3. 未着手の改善候補(優先度順、いずれも急ぎではない)

- **量子化済みチェックポイントの事前保存**: `H3_LOWVRAM=1`/`group` はリクエストごとに
  TE/transformerを再ロードするため ~90-100s の固定費がある。bnb-4bit は
  `save_pretrained` で直列化できるはず、torchao int8 は要調査。効けば低VRAMモードの
  体感が大きく変わる
- **16GB級対応**: TEのストリーミング実行(ブロック単位でGPUへ流す)が必要。現状の床は
  TE-nf4削除版の常駐17.45GB(上記セクション参照)
- **torch.compile**: 未検証。FBC/group offload の hook との相性(graph break)確認が要る
- **torchao の C++ カーネル**: int8モードの dequant コスト(+5s)を解消しうるが
  torch>=2.11 が必要で、venv全体のリグレッションリスクが大きい(非推奨)

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

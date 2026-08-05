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

# MiniMax-H3 検証アプリ — 技術概要

**日本語** | [English](TECHNICAL_OVERVIEW.en.md)

## 1. 何をするアプリか

MiniMax H3(Hailuo 3.0)は、動画とステレオ音声を**1回のデノイズで同時に生成する**オムニモーダル33Bモデルである。音声を後段で重ねる従来型のパイプラインとは異なり、映像と音声が同一の packed sequence 上の別々の「行」として、共通の自己注意の中で一緒にデノイズされる。

本アプリはこのモデルを、diffusers の **Modular Diffusers** 経路(PR #14355 で提供される実装)で動かす検証アプリである。MiniMax-H3 の diffusers 対応は本 PR でのみ提供され、ComfyUI 実装とは独立に、diffusers 上での動作を確認するために作られた。将来 [diffusers-server](https://github.com/animede/diffusers-server) へ機能を統合するための先行ワークスペースであり、diffusers-server 本体には一切手を入れていない。

サーバは FastAPI 製で、ポート **8611** で待ち受ける。UI は単一ページ(`static/index.html`)で日英切替に対応する。

### 依存関係

| 依存 | バージョン / 固定先 | 理由 |
|---|---|---|
| diffusers | `f37ab93e621d5ce206c9662e8291ca8b67d9c555`(PR #14355 マージ最終形) | MiniMax-H3 の Modular Pipeline 実装はこの PR にのみ存在する |
| transformers | `5.14.1` 以上 | `Qwen3VLProcessor.create_mm_token_type_ids` が必要(5.1.0 には無い) |
| torch | `2.9.0`(cu128) | CUDA 12.8 系に対応 |
| accelerate / safetensors / huggingface_hub | 通常の最新系 | モデルロード |
| bitsandbytes | `0.49.0` | text_encoder の NF4 量子化(既定経路で必須) |
| torchao | `0.17.0` | transformer の int8 量子化(`0.18` 以降は torch>=2.11 要求のため未採用) |
| av / fastapi / uvicorn | `16.0.1` / `0.104.1` / `0.24.0` | 動画・音声の多重化と Web API |

diffusers は**コミット固定**で運用する。全経路(t2i/t2va/バッチ/ref2va/ref バッチ)を旧ピンとの同一 seed MD5 一致で回帰確認しており、これより先へ進める場合も同じ手順を踏む方針である。

---

## 2. 提供する機能

### 生成モード

| モード | 入力 | 出力 | エンドポイント |
|---|---|---|---|
| T2VA | テキストプロンプト | 動画+ステレオ音声 | `POST /api/t2va` |
| FL2VA | テキスト + 先頭/末尾フレーム画像(どちらか一方以上) | 動画+ステレオ音声 | `POST /api/fl2va` |
| Ref2VA | テキスト + 順序付き参照(画像最大9・動画最大3・音声最大3、計12) | 動画+ステレオ音声 | `POST /api/ref2va` |
| T2I(静止画) | テキストプロンプト | 静止画(PNG)+ 超短尺 mp4 | `POST /api/t2i` |
| Ref2I(参照付き静止画) | テキスト + 参照 | 静止画(PNG) | `POST /api/ref2va`(`still=1`) |

T2I・Ref2I は「超短尺動画を生成し中央フレームを取り出す」ことで画像生成の代用にするモードである。価値は専用 T2I モデルに対する速度ではなく、**H3 と画風が完全に一致する静止画**を FL2VA の先頭フレームや Ref2VA の参照として使えることにある。

### バッチ生成

| エンドポイント | 内容 | 共通化されるもの | 変えられるもの |
|---|---|---|---|
| `POST /api/t2i_batch` | 静止画のバッチ(最大24場面) | frames・resolution・steps・seed | プロンプト(1行=1場面) |
| `POST /api/ref2i_batch` | 参照付き静止画のバッチ | references・frames・resolution・steps | プロンプト(場面ごと) |
| `POST /api/ref2va_batch` | 参照付き動画のバッチ | references・seconds(全場面共通・必須) | プロンプト(場面ごと) |

いずれも `H3_LOWVRAM=1` 環境でのモデルのロード/解放の固定費を、バッチ全体で1回に償却する設計である(詳細は §4)。`H3_LOWVRAM=1` 以外のモード(大モデル常駐)では位相並べ替えの利得がないため、同じ API のまま逐次生成にフォールバックする。

### LLM プロンプト強化

`POST /api/prompt/enhance` がローカル LLM(既定 `H3_LLM_URL=http://127.0.0.1:64650`、gemma4-31B Q4_K_M を想定)を使い、プロンプトを H3 公式スキルの記法(`h3-official`)へ整形する。

- **構造**: T2VA は3フィールド(`integrated_multimodal_description` / `overall_soundscape` / `non_diegetic_music`)、Ref2VA は6フィールド。`[Shot n]` のカット記法、`<d>[言語] 台詞</d>` の話者付き台詞タグを公式仕様どおりに出力する。
- **バリデータ**(`core/prompt_check.py`、規則 F1〜F8): フィールド順序・先頭ショットの時刻無し・カット時刻の厳密増加・尺内判定・ショット尺の下限・`<d>` タグ整合・台詞の尺内収まり・話者ID の8規則を機械判定する。F5(ショット尺下限)・F7(台詞の尺内収まり)は公式仕様に無い、本アプリが実用上追加した規則である。
- **修復ループ**(`core/llm.py` の `enhance_prompt_checked`): 違反を検出したら内容を突きつけて再生成する(最大2回、`H3_OFFICIAL_MAX_REPAIRS`)。違反が増える修復案は破棄する。入力自体が実行不可能(台詞が尺に収まらない等)な場合は LLM に投げる前に判定し、理由付きで拒否する。
- 生成はブロックしない。バリデータの指摘はステータス欄に表示するのみで、最終判断は人間(プロンプト編集)に委ねる。

### hires-fix と turbo

- **2段生成(hires-fix)**: `/api/t2va` の `upscale=1`(既定OFF)。低解像度で前半をデノイズし、映像潜在の x0 推定値のみを空間2倍補間、フレッシュノイズを再注入して高解像度で仕上げる。
- **turbo LoRA**(`H3_TURBO_LORA`、既定OFF): 4/8ステップ蒸留 LoRA を適用し、デノイズの反復回数そのものを減らす。

両者の詳細な数値と成立条件は §4・§6 で扱う。

### UI

単一ページ UI は5タブ・2段組みで構成される(動画系タブ: T2VA / FL2VA / Ref2VA、静止画系タブ: T2I / Ref2I)。各タブはバッチ生成チェック(1行=1場面)を持つ。生成結果は `outputs/` 配下の mp4/PNG をタイル表示するギャラリーに集約され、動画は選択順での無劣化連結(`concat demuxer + -c copy`、パラメータ不一致時のみ再エンコード)に対応する。日英切替を持つ。

---

## 3. アーキテクチャ

### Modular Pipeline ブロックを手動で個別に呼ぶ設計

MiniMax-H3 の diffusers 対応は Modular Diffusers のブロック群として提供される。本アプリは `ModularPipeline` を丸ごと呼ぶのではなく、**個々のブロックを自前のコードから順に呼び出す**設計を採る。

```
MiniMaxH3SetupStep            キャンバス解決・フレーム数の 17n+5 整列・キーフレーム準備
MiniMaxH3TextEncoderStep      プロンプトエンコード
MiniMaxH3PrepareLayoutStep    packed sequence のレイアウト・rotary 位置
MiniMaxH3PrepareLatentsStep   潜在の初期化
MiniMaxH3SetTimestepsStep     video/audio 2系統のシグマ格子
MiniMaxH3DenoiseStep          デノイズループ
MiniMaxH3VideoDecodeStep /
MiniMaxH3AudioDecodeStep      デコード
```

この分解の理由は、**フェーズ(位相)ごとに「今どのモデルを GPU に載せておくか」を制御する必要がある**ためである。パイプラインを丸ごと呼び出す標準的な使い方では、この制御点自体が存在しない。VRAM が全コンポーネント(text_encoder + transformer + VAE 群で約144GB)を同時に載せられない環境では、位相の切れ目でモデルの load/free を差し込めることが設計の前提になる。同じ理由で、hires-fix のようにデノイズループの途中に処理を差し込む改造も、ブロックを自前で駆動していなければ実装できない。

この設計の代償は、`get_block_state()` / `set_block_state()` / `PipelineState` といった diffusers 内部の state 契約に強く依存することである。ブロックの出力(`num_frames`・`keyframes`・latent 形状等)は `PipelineState` に格納され、`get_block_state()` は宣言された入力しかマップしないため、出力は `state.get(名前)` で読む必要がある。

### 位相(フェーズ)の構造

生成1件は次の位相を順に通過する。各位相の境界が、モデルの load/free を差し込む単位になる。

```
setup → encode → layout/latents/timesteps → denoise → after-denoise → decode
```

- **setup**: キャンバスサイズとフレーム数を H3 の規則(32の倍数・`17n+5` フレーム)に整列する。
- **encode**: text_encoder(および FL2VA のキーフレーム、Ref2VA の参照)をエンコードする。
- **layout/latents/timesteps**: packed sequence のレイアウトと rotary 位置、潜在の初期化、video/audio 2系統のシグマ格子を組み立てる。text_encoder に依存する情報がここでまだ必要になる場合があるため、モードによっては text_encoder を常駐させたままこの位相を実行する(§4・§5 参照)。
- **denoise**: transformer(または transformer_ref)によるデノイズループ。VRAM 制約下ではこの位相がピーク VRAM を生む。
- **decode**: video VAE と audio VAE でデコードする。

### 単一 pipe シェルに transformer と transformer_ref の両スロットが載る構造

Ref2VA は専用チェックポイント `transformer_ref/`(クラス・config は `transformer` と同一で重みのみ別)を使う。text_encoder・VAE 群・processor は両変種で共有し、単一のパイプラインシェルが `transformer` と `transformer_ref` の両スロットを持つ。VRAM に余裕がある構成(int8 両常駐、§5 参照)では両方を同時常駐させ、T2VA⇔Ref2VA の切替コストを消す。VRAM が厳しい構成では「アクティブな片方だけを常駐させ、変種切替時に解放→再ロードする」方式に切り替わる(`/api/status` の `active_variant` で現在の常駐変種を確認できる)。

### サーバ構成

- FastAPI 単一プロセス。生成は**同時1件までのグローバルロック**で直列化する(GPU を占有する処理を並行させないため)。
- 長時間かかる生成に対して `GET /api/progress` で進捗をポーリングできる。
- `GET /api/status` がロード状態・VRAM/RAM の実測値を返す。
- 即時反映設定(FirstBlockCache・Sage Attention・Turbo LoRA)はリクエストパラメータとして送ることができ、生成ロック取得後・デノイズ前に適用される。再ロードが必要な設定(量子化方式・低VRAMモード・video VAE精度)は `POST /api/settings/apply` で明示的に切り替える(プロセスは再起動せず、runner 内でモデルを解放して再ロードする)。

---

## 4. 各種方式の統合

### 量子化

| 対象 | 方式 | 効果 |
|---|---|---|
| transformer | torchao `Int8WeightOnlyConfig(version=2)`(`H3_TRANSFORMER_QUANT=int8`) | 66.3GB → **34.0GB** |
| text_encoder | bitsandbytes NF4(`H3_TE_QUANT=bnb-4bit`、既定、compute_dtype=bf16) | 66.71GB → **21.02GB** |
| text_encoder 未使用上位層削除 | `H3_TE_PRUNE=1` | nf4 21.02GB → **17.45GB**(-17%)、bf16 66.71GB → 53.06GB(-20%) |

text_encoder(Qwen3-VL-32B、64層)は `hidden_states[50]` しか実際には読まれない。`H3_TE_PRUNE=1` は 51層(0〜50、`layers[50]` の出力自体は読まれないが計算だけは実行する)で構築し、未使用の52〜64層目・最終 `norm`・`lm_head` を一度もロードしない。**50層ちょうどに切り詰めると誤った値になる**(transformers の `tie_last_hidden_states` 機構が、捕捉タプルの最後の要素を最終 norm 適用後の値で上書きするため)。51層への削減が正しい境界であり、64層版の `hidden_states[50]` と `torch.equal` でビット一致することを確認済み。

int8 量子化・NF4 量子化・層削除のいずれも、削除・量子化の有無で出力 mp4/PNG がバイト完全一致(MD5一致)することを確認しており、数学的に無影響な最適化として扱っている。

### Attention

| 方式 | 環境変数 | 効果 |
|---|---|---|
| Sage Attention 2.2.0(sm_120 向けソースビルド) | `H3_ATTN_BACKEND=sage`(既定) | デノイズ 118s → **104s(-12%)** |
| FirstBlockCache | `H3_CACHE=fbc`(既定)、`H3_CACHE_THRESHOLD=0.05`(既定) | デノイズ 157s → **118s(-25%)**、30step中7スキップ |

FirstBlockCache は、ステップ間で transformer 最初のブロックの残差変化が小さいとき、残りの計算をスキップする diffusers 公式のキャッシュ機構である。threshold を 0.1 まで上げると 1.92倍まで高速化するが、構図が目視で分かる程度にドリフトするため既定にはしていない(opt-in)。品質は PSNR 31.8〜34.3dB・音声相関 0.979 で目視上区別困難と判定している。Sage Attention は完全決定論(同一 seed で2本バイト一致)であり、PSNR 21dB は int8-QK 近似による軌道ドリフトであって劣化ではないと判定している。

両者は独立したレイヤーで動作し併用可能(sage + threshold 0.1 でデノイズ -43%)。

### 蒸留(Turbo LoRA)

`H3_TURBO_LORA`(既定OFF、リクエストの `turbo=1` でも opt-in 可能)は 4/8ステップ蒸留 LoRA を適用し、デノイズの反復回数自体を削減する。既定は **lightx2v** 形式(`lightx2v/Minimax-h3-Turbo`、DMD 蒸留、Apache 2.0、rank128・312 Linear対象、既定4ステップ)。

- **適用係数**(`H3_TURBO_LORA_SCALE`)は **0.094**。LoRA 配布元記載の 0.75 は ComfyUI の alpha 折り込みを前提にした値で、生の B・A に 0.75 をそのまま掛けると 30steps でも完全ノイズ化する。
- **int8 量子化 transformer との併用が可能な理由**: lightx2v 形式のキーは diffusers ネイティブ(to_q/to_k/to_v が分離)であり、適用に `fuse_projections()`(`torch.cat` を要求する)を必要としない。旧世代の comfy 形式(Ostris 版、`qkv_proj` 融合)は `torch.cat` を要求し、int8 量子化された `Int8Tensor` には `aten.cat` カーネルが未実装のため int8/低VRAMモードでは使用不可のまま。適用関数はキー形式を自動判別する。
- **併用制限**: `H3_LOWVRAM=group` とは形式を問わず併用不可(`enable_group_offload` の `cpu_param_dict` が有効化時点で固定されるため)。
- turbo 有効時は FBC を自動的に無効化する。

### オフロード

`H3_LOWVRAM=group`(24-32GB級)は、diffusers の `enable_group_offload(offload_type="block_level", num_blocks_per_group=1, use_stream=...)` を用い、int8 量子化した transformer を**ホスト RAM に常駐**させたまま、denoise の各ステップで必要なブロック(50層中1〜2層、約0.68GB/個)だけを都度 GPU へ出し入れする block-level group offload である。transformer はプロセス起動時に一度だけロードされ、リクエストをまたいで常駐し続ける。

`device_map={"transformer": "cpu"}` で CPU 上にロードした場合でも int8 量子化は正しく適用される(370/370層が `Int8Tensor` 化されることを実機確認済み)。`use_stream=True` + `low_cpu_mem_usage=True` の組み合わせ(API の既定値はどちらも False。省メモリ目的で両方を有効にすると踏む)は torchao の `Int8Tensor` に対して `cannot pin 'torch.cuda.CharTensor'` で確実にクラッシュするバグがあり、`low_cpu_mem_usage=False`(`H3_GROUP_OFFLOAD_LOW_CPU_MEM`、既定0=False)を採用することで回避している。この設定は onload が4〜5倍速くなる副次効果もある(0.04〜0.07s/ブロック 対 0.1〜0.26s/ブロック)。

### 固定費の削減3段

`H3_LOWVRAM=1`(48GB級)は、TE(17.45〜21GB)と transformer-int8(34GB)を同時常駐できないため、毎リクエストで load/free を繰り返す。この固定費を3段階で削減した。

1. **量子化済み text_encoder のディスクキャッシュ**(`H3_TE_PREQUANT`、既定ON): bnb-4bit 量子化後の重みを一度保存し、以降はロードのみで済ませる。TE ロード平均 53.0s → **29.5s**。
2. **TE を別GPUへ常駐**(`H3_TE_DEVICE=cuda:1`): 2枚目GPUに TE を常駐させ、以降のリクエストで TE ロード自体をゼロにする。t2i turbo 4steps の定常時間は平均78.4s → **約35s(-55%)**。
3. **transformer 常駐**(`H3_KEEP_TRANSFORMER=1`): デコード位相でも transformer を解放せず常駐させたままにする。成立条件は3つ(§5 参照)。t2i turbo 4steps は定常 **9.7s/枚** まで短縮。

出力の等価性はいずれの段階でも同一 seed の MD5/PNG 完全一致で確認済み(TE 別GPU化のみ、sm_120 と sm_89 のアーキテクチャ差による丸め誤差でビット不一致になるが、相対RMS差0.084%と軌道ドリフトの水準にとどまる)。

### video VAE の fp16 化

`H3_VIDEO_VAE_FP16=1` は video VAE の重みのみを fp16 化する(9.70GB → 4.85GB、デコードピーク 16.29GB → 約11.4GB)。audio VAE は fp32 のまま一切キャストしない(bf16化すると生成音声の音量が約20dB小さくなる既知の問題があるため)。品質は全124フレーム平均 PSNR **39.97dB**(min 39.08)で目視区別不能。

### 参照バッチの KV プレフィックス共有

`H3_REF_PREFIX_CACHE`(既定1)は、ref バッチ(ref2i_batch / ref2va_batch)のエンコード位相で、参照ラベル+ビジョン(約4,104トークン、約65秒/場面)の Qwen3-VL エンコードが場面ごとに重複していた問題を解消する。ref2va のトークン列は「参照が前置・プロンプトは末尾に verbatim 追記」という構造であり、条件付け元が因果 LM であることから、**参照プレフィックスの表現はプロンプトに依存しない**。プレフィックスを1回だけ `use_cache=True` で通して `DynamicCache` に焼き、場面ごとにはプロンプト末尾(14〜33トークン、約0.2秒)だけをキャッシュ継続する。

プレフィックス部分の `hidden_states[50]` はフル計算と `torch.equal` でビット一致する。プロンプト末尾側には相対RMS約1.5%の丸め差が残るが、位置オフセットを意図的に壊したネガティブコントロールでは相対RMS 27〜30%(20倍)に跳ねることから、これが正しい計算の丸めノイズであり、ロジックバグではないことを確認済み。効果: ref2i バッチのエンコード位相 212.5s → **83.1s**、1枚あたり 164.9s → **116.7s(-29%)**。

### バッチの位相並べ替え

`H3_LOWVRAM=1` の毎リクエスト固定費(TE ロード + transformer ロード、約90〜110秒)を、**位相をリクエスト単位からバッチ単位へ並べ替える**ことでバッチ全体で1回に償却する。

```
entry   : [何も常駐せず]
encode  : [TE-nf4]        全場面の setup/エンコード/layout/latents/timesteps
denoise : [transformer]   全場面を順にデノイズ
decode  : [vae ペア]      全場面をデコード → 保存(場面ごとに保存しながら進む)
```

場面間で共有される可変状態のリセットが実装の要になる。スケジューラは sigmas/timesteps の値が全場面同一(同じ幾何・ステップ数)なので `_step_index = None` に戻すだけでよく(`MiniMaxH3Scheduler.step()` が timestep 値から index を再導出するため)、FirstBlockCache は場面ごとに `_reset_stateful_cache()` + `cache_context` を呼ぶ。逐次生成との mp4/PNG MD5 一致で、位相並べ替えが数学的に無影響であることを実証している。

---

## 5. VRAM容量別の扱い

### 容量から構成を導出する方法

モードは VRAM 容量の関数として導出できる。GPU を替えたら、記憶で表を引くのではなく、次の部品表と不等式から再導出する。

**部品表(すべて実測)**

| 部品 | サイズ |
|---|---|
| text_encoder bf16 | 66.71GB(51層削除で53.06GB) |
| text_encoder nf4 | 21.02GB(51層削除で17.45GB) |
| transformer bf16 | 66.3GB |
| transformer int8 | 34.0GB |
| transformer_ref bf16 / int8 | 61.7GB / 約34GB |
| vae + audio_vae(fp32) | 11.0GB |
| デノイズ活性化 | 約5〜6.6GB(768²・5秒で実測6.6GB) |
| デコードのピーク | 16.29GB(video VAE fp16なら約11.4GB) |
| ref2va の参照エンコード追加分 | TEに対して+3.2GB以上(2048px短辺の vision tower、実測の下限) |
| CUDAコンテキスト等(非PyTorch) | 約1GB |

**満たすべき不等式(局面ごとに独立)**。同時に載る必要があるのは各局面の中だけで、局面をまたいで合計する必要はない。

```
実効予算 = カタログ容量 − 単位差(約0.5GB) − CUDAコンテキスト等(約1GB)

エンコード : TE                                    ≤ 実効予算
デノイズ   : transformer + 活性化(約6.6GB)          ≤ 実効予算
デコード   : デコードピーク(16.29 / fp16なら11.4)   ≤ 実効予算
```

リクエスト間で常駐させたいものがあれば、その分を各局面に足す(例: TE を常駐させたままデノイズしたいなら `TE + transformer + 活性化 ≤ 容量`)。

> **単位の罠**: `nvidia-smi` は MiB、PyTorch の OOM メッセージは GiB、本アプリのログは GB(10進)であり、20GB カードは `nvidia-smi` 表示で21.47GB(10進)だが PyTorch から見える実効容量は約20.99GB(10進)。ここにさらに非PyTorch分約1GBが引かれるため、カタログ容量をそのまま予算にすると約1.5GB過大評価する。

### 容量別の推奨構成表

2026-08-10 更新: 投影TE NF4(常駐3.11GB)とデコード位相の削減(fp16+uint8修正で7.53GB)を
反映した2経路の表。「実測済」以外は導出値。投影TE の制約(`<d>` タグ不可・細部近似・
ref2va vision 未検証)は §4 参照。

| 容量(実効) | 32B TE 経路 | 投影TE(NF4)経路 |
|---|---|---|
| 96GB | bf16 TE+transformer 常駐(実測済) | 不要 |
| 48GB(~49.8) | `H3_LOWVRAM=1` 毎回載せ替え(実測済)。2nd GPU 20GB併用で 9.7s/44.2s(実測済) | 全部同時常駐の見込み 44.7GB(余裕~5.1GB、**未実測**・要ガード改修) |
| 32GB(~30.5) | `group`(nf4 21+ブロック1.4+活性化6.6=29 でぎりぎり) | `group` で余裕(~11.1GB) |
| 24GB(~22.4) | `group`+`H3_TE_PRUNE=1` 必須(実測済) | `group` で余裕(~11.1GB) |
| 16GB(~15.2) | 不可(TE 17.45 が載らない) | **`group` で成立見込み 11.1GB**(**未実測**)。従来「不可」の16GBに道 |

2nd GPU を併用する場合の要件は次節。低VRAM機ほど投影TEの効きが大きい(TE を外に
出せば main は group のブロック+活性化 ~8GB まで下がる)。

### 2枚目GPUに TE を置く場合の要件

`H3_TE_DEVICE` に GPU を指定すると TE はそのGPUに常駐しつづけ、一切解放されない(常駐が目的のため)。TE用GPU側の実効予算に応じて用途が変わる。

| TE | 必要量 | 成立するカード |
|---|---|---|
| 32B pruned nf4 / t2va系 | 17.76GB(実測) | 20GB以上(余裕約1.9GB) |
| 32B pruned nf4 / ref2va | 20.67GB以上(実測。TE 17.45 + 参照エンコード3.22以上) | 24GB以上(20GBは204MB不足でOOM実測) |
| 投影4B bf16 | 8.88GB(実測)+ε | 12GB(薄い)/ 16GB以上 |
| 投影4B NF4 | 3.11GB(実測)+ε | **6〜8GB級で足りる見込み**(導出・未実測) |

ref2va には実効20.7GB以上、つまりカタログ22.2GB以上のGPUが必要と導出される(24GBカードなら実効約22.4GBで余裕約1.7GB、ただし参照2枚以上ではさらに要求が増えるため保証はできない)。

さらに `H3_KEEP_TRANSFORMER=1` を重ねると、デコード位相でも transformer を解放しない構成が成立する。成立条件は3つ全て必須:

1. `H3_LOWVRAM=1`(`group` は対象外)
2. `H3_TE_DEVICE` 設定済み(TEが別GPUに無いと、常駐transformer-int8 34.3GB + TE-nf4 17.45GB = 51.75GBでエンコード位相が先に破綻する)
3. `H3_VIDEO_VAE_FP16=1`(fp32デコードでは transformer 34.3GB + デコードピーク16.29GB = 50.6GBで48GBに入らない。fp16なら45.7GBで入る)

### 推奨起動コマンド(48GB+20GB の現行構成)

```bash
H3_LOWVRAM=1 H3_TE_PRUNE=1 H3_TE_DEVICE=cuda:1 venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
```

さらに固定費をほぼ消したい場合は `H3_VIDEO_VAE_FP16=1 H3_KEEP_TRANSFORMER=1` を追加する。

GPU0(48GB)に transformer 用途を固定し、GPU1(20GB)を TE 常駐用に使う2枚構成が前提。GPU0のみを見せたい場合(TE を別GPUに置かない構成)は `CUDA_VISIBLE_DEVICES=0` を付けて `H3_LOWVRAM=1` のみで起動する。

より詳細な、フェーズごとの常駐物の全パターンは [docs/RESIDENCY.md](RESIDENCY.md) を参照。

---

## 6. 性能

### モード別・構成別の実測

**768×768・5秒(124フレーム)・30steps を基準とする。**

| 構成 | ピークVRAM | t2va 所要 |
|---|---|---|
| 96GB(既定) | 92GB | 約160s |
| 80GB級(`H3_TRANSFORMER_QUANT=int8`) | 59.7GB | 約160s |
| 48GB級(`H3_LOWVRAM=1`) | 38.9GB | 約215s |
| 32GB級(`H3_LOWVRAM=group`) | 28.7GB | 約280s |
| 18GB級(`H3_LOWVRAM=group H3_TE_PRUNE=1`) | 17.7GB | 約280〜320s |

**RTX PRO 5000 48GB + `H3_LOWVRAM=1`(現行構成、2026-08-07以降)での各モード実測**:

| モード | 所要時間 | 備考 |
|---|---|---|
| t2va 単発(品質重視30steps) | 351s | |
| t2va 単発(turbo 4steps) | **143s** | |
| t2i(静止画、turbo 4steps) | **94s** | |
| t2i(turbo + `H3_TE_DEVICE` + `H3_KEEP_TRANSFORMER`) | **9.7s/枚**(定常) | デノイズ4.32s + デコード1.5s |
| t2i_batch(静止画バッチ、3場面) | 67.5s/枚 | 限界コスト約31s/枚 |
| ref2i_batch(参照付き静止画、3場面) | 116.7s/枚 | KVプレフィックス共有込み |
| ref2va_batch(参照付き動画、2場面・5秒) | 401.6s/本 | 限界コスト約330s/本(場面数増で約32%短縮に漸近) |

### 高速化の系譜

| 段階 | リクエスト時間(768²・5秒) |
|---|---|
| 初期(bf16 TE 入れ替え) | 245s |
| + TE bnb-4bit化 | **185s** |
| + FirstBlockCache(0.05) | デノイズ157→**118s** |
| + Sage Attention | デノイズ118→**104s** |
| 現既定(96GB機) | **約160s** |
| + FBC 0.1(opt-in) | 約125s |
| + Turbo LoRA 8steps(opt-in) | **約88s** |
| + Turbo 4steps(ドラフト用途) | 約40s |

48GB機での固定費削減の系譜(t2i turbo 4steps): 157s(GPU交換直後)→ 83.2s(`H3_TE_PREQUANT`)→ 約35s(`H3_TE_DEVICE`)→ **9.7s**(`H3_KEEP_TRANSFORMER`)。t2va 5秒・768²は turboなし30steps 351.4s → turbo 143s → 60.5s → **44.2s**(8.0倍)。

### ピークVRAMの実測値

| 局面 | 内訳 | 実測(48GB機、`H3_LOWVRAM=1 H3_TE_DEVICE=cuda:1`) |
|---|---|---|
| デノイズ(ピーク) | transformer-int8 34.3GB + 活性化約6.6GB | 40.9GB |
| デコード | vaeペア11.3GB + バッファ | (デノイズ後、transformer解放済み) |
| `H3_KEEP_TRANSFORMER=1` 併用時のデコード | transformer 34.03GB常駐 + fp16デコード | 44.15GB(導出予測45.7GBに対し実測44.15GB) |

デノイズとデコードは時間的に重ならない(transformer はデコード直前に必ず解放される、`H3_KEEP_TRANSFORMER=1` を除く)。ピークは通常デノイズ時に出る。

---

## 7. 品質と等価性の担保

### 同一seed MD5一致による回帰確認

数学的に無影響であるべき改造(層の削除・位相の並べ替え・キャッシュのリセット・量子化そのものの決定性)は、同一 seed で生成した出力(mp4/PNG)のバイト完全一致(MD5一致)まで確認する。これにより「たぶん同じ」ではなく「バイト一致」で等価性を示せる。適用例: text_encoder 51層削除、バッチの位相並べ替え、FBC のリセット処理、turbo 本実装とスパイク検証の一致、int8 量子化と bf16 の切替、TE プリロードキャッシュなど。diffusers のバージョンを上げる場合も同じ手順(t2va の同一 seed MD5 一致)で回帰確認する方針を取っている。

### PSNRによる劣化とドリフトの区別

Sage Attention の PSNR は基準比21dB、int8量子化は19dBであるが、いずれも劣化ではなく**軌道のドリフト**として扱う。拡散モデルは初期の微小な計算誤差が以後のステップ全体を分岐させるため、PSNR は「同じ絵か」ではなく「同じ軌道か」を測る指標になる。目視で区別できないこと、同一 seed の2本がバイト一致する(完全決定論)ことを併用して、劣化ではなくドリフトであると判定している。video VAE の fp16 化のように、量子化を伴わない改造では PSNR 39.97dB という高い値そのものを品質指標として扱う。

### 音声の言語検証(ASR)

生成された台詞入り動画の音声について、指定言語で発話されているかを確認する(ASRベースの検証)。h3-official モードの構造適合検証では、台詞タグ `<d>[Japanese] ...</d>` の言語指定が実際の音声出力と対応することを確認対象としている。

### 数値を目視で判断しない方針

VRAM・所要時間・PSNR・MD5・ASR判定など、品質や性能に関する主張はすべて実測値かバイト一致の確認に基づく。目視確認は併用する情報の一つであり、単独では判定根拠にしない。

---

## 8. 設定リファレンス

### 主要な環境変数

| 変数 | 既定値 | 効果 |
|---|---|---|
| `H3_TE_QUANT` | `bnb-4bit` | text_encoderの量子化方式(`none`はbf16で66.7GB) |
| `H3_TE_PRUNE` | `0` | TEの未使用上位レイヤー削除(出力は不変、nf4で-3.6GB) |
| `H3_TE_DEVICE` | (空) | TEを指定GPUに常駐させ、解放しない(例: `cuda:1`) |
| `H3_TE_PREQUANT` | `1` | 量子化済みTE重みのディスクキャッシュ(ロード時間短縮) |
| `H3_TE_PREQUANT_DIR` | `models/prequant` | キャッシュ保存先 |
| `H3_TE_PREQUANT_MIN_FREE_GB` | `25` | この空きディスクを下回ると保存をスキップ(生成は継続) |
| `H3_TRANSFORMER_QUANT` | `none` | `int8`でtransformerを66.3GB→34GBに量子化 |
| `H3_LOWVRAM` | `0` | `1`=48GB級のフェーズ循環 / `group`=24-32GB級のブロック単位オフロード |
| `H3_KEEP_TRANSFORMER` | `0` | `H3_LOWVRAM=1`下でtransformerをデコード位相でも解放しない(3条件必須、§5参照) |
| `H3_VIDEO_VAE_FP16` | `0` | video VAEをfp16化(audio VAEは対象外) |
| `H3_CACHE` | `fbc` | FirstBlockCache有効化(`none`で無効) |
| `H3_CACHE_THRESHOLD` | `0.05` | FBCのキャッシュスキップ判定しきい値 |
| `H3_ATTN_BACKEND` | `sage` | Sage Attention使用(`default`でSDPAへ) |
| `H3_HIRES_DENOISE` | `0.35` | hires-fixパス2のデノイズ強度 |
| `H3_TURBO_LORA` | `0` | 4/8ステップ蒸留LoRAの既定有効化 |
| `H3_TURBO_LORA_REPO` | `lightx2v/Minimax-h3-Turbo` | turbo LoRAの配布元 |
| `H3_TURBO_LORA_FILE` | `minimax_h3_fl2v_turbo_4step_v0.1.safetensors` | turbo LoRAのファイル名 |
| `H3_TURBO_LORA_SCALE` | (形式別の実測既定、lightx2vは0.094) | LoRA適用係数 |
| `H3_GROUP_OFFLOAD_BLOCKS` | `1` | groupオフロード時の同時転送ブロック数 |
| `H3_GROUP_OFFLOAD_USE_STREAM` | `1` | groupオフロードのストリーム転送 |
| `H3_GROUP_OFFLOAD_LOW_CPU_MEM` | `0` | `1`でRAM節約優先(onloadは遅くなる) |
| `H3_GROUP_OFFLOAD_MIN_RAM_GB` | `40` | groupモード起動に必要な空きRAMの下限 |
| `H3_VAE_SMALLCLIP_FIX` | `1` | 超短尺(静止画モード)でのVAEデコード修正 |
| `H3_REF_PREFIX_CACHE` | `1` | 参照バッチのKVプレフィックス共有 |
| `H3_LLM_URL` | `http://127.0.0.1:64650` | プロンプト強化に使うローカルLLM |

このほかにも診断・デバッグ用の環境変数があるが、通常運用で変更するのは上記が中心である。UIから即時反映できる項目(FBC・Sage・Turbo)は、恒久的に変えたい場合のみ環境変数で指定すればよい。

### APIエンドポイント一覧

| パス | 主なパラメータ | 戻り値 |
|---|---|---|
| `GET /` | — | UI(index.html) |
| `GET /api/status` | — | ロード状態・VRAM/RAM実測 |
| `GET /api/progress` | — | 生成中の進捗 |
| `GET /api/settings` | — | 現在の再ロード系設定値と選択肢 |
| `POST /api/settings/apply` | 量子化方式・低VRAMモード等 | モデルの解放・再ロード実行結果 |
| `POST /api/t2va` | `prompt`, `resolution`/`height`+`width`, `seconds`, `num_inference_steps`, `seed`, `upscale` | 動画+音声(mp4) |
| `POST /api/fl2va` | 上記 + `image` / `last_image` | 動画+音声(mp4) |
| `POST /api/t2i` | `prompt`, `frames`(22既定\|5), `resolution`/`height`+`width`, `seed` | 超短尺mp4 + 中央フレームPNG |
| `POST /api/t2i_batch` | `prompts`(最大24) + 共通パラメータ | 場面ごとのPNG/mp4 |
| `POST /api/ref2va` | `prompt`, 参照ファイル群(画像/動画/音声), `seconds`, `still`, `frames` | 動画+音声、または`still=1`でPNG |
| `POST /api/ref2i_batch` | `references` + `prompts`(最大24) | 場面ごとのPNG |
| `POST /api/ref2va_batch` | `references` + `prompts` + `seconds`(必須) | 場面ごとの動画+音声 |
| `POST /api/prompt/enhance` | `prompt`, `mode`, `task`, `lang` | 強化済みプロンプト + `violations`/`warnings`/`check_report` |
| `GET /api/outputs` | — | `outputs/`直下のmp4/PNG一覧 |
| `POST /api/outputs/delete` | ファイル名 | 削除結果(パストラバーサル対策済み) |
| `POST /api/outputs/concat` | 選択ファイル順 | 連結mp4(無劣化 or 再エンコード) |

---

設計判断の背景・実装時に踏んだ罠の詳細は内部資料 [docs/internal/TECHNICAL_REPORT.md](internal/TECHNICAL_REPORT.md) を参照。運用手順・実測値の一次情報は [README.md](../README.md)、VRAM常駐設計の詳細は [docs/RESIDENCY.md](RESIDENCY.md) を参照。

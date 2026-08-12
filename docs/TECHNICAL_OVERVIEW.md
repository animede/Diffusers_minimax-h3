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

いずれも `H3_LOWVRAM=1` 環境でのモデルのロード/解放の固定費を、バッチ全体で1回に償却する設計である(詳細は §4)。位相並べ替えの実装は **`H3_LOWVRAM=1` 専用**で、それ以外のモード(大モデル常駐)では利得がないため、同じ API のまま黙って逐次生成にフォールバックする。

> **2026-08-12 時点での位置づけの変化**: transformer をリクエスト間も常駐させられるようになり、償却すべきロード固定費そのものが消えたため、**`t2i_batch` はもう効かない**(3場面で 0.94倍 = わずかに遅い。常駐化前は 157s → 67.5s の 2.3倍だった)。今も効くのは**参照系バッチ(`ref2i_batch` / `ref2va_batch`)だけ**で、そこで共有されるのはモデルのロードではなく参照のビジョンエンコード(約47s/場面)である(§4・§6)。

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

Ref2VA は専用チェックポイント `transformer_ref/`(クラス・config は `transformer` と同一で重みのみ別)を使う。text_encoder・VAE 群・processor は両変種で共有し、単一のパイプラインシェルが `transformer` と `transformer_ref` の両スロットを持つ。VRAM に余裕がある構成(int8 両常駐、§5 参照)では両方を同時常駐させ、T2VA⇔Ref2VA の切替コストを消す(ただし int8 でも両常駐で **74.3GB** に達するので、48GB 級1枚では選べない。§6 参照)。VRAM が厳しい構成では「アクティブな片方だけを常駐させ、変種切替時に解放→再ロードする」方式に切り替わる(`/api/status` の `active_variant` で現在の常駐変種を確認できる)。

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

### 投影テキストエンコーダ(低VRAM対応の要)

`H3_TE_PROJ`(既定OFF)は、32B の text_encoder を **Qwen3-VL-4B + 学習済み線形写像**に置き換える経路である。4B の `hidden_states[tap=24]`(2560次元)を、学習済みの投影行列 W(2560→5120、[ClipProj-MiniMax-H3](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3))で 32B TE の埋め込み空間(5120次元)へ射影する。統計正規化を挟む線形写像で、token 0 はアテンションシンク(`sink_out`)に固定する。

```
cond = ((h - mean_in) / std_in) @ W * std_out + mean_out     # h = 4B の hidden_states[24]
cond[:, 0] = sink_out                                        # token 0 はアテンションシンク
```

- **NF4 量子化(`H3_TE_PROJ_QUANT=bnb-4bit`、既定)で常駐 3.11GB**。32B nf4 削除版の 17.45GB に対し 1/5 以下で、これが低VRAM対応の床を一気に下げる主因になる(投影行列は bf16 用のものをそのまま fp32 で適用する。配布元の int8 用行列を使うとかえってズレが増える)。
- **品質は「同じ絵」ではなく「同等品質の別の絵」**。32B との PSNR は静止画 22.4dB・動画 14.98dB だが、これは劣化ではなく「同じ指示の別テイク」(構図・色調・時間帯の解釈は一致し、細部の指定は落ちる)。投影は原理的に近似なので、量子化や層削除と違い MD5 では判定できず、PSNR + 目視 + VRAM 実測で判定する。動画では静止画で見えた鮮鋭度低下は出ず、ちらつき(2階差分)はむしろ 11% 少なかった。bf16 vs NF4 の直接 PSNR は 34.45dB で量子化の影響はほぼゼロ。
- **ビジョン/参照経路も目視検証済み**(2026-08-11)。投影行列はテキストのみで校正されているため vision 品質は未知だったが、参照人物の顔・前髪・髪の長さ・服の色・小物まで一貫して反映され、32B 正解と同水準だった。実装は校正の事実の記録として一度だけ `logger.warning` を出す。
- **制約: H3 固有の台詞タグ `<d>`/`</d>`(id≥151669)は 4B の語彙外**。これらを含むプロンプトは黙って別物を送らず `ValueError` で明示的に拒否する。台詞は音声参照(`fully_copy`)側で入れられる。
- **設定 API との関係**: 投影TEは env 専用ではなく、UI の再ロード設定パネル(`/api/settings/apply` の `te_proj`/`te_proj_quant`)からも切替可能。ON のとき 32B TE 向けの `te_quant`/`te_prune` は API 側で 400 拒否する(判定は適用後の値ベース)。往復リロードの E2E で、UI 経路の ON = env 経路の NF4 と PNG MD5 完全一致、OFF = 32B ベースラインと完全一致を確認済み。

`H3_TE_PROJ` は他の全最適化(量子化・常駐制御・turbo など、すべて MD5 一致で無影響を証明できる)とは種類が違い、原理的な近似である点が本質的に異なる。低VRAM対応の設計ストーリー(NF4 3.11GB の常駐 + `H3_LOWVRAM=group` との同居 + fp16 デコード)で 16GB 単体を成立させる中核部品である(§5 参照)。

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
- **参照経路(`transformer_ref`)でも成立**(2026-08-12 に目視確認、従来は未検証)。4ステップでも参照人物の顔・髪型・服の色・小物まで忠実さが保たれ、動画も先頭〜末尾で人物が一貫する。スケールは `transformer` と同じ 0.094 のままでよい。あわせて、参照系3エンドポイント(`/api/ref2va`・`/api/ref2i_batch`・`/api/ref2va_batch`)だけが `num_inference_steps` の既定を 30 にハードコードしており、turbo 時のステップ既定(4)を拾わない噛み合わせ不良を修正した(蒸留 LoRA を付けたまま30ステップ回していた)。

### オフロード

`H3_LOWVRAM=group`(24-32GB級)は、diffusers の `enable_group_offload(offload_type="block_level", num_blocks_per_group=1, use_stream=...)` を用い、int8 量子化した transformer を**ホスト RAM に常駐**させたまま、denoise の各ステップで必要なブロック(50層中1〜2層、約0.68GB/個)だけを都度 GPU へ出し入れする block-level group offload である。transformer はプロセス起動時に一度だけロードされ、リクエストをまたいで常駐し続ける。

`device_map={"transformer": "cpu"}` で CPU 上にロードした場合でも int8 量子化は正しく適用される(370/370層が `Int8Tensor` 化されることを実機確認済み)。`use_stream=True` + `low_cpu_mem_usage=True` の組み合わせ(API の既定値はどちらも False。省メモリ目的で両方を有効にすると踏む)は torchao の `Int8Tensor` に対して `cannot pin 'torch.cuda.CharTensor'` で確実にクラッシュするバグがあり、`low_cpu_mem_usage=False`(`H3_GROUP_OFFLOAD_LOW_CPU_MEM`、既定0=False)を採用することで回避している。この設定は onload が4〜5倍速くなる副次効果もある(0.04〜0.07s/ブロック 対 0.1〜0.26s/ブロック)。

### 固定費の削減3段

`H3_LOWVRAM=1`(48GB級)は、TE(17.45〜21GB)と transformer-int8(34GB)を同時常駐できないため、毎リクエストで load/free を繰り返す。この固定費を3段階で削減した。

1. **量子化済み text_encoder のディスクキャッシュ**(`H3_TE_PREQUANT`、既定ON): bnb-4bit 量子化後の重みを一度保存し、以降はロードのみで済ませる。TE ロード平均 53.0s → **29.5s**。
2. **TE を別GPUへ常駐**(`H3_TE_DEVICE=cuda:1`): 2枚目GPUに TE を常駐させ、以降のリクエストで TE ロード自体をゼロにする。t2i turbo 4steps の定常時間は平均78.4s → **約35s(-55%)**。
3. **transformer 常駐**(`H3_KEEP_TRANSFORMER=1`): デコード位相でも transformer を解放せず常駐させたままにする(成立条件は §5 参照)。**48GB機**(`H3_LOWVRAM=1` + TE を2枚目GPUへ)で t2i turbo 4steps は定常 **9.7s/枚** まで短縮。

出力の等価性はいずれの段階でも同一 seed の MD5/PNG 完全一致で確認済み(TE 別GPU化のみ、sm_120 と sm_89 のアーキテクチャ差による丸め誤差でビット不一致になるが、相対RMS差0.084%と軌道ドリフトの水準にとどまる)。

### デコード窓の解放停止を plain モードへ拡張(2026-08-12)

`H3_KEEP_TRANSFORMER` は当初 `H3_LOWVRAM=1` 専用だったが、実は **plain モード(`H3_LOWVRAM=0`、大モデル常駐)もデコード窓のあいだだけ transformer を解放し、直後に再ロードしていた**(実測 11.9〜12.3s/リクエスト)。この解放は「TE-nf4 21GB + transformer bf16 66.3GB + VAE fp32 11GB = 98.5GB > 96GB」という **32B TE を計算用GPUに同居させる前提**の収支から来たもので、TE が計算用GPUに居ない構成では前提が成立しない(bf16 66.3 + fp16 デコード 11.4 = 77.7GB)。残り2条件がそのまま plain モードの成立条件にもなっているので、**成立条件の1つ目を「`H3_LOWVRAM` が `group` でないこと」へ緩めるだけ**でよかった(解放をスキップする分岐は元から共通で、再ロード側の `_ensure_transformer` は冪等なので追加実装は不要)。

効果は 96GB機の int8 単騎で **19.58s → 7.40s(2.6倍)**。同条件で `H3_KEEP_TRANSFORMER` を 0/1 で切り替えた PNG は **MD5 完全一致**(`596a718e4b5cf9a0b907d2ec479225d2`)であり、解放の停止は数学的に無影響である。構成別の実測は §6。

### video VAE の fp16 化

`H3_VIDEO_VAE_FP16=1` は video VAE の重みのみを fp16 化する(9.70GB → 4.85GB、デコードピーク 16.29GB → 約11.4GB)。audio VAE は fp32 のまま一切キャストしない(bf16化すると生成音声の音量が約20dB小さくなる既知の問題があるため)。品質は全124フレーム平均 PSNR **39.97dB**(min 39.08)で目視区別不能。

> **デコードピークの数値について**: 16.29GB / 約11.4GB はいずれも、後述の「デコード逆正規化の CPU 化」(2026-08-11)より**前**に測った値である。この修正で全長 fp32 のテンソルが GPU 側から消えたため実際のピークはこれより低いはずだが、**修正後の再計測は未実施**。本文中の 16.29 / 11.4GB を使った収支計算(§5 の不等式・`H3_KEEP_TRANSFORMER` の成立条件)は、そのぶん保守側に倒れている。

### 低VRAMゴールの達成と、そのための3件の修正(2026-08-11〜12)

投影TE(NF4 3.11GB)+ `H3_LOWVRAM=group` + fp16 デコードの組み合わせで、**最終ゴール2つを達成した**:

- **ゴールA(単体16GB)**: 実 RTX 4060 Ti 16GB **単体**で t2i/t2va/ref2i/i2va/音声参照/768×1344 が全部動く(ピーク 7.4〜15.2GB、実質上限は 768×1344/5秒)。
- **ゴールB(8GiB×2)**: 計算側 8GiB + TE側 8GiB で t2i/t2va 5秒 768²/ref2i まで(ピーク 6.4〜7.23GB)。動画の参照系は参照トークンの VRAM 加算で不可(§5)。

達成の過程で、低VRAMゴールでしか併用されない組合せに潜んでいた3件を修正した。いずれも設計の一部として恒久適用している:

1. **デコード逆正規化の CPU 化**: 上流 `MiniMaxH3VideoDecodeStep` の末尾は、fp16 の全長デコード結果を GPU 上で一括 fp32 化していた(768²・124フレームで 838MiB の一時テンソル)。runner 側サブクラス `_cpu_norm_video_decode_step()` で **fp16 のまま CPU へ移してから逆正規化**する。要素毎の fp32 演算は縮約が無く CPU/GPU で丸めが一致するため出力はビット同一で、同一 seed PNG MD5 完全一致で実証済み。全経路に無条件適用し、どの構成でもデコード位相のピークを全長 fp32 数本ぶん下げる(8GiB×2 で t2va が入る直接の条件)。
2. **`vae.encode` への fp16 autocast**: `H3_VIDEO_VAE_FP16=1` は VAE 重みを fp16 化するが、上流は**デコードには自前 autocast があり、エンコード側(参照・キーフレーム条件付け)には無い**という非対称があった。このため fp16 VAE × 参照は VRAM 量に関係なく dtype 不整合で即死する。`_load_vae` で `vae.encode` をデコード側と対称の fp16 autocast でラップして解消した。
3. **group 解放時の pinned ホストキャッシュ返却**: group offload は int8 重み ~34GB を pinned host memory に置く。del+gc では torch のホスト側キャッシングアロケータに残り OS に返らない(`empty_cache()` はデバイス側のみ)ため、t2va↔ref2va のモード切替が RAM ガードで恒久的に拒否されていた。解放時に `torch._C._host_emptyCache()`(私有API、getattr ガード付き)を group モードでのみ呼ぶことで、切替が初めて成立した。

罠の詳細な経緯は内部資料の追補 B1〜B11 を参照。

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

**バッチ経路自体のオーバーヘッドはゼロ**で、バッチのステップ時間は単発と完全一致する(ref2i 2.598s vs 単発 2.599s、i2va 7.321s vs 7.323s)。したがってバッチの利得は「共有できる固定費を何回払わずに済むか」だけで決まり、`節約 = 共有できる固定費 ×(場面数-1)/場面数` という単純なモデルが実測とよく合う(ref2i 3場面 実測32.3 vs 予測31.3s/枚、i2va 2場面 実測28.1 vs 予測23.5s/本)。

2026-08-12 時点の制約と位置づけ:

- **`t2i_batch` はもう効かない**(0.94倍)。共有すべきロード固定費が transformer 常駐で消えたため(常駐化前は 157s → 67.5s の 2.3倍だった)。効くのは参照系バッチだけで、そこで共有されるのは参照のビジョンエンコード(約47s/場面)である。
- 位相並べ替えは **`H3_LOWVRAM=1` 専用**(`app.py` の分岐)。それ以外の構成では同じ API のまま黙って逐次生成に落ちる。
- **参照バッチは `H3_TE_DEVICE` と併用不可**: TE用GPUに 24GB 以上を要求するガード(§5)で 400 拒否される(`ValueError` → `HTTPException(400)`)。この閾値は 32B TE の vision 活性化を前提にしたもので、常駐 3.11GB の投影TE には過大である(**要見直し・未修正**)。

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
| デコードのピーク | 16.29GB(video VAE fp16なら約11.4GB)。いずれも逆正規化 CPU 化より前の実測値・再計測未実施(§4) |
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

2026-08-10 更新 / 2026-08-11・08-12 実測反映: 投影TE NF4(常駐3.11GB)とデコード位相の削減
(fp16+uint8修正で7.53GB)を反映した2経路の表。「実測済」以外は導出値。投影TE の制約
(`<d>` タグ不可・細部近似。ref2va vision 経路は 2026-08-11 に目視検証済み)は §4 参照。

| 容量(実効) | 32B TE 経路 | 投影TE(NF4)経路 |
|---|---|---|
| 96GB | bf16 TE+transformer 常駐(実測済) | 32B TE を計算用GPUから外す手段として有効。bf16 transformer + 投影TE + 解放停止が現時点の最速(t2i 6.89〜7.08s、ピーク 74.2〜77.3GB、実測済・§6) |
| 48GB(~49.8) | `H3_LOWVRAM=1` 毎回載せ替え(実測済)。2nd GPU 20GB併用で 9.7s/44.2s(48GB機で実測済) | **全部同時常駐が成立**(ガード改修済み)。int8 transformer + 投影TE + 解放停止で **ピーク 45.6GB**(96GB機での実測。実効予算 ~49.8GB に対し余裕 ~4.2GB)。**48GB 実機での確認は未実施** |
| 32GB(~30.5) | `group`(nf4 21GB+ブロック1.4GB+活性化6.6GB=29GB でぎりぎり) | `group` で余裕(~11.1GB) |
| 24GB(~22.4) | `group`+`H3_TE_PRUNE=1` 必須(実測済) | `group` で余裕(~11.1GB) |
| 16GB(~15.2) | 不可(TE 17.45GB が載らない) | **実測済(2026-08-11)**: 実 RTX 4060 Ti 16GB **単体**で t2i/t2va/ref2i/i2va/音声参照/768×1344 全部完走(ピーク 7.4〜15.2GB)。従来「不可」だった16GBに道が開いた |

2nd GPU を併用する場合の要件は次節。低VRAM機ほど投影TEの効きが大きい(TE を外に
出せば main は group のブロック+活性化 ~8GB まで下がる)。**8GiB×2**(計算側 8GiB + TE側
8GiB)も 2026-08-11 に実測済みで、t2i/t2va 5秒 768²/ref2i まで成立する(ピーク 6.4〜7.23GB)。
動画の参照系(i2va/音声参照)は参照トークンの VRAM 加算(t2va 7.23GB → i2va 9.41GB)で
8GiB 予算を超えるため 16GB 単体でのみ動く。速度と品質の正直な特性は §6 を参照。

### 2枚目GPUに TE を置く場合の要件

`H3_TE_DEVICE` に GPU を指定すると TE はそのGPUに常駐しつづけ、一切解放されない(常駐が目的のため)。TE用GPU側の実効予算に応じて用途が変わる。

| TE | 必要量 | 成立するカード |
|---|---|---|
| 32B pruned nf4 / t2va系 | 17.76GB(実測) | 20GB以上(余裕約1.9GB) |
| 32B pruned nf4 / ref2va | 20.67GB以上(実測。TE 17.45 + 参照エンコード3.22以上) | 24GB以上(20GBは204MB不足でOOM実測) |
| 投影4B bf16 | 8.88GB(実測)+ε | 12GB(薄い)/ 16GB以上 |
| 投影4B NF4 | 3.11GB(実測)+ε | **8GiB級で実測済(2026-08-11)**(8GiB×2 の TE 側で t2i/t2va/ref2i 完走)。6GB級は導出 |

ref2va には実効20.7GB以上、つまりカタログ22.2GB以上のGPUが必要と導出される(24GBカードなら実効約22.4GBで余裕約1.7GB、ただし参照2枚以上ではさらに要求が増えるため保証はできない)。runner はこれをガードとして実装しており、TE用GPUの総容量が **24GB 未満なら ref2va を明示的に拒否**する(`_te_external_usable_for()`)。

> **この 24GB 閾値は 32B TE 前提のまま**である。判定に使っているのは 32B TE の vision 活性化を基準にした値で、常駐 3.11GB の投影TE には過大であり、16GB のカードを TE用GPU にしている構成が弾かれる(参照バッチが `H3_TE_DEVICE` と併用できない原因でもある)。**要見直し・未修正**。

さらに `H3_KEEP_TRANSFORMER=1` を重ねると、デコード位相でも transformer を解放しない構成が成立する。成立条件は3つとも必須で、欠けていれば import 時に `RuntimeError` で明示的に落ちる(条件1は 2026-08-12、条件2は 2026-08-11 に緩和した。一次情報は `core/runner.py` の `H3_KEEP_TRANSFORMER` のガードとその設計コメント):

1. `H3_LOWVRAM` が **`group` でないこと**(`1` でも `0`(plain)でも可)。`group` だけは対象外で、そもそも transformer を CPU 側に常駐させてブロック単位で出し入れする別設計なので無関係である。`1` では毎リクエストの再ロード固定費が、`0`(plain)ではデコード窓ごとの解放/再ロード(11.9〜12.3s)が消える。
2. `H3_TE_DEVICE` **または** `H3_TE_PROJ` のいずれかが設定済み。要するに **32B TE を計算用GPUに同居させないこと**で、同居させるとデコードより先に**エンコード位相**が破綻する(TE-nf4 17.45GB + 常駐 transformer-int8 34.3GB = 51.75GB > 実効予算 ~49.8GB)。投影TE は NF4 で 3.11GB しかないため、3.11 + 34.03 + デノイズ活性化 6.6GB = 43.7GB で同一GPUに同居でき、この条件を単独で満たす。
3. `H3_VIDEO_VAE_FP16=1`(fp32デコードでは transformer 34.3GB + デコードピーク16.29GB = 50.6GBで48GBに入らない。fp16なら45.7GBで入る)

### 推奨起動コマンド

**48GB+20GB の2枚構成(32B TE 経路)**:

```bash
H3_LOWVRAM=1 H3_TE_PRUNE=1 H3_TE_DEVICE=cuda:1 venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
```

さらに固定費をほぼ消したい場合は `H3_VIDEO_VAE_FP16=1 H3_KEEP_TRANSFORMER=1` を追加する(48GB機で t2i 定常 9.7s/枚・t2va 5秒 44.2s)。GPU0(48GB)に transformer 用途を固定し、GPU1(20GB)を TE 常駐用に使う2枚構成が前提。GPU0のみを見せたい場合(TE を別GPUに置かない構成)は `CUDA_VISIBLE_DEVICES=0` を付けて `H3_LOWVRAM=1` のみで起動する。**ref2va を使うときは `H3_TE_DEVICE` を外す**(TE用GPUが20GBだと上記のガードで拒否される)。

**GPU 1枚・投影TE 経路(2026-08-12 の最速級。96GB機で実測、ピーク 45.6GB)**:

```bash
H3_TRANSFORMER_QUANT=int8 H3_KEEP_TRANSFORMER=1 H3_VIDEO_VAE_FP16=1 \
  H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_TURBO_LORA=1 \
  venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
```

int8 transformer と投影TE(NF4)を1枚のGPUに同時常駐させ、デコード窓の解放も止める構成。ピーク 45.6GB は 48GB級の実効予算 ~49.8GB に収まるが、**48GB 実機での確認は未実施**(96GB機で測ったピークが48GB予算に収まることの確認まで)。なお実測時は2枚挿しの 96GB機で 1枚だけを見せるために `CUDA_VISIBLE_DEVICES=0` を前置している(GPUが元々1枚の機体では不要)。

**16GB 単体 / 8GiB×2(投影TE + group offload)**: 起動例は README の「VRAM級別クイックスタート」を参照。`H3_LOWVRAM=group H3_TE_PROJ=… H3_VIDEO_VAE_FP16=1` が中核で、sm_120 以外では `H3_ATTN_BACKEND=default` が必須(既定の sage は sm_120 専用ビルド)。

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

この表は 2026-08-12 より前の基準値である。解放停止後の 96GB機・int8 単騎なら、同じ 30steps でも t2va は 155.0s(FBCなし)/ 121.5s(FBCあり)まで下がる(後述の「モード別の到達点」)。

> **どちらの箱の値か**: 以下、**96GB機** = RTX PRO 6000 Blackwell 96GB + 増設 RTX 4060 Ti 16GB、**48GB機** = RTX PRO 5000 Blackwell 48GB + RTX 4000 SFF Ada 20GB。同じ数値でも箱が違えば比較できないので、表ごとに明記する。

**48GB機・`H3_LOWVRAM=1`(2026-08-07以降)での各モード実測**:

| モード | 所要時間 | 備考 |
|---|---|---|
| t2va 単発(品質重視30steps) | 351s | |
| t2va 単発(turbo 4steps) | **143s** | |
| t2i(静止画、turbo 4steps) | **94s** | |
| t2i(turbo + `H3_TE_DEVICE` + `H3_KEEP_TRANSFORMER`) | **9.7s/枚**(定常) | デノイズ4.32s + デコード1.5s |
| t2i_batch(静止画バッチ、3場面) | 67.5s/枚 | 限界コスト約31s/枚 |
| ref2i_batch(参照付き静止画、3場面) | 116.7s/枚 | KVプレフィックス共有込み |
| ref2va_batch(参照付き動画、2場面・5秒) | 401.6s/本 | 限界コスト約330s/本(場面数増で約32%短縮に漸近) |

### 解放停止後の最速構成の比較(96GB機、2026-08-12)

**turbo 4steps・768²・定常値**(サーバ起動後の2回目以降。t2va は 5秒 = 124フレーム)。TE はすべて投影TE(NF4、3.11GB)。

| 構成 | t2i | t2va 5秒 | ピーク | GPU |
|---|---|---|---|---|
| bf16 + TE@GPU1 + 解放停止 | **6.89s** | **26.8s** | 74.2GB + 3.2GB | 2枚 |
| bf16 単騎(TE も GPU0)+ 解放停止 | 7.08s | 27.04s | 77.3GB | 1枚 |
| **int8 単騎 + 解放停止(実用最適)** | **7.40s** | **28.13s** | **45.6GB** | 1枚 |
| int8 + `H3_LOWVRAM=1` + KEEP + TE@GPU1 | 7.65s | 28.56s | 42.5GB | 2枚 |
| int8 単騎・解放あり(`H3_KEEP_TRANSFORMER=0`) | 19.58s | — | 39.8GB | 1枚 |
| bf16 + TE@GPU1・解放あり | 19.9s | 40.0s | 68.9GB | 2枚 |

- bf16 のデノイズは int8 より速い(t2i 2.05〜2.07s vs 2.39〜2.40s、t2va 14.05s vs 14.81s)が、**77GB 級のカードが要る**。int8 単騎との差は 7% 程度なので、どちらを取るかは持っているカード次第である。
- **TE の置き場所はもう効かない**: 単騎 7.08s と 2枚 6.89s の差は 2.7%。「TE の置き場所で t2i が2倍違う」(15.23s vs 7.65s)は**解放が残っていた時代の現象**で、解放を止めた後は1枚で足りる。
- 最速構成のピークが 45.6GB であることが示すとおり、速さの源は容量ではなく**固定費ゼロ運用**の側にある。

### モード別の到達点(96GB機・int8 単騎・解放停止、2026-08-12)

**turbo 4steps・768²・定常値**。動画は5秒=124フレーム、静止画は22フレーム。

| モード | 単発 | 連番なら1件あたり | デノイズ | ピーク |
|---|---|---|---|---|
| t2i 768² | 7.40s | 7.9s(0.94倍 = 効果なし) | 2.40s | 45.0GB |
| t2va 5秒 768² | 28.13s | (バッチAPIなし) | 14.94s | 45.6GB |
| ref2i 768²(参照付き静止画) | 79.3s | 47.0s(1.69倍) | 7.8s | 45.4GB |
| i2va 5秒 768²(画像参照→動画) | 103.1s | 75.0s(1.37倍) | 22.0s | 45.9GB |

turbo なし30ステップでは t2i 27.42s / t2va 155.0s(FBCなし)、21.41s / 121.5s(FBCあり)、ref2i 148.4s、i2va 290.3s。

- **初回コスト**: 起動時に transformer + VAE を常駐させるのに約50秒(プロセスに1回だけ)。参照系だけは `transformer_ref` を最初の参照リクエストでコールドロードするため、**初回のみ +55秒**(ref2i 実測: 初回 134.7s → 定常 79.3s)。
- **全モードが 45〜46GB に収まる**が、**t2va 系と参照系を同じプロセスで混ぜると両方の transformer が常駐して 74.3GB**(t2va に戻ったときのピークは 77.3GB)。48GB 1枚で運用するならプロセスをモード別に分けること。
- **参照系の律速はデノイズではなく参照のビジョンエンコード**。i2va 103.1s の内訳(ログのタイムスタンプ): 参照ビジョンエンコード約47s / デノイズ 22.0s / デコード+VAE往復 約10s / 参照VAEエンコード 約6s / 末尾で t2va 用 transformer を再ロード 約13s。turbo の効きが t2va(5.5倍)より小さく i2va で 2.8倍に留まるのはこのため。**リクエスト跨ぎの参照エンコードキャッシュ(47s)と「t2va へ戻さない」判断(13s)はいずれも未実装**。

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

48GB機での固定費削減の系譜(t2i turbo 4steps): 157s(GPU交換直後)→ 83.2s(`H3_TE_PREQUANT`)→ 約35s(`H3_TE_DEVICE`)→ **9.7s**(`H3_KEEP_TRANSFORMER`、16倍)。t2va 5秒・768²は turboなし30steps 351.4s → turbo 143s → 60.5s → **44.2s**(8.0倍)。

この **9.7s / 44.2s は 48GB機の記録**であり、その構成での値としては今も正しい。ただし**現在の最速ではない**: 2026-08-12 に 96GB機の int8 単騎(解放停止)が **7.40s / 28.13s**、2枚 bf16 なら **6.89s / 26.8s** で更新した(上表)。箱が違うので単純比較はできないが、更新の主因はカードではなく「デコード窓の解放を止めたこと」である。

### ピークVRAMの実測値

| 局面 | 内訳 | 実測(48GB機、`H3_LOWVRAM=1 H3_TE_DEVICE=cuda:1`) |
|---|---|---|
| デノイズ(ピーク) | transformer-int8 34.3GB + 活性化約6.6GB | 40.9GB |
| デコード | vaeペア11.3GB + バッファ | (デノイズ後、transformer解放済み) |
| `H3_KEEP_TRANSFORMER=1` 併用時のデコード | transformer 34.03GB常駐 + fp16デコード | 44.15GB(導出予測45.7GBに対し実測44.15GB) |

デノイズとデコードは時間的に重ならない(transformer はデコード直前に必ず解放される、`H3_KEEP_TRANSFORMER=1` を除く)。ピークは通常デノイズ時に出る。

### 低VRAM構成の速度と品質の正直な特性

16GB 単体・8GiB×2 は**成立する**(全機能が完走する)が、速度と品質には構成固有の癖がある。数値は誤解を避けるため出典を明示しておく。

**実測(96GB機に増設した RTX 4060 Ti 16GB での測定、`H3_LOWVRAM=group` + 投影TE(NF4)+ fp16 デコード + SDPA、30ステップ、2026-08-11)**:

| 構成 | t2i 768² | t2va 5秒 768² | ref2i | i2va(画像参照) | 音声参照 | 768×1344 5秒 |
|---|---|---|---|---|---|---|
| 実 16GB 単体(TE 同居) | 498s・ピーク7.4GB | 25分・11.4GB | (8GiB×2 で成立のため未実施) | 39分・9.41GB | 54分・11.96GB | 66分・13.37GB(nvidia実測15.2GB=実質上限) |
| 8GiB×2(計算側+TE側、バラスト模擬) | 512s・6.4GB | 25.6分・7.23GB | 17.7分・6.69GB | × デノイズOOM | × デノイズOOM | × デノイズOOM |

**8GiB×2 は ref2i まで**である。動画の参照系は参照トークンの分だけ系列が長く、最短の 768²/5秒ですらデノイズ活性化が 8GiB に入らない(必要量は実測 9.41GB)。**16GB 単体なら参照系も全部動く**。

- **group は毎ステップ重み転送律速**。検証に使った箱の2番スロットが **Gen3 x4(実効 ~3.5GB/s)**のため、16GB 単体で t2i 16.5s/step・t2va 49.8s/step と重い。これは「16GB カードの性能」ではなく「このスロット」の値であり、**Gen4 x16 のまともなスロットなら転送は ~1/8** になる(nvidia-smi の `pcie.link.gen/width` で確認できる)。カードの演算性能と混同しないこと。
- **int8+SDPA 軌道では FirstBlockCache が効かない**(`cache_skipped_steps: 0`、閾値 0.05)。PRO 6000 + sage + bf16 の軌道では大半のステップが省けていたのと対照的で、int8+SDPA では残差が閾値を下回らない。「効かない」=「壊れている」ではなく軌道依存で、閾値調整に最大2倍の余地はあるが未検証。
- **同一 seed でも構成をまたぐと構図が変わる**。int8+SDPA は生成の軌道を別のアトラクタへ移し、PSNR は vs 32B 基準 7.40dB と数値上は壊滅するが、目視ではむしろプロンプト忠実度が向上した実例がある(前段の bf16+sage がアニメ調の夕景に収束していたのに対し、写実的な雪山と朝霧の湖になった)。**構成をまたぐ PSNR/MD5 比較はもともと成立しない**(§7 の PSNR の扱いを構成間へ拡張したもの)。構成間の品質は目視で判断する。

なお 16GB 単体の検証は sage が sm_120 専用ビルドのため `H3_ATTN_BACKEND=default`(SDPA)で走らせる必要がある(増設カードは sm_89)。

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
| `H3_TE_PROJ` | (空) | 32B TEを投影4B TE(Qwen3-VL-4B + 学習済み線形写像)に置換。HFリポジトリID or ローカルパス。低VRAM対応の核(§4) |
| `H3_TE_PROJ_QUANT` | `bnb-4bit` | 投影4Bの量子化(`none`/`bnb-4bit`/`bnb-8bit`)。NF4で常駐3.11GB。32B用の`H3_TE_QUANT`とは別物 |
| `H3_TE_PREQUANT` | `1` | 量子化済みTE重みのディスクキャッシュ(ロード時間短縮) |
| `H3_TE_PREQUANT_DIR` | `models/prequant` | キャッシュ保存先 |
| `H3_TE_PREQUANT_MIN_FREE_GB` | `25` | この空きディスクを下回ると保存をスキップ(生成は継続) |
| `H3_TRANSFORMER_QUANT` | `none` | `int8`でtransformerを66.3GB→34GBに量子化 |
| `H3_LOWVRAM` | `0` | `1`=48GB級のフェーズ循環 / `group`=24-32GB級のブロック単位オフロード |
| `H3_KEEP_TRANSFORMER` | `0` | transformerをデコード位相でも解放しない。`H3_LOWVRAM` が `group` 以外(`1`/`0` どちらでも可)、`H3_TE_DEVICE` または `H3_TE_PROJ` のいずれか、`H3_VIDEO_VAE_FP16=1` の3条件が必須(§5参照) |
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

# 機能と設定の詳細(minimax-h3)

**日本語** | [English](FEATURES.en.md)

README から分離した、各機能・環境変数の詳細説明と実測値。各節は README にあった原文をそのまま移したもので、節内の日付はその機能を実装・実測した日を示す。使い方の要点は [README.md](../README.md)、時系列の作業記録は [HISTORY.md](HISTORY.md) を参照。

- [性能の概要](#性能の概要)
- [VRAM 構成と常駐設計](#vram-構成と常駐設計)
- [text_encoder](#text_encoder)
- [transformer とデノイズ](#transformer-とデノイズ)
- [生成モード](#生成モード)
- [UI](#ui)
- [LLM プロンプト強化](#llm-プロンプト強化)

## 性能の概要

### 高速化の効果(実測まとめ)

**このリポジトリの中心的な発見: 律速はデノイズではなく「モデルの load/free という固定費」
だった。** 以下は各節の実測値を1か所に集めたもの(再測定ではない)。

#### 累積: 素の構成から現在まで

48GB機(PRO 5000 48GB + RTX 4000 20GB)・`H3_LOWVRAM=1` 系・768²・**定常値**
(初回のモデルロードを含まない2回目以降。t2va は 5秒 = 124フレーム)。

| 段階 | t2i 1枚 | t2va 5秒 |
|---|---|---|
| 素の構成(30steps、turbo なし) | 157s | 351.4s |
| + lightx2v turbo LoRA(4steps) | 157s | 143s |
| + `H3_TE_PREQUANT`(量子化TEのディスクキャッシュ) | 83.2s | — |
| + `H3_TE_DEVICE`(TE を2枚目GPUへ) | 約35s | 60.5s |
| + `H3_KEEP_TRANSFORMER`(transformer 常駐) | **9.7s** | **44.2s** |
| | **16倍** | **8.0倍** |

デノイズ自体を速くする turbo だけでは **2.6倍で頭打ち**。残りの3倍強は
**すべて固定費の撤廃**から来ている(turbo なし 30steps のままでも、固定費撤廃だけで
t2i は 157s → 51.1s と3倍速い)。

#### モード別の到達点(全モードを同一構成で実測、2026-08-12)

構成: **int8 transformer 常駐 + 投影TE(NF4)+ デコード窓の解放停止 + turbo 4steps + sage +
fp16 デコード**、**GPU 1枚**(96GB機で実測。`H3_TRANSFORMER_QUANT=int8 H3_KEEP_TRANSFORMER=1
H3_VIDEO_VAE_FP16=1 H3_TE_PROJ=… H3_TURBO_LORA=1`)。

| モード | 単発(最速) | **連番なら1件あたり** | デノイズ | ピーク | 時間を食っている残りの固定費 |
|---|---|---|---|---|---|
| **t2i** 768² | **7.40s** | 7.9s(**0.94倍 = 使う意味なし**) | 2.40s | 45.0GB | VAE の CPU↔GPU 往復 ~3.3s |
| **t2va** 5秒 768² | **28.13s** | (バッチAPIなし) | 14.94s | 45.6GB | デコード 7.05s |
| **ref2i** 768²(参照付き静止画) | 79.3s | **47.0s(1.69倍)** | 7.8s | 45.4GB | 参照ビジョンエンコード **~47s** |
| **i2va** 5秒 768²(画像参照→動画) | 103.1s | **75.0s(1.37倍)** | 22.0s | 45.9GB | 同 ~47s + t2va transformer の再ロード ~13s |

**計測条件**(この表の数値の読み方):

- **すべて定常値。サーバ起動時の初回ロードは含まない**(2回目以降のリクエスト)。
  起動時に transformer + VAE を常駐させるのに **約50秒**かかるが、これはプロセスに1回だけ。
- **参照系(ref2i / i2va)は初回リクエストだけ別**: `transformer_ref` は起動時ではなく
  最初の参照リクエストでロードされるため、**初回のみ +55秒**(ref2i 実測: 初回 134.7s →
  定常 79.3s)。以後は常駐するので定常値になる。
- 解像度 **768×768** 固定。動画は 5秒 = **124フレーム**(24fps)、静止画は **22フレーム**。
- **turbo LoRA の 4ステップ**(`H3_TURBO_LORA=1`)。FirstBlockCache は turbo により自動 OFF。
  attention は sage。seed とプロンプトは各モードで固定。
- 「連番なら1件あたり」= 同じ参照/設定で複数件まとめて作るときの1件あたり
  (`/api/t2i_batch`, `/api/ref2i_batch`, `/api/ref2va_batch`)。実測は t2i/ref2i が3場面、
  i2va が2場面。**バッチ経路は `H3_LOWVRAM=1` 専用**なので単発列とは構成が違う —
  つまり「1件だけ作る最速手段」と「N件作る最速手段」を並べた表。

- **全モードが 45〜46GB に収まる** = 48GB カード1枚で全機能が最速級で動く。
  ただし **t2va 系と参照系を同じプロセスで混ぜると両方の transformer が常駐して 74.3GB**
  になる(実測。t2va に戻ったときのピークは 77.3GB)。48GB 1枚で運用するなら
  **プロセスをモード別に分ける**こと。
- **参照系の律速はデノイズではなく参照のビジョンエンコード ~47s**。turbo の効きが
  t2va(5.5倍)より小さく i2va で 2.8倍に留まるのはこのため。
- 2枚構成 + bf16 transformer なら **t2i 6.89s / t2va 26.8s** がこれまでの最速だが、
  77GB 級のカードが要る(差は 7% 程度)。詳細は日付節 2026-08-12。

#### 連番生成(バッチ)について分かったこと

- **t2i のバッチはもう使う意味がない**(0.94倍 = わずかに遅い)。バッチは元々
  「モデルのロード固定費をバッチ1回に償却する」機能で、**常駐化で償却すべき固定費が
  消えた**ため(常駐化前は 157s → 67.5s の 2.3倍だった)。
- **参照系だけは今も効く**。共有されるのが**ロードではなく参照ビジョンエンコード ~47s**
  だから。**バッチのステップ時間は単発と完全一致**しており(ref2i 2.598s vs 単発 2.599s、
  i2va 7.321s vs 7.323s)、バッチ経路自体のオーバーヘッドはゼロ — 差は純粋に
  「47s のエンコードを何回払うか」だけ。
- **場面数が増えるほど効く**: 節約は `47×(場面数-1)/場面数`。i2va なら 2場面 1.37倍 →
  3場面 1.44倍 → 5場面 1.57倍。同じキャラクターで長い物語を作る用途ほど有利。
- **「共有すれば縮む」モデルは静止画・動画とも成立**(実測削減 vs 予測: ref2i 3場面が
  32.3 vs 31.3s/枚、i2va 2場面が 28.1 vs 23.5s/本)。→ **参照エンコードをリクエスト跨ぎで
  キャッシュすれば、単発の繰り返しでも同じ短縮が得られる**見込み(未実装、日付節 2026-08-12)。
- **制約2つ**: バッチ経路は `H3_LOWVRAM=1` 専用で、それ以外では**黙って逐次実行に落ちる**。
  参照バッチは `H3_TE_DEVICE` と併用不可(「TE用GPUに 24GB 以上」のガード。**32B TE の
  vision 活性化を前提にした閾値**で 3.11GB の投影TEには過大 — 16GB の 4060 Ti が弾かれる)。
  どちらも**要見直し**。

> **計測の落とし穴(2026-08-12 に実際に踏んだ)**: バッチ側だけ `height`/`width` を渡さずに
> 単発と比べると、**バッチはサーバ既定の 16:9 キャンバス(1344×768)で生成される**ため
> 1.75倍のピクセル数になり、「バッチはステップが 1.75 倍遅い」という**存在しない現象**が
> 見える。ピクセル比 1344×768÷768² = 1.75 と実測のステップ時間比 1.753 が一致したことで
> 判明した。**モード間・経路間の比較では解像度を必ず明示すること。**

### 速度の正直な注記

**低VRAM構成は動くが遅い。** `H3_LOWVRAM=group` は毎ステップ int8 重み(~34GB)を CPU→GPU に
流すため、転送が律速になる。上の対応表の 16GB / 8GB の実測値は **PCIe Gen3 x4 スロット**での
もので、`16.5s/step`(t2i)〜`51s/step`(t2va)のほとんどは転送時間。**Gen4 x16 のまともな
スロットなら転送は約1/8**になるので、これは「16GBカードの性能」ではなく「この箱のスロット」の値。
また int8+SDPA の軌道では FirstBlockCache が効かず(`cache_skipped_steps: 0`)、閾値調整で
最大2倍の短縮余地がある(未検証)。

品質については、投影TE(PSNR 22.4dB vs 32B)も int8+SDPA も、同一seedでも構図が変わる
(**軌道の分岐であって劣化ではない**)。むしろ実測では int8+SDPA の方がプロンプト忠実度が
上がったケースもあり、参照(ビジョン)経路も目視で 32B 正解と同水準の忠実度を確認済み。
**構成をまたぐ品質は PSNR/MD5 では判定できないので目視で見ること。**

> **追記(2026-11-17)**: 上の記述は 2026-08 時点のもので、次の 3 点が変わっている。
> - 「上の対応表」は分離後は次節の「[VRAM×機能マトリクス](#vram機能マトリクス2026-08-11-実測)」を指す。
> - **turbo LoRA(4 ステップ)が `H3_LOWVRAM=group` でも使えるようになった**(自作ストリーマ + 起動時 `H3_TURBO_LORA=1`)。転送回数がステップ数に比例して減るため、転送律速の「動くが遅い」はほぼ解消できる。A4000 16GB(Ampere sm_86、PCIe Gen4 x4、SDPA)の実測: t2va 5 秒・768² が 30 ステップ 790 秒(peak 8.0GB)→ 4 ステップ 151 秒(peak 7.0GB)。8GB 級の指定(`H3_TE_STREAM=1 H3_TE_DIET=1`、VAE デコーダのストリーミング、`H3_VRAM_LIMIT_GB=8` で模擬)では 768² 209 秒・3.08GB、1024² 376 秒・3.73GB、fl2va 225 秒・3.37GB、ref2va 299 秒・3.02GB(README の対応表参照)。
> - 「int8+SDPA の軌道では FirstBlockCache が効かない」は現在は当てはまらない。同じ A4000(int8 + SDPA、30 ステップ)で t2i は 7 ステップ、t2va は 6 ステップ分がスキップされた(`cache_skipped_steps`)。
> 品質に関する段落(構成をまたぐと同一 seed でも構図が変わるのは軌道の分岐であって劣化ではない、目視で判定する)は変わらない。

## VRAM 構成と常駐設計

### VRAM×機能マトリクス(2026-08-11 実測)

○ = 実測で完走 / △ = 導出見込み(未実測)/ × = OOM。**時間は定常値**(初回のモデル
ロードを含まない2回目以降)、**ピークは torch 計測の割当ピーク**。速度はいずれもこの箱の
**PCIe Gen3 x4 スロット + sm_89(4060 Ti)+ SDPA** での値で、**Gen4 x16 なら重み転送は
約1/8**になる(下の「速度の正直な注記」参照)。16GB/8GiB 行は 30ステップ、
48GB+20GB 行は turbo 4ステップ。

| 構成 | t2i 768² | t2va 5秒 768² | ref2i | i2va(画像参照) | 音声参照 | 768×1344 5秒 |
|---|---|---|---|---|---|---|
| **実 RTX 4060 Ti 16GB 単体** | ○ 498s・ピーク7.4GB | ○ 25分・11.4GB | ○ | ○ 39分・9.41GB | ○ 54分・11.96GB | ○ 66分・13.37GB(実15.2GB=上限) |
| **8GiB×2**(計算+TE、バラスト模擬) | ○ 512s・6.4GB | ○ 25.6分・7.23GB | ○ 17.7分・6.69GB | × OOM | × OOM | × OOM |
| **12GB 単体** | △ 見込み | △ | △ | △ | ×に近い(実14.7GB) | × |
| **48GB+20GB**(2枚、turbo LoRA) | ○ 9.7s | ○ 44.2s | ○ | ○ | ○ | ○ |

- 12GB 単体は未実測。t2va 実ピーク 11.4GB / i2va 9.41GB(実11.2GB)が 12GiB に収まる計算だが、
  音声参照(実14.7GB)・768×1344(実15.2GB)は超える。
- 16GB / 8GB 構成の鍵は **投影TE**(Qwen3-VL-4B + 学習済み線形写像 ClipProj、NF4 で 3.11GB。
  32B の text_encoder の代替)+ `H3_LOWVRAM=group`(transformer int8 をブロック単位で
  ストリーム、GPU常駐は ~1.4GB)+ video VAE の fp16 デコード。詳細は日付節 2026-08-10〜11。
- **測定環境は2台のマシンで共有**: 16GB / 8GB 列は **96GB機(RTX PRO 6000 + 増設 4060 Ti 16GB)**、
  48GB+20GB 列は **48GB機(RTX PRO 5000 48GB + RTX 4000 SFF Ada 20GB)** での実測。旧版の実測
  (96GB単騎時代)は各日付節にそのまま残してある。

### VRAM対応表と主な環境変数(早見表)

用途に応じた起動例。上段5行は 96GB 機(PRO 6000)でのバラスト実測(32B TE 使用)、
下段3行は **投影TE を使う実カード実測**(2026-08-11。詳細は冒頭の
「[VRAM×機能マトリクス](#vram機能マトリクス2026-08-11-実測)」と日付節 2026-08-11)。

| GPU | 起動時の指定 | t2va 実測(768²・5秒) |
|---|---|---|
| 96GB | (指定なし=既定) | peak 92GB / 約160秒 |
| 80GB級 | `H3_TRANSFORMER_QUANT=int8` | peak 59.7GB / 約160秒 |
| 48GB級 | `H3_LOWVRAM=1` | peak 38.9GB / 約215秒 |
| 32GB級 | `H3_LOWVRAM=group` | peak 28.7GB / 約280秒(2026-11-15 の自作ストリーマで denoise は約 1.7 倍速、下の `H3_GROUP_STREAM`) |
| 18GB級 | `H3_LOWVRAM=group H3_TE_PRUNE=1` | peak 17.7GB / 約280〜320秒 |
| **16GB 単体**(実 4060 Ti) | `H3_LOWVRAM=group H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_VIDEO_VAE_FP16=1` | peak 11.4GB / 約25分※ |
| **6GB 単体 / 8GB 単体**(2026-11-15) | 上の行 + `H3_TE_STREAM=1 H3_TE_DIET=1` | 768²: 4.62GB / 1024²: 6.41GB(MD5 一致の既定。48GB 機で 272 / 552 秒)。空きが足りないときは attention をヘッド分割して 1024² 4.79GB(別サンプル相当、+15%)、`H3_CACHE=none` なら 3.03GB。ホスト RAM 32GB / 16GB は量子化キャッシュから直接流す(1K は同速、512² は約 2 倍)。Gen3 x4 の箱では上の※と同じ転送律速 |
| **12GB 単体**(見込み・未実測) | 16GB 単体と同じ | 実ピーク11.4GBが12GiBに入る計算。参照系・768×1344は不可 |
| **8GB×2**(バラスト実測) | 16GB 単体の指定 + `H3_TE_DEVICE=cuda:1` | peak 7.23GB / 約25.6分※ |

※ 低VRAM実カード列の時間は PCIe Gen3 x4 スロット + sm_89(SDPA)での実測。Gen4 x16 なら
重み転送は約1/8(「[速度の正直な注記](#速度の正直な注記)」参照)。sm_120 以外のカードは
`H3_ATTN_BACKEND=default` も必要。

```bash
# 例: 32GB級GPUで起動する
H3_LOWVRAM=group H3_TE_PRUNE=1 venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
# 例: 16GB カード単体で起動する(sm_89 なら H3_ATTN_BACKEND=default も付ける)
H3_LOWVRAM=group H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_VIDEO_VAE_FP16=1 \
  venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
```

主な環境変数(既定値。多くは**UIからも切り替えられる**ので、恒久的に変えたい時だけ指定する):

| 変数 | 既定 | 意味 |
|---|---|---|
| `H3_PROFILE` | `auto` | 2026-11-17 追加。主プロファイル系の変数が未指定なら総 VRAM から 96/80/48/32/16/8GB 級の既定値一式を入れる(`core/profile.py`)。32GB 以下の級では turbo LoRA(ref2v / fl2v の 4 ステップ)も ON にする(group では起動時に巻く必要があるため)。`file` ストリーミングでも可(LoRA 約 1.2GB はホスト RAM から単位ごとに流す、2026-11-18)。`96gb`〜`8gb` で強制、`off` で無効。sage が使えなければ `H3_ATTN_BACKEND=default` も入れる |
| `H3_TE_QUANT` | `bnb-4bit` | text_encoder の量子化(`none` は bf16 で 66.7GB) |
| `H3_TE_PRUNE` | `0` | TE の未使用上位レイヤー削除(出力は不変、-3.6GB) |
| `H3_TE_PROJ` | (無効) | **投影TE**: 32B TE を Qwen3-VL-4B+線形写像で代替(repo id か .safetensors パス。品質は近似、日付節 2026-08-10 参照) |
| `H3_TE_PROJ_QUANT` | `bnb-4bit` | 投影TE 4B の量子化(NF4 で 3.11GB。`none`=bf16 8.88GB / `bnb-8bit`) |
| `H3_TE_DEVICE` | (無効) | TE を2枚目GPUへ常駐(例 `cuda:1`。32B TE は 20GB 級、投影TE なら 8GB 級で可) |
| `H3_TRANSFORMER_QUANT` | `none` | `int8` で transformer を 66.3→34GB |
| `H3_PRUNED` | `0` | ref2va の transformer_ref を AdaLN-pruned 版(`multimodalart/MiniMax-H3-Pruned`)で読む(常駐 34→約21GB、`H3_TRANSFORMER_QUANT=int8` 併用必須。HyperFlow/`H3_LOWVRAM=group`/`H3_ADALN_PRECOMP` とは非互換) |
| `H3_PRUNED_QUANT` | `int8wo` | pruned の量子化方式: `int8wo`(既定、4.60 s/step・peak 28GB)/ `int8dyn-convrot`(旧既定、7.60 s/step)/ `int8wo-convrot` / `fp8` / `fp8-convrot` / `bf16`。方式ごとに別キャッシュ(`models/prequant/transformer_ref_pruned_*`) |
| `H3_LOWVRAM` | `0` | `1`=48GB級のフェーズ循環 / `group`=32GB級以下の block offload |
| `H3_KEEP_TRANSFORMER` | `0` | transformer 常駐で再ロード固定費を撤廃(成立条件は該当節参照) |
| `H3_CACHE` / `H3_CACHE_THRESHOLD` | `fbc` / `0.05` | FirstBlockCache(デノイズ -25%。int8+SDPA 軌道では不発の実測あり) |
| `H3_ATTN_BACKEND` | `sage` | sage は sm_120 専用ビルド。**それ以外のGPUは `default`(SDPA)必須** |
| `H3_TURBO_LORA` | `0` | 4/8ステップ蒸留LoRA(初回に約780MBをDL)。2026-11-17 から group でも可(`H3_GROUP_STREAM=own` + 起動時指定。`file` ストリーミングでは当初 LoRA 約 1.2GB が GPU 常駐で、8GB 級 + 8GB 上限の模擬で 768² 205 秒・4.19GB、1024² 376 秒・4.85GB、ref2va 768² 276 秒・4.13GB。同日 15 時にホスト RAM から単位ごとに流す方式に変え(`H3_FILE_STREAM_HOST_MIN_MB`)、6GB 上限の模擬で 768² 211 秒・2.82GB、1024² 380 秒・3.48GB。実機の Windows 11 + RTX 4060 Laptop 8GB / RAM 16GB と Windows 10 + RTX 3050 6GB / RAM 16GB で動作確認済み(生成時間は公平な条件で測り直してから載せる。重みの読み出しが律速で SSD の速さに依存)。A4000 16GB で t2va 5 秒 768² が 4 ステップ 151 秒。8GB 級の指定 + 8GB 上限の模擬で 768² 209 秒・3.08GB、1024² 376 秒・3.73GB、ref2va 768² 299 秒・3.0GB、fl2va 768² 225 秒・3.4GB。なお自作ストリーマの group モードでは ref2va が layout の device 不一致で落ちる退行があり、同日に修正) |
| `H3_ALLOW_WEIGHT_DOWNLOAD` | `0` | transformer / transformer_ref の bf16 重み(各 ~66GB)がキャッシュに無いとき、`1` のときだけ HF から取得する(既定は取得せず、エラーで案内する) |
| `H3_PIKA_RENOISE` | `0` | Pika H3 Flex の turbo LoRA 用 renoise サンプラー(実験) |
| `H3_BLOCK_COMPILE` | `0` | `1` で turbo LoRA 適用後に transformer_ref の 50 ブロックを torch.compile(attention 本体は graph break)。RTX PRO 6000・1024×768 で 1 step 約 -13%、初回のみ +5〜14s |
| `H3_LORA_FUSE` | `0` | turbo LoRA の要素ごとの演算を減らす: `1`=scale 1.0 の乗算を省く(ビット一致)/ `2`=加算を GEMM に融合。`H3_BLOCK_COMPILE` 併用時は上乗せなし |
| `H3_VIDEO_VAE_FP16` | `0` | video VAE を fp16 化(デコードのピークを削減。16GB 以下では必須) |
| `H3_LLM_URL` | `http://127.0.0.1:64660` | プロンプト強化に使うローカルLLM(任意機能) |
| `H3_LLM_API_KEY` | なし | LLMサーバのAPIキー(`Authorization: Bearer` で送る)。環境変数が無ければリポジトリ直下の `config.local.env`(git管理外)の同名行を読む |
| `H3_LLM_LOCAL` / `H3_LLM_LOCAL_GGUF` | `auto` / なし | 2026-11-17 追加。外部 LLM に接続できないとき CPU 上の内蔵 LLM(Gemma 4 E4B、GGUF Q4_0)へフォールバック。実体は llama.cpp のビルド済み `llama-server` の子プロセス(`core/local_llm.py` が自動取得・起動、コンパイラ不要)。`1` で内蔵のみ、`0` で無効。`H3_LLM_LOCAL_IDLE_S`(既定 300)で未使用時に停止、`H3_LLM_LOCAL_STOP_ON_GENERATE`(auto = RAM 32GB 未満)で生成開始時に停止 |

### 2026-10 更新: 24GB/32GB 単騎でリアルタイム会話生成

8月の初版以降、会話アプリ用途(ref2va 連続生成)に向けた大幅な高速化・低VRAM化を
本番リポジトリ(diffusers-movie-server)側で行い、本リポジトリへ再同期した。到達点:

| 構成 | 条件 | 結果 |
|---|---|---|
| **32GB 単騎(RTX 5090 級)** | 320×448・141f(5.875s 分) | **5.57s = 実時間の 0.95 倍(リアルタイム)**、peak 25.6GB |
| 24GB 単騎(RTX 4090 級) | 320×448・141f | 7.1s = 1.21 倍(短い応答なら実用圏)、peak 20.4GB |

主な追加技術:
- **ck-w4a8 量子化**(`H3_PRUNED_QUANT=ck-w4a8` / `H3_TRANSFORMER_QUANT=ck-w4a8`):
  ComfyUI の W4A8 int8 カーネル(pip の `comfy-kitchen==0.2.37`、要インストール)。
  AdaLN-pruned(multimodalart)と併用で 33B が**常駐 12.5〜14.9GB・denoise -28%・品質目視同等**
- **pinned 退避**(`H3_VAE_PINNED` / `H3_BASE_PINNED` / `H3_REF_PINNED`):
  モデルの退避を pinned メモリのポインタ付替えに(退避 0.3s / 復帰 2s)
- **TE 削減**(`H3_TE_DIET` / `H3_TE_STREAM`)、量子化済みキャッシュ、参照 latent/prefix キャッシュ
- lightx2v turbo 4step 蒸留(`H3_TURBO_LORA=1`)

単騎実行の環境変数一式(32GB 級。24GB 級は `H3_KEEP_REF2VA_VAE=0` に変える):

```bash
H3_LOWVRAM=1 H3_TE_PRUNE=1 H3_VIDEO_VAE_FP16=1 H3_TRANSFORMER_QUANT=ck-w4a8 H3_BASE_PINNED=1 H3_PRUNED=1 H3_PRUNED_QUANT=ck-w4a8 H3_REF_PINNED=1 H3_FL2VA_KEEP_TE=1 H3_KEEP_REF2VA=1 H3_KEEP_REF2VA_VAE=1 H3_REF_LATENT_CACHE=1 H3_DECODE_STREAM=1 H3_DECODE_VAE=light H3_VAE_PINNED=1 H3_TE_DIET=1 H3_VAE_SPLIT=1 H3_REF_PREFIX_CACHE_SINGLE=1 H3_REF_PREFIX_PARK=1 H3_TE_STREAM=1 H3_TURBO_LORA=1 ./run.sh
```

**注意点(必読)**:
1. **GPU 専有前提**: 空き VRAM 23GiB(24GB級)/ 28GiB(32GB級)以上。TTS・LLM は同じ GPU に同居不可
2. **初回キャッシュ作成**: ref 側は multimodalart/MiniMax-H3-Pruned の bf16(~38GB DL)から
   自動量子化(**空きホスト RAM 45GB 必要**。24GB カード上での初回量子化は未検証)。
   base 側は Kijai/MiniMax-H3-experimental の `minimax_h3_fl2va_pruned_w4a8_mixed.safetensors`
   (12.5GB)を `scripts/convert_kijai_w4a8.py` で変換(GPU 不要)
3. **ホスト RAM**: pinned 常駐 ~45GiB。**RAM 64GB 機は `H3_REF_PINNED` か `H3_TE_STREAM` を外す**
4. **24GB で `H3_KEEP_REF2VA_VAE=1` にしないこと**(発話⇄待機のプロンプト切替で OOM を実測)

会話アプリとしての実例は [Realtime_Narration_Video](https://github.com/animede/Realtime_Narration_Video)(+ 音声会話クライアント [Realtime_Conversation_Video](https://github.com/animede/Realtime_Conversation_Video))を参照。

### VRAM/RAM 設計 (重要、実測に基づく)

このマシンは VRAM 96GB に対し RAM は 94GB。**text_encoder は bf16ネイティブ配布で
実測66.73GB**(fp32配布を仮定した「bf16化で半分の33GB」という当初推定は誤りだった)。
transformer bf16 66.3GB / vae+audio_vae fp32 計11GB と合わせると約144GBになり、
**VRAMにもRAMにも同時に載らない**。diffusers の
`ComponentsManager.enable_auto_cpu_offload()`(全コンポーネントを定常的にRAM常駐させ、
アクティブな1つだけをGPUへ出す方式)は RAM 94GB では成立しないため採用していない。

TEのロード方式は環境変数 `H3_TE_QUANT` で選択する(既定 `bnb-4bit`、2026-08-04 に
A/B検証して既定化)。

#### `H3_TE_QUANT=bnb-4bit` (既定)

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

#### `H3_TE_QUANT=none` (bf16 TE、旧方式)

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

### 実測値 (RTX PRO 6000 Blackwell 96GB, 768×768, 124フレーム=5.17秒, 30steps)

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

### 48GB級VRAM対応 (`H3_LOWVRAM`、既定 `0`)

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
- ~~**量子化済みチェックポイントの事前保存によるロード時間短縮**: 未調査~~
  → **TE は 2026-08-08 に実装済み**(`H3_TE_PREQUANT`、既定ON。TEロード 53.0s→29.5s、
  リクエスト合計 -35%、MD5一致で等価性確認済み。上記「量子化済み text_encoder の
  ディスクキャッシュ」節を参照)。transformer int8 の直列化は**未検証のまま** —
  保存に約34GBを要し、この箱のディスク空き(43GB)では採用自体が非現実的なため。

### 24〜32GB級VRAM対応 (`H3_LOWVRAM=group`、2026-08-05追加)

#### 2026-11-15: ブロック転送を自作の窓付き先読みに置き換え(`H3_GROUP_STREAM`、既定 `own`)

`H3_LOWVRAM=group` のブロック転送は、diffusers の `enable_group_offload(block_level, use_stream=True)` から
`core/blockstream.py` の自作ストリーマに置き換えた(`H3_GROUP_STREAM=diffusers` で従来に戻せる)。
TE の `H3_TE_STREAM`(`core/testream.py`)と同じ設計で、50 ブロック(int8、計 30.1GiB)をピン留めした
ホスト RAM に置き、コピー専用 stream で window 個(`H3_GROUP_OFFLOAD_BLOCKS`+1、既定 2)先のブロックを
送りながら計算する。ブロック以外(埋め込み・token_refiner・出力層、bf16 で 1.6GiB)は GPU に常駐。
torchao `Int8Tensor` は外側の wrapper ではなく内側の `qdata` / `scale` / `zero_point` を差し替える
(diffusers の `_restore_torchao_tensor` と同じ方式)。flux3-klein-studio の 6GB 対応で実証した設計の移植。

実測(RTX PRO 5000 Blackwell 48GB の GPU0、`H3_LOWVRAM=group H3_TE_PRUNE=1`、t2va 512²×5 秒 124f、30 step、seed 1、2 回ずつ):

| | diffusers 版(`H3_GROUP_STREAM=diffusers`) | 自作版(`own`、既定) |
|---|---:|---:|
| デノイズ、`H3_CACHE=none` | 164.9〜165.0 秒(5.68 秒/step) | **96.4〜96.5 秒(3.32 秒/step)** |
| デノイズ、`H3_CACHE=fbc`(既定) | 165.5〜167.6 秒(FBC が 1 step も飛ばない) | **85.6〜86.1 秒(FBC が 6/30 step を飛ばす)** |
| ピーク VRAM | 17.7〜18.3GB | 19.5〜20.0GB(ブロック以外の常駐 +1.6GiB) |
| 出力(MD5) | `H3_CACHE=none` で自作版と**完全一致** | 同左 |
| ロード(ピン留め) | 20 秒(34GB を pinned 辞書へ) | 15 秒(ブロックごとに 1 本のバッファへ) |

- 速さの差は転送方式そのもの(FBC オフ同士で 1.71 倍)。既製品はグループごとのフック固定費と、CPU が計算に先回りして
  GPU 側の解放が遅れる構造のため(`docs/lowvram-techniques.md` の 4.2 節と同じ)。
- **FirstBlockCache が group モードでも効くようになった**。diffusers 版では FBC が 1 step も飛ばしておらず
  (「速度の正直な注記」の `cache_skipped_steps: 0` はこれ)、自作版では閾値 0.05 で 30 step 中 6 step を飛ばす。
  その分だけ出力は FBC オフと変わる(FBC 本来の挙動)。完全一致が要るときは `H3_CACHE=none`。
- group モードの transformer は、量子化済みキャッシュ(`models/prequant/transformer_int8`)があればそれを CPU に読む
  ようにした(bf16 原本が HF キャッシュに無くても起動できる。他モードと同じ fail-open)。キャッシュから読んだ
  Int8Tensor は `requires_grad=False` に固定する(diffusers 版の `swap_tensors` が autograd を辿って
  `aten.view unimplemented` で落ちるため)。bf16 原本を消した箱では `meta.json` の `source_snapshot` の照合を省略する。
- 解放は `BlockStreamer.release()`(`cudaHostUnregister`)で行い、`_host_emptyCache` に頼らない。
- ホスト RAM: ピン留め 30.1GiB(diffusers 版の pinned 辞書 34GB と同程度)。RAM ガード(`H3_GROUP_OFFLOAD_MIN_RAM_GB`)はそのまま。

#### 2026-11-15: 低 VRAM の到達点 — 16GB 構成を 8GB / 6GB へ(3 段のストリーミング)

同じ設計を、group モードで GPU に残っていた残り 3 つにも当てた。いずれも配置換えだけで**出力は MD5 一致**。

1. **投影 TE の LM 層**(`H3_TE_STREAM=1` が `H3_TE_PROJ` にも効くようにした)。4B の 36 層 1.9GiB をピン留めホストに置き、
   エンコード中だけ流す。group モードは TE をデノイズ中も常駐させる段取りなので、デノイズのピークが下がる。
2. **token_refiner の 2 ブロック**(`H3_GROUP_STREAM_REFINER`、既定 1)。int8 化の対象外で bf16 1.5GiB が常駐していた。
3. **video VAE デコーダの 36 ブロック**(`H3_GROUP_STREAM_VAE`、既定 `auto` = 最初のデコードで空き VRAM が 8GB 未満なら流す)。
   VAE 5.2GiB(fp16)の 4.8GiB がデコーダで、1・2 のあとはここがピークだった。タイルごとに送り直すのでデコードは遅くなる。

実測(RTX PRO 5000 48GB の GPU0、`H3_LOWVRAM=group H3_TE_PROJ=… H3_VIDEO_VAE_FP16=1 H3_TE_STREAM=1`、t2va 5 秒 30 step、seed 1):

| | 基準(自作ストリーマのみ) | +1 投影 TE | +2 token_refiner | +3 VAE デコーダ |
|---|---:|---:|---:|---:|
| 512²×5 秒 ピーク | 8.59GB | 8.33GB | 6.79GB | **4.67GB** |
| 768²×5 秒 ピーク | 10.78GB | 9.54GB | 7.99GB | **7.45GB** |
| 512² デノイズ / デコード | 90 / 6 秒 | 86 / 6 秒 | 90 / 6 秒 | 89 / 17 秒 |
| 768² デノイズ / デコード | 200 / 11 秒 | 193 / 11 秒 | 195 / 11 秒 | 195 / 30 秒 |

起動直後の GPU 常駐は 0.18GB。残るピークはデノイズの活性で、これは解像度と長さで決まる。

4. **feed-forward のトークン分割**(`H3_FF_CHUNK`、既定 `auto` = group モードで 8192 トークン、`0` で無効)。ブロックで最大の
   一時メモリは SwiGLU の射影(トークンごとに 2 × ffn_dim 14336 の bf16 = 57KB)で、トークンを 8192 個ずつ通すと
   一時メモリがその分だけになる。トークンごとの演算なので結果は同じ(MD5 一致)。

| 1K を含む活性のピーク(デノイズ段階、同構成) | 分割なし | `H3_FF_CHUNK=8192` |
|---|---:|---:|
| 768²×5 秒 | 7.45GB | **6.26GB** |
| 1024²×5 秒(3.8 万トークン、13.9 秒/step) | 10.73GB | **9.34GB** |

残るのは attention の K・V(1024² で各 0.55GB)と出力・残差ストリームで、正確なままでは削れない下限に近い。

5. **attention のクエリ分割**(`H3_ATTN_Q_CHUNK`、既定 0 = 無効、`core/attnchunk.py`)。K・V は丸ごと作り、Q の射影・
   QK ノルム・RoPE・カーネル・出力射影を 8192 トークンずつ回す(結果は MD5 一致)。効くのは 1K 以上で、
   1024²×5 秒のデノイズのピークが 9.34 → **8.53GB**、768² では差なし(6.26 → 6.38)。2〜4% 遅くなる(1024² で
   13.9 → 14.4 秒/step)ので既定では切ってあり、1K を小さいカードで回すときに `H3_ATTN_Q_CHUNK=8192` を足す。

ここまでで 1024²×5 秒は 8.5GB。さらに 3 つで「後ちょっと」を詰めた(いずれも MD5 一致)。

6. **メモリ節約版のブロック forward**(`H3_LEAN_BLOCK`、既定 `auto` = group モードで 8192 トークン、`core/leanblock.py`)。
   元の forward は norm1 の出力・attention の出力・norm2 の出力・FF の出力を全トークン分(1024² で各 0.41GB)作ってから
   足す。正規化 → 変調 → 射影/attention/FF → ゲート → 残差加算をチャンクで通し、全トークン分の中間は入力・出力・
   K・V の 4 本だけにする。4(FF 分割)と 5(クエリ分割)を包含する。norm をチャンクごとに 2 回計算する分、1024² で約 7% 遅い。
7. **流す単位をブロックの 1/3 に**(`H3_GROUP_STREAM_UNITS=half`、既定)。adaln_proj → attn → ff の順に 150 単位で流し、
   常駐する先読み分が 2 ブロック(1.3GiB)から小さい 2 単位になる。
8. **投影 TE にも `H3_TE_DIET=1`**(opt-in)。lm_head の解放と埋め込み表(151936 × 2560)の CPU ブリッジ化で 1.45GiB。

| ピーク VRAM(同構成 + `H3_TE_DIET=1`、VAE を流した小カード相当の総合値) | 最初(自作ストリーマのみ) | 1〜3(ストリーミング 3 段) | 全部(4〜8) |
|---|---:|---:|---:|
| 512²×5 秒 | 8.59GB | 4.67GB | **3.27GB** |
| 768²×5 秒 | 10.78GB | 7.45GB | **4.85GB** |
| 1024²×5 秒 | 10.73GB(デノイズ) | — | **6.82GB** |

ここまでが午前。午後に 1024²×5 秒の内訳を測り直す(`H3_LEAN_DEBUG=1` がブロック 0〜2 の allocated / peak を出す)と、
トークンは 38,321(動画 31,744 + 音声 + テキスト)で、残差ストリーム 1 本が 0.41GB、K・V が各 0.55GB、ブロック内の一時
メモリは attention 段が最大(K・V 1.1GB + sage がカーネル内で作る int8 の写し + チャンク分)だった。さらに 4 つ。

9. **fp32 パッチ埋め込みの丸め**(`core/leanblock.py` の `shrink_prologue`、`H3_LEAN_BLOCK` と一緒に入る)。transformer の
   forward は `proj_in`(fp32 のまま残す層)の出力を全トークン分作って bf16 に丸めて詰めるが、丸める前の fp32 の方が
   ローカル変数としてブロックのループ中ずっと残る(1024² で 0.68GB)。射影の出口で同じ丸めをしてから返すので値は
   同じまま、残るのが半分になる。1024² 6.82 → **6.41GB**、768² 4.85 → **4.62GB**(MD5 一致)。
10. **残差の in-place 更新**(同)。ブロック 2 以降は入力バッファに直接書く(各チャンクは自分の行を読み終えてから書く
    ので結果は同じ。MD5 一致)。FBC はブロック 0 の入力と出力を参照するのでブロック 0・1 は新しいバッファに書き、
    FBC が持つ残差 3 本(head_block_output / head_block_residual / tail_block_residuals、1024² で計 1.2GB)と合わせると
    ピークはブロック 1 のまま → FBC 有効時の総合値は変わらない。`H3_CACHE=none` なら全ブロック in-place で 1 本下がる。
11. **ヘッド分割の attention**(`H3_ATTN_HEAD_GROUPS`、既定 `auto`)。K・V・Q を 56 ヘッドの G 分割で順に作る(to_q/to_k/to_v
    の重みの行スライス。torchao の Int8Tensor はストリーマが中身だけ GPU に置くので、中身をスライスして組み直す)。
    各グループの attention 出力は全ヘッド分の 1 本(attn_out)に書き、出力射影はそこから全ヘッドまとめて行う。同時に
    持つのが K・V の 2 本から attn_out + K/G + V/G になり、sage がカーネル内で作る K・V の写しも 1/G になる。
    正規化と変調を G 回やり直すので遅い(G=4 +13%、G=8 +15〜35%、G=28 +70%)。**計算は同じだがビット一致はしない**:
    GEMM の行数(7168/G)と attention カーネルのヘッド数が変わるとカーネル選択と加算順序が変わり、bf16 の最下位
    ビットが動くことがある(sage は K の平均を引く前処理がヘッド数とトークン数の組み合わせで揃ったり揃わなかったり
    する)。30 ステップで増幅されて別のサンプル相当の動画になる(基準との PSNR は 1024² 34.4dB / 768² 38.7dB。FBC の
    有無の差が 34.2dB なので、既定で使っている FBC の近似と同程度の差で、品質低下ではない)。そのため **auto は G=1 で
    収まるときは必ず G=1** を選び、収まらないときだけ 4 → 7 → 8 → 14 → 28 の順に空き VRAM(`H3_VRAM_LIMIT_GB` の上限も
    見る)に入る最小の G を使う(生成の最初の forward で決めて、その生成の間は固定。FBC の残差 3 本と 1GiB の余裕を先に引く)。G=4 固定で 1024² 5.09GB(16.7 秒/step、+8%)、G=8 固定で 1024² **4.79GB**(17.9 秒/step)、768² **3.95GB**
    (8.8 秒/step)。5.8GB 上限で auto に任せると G=4 → 28 を選んで 4.79GB(26 秒/step)。ブロック単位の診断で一致した
    組み合わせ(1024² の G=4)でも 30 ステップの実機出力は MD5 が変わったので、G>1 は常に「別サンプル」と考える。
12. **FBC を切る**(`H3_CACHE=none`)。FBC の残差 3 本が消え、全ブロック in-place になる。G=8 と合わせて 1024²×5 秒の
    デノイズが **3.03GB**(30 ステップ全部計算するので 23 秒/step。FBC 自体が別サンプル相当の近似なので出力は変わる)。

| ピーク VRAM(同構成、総合値) | 〜8(午前) | 9〜10(MD5 一致、既定) | + ヘッド分割 G=8(別サンプル) | + `H3_CACHE=none` |
|---|---:|---:|---:|---:|
| 512²×5 秒 | 3.27GB | **2.90GB** | — | — |
| 768²×5 秒 | 4.85GB | **4.62GB** | 3.95GB | — |
| 1024²×5 秒 | 6.82GB | **6.41GB** | 4.79GB | 3.03GB |

正確なまま(MD5 一致)では **512² 2.90GB / 768² 4.62GB / 1024² 6.41GB** が下限で、残りは K・V と残差ストリームと FBC の残差 3 本。
ヘッド分割を許せば **1024²×5 秒が 6GB カード**に入る(auto が必要なときだけ使う)。`/api/status` の
`group_stream_refiner` / `group_stream_vae` / `group_stream_units` / `ff_chunk` / `attn_q_chunk` / `lean_block` /
`attn_head_groups` と、直近の forward が選んだ `lean_block_last`(トークン数・G・in-place の開始ブロック)で状態を
確認できる(生成結果の JSON にも `lean_block_last` が入る)。

#### ホスト RAM 32GB / 16GB の PC: 重みをファイルから直接流す(`H3_GROUP_STREAM_HOST=file`、2026-11-15)

ここまでの low VRAM 構成は int8 transformer 全体(30GiB)をピン留めしたホスト RAM に置くので、RAM 48GB 級が前提だった。
`H3_GROUP_STREAM_HOST=file`(`core/filestream.py`。minimax-h3-image-server の `app/filestream.py` と同じ方式)は、
量子化済みキャッシュ(safetensors、`from_pretrained` が mmap で読むので RAM に展開されない)から単位(ブロックの 1/3)ごとに
8 スレッドで 32MB ずつ並列に読み、ピン留めした受け皿(3 本 × 0.24GiB、2 単位先読み)経由で GPU へ流す。必須のホスト RAM は
その受け皿と token_refiner の受け皿(2 × 0.36GiB)だけで、残りは OS のページキャッシュに乗る分だけ速くなる。
キャッシュに収まらない末尾の単位は読むたびに `posix_fadvise(DONTNEED)` で捨て、巡回アクセスで全部が LRU に押し出される
のを防ぐ(捨てる数は auto: 「原本 + この後ピン留めする TE / VAE の分 − 使える RAM(cgroup の上限も見る)」)。
file のときは VAE デコーダの流し元もピン留めしない(fp16 の写しをそのまま使う)。重みは同じ int8 なので出力は同じ。

- `H3_GROUP_STREAM_HOST`: `auto`(既定。搭載 RAM が `H3_GROUP_STREAM_FILE_BELOW_GB` = 48GB 未満なら file)/ `pinned` / `file`。
  bf16 原本から量子化した(キャッシュが無い)ときは pinned に落ちる。`H3_GROUP_STREAM_DROP_UNITS`(auto / 数)、
  `H3_GROUP_STREAM_RING`(受け皿の本数、既定 3)。`H3_FILE_STREAM_HOST_MIN_MB`(既定 1): ファイルに無いテンソル(turbo LoRA の
  lora_a / lora_b、計 1.24GiB)のうちこのサイズ以上のものは GPU に常駐させず、ホスト RAM(ページャブル)に置いて単位ごとに
  受け皿へ memcpy して流す(2026-11-18。常駐だと 6GB のカードで Windows が共有メモリへ退避して極端に遅くなる、RTX 3050 で実測)。`/api/status` の `group_stream_host` / `group_stream_file`。
- Windows は `fadvise` が無いので捨てる制御は効かない(読み自体は `FILE_FLAG_SEQUENTIAL_SCAN` で動く。画像サーバの
  RTX 3050 / RAM 16GB 機で実証した方式)。2026-11-18: `FILE_FLAG_SEQUENTIAL_SCAN` は「RAM に収まらないもの」(30GB の
  transformer)にだけ付け、収まるもの(VAE デコーダ 4.5GB、投影 TE 1.9GB)は通常のキャッシュ付きで開く。前者を全部に
  付けていたため、RTX 4060 Laptop / RAM 16GB でデコード中にタイルごとの再読込が毎回 NVMe から走っていた
  (デコード 124 秒、NVMe 約 1GB/s で張り付き)。判定は「重み + 1GiB + 予約 ≤ 使える RAM」、ログの `page cache kept / not kept`。
- 同日: デコーダを **GPU に置き切る**経路(`H3_VAE_DECODER_RESIDENT`、既定 auto、`core/filestream.py` の `ResidentLoader`)。768²・5 秒の
  デコードは 7 クリップ × 16 タイル ≒ 150 パスで、タイルごとに 36 ブロック 4.5GB をホストから送り直す方式は RAM 16GB の PC では
  その転送(ページキャッシュ → 受け皿 → GPU)がデコード時間の大半だった(RTX 4060 Laptop: 1 パス 0.77 秒のうち 0.68 秒が読み待ち、
  デコード約 90 秒)。空き VRAM が「4.5 + 0.5 + 出力 + 余裕 0.5GiB」以上なら 1 回読んで GPU に置いたままにする。A4000 の 8GB 上限で
  強制して実測: デコード 86 → 34 秒、全体 201 → 152 秒、ピーク 7.03GB、出力ハッシュ一致。8GB のカードは表示用の分を引くと
  768² では足りない(auto は取らない)。12GB 以上のカードに RAM 16GB の組み合わせで効く。
- 量子化キャッシュは `scripts/prepare_lowvram.py` で先に作る(bf16 シャードを 1 つずつ落として量子化して消す。RAM 16GB で可)。
  Ref2VA / Ref2I は `transformer` とは別の重み `transformer_ref/`(66GB、全シャードの内容が異なる)を使うので、
  `--ref` で `transformer_ref_int8` も作る(+32GB)。2026-11-18 まで ref2va の group 読み込みはキャッシュの有無を見る前に
  40GB の RAM ガードを通していたため、低 RAM 機では参照モードが必ず止まっていた。base 側と同じく「キャッシュがあって
  file で流すならガード無し」にし、キャッシュが無いときは `--ref` の案内で止める。

実測(48GB 機の GPU0、RAM の上限を cgroup で模擬し、量子化キャッシュのページキャッシュを捨ててから開始。この機の
NVMe は直読み 3.3GB/s。出力はすべて pinned と MD5 一致):

| 原本の置き場所 | 必須ホスト RAM | 512²×5 秒 | 1024²×5 秒 | 備考 |
|---|---:|---:|---:|---|
| pinned(参考) | 約 38GiB(transformer 30 + refiner 1.4 + TE 1.9 + VAE 4.5) | 2.98 秒/step | 15.1 秒/step | VRAM 2.90 / 6.41GB |
| file、RAM 制限なし(キャッシュ温) | 約 8GiB | 3.11 秒/step | — | VRAM 2.49GB(GPU の先読みが 1 単位少ない) |
| file、RAM 28GB 上限(32GB 機相当) | 同上 | 6.1 秒/step | **14.7 秒/step** | 150 単位中 94 をディスク直読み。1K は読みが計算に隠れる |
| file、RAM 16GB 上限 | 同上 | 8.0 秒/step | — | 全 150 単位をディスク直読み(30GiB/step)、MD5 一致 |
| file、RAM 12GB 上限(16GB 機相当)、VAE・TE もファイルから(下記) | 約 2GiB | 8.0 秒/step | **15.4 秒/step** | MD5 一致。VAE は空き RAM があればデコードの間だけピン留め |

1K では読み(最悪 30GiB/step ÷ 3.3GB/s ≈ 9 秒)が 15 秒の計算に隠れて pinned と同じ速さ、512² は読み律速で約 2 倍かかる。
**RAM 32GB の PC で 1024²×5 秒が pinned と同速、RAM 16GB でも 1K まで通る**(速度はディスクの連続読み出し速度に依存)。
最初の 12GB 上限の試験では、デコードの直前にプロセスの匿名メモリが約 12GB に達して落ちた(`proc_anon_gb` の内訳:
VAE の fp16 の写し 4.8、受け皿 1.6、TE の層のピン留め 1.9〜2.4、基本 0.6、デコードのフレームバッファ)。残っていた 2 つも
同じ方式に乗せて、RAM 16GB 相当で 1K を通した:

- **VAE デコーダの fp16 キャッシュ**(`core/vaecache.py`、`models/prequant/vae_decoder_fp16/`、4.5GiB、初回に 2 秒で作る)。
  file モードでは起動時にデコーダブロックを fp16 に写さず(写しが 4.8GiB の匿名メモリだった)、ブロックを 0 要素の
  プレースホルダにしておく。デコードのときは空き RAM を見て、**4.5GiB + 2GiB あればキャッシュからピン留めに 1.4 秒で
  読み込み、デコードの間だけ持つ**(pinned と同じ速さ)。無ければタイルごとにファイルから流す(1K のデコード 48 → 95 秒)。
  終わったらピン留めを返す。値は `vae.to(float16)` と同じ丸めなので出力は同じ。
- **投影 TE の量子化済みキャッシュ**(`H3_TE_PROJ_PREQUANT=1`、既定。`models/prequant/te_proj_<model>_bnb-4bit/`)。
  bnb-4bit の TE を一度だけ保存し(2 秒)、次回から読む(ロードはこの機では bf16 からの量子化と同じ 9〜10 秒だが、読む量が
  bf16 8GB から nf4 2.2GB になるのでディスクの遅い PC で効く)。
  file モードでは LM 層 36 本(1.9GiB)をこのファイルから直接流し、ピン留めしない(受け皿 2 × 0.06GiB)。

これで 12GB 上限(16GB 機相当)でも **1024²×5 秒が完走**(15.4 秒/step、デコードはファイル流しで 97 秒、MD5 一致)、
768² も 8.3 秒/step(VAE をデコードの間だけピン留めしてデコード 30 秒、MD5 一致)。必須のホスト RAM は受け皿 3 本 + TE の受け皿 2 本の約 1GiB と
プロセス本体で、約 2GiB。

##### VRAM 級 × ホスト RAM 級の早見表(t2va 5 秒・30 ステップ、既定の低 VRAM 構成)

速度はこの機(RTX PRO 5000)の秒/step。RTX 3050 などではその数倍かかる。○ = MD5 一致のまま入る、△ = ヘッド分割
(別サンプル相当)か `H3_CACHE=none` で入る、✗ = 入らない。

| VRAM \ ホスト RAM | 16GB(file、全単位をディスクから) | 32GB(file、一部をディスクから) | 48GB 以上(pinned) |
|---|---|---|---|
| **6GB** | 512² ○ 8.0 / 768² ○ 8.3。1K △ 約 17(ヘッド分割、デコードはファイル流しで 97 秒) | 512² ○ 6.1 / 768² ○ 7.1。1K △ 16.8(auto が G=4 を選び 4.4GB) | 512² ○ 3.0 / 768² ○ 7.1。1K △(ヘッド分割で 4.8GB、`H3_CACHE=none` で 3.0GB) |
| **8GB** | 同上。1K ○ 15.4(5.8GB、デコード 97 秒) | 同上。1K ○ 14.7(5.8GB) | 同上。1K ○ 15.1(6.4GB) |
| **12〜16GB** | 同上 | 同上。空き VRAM が 8GB 以上なら VAE を流さずデコードが速い(768² 31 → 11 秒級) | 同上 |
| **24GB 以上** | 同上 | 従来の表(`H3_LOWVRAM=1` 系)も選べる | 同左 |

- VRAM が決めるのは「何が入るか」: 512²(2.9GB)と 768²(4.6GB)は 6GB で入り、1K は 8GB なら MD5 一致のまま
  (pinned 6.4GB / file 5.8GB)、6GB ではヘッド分割(auto が必要なときだけ使う)か `H3_CACHE=none`。
- ホスト RAM が決めるのは「速さ」: 48GB 以上はピン留めで最速、32GB は 768² と 1K が同速で 512² が 2 倍、16GB は
  512² が 2.7 倍・768² が 1.1 倍で、1K も通る(VAE と TE もファイルから流す)。
- file モードは GPU の先読みが 1 単位少ないぶん VRAM ピークも下がる(512² 2.49 / 768² 3.71 / 1K 5.76GB)。
  file で VAE デコーダをピン留めするかは `H3_GROUP_STREAM_VAE_PIN`(auto = RAM 24GB 以上。しないと 768² のデコードが
  31 → 53 秒)。

`H3_LOWVRAM=1`(48GB級)は transformer(34GB)を毎リクエスト GPU に丸ごとロードするため、
24〜32GB級カードでは transformer 単体でも収まらない。`H3_LOWVRAM=group` は
diffusers-server(姉妹プロジェクト)の CLAUDE.md #33/#34/#37 で確立された
「block-level group offload」パターンをこのプロジェクトに移植したもので、transformer を
**ホストRAMに常駐**させたまま、denoise の各ステップで必要なブロック(50層中1〜2層、
~0.68GB×1〜2)だけを都度 GPU へ出し入れする。transformer は**プロセス起動時に一度だけ
ロードされ、リクエストをまたいで常駐し続ける**(`H3_LOWVRAM=1` のような毎リクエスト
再ロードは発生しない)。

#### PR側の「streamed offload時のload-time量子化」調査結果

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

#### 実装の要点

- `_ensure_transformer_group()`(`core/runner.py`)が
  `device_map={"transformer": "cpu"}` + `TorchAoConfig(Int8WeightOnlyConfig)` で
  CPU上に量子化ロードしてから `enable_group_offload(offload_type="block_level",
  num_blocks_per_group=1, use_stream=..., low_cpu_mem_usage=...)` を呼ぶ
  (transformer_refも同型の `_ensure_transformer_ref_group()`)。
- TEは `H3_LOWVRAM=1` と同じくbnb-4bit必須(起動時強制)。t2vaの定常状態では
  TEはリクエストをまたいで常駐する(transformerがそもそも常駐するため、
  TEも常駐させておいた方がリクエストごとの再ロードコストを避けられる)。

#### 【重大な発見】`use_stream=True` + `low_cpu_mem_usage=True` の併用は
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
| `use_stream=True, low_cpu_mem_usage=True`(併用時) | **クラッシュ** | - |

(2026-08-10 訂正: 当初この組み合わせを「diffusers既定」と記載していたが誤り。`apply_group_offloading` の API 既定値は `use_stream=False, low_cpu_mem_usage=False` で、省メモリ目的で両方をオプトインしたときに踏む)
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

#### choreography最終形(フェーズ×常駐物×ピーク)

| フェーズ | 常駐する大きいもの | 備考 |
|---|---|---|
| 起動時preload | transformer(int8, CPU常駐+groupoffloadフック) | 約90秒(CPUロード70s+pin化22s) |
| t2va encode | TE-nf4(GPU,21GB) + transformer(CPU) | |
| t2va denoise | TE-nf4(GPU,21GB) + transformerの1-2ブロック(GPU,~1.4GB) | |
| t2va decode | vaeペア(GPU,~11GB) + transformerの1-2ブロック | **TEはこの窓だけ強制解放**(下記参照)、decode後に再ロード |
| ref2va参照エンコード | vaeペア(GPU,11GB) | TEはこの窓だけ強制解放(下記参照) |
| ref2va denoise | TE-nf4(GPU,21GB) + transformer_refの1-2ブロック | |
| リクエスト間定常 | transformer(CPU) + TE-nf4(GPU) | ref2va後はtransformer_refが未ロードに戻る(t2va↔ref2va切替のたび再ロード) |

#### 【実装中に発見・修正した2つ目のバグ】decode窓・参照エンコード窓でのTE強制解放が必要だった

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

#### MD5一致チェックの結果

同一seed(768²・5秒・30steps・キツネのプロンプト、seed=12345)で、通常int8モード
(`H3_LOWVRAM`未指定、`H3_TRANSFORMER_QUANT=int8`、FBC `H3_CACHE=fbc`有効)の
出力と `H3_LOWVRAM=group` の出力を比較したところ、**FBCのキャッシュスキップ判定が
実行経路の違いで異なった**(`cache_skipped_steps`が6→0)ため素朴な比較ではmp4が
不一致だった。FBCは前ステップとの残差の類似度という数値的に鋭敏な判定のため、
数学的に等価な演算でも経路が変わればスキップ判定が変わりうる(劣化ではない)。
両モードとも `H3_CACHE=none` に揃えて再比較したところ、**mp4がバイト完全一致
(md5一致)**することを確認した。group offloadの計算内容は既存経路と数学的に
同一であることの裏付け。

#### 計測表: 32GB制限・24GB制限プローブ

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

#### VAE tiling調査結果

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

#### FBC/sage共存の確認結果

全てのballast検証(32GB・24GB双方)を通じて `H3_CACHE=fbc`(既定)・
`H3_ATTN_BACKEND=sage`(既定)を有効にしたまま実行し、group offloadのフックと
競合するエラーは一切発生しなかった(FBCはブロック単位の計算スキップ判定、
group offloadはブロックのGPU常駐管理、sageは各ブロック内部のattention実装、と
三者は独立したレイヤーで動作するため)。MD5一致チェック(`H3_CACHE=none`で
実施)とは別に、既定のFBC有効設定での通し実行(32GB制限のt2va 2本)が完走した
ことも確認済み。

#### Ref2VAのRAM制約(既知の制限、未解決)

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

### 2026-08-09: H3_KEEP_TRANSFORMER — transformer 常駐で毎リクエスト再ロード固定費を撤廃

`H3_LOWVRAM=1` はデコード直前に transformer を解放し、次リクエストで**毎回**再ロードする
(実測 14.8〜32.7s の固定費、`docs/RESIDENCY.md` §5.5)。`H3_KEEP_TRANSFORMER=1` はこの
解放をスキップし、transformer を GPU に常駐させたまま次のリクエストに持ち越すことで
この固定費を初回ロードの1回だけに収束させる。

**成立条件は3つとも必須**(`core/runner.py` の import 時ガードで強制、欠けたら
`RuntimeError`)。VRAM 収支の理由:

1. `H3_LOWVRAM=1`(raw `"1"` のみ。`group` は transformer をホストRAM常駐+ブロック
   offload する別設計のため対象外)
2. `H3_TE_DEVICE` 設定済み(TE を別GPUに常駐)— **これが無いとエンコード位相が先に
   破綻する**: TE-nf4 17.45GB + 常駐transformer-int8 34.3GB = 51.75GB が実効予算を
   超える
3. `H3_VIDEO_VAE_FP16=1` — encode 位相は TE を別GPUに逃がすことで回避できるが、
   decode 位相は fp32 デコードピーク16.29GBのままだと transformer 34.3 + 16.29 =
   50.6GB で実効予算(~49.8GB)を超える。fp16 化した VAE のデコードピーク ~11.4GB
   なら 34.3 + 11.4 = 45.7GB で収まる(video VAE fp16 の品質影響は既知実測 PSNR
   **39.97dB**・目視差なし、詳細は「video VAE の fp16 化」節)

既定(`H3_KEEP_TRANSFORMER=0`)は挙動不変。**ref2va(`transformer_ref`)は対象外** —
このフラグの範囲外で、従来どおり毎回解放される。

```bash
# 推奨起動コマンド (48GB GPU0 + 20GB GPU1)
H3_LOWVRAM=1 H3_TE_PRUNE=1 H3_TE_DEVICE=cuda:1 H3_VIDEO_VAE_FP16=1 H3_KEEP_TRANSFORMER=1 \
  venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
```

**実機 E2E 実測(2026-08-09、48GB GPU0 + 20GB GPU1、上記起動構成)**:

- transformer(int8) は初回リクエストで1回だけロード(32.0s)。以降のリクエストで
  再ロードなし(サーバーログで確認)
- t2i turbo 4steps 定常: **9.7s/枚**(denoise 4.32s、decode 1.5s)。peak VRAM 41.97GB
  (デノイズ時)
- t2i steps=30 定常: 51.1s(denoise 45.7s)。peak 41.97GB
- t2va 5秒 turbo 4steps: **44.2s**(denoise 26.05s、decode 10.81s)。**peak VRAM
  44.15GB = デコード位相**(transformer 34.03GB 常駐 + fp16 デコード)。導出予測
  45.7GB に対し実測 44.15GB、カタログ 48.9GB に対し余裕 ~4.8GB
- nvidia-smi 実測ピーク 42,620 MiB(1秒サンプリング。瞬間ピークは torch 計測の
  44.15GB が正)
- 同一seed 出力等価性: フラグOFFのベースライン(他は同条件)と PNG MD5 **完全一致**
  (seed=11、md5 `665eadddea8f34298a1b5b89e69d4bd0`)。ベースライン側は total 63.27s
  (transformer ロード込み)/ peak 36.4GB

**高速化の系譜**(48GB GPU0・`H3_LOWVRAM=1` 系、t2va は 768²・5秒):

| | t2i | t2va 5秒 |
|---|---|---|
| turboなし 30steps(GPU交換直後の素の構成、08-07) | 157s | 351.4s(デノイズ197.7s) |
| lightx2v turbo 4steps 導入(08-07朝) | 157s | 143s |
| + `H3_TE_PREQUANT` | 83.2s | — |
| + `H3_TE_DEVICE` | ~35s | 60.5s |
| + `H3_KEEP_TRANSFORMER`(本節) | **9.7s** | **44.2s** |

t2va は素の構成比 **8.0倍**(351.4s → 44.2s)。turbo だけでは 2.6倍(デノイズは7.6倍
短縮されるがロード固定費 ~110s が残る)で、残りは固定費撤廃
(`H3_TE_PREQUANT`/`H3_TE_DEVICE`/`H3_KEEP_TRANSFORMER`)の寄与。なお本節構成での
turboなし 30steps は t2i で 51.1s(上記実測)— 素の 157s に対し、30steps のままでも
固定費撤廃だけで3倍速い。

詳細な VRAM 収支の導出・位相×常駐表は `docs/RESIDENCY.md` §5.5・§5.6 を参照。

### text_encoder を別GPUへ常駐 (`H3_TE_DEVICE`、既定 ``、2026-08-09)

`H3_LOWVRAM=1` が毎リクエストで TE を再ロードする根本原因は、**デノイズ中に TE の
置き場所が無いこと**にある(48GB では transformer-int8 34GB + 活性化 5GB = 39GB で、
残り 9GB に TE の 17.45GB は入らない)。TE を2枚目のGPUへ逃がせばこの再ロードが消える。

```bash
# 例: TE を cuda:1 に常駐させる (CUDA_VISIBLE_DEVICES は設定しない = 両GPUを見せる)
H3_LOWVRAM=1 H3_TE_PRUNE=1 H3_TE_DEVICE=cuda:1 venv/bin/python -m uvicorn app:app --port 8611
```

**実測(RTX PRO 5000 48GB + RTX 4000 SFF Ada 20GB、t2i turbo 4steps、768²)**:

| | 1回目 | 定常(2回目以降) |
|---|---|---|
| 従来(TE も cuda:0) | 81.3s | 67.6〜86.2s(平均 **78.4s**) |
| **TE を cuda:1 へ** | 88.2s(TE初回ロード込み) | 33.6〜36.7s(**約35s、-55%**) |

TE のロードは**プロセス起動後1回だけ**。GPU1 に 17GB 常駐したまま、GPU0 はアイドル時
2.3GB まで下がる。

- **ref2va は自動的に拒否される**(20GB級の場合): 2048px 短辺の参照を vision tower に
  通す活性化が入らず OOM することを実測済み(19.25GB 使用中に 204MB 不足)。
  TE用GPUが 24GB 未満なら理由を添えて 400 を返す — 「動くはず」で走らせて OOM させない。
  24GB 以上なら自動的に許可される。t2va/fl2va/t2i とその各バッチは併用可能
- **PCIe 幅は問題にならない**: GPU1 は Gen4 x4(x16対応カードだが4レーン接続)だが、
  TE は起動時に一度載せるだけで、毎リクエストの転送は prompt_embeds の約42MB のみ(約6ms)

**出力は従来構成とビット一致しない**(PNG の MD5 が異なる)。原因は **sm_120(Blackwell)
と sm_89(Ada)のアーキテクチャ差による丸め**で、実測した `hidden_states[50]` の
**相対RMS差は 0.084%**(max_abs_diff 3.5、値のRMS 57.4)。これは既に採用済みの
参照プレフィックス共有の差(1.5%)より**20倍小さく**、バグの水準(ネガティブ
コントロール 27-30%)とは桁が違う。Sage Attention や int8 と同種の軌道ドリフトであり、
目視でも品質は同等。**ビット再現が要る対照実験では `H3_TE_DEVICE` を外すこと**。

#### 実装の要点(`_execution_device` の罠、再び)

`_execution_device` は components 順(`text_encoder, tokenizer, processor, vae, ...`)で
最初の nn.Module のデバイスを返す(実装を読んで確認)。**TE を cuda:1 に置くと
これが cuda:1 を返し**、layout/latents/timesteps も decode も別GPUにテンソルを作る。

最初は「layout の窓だけ塞ぐ」実装にしたが、**decode でも発火した**
(`latents = latents * latents_std + latents_mean` が
`Expected all tensors to be on the same device, cuda:0 and cuda:1` で失敗、実機で再現)。
窓を個別に塞ぐ方式は漏れるため、**「既定では TE をパイプから外しておき、エンコード中
だけ繋ぐ」**(`_te_attached()`)という反転した設計に変更した。これなら新しい経路が
増えても安全側に倒れる。モジュール実体は `self._te_module` が保持し続けるので、
外しても解放されない。

layout/latents/timesteps の窓ではさらに `_pin_execution_device_to_compute()` で
vae も一時的に外す(TE を外した後の次の nn.Module が CPU 上の vae になり、今度は
`_execution_device` が cpu を返してしまうため。transformer が最初に見つかるようにする)。

### video VAE の fp16 化 (`H3_VIDEO_VAE_FP16`、既定 `0`)

`H3_VIDEO_VAE_FP16=1` で video VAE の重みだけを fp16 化する(9.70→4.85GB、デコード
ピーク 16.29→~11.4GB)。**audio VAE は絶対に触らない**(bf16化で-20dBの既知問題)。
- 品質: 全124フレームの平均PSNR **39.97dB**(min 39.08)で目視区別不能。デコード計算は
  元々autocast fp16のため重みfp16化の影響が小さい
- **実装の罠**: `AutoencoderKLMiniMaxH3._keep_in_fp32_modules` が encoder/decoder等を
  強制fp32に戻すため、`from_pretrained(dtype=fp16)` は効かない(実機確認)。ロード後に
  `.to(torch.float16)` を明示的に呼ぶ必要がある
- 既定OFFでは既存基準とMD5一致(回帰ゼロ)

## text_encoder

### text_encoder の未使用上位レイヤー削除 (`H3_TE_PRUNE`、既定 `0`、2026-08-06追加)

MiniMax-H3 の text_encoder(Qwen3-VL-32B、64層)は
`hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER]`(=50)しか読まない
(`diffusers/modular_pipelines/minimax_h3/encoders.py`)。`H3_TE_PRUNE=1` は
text_encoder を **51層だけ**(0〜50、`MINIMAX_H3_TEXT_ENCODER_LAYER + 1`)で構築し、
未使用の52〜64層目 + 最終`norm` + `lm_head`(重み換算で14層分、bnb-4bit実測 ~3.6GB /
bf16実測 ~13.6GB)を一度もロードしない。既定 `0` は完全無変更(この分岐自体が
一切呼ばれない)。

#### なぜ「50層」ちょうどではなく「51層」なのか(このタスクで発見・検証したtransformers側の罠)

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

#### 実装

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

#### 削除後TEの実測サイズ

| 精度 | 削除前 | 削除後(51層) | 削減 |
|---|---|---|---|
| bnb-4bit nf4 | 21.02GB | **17.45GB** | -3.57GB (-17%) |
| bf16 | 66.71GB | **53.06GB** | -13.65GB (-20%) |

nf4は量子化で1層あたりのサイズがbf16の約1/4に圧縮されるため、削減の絶対量もbf16より
小さい(相対削減率はほぼ同じ)。

#### MD5一致チェックの結果

同一seed(768²・5秒・30steps・キツネのプロンプト、seed=12345、`H3_CACHE=none`で
FBCの経路依存を排除)で、`H3_TE_PRUNE=0`(削除なし)と`H3_TE_PRUNE=1`(削除あり)の
出力を比較したところ、**t2va・ref2va(画像参照1枚、vision tower経由)とも
mp4がバイト完全一致(md5一致)**した。削除が数学的に無影響であることの実証。
`H3_LOWVRAM=1`・`H3_LOWVRAM=group`の各モードでも、削除の有無で出力が完全一致する
ことを確認済み(後述)。

#### 24GB級対応: 削除だけでは不十分だった(実機で発見・`H3_LOWVRAM_GROUP`側に追加修正)

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

#### `H3_LOWVRAM=1`(48GB級)でのTEロード時間短縮

削除により、`H3_LOWVRAM=1`が毎リクエスト支払うTEロード固定費が短縮される
(実機、ダミーVRAM確保で空きを~43.5GBに制限):

| | TEロード時間 | TEサイズ |
|---|---|---|
| 削除なし | 42.3s | 21.01GB |
| 削除あり | **35.0s**(-17%) | 17.44GB |

出力mp4は削除の有無でバイト完全一致(md5一致)。

#### 回帰確認

`H3_TE_PRUNE`未指定(既定`0`)の状態で同一条件のt2vaを実行し、この機能追加前に
取得していた基準mp4とバイト完全一致(md5一致)することを確認した。既定動作は
完全無変更。

### 量子化済み text_encoder のディスクキャッシュ (`H3_TE_PREQUANT`、既定 `1`、2026-08-08)

`H3_LOWVRAM=1` は毎リクエストで TE を再ロードするため、その時間がそのまま固定費になる。
この時間の大半は「元の bf16 重みを読む + その場で bnb-4bit へ量子化する」処理なので、
**量子化後の重みを一度保存しておけば次回以降は読むだけで済む**。初回ロード時に自動保存し、
以降はそこから読む。

**実測(RTX PRO 5000 48GB + `H3_LOWVRAM=1 H3_TE_PRUNE=1`、t2i turbo 4steps を各4回)**:

| | TE ロード | リクエスト合計 |
|---|---|---|
| キャッシュ無効(従来) | 46.5〜55.8s(平均 **53.0s**) | 108.7〜150.0s(平均 **128.6s**) |
| **キャッシュ有効(既定)** | 21.1〜34.3s(平均 **29.5s**) | 75.3〜90.6s(平均 **83.2s**) |
| 差 | **-23.5s(1.8倍)** | **-45.4s(-35%)** |

**等価性**: 同一 seed の生成物が **PNG で MD5 完全一致**(seed 3・4 で確認)。加えて
プローブ(`scripts/probe_prequant_equivalence.py`)で `hidden_states[50]` が
`torch.equal` でビット一致(max_abs_diff 0.0)することも確認済み。bnb-4bit の量子化は
決定的なので当然の結果だが、速いだけで採用しないのが本プロジェクトの流儀。

- キャッシュ先: `models/prequant/te_<quant>_prune<0|1>/`(.gitignore 済み)。
  **設定ごとに別ディレクトリ**にしてある — TE_QUANT / TE_PRUNE を変えると重みの中身が
  別物になるため、同じ場所を使い回すと設定切替時に古い重みを読む事故が起きる
- 消費ディスク: **17.44GB**(`H3_TE_PRUNE=1`)/ 未削除なら ~21GB
- `H3_TE_PREQUANT=0` で完全に無効化(導入前と同一挙動)。空きディスクが
  `H3_TE_PREQUANT_MIN_FREE_GB`(既定 25GB)を下回る場合は**保存をスキップして生成は続行**する
  (キャッシュは高速化であって機能ではない)。保存は一時ディレクトリへ書いてから rename
  するので、途中で落ちても中途半端なキャッシュが「有効」に見えることはない

**測定上の注意(この検証で踏んだ罠)**: 保存直後に測ると 17.44GB がページキャッシュに
載っているため **2.6秒**という非現実的な値が出る。実運用ではリクエスト間に transformer
(34GB)をディスクから読むため TE のページキャッシュが押し出され、実測は 21〜34秒に
落ち着く(リクエストを重ねるほど遅くなり 34秒前後で頭打ち)。**それでも従来経路の
46〜56秒より速い**が、プローブの数値をそのまま性能として報告してはいけない。

### 2026-08-10: 投影TE(Qwen3-VL-4B + 学習済み線形写像)— 実装と実測、未検証項目

`H3_TE_PROJ` として実装(既定OFF、既存挙動は1バイトも変わらない)。TE を Qwen3-VL-32B から
**Qwen3-VL-4B + 学習済み投影行列**([ClipProj-MiniMax-H3](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3))へ
置き換える経路。狙いは48GB機で TE と transformer を**同時常駐**させ、載せ替えの固定費を消すこと。

```
cond = ((h - mean_in) / std_in) @ W * std_out + mean_out     # h = 4B の hidden_states[24]
cond[:, 0] = sink_out                                        # token 0 はアテンションシンク
```

#### 実測(96GB機、t2i 768²・30steps・seed 4242・同一プロンプト)

| | 32B TE(現行) | **投影4B TE** |
|---|---|---|
| PSNR | — | **22.64 dB** |
| 鮮鋭度(ラプラシアン分散) | 48 | **37**(-23%) |
| 生成時間 | 65.7s | **43.7s** |
| ピークVRAM | 88.73GB | **76.1GB** |
| **TE の GPU 実占有** | 21.02GB (nf4) | **8.88GB** |

**品質は「同じ絵」ではなく「同等品質の別の絵」。** 構図・色調・時間帯の解釈は一致するが、
細部の指定は落ちる(実例: プロンプトにない範囲だが、32B版にあった手前の睡蓮の葉と草が
投影版では水面に置き換わった)。破綻はなく実用水準。これは投影の性質(配布元の報告で
test cosine 0.712)どおりで、**プロンプトの大意は保持し細部は失う**と理解するのが正確。

**この変更はこれまでの全最適化と種類が違う。** 量子化・常駐制御・turbo などは全て同一seed
MD5一致で「数学的に無影響」を証明できたが、投影TEは**原理的に近似**なので MD5 は使えない。
判定は PSNR + 目視 + VRAM 実測による。

#### トークナイザの調査結果(実装方針の根拠)

| 対象 | 結果 |
|---|---|
| 通常テキスト・日本語 | **ID 完全一致**(H3 側のトークナイザをそのまま使える) |
| `<d>` / `</d>` | H3 は 151669/151670 の**単一トークン**、4B は語彙外で分解される |
| `<\|cutoff\|>` `<\|lyrics_*\|>` `<\|caption_*\|>` | H3 固有(151671〜675)だが**公式ガイドも本アプリも未使用** |
| `[Shot n]` `<cutoff>` `<scenetrans>` | **通常テキスト**(特殊トークンではない)。マルチショット構成は影響なし |

→ 実使用で影響を受けるのは**台詞タグ `<d>` だけ**。実装では id>=151669 を含むプロンプトを
`ValueError` で**明示的に拒否**する(黙って別物を送らない)。台詞は音声参照(`fully_copy`)で
入れられることを同日に実証済み(上記セクション参照)なので、運用上の回避策がある。

#### 2枚目GPUの要件が下がる

`docs/RESIDENCY.md` §5.4 の導出式に実測 8.88GB を入れ直すと、`H3_TE_DEVICE` 用のカード要件が
**20GB → 16GB(おそらく12GB)** に下がる。

| 2枚目GPU | 実効予算 | 32B TE (17.45GB) | 投影4B TE (8.88GB) |
|---|---|---|---|
| 12GB | ~10.5GB | 不可 | **成立見込み**(必要 ~9.2GB、余裕 ~1.3GB) |
| 16GB | ~14.5GB | 不可 | **成立見込み**(余裕 ~5.3GB) |
| 20GB | ~19.7GB | t2va のみ成立・ref2va は OOM | 成立(余裕 ~10.5GB) |

**12GB は実機確認なしに断言しない。** 以前 20GB で「導出上は入るはずが実測 OOM」を
経験しており(単位の罠、§5.1)、余裕1.3GB は薄い。

#### 追記(同日): NF4 量子化オプションと動画での品質確認

**`H3_TE_PROJ_QUANT`**(既定 `none` / `bnb-4bit` / `bnb-8bit`)を追加した。4B 自体を
量子化する **4B 専用フラグ**で、32B 用の `H3_TE_QUANT` とは別物(排他ガードの意図を参照)。

**投影行列は bf16 用のものをそのまま使うのが正解**(実測)。量子化による条件付けのズレは
NF4 で相対RMS 0.61〜0.96%(cosine 1.0000)。配布元の int8_convrot 用行列を使うと
かえってズレが増える(1.02〜2.97%)— あれは ComfyUI の量子化方式専用の校正なため。

| 構成 | TE 常駐 | t2i (30steps) | t2va 5秒 (30steps) | ピーク(t2va) |
|---|---|---|---|---|
| 32B TE | 21.02GB | 65.7s | 162.1s | 91.9GB |
| 投影4B bf16 | 8.88GB | 43.7s | — | — |
| **投影4B NF4** | **3.11GB** | **33.5s** | **143.5s** | **74.3GB** |

**動画でも品質確認済み**(768²・5秒・124frames・同一seed 555):
静止画で見えた鮮鋭度低下(-23%)は動画では出ず(187 vs 191)、**ちらつき(2階差分)は
むしろ 11% 少ない**(7.51 → 6.67)。bf16 vs NF4 の直接比較は PSNR 34.45dB で量子化の
影響はほぼゼロ。32B との PSNR 14.98dB は「劣化」ではなく「同じ指示の別テイク」
(32B版には対向者が現れ、投影版は少女単独、など解釈の分岐)。

**設定APIとの関係**: 投影TEは現状 **env 専用**(`H3_TE_PROJ`/`H3_TE_PROJ_QUANT`)。
`/api/settings/apply` から te_quant/te_prune を変えようとすると、runner の import 時
ガードを素通りした上で**適用されないのにスナップショットは変わったと報告する**穴が
あったため、投影TE有効時はこれらのフィールドを 400 で拒否するようにした
(`/api/settings` のスナップショットに読み取り専用の `te_proj`/`te_proj_quant` を追加)。

#### 追記(同日その2): NF4 を既定化、UI からの切り替えに対応

- **`H3_TE_PROJ_QUANT` の既定を `bnb-4bit` に変更**(品質実測に基づく。none/bnb-8bit は
  明示指定で選択可)。既定変更に伴い「量子化指定あり+投影OFF」ガードは**明示指定
  (`in os.environ`)のときだけ**発火するよう修正 -- 既定値のままの通常ユーザーを
  誤って落とさないため
- **UI の再ロード設定パネルから投影TEを切替可能に**(`/api/settings/apply` の
  `te_proj` / `te_proj_quant`)。ON のとき te_quant/te_prune のコントロールは無効化され、
  API 側も 400 で拒否する(判定は「適用後の値」ベース: OFF に戻しながら te_quant を
  変えるのは合法)

**往復リロード E2E(実測、96GB機)**: OFF→ON 47.9s / ON→OFF 29.4s / 再ON 27.3s。
各段で同一 seed 生成の PNG MD5 を照合し、**UI 経路の ON = env 経路の NF4 と完全一致**、
**OFF = 32B ベースラインと完全一致**、**再 ON = 1回目の ON と完全一致**(状態残留なし、
投影行列の再ロード正常)。ON 中の te_quant 変更は 400。

#### 未検証(重要)

- **48GB機での TE+transformer 同時常駐** — **本命**。8.88 + 34(int8) = 42.9GB で載る見込みだが
  未実測。これが成立して初めて導入の意味がある(96GB機では元々同時に載るため差が出ない)
- **参照経路(ref2va)** — 投影行列は**テキストのみで校正**されており、vision 特徴が正しく
  写るかは未知。実装は動くようにしてあるが、一度だけ `logger.warning` を出す
  → **2026-08-11 に目視検証済み: 参照は明確に効く**(顔・髪・服・小物まで反映、32B 正解と
  同水準。「2026-08-11: ref2va/i2va はどこまで動くか」の節参照)。warning は校正の事実の
  記録として残置
- **音声参照 + `fully_copy` との組み合わせ**(台詞を音声から入れる運用が投影TEでも成立するか)
- **4B の TE ロード 80.5s**(初回DL込み)。定常のロード時間は未測定

## transformer とデノイズ

### transformer int8量子化 (`H3_TRANSFORMER_QUANT`、既定 `none`)

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

#### torchao の TF32 副作用の抑止 (`set_inductor_config=False`、2026-10-01)

torchao の `quantize_()` は config の `set_inductor_config=True`(既定)だと
`recommended_inductor_config_setter()` を呼び、`torch.set_float32_matmul_precision("high")`
(fp32 matmul の TF32 化)等を**プロセス全体**に立てる。fresh 量子化を行ったプロセスだけ
以後の fp32 計算(audio_vae 等)の数値が変わり、キャッシュ読み込みだけのプロセスと出力が
一致しなくなるため、本リポジトリの全 config(上記 `Int8WeightOnlyConfig(version=2)` と
`core/pruned.py` の各方式)は `set_inductor_config=False` で作る。量子化済み重みは
変わらず(フラグの副作用だけ)、キャッシュの `meta.json` も不変なので既存キャッシュは
そのまま使える。`/api/status` の `float32_matmul_precision` が `"highest"` のままで
あることで観測できる。

### FirstBlockCache によるデノイズ高速化 (`H3_CACHE`、既定 `fbc`)

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

### Sage Attention (`H3_ATTN_BACKEND`、既定 `sage`)

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

### Turbo LoRA (`H3_TURBO_LORA`、既定 `0`、2026-08-06 / **2026-08-08 lightx2v 版へ切替**)

> **2026-08-08 更新: 既定 LoRA を lightx2v 版へ切り替え、48GB(int8/低VRAM)でも
> turbo が使えるようになった。**
>
> - 既定: `H3_TURBO_LORA_REPO=lightx2v/Minimax-h3-Turbo` /
>   `H3_TURBO_LORA_FILE=minimax_h3_fl2v_turbo_4step_v0.1.safetensors`(DMD蒸留、
>   Apache 2.0、rank128・312 Linear対象)。既定ステップ数は形式連動
>   (lightx2v=4 / Ostris に戻すと 8)
> - **int8 対応の理由**: キーが diffusers ネイティブ(to_q/to_k/to_v 分離)で
>   `fuse_projections()` 不要 → `torch.cat` を呼ばない → `Int8Tensor` 非互換を
>   踏まない。適用関数は形式自動判別(`detect_turbo_lora_format`、comfy 署名
>   `qkv_proj` を先に見る — Ostris 版も `token_refiner.` キーを持つため順序が重要)
> - **適用係数**(`H3_TURBO_LORA_SCALE`、空=形式別の実測既定): lightx2v は **0.094**
>   (Kijai 記載の 0.75 は ComfyUI の alpha 折り込み前提。生の B・A に 0.75 を掛けると
>   30steps でも完全ノイズ化 — 強度スイープはスパイク節参照)。Ostris は 1.0 のまま
>   (scale=1.0 は恒等なので旧挙動と bit 一致)
> - **E2E 実測**(RTX PRO 5000 48GB + `H3_LOWVRAM=1`、768²): t2va 4steps
>   **総所要 143s / デノイズ 29s**(非turbo 30steps は 351s)。**t2i(静止画)×turbo は
>   デノイズ 5.0s / 総所要 94s**。本実装経路の出力はスパイク出力と **mp4 md5 完全一致**。
>   turbo 連続2本で lowvram の再ロード後の再適用も正常、turbo=0 で通常経路(FBC 有効)
>   へ正しく復帰
> - **既知の性質**: turbo 時は音声レベルが非turbo比で大きめ(rms 0.018-0.042 vs
>   0.007。白色雑音ではないが強度によっては peak が 1.0 に近づく)
> - **併用制限**: `H3_LOWVRAM=group` とは形式を問わず併用不可(`enable_group_offload`
>   の `cpu_param_dict` が有効化時点で固定され、後から追加する LoRA バッファが
>   offload サイクルから欠落するリスク — 未検証のまま解禁しない)。comfy 形式
>   (Ostris)× int8 は従来どおり不可(リポジトリ名で予備判定+適用時に実ファイルの
>   キーで最終判定)。ref2va への適用(transformer_ref)は未検証のまま
> - v0.1(2026-08-07 公開)のため引き続き**既定OFF**(UI/リクエストの turbo で opt-in)

以下は Ostris 版 (comfy 形式) 時代の記録(bf16 経路では現在も有効):

`H3_TURBO_LORA_REPO=larryvrh/MiniMax-H3-Turbo-Lora` で Ostris氏学習中の4/8ステップ
蒸留LoRA(Apache 2.0、rank64・259 Linear対象)を適用し、既定ステップ数を8にする。
**プレビュー版LoRA(学習途上)のため既定OFF**。完成版が出たら再評価する。

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

#### turbo × 他機能の組み合わせ検証(2026-08-06)

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

### ステップ数の指針(蒸留モデル、実測 2026-08-05)

`num_inference_steps` はAPI/UIパラメータ。30(検証既定)に対し **20で-15%、16で-31%** の
デノイズ短縮。16/20とも単フレーム品質・時間方向の安定性に破綻なし(ただし構図は
ステップ数で変わる)。ドラフト用途は16-20、本番は30が目安。ステップを減らすと
FBCのスキップ機会も減る(30stepsで7スキップ→16stepsで0)ため、効果は単純比例しない。

### 2段生成による2xアップスケール (`/api/t2va` の `upscale=1`、既定OFF)

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

### 任意サイズ・秒数の丸め (2026-08-06)

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

## 生成モード

### Ref2VA (オムニ参照生成、`/api/ref2va`)

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

### 静止画モード (T2I、`/api/t2i`、2026-08-07 本実装)

H3を「超短尺動画→静止画取り出し」で画像生成の代用にするモード。前段の超短尺プローブ
(下記)の結論をそのまま本実装した。価値は「**H3と画風が完全に一致する静止画**」を
FL2VA の先頭フレームや Ref2VA の参照画像として作れること(速度は固定費支配のため
専用T2Iモデルには勝てない)。

- **API**: `POST /api/t2i` — `prompt` / `frames`(22既定 or 5)/ `resolution` or
  `height`+`width` / `num_inference_steps` / `seed` / 即反映パラメータ
  (cache/cache_threshold/attn/turbo)。レスポンスは超短尺 mp4 と**中央フレームの PNG**
  (`image_url`、`t2i_<ts>.png`)の両方
- **UI**: 静止画段の「T2I」タブ(秒数・upscale欄は非表示、フレーム数選択のみ)。
  ギャラリーにも PNG がタイル表示される(削除可・連結不可)
- **フレーム数**: 22 (0.917s) が既定。5 (0.208s) は実験的だが下記の VAE 修正で動く
- **実装の3点セット**(`core/runner.py`):
  1. `generate(still=True, still_frames=...)` — diffusers 側の最小尺5秒バリデーション
     (`MINIMAX_H3_MIN_DURATION`)は `MiniMaxH3SetupStep` の呼び出し1回分だけ
     `_relaxed_min_duration()` で緩和(生成は generation_lock で直列なので漏れない)
  2. **VAE 小クリップデコード修正** (`H3_VAE_SMALLCLIP_FIX`、既定1):
     `AutoencoderKLMiniMaxH3._decode()` は潜在1-2フレームで num_chunks=0 →
     `torch.cat([])` で落ちる上流バグがある(プローブで実機再現)。runner 側の
     monkeypatch で「1チャンク未満なら全トークンを単一 `_decode_clip()` で復号し、
     `frame_pre_padding`(3) とパディング末尾を切り落とす」分岐を追加。潜在2フレーム
     ×4 − 3 = 5ピクセルフレームで幾何が一致する。通常動画 (num_chunks>=1) は元実装へ
     そのまま委譲するため影響なし(venv の diffusers 本体は無変更)
  3. **decode 例外時の steady state 復元**: プローブで観測した「decode 例外 →
     transformer drop 済みのまま復元コード未実行 → ~98.5GB 残留で後続リクエストが
     連鎖 OOM」への対策として、`generate()`/`generate_ref2va()` の decode 部を
     try/except 化し、例外時も `_restore_decode_steady_state()`(VAE を CPU へ・
     モードごとの transformer/TE 再ロード)を実行してから再送出する。
     **実機検証済み** (2026-08-07): 一回限りの失敗注入フック `H3_DEBUG_FAIL_DECODE=1`
     (`os.environ.pop` で読むため同一プロセスの次リクエストから通常動作)で decode を
     故意に失敗させ、500 応答後に GPU が ~2GB まで解放され、**同一プロセスの次の
     リクエストが正常に 200 を返す**ことを確認した

**実測 (2026-08-07、RTX PRO 5000 48GB + `H3_LOWVRAM=1`、768²・30steps・fbc+sage・
seed 12345)**:

| frames | デノイズ | デコード | ピークVRAM | 合計(毎リクエストの再ロード込み) | 品質(目視) |
|---|---|---|---|---|---|
| 22 (0.917s) | 29.0s (1.0s/step) | 1.8s | 35.0GB | 157s | 破綻なし・高品質 |
| 5 (0.208s) | 9.1s | 0.67s | 35.2GB | 125s | 破綻なし(史上初の実デコード成功) |

5フレームの音声 (0.208s) も非無音 (rms 0.055)。

#### 連続生成 (物語の場面画像などプロンプト違いの連番) の実測: **group モードは静止画に不利**

「常駐モード (`H3_LOWVRAM=group`) なら再ロード固定費が消えて連続生成が速いはず」という
仮説を、物語3場面のプロンプト (frames=22・768²・30steps・seed固定) の3連続実行で実測した
(2026-08-07、RTX PRO 5000 48GB)。**結果は逆で、lowvram=1 より約1.5倍遅い**:

| モード | 1枚目 | 定常 (2枚目以降) | 定常の内訳 |
|---|---|---|---|
| `H3_LOWVRAM=1` | 157s | **~157s** | TE+transformerロード ~120s + デノイズ 29s + デコード 2s |
| `H3_LOWVRAM=group` | 310s (TEコールド込み) | **~240s** | デノイズ **130s** + デコード 2s + デコード後のTE再ロード **75-97s** |

- **デノイズ 29s → 130s の悪化が本質**: group offload は1ステップごとに int8
  transformer 全 ~34GB をブロック単位で PCIe 転送する。転送コストはシーケンス長に
  無関係な固定費なので、計算が軽い超短尺 (22フレーム) では転送が完全に支配する
  (124フレームの通常動画では計算時間に隠れて相対的に軽い、という設計前提が
  静止画では崩れる)
- group モードはデコード窓で TE を force-free して後で再ロードする(24-32GB級の
  ヘッドルーム設計)。この箱では TE NF4 ロードが実測 75-97s かかる(96GB時代の
  15-40s より大幅に遅い)ため、この再ロードも定常コストに乗る
- 品質は lowvram=1 と同様に良好(目視)。ピークVRAMは 25.1GB
- **結論**: この箱での静止画の連続生成は **`H3_LOWVRAM=1` のまま回すのが最速
  (~157s/枚)**。さらに縮めるのは下の `/api/t2i_batch`(この実測を受けて実装済み)

#### 静止画のバッチ生成 (`/api/t2i_batch`、2026-08-07 本実装)

上の実測を受けて、`H3_LOWVRAM=1` の固定費 (~110s: TEロード75-97s + transformerロード
~20-35s) を**バッチ全体で1回に償却**するエンドポイントを実装した
(`core/runner.py` の `generate_still_batch()`)。generate() の lowvram choreography を
位相順に並べ替える:

    entry   : [nothing big resident]
    encode  : [TE-nf4]         全場面の setup/エンコード/layout/latents/timesteps
    denoise : [transformer]    全場面を順にデノイズ
    decode  : [vae pair]       全場面を順にデコード → PNG/mp4 保存(場面ごとに保存
                               しながら進むので途中失敗でも完了分は残る)

- **API**: `POST /api/t2i_batch` — multipart で `prompts` を場面数ぶん繰り返し送る
  (最大24場面)。`frames`/`resolution`/`num_inference_steps`/`seed`/即反映パラメータは
  全場面共通(変えられるのはプロンプトのみ)。UI は T2I タブの
  「バッチ連続生成(1行=1場面)」チェックで、プロンプト欄の1行を1場面として送る
- **場面間の共有状態リセット**(実装の要): スケジューラは sigmas/timesteps の値が
  全場面同一(同じ幾何・ステップ数)なので、デノイズ直前に `_step_index = None` に
  戻すだけでよい(`MiniMaxH3Scheduler.step()` は index を timestep 値から再導出する)。
  FirstBlockCache は場面ごとに `_reset_stateful_cache()` + `cache_context`
  (generate() の per-request リセットの per-scene 版)
- **等価性の実証**: 同一プロンプト・seed の逐次 `/api/t2i` とバッチの場面1で
  **mp4 と PNG がバイト完全一致**(md5一致)。位相並べ替えは数学的に無影響
  (なお group モード出力とは一致しない — FBC の効き方が実行経路で変わる既知の現象で、
  バッチ固有の差ではない)
- `H3_LOWVRAM=1` 以外のモード(大モデル常駐)では位相並べ替えの利得がないため、
  同じ API のまま逐次 generate() にフォールバックする(レスポンス形式は同一)

**実測 (2026-08-07、RTX PRO 5000 48GB + `H3_LOWVRAM=1`、3場面・frames=22・768²・30steps)**:

| 方式 | 合計 | 1枚あたり | 内訳 |
|---|---|---|---|
| 逐次 `/api/t2i` ×3 | ~471s | **157s** | 毎回 ロード~120s + デノイズ29s + デコード2s |
| `/api/t2i_batch` (3場面) | **202.6s** | **67.5s** | ロード~110s(1回) + エンコード0.9s + デノイズ84.8s + デコード6.9s |

場面追加の限界コストは **~31s/枚**(デノイズ28-29s + デコード2.3s)なので、場面数が
増えるほど1枚あたりは 31s に漸近する(10場面で ~42s/枚、24場面で ~36s/枚の計算)。
ピークVRAMは 35.0GB で1枚生成と同じ(場面ごとに増えるのは潜在+prompt_embedsの数十MBのみ)。

#### 参照付き静止画 (ref2i、`/api/ref2va` の `still=1` と `/api/ref2i_batch`、2026-08-07 本実装)

下記スパイクの成立を受けて本実装した。**キャラクター参照から場面ごとの一貫した
静止画**を作るモード。作った静止画は FL2VA の先頭フレームや次の Ref2VA の参照に回せる
(物語の複数動画生成の素材作り)。

- **単発**: `POST /api/ref2va` に `still=1` + `frames`(22既定|5)を追加で渡す。
  `seconds` は無視され、中央フレーム PNG (`ref2i_<ts>.png`) がレスポンスの
  `image_url` に付く。実装は t2i と同じ3点セット (setup step の間だけの尺ゲート緩和・
  VAE 小クリップ修正・既存の decode 例外クリーンアップ) の流用
- **バッチ**: `POST /api/ref2i_batch` — 共通の `references` + `prompts`(最大24場面)。
  `H3_LOWVRAM=1` では `generate_ref2i_batch()`(t2i_batch と同じ位相並べ替えの
  ref2va 版: TE常駐で全場面エンコード → VAE窓1回で全場面の参照VAEエンコード →
  layout/timesteps → transformer_ref 1回ロードで全デノイズ → まとめてデコード)。
  他モードは逐次フォールバック。スケジューラ/FBC の場面間リセットは t2i_batch で
  md5 一致まで実証済みの同じ手法
- **UI**: 静止画段の「Ref2I」タブ(2026-08-09 に Ref2VA タブ内の「静止画」チェックから
  独立タブへ移動。下記「タブの2段組み」参照)+「バッチ連続生成(1行=1場面)」チェック

**実測 (2026-08-07、RTX PRO 5000 48GB + `H3_LOWVRAM=1`、赤ずきん参照1枚・3場面・
frames=22・768²・30steps)**: 合計 494.7s = **164.9s/枚**(逐次 ~265s/枚の1.6倍速)。
内訳: エンコード位相 212.5s + デノイズ 178.1s (57.7-60.2s/場面) + デコード 6.6s。
ピークVRAM 36.8GB。品質・キャラ一貫性とも良好(3場面とも赤マントを保持し
各プロンプトに追従)。

- ~~既知の改善余地: 参照ビジョンエンコードが場面ごとに走る~~ → **下記
  「参照プレフィックスの KV キャッシュ共有」で解決済み (2026-08-08)**

#### 参照プレフィックスの KV キャッシュ共有 (`H3_REF_PREFIX_CACHE`、既定1、2026-08-08 本実装)

ref バッチ (`generate_ref_batch` = ref2i_batch/ref2va_batch) のエンコード位相で、
参照ラベル+ビジョン (~4104トークン、~65s/場面) の Qwen3-VL エンコードが場面ごとに
重複していた問題の解消。ref2va のトークン列は「参照が前置・プロンプトは末尾に verbatim
追記」(packing_ref2va の `build_ref2va_presentation` で確認) で、条件付け元の Qwen3-VL は
因果 LM なので、**参照プレフィックスのテキスト表現はプロンプトに依存しない** —
プレフィックスを1回だけ `use_cache=True` で通して `DynamicCache` に焼き、場面ごとには
プロンプト末尾 (14-33トークン、~0.2s) だけをキャッシュ継続する
(`core/runner.py` の `_encode_ref_prompts_shared_prefix()`)。

**検証 (`scripts/probe_ref_prefix_cache.py`、transformers 5.14.1 の
modeling_qwen3_vl.py を読んだ上で実機実測)**:

- プレフィックス部分の hidden_states[50] はフル計算と**ビット一致** (`torch.equal`)
- プロンプト末尾部分は相対RMS ~1.5% の丸め差が残る。原因はカーネル/GEMM のタイル経路が
  系列長で変わること — eager 固定でも同水準 (sdpa 固有ではない)。**ロジックバグでない
  ことはネガティブコントロールで確定**: わざと位置オフセットを壊した継続は相対RMS
  27-30% (20倍) に跳ねる。1.5% は「正しい計算の丸めノイズ」の水準
- 継続呼び出しのレシピ (罠3点): `attention_mask=None`
  (`compute_3d_position_ids` の arange 分岐 = past長からの連番 + rope_deltas 加算に
  乗せる)、`mm_token_type_ids`/`pixel_values`/`grid_thw` 系は**全て None**
  (`image_grid_thw` を渡すと `model.rope_deltas` が再計算・上書きされる)、場面ごとに
  `DynamicCache.crop(prefix_len)` で切り戻して直列再利用。`rope_deltas` は
  Qwen3VLModel の**インスタンス状態**なので、プレフィックス→全継続の間に他の TE 呼び出しを
  挟んではならない (ヘルパー1呼び出し内で完結させる設計)

**E2E 実測 (ref2i_batch、赤ずきん参照1枚・同一3場面・seed 12345・768²・30steps)**:

| | エンコード位相 | 合計 | 1枚あたり |
|---|---|---|---|
| 共有なし (H3_REF_PREFIX_CACHE=0 相当) | 212.5s | 494.7s | 164.9s |
| **共有あり (既定)** | **83.1s** | **350.1s** | **116.7s (-29%)** |

品質: 共有なしの同一 seed 出力との PSNR 21.9-27.4dB。バッチ出力は従来経路と
ビット一致しない (プロンプト末尾側の ~1.5% 丸め差が軌道をドリフトさせる —
Sage Attention 既定化時と同種・同水準の epsilon 級ドリフト) が、目視で構図・品質・
キャラ一貫性とも同等。ビット再現が要る対照実験は `H3_REF_PREFIX_CACHE=0` で従来経路に
戻せる。ref2va_batch (動画) もエンコード位相は同一コードなので同じ削減 (~130s/3場面)
がそのまま効く。参照VAEエンコード (~数s/場面) は共有せず場面ごとのまま
(効果が小さく、状態の別名参照リスクを増やさないため)。

#### 単発リクエスト間でのプレフィックス再利用 (`H3_REF_PREFIX_CACHE_SINGLE`、既定0、2026-08-24)

上の共有プレフィックスを、バッチだけでなく **単発 `/api/ref2va` のリクエスト間** でも
持ち越す。同じ参照画像に別々の音声・別々のプロンプトを当てる運用(mv_studio の
リップシンク生成がまさにこれ)で、~52s の参照ビジョンエンコードが2回目以降まるごと消える。

**既定 OFF**。プロセス寿命の KV キャッシュ(実測 1.04GiB VRAM)をリクエストをまたいで
持つため、「既存挙動を変えない」を優先している。有効化は gateway の overrides:
`{"backend":"h3","preset":"96gb","overrides":{"H3_REF_PREFIX_CACHE_SINGLE":"1"}}`。

**成立条件(コードが機械的に判定し、外れたら黙って従来経路へ落ちる)**:

- **画像参照のみ**(音声参照は何本あってもよい)。音声は `_build_presentation` が
  `"<Audio j>: "` のラベルしか出さず波形が conditioner に届かないため、**音声の中身も
  長さも変わってプレフィックスのトークン列はビット単位で不変**。
- **動画参照が1つでもあると使わない**(動画はピクセルがプレフィックスに入るため)。
  この場合はキャッシュを破棄して従来のフル計算に落ちる。

**キー**はファイル名ではなく「実際にプレフィックス forward へ入る値」
(トークン列 + 画像テンソルのバイト列 + `image_grid_thw`)の SHA-256。参照の枚数・順序・
解像度・ピクセルの中身のどれが変わっても必ず別キーになる(キー計算は実測 0.14-0.35s)。

**バッチ版に無い安全装置(`_encode_ref2va_prompt_prefix_cached` の docstring 参照)**:

1. `model.rope_deltas` は `Qwen3VLModel` の**インスタンス状態**で、プレフィックスが書き
   継続が読む。リクエストをまたぐと間に t2va のエンコードや `/api/prompt/enhance` が
   挟まって上書きされうるので、プレフィックス時の値を clone して持ち、**継続の直前に
   必ず書き戻す**。
2. `weakref` で text_encoder の同一性を見る。TE が解放/再ロード/量子化変更されたら無効。
   `_free_text_encoder()` 側でも能動的に捨てる(VRAM を即返すため)。
3. 継続のたびに `DynamicCache.crop(prefix_len)`(例外で抜けた場合の保険として、継続前にも
   長さを確認して切り戻す)。

**実測(96GB 機、preset=96gb + `H3_TRANSFORMER_QUANT=int8`、768x448・8秒・turbo・
4steps・seed 777、画像1枚+音声1本)**:

| | 1本目 | 2本目 | ピークVRAM |
|---|---|---|---|
| OFF(既定・従来経路) | 136.6s | **94.5s** | 73.79GB |
| **ON** | 140.4s(MISS、プレフィックス 52.5s) | **41.9s (-56%)** | 74.92GB (+1.1GB) |

- 同じ画像+**別音声**(8.0s→6.6s の別ファイル)も HIT: 37.0s。
- 参照画像を差し替えると `cache cleared` → MISS(93.6s)→ 元に戻すとまた MISS(94.1s、
  ただしキーは元の値に戻る=キーは決定論的)。
- ref バッチ経路も resident モードでは逐次 `generate_ref2va()` なので恩恵を受ける:
  ref2i_batch の2場面目が 71s → **21s**。

**数値**: MISS(その場でプレフィックス計算)と HIT(再利用)の出力は**ビット一致**
(192フレーム・音声とも `max diff = 0`)。つまりリクエスト跨ぎの再利用そのものは
完全に等価。一方、従来経路(丸ごと1本でエンコード)とはビット一致しない — 
プレフィックス/継続に分割することによる ~1.5% の丸め差が turbo 4steps で増幅されるためで、
バッチ経路(`H3_REF_PREFIX_CACHE`)が既に受け入れているのと同じ性質・同じ原因:

| 比較 | 平均画素差 | PSNR | 音声相関 |
|---|---|---|---|
| ON(HIT) vs OFF ※入力が完全同一のケース | 2.46/255 | 31.7dB | 0.9909 |
| **ON(HIT) vs OFF ※本来の用途(同じ画像・別音声)** | **12.14/255** | **20.45dB** | 音声 max diff 24943 |
| ON(MISS) vs ON(HIT) | **0(ビット一致)** | ∞ | 1.0 |

**2行目が本機能の狙いのケース**で、1行目より 11dB 悪い(画素の 67% が差 > 2)。
目視では構図・キャラクター同一性・口の形まで区別がつかず、これは劣化ではなく
**軌道の分岐**(同等品質の別の絵)だが、31.7dB だけを見て「ほぼ同一」と判断しないこと。

「MISS と HIT がビット一致」= **リクエストを跨いだ再利用そのものは完全に等価**。
差は「プレフィックス/継続に分割したこと」だけに由来する。

ビット再現が要る対照実験は `H3_REF_PREFIX_CACHE_SINGLE=0`(既定)のままにすること。

**H3_LOWVRAM 系との関係**: lowvram は毎リクエスト TE を解放するので、キャッシュはその
たびに捨てられる(= 効果なし)。実機で 48gb-lowvram + ON を確認済み: 毎回
`cache cleared: text_encoder freed` → 次リクエストは MISS、VRAM リークも例外もなし
(TE prune 併用の 51 層 TE ではキャッシュサイズも 0.84GiB に自動追従する)。

#### 参照付き動画のバッチ生成 (`/api/ref2va_batch`、2026-08-08 本実装)

ref2i と同じ位相並べ替え(実装は同一メソッド `generate_ref_batch(still=False)`)を
**通常尺の動画**に適用したもの。物語の各場面の動画を、共通参照+プロンプト違いで
連続生成する。尺 (`seconds`) は全場面共通で必須(スケジューラの sigmas/timesteps 値の
場面間同一性が位相並べ替えの前提のため。音声参照からの尺自動導出もバッチでは不可)。
UI は Ref2VA / Ref2I タブの「バッチ連続生成(1行=1場面)」チェック(タブと独立に
使える: Ref2I+バッチ = ref2i_batch、Ref2VA+バッチ = ref2va_batch)。

**実測 (2026-08-08、RTX PRO 5000 48GB + `H3_LOWVRAM=1`、赤ずきん参照1枚・2場面・
5秒・768²・30steps)**: 合計 803.2s = **401.6s/本**(逐次 ~485s/本の17%短縮@2場面)。
内訳: エンコード位相 151.3s + デノイズ 503.4s (247.9-255.5s/場面) + デコード 25.4s。
ピークVRAM 40.5GB。場面追加の限界コストは ~330s/本(ビジョンエンコード ~65s +
デノイズ ~250s + デコード ~13s)なので、場面数が増えると**約32%短縮**に漸近する。
動画はデノイズ支配のため静止画バッチほど劇的ではないが、品質・キャラ一貫性は
逐次と同様に良好(2場面とも赤マントを保持しプロンプトに追従)。

#### スパイク: Ref2VA×超短尺 (参照付き静止画、2026-08-07、実測済み → 上記の通り本実装済み)

「t2va で検証済みの22フレーム超短尺が、参照パッキング (packing_ref2va: 参照行が
生成行より多くなる) と重なっても品質が崩れないか」を実測した
(`scripts/probe_ref2va_short.py`、本体コード無変更・プローブ内 monkeypatch のみ)。
成立すれば**キャラクター一貫の場面静止画**が量産でき、物語の複数動画生成の素材作りが
速くなる。

条件: 参照 = 本リポジトリの t2i 生成物 (赤ずきんの少女 PNG) 1枚、同一プロンプト
(`<Picture 1>` 参照付き)・seed 12345・30steps・768²。RTX PRO 5000 48GB +
`H3_LOWVRAM=1`。

| 条件 | デノイズ | デコード | ピークVRAM | wall (逐次・ロード込み) | 品質(中央フレーム目視) |
|---|---|---|---|---|---|
| short22 (0.917s) | 58.1s | 1.7s | 36.7GB | 265s | **破綻なし・高品質** |
| baseline (5.0s) | 243.1s | 10.6s | 40.7GB | 485s | 破綻なし(アンカー) |

- **判定: 成立 (GO)**。分布外の重なり (超短尺×参照パッキング) でも品質は崩れず、
  参照の衣装・小物 (赤マント・ランタン・白いドレス)・画風は short22 でも保持された。
  構図もプロンプト (苔むした岩に座る) に正しく追従
- デノイズは t2va の22フレーム (29s) の約2倍 (58s) — 参照行のぶんパック列が伸びる
  ため。それでも 5 秒基準の 1/4
- 顔の同一性の厳密さは参照画像の写り (この実験では後ろ姿気味) に依存する。これは
  参照の性質であって超短尺の欠陥ではない
- 本実装 (API/UI 化、`generate_still_batch` の ref2va 版) は**未着手** — t2i と同じ
  3点セット (尺ゲート緩和・VAE 小クリップ修正・例外クリーンアップ) は流用可能で、
  バッチ位相並べ替えも t2i_batch と同じ手法 (スケジューラ `_step_index` リセット +
  per-scene FBC リセット、参照が全場面共通なら参照エンコード1回共有) が使える見込み

#### 前段の超短尺プローブ (2026-08-07、実測記録)

フレーム数は 17n+5 刻み(最小5=0.208秒、次が22=0.917秒)。プローブは monkeypatch のみで
検証した(`scripts/probe_short_frames.py` + `_one.py`)。当時の環境は RTX PRO 6000 96GB・
既定モード。

| フレーム数 | デノイズ(30steps) | デコード | 結果 |
|---|---|---|---|
| 5 (0.208s) | 完走 | **失敗** | VAEチャンク分割デコードの境界バグ(潜在2フレームで num_chunks=0 → `torch.cat([])`)→ **上記2で修正済み** |
| **22 (0.917s)** | **13.5s**(124fの1/7.6) | 1.2s | **成功・品質は5秒基準と遜色なし**(目視確認済み) |
| 124 (5s、基準) | 102.2s | 6.3s | 成功 |

- 22フレームは学習分布外(公式下限5秒の1/5)でも品質が崩れず、音声も非無音
- **教訓**: プローブ中、5フレームの例外がデコード後のクリーンアップ(transformer再ロード)
  前に発生し、~98.5GB残留で後続リクエストがOOMする連鎖を観測 → 上記3のクリーンアップ
  経路として本実装に反映済み

## UI

### UIからの設定切替 (2026-08-06)

環境変数でしか変えられなかったオプトイン設定を、性質で2つに分けてUIから操作できる。

#### 即反映(生成ボタン直上のチェックボックス、再ロード不要)

FirstBlockCache(+しきい値)・Sage Attention・Turbo LoRA。リクエストごとのパラメータ
(`cache` / `cache_threshold` / `attn` / `turbo`)として送り、`MiniMaxH3Runner.
apply_instant_settings()` が生成ロック取得後・denoise前に常駐transformerへ適用する
(`disable_cache()`/`enable_cache()`、`set_attention_backend()`、`_TurboLoRALinear.enabled`)。
**未指定なら従来どおりプロセス既定**なので既存のcurl/スクリプトは無変更で動く。
turbo有効時はFBCを自動的に無効化する(元の安全規則を踏襲)。

実測(同一seed、再起動なしで切替): FBC on 100.8s(6スキップ)/ off 129.3s(0スキップ)、
Sage on 129.3s / off(native) 158.5s、Turbo on(8steps) 38.9s。

#### 再ロードが必要(ヘッダの折りたたみ + 「適用(再ロード)」ボタン)

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

### 生成済み動画ギャラリー (2026-08-06)

結果表示の下に `outputs/` 直下の mp4 をタイル表示する。サムネイルはサーバで生成せず
`<video preload="metadata" src="....mp4#t=0.1">` に先頭フレームを描かせる(依存追加なし。
Ref2VAの参照タイルと同じ手法)。

- `GET /api/outputs`: **直下の *.mp4 と *.png**(静止画モードの生成物。`type` フィールド
  "video"/"image" で区別。`outputs/ab_*` 等の検証資料は対象外)。
  尺/解像度は ffprobe で取得し mtime+size をキーにメモリキャッシュ
- `POST /api/outputs/delete`: **パストラバーサル対策**(`/`・`\`・`..` を拒否し、
  resolve 後に `outputs/` 直下であることを検証=シンボリックリンク経由の脱出も遮断)。
  UIは `confirm()` 必須。実機で `../app.py` `/etc/passwd` `ab_*/...` 等が400になることを確認済み
- `POST /api/outputs/concat`: **連結順は「チェックした順」**(表示順は新しい順)。
  全入力のパラメータが一致すれば `concat demuxer + -c copy`(**再エンコードなし=劣化ゼロ**)、
  不一致なら `filter_complex` で先頭動画の解像度へ揃えて再エンコード(無音入力には
  `anullsrc` を合成)。2本未満は400、連結同士の同時実行は409(GPUを使わないので
  generation_lock は取らない)。**PNG(静止画)は連結対象外**(ffprobe が PNG も
  video stream として読めてしまうため、拡張子で明示的に400にする。UI側も選択に
  PNG が混ざると連結ボタンを無効化する)

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

## LLM プロンプト強化

### LLMプロンプト強化(2026-08-04追加)

ローカルLLM(gemma4-31B、OpenAI互換 `/v1/chat/completions`)で、入力プロンプトを
H3公式ガイドの形式へ整形する。クラウド版Hailuo AIの内部プロンプト整形層のローカル再現。

- 接続先: 環境変数 `H3_LLM_URL`(既定 `http://127.0.0.1:64660`)。接続不可時は502
  (生成機能には影響しない)
- `POST /api/prompt/enhance` {text, mode, seconds, task, lang}
- モード(UIの「LLM強化」ボタン+モード選択):
  - `storyboard`(既定): マルチショットのCUTタイムコード形式へ展開(総尺=seconds、2〜3カット、
    ハードカット・被写体同一性維持・カット毎の音指示。焦点距離は35/50/65/100mmに制限)。
    これはH3公式ドキュメントの記法ではなく本アプリの独自案(下記 `h3-official` 参照)
  - `brief`: 公式ブリーフ形式(シーン→被写体→アクション→カメラ→音→終わり方)の単一ショット詳細化
  - `h3-official`(2026-08-07追加): MiniMax公式スキル `h3-prompt-writing`
    (`MiniMax-AI/MiniMax-H3` リポジトリの `skills/h3-prompt-writing/`)のフィールド構造・
    記法へ厳密準拠。`task`(UIの現在のタブ = t2va/fl2va/ref2va)で参照ガイドを切替える:
    t2va/fl2va は `references/base-en.txt` の3フィールド形式
    (`integrated_multimodal_description` / `overall_soundscape` / `non_diegetic_music`、
    `[Shot N] At MM:SS.SS` のカット記法・`<d>[言語] ...</d>` の台詞逐語保持)、ref2va は
    `references/ref-en.txt` の6フィールド形式(`subject_definitions` / `summary` /
    `retention_analysis` / `detailed_description` / `overall_soundscape` /
    `non_diegetic_music`)。出力は公式既定の英語(`lang=en`)、`lang=ja` で書き換え本文のみ
    日本語化も可能(フィールド名・`[Shot n]`等のラベル・タイムコード記法・`<d>`内の台詞は
    公式ルールどおり英語/原語のまま)。リファレンス本文は本リポジトリに同梱しない
    (ライセンス表記が無いリポジトリのため)。`venv/bin/python scripts/fetch_h3_skill.py`
    で `skills_cache/`(.gitignore 済み)へ事前取得しておく必要があり、未取得時は
    400+取得コマンド案内のエラーになる。システムプロンプトはSKILL.md+参照ガイド全文を
    要約せず投入(約18.8KB=t2va/fl2va、約26.6KB=ref2va)。
  - `translate`: 過剰創作なしの英訳(CUT構造は保持)
- 強化結果はプロンプト欄を置き換え(編集可)、「元に戻す」で1世代復元。生成結果には
  使用プロンプト全文を折りたたみ表示(強化あり/なし・モード間の比較評価用)
- UIに手書き用「プロンプトガイド」チートシート(公式ブリーフ構造・CUT記法・実測カット精度±1秒)を同梱
- 実LLM検証済み(gemma4-31B Q4_K_M): 3モード(brief/storyboard/translate)とも形式に従う
  ことを確認(storyboardはタイムコード合計・焦点距離制限も遵守、応答11〜17秒)。
  `h3-official` は2026-08-07追加時点でLLMサーバ未接続のため生成A/Bのみ実施(下記参照)、
  LLM経由での実応答確認は別途必要

### 公式スキル(h3-prompt-writing)モードの実機検証 (2026-08-07)

ローカルLLM(gemma4-31B)接続下で `h3-official` モードを実測した結果。

**構造適合**: 生成されたプロンプトは公式仕様に完全準拠していた。
- T2VA: 3フィールド(`integrated_multimodal_description` / `overall_soundscape` /
  `non_diegetic_music`)、`[Shot 1]` は時刻なし、`[Shot 2] At 00:05.000` の3桁記法、
  台詞は `<d>[Japanese] おかえり</d>` と原語で保持、話者ID `(S1)` も付与。応答6.2秒
- Ref2VA: 6フィールドすべて出力、`<Subject n>`/`<Picture n>`/`<Audio n>` のラベルと
  `fully_preserved` / `fully_copy` 等の関係マーカーも使用。応答12.8秒

**カット位置の実測**(768²・10秒・turbo 8steps、LLM生成プロンプト、各2seed。
ffmpeg のシーン検出 `gt(scene,0.15)`):

| 記法 | 指示 | 検出(seed 12345 / 777) | ズレ |
|---|---|---|---|
| 公式 `[Shot 2] At 00:05.000` | 5.000秒 | 4.875 / 4.875 | **-0.125秒(両seedで同一)** |
| 独自 `CUT 2 [6-10秒]` | 6.0秒 | 6.083 / 6.333 | +0.083 / +0.333秒 |

公式記法の方が**ばらつきが小さい**(2seedで完全同一)。ただし独自CUT記法も0.1〜0.3秒の
ズレに収まっており、**当初の手書きプロンプトによる1試行(独自記法+1.0秒)ほどの差はない**。
どちらも実用範囲で、公式記法は再現性が高い、というのが実測の結論。

**実装上の罠(実機で再現・修正)**: `lang=ja` を指定しても**英語で返ってくる**。
言語指示をシステムプロンプトの冒頭に置くと、その後ろに続く15.8KBの英語リファレンス
(英語出力の実例が並ぶ)に上書きされるため。**言語指示はリファレンス本文より後ろ、
最終指示として置くこと**で解消した(diffusers-server の Tポーズ実装で確立した
「プロンプトは末尾が最強」と同じ原理)。修正後は地の文だけが日本語になり、
フィールド名・`[Shot n]`・タイムコード・`<d>` タグ内の台詞は英語/原語のまま維持される。

### h3-official の品質保証: 検証 + 修復ループ (2026-08-08)

「LLMの性能限界なのか、プロンプトで直るのか」を推測で議論しないために、**まず故障率を
実測**し(`scripts/probe_h3official_compliance.py`、5入力×3回)、その結果に基づいて
対策した。LLM は gemma4-31B Q4_K_M / llama.cpp、**n_ctx=7680**。

#### ベースライン実測: 失敗は1クラスに集中していた

| 故障クラス | 発生率 |
|---|---|
| 構造・記法(フィールド名・`[Shot n]`・タイムコード・`<d>`タグ・話者ID) | **0/15 (0%)** |
| 時間配分 | 6/15 (40%) |
| 文脈溢れ | 0/15(余裕 2,900+ トークン) |

**「入力が変われば別の問題が出るのでは」という懸念は良い方に外れた** — 違反は散らばらず
時間配分の1クラスに集中していた。しかもその6件は性質が2つに分かれる:

- **入力が物理的に不可能 (3/6)**: 台詞の推定発話時間が尺を超えている。実例は38文字の
  独白(推定9.5秒)を9秒尺で要求したケースで、**LLMは1ショットに9秒全部を割く最善の
  配分をしていた**。公式仕様が台詞の逐語保持を要求する以上、短縮という逃げ道も無い
  → **どのモデルでも解決不能。ユーザーに返すべき情報**
- **LLMの配分ミス (3/6)**: 尺には収まるのに配置が悪い → 指摘すれば直せる部類

文脈: t2va は system 4,397 tok / 7,680 で余裕あり。**ref2va は 6,191 tok (81%) で余裕
約1,100 トークンしかない**(実測では6フィールド揃って正常終了したが、入力が長い場合は要注意)。

#### 対策(3層)

1. **バリデータ** (`core/prompt_check.py`): 決定的に判定できる規則を実装
   (F1 3フィールド / F2 先頭ショットに時刻なし / F3 カット時刻の厳密増加 / F4 尺内 /
   F5 ショット尺の下限 / F6 `<d>`タグと言語タグ / F7 台詞がショットに収まる /
   F8 話者ID)。F5・F7 は公式仕様に無い**本アプリの実用規則** — 公式は「カット時刻が
   尺内」としか言わないため、5秒尺で4.5秒にカットして最終ショット0.5秒、という
   字義どおりだが使えない出力を許してしまう(実際に発生)。意味的整合性(1ショット内で
   画角が混在する等)は近似ルールで**警告**にとどめる。**手書きプロンプトにも効く**
2. **システムプロンプト改善**: 英語版wrapperには**ガイド本文の後ろに何も無かった**
   (日本語版だけ `lang=ja` 修正時に後方ブロックを持っていた)。既定は `lang=en` なので、
   尺の制約を含む全指示が15.8KBのガイドに埋もれていた。両版の**末尾**に時間配分6ルール
   (ショット尺の下限・台詞の発話時間見積り・ショット数の目安・1ショット1構図・
   画面外音声の定型句・話者ID)を追加
3. **修復ループ** (`core/llm.py` の `enhance_prompt_checked`): 違反を検出したら
   その内容を日本語で突きつけて再生成(最大2回、`H3_OFFICIAL_MAX_REPAIRS`)。
   **違反が増える修復案は破棄**する(修復が別の箇所を壊す事故の防止)。
   入力の実現不可能性は**LLMに投げる前**に判定し `InfeasibleInputError` → 400 + 助言

#### 改善後の実測(同一条件)

| | clean | 入力不可を検出 | **違反が残存** |
|---|---|---|---|
| ベースライン | 9/15 | 0 | **6/15** |
| **改善後** | **12/15** | **3/15**(正しい挙動) | **0/15** |

**未解決の違反がゼロになった。** 全ケースが「妥当なプロンプトを返す」か「実行不能な理由を
助言付きで返す」かのいずれかになる。所要時間の中央値は 8.5s → 9.2s(+8%)で、修復ループは
必要なときだけ発動する(発動時は約2倍の21秒)。

API は `/api/prompt/enhance` の h3-official 応答に `violations` / `warnings` /
`check_report` / `attempts` / `repaired` を追加。UI は残った指摘をステータス欄に表示する
(**生成はブロックしない** — プロンプトは編集可能なので人間が最終ゲート)。


# どのモードで、いつ何がロード/解放されるか — 常駐リファレンス

モード・量子化・turbo・TE配置の組み合わせが増えたため、**「この設定のこの局面で GPU に何が載っているか」**を一枚で引けるようにしたもの。すべて `core/runner.py` のコードから導き、実測ログで裏を取っている。

> **このドキュメントを作った理由**: 2026-08-09、ピーク VRAM の内訳を「transformer + デコード時のVAE」と説明したが**誤りだった**(実際は transformer をデコード前に解放している)。組み合わせが多すぎて記憶で答えると間違える、という実例。以後はここを見ること。

---

## 0. まず結論 — 現行構成(この箱)

```bash
H3_LOWVRAM=1 H3_TE_PRUNE=1 H3_TE_DEVICE=cuda:1
```

| 局面 | GPU0 (48GB) に載っているもの | GPU1 (20GB) |
|---|---|---|
| アイドル | (ほぼ空、~1.5GB) | **TE-nf4 17.4GB(常駐しっぱなし)** |
| エンコード | — | TE-nf4 17.4GB |
| **デノイズ** | **transformer-int8 34.3GB + 活性化 ~6.6GB = 40.9GB** ← **ピーク** | TE-nf4 |
| (transformer 解放) | 0.3GB | TE-nf4 |
| デコード | vae ペア 11.3GB + バッファ | TE-nf4 |
| 終了時 | vae を CPU へ → 1.2GB | TE-nf4(解放しない) |

**重要**: デノイズとデコードは**時間的に重ならない**。transformer はデコードの直前に必ず解放される(`generate()` 3792-3793行)。ピーク 40.9GB は**デノイズ時**のもの。

---

## 1. 何が「常駐」で何が「毎回」か — 3つの軸

判断は次の3つの独立した軸で決まる。混乱の元はこれらが直交していること。

| 軸 | 環境変数 | 効果 |
|---|---|---|
| **A. どこまで常駐させるか** | `H3_LOWVRAM` (0/1/group) | 最重要。下記 §2 の表を決める |
| **B. TE をどのGPUに置くか** | `H3_TE_DEVICE` | 設定すると TE は**一切解放されなくなる**(別GPU常駐) |
| **C. ロードを速くするか** | `H3_TE_PREQUANT` | 「いつロードするか」は変えない。**ロードの中身だけ**が速くなる |

**turbo (`H3_TURBO_LORA` / リクエストの `turbo=1`) はこの表に影響しない。** LoRA は transformer にラッパーを被せるだけで、常駐の増減は起きない(初回のみ 1.38GB の重み読み込みとラップ処理)。FBC が強制OFFになるのと、既定ステップ数が変わるだけ。

---

## 2. `H3_LOWVRAM` 別・フェーズ×常駐物

### `H3_LOWVRAM=0` + `H3_TE_QUANT=bnb-4bit`(96GB級の既定)

TE と transformer が**両方とも常駐しっぱなし**。VAE だけがフェーズごとに GPU を往復する。

| 局面 | 常駐 | 備考 |
|---|---|---|
| 定常(リクエスト間) | transformer 66.3GB + TE-nf4 21GB = **87.5GB** | VAE は CPU |
| エンコード | 同上 | |
| デノイズ | 同上 + 活性化 | |
| **デコード** | **transformer を解放** → vae ペア 11GB | 3つ同時は 98.5GB で 96GB を超えるため |
| デコード後 | transformer を再ロード(定常へ復帰) | |

### `H3_LOWVRAM=1`(48GB級) ← **この箱**

**リクエスト間は何も常駐しない**のが原則。ただし `H3_TE_DEVICE` を設定すると TE だけは例外。

| 局面 | `H3_TE_DEVICE` 未設定 | `H3_TE_DEVICE=cuda:1` |
|---|---|---|
| 入口 | transformer を解放(前回の残り) | 同左 |
| エンコード | **TE をロード** 17.4GB | TE は GPU1 に常駐済み(**ロード不要**) |
| layout/latents/timesteps | TE 常駐のまま | TE をパイプから外し transformer をロード |
| (TE 解放) | **TE を解放** | **解放しない** |
| デノイズ | transformer-int8 34.3GB + 活性化 | 同左 |
| (transformer 解放) | **解放** | **解放**(現状) |
| デコード | vae ペア 11.3GB | 同左 |
| 出口 | vae を CPU へ。**何も再ロードしない** | 同左 |

**毎リクエストの固定費**:

| | TE ロード | transformer ロード | 合計 |
|---|---|---|---|
| 素の構成 | 46.5〜55.8s | 32.5s | **~85s** |
| + `H3_TE_PREQUANT=1` | 21〜34s | 32.5s | **~55s** |
| + `H3_TE_DEVICE=cuda:1` | **0s** | 14.8〜32.7s | **~25s** |

**`H3_KEEP_TRANSFORMER=1` を重ねた場合**(`H3_TE_DEVICE` 設定必須、詳細は §5.5): 上の
表の「(transformer 解放)」行と「デコード」行が変わる — transformer は**デコード位相でも
解放されず**、**リクエストをまたいでも常駐したまま**になる(「入口」の解放も発生しない)。
常駐し続けるのは初回ロードの1回のみで、transformer ロードの固定費(14.8〜32.7s)は
初回だけに収束する。デコード位相のピークは transformer 34.3GB 常駐 + fp16 デコード
~11.4GB(`H3_VIDEO_VAE_FP16=1` 必須)。実測は §5.5 参照。

### `H3_LOWVRAM=group`(24-32GB級)

transformer は**ホストRAMに常駐**し、ブロック単位(1〜2個、~1.4GB)で GPU を往復する。

| 局面 | GPU上 | 備考 |
|---|---|---|
| 起動時 | transformer は CPU 常駐(int8, group-offload フック付き) | プロセス生存中ずっと |
| エンコード | TE-nf4 21GB | |
| デノイズ | TE-nf4 21GB + transformer の 1-2 ブロック ~1.4GB | |
| **デコード** | **TE を強制解放** → vae ペア 11GB | TE(21GB)+デコード(16.3GB)=37GB が 32GB を超えるため |
| デコード後 | TE を再ロード | 次リクエストが TE を先に要るため |

**注意**: このモードだけ「デコードで解放されるのは transformer ではなく TE」。group-offload された transformer の GPU 占有はもともと小さいので、譲るのは TE の側になる。

---

## 3. `H3_TE_DEVICE` を設定したときの追加の約束

- TE は **`_free_text_encoder()` を呼んでも解放されない**(`force=True` でも)。解放しないことが目的だから
- TE は**普段パイプから外れている**。`_te_attached()` の窓(エンコード中)だけ繋がる
  - 理由: `_execution_device` は components 順で最初の nn.Module のデバイスを返すため、繋ぎっぱなしだと layout も **decode も**別GPUにテンソルを作って落ちる(実機で再現)
- layout/latents/timesteps の窓では `_pin_execution_device_to_compute()` が **vae も一時的に外す**(TE を外した次に見つかるのが CPU 上の vae になり、今度は `cpu` が返るため)
- **ref2va は TE用GPUが 24GB 未満なら 400 で拒否**される(20GB で OOM を実測済み)

---

## 4. よくある誤解

| 誤解 | 実際 |
|---|---|
| 「ピーク 40.9GB は transformer + デコードVAE」 | **違う**。ピークはデノイズ時(transformer 34.3 + 活性化 6.6)。デコード時は transformer 解放済みで VAE 11.3GB |
| 「`H3_TE_PREQUANT` で TE が常駐するようになる」 | **違う**。ロードするタイミングは変わらない。**ロードが速くなるだけ**(53s→29.5s) |
| 「turbo にすると VRAM が増える/減る」 | ほぼ変わらない。LoRA はラッパーで、初回に 1.38GB を読むだけ |
| 「`H3_LOWVRAM=group` は 1 より常に省メモリで遅い」 | 省メモリは正しいが、**静止画では group の方が遅い**(ブロック転送が固定費で、計算が軽い超短尺では転送が支配する) |
| 「TE を別GPUに置けば固定費がゼロになる」 | TE 分だけ。**transformer のロード(14.8〜32.7s)は残る**(デコード前に解放しているため)。ただし `H3_KEEP_TRANSFORMER=1` を重ねればこれも初回のみに収束する(§5.5) |

---

## 5. VRAM 容量が変わったら — 表を覚え直さず「導出」する

**モードは VRAM 容量の関数**である。GPU を替えたら下記の部品表と不等式から再導出すること
(この表自体、96GB 機で作られ 48GB 機で作り直した経緯がある)。

### 5.1 部品表(すべて実測)

| 部品 | サイズ |
|---|---|
| text_encoder bf16 | 66.71GB(51層削除で 53.06GB) |
| **text_encoder nf4** | **21.02GB(51層削除で 17.45GB)** |
| transformer bf16 | 66.3GB |
| **transformer int8** | **34.0GB** |
| transformer_ref bf16 / int8 | 61.7GB / ~34GB |
| vae + audio_vae (fp32) | 11.0GB |
| デノイズ活性化 | ~5〜6.6GB(768²・5秒で実測 6.6GB) |
| **デコードのピーク** | **16.29GB**(video VAE fp16 なら ~11.4GB) |
| ref2va の参照エンコード追加分 | TE に対して **+3.2GB 以上**(2048px 短辺の vision tower。実測の下限) |
| **CUDA コンテキスト等(非PyTorch)** | **~1GB**(忘れやすい。下記の罠を参照) |

> **単位の罠(この表を作る過程で実際に間違えた)**: `nvidia-smi` は **MiB**、PyTorch の
> OOM メッセージは **GiB**、本アプリのログ(`gpu_mem_gb()`)は **GB(10進)**。
> 20GB カードは `nvidia-smi` で 20475 MiB = **21.47 GB(10進)** だが、PyTorch から見える
> 容量は 19.55 GiB = **20.99 GB(10進)**。**約 0.5GB の差**があり、さらに非PyTorch分
> ~1GB が引かれる。**カタログ容量をそのまま予算にすると 1.5GB ほど過大評価する**
> (この見落としで「ref2va は 20GB でも入る」と誤って導出しかけた。実測は OOM)。

### 5.2 満たすべき不等式(局面ごとに独立)

同時に載る必要があるのは**各局面の中だけ**。局面をまたいで合計する必要はない。

```
実効予算 = カタログ容量 − 単位差(~0.5GB) − CUDAコンテキスト等(~1GB)

エンコード : TE                                    ≤ 実効予算
デノイズ   : transformer + 活性化(~6.6GB)          ≤ 実効予算
デコード   : デコードピーク(16.29 / fp16なら11.4)   ≤ 実効予算
```

**リクエスト間で常駐させたいものがあれば、その分を各局面に足す。** ここが設計の分岐点になる。

- TE を常駐させたまま デノイズ したい → `TE + transformer + 活性化 ≤ 容量`
- transformer を常駐させたまま デコード したい → `transformer + デコードピーク ≤ 容量`

### 5.3 容量から導かれるモード

| 容量 | 成立する構成 | 律速(なぜそれ以上できないか) |
|---|---|---|
| **96GB** | bf16 のまま TE+transformer 常駐(87.5GB) | デコード時に 3つ同時(98.5GB)が入らず transformer を譲る |
| **80GB** | int8 で TE+transformer+transformer_ref 常駐(89GB) | 同上 |
| **48GB** | `H3_LOWVRAM=1`。TE(17.45)+transformer(34)+活性化(6.6)= **58GB で入らない** → 毎回入れ替え | **TE の置き場所が無いこと**が固定費の原因 |
| **32GB** | `H3_LOWVRAM=group`。transformer を RAM 常駐、ブロック単位で流す | TE(21)+デコード(16.3)=37GB > 32GB なのでデコード時は TE を譲る |
| **24GB** | `group` + `H3_TE_PRUNE=1` | TE-nf4 17.45GB が予算の大半。削除しないと入らない |
| **18GB** | 同上(実測の床) | **TE-nf4 削除版 17.45GB そのものが床** |
| 16GB | **不可** | TE のロード終盤で OOM。突破には TE のストリーミング実行が要る |

### 5.4 TE を別GPUに置く場合(`H3_TE_DEVICE`)

TE用GPUに必要な容量は用途で変わる。

現行 20GB カードの**実効予算は約 19.7GB**(カタログ 21.47GB − 単位差 0.5 − 非PyTorch ~1)。

| 用途 | 必要量(実測) | 20GB での可否 |
|---|---|---|
| t2va / fl2va / t2i | 17.76GB | **成立**(余裕 ~1.9GB) |
| ref2va | **20.67GB 以上**(TE 17.45 + 参照エンコード 3.22 以上) | **不可**(204MB 不足で OOM) |

**→ ref2va に必要なのは実効 20.7GB 以上、つまりカタログ 22.2GB 以上。** 24GB カード
(実効 ~22.4GB)なら余裕 ~1.7GB で成立する見込み。これが RTX PRO 4000 Blackwell 24GB へ
交換する具体的な根拠(**容量だけが理由**で、PCIe 幅や世代は無関係 — §3 参照)。

ただし余裕 1.7GB は薄く、参照画像を 2枚以上にするとさらに要求が増える(実測では
2枚でも OOM した)。**24GB でも ref2va の複数参照は保証できない。**

### 5.5 `H3_KEEP_TRANSFORMER=1` — transformer 常駐のままデコード(2026-08-09 実測・成立)

**transformer を常駐させたままデコードする**構成。固定費(transformer ロード 14.8〜32.7s、
`H3_LOWVRAM=1` では毎リクエスト発生)が消える。5.3 の不等式どおり fp32 デコードでは
入らないが、`H3_VIDEO_VAE_FP16=1` を使うと入る:

```
transformer 34.3 + デコードピーク 16.29 = 50.6GB  > 48GB   ← 不可(fp32 VAE)
transformer 34.3 + 11.4 (H3_VIDEO_VAE_FP16=1)   = 45.7GB  < 48GB  ← 導出予測
```

`core/runner.py` の `H3_KEEP_TRANSFORMER=1` として実装済み。成立条件は import 時ガードで
強制する(3つとも必須、欠けたら `RuntimeError`):

1. `H3_LOWVRAM=1`(raw `"1"` のみ。`group` は対象外 — 別設計のため無関係)
2. `H3_TE_DEVICE` 設定済み(TE が別GPU。**さもないとエンコード位相が先に破綻する**:
   TE-nf4 17.45GB + 常駐transformer-int8 34.3GB = 51.75GB > 実効予算)
3. `H3_VIDEO_VAE_FP16=1`(fp32 では上記のとおり 50.6GB で入らない)

既定(`H3_KEEP_TRANSFORMER=0`)は挙動不変。`transformer_ref`(ref2va 経路)は対象外で
従来どおり毎回解放される。

**実機 E2E 実測(2026-08-09、48GB GPU0 + 20GB GPU1、`H3_LOWVRAM=1 H3_TE_PRUNE=1
H3_TE_DEVICE=cuda:1 H3_VIDEO_VAE_FP16=1 H3_KEEP_TRANSFORMER=1`)**:

- transformer(int8) は初回リクエストで1回だけロード(32.0s)。以降のリクエストで再ロード
  なし(サーバーログで確認)
- t2i turbo 4steps 定常: **9.7s/枚**(denoise 4.32s、decode 1.5s)。peak VRAM 41.97GB
  (デノイズ時)
- t2i steps=30 定常: 51.1s(denoise 45.7s)。peak 41.97GB
- t2va 5秒 turbo 4steps: **44.2s**(denoise 26.05s、decode 10.81s)。**peak VRAM
  44.15GB = デコード位相**(transformer 34.03GB 常駐 + fp16 デコード)。上記の導出予測
  45.7GB に対し実測 44.15GB、カタログ 48.9GB に対し余裕 ~4.8GB
- nvidia-smi 実測ピーク 42,620 MiB(1秒サンプリング。瞬間ピークは torch 計測の
  44.15GB が正)
- 同一seed 出力等価性: フラグOFFのベースライン(他は同条件)と PNG MD5 **完全一致**
  (seed=11、md5 `665eadddea8f34298a1b5b89e69d4bd0`)。ベースライン側は total 63.27s
  (transformer ロード込み)/ peak 36.4GB
- 高速化の系譜: t2i turbo 157s(08-07朝)→ 83.2s(`H3_TE_PREQUANT`)→ ~35s
  (`H3_TE_DEVICE`)→ **9.7s**(`H3_KEEP_TRANSFORMER`)。t2va 5s(768²):
  turboなし30steps **351.4s**(48GB素の構成)→ turbo 143s → 60.5s → **44.2s**(8.0倍)

### 5.6 導出式の検算(全ケースで実測と一致)

上の式に既知の実測を通したところ、成否の判定が**すべて実測と一致**した。

| ケース | 必要 / 実効予算 | 判定 | 実測 |
|---|---|---|---|
| TE用GPU 20GB・t2va | 17.76 / 19.97 | OK | ○ 成立 |
| TE用GPU 20GB・ref2va | 20.67 / 19.97 | NG | ○ OOM |
| TE用GPU 24GB・ref2va | 20.67 / 24.27 | OK | (未実測・期待値) |
| GPU0 48GB・デノイズ | 40.60 / 49.81 | OK | ○ peak 40.89 |
| GPU0 48GB・TE常駐のままデノイズ | 58.05 / 49.81 | NG | ○ だから毎回入れ替えている |
| GPU0 48GB・transformer常駐のままデコード | 50.29 / 49.81 | NG | (fp32 VAEでは未実装のまま) |
| 同上 + video VAE fp16(`H3_KEEP_TRANSFORMER=1`) | 45.70 予測 / 49.81 | OK | ○ 実測 44.15GB(§5.5、2026-08-09) |

**GPU を替えたらこの式に新しい容量を入れて再導出すること。** 表を覚え直す必要はない。

## 6. 実行中に確認する方法

```bash
# runner が今どう思っているか
curl -s http://127.0.0.1:8611/api/status | python3 -m json.tool | grep -E "loaded|on_gpu|peak|allocated"

# 実際のGPU占有(プロセス単位)
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv

# ロード/解放の履歴(これが一番確実)
grep -E "transformer (loaded|freed)|text_encoder|vae/audio_vae ->" logs/server.log | tail -20
```

ログの各行に `gpu={'allocated_gb':..., 'peak_gb':...}` が付いているので、**推測せずここを読むこと**。§0 の表もこのログから起こしている。

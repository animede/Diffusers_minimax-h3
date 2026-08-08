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
| 「TE を別GPUに置けば固定費がゼロになる」 | TE 分だけ。**transformer のロード(14.8〜32.7s)は残る**(デコード前に解放しているため) |

---

## 5. 実行中に確認する方法

```bash
# runner が今どう思っているか
curl -s http://127.0.0.1:8611/api/status | python3 -m json.tool | grep -E "loaded|on_gpu|peak|allocated"

# 実際のGPU占有(プロセス単位)
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv

# ロード/解放の履歴(これが一番確実)
grep -E "transformer (loaded|freed)|text_encoder|vae/audio_vae ->" logs/server.log | tail -20
```

ログの各行に `gpu={'allocated_gb':..., 'peak_gb':...}` が付いているので、**推測せずここを読むこと**。§0 の表もこのログから起こしている。

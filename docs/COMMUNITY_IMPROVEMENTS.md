# コミュニティ改良の取り込み作業一覧(MiniMax H3)

オリジナル(MiniMaxAI/MiniMax-H3 + diffusers PR #14355)を起点に、ComfyUI コミュニティ
などで出た改良を本アプリ(diffusers 経路)へ取り込んだ作業の記録。2026-08-04〜08-06。

判定はすべて **同一 seed の実機 A/B** に基づく。「品質同等」は目視 + PSNR + 音声相関で
確認したもの、「等価」は mp4 の MD5/バイト一致まで確認したものを指す。

---

## A. 取り込んだもの(コミュニティ発)

### A-1. FirstBlockCache(ComfyUI の EasyCache 相当) — 既定ON

| | |
|---|---|
| 出典 | ComfyUI コミュニティ(kijai 氏の EasyCache 活用法。@gosrum / @umiyuki_ai のポスト) |
| コミュニティ報告 | EasyCache で 1.67倍、Sage 併用で 6分→2分半 |
| 取り込み方 | diffusers 公式の `FirstBlockCache`(同系統のステップ間キャッシュ)を採用。`H3_CACHE`/`H3_CACHE_THRESHOLD` |
| 実測 | デノイズ 157s → **118s(-25%)**、30step中7スキップ。threshold 0.1 なら 81.5s(1.92倍)だが構図ドリフト |
| 品質 | PSNR 31.8〜34.3dB、音声相関 0.979。目視で区別困難 |
| 判定 | **threshold 0.05 を既定化**。0.1 は opt-in |
| 罠 | H3 ブロックは PR ブランチの `TransformerBlockRegistry` 未登録 → 自前登録が必要。リクエスト毎に `_reset_stateful_cache()` + `cache_context()` 必須(同一seed連続2本のバイト一致で検証) |
| コミット | `14afdfc` |

### A-2. Sage Attention — 既定ON

| | |
|---|---|
| 出典 | 同上のポスト(「Sage のみでも25%短縮」) |
| 取り込み方 | sm_120 向けに **thu-ml/SageAttention をソースビルド**(Linux 向け事前ビルド wheel は存在せず、公開されているのは Windows 版のみ)。`H3_ATTN_BACKEND` |
| 実測 | デノイズ 118s → **104s(-12%)**。コミュニティ報告の -25% には届かず |
| 品質 | 完全決定論(同一seed 2本バイト一致)。目視同等(PSNR 21dB は int8-QK 近似による軌道ドリフトで劣化ではない) |
| 判定 | **既定化**(`H3_ATTN_BACKEND=default` で従来 SDPA に戻せる) |
| 罠 | ビルドは `MAX_JOBS=4 NVCC_THREADS=2` + systemd-run メモリ上限必須(無制限並列 nvcc はホストRAM枯渇でシステム巻き添え事故歴)。`CUDA_HOME=/usr/local/cuda-12.8` の明示が必要(既定の cuda-13.0 は torch cu128 と不一致でビルド失敗) |
| コミット | `9c7e6a6`、`scripts/build_sageattention.sh` |

### A-3. Latent アップスケーラ(2段生成 hires-fix) — opt-in

| | |
|---|---|
| 出典 | [Tr1dae/ComfyUI-MiniMaxH3_LatentUpscaler](https://github.com/Tr1dae/ComfyUI-MiniMaxH3_LatentUpscaler)(@umiyuki_ai のポスト経由) |
| 仕組み | 低解像度で前半デノイズ → 映像 latent のみ空間2x補間 → 再ノイズ → 高解像度で仕上げ。学習済みアップスケーラは使わない |
| 取り込み方 | `/api/t2va`・`/api/i2v` の `upscale=1`、`H3_HIRES_DENOISE` |
| 実測 | 768²→**1536²** が 645s / peak 88.0GB(upscale=0 は 181s / 92.1GB) |
| 品質 | 構図一致のまま毛並み等の実ディテールが乗る。背景細部は再デノイズで軽微にドリフト(hires-fix の性質) |
| 判定 | **opt-in**(既定OFF) |
| 罠 | **補間対象はノイズ付き latent ではなく x0 推定値**。ノイズ付きを補間すると市松ノイズが増幅され全面ノイズ化する(実機で再現 → 参考実装も `denoised_output` を使っていることを確認して修正)。解像度変更時は `build_packed_sequence()` と `row_timestep_plan` の再構築が必要 |
| コミット | `e9a45a7` |

### A-4. Turbo LoRA(4/8ステップ蒸留) — opt-in

| | |
|---|---|
| 出典 | [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)(Ostris 氏学習中、Apache 2.0)。@PhotogenicWeekE のポストで実用情報を取得 |
| コミュニティ報告 | 8 steps で 1280×736/10秒が 272s→160s。**「4〜7 steps はダメ」** |
| 取り込み方 | `H3_TURBO_LORA=1`(既定8steps)。ComfyUI リパック版ではなくオリジナルを使用 |
| 実測 | 8steps **87.7s(-46%)** / 16steps 98.4s / 4steps 39.6s(基準30stepsは163.5s) |
| 品質 | 8steps は基準に迫る。16steps は基準同等。4steps は柔らかめ・音声弱め |
| 判定 | **opt-in**(LoRA が「デモ/プレビュー・学習途上」と作者明記のため既定OFF。完成版が出たら差し替え + A/B のみで既定化判断できる状態にしてある) |
| 新事実 | **「4〜7 steps はダメ」は ComfyUI 標準サンプラー起因の可能性が高い**。本実装(video shift 12 / audio shift 3 のデュアルスケジュールを正しく積分)では 4steps でも音声破損は起きなかった。なお シフト配線は改修不要で、PR 実装の既定値が既に正しいことを sigma 格子のビット一致で確認済み |
| 罠 | LoRA キーは ComfyUI 命名の fused-QKV 形式 → `fuse_projections()` + ランタイムデルタで適用。**`fuse_projections()` は旧 to_q/k/v を削除せず +12.8GB リークする**(実 OOM で発見、明示 delete で対処) |
| コミット | `2ab3100` |

### A-5. block-level group offload(24〜32GB級対応) — opt-in

| | |
|---|---|
| 出典 | ComfyUI の「INT8 + layerwise offload で 24GB 動作」報告 + PR #14355 の streamed offload 対応 + 姉妹プロジェクト(diffusers-server)で確立した知見 |
| 取り込み方 | `H3_LOWVRAM=group`。int8 transformer を CPU ロード → `enable_group_offload(block_level, num_blocks_per_group=1, use_stream=True)` |
| 実測 | 32GB 制限で **peak 28.7GB**、デノイズ 220.8s(常駐比 ~2.1倍) |
| 等価性 | 通常 int8 モードと **mp4 MD5 一致** |
| 判定 | **opt-in** |
| 罠 | **`use_stream=True` + `low_cpu_mem_usage=True`(diffusers 既定)は torchao `Int8Tensor` の pin_memory でクラッシュ**(`cannot pin 'torch.cuda.CharTensor'`)。`low_cpu_mem_usage=False` で回避(onload も4〜5倍高速化)。→ diffusers 本体に報告する価値のある実バグ |
| コミット | `26bf434` |

---

## B. 調査した結果、取り込まなかったもの

| 項目 | 出典 | 結論 |
|---|---|---|
| **VAE の device/dtype キャスト修正** | [ComfyUI commit 16e3f30](https://github.com/Comfy-Org/ComfyUI/commit/16e3f3034f2bba1fff6c70cbd759339778555cd6)(@PhotogenicWeekE) | **不要**。ComfyUI 独自の重み管理(レイヤー単位 compute 時キャスト)で生 `nn.Parameter` が素通りするのが原因のクラッシュ修正。diffusers はモジュール丸ごと `.to(device)` で移動し、VAE fp32 固定が PR の明文契約のため同種の不一致が起きない(実コードで確認) |
| **NVFP4 版 TE** | Comfy-Org(14.6GB、ComfyUI形式)/ RedHatAI(20.4GB、compressed-tensors) | **不採用**。ComfyUI 形式は fp4 演算の自前実装が必要、RedHatAI 版は nf4 の 21GB とほぼ同サイズで 24GB 問題を解決しない(実行系も vLLM 前提) |
| **hub 系 attention backend** | diffusers `flash_hub` / `sage_hub` | **不成立**。Hub 側に torch 2.9 向けビルドが存在しない(利用可能は 2.10〜2.12)。環境の問題ではない |
| **ComfyUI の pruned TE ファイル(47.97GB)** | Comfy-Org/MiniMax-H3 | ファイル自体は使わず、**同じ削減を自前で導出して実装**(下記 C-1)。ComfyUI 形式の変換を避け、既存の bnb-nf4 経路をそのまま使えるため |
| **torchao の C++ カーネル** | torchao 0.18 | **見送り**。torch>=2.11 要求で venv 全体のリグレッションリスクが大きい。0.17 の pure-Python フォールバックで運用中 |

---

## C. コミュニティの知見から着想を得て、自前で実装したもの

### C-1. text_encoder 未使用上位レイヤー削除(`H3_TE_PRUNE`)

ComfyUI 配布の bf16 TE が 47.97GB(フル 62.13GB 比 -14.2GB)である理由を調べ、
**H3 は TE の `hidden_states[50]` しか読まない**(51層目以降の14層 ≈ 13GB は死荷重)
ことを突き止めて自前で実装した。

- 実測: TE-nf4 **21.02 → 17.45GB(-17%)**、bf16 66.71 → 53.06GB
- 等価性: t2va・ref2va とも削除なし版と **mp4 MD5 一致**(64層版との `torch.equal` も確認)
- **罠(最重要): 50層ちょうどに削ると transformers の `tie_last_hidden_states` が
  `hidden_states[50]` を最終 norm 適用後の値で静かに上書きし、数値が別物になる。
  51層に削るのが正解**(diffusers 側に `num_layers <= 50 → raise` のガードがあるのは
  まさにこの境界のため)
- これにより 24GB級(実測20GBでも動作)対応が完成した
- コミット: `2d31424`

### C-2. transformer int8 量子化 + 両変種同時常駐

PR #14355 のドキュメント記載レシピ(`Int8WeightOnlyConfig`)を適用し、さらに
transformer と transformer_ref を同時常駐させて Ref2VA ⇔ T2VA の切替コストを消した。

- 実測: transformer 66.3 → **34.0GB**。ref2va 523s → **463-471s**、変種切替の 66GB 級
  再ロードが消滅
- 罠: `expandable_segments:True` が必須(int8 のロード/解放サイクルで断片化し、
  「54GB しか使っていないのに 15GB 確保失敗」が実機再現)
- コミット: `236a424`、`435f831`

### C-3. その他の自前実装

| 項目 | 内容 | コミット |
|---|---|---|
| TE bnb-4bit | TE 66.7GB → 21.0GB。245s → 185s(TE⇔transformer 入れ替えの消滅) | `6526b61` |
| 48GB級フェーズ循環 | TE と transformer を同時常駐させない。t2va peak 38.9GB | `2bb3127` |
| video VAE fp16 | デコードピーク 16.3 → 11.4GB、PSNR 39.97dB。罠: `_keep_in_fp32_modules` が dtype 指定を無効化 | `a6c5ffa` |
| ローカル LLM プロンプト強化 | storyboard / brief / translate の3モード | `02e311f` |
| Ref2VA | 画像9/動画3/音声3 の順序付き参照(PR の機能を配線。OOM 3件を実機で解決) | `ca5d912` |

---

## D. 到達点(768²・5秒)

| 構成 | リクエスト |
|---|---|
| 初期(bf16 TE 入れ替え) | 245s |
| 現既定(bnb-4bit + FBC 0.05 + Sage) | **~160s** |
| + FBC 0.1(opt-in) | ~125s |
| + Turbo LoRA 8steps(opt-in) | **~88s** |
| + Turbo 4steps(ドラフト用途) | ~40s |

VRAM 下限: 96GB → **~18GB**(`H3_LOWVRAM=group H3_TE_PRUNE=1`)。
各段はすべて同一 seed の MD5 一致または品質 A/B で等価性を確認済み。

# MiniMax-H3 を diffusers で動かす — 技術レポート

**日本語** | [English](TECHNICAL_REPORT.en.md)

> **内部資料(作業記録)**。これは開発中に踏んだバグ・失敗・回り道までを含む**作業経過の
> 記録**で、外部向けの技術説明ではない。仕様・設計・性能を知りたい場合は
> **[docs/TECHNICAL_OVERVIEW.md](../TECHNICAL_OVERVIEW.md)** を読むこと。
> この文書の価値は「なぜその設計になったか」「同じ罠をもう一度踏まないため」にある。

**対象期間**: 2026-08-04 〜 2026-08-08(44コミット)
**対象**: 本リポジトリ `minimax-h3`(MiniMax H3 / Hailuo 3.0 の機能確認用スタンドアロンアプリ)
**実装規模**: 約 10,700 行(`core/runner.py` 4,943 / `static/index.html` 1,839 / `README.md` 1,441 / `app.py` 1,003 / ほか)

本レポートは、運用ドキュメントである [README.md](../../README.md) とは別に、**「何を作ったか」ではなく「なぜそう作ったか、何につまずいたか、どう確かめたか」**を残すことを目的とする。数値はすべて実機実測であり、測定条件を併記する。個々の操作手順は README を参照。

---

## 1. 課題設定

MiniMax H3 は、動画とステレオ音声を**1回のデノイズで同時に生成する**オムニモーダル 33B モデルである。音声を後から重ねるのではなく、映像と音声が同一の packed sequence 上の別の行として一緒に denoise される点が、既存の動画生成モデルと決定的に異なる。

本プロジェクトの目的は、このモデルを **diffusers 経路で**動かし、将来 `diffusers-server` へ統合するための知見を先行して得ることだった。ComfyUI 実装は既に存在したが、統合先が diffusers ベースであるため、ComfyUI の資産をそのまま使うのではなく **diffusers 上で同等以上を達成できるか**を検証する必要があった。

### 1.1 最初から分かっていた最大の制約

コンポーネントの実測サイズは次のとおりで、**VRAM にも RAM にも同時に載らない**。

| コンポーネント | 精度 | 実測サイズ |
|---|---|---|
| text_encoder (Qwen3-VL-32B) | bf16(ネイティブ配布) | **66.73 GB** |
| transformer | bf16 | 66.3 GB |
| transformer_ref (ref2va 用、別チェックポイント) | bf16 | 61.7 GB |
| vae + audio_vae | fp32 | 計 11 GB |

合計 約 144 GB に対し、初期環境は **VRAM 96 GB / RAM 94 GB**。

ここで最初の誤りを1つ記録しておく。当初は「text_encoder は fp32 配布だろうから bf16 化すれば 33GB になる」と見積もっていたが、**実測すると配布時点で既に bf16 で 66.73GB** だった。この種の「推定で設計して実測で崩れる」パターンが以降も繰り返されたため、本プロジェクトは**推定値を設計の前提にしない**方針を早期に採用した。

---

## 2. diffusers で動かすための設計

### 2.1 Modular Diffusers のブロックを自前で呼ぶ

MiniMax-H3 の diffusers 対応は **PR #14355 でのみ提供**されており、2026-08-06 時点で未マージ(draft)である。本アプリはこの PR ブランチの **`abc5e9b`(2026-08-02)にコミット固定**して使っている。

さらに、`ModularPipeline` を丸ごと呼ぶのではなく、**個々のブロックを自前で順に呼ぶ**設計を採った。

```
MiniMaxH3SetupStep            → キャンバス解決・フレーム数の 17n+5 整列・キーフレーム準備
MiniMaxH3TextEncoderStep      → プロンプトエンコード(staticmethod を直接呼ぶ)
MiniMaxH3PrepareLayoutStep    → packed sequence のレイアウト・rotary 位置
MiniMaxH3PrepareLatentsStep   → 潜在の初期化
MiniMaxH3SetTimestepsStep     → video/audio 2系統のシグマ格子
MiniMaxH3DenoiseStep          → デノイズループ
MiniMaxH3VideoDecodeStep / MiniMaxH3AudioDecodeStep → デコード
```

**なぜ分解したか**: メモリ制約の解決には「どのフェーズで何が GPU に載っているか」を1ステップ単位で制御する必要があった(§3)。パイプラインを丸ごと呼ぶと、この制御点が存在しない。加えて hires-fix(§5.1)のように**ループの途中に処理を差し込む**改造も、ブロックを自前で駆動していなければ実装できない。

**代償**: `get_block_state()` / `set_block_state()` / `state.get(...)` / `row_timestep_plan` といった **state 契約に強く依存する**ことになった。PR には 8/4〜8/5 に state contract の refactor(`8ab3662`, `99ced1b`)が入っており、最新版へ追従すると壊れる公算が高い。このため README に「**安易に上げないこと**」と明記し、追従する際は「まず t2va の同一 seed MD5 一致で回帰確認」という手順を定めた。

### 2.2 採用しなかった選択肢

`ComponentsManager.enable_auto_cpu_offload()` は diffusers の標準的な解だが、**採用しなかった**。この方式は全コンポーネントを CPU 常駐にしてアクティブなものだけを GPU へ出すため、144GB を RAM に同時保持しようとする。RAM 94GB の本機では成立しない。

---

## 3. メモリ設計 — フェーズ×常駐物

本プロジェクトの中核は「**どのフェーズで何を GPU に置くか**」の設計であり、最終的に5つのモードに整理された。

### 3.1 モード一覧と実測

| 想定 GPU | 起動指定 | ピーク VRAM | t2va 所要(768²・5秒・30steps) |
|---|---|---|---|
| 96GB | (指定なし=既定) | 92 GB | 約 160 s |
| 80GB級 | `H3_TRANSFORMER_QUANT=int8` | 59.7 GB | 約 160 s |
| 48GB級 | `H3_LOWVRAM=1` | 38.9 GB | 約 215 s |
| 32GB級 | `H3_LOWVRAM=group` | 28.7 GB | 約 280 s |
| 18GB級 | `H3_LOWVRAM=group H3_TE_PRUNE=1` | 17.7 GB | 約 280〜320 s |

**VRAM 下限を 96GB → 約 18GB まで下げた**ことになる。各段は同一 seed の MD5 一致または品質 A/B で等価性を確認済み。

### 3.2 設計の考え方: 「短い一方通行」だけを許す

姉妹プロジェクト `diffusers-server` の教訓から、**「60GB級のモジュールを毎ステップ入れ替える」パターンは禁止**という制約を最初から置いた。過去にスワップ暴走でシステムを巻き添えにした事故があったためである。

代わりに採ったのが「短い窓・一方通行」の原則である。各モジュールの移動は**特定のフェーズ1つに限定された片道の移動**でなければならない。

たとえば既定の `bnb-4bit` モードでは:

```
定常状態    : [transformer 66.3GB + TE-nf4 21GB] = 87.5GB 常駐
encode      : 上記のまま(VAE は CPU に退避)
denoise     : 上記のまま
decode      : transformer を落とし [vae ペア 11GB] を GPU へ  ← 約9秒の窓
decode 後   : VAE を CPU へ戻し transformer を再ロード(定常状態へ復帰)
```

transformer(66.3) + TE-nf4(21.0) + vae ペア(11.0) = **98.5GB** は、活性化バッファを数える前から 95.6GB を超える。実際に3つ常駐させたまま decode を試みて `Tried to allocate 30.00 MiB`(アロケータは 93.7GB で飽和)で OOM した。decode は transformer を一切使わないため、**transformer が譲る**のが正解だった。

### 3.3 `H3_LOWVRAM=1`(48GB級): フェーズ循環

48GB では TE-nf4(21GB)と transformer-int8(34GB)の合計 55GB すら同時に載らない。そこで**リクエスト間に何も常駐させない**設計にした。

```
entry    : [何も常駐せず]
encode   : [TE-nf4 21GB]                      ← transformer は解放済み
(TE 解放)
denoise  : [transformer-int8 34GB + 活性化 ~5GB ≈ 39GB]
(transformer 解放)
decode   : [vae ペア ~11GB + デコードバッファ]
(VAE を CPU へ。次回のための先読みロードはしない)
```

代償は明確で、**毎リクエストに TE ロード + transformer ロードの固定費(約 90〜100 秒)**がかかる。この固定費が後にバッチ機能(§5.4)の動機になる。

### 3.4 `H3_LOWVRAM=group`(24-32GB級): ブロック単位ストリーミング

さらに下げるには、34GB の transformer 自体を GPU に置けない。diffusers の `enable_group_offload(block_level, num_blocks_per_group=1, use_stream=True)` を使い、transformer を**ホスト RAM に常駐させ、50 ブロックのうち 1〜2 個(約 0.68GB/個)だけを逐次 GPU へ流す**。

これは CLAUDE.md が禁じる「モジュール丸ごとの入れ替え」には当たらない。group-offload されたモジュールの「常駐場所」は CPU 側であり、フックが小さなブロック単位の GPU 訪問を自動管理する、というのが設計上の整理である。

---

## 4. 踏んだ罠 — 実機で再現し、修正したもの

本プロジェクトで最も価値のある記録は、おそらくこの章である。いずれも**推測ではなく実機で再現し、原因を特定して修正した**ものに限る。

### 4.1 `_execution_device` の解決順序による device mismatch

**症状**: TE を解放した直後から、transformer の `rope()` 内部で cuda/cpu の device mismatch が発生。

**原因**: `MiniMaxH3ModularPipeline._execution_device` は、`self.components` の**挿入順で最初に見つかった `nn.Module` のデバイス**を返す。順序は `text_encoder, tokenizer, processor, vae, scheduler, ...`。TE を解放すると、次に見つかるのが **CPU に退避中の `vae`** になり、`_execution_device` が静かに `cpu` を返すようになる。

**解決**: TE の解放を、`layout_step` / `latents_step` / `timesteps_step` が**終わった後**まで遅延させる。これらのステップが `_execution_device` を必要とする最後の場所であり、一度テンソルが正しいデバイス上に実体化されてしまえば、以降 TE を解放しても影響しない。

この罠は `H3_LOWVRAM=1` でより厄介だった。同モードでは「transformer を先にロードすれば解決」という手も使えない。`vae` は解放されず CPU 上に**存在し続ける**ため、`_execution_device` の走査は transformer に到達しないからである。

### 4.2 autograd グラフが TE を GPU に固定する

**症状**: TE を解放したはずなのに約 50GB が GPU に残る。

**原因**: `MiniMaxH3TextEncoderStep.encode_prompt` は素の `staticmethod` であり、`@torch.no_grad()` はバイパスしたブロックの `__call__` 側に付いている。自前でブロックを分解した(§2.1)結果、**no_grad が外れていた**。

**解決**: 呼び出しを明示的に `with torch.no_grad():` で囲む。分解設計の副作用として生じた罠であり、パイプラインを丸ごと呼んでいれば起きなかった。

### 4.3 `use_stream=True` + `low_cpu_mem_usage=True` は torchao int8 で必ず壊れる

**症状**: group offload モードで、デノイズの**最初の** transformer ブロックの forward が
`RuntimeError: cannot pin 'torch.cuda.CharTensor' only dense CPU tensors can be pinned` で失敗。

**原因**: `low_cpu_mem_usage=True`(API 既定は False)は `enable_group_offload()` 時点での事前ピンをスキップし、**毎回の onload 時にピンする**経路に切り替わる。torchao の `Int8Tensor` に対してこの遅延ピン経路は機能しない。`use_stream=True` と組み合わさったときだけ発火する。

**解決**: `low_cpu_mem_usage=False` を既定にした。名前に反して**これが安全側**である。実機を使った最小再現(int8 量子化した nn.Linear のスタック)まで落として、この組み合わせが無条件のトリガであることを確認した。

副産物として **onload が 4〜5倍高速化**した(0.1〜0.26s → 0.04〜0.07s)。ピン留めメモリはページアウトできないため DMA 転送が速い。代償は事前ピンの一括コスト(約 22 秒、RAM 約 15.7GB)であり、これは `H3_GROUP_OFFLOAD_MIN_RAM_GB` のガードで担保している。

**これは diffusers 本体に報告する価値のある実バグ**である。

### 4.4 `fuse_projections()` が元の重みを消さない

**症状**: turbo LoRA 適用後、transformer の常駐サイズが 66.3GB → **79.08GB**(+12.8GB)に増え、次の TE ロード(21GB)が OOM。

**原因**: `fuse_projections()` は `to_q`/`to_k`/`to_v` を `to_qkv` に**コピー**するが、**元の3つを削除しない**。`unfuse_projections()` を読んでも `to_qkv` を消すだけで元を復元する処理は無い。結果、融合版と非融合版の両方を保持した状態になる。

**解決**: 融合直後に `del module.to_q / to_k / to_v`。`to_qkv` の重みは `torch.cat` による独立したコピーでありビューではないため安全。

### 4.5 int8 の断片化 — 「54GB しか使っていないのに 15GB 確保失敗」

**症状**: int8 両常駐モードで、2回目の ref2va リクエストの transformer 再ロードが `from_pretrained` の `_caching_allocator_warmup` 内で失敗(`Tried to allocate 15.43 GiB`、実割当は 92.55GB 中 54.44GB)。総常駐予算(89GB)は 95.6GB に十分収まっている。

**原因**: 容量不足ではなく**断片化**。int8 の load/free サイクルの繰り返しが、アロケータに約 37GB の「予約済みだが未割当」の半端なブロックを残していた。

**解決**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を **`import torch` より前**に設定(PyTorch はこれをアロケータ初期化時に一度だけ読む)。単一の仮想アドレス予約を伸縮させる方式に変わり解消。本プロジェクトはヘッドルームが常にタイトなため、opt-in ではなく**無条件**で設定している。

### 4.6 FirstBlockCache が H3 ブロックを知らない

**症状**: `enable_cache()` 後、最初のデノイズステップで `ValueError`。

**原因**: FBC はブロッククラスごとのメタデータを `TransformerBlockRegistry` から引くが、PR ブランチには `MiniMaxH3TransformerBlock` の登録が無い。

**解決**: runner 側で `TransformerBlockMetadata(return_hidden_states_index=0, return_encoder_hidden_states_index=None)` を登録。H3 はクロスアテンションを持たない(テキストは packed sequence の行として扱われる)ため `encoder_hidden_states` の枠は `None` が正しい。**venv の diffusers 本体は無改変**という原則を守った。

### 4.7 hires-fix: ノイズ付き潜在を補間すると市松模様になる

**症状**: 2段生成で、パス2の出力が市松状のノイズに覆われて全面崩壊。

**原因**: **ノイズ付きの `x_t` を空間補間していた**。補間はノイズの高周波成分を増幅する。

**解決**: 補間対象を **x0 推定値**(`denoised_output` 相当)に変更し、パス2の開始シグマで改めてノイズを載せる。参考実装(ComfyUI ノード)も `denoised_output` を使っていることを確認して裏を取った。

### 4.8 `_keep_in_fp32_modules` が dtype 指定を無効化する

**症状**: `from_pretrained(dtype=torch.float16)` で VAE を読んでも、全パラメータが fp32(9.70GB)のまま。

**原因**: `AutoencoderKLMiniMaxH3._keep_in_fp32_modules = ["encoder", "decoder", "quant_conv", "post_quant_conv"]` が実質モジュールツリー全体を覆っており、diffusers の `load_model_dict_into_meta` が `dtype=` を無視して fp32 に強制する。

**解決**: `from_pretrained` の**後**に明示的に `.to(torch.float16)` を呼ぶ。

### 4.9 TE を 50層に削ると、値が静かに別物になる

これは本プロジェクトで最も発見が難しかった罠である。

**背景**: H3 は TE の `hidden_states[50]` しか読まない。51層目以降(14層 ≈ 13GB)は死荷重である。

**症状**: 素直に「50層」に削ると、`hidden_states[50]` が元の 64層モデルと**最大絶対差 約 1.5e4** の別物になる(量子化ノイズの水準ではない)。

**原因**: `Qwen3VLTextModel.forward` は `@capture_outputs` で包まれており、既定の `tie_last_hidden_states=True` が**最後の hidden_states エントリを `outputs.last_hidden_state`(最終 norm 適用後の値)で無条件に上書きする**。64層モデルではインデックス 50 は末尾から遠いので影響しないが、50層に削るとインデックス 50 が**唯一かつ最後**のエントリになり、post-norm の値にすり替わる。

**解決**: **51層**に削る。`layers[50]` は実行されるが出力は読まれない。これによりインデックス 50 が再びスタック中間に戻る。`torch.equal` で bf16・nf4 とも元と**ビット一致**することを確認した。

なお diffusers 側に `num_layers <= 50 → raise` のガードがあるのは、まさにこの境界のためである。**51 は、そのガードを通り、かつ置換を回避する最小値**だった。

**成果**: TE-nf4 21.02GB → **17.45GB**(-17%)、bf16 66.71GB → 53.06GB(-20%)。t2va・ref2va とも**削除有無で mp4 が MD5 一致**。

### 4.10 超短尺で VAE デコードが落ちる

**症状**: 5フレーム(0.208秒)の動画で、デノイズは完走するがデコードが `ValueError: torch.cat(): expected a non-empty list of Tensors` で失敗。

**原因**: `AutoencoderKLMiniMaxH3._decode()` のチャンク数計算 `num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)` が、潜在2フレーム(= `tokens_chunk_size(5)` − `token_drop(3)` ちょうど)で **0** になり、ループが一度も回らない。

**解決**: runner 側の monkeypatch(`H3_VAE_SMALLCLIP_FIX`、既定 ON)で「1チャンク未満なら全トークンを単一の `_decode_clip()` で復号し、`frame_pre_padding`(3)とパディング末尾を切り落とす」分岐を追加。潜在2フレーム × temporal_ratio(4) − 3 = 5 ピクセルフレームで幾何が一致する。通常の動画は元実装へそのまま委譲するため影響なし。

### 4.11 decode 例外時のクリーンアップ漏れ(連鎖 OOM)

**症状**: 上記 4.10 の例外が発生した後、**無関係な後続リクエスト2本が連続で OOM**(約 98.5GB 残留)。

**原因**: 例外が decode フェーズで送出されると、`generate()` 自身の「transformer を再ロードして定常状態へ復帰する」後始末が**実行されない**。TE-nf4 と transformer が両方常駐したままの、bookkeeping と実際の占有が食い違った状態が残る。

**解決**: decode 部を try/except で囲い、**例外時も必ず** `_restore_decode_steady_state()` を通してから再送出する。`generate()` と `generate_ref2va()` の両方に適用。一回限りの失敗注入フック(`H3_DEBUG_FAIL_DECODE`)で「500 応答 → GPU が約 2GB まで解放 → 同一プロセスの次リクエストが正常に 200」を実機確認した。

### 4.12 参照カウントが残って VRAM が返らない

**症状**: `_free_text_encoder(force=True)` はログ上実行されているのに `allocated_gb` が 87.5GB のまま下がらず、直後のデノイズが OOM。

**原因**: `_sync_shared_components_to_ref()` によって `self._pipe_ref.text_encoder` は `self._pipe.text_encoder` と**同一オブジェクト**を指している。片方だけ `del` しても参照カウントが残る。

**解決**: 両方の参照を明示的に削除する。「解放したつもり」が最も危険なパターンであり、以降は解放後に必ず `allocated_gb` を実測して確認する運用にした。

### 4.13 例外の握りつぶしによる誤帰属

**症状**: `AttributeError: 'NoneType' object has no attribute 'enable_cache'`。

**原因**: `load_components()` が内部で例外を `try/except Exception: continue` で握りつぶしており、ロード失敗を呼び出し側へ伝えない。結果 `self._pipe.transformer` が `None` のまま後続処理へ進み、**本当の失敗原因とは無関係な場所**でエラーになる。

**解決**: ロード直後に明示的な None チェックを入れ、失敗をその場で報告する。上流が失敗を飲み込む場合、**呼び出し側で境界を設ける**必要がある。

### 4.14 ref2va の transformer_ref ロード順序

**症状**: 初回 ref2va で `vae._encode_clip()` の 8 秒目に `Tried to allocate 98.00 MiB` で OOM。2回目以降は `_vae_to_gpu()` の時点で `allocated_gb: 98.81` で OOM。

**原因**: 参照 VAE エンコードは `vae` を GPU に載せた状態で行うが、`transformer_ref`(66.3GB)+ TE-nf4(21GB)+ vae ペア(11GB)= 98.3GB が既に予算超過。さらに2回目以降は、前回の decode 後に復元された transformer_ref が常駐したままだった。

**解決**: 参照 VAE エンコードの**前**に `transformer_ref` を毎回明示的に解放し、エンコード完了後にロードする順序へ変更。fl2va のキーフレームエンコードと同型の順序制約である。

### 4.15 GPU 以外の罠

- **HTML5 のフォーム検証が丸め前の入力を弾く**: 幅・高さの `<input type=number>` に `min`/`max`/`step="32"` を付けると、丸める前の端数(例: 1000×700)が `stepMismatch` でブロックされ、**送信自体が起きない**。「偶然 valid な値だけ通る」という分かりにくい症状になる。丸め前提の欄に制約属性を付けないことで解消。
- **チェックボックス OFF は空文字ではなく `0` を明示送信する**: 空文字は「未指定=サーバ既定を使う」と解釈されうるため、`H3_TURBO_LORA=1` で起動したサーバではチェックを外しても turbo が無効にならない恐れがある。
- **ギャラリーの全タイル再構築でちらつく**: 選択のたびに全タイルを作り直すと `<video>` が全数再生成される(95枚で顕著)。選択状態はバッジ/チェックの**差分更新**にする。

### 4.16 その他

- **`_sync_shared_components_to_ref()` の順序**: 共有コンポーネントの ref パイプラインへの同期は、TE ロードの**後**でなければならない。`components` は約束ではなくライブな属性読み取りであり、先に同期すると `text_encoder = None` が焼き付く。`H3_LOWVRAM` は TE を先読みしないため、このモードでのみ `AttributeError: 'NoneType' object has no attribute 'config'` として顕在化した。
- **`MiniMaxH3AdaLayerNormModulation` の `.weight.dtype` 直読み**: Linear をラッパーで置換すると `AttributeError`。`_TurboLoRALinear` に `.weight` / `.bias` の透過プロパティを持たせて解決。姉妹プロジェクトが JoyAI の `PatchifyLinear` で踏んだのと同型。
- **audio_vae を bf16 にしてはならない**: 生成音声が約 20dB 小さくなる既知の上流バグ。fp32 固定を厳守。
- **VAE タイルサイズは効かない**: デコードピーク削減のため 256 → 192 → 128 → 96 と縮小したが、ピーク VRAM は **16.29GB のまま不変**(所要時間だけ 5.9s → 10.5s に悪化)。空間タイリングとは独立した固定コストが律速であると結論。
- **turbo LoRA の形式判定は順序が重要**: comfy 署名(`qkv_proj`)を先に見ないと誤検出する。Ostris 版も `token_refiner.blocks.*` のキーを持つため、プレフィックスだけで diffusers 形式と判定してしまう(実ファイルで再現して修正)。

---

## 5. 機能拡張

### 5.1 2段生成による 2x アップスケール(hires-fix)

低解像度で前半をデノイズ → 映像潜在の x0 推定値のみ空間 2x 補間 → 再ノイズ → 高解像度で仕上げる。学習済みアップスケーラは使わない。

- 768² → **1536²** が 645s / peak 88.0GB(`upscale=0` は 181s / 92.1GB)
- 音声潜在は空間軸を持たないため触らない
- 解像度変更時は `build_packed_sequence()` と `row_timestep_plan` の再構築が必要
- スケジューラの `_step_index` はパス間でリセットしない(シグマ軌道を継続させる)

### 5.2 Ref2VA(オムニ参照生成)

画像9・動画3・音声3(計12)の順序付き参照から動画+音声を生成。PR の機能を配線し、実装中に OOM を3件解決した。参照 VAE エンコードは `vae` を GPU に載せた状態で行う必要があるため、**transformer_ref のロードより前**に実行する順序が必須。

### 5.3 静止画モード(T2I / ref2i)

「超短尺動画 → 中央フレーム取り出し」で画像生成の代用にする。

| フレーム数 | デノイズ | デコード | 品質 |
|---|---|---|---|
| 22 (0.917s) | 29.0 s | 1.8 s | 破綻なし・高品質 |
| 5 (0.208s) | 9.1 s | 0.67 s | 破綻なし(4.10 の修正により初めて成功) |
| 124 (5s、基準) | 197.7 s | 10.6 s | — |

学習分布外(公式下限 5 秒の 1/5)でも品質が崩れず、音声も非無音だった。価値は速度ではなく「**H3 と画風が完全に一致する静止画**」を FL2VA の先頭フレームや Ref2VA の参照として作れる点にある。

参照付き静止画(ref2i)も成立した。参照パッキング(参照行が生成行より多い)と超短尺が重なっても品質は崩れず、参照の衣装・小物・画風が保持された(デノイズ 58.1s、5秒基準の 1/4)。

### 5.4 バッチ生成 — 固定費をバッチ全体で1回に償却

`H3_LOWVRAM=1` の毎リクエスト固定費(約 110 秒)を、**位相を並べ替える**ことで償却する。

```
entry   : [何も常駐せず]
encode  : [TE-nf4]        全場面のエンコードをまとめて実行
denoise : [transformer]   1回ロードして全場面を順にデノイズ
decode  : [vae ペア]      全場面をデコード(場面ごとに保存)
```

場面間で共有される可変状態のリセットが実装の要である。スケジューラはシグマ値が全場面同一なので `_step_index = None` に戻すだけでよく、FirstBlockCache は場面ごとに `_reset_stateful_cache()` する。**この等価性は、逐次生成との mp4/PNG の MD5 一致で実証した**。

| 経路 | 逐次 | バッチ | 短縮 |
|---|---|---|---|
| 静止画 (t2i、3場面) | 157 s/枚 | **67.5 s/枚** | 限界コスト ~31 s/枚 |
| 参照付き静止画 (ref2i、3場面) | 164.9 s/枚 | **116.7 s/枚** | -29% |
| 参照付き動画 (ref2va、2場面) | ~485 s/本 | **401.6 s/本** | -17%(限界 ~32%) |

### 5.5 参照プレフィックスの KV キャッシュ共有

ref バッチのエンコード位相で、参照ラベル+ビジョン(約 4,104 トークン、約 65 秒/場面)の Qwen3-VL エンコードが場面ごとに重複していた。

ref2va のトークン列は「参照が前置・プロンプトは末尾に verbatim 追記」であり、条件付け元は**因果 LM** である。したがって**参照プレフィックスの表現はプロンプトに依存しない**。プレフィックスを1回だけ `use_cache=True` で通して `DynamicCache` に焼き、場面ごとにはプロンプト末尾(14〜33トークン、約 0.2 秒)だけをキャッシュ継続する。

**検証**: プレフィックス部分の `hidden_states[50]` はフル計算と `torch.equal` で**ビット一致**。プロンプト末尾部分には相対 RMS 約 1.5% の丸め差が残るが、これがロジックバグでないことを**ネガティブコントロール**で確定させた — わざと位置オフセットを壊した継続は相対 RMS 27〜30%(20倍)に跳ねる。1.5% は「正しい計算の丸めノイズ」の水準である。

**罠**(transformers 5.14.1 の実装を読んで判明): 継続呼び出しでは `attention_mask=None`、`mm_token_type_ids` / `pixel_values` / `grid_thw` 系は**すべて None** にする(`image_grid_thw` を渡すと `model.rope_deltas` が再計算・上書きされる)。`rope_deltas` は `Qwen3VLModel` の**インスタンス状態**なので、プレフィックス→全継続の間に他の TE 呼び出しを挟んではならない。

**成果**: ref2i バッチのエンコード位相 212.5s → **83.1s**、1枚あたり 164.9s → **116.7s**(-29%)。

---

## 6. 高速化の積み上げ

| 段階 | リクエスト時間(768²・5秒) | 品質検証 |
|---|---|---|
| 初期(bf16 TE 入れ替え) | 245 s | — |
| + TE bnb-4bit 化 | **185 s** | 同一 seed でフレーム・音声とも同水準 |
| + FirstBlockCache (0.05) | デノイズ 157→**118 s** | PSNR 31.8〜34.3dB、音声相関 0.979 |
| + Sage Attention | デノイズ 118→**104 s** | 完全決定論(同一 seed でバイト一致) |
| **現既定** | **~160 s** | |
| + FBC 0.1(opt-in) | ~125 s | 構図が目視で分かる程度にドリフト |
| + Turbo LoRA 8steps(opt-in) | **~88 s** | 基準に迫る |
| + Turbo 4steps(ドラフト) | ~40 s | 柔らかめ |

### 6.1 PSNR を「劣化」と読まないこと

Sage Attention の PSNR は基準比 **21dB**、int8 量子化は **19dB** である。数値だけ見れば大きな劣化に見えるが、**いずれも劣化ではなく軌道のドリフト**である。拡散モデルは初期の微小な差が軌道全体を分岐させるため、PSNR は「同じ絵か」ではなく「同じ軌道か」を測っている。実際、目視では区別できず、同一 seed の2本はバイト一致(完全決定論)する。

この解釈を誤ると、有効な最適化を「品質劣化」として不採用にしてしまう。本プロジェクトでは**目視 + 音声相関 + 決定論性**を併用して判定した。

### 6.2 ビルドで環境を壊さない

SageAttention には Linux 向けの事前ビルド wheel が存在せず(公開されているのは Windows 版のみ)、sm_120 向けにソースからビルドする必要があった。ここで2つの実務的な注意がある。

- **無制限の並列 nvcc はホスト RAM を枯渇させ、システムを巻き添えにする**(過去に事故歴あり)。`MAX_JOBS=4 NVCC_THREADS=2` に絞り、`systemd-run` のメモリ上限付きで実行する
- **`CUDA_HOME=/usr/local/cuda-12.8` の明示が必要**。既定の cuda-13.0 は torch の cu128 と不一致でビルドが失敗する

なお diffusers の hub 系 attention backend(`flash_hub` / `sage_hub`)は、Hub 側に torch 2.9 向けビルドが存在しない(利用可能は 2.10〜2.12)ため**不成立**だった。これは環境の問題ではない。

---

## 7. 検証手法

推測を排するために、次の4手法を使い分けた。

| 手法 | 用途 | 適用例 |
|---|---|---|
| **同一 seed の mp4 MD5 一致** | 数学的な等価性の証明 | TE 51層削除、バッチの位相並べ替え、FBC の per-request リセット、turbo 本実装とスパイクの一致 |
| **PSNR + 音声相関 + 目視** | 近似手法の品質判定 | FBC、Sage、int8、video VAE fp16(PSNR 39.97dB) |
| **VRAM バラスト** | 未所持の GPU 容量での動作検証 | 32GB / 24GB / 22GB / 20GB / 18GB / 16GB の各級 |
| **プローブスクリプト** | 本体を汚さない仮説検証 | 17本(`scripts/probe_*.py`) |

**プローブ方式の要点**は、本体コードを一切変更せずに検証できることである。たとえば超短尺の検証では、diffusers とアプリ双方の 5 秒バリデーションをプローブ内の monkeypatch だけで回避した。仮説が棄却されれば本体は無傷で済む。

**サブプロセス分離**も重要な教訓だった。5フレームの例外が同一プロセスの後続リクエストを汚染した(§4.11)ため、以降のプローブは1生成につき1プロセスに分離している。「例外が既に GPU 状態を壊した**後**の挙動」ではなく「その条件を単独で測る」ことが目的だからである。

### 7.1 プローブスクリプト一覧(`scripts/`)

| スクリプト | 検証対象 |
|---|---|
| `probe_t2va.py` | T2VA 基本経路の回帰。UI より先に動作確認する |
| `probe_group_offload.py` | `device_map="cpu"` + torchao int8 が CPU 上でも量子化されるか(370/370層を確認) |
| `probe_group_offload_forward.py` | §4.3 の pin_memory クラッシュをダミー int8 スタックで単離再現 |
| `probe_group_offload_fix.py` | 修正候補2案(`use_stream=False` / `low_cpu_mem_usage=False`)の対照実験 |
| `probe_vae_tile_size.py` | VAE デコードを単体でベンチし、タイルサイズの効果を測定(無関係と判明) |
| `probe_te_prune.py` 〜 `_4.py` | §4.9 の 50層 vs 51層。`_3` は重みのバイト一致、`_4` は全 hidden_states を保存して分岐点を特定 |
| `probe_turbo_lora_apply.py` | Ostris 版 turbo LoRA の適用単体検証(259層ラップ・forward 差分) |
| `probe_lightx2v_turbo.py` | lightx2v 版の int8 適用可否(Q1)と 4steps 品質・強度スイープ(Q2) |
| `probe_short_frames.py` / `_one.py` | 5/22/124 フレームの超短尺。サブプロセス分離方式 |
| `probe_ref2va_short.py` | Ref2VA × 超短尺(参照パッキングとの重なり)の品質 |
| `probe_ref_prefix_cache.py` | §5.5 のプレフィックス共有のビット一致性・ネガティブコントロール |
| `probe_hires_vae_upscale.py` | hires-fix の補間済み潜在を VAE がデコードできるかの単離検証 |
| `probe_h3official_compliance.py` | §10.2 の故障率ベースライン測定(機械判定規則 F1〜F9) |
| `vram_ballast.py` | ダミー CUDA テンソルで空き VRAM を任意の値まで減らす(バラスト本体) |

全プローブは「本体コード無変更、プローブ内 monkeypatch のみ」または「本体を読んでの独立再実装」の方針で書かれており、**venv の diffusers 本体には一切手を入れていない**。

---

## 8. 環境変化 — GPU 交換(2026-08-07)

作業期間の途中で、実験機の GPU が **RTX PRO 6000 Blackwell 96GB** から **RTX PRO 5000 Blackwell 48GB + RTX 4000 SFF Ada 20GB** の2枚構成へ交換された。

影響は大きい。48GB 単騎では既定モード(bf16 transformer 66.3GB)が**物理的にロードできない**ため、起動に `H3_LOWVRAM=1` か `group` が必須になった。同一設定でも旧 GPU 比で遅くなる(`H3_LOWVRAM=1` の t2va が 215s → 351s)。20GB の RTX 4000 は床(約 18GB)に対し余裕が 2GB しかなく、sm_89 のため sm_120 向けにビルドした SageAttention も使えない。

**この交換が、後続の設計判断を方向づけた**。低 VRAM モードの固定費が支配的になったため、バッチ生成(§5.4)とプレフィックス共有(§5.5)の価値が相対的に大きくなり、また「48GB で turbo を使えないか」という問いが turbo LoRA の再検討(§9)につながった。

---

## 9. Turbo LoRA — 「不可」から「解禁」へ

### 9.1 最初の LoRA(Ostris 版)が int8 で動かなかった理由

`larryvrh/MiniMax-H3-Turbo-Lora` は ComfyUI 由来の**融合 QKV**(`qkv_proj`)を対象にしている。適用には `attn.fuse_projections()` が必要で、その実体は `torch.cat([to_q.weight, to_k.weight, to_v.weight])` である。int8 量子化下では `to_q`/`to_k`/`to_v` は torchao の `Int8Tensor` になっており、**`Int8Tensor` には `aten.cat` カーネルが登録されていない**。

結果、`H3_LOWVRAM=1` / `group` / int8 両常駐のいずれでも `NotImplementedError` で確実に失敗する。これは呼び出し順序の問題ではなく(group offload のフックに到達する前に失敗する)、**カーネルの不在**であるため回避不能と判断し、起動時に拒否する実装にした。

### 9.2 lightx2v 版で解決

`lightx2v/Minimax-h3-Turbo`(DMD 蒸留、Apache 2.0)は **キーが diffusers ネイティブ**で `to_q`/`to_k`/`to_v` が分離している。safetensors のヘッダを直接読んで確認したところ、`ff.net.0.proj` の `lora_B` が `(28672, 128)` と diffusers の SwiGLU(ゲート込み `dim_out*2`)に完全一致しており、**diffusers のモジュール構成そのものに対して学習された**ことが裏付けられた。

融合が不要 → `torch.cat` を呼ばない → **int8 でも適用できる**。実際に int8 transformer へ 312 モジュールを 0.6 秒で適用でき、例外は出なかった。

### 9.3 罠: strength は 0.75 ではなく 0.094

配布元の記載は「4 steps、LoRA strength 0.75」だが、**この値をそのまま生の B·A に掛けると 30 steps でも出力が完全にノイズ化する**。

原因は、ComfyUI 側が alpha を折り込んで適用するのに対し、本実装は生のデルタに直接掛けている点にある。`0.75 × (alpha/rank) = 0.75 × 16/128 ≈ 0.094` という換算仮説を立て、強度スイープで実測した。

| strength | 4 steps の結果 | 音声 rms / peak |
|---|---|---|
| 0.75(記載値そのまま) | **完全にノイズ**(30 steps でも同じ) | 0.083 / 0.43 |
| 0.15 | 良好(背景がやや軟らかい) | 0.069 / 0.79 |
| 0.10 | 良好 | 0.065 / **1.05(クリップ)** |
| **0.094** | **最良**(毛並み・松葉までシャープ) | 0.039 / 0.70 |

30 steps でも崩れることを確認したことで、「ステップ数の問題ではなく強度の問題」と切り分けられた点が重要である。

**成果**(RTX PRO 5000 48GB + `H3_LOWVRAM=1`、768²・5秒): デノイズ **197.7s → 26.1s(7.6倍)**、総所要 **351.4s → 135.2s(2.6倍)**。静止画との組み合わせではデノイズ 5.0 秒。

**残る制限**: `H3_LOWVRAM=group` とは形式を問わず併用不可(`enable_group_offload` の `cpu_param_dict` が有効化時点で固定されるため、後から追加する LoRA バッファが offload サイクルから欠落するリスク。未検証のまま解禁しない)。音声レベルが非 turbo 比で大きめ。ref2va への適用は未検証。

---

## 10. LLM プロンプト強化と、その品質保証

### 10.1 「プロンプトは末尾が最強」

MiniMax 公式の `h3-prompt-writing` スキル(SKILL.md + リファレンス、約 15.8KB)をシステムプロンプトへ全文投入するモードを実装した際、`lang=ja` を指定しても**英語で返ってくる**問題が起きた。

原因は、言語指示をシステムプロンプト**冒頭**に置いていたことである。その後ろに続く 15.8KB の英語リファレンス(英語出力の実例が並ぶ)に上書きされていた。**言語指示をリファレンス本文より後ろ、最終指示として置く**ことで解消した。

この「後方の指示ほど強く効く」という発見は、後に英語版 wrapper の欠陥発見(§10.3)につながる。

### 10.2 「性能限界か」を測ってから判断する

LLM 出力の不具合に対し、「モデルの性能限界か、プロンプトで直るか」を推測で議論せず、**まず故障率を実測**した(5入力 × 3回、gemma4-31B Q4_K_M、n_ctx 7680)。

| 故障クラス | 発生率 |
|---|---|
| 構造・記法(フィールド名・`[Shot n]`・タイムコード・`<d>` タグ・話者ID) | **0/15(0%)** |
| 時間配分 | 6/15(40%) |
| 文脈溢れ | 0/15 |

**「入力が変われば別の問題が出るのでは」という懸念は良い方に外れた** — 違反は散らばらず1クラスに集中していた。さらにその6件は性質が2つに分かれた。

- **入力が物理的に不可能(3/6)**: 台詞の推定発話時間が尺を超えている。実例では 38 文字の独白(推定 9.5 秒)を 9 秒尺で要求しており、**LLM は1ショットに9秒すべてを割く最善の配分をしていた**。公式仕様が台詞の逐語保持を要求する以上、短縮という逃げ道も無い。**どのモデルでも解決不能**
- **LLM の配分ミス(3/6)**: 尺には収まるのに配置が悪い。指摘すれば直せる部類

つまり「LLM の限界」に該当したのは**全体の2割**であり、残りは仕様上の必然と入力側の問題だった。

### 10.3 対策(3層)

1. **バリデータ**(`core/prompt_check.py`): 決定的に判定できる8規則。うち「ショット尺の下限」「台詞がショットに収まる」は**公式仕様に無い、本アプリが足す実用規則**である。公式は「カット時刻が尺内」としか言わないため、5秒尺で 4.5 秒にカット(最終ショット 0.5 秒)という**字義どおりだが使えない出力**を許してしまう。手書きプロンプトにも効く
2. **システムプロンプト改善**: 調査の結果、**英語版 wrapper にはガイド本文より後ろの指示が1つも無かった**。日本語版だけが §10.1 の修正で後方ブロックを持っており、既定は `lang=en` なので、尺の制約を含む全指示が 15.8KB のガイドに埋もれていた。両版の末尾に時間配分6ルールを追加
3. **修復ループ**: 違反を日本語で突きつけて再生成(最大2回)。**違反が増える修復案は破棄**する。入力の実現不可能性は LLM に投げる前に判定し 400 + 助言を返す

**結果**(同一条件):

| | clean | 入力不可を検出 | **違反が残存** |
|---|---|---|---|
| ベースライン | 9/15 | 0 | **6/15** |
| 改善後 | **12/15** | **3/15**(正しい挙動) | **0/15** |

所要時間の中央値は 8.5s → 9.2s(+8%)。修復ループは必要なときだけ発動する。

### 10.4 文脈予算

LLM は n_ctx **7,680**。システムプロンプトの実測は t2va 4,832 トークン(63%)、**ref2va 6,626 トークン(86%)**。t2va は余裕があるが、**ref2va は残り約 1,050 トークンで入力と出力を賄う必要がある**。現状は正常に動作しているが、入力が長い場合の挙動は未確認であり、既知のリスクとして記録する。

---

## 11. 現在の到達点

**環境**: RTX PRO 5000 Blackwell 48GB / RAM 94GB / `H3_LOWVRAM=1`

| 用途 | 所要時間 |
|---|---|
| t2va 単発(品質重視 30steps) | 351 s |
| t2va 単発(turbo 4steps) | **143 s** |
| 静止画 t2i(turbo 4steps) | **94 s** |
| 静止画バッチ(3場面) | 67.5 s/枚(限界 ~31 s/枚) |
| 参照付き静止画バッチ(3場面) | 116.7 s/枚 |
| 参照付き動画バッチ(2場面) | 401.6 s/本(限界 ~330 s/本) |

物語制作のパイプラインとしては「**t2i でキャラクターを確立 → ref2i_batch で全場面の静止画 → ref2va_batch で各場面を動画化**」が一通り繋がった状態にある。

### 11.1 積み残し

**外部イベント待ち**
- diffusers PR #14355 のマージ(§2.1 の理由により安易に追従しない。追従時は同一 seed MD5 で回帰確認)
- lightx2v turbo LoRA は v0.1。コミュニティでの実績蓄積を待って既定 ON を再検討

**未着手の改善候補**
- 量子化済みチェックポイントの事前保存(低 VRAM モードの固定費 90〜100 秒の解消)
- 16GB 級対応(TE のストリーミング実行が必要。現状の床は 17.45GB)
- `torch.compile`(FBC / group offload のフックとの相性が未検証)

**既知の未解決問題**
- `H3_LOWVRAM=group` で t2va 実行後の ref2va がホスト RAM ガードで拒否される(94GB 機でのみ確認。RAM 48GB 以上の余裕がある機では起きない可能性が高いが未検証)
- ref2va のシステムプロンプトが n_ctx の 86% を占める(§10.4)
- turbo の ref2va 適用、turbo × group offload はいずれも未検証

---

## 12. 総括 — 再利用できる教訓

本プロジェクトから、次のプロジェクトへ持ち越す価値のある教訓を挙げる。

1. **推定値を設計の前提にしない**。text_encoder のサイズ(33GB と推定 → 実測 66.73GB)、TE-nf4 のサイズ(17-18GB と推定 → 実測 21GB)、いずれも推定が外れた。実測してから設計する。

2. **数学的な等価性は MD5 で証明できる**。「たぶん同じ」ではなく「バイト一致」で示せる改造(層の削除、位相の並べ替え、キャッシュのリセット)は、必ずそこまで確認する。これにより回帰の不安なく大きな改造ができた。

3. **PSNR の低下が必ずしも劣化ではない**。拡散モデルでは軌道の分岐と品質劣化を区別する必要がある。目視・決定論性・音声相関を併用する。

4. **失敗を検出可能にする方が、失敗をゼロにするより現実的**。LLM の出力品質はプロンプト改善だけでは 100% にならないが、バリデータと修復ループを足せば「妥当な出力」か「実行不能な理由の説明」のどちらかを常に返せる(§10.3)。

5. **例外経路のクリーンアップは、正常系と同じだけ重要**。§4.11 の連鎖 OOM は、例外が1回起きただけで後続の無関係なリクエストを2本潰した。

6. **スパイク → 実測 → 本実装の順序**。「動くはず」で本実装に入らず、本体を汚さないプローブで白黒つけてから進めた。turbo LoRA の int8 対応(§9.2)は、この順序だったからこそ、キー形式という決定的な差分を早期に発見できた。

7. **上流のコードを読む**。本レポートの罠の大半は、diffusers / transformers / torchao の実装を実際に読んで初めて原因が分かったものである。特に §4.9(`tie_last_hidden_states`)と §4.3(`pin_memory`)は、ドキュメントからは決して分からなかった。

---

# 追補: 2026-08-09 の作業(12コミット)

本編は 2026-08-04〜08-08 を対象とする。以降に発生した罠と教訓をここに追補する。
内容は「diffusers マージ版への追従」「transformer 常駐」「UI 再編と国際化」。
**確定した仕様は [docs/TECHNICAL_OVERVIEW.md](../TECHNICAL_OVERVIEW.md) 側にある**ので、
ここには罠と教訓だけを書く。

## A1. `H3_KEEP_TRANSFORMER` — 導出が実測より 1.55GB 保守的だった

デコード位相でも transformer を解放しない構成。導出では
`transformer 34.3 + fp16デコードピーク 11.4 = 45.7GB` と見積もり、実効予算 49.8GB に対し
余裕 ~4GB と判断して着手した。**実測ピークは 44.15GB**(t2va 5秒)で、導出より 1.55GB
小さかった。導出が保守側に外れる分には安全だが、「余裕 4GB」を根拠に可否を判断していた
ので、実測するまで成立を断言しなかったのは正しい運用だった。

t2i 定常 9.7s/枚、t2va 5秒 44.2s。同一 seed で PNG/MP4 の MD5 完全一致。

## A2. マージ版追従で踏んだ `_execution_device` の罠2件(いずれも本編 §4 の再来)

本編 §4 で「最も罠の多い場所」と書いた `_execution_device` が、マージ版でまた2件出た。
**同じ関数が原因でも、原因の形は毎回違う**という実例。

1. **components の順序が変わった**。旧: `text_encoder, tokenizer, processor, vae,
   scheduler, audio_scheduler, transformer, video_processor, audio_vae` →
   新: `image_processor, text_encoder, tokenizer, processor, vae, **audio_vae**,
   scheduler, audio_scheduler, transformer_ref, transformer, video_processor`。
   `audio_vae` が transformer より前に来たため、`_pin_execution_device_to_compute()` が
   text_encoder と vae しか外していないと、**CPU 常駐の audio_vae が最初の nn.Module として
   拾われ** `_execution_device` が `cpu` を返す。新レイアウトステップは出力を
   `_execution_device` に置く契約なので position_ids が CPU に作られ、rope() 内で
   device mismatch。→ 窓で audio_vae も外す。
   **教訓**: この種のピン留めは「外すモジュール名」ではなく「transformer より前に来る
   nn.Module すべて」を外す意図で書くべきだった。順序はライブラリ側の都合で変わる。

2. **バッチ経路では `_execution_device` がそもそも解決できない**。バッチは位相を
   並べ替えて「エンコード位相で layout/latents/timesteps まで済ませ、transformer の
   ロードはデノイズ位相まで遅延」する。この時点では transformer 未ロード + TE デタッチ済みで、
   スキャンが CPU 常駐物かフォールバックの `cpu` に落ちる。値自体は正しく作られるので、
   デノイズ直前に `_scene_state_to_compute()` で明示的に運べば済む。
   **教訓**: 位相を並べ替える設計は、位相に紐づく暗黙の前提(ここでは「この時点で
   計算用GPUが解決できる」)も一緒にずらす。

## A3. 検証環境の制約 — preview のページは `hidden` で描画されない

UI 検証で15分溶かしたので記録する。preview のブラウザは
`document.visibilityState === 'hidden'` の状態にあり、**描画(コンポジット)が走らない**。
帰結:

| 症状 | 原因 | 代替手段 |
|---|---|---|
| `preview_screenshot` が30秒でタイムアウト | 描画されないため合成できない。ギャラリーに `<video>` 165本があると確実に詰まる | DOM を `preview_eval` で実測する(テキスト・位置・高さ・可視性) |
| `IntersectionObserver` が発火しない | 非表示ページでは交差判定が走らない | 遅延読み込みの検証は**ネットワークログ**(mp4 リクエスト数 165→0)で行う |
| `preview_click`(座標クリック)が当たらない | ヒットテストにレイアウト/コンポジットが要る | `element.click()` を `preview_eval` から呼ぶ |

**教訓**: この環境での UI 検証は「見た目」ではなく **DOM の実測とネットワークログ**で行う。
スクリーンショットが撮れないことをコードの不具合と誤認しないこと(実際、遅延読み込みが
「効いていない」と一度誤判定しかけた)。

## A4. エージェント運用の事故 — 再委任の連鎖(**再発**)

[[stdgen-spike-verdict]] に記録済みの「孫エージェント事故」が再発した。調査を委任した
エージェントが、自分で作業せず**さらに別のエージェントを起動**し、それが完了報告として
「エージェントを起動しました」とだけ返す、という連鎖が3回起きた。停止指示を出したが、
1体は生き残って完走し、約23万トークンを重複消費した。

**対策(後半の委任で実際に機能した)**: 委任プロンプトの冒頭に
**「あなた自身が Read/Edit/Bash で完結させること。サブエージェントの起動(Agent ツール)は
禁止」**を明記する。以後の6件の委任では一度も再発していない。

**副次的な教訓**: 事故った調査の成果物自体は有用だった(私の直接調査より深い diff レベルの
監査で、私の報告2点を訂正できた)。ただしトークンコストは重複していたので、**委任前に
禁止事項を書く方がはるかに安い**。

## A5. 移行の回帰は「基準値を先に採る」

diffusers を上げる前に、**旧ピンに戻して基準の MD5 を採取した**(t2i/t2va/バッチは
既存の当日実測を流用、ref 系は main + abc5e9b に戻して15分かけて採取)。この順序でないと
「移行後の出力が正しいか」を判定する術がなくなる。

採取した基準値は **[docs/internal/regression_baselines.json](regression_baselines.json)**
に保存した(scratchpad は消えるため)。次に diffusers を上げるときは、このファイルの
`cases` を同じ条件で再生成して md5 を突き合わせればよい。abc5e9b で採取し、f37ab93 追従後も
全件一致したので、**現行 f37ab93 の期待値でもある**。

## A6. ドキュメントの層分け

本レポートは「作業記録(内部資料)」に降格し、外部向けの技術説明は
[docs/TECHNICAL_OVERVIEW.md](../TECHNICAL_OVERVIEW.md) に分離した。両者は読者が違う:
仕様・性能を知りたい人に失敗談の混じった経過を読ませない、逆に**同じ罠を避けたい開発者に
とっては経過こそが価値**、という切り分け。

日英両対応も同時に行った(別ファイル + 相互リンク。同一ファイル内併記は、実測値を
追記し続けるこのリポジトリでは必ず片方が腐るため採らなかった)。

---

# 追補: 2026-08-11〜12(低VRAMゴール達成戦)

本編・A追補は 48GB 機での固定費削減までを扱う。ここでは **投影TE(Qwen3-VL-4B + 学習済み
線形写像)を軸に、最終ゴール2つ(実16GBカード単体で全機能 / 8GiB×2 で t2va まで)を
達成する過程で踏んだ罠と教訓**を記録する。確定仕様は
[docs/TECHNICAL_OVERVIEW.md](../TECHNICAL_OVERVIEW.md) と
[docs/RESIDENCY.md](../RESIDENCY.md) 側にあるので、ここには罠と教訓だけを書く。

## B1. plain モード × `H3_TE_DEVICE` の device 不一致(`_execution_device` 罠の新変種)

投影TE(4B NF4)を `H3_TE_DEVICE=cuda:1` に外出しして **lowvram なし(plain)**で走らせた
初走行で、rope の中で device mismatch クラッシュした。原因は本編 §4・A2 と同じ
`_execution_device` だが、**発火位置がまた違った**。

- TE を切り離すと `_execution_device` は CPU 常駐の audio_vae に落ちる(A2 の1件目と同型)。
  A2 で `_pin_execution_device_to_compute()` に audio_vae 除外を足したが、それは
  lowvram 経路の窓だけだった。
- plain(非 lowvram)には別の2分岐(catch-all / bnb-4bit+fl2va)があり、その layout〜
  timesteps が **ピンの窓の外**にあった。plain × `H3_TE_DEVICE` は今回が初走行で、この
  組合せでしか通らない。
- **修正**: `self._te_external` が真のときだけ、これらの分岐も
  `_pin_execution_device_to_compute()` で包む(TE を同居させる通常経路はバイト単位で不変)。

**教訓**: A2 で得た「transformer より前の nn.Module を全部外す」という意図は正しかったが、
**その意図を適用する窓が経路ごとに複数ある**ことを見落としていた。ピン留めの導入時は
「どの経路がこの窓を通るか」を経路の全列挙で確認する。新しい env の組合せは、既存コードの
「今まで一度も通っていない分岐」を初めて通す。

## B2. 上流投影行列の silent 再校正(同名でない更新だが、実質別の関数)

`H3_TE_PROJ` の既定ファイルが 404 になった。配布元が旧
`h3_qwen3vl_4b_tap24.safetensors` を `obsolete/` へ退避し、**再校正版**
`mmh3-4b-ClipProj.safetensors` に置換していた(学習 1,666→5,664 プロンプト、
cos_test 0.711→0.717)。

- ファイル名が変わったので 404 で気づけたが、**中身も別物**だった。旧行列 W との cosine は
  0.9596 で、これは「実質別の関数」の水準。名前だけ見て「同じものの再アップロード」と
  扱っていたら、品質の差分を投影の性質と混同しかねなかった。
- **どう確定したか**: 旧行列がローカル HF キャッシュ(snapshot `3f762f19`)に残存していた
  ため、新旧を同 seed で走らせて **PNG MD5 で別物であること・W のハッシュで別テンソルで
  あること**を確定できた。旧行列は `H3_TE_PROJ` に絶対パスを渡せば再現可能。
- 新旧とも 32B 基準に対する PSNR は同水準(22.49→22.36dB)、新旧 NF4 同士の PSNR は
  39.29dB で「再校正+GPU違いでも実質同一の画」。

**教訓**: **配布物の同名/別名更新を疑え**。HF リポジトリのファイルは黙って中身が変わる。
ローカルキャッシュに旧版が残っていると新旧のハッシュ比較ができる — 追従の前に
`~/.cache/huggingface` の snapshot を消さないこと(消す前にハッシュを控える)。

## B3. sage attention は sm_120 専用ビルド → sm_89 では `H3_ATTN_BACKEND=default` 必須

増設した RTX 4060 Ti は **sm_89**。本アプリの sage は sm_120 向けにソースビルドした
2.2.0 で、sm_89 カード上では使えない。16GB 単体検証の起動は
`H3_ATTN_BACKEND=default`(SDPA)に戻す必要がある。

**教訓**: attention バックエンドはカードの compute capability に紐づく。GPU を替えたら
まず sage の対応 arch を確認し、非対応なら SDPA に落とす(既定 `sage` のまま起動すると
落ちる)。後述の B4 とも絡む: SDPA へ落とすと軌道が変わり FBC も効かなくなる。

## B4. int8+SDPA 軌道では FBC が0スキップ+同一seedで構図が変わる → 構成間品質は目視で

16GB 単体(int8 group + SDPA)の生成で、**FBC が1ステップも省けていなかった**
(`cache_skipped_steps: 0`、閾値 0.05)。PRO 6000 + sage + bf16 の軌道では大半のステップが
省けていたのと対照的で、int8+SDPA の軌道では残差が閾値を下回らない。

同時に、PSNR は vs 32B 基準 7.40dB / vs 前段 7.43dB と数値上は壊滅だが、**目視では劣化
ではなかった**: 前段(bf16+sage)は seed 4242 でアニメ調の夕景の湖(プロンプトの
"photorealistic"/"snow-capped peaks" 不反映)に収束していたのに対し、今回(int8+group+SDPA)
は写実的な雪山と朝霧の湖で、**プロンプト忠実度はむしろ向上**していた。

**教訓**: int8+SDPA は生成の軌道を別のアトラクタへ移す。**構成をまたぐ PSNR/MD5 比較は
もともと成立しない**(既知の int8 軌道分岐 ~19dB のさらに先)。構成間の品質は目視で判断
すること。FBC のスキップ率も軌道依存であり、「効かない」=「壊れている」ではない
(閾値調整で最大2倍の短縮余地はあるが未検証、品質とのトレードオフ)。

## B5. 遅さの原因切り分け — この箱の2番スロットが Gen3 x4(カードの性能と混同しない)

16GB 単体で t2i 16.5s/step、t2va 49.8s/step と非常に遅かった。group offload は毎ステップ
int8 重み ~34GB を CPU→GPU に流すが、4060 Ti が刺さっているスロットは
**Gen3 x4(実効 ~3.5GB/s)**。転送だけで ~10s/step になり、実測と整合する。

- **どう確定したか**: `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current`
  でスロットが Gen3 x4 であることを直接確認した。デノイズ計算そのものではなく毎ステップ
  転送が律速。
- **Gen4 x16 のまともなスロットなら転送は ~1/8** になるので、この時間は「16GB カードの
  性能」ではなく「この箱のスロット」の値。t2va の 49.8s/step はこれに 124 フレーム分の
  SDPA 計算(sm_89)が乗ったもの。

**教訓**: group モードの速度は PCIe 帯域律速。遅さを見たら**カードの演算性能ではなく
スロットの Gen/幅を疑う**(nvidia-smi で確認できる)。この箱固有の値をカードの一般的性能と
して記録しないこと。

## B6. デコード末尾の一括 fp32 化 838MiB でデノイズ完走後に OOM → 逆正規化を CPU へ

8GiB×2 の t2va 初回試行は、**デノイズを 29/29 完走した後、デコード末尾の1行**で OOM した。
上流 `decoders.py` の `MiniMaxH3VideoDecodeStep.__call__` 最終行
`(video.float() * pixel_std + pixel_mean).clamp(0,1)` が fp16 の全長デコード結果を GPU 上で
一括 fp32 化する — 768²・124フレームで 124×768×768×3×4B = **838MiB** の一時確保
(OOM メッセージの "Tried to allocate 838.00 MiB" と完全一致)。本編で
`frames_to_uint8` に潰した「デコード末尾の一括変換」と同族の問題が、正規化側にも残っていた。

- **対処**: venv 無改変ルールどおり runner 側サブクラス(`_cpu_norm_video_decode_step()`、
  f37ab93 の `__call__` を複製して1点だけ変更): **fp16 のまま CPU へ移してから逆正規化**。
- **等価性の論法**: 要素毎の fp32 mul/add/clamp は縮約も FMA 融合もないため、CPU と GPU で
  IEEE754 の丸めが一致する → 出力はビット単位で同一。**適用前後で同一 seed の PNG MD5
  完全一致(`1a2a136b61234b4917465604ac35cca2`)を実測確認**した(「同じはず」ではなく
  バイト一致で証明)。追加コストは fp16 全長 ~420MiB の PCIe 転送1回のみ。
- 全経路(t2va/t2i/ref2va/バッチ)に無条件適用。どの構成でもデコード位相のピークから
  全長 fp32 数本ぶんが消える。

**教訓**: 「デノイズが通ってもデコードで落ちる」パターンは末尾の一括 dtype 変換を疑う。
要素毎演算の CPU/GPU 等価性は縮約が無いことが条件で、これが成り立つときは丸めが一致する
ので **MD5 で証明できる**(近似ではなく等価な移設)。

## B7. `H3_VIDEO_VAE_FP16=1` × 参照の dtype 即死(decode は autocast あり・encode は無し)

参照系(ref2va / fl2va)を `H3_VIDEO_VAE_FP16=1` 構成で走らせると、VRAM 量に関係なく
(96GB でも)`Input type (float) and bias type (c10::Half) should be the same` で即死した。

- **なぜ**: `H3_VIDEO_VAE_FP16=1` は VAE 重みを fp16 に恒久キャストする。**デコード**は
  上流ステップが自前の fp16 autocast を張るので整合するが、**エンコード**側
  (encoders.py の `encode_vae_condition` — ref2va の参照と fl2va のキーフレーム条件付けが
  使用)は autocast なしで、内部で明示的に fp32 化した画素を `vae.encode()` に渡す。
  上流の **decode は autocast あり・encode は無しという非対称**が原因。
- **なぜ今まで潜伏したか**: これまでの ref2va 回帰がすべて fp32 VAE 構成だったため、
  **fp32 VAE 構成の回帰だけでは永遠に見つからなかった**。低VRAMゴールで初めて
  `H3_VIDEO_VAE_FP16=1` × 参照ありを併用して発覚した。
- **修正**: `_load_vae` で fp16 化直後に `vae.encode` をデコード側と対称の fp16 autocast で
  ラップ。精度は設計内(`encode_vae_condition` は元々、結果を
  `latents.to(torch.float16).float()` と自分で fp16 に丸めてから返す)。

**教訓**: 上流が **decode/encode で autocast の張り方が非対称**なことがある。片方の経路
(fp32 VAE)だけで回帰を回していると、もう片方の経路特有のバグは構造的に検出できない。
回帰マトリクスは「フラグ × 機能」の直積で、少なくとも危険な組合せ(fp16 VAE × 参照)は
明示的に含める。

## B8. group の pinned ~34GB がホストキャッシングアロケータに残留(empty_cache はデバイス側のみ)

group モードで **t2va→ref2va のモード切替が恒久的に不可能**だった。原因は VRAM ではなく
**ホスト側 pinned RAM の残留**。

- group offload(use_stream=True)は int8 重み ~34GB を **pinned memory** に置く。
  `_free_transformer` の del+gc ではページが **torch のホスト側キャッシングアロケータに
  保持されたまま OS に返らず**(`torch.cuda.empty_cache()` はデバイス側のみ)、
  MemAvailable が ~34GB 不足したままになる。後続の transformer_ref ロードが RAM ガード
  (40GB 要求)で拒否される。
- **どう確定したか**: 解放後の avail を実測すると 38.6GB(RssShmem 45.6GB 残留)。
  「解放したのに RAM が戻らない」ことを数値で確認できた。
- **修正**: `_free_transformer` / `_free_transformer_ref` に、group モード時のみ
  `torch._C._host_emptyCache()`(私有APIのため `getattr` ガード付き)を追加。修正後は
  解放直後に avail が **85.8GB** へ完全回復し、t2va→ref2va 切替が group モードで初めて成立。

**教訓**: **ホスト側 pinned キャッシュはデバイス解放と別勘定**。`empty_cache()` はデバイス
側のみで、pinned ホストページは `_host_emptyCache`(私有API)でしか OS に返らない。
group モードのように大量の pinned を使う構成では、モード切替時にこれを明示的に返す。
私有APIなので `getattr` で存在を確認してから呼ぶ(torch バージョン差で消えうる)。

## B9. VRAMバラスト模擬の方法論(実カードよりわずかに厳しめになる理由)

8GiB×2 の検証は、増設した実 16GB カードと 96GB カードを
`scripts/vram_ballast.py --target-free-gb 7.9`(**GiB 単位**)で headless 8GiB カード相当に
制限して模擬した。計算側 = 4060 Ti、TE側 = PRO 6000
(`CUDA_VISIBLE_DEVICES=1,0` でアプリの cuda:0/cuda:1 に割り当て)。

- バラストは指定した空き量を残して残りを埋める。`--target-free-gb` は GiB 指定なので、
  8GiB カードの実効予算より **わずかに厳しめ**になる(バラスト自身のアロケータ断片化や
  CUDA コンテキストが、実カードの「素の空き」より少し多く食う方向)。つまり
  「バラストで通れば実カードでも通る」側に倒れており、可否判定として安全側。
- 実カード(ゴールA)とバラスト(ゴールB)を両方回したのは、**バラストが実カードを
  正しく近似するか**を実カード側で裏取りするため。

**教訓**: バラスト模擬は実カードより厳しめに倒れるので**成立判定には安全**だが、
**速度は模擬にならない**(PCIe 帯域・compute capability は実スロット/実カード固有)。
容量の可否はバラストで、速度は実カードで測る。

## B10. サブエージェント検証の運用(Bash 10分上限 → nohup+ポーリング)

低VRAMの生成は 1本 25〜66 分かかる(この箱の Gen3 x4 スロットのため)。検証を委任した
サブエージェントの Bash には 10 分上限があり、そのままでは長尺生成を待てない。

- **対処**: 生成は `nohup ... &` でバックグラウンド起動し、`logs/server.log` を
  ポーリングして完了を検出する。curl が途中で切れても**サーバ側の生成は継続**し、
  結果は `outputs/` と `server.log` から回収できる(生成はグローバルロックで直列化されて
  いるので、切断でジョブが失われることはない)。
- 委任プロンプト冒頭の「再委任禁止・自身で完結」も継続適用(A4 の教訓)。

**教訓**: 長尺ジョブの検証は「同期で待つ」のではなく「起動+ポーリング+ログ回収」で組む。
curl の切断とサーバ側ジョブの生死を混同しない(HTTP コネクションはジョブの寿命ではない)。

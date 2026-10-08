#!/usr/bin/env bash
# minimax-h3 バックエンド起動スクリプト (diffusers-movie-server 統合用)
#
# - 内部ポートは既定 8631 (gateway 8630 からのパススルー先)。H3_PORT で上書き可。
# - H3_* 環境変数は呼び出し側 (gateway / 手動起動) から素通しする。
#   秘匿値 (LLM URL 等) はここにハードコードしない。
# - venv は当面 /home/animede/minimax-h3/venv への symlink (Phase 0 方針、
#   完全独立再構築の手順は VENV_REBUILD.md 参照)。
set -euo pipefail

cd "$(dirname "$0")"

PORT="${H3_PORT:-8631}"

# --- MV 本番の既定値 (2026-09-02 追加) -------------------------------------
# いずれも「コード既定と異なるが、実測に基づいて本番で使うと決めた値」。
# 以前はプロセスの環境変数にしか存在せず、バックエンドを再起動するたびに
# 失われていた (実際に 2026-09-02 に2回失いかけた)。ここに書いておけば
# 手動起動でも gateway 起動でも効く。すべて `${VAR:-...}` なので、
# 呼び出し側 (gateway のプリセット/トグル、手動 export) が明示すればそちらが勝つ。
#
#   H3_TURBO_LORA_FILE          core/runner.py の既定は fl2v v1.0 768p。
#                               ref2v 専用蒸留を 2026-08-26 に A/B して採用
#                               (seed 9999 の二重露光が解消。commit 025e0c0)。
#                               2026-09-03 追記: ref2v v1.0(8step・768p)公開、当初は
#                               denoise 約2倍のため v0.1 のままとしていた。
#                               2026-09-09 更新: 9/3 の A/B は shift 自動判定バグ
#                               (`_768p`→6 が ref2v v1.0(訓練12)に誤適用)の条件で
#                               走っていた。バグ修正後の A/B(outputs/
#                               ab_8step4_20260909/)で **v1.0 を 4steps で回す構成が
#                               v0.1 4step と同コストで品質上(ユーザー判定: v0.1 は
#                               オーバーアクション、v1.0 はリップシンク問題なし)**
#                               となり、フリート既定を v1.0(steps 4)へ差し替えた。
#   H3_TURBO_STEPS_DEFAULT      turbo 時の既定ステップ数。2026-09-09 から v1.0
#                               (8step 蒸留)を 4steps で回す運用が既定
#                               (同日 A/B で 8steps の優位は確認できず。
#                               8steps は V3 の「バランス」tier からのみ選択)。
#   H3_VOCAL_LOCK               生成音声行の凍結。リップシンクのずれ対策 (commit 7faab45)
#   H3_REF_PREFIX_CACHE_SINGLE  参照 prefix キャッシュを1件に絞る
#
# **H3_TRANSFORMER_QUANT はここに書かないこと**: gateway の H3 プリセットは
# `96gb` が env={} (bf16)、`96gb-int8` だけが int8 を渡す設計になっている。
# ここで既定値を与えると `96gb` を選んでも黙って int8 になってしまう。
export H3_TURBO_LORA_FILE="${H3_TURBO_LORA_FILE:-minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors}"
export H3_TURBO_STEPS_DEFAULT="${H3_TURBO_STEPS_DEFAULT:-4}"
export H3_VOCAL_LOCK="${H3_VOCAL_LOCK:-1}"
export H3_REF_PREFIX_CACHE_SINGLE="${H3_REF_PREFIX_CACHE_SINGLE:-1}"
# ---------------------------------------------------------------------------

exec venv/bin/python -u -m uvicorn app:app --host 0.0.0.0 --port "$PORT"

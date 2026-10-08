"""Kijai/MiniMax-H3-experimental の w4a8_mixed(comfy_quant 形式)を、本プロジェクトの
pruned CKW4A8 キャッシュ形式(core/pruned.py の CKW4A8Linear 用 state_dict)へ変換する。

対象ファイル(2026-10-08 時点、両者は完全同一構造を header 突合で確認済み):
  - minimax_h3_fl2va_pruned_w4a8_mixed.safetensors  (base: t2va/fl2va 用、12.5GB)
  - minimax_h3_ref2va_pruned_w4a8_mixed.safetensors (ref2va 用。自前 ck ref キャッシュの
    検証用 — 同じ multimodalart bf16 由来なので s_channel が行単位でほぼ一致するはず)

形式対応(docs/h3-w4a8-kijai-header.json と自前キャッシュの実測から導出):
  - 量子化テンソル名: weight→qdata / weight_s_rel→s_rel / weight_s_channel→s_channel /
    weight_codebook→codebook(group16・codebook・convrot256 = 自前 CKW4A8Linear と同一)
  - 融合 QKV(qkv_proj [3*7168, ...])は out 次元の行3分割で to_q/to_k/to_v へ(無損失。
    codebook は分割後の3層で共有値を複製)。mlp.fc1/fc2 → ff.net.0.proj / ff.net.2。
  - blocks.N → transformer_blocks.N、token_refiner.blocks.N → token_refiner.refiner_blocks.N
    (refiner は bf16 のまま、同じ行3分割)。
  - adaln_proj.linear.bias(折り込み済み定数項)→ adaln_proj.folded_bias(f32)。
    final_layer.adaln_proj.* → norm_out.*。adaln_t_table → time_embedder.table。
  - adaln_basis / adaln_mean は Kijai のファイルに存在しない(彼の折り込み分解の基底は
    非公開)。**forward では未使用**(remote code 上 LoRA 射影 project_adaln_lora 専用)
    なのでゼロで登録する。⇒ この変換物には released(2688幅)AdaLN LoRA を射影できない
    (turbo LoRA の adaln_proj 因子はスキップする運用が前提)。
  - rope.inv_freq は本プロジェクトの remote code では動的生成のため破棄。
  - comfy_quant メタ blob(U8)は破棄(レイアウト情報は上記で固定)。

出力: <dst>/pruned_ck_w4a8_state.pt + config.json(multimodalart pruned の
transformer_ref/config.json を複製 — base/ref 同一アーキ)。meta.json は書かない
(ランナー側のロード経路が専用 env で明示される設計のため)。

使い方:
  venv/bin/python scripts/convert_kijai_w4a8.py --src <kijai .safetensors> --dst <cache_dir>
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open

QKV_SPLIT = {"to_q": 0, "to_k": 1, "to_v": 2}
QUANT_SUFFIX = {
    "weight": "qdata",
    "weight_s_rel": "s_rel",
    "weight_s_channel": "s_channel",
    "weight_codebook": "codebook",
}


def convert(src: Path, dst: Path, pruned_config_dir: Path) -> None:
    out: dict[str, torch.Tensor] = {}
    dropped: list[str] = []

    def put(key: str, tensor: torch.Tensor):
        assert key not in out, f"二重書き込み: {key}"
        out[key] = tensor

    with safe_open(str(src), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        scale_keys = {k for k in keys if k.endswith(".weight_scale")}
        for k in keys:
            if k.endswith(".comfy_quant") or k == "rope.inv_freq" or k in scale_keys:
                dropped.append(k) if k not in scale_keys else None
                continue
            t = f.get_tensor(k)
            # ref2va 版の token_refiner は int8 per-row scale + ConvRot(group 256)
            # **回転済み**で格納される(comfy_quant メタ: {"format":"int8_tensorwise",
            # "convrot":true})。単純な t*scale のデクオンタイズでは回転空間のままになり
            # 出力が壊れる(実測 corr 0.06)ため、未対応として明示拒否する。
            # fl2va(base)版の refiner は bf16 のままなのでこの分岐自体を通らない。
            if f"{k}_scale" in scale_keys:
                raise RuntimeError(
                    f"{k}: int8+weight_scale(ConvRot 回転済み)の refiner は未対応です。"
                    "逆回転の実装が必要(fl2va 版は bf16 のため対象外)。"
                )
            # --- トップレベルの単純リネーム ---
            top = {
                "adaln_t_table": "time_embedder.table",
                "audio_patch_proj.weight": "audio_proj_in.weight",
                "audio_patch_proj.bias": "audio_proj_in.bias",
                "condition_proj.weight": "context_embedder.weight",
                "condition_proj.bias": "context_embedder.bias",
                "final_layer.adaln_proj.linear.weight": "norm_out.linear.weight",
                "final_layer.adaln_proj.linear.bias": "norm_out.folded_bias",
                "final_layer.norm.weight": "norm_out.norm.weight",
                "final_layer.audio_out.weight": "audio_proj_out.weight",
                "final_layer.audio_out.bias": "audio_proj_out.bias",
                "final_layer.video_out.weight": "proj_out.weight",
                "final_layer.video_out.bias": "proj_out.bias",
                "video_patch_proj.weight": "proj_in.weight",
                "video_patch_proj.bias": "proj_in.bias",
            }
            if k in top:
                nk = top[k]
                if nk == "norm_out.folded_bias":
                    t = t.to(torch.float32)
                elif nk == "norm_out.linear.weight":
                    t = t.to(torch.bfloat16)
                put(nk, t)
                continue
            # --- blocks / token_refiner ---
            parts = k.split(".")
            if parts[0] == "token_refiner":
                # token_refiner.blocks.N.xxx -> token_refiner.refiner_blocks.N.xxx(bf16、量子化なし)
                if parts[1] == "final_norm":
                    put("token_refiner.final_norm.weight", t)
                    continue
                assert parts[1] == "blocks", k
                idx, rest = parts[2], ".".join(parts[3:])
                base = f"token_refiner.refiner_blocks.{idx}"
                if rest == "attn.qkv_proj.weight":
                    q, kk, v = t.chunk(3, dim=0)
                    for name, chunk in (("to_q", q), ("to_k", kk), ("to_v", v)):
                        put(f"{base}.attn.{name}.weight", chunk.contiguous())
                elif rest == "attn.out_proj.weight":
                    put(f"{base}.attn.to_out.0.weight", t)
                elif rest in ("attn.q_norm.weight", "attn.k_norm.weight"):
                    which = "norm_q" if "q_norm" in rest else "norm_k"
                    put(f"{base}.attn.{which}.weight", t)
                elif rest == "mlp.fc1.weight":
                    # GEGLU 融合の半分順が diffusers と逆(main blocks の量子化 fc1 と
                    # 同じ取り違え。multimodalart ref の refiner と相関比較で確定:
                    # as-is corr 0.0004 / 入替 0.9995)。
                    hh = t.shape[0] // 2
                    put(f"{base}.ff.net.0.proj.weight",
                        torch.cat([t[hh:], t[:hh]], dim=0).contiguous())
                elif rest == "mlp.fc2.weight":
                    put(f"{base}.ff.net.2.weight", t)
                elif rest in ("norm1.weight", "norm2.weight"):
                    put(f"{base}.{rest}", t)
                else:
                    raise RuntimeError(f"未知の refiner キー: {k}")
                continue
            assert parts[0] == "blocks", f"未知のキー: {k}"
            idx = parts[1]
            base = f"transformer_blocks.{idx}"
            rest = ".".join(parts[2:])
            if rest.startswith("adaln_proj.linear."):
                if rest.endswith(".weight"):
                    put(f"{base}.adaln_proj.linear.weight", t.to(torch.bfloat16))
                else:  # bias = 折り込み済み定数項
                    put(f"{base}.adaln_proj.folded_bias", t.to(torch.float32))
                continue
            if rest in ("norm1.weight", "norm2.weight"):
                put(f"{base}.{rest}", t)
                continue
            if rest in ("attn.q_norm.weight", "attn.k_norm.weight"):
                which = "norm_q" if "q_norm" in rest else "norm_k"
                put(f"{base}.attn.{which}.weight", t)
                continue
            # 量子化テンソル(qkv_proj / out_proj / mlp.fc1 / mlp.fc2 × 4サフィックス)
            mod, suffix = rest.rsplit(".", 1)
            new_suffix = QUANT_SUFFIX[suffix]
            if mod == "attn.qkv_proj":
                if new_suffix == "codebook":
                    for name in QKV_SPLIT:
                        put(f"{base}.attn.{name}.codebook", t.clone())
                else:
                    chunks = t.chunk(3, dim=0)
                    for name, pos in QKV_SPLIT.items():
                        put(f"{base}.attn.{name}.{new_suffix}", chunks[pos].contiguous())
            elif mod == "attn.out_proj":
                put(f"{base}.attn.to_out.0.{new_suffix}", t)
            elif mod == "mlp.fc1":
                # Kijai の fc1 は GEGLU 融合の2半分が diffusers の ff.net.0.proj と
                # 逆順([up; gate] vs [gate; up])。出力次元 (dim 0) で半分を入れ替える。
                # qdata [N,K/2] / s_rel [N,K/16] / s_channel [N] はいずれも dim0=出力行。
                # codebook [16] は行に依存しないのでそのまま。
                # (実機で特定: 入替え無しだと qdata 一致率 0.6%、入替えで 99.4% —
                #  s_channel は行エネルギーで入替えに無関係に相関 ~0.1 を示すため
                #  この取り違えは s_channel 検証では検出できない。出力は全面ノイズ化する)
                if new_suffix != "codebook":
                    h = t.shape[0] // 2
                    t = torch.cat([t[h:], t[:h]], dim=0).contiguous()
                put(f"{base}.ff.net.0.proj.{new_suffix}", t)
            elif mod == "mlp.fc2":
                put(f"{base}.ff.net.2.{new_suffix}", t)
            else:
                raise RuntimeError(f"未知の量子化モジュール: {k}")

    # LoRA 射影専用バッファ(forward 未使用)。multimodalart の adaln_affine.safetensors が
    # まさに adaln_basis + adaln_mean の実体なのでそれを載せる。注意: Kijai の base は
    # adaln_t_table が ref と完全一致ではない(max diff 0.014 実測)ため、base では
    # この basis/mean による AdaLN LoRA 射影は「近似」になる(forward には無関係)。
    affine = pruned_config_dir / "adaln_affine.safetensors"
    with safe_open(str(affine), framework="pt", device="cpu") as af:
        put("adaln_basis", af.get_tensor("adaln_basis").to(torch.float32))
        put("adaln_mean", af.get_tensor("adaln_mean").to(torch.float32))

    dst.mkdir(parents=True, exist_ok=True)
    torch.save(out, str(dst / "pruned_ck_w4a8_state.pt"))
    shutil.copyfile(pruned_config_dir / "config.json", dst / "config.json")
    print(f"変換完了: {len(out)} tensors -> {dst}(破棄 {len(dropped)}: comfy_quant メタ / rope.inv_freq)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--config-dir", default=None,
                    help="pruned config.json の場所(既定: multimodalart キャッシュの transformer_ref)")
    a = ap.parse_args()
    cfg = Path(a.config_dir) if a.config_dir else next(
        Path("/home/animede/.cache/huggingface/hub/models--multimodalart--MiniMax-H3-Pruned/snapshots").glob("*/transformer_ref")
    )
    convert(Path(a.src), Path(a.dst), cfg)

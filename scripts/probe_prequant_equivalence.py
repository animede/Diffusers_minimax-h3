"""案1の等価性検証: 事前量子化して保存した TE が、現行経路と同じ出力を出すか。

`probe_prequantized_ckpt.py` でロードが 66.9s -> 2.6s (25.7倍) になることは分かったが、
**速いだけでは採用できない**。このプロジェクトの流儀に従い、出力が数値的に同一である
ことを確かめる。

比較するのは `hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER]`(H3 が条件付けに使う唯一の
値)。bnb-4bit の量子化は決定的なはずなので、保存→再ロードでビット一致するのが期待値。
一致しなければ「速いが等価でない」= 採用不可(または要調査)と判定する。

手順:
  1. 現行経路(元重み + その場でNF4量子化)で TE をロードし、hidden_states[50] を取る
  2. save_pretrained で保存
  3. 解放し、保存物から再ロードして同じ入力で hidden_states[50] を取る
  4. torch.equal / max abs diff で比較

実行:
  CUDA_VISIBLE_DEVICES=0 H3_TE_PRUNE=1 venv/bin/python scripts/probe_prequant_equivalence.py
"""
import gc
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

OUT_DIR = BASE_DIR / "outputs" / "prequant"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR = OUT_DIR / "text_encoder_nf4_equiv"
KEEP = os.environ.get("H3_PROBE_KEEP") == "1"

PROMPTS = [
    "A red fox walks through a snowy forest at dawn, birds chirping",
    "制服の少女が坂道を歩く。The girl (S1) says: <d>[Japanese] 今日はいい天気だね。</d>",
]


def encode_all(pipe, device):
    """各プロンプトの hidden_states[50] を取る(H3 が実際に読む値)。"""
    import torch

    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3TextEncoderStep

    out = []
    for p in PROMPTS:
        with torch.no_grad():
            emb, tags = MiniMaxH3TextEncoderStep.encode_prompt(
                pipe, p, None, device=device, dtype=torch.bfloat16
            )
        out.append((emb.detach().to("cpu"), tags.detach().to("cpu")))
    return out


def main():
    import torch

    from core.runner import DEVICE, MiniMaxH3Runner

    runner = MiniMaxH3Runner(OUT_DIR)

    # --- 1. 現行経路 ---
    logging.info("=== 現行経路でロード ===")
    t0 = time.time()
    runner._load_text_encoder(None)
    baseline_load = time.time() - t0
    logging.info("ロード %.1fs / GPU %.2fGB", baseline_load, torch.cuda.memory_allocated() / 1e9)
    base_out = encode_all(runner._pipe, DEVICE)

    # --- 2. 保存 ---
    if SAVE_DIR.exists():
        shutil.rmtree(SAVE_DIR)
    t0 = time.time()
    runner._pipe.text_encoder.save_pretrained(str(SAVE_DIR))
    save_s = time.time() - t0
    save_gb = sum(f.stat().st_size for f in SAVE_DIR.rglob("*") if f.is_file()) / 1e9
    logging.info("保存 %.1fs / %.2fGB", save_s, save_gb)

    # --- 3. 解放して保存物から再ロード ---
    runner._free_text_encoder(force=True)
    gc.collect()
    torch.cuda.empty_cache()
    logging.info("解放後 GPU %.2fGB", torch.cuda.memory_allocated() / 1e9)

    from transformers import AutoModelForImageTextToText

    logging.info("=== 保存物から再ロード ===")
    t0 = time.time()
    te2 = AutoModelForImageTextToText.from_pretrained(
        str(SAVE_DIR), dtype=torch.bfloat16, device_map="cuda"
    )
    reload_s = time.time() - t0
    logging.info("再ロード %.1fs / GPU %.2fGB", reload_s, torch.cuda.memory_allocated() / 1e9)

    # 再ロードした TE をパイプラインへ差し込んで同じ経路で比較する
    runner._pipe.text_encoder = te2
    runner._text_encoder_loaded = True
    new_out = encode_all(runner._pipe, DEVICE)

    # --- 4. 比較 ---
    results = {
        "te_prune": os.environ.get("H3_TE_PRUNE", "0"),
        "baseline_load_s": round(baseline_load, 1),
        "save_s": round(save_s, 1),
        "save_gb": round(save_gb, 2),
        "reload_s": round(reload_s, 1),
        "speedup_x": round(baseline_load / reload_s, 2),
        "cases": [],
    }
    all_equal = True
    for i, ((a, ta), (b, tb)) in enumerate(zip(base_out, new_out)):
        eq = bool(torch.equal(a, b))
        tags_eq = bool(torch.equal(ta, tb))
        diff = float((a.float() - b.float()).abs().max())
        rel = float((a.float() - b.float()).pow(2).mean().sqrt() / (a.float().pow(2).mean().sqrt() + 1e-9))
        results["cases"].append({
            "prompt": PROMPTS[i][:40], "shape": list(a.shape),
            "bitwise_equal": eq, "tags_equal": tags_eq,
            "max_abs_diff": diff, "rel_rms_diff": rel,
        })
        all_equal &= eq and tags_eq
        logging.info("case%d: bitwise=%s tags=%s max_abs_diff=%.3e rel_rms=%.3e",
                     i, eq, tags_eq, diff, rel)

    results["all_bitwise_equal"] = all_equal
    results["verdict"] = (
        "等価(ビット一致)。事前量子化保存は採用可能"
        if all_equal else
        "ビット不一致。差分の大きさを見て採用可否を判断すること"
    )
    out = OUT_DIR / "equivalence_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logging.info("=== %s ===", results["verdict"])
    logging.info("wrote %s", out)

    if not KEEP and SAVE_DIR.exists():
        shutil.rmtree(SAVE_DIR)
        logging.info("保存物を削除 (H3_PROBE_KEEP=1 で残せる)")


if __name__ == "__main__":
    main()

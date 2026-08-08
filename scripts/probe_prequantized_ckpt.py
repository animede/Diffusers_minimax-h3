"""案1の検証: 量子化済みチェックポイントの事前保存でロード固定費を削れるか。

問い: `H3_LOWVRAM=1` の毎リクエスト固定費(実測 85-105秒)のうち、text_encoder の
NF4 ロードが 36-53秒、transformer の int8 ロードが約33秒を占める。この時間の大半は
「元の重みをディスクから読む + その場で量子化する」処理なので、**量子化後の重みを
一度保存しておけば、次回以降は量子化を省いて読むだけで済むはず**。

検証する2つ:
  A) text_encoder (bnb-4bit NF4): `save_pretrained()` で直列化できるか。できるなら
     再ロード時間はどれだけ短くなるか。bitsandbytes は 4bit の直列化保存に対応して
     いる可能性が高いが、このプロジェクトでは未検証(README の改善候補筆頭)
  B) transformer (torchao int8): 同上。torchao の直列化は未調査

各ステップは実機の時間とディスクサイズを測る。**保存できない場合はその例外を記録する**
(「できなかった」も同じくらい価値のある結果)。保存先は既定で /tmp 配下ではなく
`outputs/prequant/` (.gitignore 済み) に置き、ディスクを大量に消費するため検証後に
削除できるようにする。

本体コードは無変更(このプローブ内でロード/保存を再現するのみ)。

実行 (サーバ停止中に):
  CUDA_VISIBLE_DEVICES=0 venv/bin/python scripts/probe_prequantized_ckpt.py
  H3_PROBE_KEEP=1 ... で保存物を残す (既定は測定後に削除)
"""
import json
import logging
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

OUT_DIR = BASE_DIR / "outputs" / "prequant"
OUT_DIR.mkdir(parents=True, exist_ok=True)
KEEP = os.environ.get("H3_PROBE_KEEP") == "1"


def dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def free_all(runner):
    import gc

    import torch

    runner._free_text_encoder(force=True)
    runner._free_transformer()
    gc.collect()
    torch.cuda.empty_cache()


def probe_text_encoder(runner, results):
    """A) bnb-4bit NF4 の text_encoder を save_pretrained → 再ロードで計測。"""
    import torch

    rec = {"component": "text_encoder", "quant": "bnb-4bit-nf4",
           "te_prune": os.environ.get("H3_TE_PRUNE", "0")}
    save_dir = OUT_DIR / "text_encoder_nf4"

    # --- 基準: 現行経路のロード時間 ---
    logging.info("[TE] 基準ロード (現行経路: 元重み読み込み + その場でNF4量子化)")
    t0 = time.time()
    runner._load_text_encoder(None)
    rec["baseline_load_s"] = round(time.time() - t0, 1)
    te = runner._pipe.text_encoder
    rec["gpu_after_load_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
    logging.info("[TE] 基準ロード %.1fs / GPU %.2fGB", rec["baseline_load_s"], rec["gpu_after_load_gb"])

    # --- 保存できるか ---
    if save_dir.exists():
        shutil.rmtree(save_dir)
    t0 = time.time()
    try:
        te.save_pretrained(str(save_dir))
        rec["save_s"] = round(time.time() - t0, 1)
        rec["save_gb"] = round(dir_size_gb(save_dir), 2)
        rec["saveable"] = True
        logging.info("[TE] save_pretrained 成功: %.1fs / %.2fGB", rec["save_s"], rec["save_gb"])
    except Exception:
        rec["saveable"] = False
        rec["save_exception"] = traceback.format_exc()
        logging.error("[TE] save_pretrained 失敗:\n%s", rec["save_exception"])
        results.append(rec)
        return

    # --- 解放して、保存物から再ロード ---
    free_all(runner)
    logging.info("[TE] 保存物から再ロード")
    from transformers import AutoModelForImageTextToText

    for label in ("reload_cold_ish", "reload_warm"):
        t0 = time.time()
        try:
            m = AutoModelForImageTextToText.from_pretrained(
                str(save_dir), dtype=torch.bfloat16, device_map="cuda"
            )
            dt = round(time.time() - t0, 1)
            rec[f"{label}_s"] = dt
            rec[f"{label}_gpu_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
            logging.info("[TE] %s: %.1fs / GPU %.2fGB", label, dt, rec[f"{label}_gpu_gb"])
            del m
        except Exception:
            rec[f"{label}_exception"] = traceback.format_exc()
            logging.error("[TE] %s 失敗:\n%s", label, rec[f"{label}_exception"])
            break
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    if rec.get("reload_warm_s") and rec.get("baseline_load_s"):
        rec["speedup_x"] = round(rec["baseline_load_s"] / rec["reload_warm_s"], 2)
        rec["saved_s"] = round(rec["baseline_load_s"] - rec["reload_warm_s"], 1)
    results.append(rec)
    if not KEEP and save_dir.exists():
        shutil.rmtree(save_dir)


def probe_transformer(runner, results):
    """B) torchao int8 の transformer を save_pretrained → 再ロードで計測。"""
    import torch

    from core.runner import H3_TRANSFORMER_QUANT

    rec = {"component": "transformer", "quant": H3_TRANSFORMER_QUANT}
    if H3_TRANSFORMER_QUANT != "int8":
        rec["skipped"] = "H3_TRANSFORMER_QUANT != int8 (H3_LOWVRAM=1 で実行すること)"
        results.append(rec)
        return
    save_dir = OUT_DIR / "transformer_int8"

    logging.info("[TR] 基準ロード (現行経路: bf16読み込み + その場でint8量子化)")
    t0 = time.time()
    runner._ensure_transformer(None)
    rec["baseline_load_s"] = round(time.time() - t0, 1)
    tr = runner._pipe.transformer
    rec["gpu_after_load_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
    logging.info("[TR] 基準ロード %.1fs / GPU %.2fGB", rec["baseline_load_s"], rec["gpu_after_load_gb"])

    if save_dir.exists():
        shutil.rmtree(save_dir)
    t0 = time.time()
    try:
        # torchao の tensor subclass はデフォルトの safetensors 直列化に載らない
        # 可能性がある。その場合の例外を捕まえるのがこのプローブの目的。
        tr.save_pretrained(str(save_dir))
        rec["save_s"] = round(time.time() - t0, 1)
        rec["save_gb"] = round(dir_size_gb(save_dir), 2)
        rec["saveable"] = True
        logging.info("[TR] save_pretrained 成功: %.1fs / %.2fGB", rec["save_s"], rec["save_gb"])
    except Exception:
        rec["saveable"] = False
        rec["save_exception"] = traceback.format_exc()
        logging.error("[TR] save_pretrained 失敗:\n%s", rec["save_exception"].splitlines()[-1])
        results.append(rec)
        return

    free_all(runner)
    logging.info("[TR] 保存物から再ロード")
    from diffusers import MiniMaxH3Transformer3DModel

    t0 = time.time()
    try:
        m = MiniMaxH3Transformer3DModel.from_pretrained(str(save_dir), dtype=torch.bfloat16)
        m.to("cuda")
        rec["reload_s"] = round(time.time() - t0, 1)
        rec["reload_gpu_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
        logging.info("[TR] 再ロード %.1fs / GPU %.2fGB", rec["reload_s"], rec["reload_gpu_gb"])
        rec["speedup_x"] = round(rec["baseline_load_s"] / rec["reload_s"], 2)
        rec["saved_s"] = round(rec["baseline_load_s"] - rec["reload_s"], 1)
        del m
    except Exception:
        rec["reload_exception"] = traceback.format_exc()
        logging.error("[TR] 再ロード失敗:\n%s", rec["reload_exception"].splitlines()[-1])
    results.append(rec)
    if not KEEP and save_dir.exists():
        shutil.rmtree(save_dir)


def main():
    from core.runner import MiniMaxH3Runner

    runner = MiniMaxH3Runner(OUT_DIR)
    results = []
    try:
        probe_text_encoder(runner, results)
        free_all(runner)
        probe_transformer(runner, results)
    finally:
        out = OUT_DIR / "probe_results.json"
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
        logging.info("=== wrote %s ===", out)
        for r in results:
            if r.get("skipped"):
                logging.info("%s: SKIP (%s)", r["component"], r["skipped"])
            elif not r.get("saveable"):
                logging.info("%s: 保存不可", r["component"])
            else:
                logging.info("%s: 基準 %.1fs -> 再ロード %s s (%s倍, %s秒節約) / 保存 %.2fGB",
                             r["component"], r["baseline_load_s"],
                             r.get("reload_warm_s") or r.get("reload_s"),
                             r.get("speedup_x"), r.get("saved_s"), r.get("save_gb", 0))


if __name__ == "__main__":
    main()

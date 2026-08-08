"""案2の検証: text_encoder を2枚目のGPUに常駐させられるか。

背景: `H3_LOWVRAM=1` が毎リクエストで TE と transformer を再ロードする根本原因は、
**デノイズ中に TE の置き場所が無いこと**にある。48GB 一枚では
transformer-int8 34GB + 活性化 ~5GB = 39GB で、残り 9GB に TE(削除版 17.45GB)は入らない。

TE を2枚目のGPUへ逃がせれば GPU0 は transformer を常駐させたままにでき、
**毎リクエストの固定費(実測 85-105秒)が丸ごと消える**見込み。

このプローブが答えるのは1点だけ:
  **現行の2枚目 (RTX 4000 SFF Ada 20GB, sm_89) に、削除版 TE-nf4 を載せて
  実際にエンコードを回したとき、ピークVRAMが 20GB に収まるか。**

収まるなら GPU 増設は不要(実装だけで済む)。収まらないなら 24GB 級への交換に意味がある。
t2va(参照なし)と ref2va(2048px短辺の参照画像を vision tower に通す = 活性化が大きい)
の両方を測る -- 後者が本当の上限になるため。

sm_89 でも問題にならない想定: SageAttention は transformer 側に適用するもので、
TE は transformers 標準の attention を使う。

本体コードは無変更(このプローブ内で TE を cuda:1 に配置して測るのみ)。

実行 (サーバ停止中に。CUDA_VISIBLE_DEVICES は設定しないこと = 両GPUを見せる):
  venv/bin/python scripts/probe_te_on_second_gpu.py
"""
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

OUT_DIR = BASE_DIR / "outputs" / "ab_te_second_gpu"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TE_DEVICE = os.environ.get("H3_PROBE_TE_DEVICE", "cuda:1")
REFERENCE_PNG = BASE_DIR / "outputs" / "t2i_1786110542_s1.png"

PROMPT = ("制服の少女と親友が坂道を歩きながら会話する。明るい住宅街、鳥の声。"
          "The girl (S1) says: <d>[Japanese] 今日は本当にいい天気だね。</d>")


def gpu_stats(idx: int) -> dict:
    import torch

    return {
        "allocated_gb": round(torch.cuda.memory_allocated(idx) / 1e9, 2),
        "reserved_gb": round(torch.cuda.memory_reserved(idx) / 1e9, 2),
        "peak_gb": round(torch.cuda.max_memory_allocated(idx) / 1e9, 2),
    }


def main():
    import torch

    if torch.cuda.device_count() < 2:
        raise SystemExit(
            f"GPUが{torch.cuda.device_count()}枚しか見えていません。"
            "CUDA_VISIBLE_DEVICES を設定せずに実行してください。"
        )
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        logging.info("cuda:%d %s %.1fGB sm_%d%d", i, p.name, p.total_memory / 1e9, p.major, p.minor)

    te_idx = int(TE_DEVICE.split(":")[1])
    total_gb = torch.cuda.get_device_properties(te_idx).total_memory / 1e9
    logging.info("TE を %s に配置して検証 (総容量 %.1fGB)", TE_DEVICE, total_gb)

    from core.runner import MiniMaxH3Reference, MiniMaxH3Runner

    runner = MiniMaxH3Runner(OUT_DIR)
    runner._ensure_pipe_shell()

    # --- TE を2枚目へロード ---
    # 本体の `_load_text_encoder()` は device_map={"text_encoder": "cuda"} 固定なので、
    # ここでは同じ引数構成のまま device_map だけを差し替えて再現する。
    from transformers import BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    config_kwargs = runner._text_encoder_config_kwargs()
    logging.info("text_encoder をロード中 (NF4, H3_TE_PRUNE=%s, device_map=%s)...",
                 os.environ.get("H3_TE_PRUNE", "0"), TE_DEVICE)
    torch.cuda.reset_peak_memory_stats(te_idx)
    t0 = time.time()
    runner._pipe.load_components(
        names=["text_encoder", "tokenizer", "processor"],
        dtype=torch.bfloat16,
        quantization_config={"text_encoder": quant_config},
        device_map={"text_encoder": TE_DEVICE},
        config=config_kwargs,
    )
    runner._text_encoder_loaded = True
    load_s = time.time() - t0
    after_load = gpu_stats(te_idx)
    logging.info("ロード完了 %.1fs / %s", load_s, after_load)

    results = {
        "te_device": TE_DEVICE,
        "te_device_name": torch.cuda.get_device_properties(te_idx).name,
        "te_device_total_gb": round(total_gb, 1),
        "te_prune": os.environ.get("H3_TE_PRUNE", "0"),
        "load_s": round(load_s, 1),
        "after_load": after_load,
        "cases": [],
    }

    dev = torch.device(TE_DEVICE)
    from diffusers.modular_pipelines.minimax_h3.before_encoder import (
        MiniMaxH3Ref2VASetupStep,
        MiniMaxH3SetupStep,
    )
    from diffusers.modular_pipelines.minimax_h3.encoders import (
        MiniMaxH3Ref2VATextEncoderStep,
        MiniMaxH3TextEncoderStep,
    )
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    def new_state(**kw):
        st = PipelineState()
        base = {"prompt": PROMPT, "image": None, "last_image": None, "height": 768, "width": 768,
                "num_frames": 124, "generator": None, "num_inference_steps": 30,
                "output_type": "pt", "attention_kwargs": None, "latents": None,
                "audio_latents": None, "condition_latents": None, "audio_condition_latents": None}
        base.update(kw)
        for k, v in base.items():
            st.set(k, v)
        return st

    # --- ケース1: t2va (参照なし) ---
    for label, res in (("t2va_768", (768, 768)), ("t2va_1344", (768, 1344))):
        rec = {"case": label, "height": res[0], "width": res[1]}
        try:
            torch.cuda.reset_peak_memory_stats(te_idx)
            state = new_state(height=res[0], width=res[1])
            _, state = MiniMaxH3SetupStep()(runner._pipe, state)
            t0 = time.time()
            with torch.no_grad():
                emb, tags = MiniMaxH3TextEncoderStep.encode_prompt(
                    runner._pipe, PROMPT, state.get("keyframes") or None,
                    device=dev, dtype=torch.bfloat16,
                )
            rec["encode_s"] = round(time.time() - t0, 2)
            rec["embed_shape"] = list(emb.shape)
            rec["stats"] = gpu_stats(te_idx)
            rec["headroom_gb"] = round(total_gb - rec["stats"]["peak_gb"], 2)
            rec["ok"] = True
            del emb, tags
        except Exception:
            rec["ok"] = False
            rec["exception"] = traceback.format_exc().splitlines()[-1]
            logging.error("[%s] 失敗: %s", label, rec["exception"])
        results["cases"].append(rec)
        logging.info("[%s] %s", label, json.dumps(rec, ensure_ascii=False))

    # --- ケース2: ref2va (2048px短辺の参照を vision tower に通す = 本当の上限) ---
    if REFERENCE_PNG.exists():
        runner._ensure_pipe_ref_shell()
        runner._sync_shared_components_to_ref()
        for n_ref in (1, 2):
            rec = {"case": f"ref2va_{n_ref}img"}
            try:
                torch.cuda.reset_peak_memory_stats(te_idx)
                refs = [MiniMaxH3Reference(image=str(REFERENCE_PNG)) for _ in range(n_ref)]
                st = PipelineState()
                for k, v in {"prompt": PROMPT, "references": refs, "height": 768, "width": 768,
                             "num_frames": 124, "generator": None, "num_inference_steps": 30,
                             "output_type": "pt", "attention_kwargs": None,
                             "latents": None, "audio_latents": None}.items():
                    st.set(k, v)
                _, st = MiniMaxH3Ref2VASetupStep()(runner._pipe_ref, st)
                t0 = time.time()
                with torch.no_grad():
                    emb, tags = MiniMaxH3Ref2VATextEncoderStep.encode_prompt(
                        runner._pipe_ref, PROMPT, st.get("prepared_references"),
                        device=dev, dtype=torch.bfloat16,
                    )
                rec["encode_s"] = round(time.time() - t0, 2)
                rec["embed_shape"] = list(emb.shape)
                rec["stats"] = gpu_stats(te_idx)
                rec["headroom_gb"] = round(total_gb - rec["stats"]["peak_gb"], 2)
                rec["ok"] = True
                del emb, tags
            except Exception:
                rec["ok"] = False
                rec["exception"] = traceback.format_exc().splitlines()[-1]
                logging.error("[%s] 失敗: %s", rec["case"], rec["exception"])
            results["cases"].append(rec)
            logging.info("[%s] %s", rec["case"], json.dumps(rec, ensure_ascii=False))
    else:
        logging.warning("参照画像が無いため ref2va ケースを省略: %s", REFERENCE_PNG)

    ok_cases = [c for c in results["cases"] if c.get("ok")]
    if ok_cases:
        worst = min(c["headroom_gb"] for c in ok_cases)
        results["min_headroom_gb"] = worst
        results["verdict"] = (
            f"全{len(ok_cases)}ケース成立、最小余裕 {worst}GB" if len(ok_cases) == len(results["cases"])
            else f"一部失敗。成立分の最小余裕 {worst}GB"
        )
    else:
        results["verdict"] = "全ケース失敗"

    out = OUT_DIR / "probe_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logging.info("=== %s ===", results["verdict"])
    logging.info("wrote %s", out)


if __name__ == "__main__":
    main()

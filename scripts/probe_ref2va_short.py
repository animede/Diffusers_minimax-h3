"""Ref2VA x 超短尺 (22フレーム) スパイク: 参照付き静止画が成立するかの品質検証。

問い: t2va では検証済みの「22フレーム超短尺 → 静止画取り出し」(README「静止画モード」)
が、参照パッキング (packing_ref2va: 参照行が生成行より圧倒的に多くなる) と重なっても
品質が崩れないか。成立すれば「キャラクター一貫の場面静止画」が量産でき、物語の
複数動画生成 (各場面の FL2VA/Ref2VA 素材作り) が大幅に速くなる。

条件 (いずれも同一参照・同一プロンプト・同一 seed・30steps・768x768):
  1. short22 : seconds=22/24 (0.917s、17n+5 の 2番目の最小値)
  2. baseline: seconds=5.0 (124フレーム、ref2va の通常尺) -- 品質アンカー

参照は本リポジトリの t2i バッチ生成物 (赤ずきんの少女、t2i_1786110542_s1.png) を使う。
「H3 生成静止画を参照に回す」という実運用ワークフローそのもの。

本体コードは無変更。5秒未満の尺は diffusers/アプリ両方のバリデーションが弾くため、
probe_short_frames_one.py と同じ手順でプローブ内 monkeypatch のみで回避する
(before_encoder/packing の MINIMAX_H3_MIN_DURATION と runner の MIN_SECONDS)。

実行 (サーバ停止中に、リポジトリ直下で):
  CUDA_VISIBLE_DEVICES=0 H3_LOWVRAM=1 venv/bin/python scripts/probe_ref2va_short.py
"""
import json
import logging
import sys
import time
import traceback
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

OUT_DIR = BASE_DIR / "outputs" / "ab_ref2va_short"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_PNG = BASE_DIR / "outputs" / "t2i_1786110542_s1.png"
PROMPT = (
    "The young girl in the red cloak from <Picture 1> sits on a mossy rock in a sunlit "
    "forest clearing, holding her lantern in her lap, soft morning light, birds chirping"
)
SEED = 12345
STEPS = 30
HEIGHT = 768
WIDTH = 768

PLAN = [
    ("short22", 22 / 24),
    ("baseline_5s", 5.0),
]


def patch_duration_gates():
    """probe_short_frames_one.py と同一の3点 patch (プローブプロセス限定)。"""
    import diffusers.modular_pipelines.minimax_h3.before_encoder as before_encoder
    import diffusers.modular_pipelines.minimax_h3.packing as packing

    import core.runner as runner_mod

    for mod, name in ((before_encoder, "MINIMAX_H3_MIN_DURATION"),
                      (packing, "MINIMAX_H3_MIN_DURATION"),
                      (runner_mod, "MIN_SECONDS")):
        old = getattr(mod, name)
        setattr(mod, name, 0.01)
        logging.info("[patch] %s.%s: %r -> 0.01", mod.__name__, name, old)


def extract_middle_frame(mp4_path: Path, out_png: Path) -> int:
    import av

    container = av.open(str(mp4_path))
    frames = [f.to_image() for f in container.decode(container.streams.video[0])]
    container.close()
    frames[len(frames) // 2].save(out_png)
    return len(frames)


def main():
    if not REFERENCE_PNG.exists():
        raise SystemExit(f"reference image not found: {REFERENCE_PNG}")

    logging.info("=== patching duration gates ===")
    patch_duration_gates()

    from core.runner import MiniMaxH3Reference, MiniMaxH3Runner, ProgressState

    runner = MiniMaxH3Runner(OUT_DIR)
    results = []
    for label, seconds in PLAN:
        logging.info("=== %s (seconds=%.4f) ===", label, seconds)
        progress = ProgressState()
        progress.update(job_id=label, started_at=time.time())
        # MiniMaxH3Reference はパスから自前デコードする。生成ごとに作り直す
        # (前の生成が内部状態を持っていても引きずらないように)。
        reference = MiniMaxH3Reference(image=str(REFERENCE_PNG))
        record = {"label": label, "requested_seconds": seconds}
        t0 = time.time()
        try:
            result = runner.generate_ref2va(
                prompt=PROMPT,
                references=[reference],
                height=HEIGHT,
                width=WIDTH,
                seconds=seconds,
                num_inference_steps=STEPS,
                seed=SEED,
                progress=progress,
            )
            record.update({k: result[k] for k in (
                "num_frames", "duration_s", "denoise_time_s", "decode_time_s",
                "avg_step_time_s", "peak_vram_gb", "audio_rms", "audio_peak",
                "mp4_path", "total_elapsed_s",
            )})
            mp4 = Path(result["mp4_path"])
            saved = OUT_DIR / f"{label}.mp4"
            saved.write_bytes(mp4.read_bytes())
            record["mp4_saved"] = str(saved)
            png = OUT_DIR / f"{label}_middle.png"
            record["n_decoded_frames"] = extract_middle_frame(saved, png)
            record["middle_png"] = str(png)
        except Exception:
            record["exception"] = traceback.format_exc()
            logging.error("[%s] EXCEPTION:\n%s", label, record["exception"])
        record["wall_time_s"] = round(time.time() - t0, 2)
        results.append(record)

    out_json = OUT_DIR / "probe_results.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    logging.info("=== wrote %s ===", out_json)
    for rec in results:
        if rec.get("exception"):
            logging.info("%s: FAILED (see json)", rec["label"])
        else:
            logging.info(
                "%s: frames=%s denoise=%.1fs decode=%.1fs peak_vram=%.1fGB wall=%.1fs",
                rec["label"], rec.get("num_frames"), rec.get("denoise_time_s"),
                rec.get("decode_time_s"), rec.get("peak_vram_gb"), rec["wall_time_s"],
            )


if __name__ == "__main__":
    main()

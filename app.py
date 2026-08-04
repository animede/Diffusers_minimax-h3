"""
MiniMax-H3 スタンドアロン検証アプリ (FastAPI)。

diffusers-server (port 8601) とは完全に独立したワークスペース。将来の統合検証のための
先行アプリで、T2VA (テキスト -> 動画+ステレオ音声) と FL2VA (先頭/末尾フレーム指定) を
提供する。生成は同時1件まで (グローバルロック)。

起動: venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
"""
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from core.llm import LLMConnectionError, VALID_MODES, enhance_prompt, get_llm_url
from core.runner import MAX_SECONDS, MIN_SECONDS, MiniMaxH3Runner, ProgressState

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("minimax_h3.app")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="MiniMax-H3 検証アプリ")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

runner = MiniMaxH3Runner(OUTPUT_DIR)

# 同時1件までのシンプルな排他制御 (diffusers-server の generation_lock と同じ考え方)
_generation_lock = threading.Lock()
_current_progress: Optional[ProgressState] = None
_progress_guard = threading.Lock()

RESOLUTION_PRESETS = {
    "768x768": (768, 768),
    "768x1344": (768, 1344),
    "1344x768": (1344, 768),
}


@app.get("/")
def index():
    # ブラウザが古いindex.htmlをキャッシュしてUI更新が見えなくなる問題の対策
    # (diffusers-server の NoCacheStaticFiles と同趣旨。JS/CSSはこのファイルにインライン)
    return FileResponse(
        str(BASE_DIR / "static" / "index.html"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/status")
def api_status():
    with _progress_guard:
        busy = _generation_lock.locked()
        progress = _current_progress.snapshot() if _current_progress else None
    return {
        "busy": busy,
        "progress": progress,
        "runner": runner.status(),
        "min_seconds": MIN_SECONDS,
        "max_seconds": MAX_SECONDS,
        "resolutions": list(RESOLUTION_PRESETS.keys()),
    }


@app.get("/api/progress")
def api_progress():
    with _progress_guard:
        if _current_progress is None:
            return {"phase": "idle"}
        return _current_progress.snapshot()


def _run_generation(
    prompt: str,
    resolution: str,
    seconds: float,
    num_inference_steps: int,
    seed: Optional[int],
    image: Optional[Image.Image],
    last_image: Optional[Image.Image],
    upscale: int = 0,
) -> dict:
    global _current_progress

    if resolution not in RESOLUTION_PRESETS:
        raise HTTPException(400, f"unknown resolution preset: {resolution}")
    height, width = RESOLUTION_PRESETS[resolution]

    if not (MIN_SECONDS <= seconds <= MAX_SECONDS):
        raise HTTPException(400, f"seconds must be between {MIN_SECONDS} and {MAX_SECONDS}, got {seconds}")

    if not prompt or not prompt.strip():
        raise HTTPException(400, "prompt is required")

    acquired = _generation_lock.acquire(blocking=False)
    if not acquired:
        raise HTTPException(409, "別の生成が進行中です。しばらく待ってから再試行してください。")

    job_id = uuid.uuid4().hex[:12]
    progress = ProgressState(job_id=job_id, phase="starting", started_at=time.time())
    with _progress_guard:
        _current_progress = progress

    try:
        result = runner.generate(
            prompt=prompt.strip(),
            height=height,
            width=width,
            seconds=seconds,
            num_inference_steps=num_inference_steps,
            seed=seed,
            image=image,
            last_image=last_image,
            progress=progress,
            upscale=upscale,
        )
        result["job_id"] = job_id
        result["video_url"] = f"/outputs/{Path(result['mp4_path']).name}"
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("generation failed")
        progress.update(phase="error", error=str(e))
        raise HTTPException(500, f"generation failed: {e}")
    finally:
        _generation_lock.release()


@app.post("/api/t2va")
def api_t2va(
    prompt: str = Form(...),
    resolution: str = Form("768x768"),
    seconds: float = Form(5.0),
    num_inference_steps: int = Form(30),
    seed: Optional[int] = Form(None),
    upscale: int = Form(0),
):
    result = _run_generation(
        prompt=prompt,
        resolution=resolution,
        seconds=seconds,
        num_inference_steps=num_inference_steps,
        seed=seed,
        image=None,
        last_image=None,
        upscale=upscale,
    )
    return JSONResponse(result)


@app.post("/api/fl2va")
def api_fl2va(
    prompt: str = Form(...),
    resolution: str = Form("768x768"),
    seconds: float = Form(5.0),
    num_inference_steps: int = Form(30),
    seed: Optional[int] = Form(None),
    image: Optional[UploadFile] = File(None),
    last_image: Optional[UploadFile] = File(None),
):
    if image is None and last_image is None:
        raise HTTPException(400, "FL2VA には image または last_image のいずれかが必要です")

    pil_image = Image.open(image.file).convert("RGB") if image is not None else None
    pil_last_image = Image.open(last_image.file).convert("RGB") if last_image is not None else None

    result = _run_generation(
        prompt=prompt,
        resolution=resolution,
        seconds=seconds,
        num_inference_steps=num_inference_steps,
        seed=seed,
        image=pil_image,
        last_image=pil_last_image,
    )
    return JSONResponse(result)


@app.post("/api/prompt/enhance")
def api_prompt_enhance(
    text: str = Form(...),
    mode: str = Form("storyboard"),
    seconds: float = Form(10.0),
):
    """ローカルLLM(H3_LLM_URL)でプロンプトをH3向けに整形する。

    GPU生成ロックとは無関係(生成中でも呼べる)。mode: brief(公式ブリーフ詳細化)/
    storyboard(マルチショットのCUTタイムコード形式、seconds に整合)/ translate(英訳のみ)。
    """
    if not text or not text.strip():
        raise HTTPException(400, "text is required")
    if mode not in VALID_MODES:
        raise HTTPException(400, f"mode は {VALID_MODES} のいずれかです: {mode!r}")
    t0 = time.time()
    try:
        result = enhance_prompt(text.strip(), mode, seconds=seconds)
    except LLMConnectionError as e:
        raise HTTPException(502, f"LLMサーバに接続できません(H3_LLM_URL={get_llm_url()}): {e}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"result": result, "mode": mode, "elapsed_s": time.time() - t0}


@app.on_event("startup")
def on_startup():
    logger.info("startup: preloading transformer/vae/audio_vae (text_encoder loaded/freed per request)")
    try:
        runner.preload_all()
        logger.info("preload done: %s", runner.status())
    except Exception:
        logger.exception("preload failed -- components will load lazily on first request instead")

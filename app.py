"""
MiniMax-H3 スタンドアロン検証アプリ (FastAPI)。

diffusers-server (port 8601) とは完全に独立したワークスペース。将来の統合検証のための
先行アプリで、T2VA (テキスト -> 動画+ステレオ音声) と FL2VA (先頭/末尾フレーム指定) を
提供する。生成は同時1件まで (グローバルロック)。

起動: venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
"""
import logging
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

import core.gallery as gallery
import core.settings as settings
from core.llm import LLMConnectionError, VALID_MODES, enhance_prompt, get_llm_url
from core.runner import (
    FPS,
    H3_TURBO_LORA,
    H3_TURBO_STEPS_DEFAULT,
    MAX_SECONDS,
    MIN_SECONDS,
    MiniMaxH3Reference,
    MiniMaxH3Runner,
    ProgressState,
    seconds_to_num_frames,
)

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

# 任意サイズ指定のための丸め規則。MiniMax-H3 のキャンバスは 32 の倍数でなければならず
# (`MINIMAX_H3_CANVAS_MULTIPLE`、`MiniMaxH3Ref2VASetupStep._check_inputs` 等が
# 満たさない値を ValueError にする)、ネイティブのキャンバスは短辺 768・最大 768x1344。
# ここでは「エラーにせず丸める」方針を取り、下限/上限でクランプしてから 32 の倍数へ
# 四捨五入する(上限はモデルカードの 2K 記載に合わせた実験用の余地。ネイティブ範囲を
# 超える指定は VRAM も品質も未検証なので UI 側で注意書きを出す)。
CANVAS_MULTIPLE = 32
CANVAS_MIN = 256
CANVAS_MAX = 2048
CANVAS_NATIVE_MAX = 1344  # ネイティブ範囲の目安(超過は実験的)


def round_canvas_value(value: int) -> int:
    """1辺を H3 の規則(32の倍数)へ丸め、[CANVAS_MIN, CANVAS_MAX] にクランプする。"""
    value = max(CANVAS_MIN, min(CANVAS_MAX, int(value)))
    return int(round(value / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE

# H3_TURBO_LORA=1: the turbo LoRA is community-verified at 8 steps (see core/runner.py's
# H3_TURBO_LORA module comment) -- default `num_inference_steps` to that instead of the
# base model's 30 so a client that does not pass the field explicitly gets a sane value
# for whichever mode the server was launched in. Still fully overridable per-request.
DEFAULT_NUM_INFERENCE_STEPS = H3_TURBO_STEPS_DEFAULT if H3_TURBO_LORA else 30


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
        # 任意サイズ/秒数のUI側プレビュー用。丸めの権威はサーバ(下の各エンドポイント)
        # だが、UIが同じ規則で「実際に使われる値」を先に見せられるように公開する。
        "constraints": {
            "canvas_multiple": CANVAS_MULTIPLE,
            "canvas_min": CANVAS_MIN,
            "canvas_max": CANVAS_MAX,
            "canvas_native_max": CANVAS_NATIVE_MAX,
            "fps": FPS,
            "frame_step": 17,   # num_frames は 17n + 5 (align_num_frames)
            "frame_offset": 5,
        },
    }


@app.get("/api/progress")
def api_progress():
    with _progress_guard:
        if _current_progress is None:
            return {"phase": "idle"}
        return _current_progress.snapshot()


@app.get("/api/settings")
def api_settings():
    """Current instant/reload-group settings + choice lists (see core/settings.py).
    The UI reads this once at load to seed its controls with the server's actual
    current state, not hardcoded defaults.
    """
    return settings.current_settings_snapshot()


@app.post("/api/settings/apply")
def api_settings_apply(
    transformer_quant: Optional[str] = Form(None),
    te_quant: Optional[str] = Form(None),
    te_prune: Optional[bool] = Form(None),
    lowvram: Optional[str] = Form(None),
    video_vae_fp16: Optional[bool] = Form(None),
):
    """Apply a new reload-group configuration: unloads every big model and reloads
    under the new settings (core.settings.apply_reload_settings -> runner.unload_all()
    + runner.preload_all(), no process restart -- see that function's docstring).

    409 while a generation is in progress (same generation_lock as /api/t2va etc, so an
    apply can never race an in-flight denoise loop or vice versa). 400 for an invalid/
    unverified combination (mirrors core/runner.py's own startup validation).
    """
    fields = {}
    if transformer_quant is not None:
        fields["transformer_quant"] = transformer_quant
    if te_quant is not None:
        fields["te_quant"] = te_quant
    if te_prune is not None:
        fields["te_prune"] = te_prune
    if lowvram is not None:
        fields["lowvram"] = lowvram
    if video_vae_fp16 is not None:
        fields["video_vae_fp16"] = video_vae_fp16

    acquired = _generation_lock.acquire(blocking=False)
    if not acquired:
        raise HTTPException(409, "生成が進行中のため設定を適用できません。完了を待ってから再試行してください。")
    global _current_progress
    job_id = uuid.uuid4().hex[:12]
    progress = ProgressState(job_id=job_id, phase="starting", started_at=time.time())
    with _progress_guard:
        _current_progress = progress
    try:
        progress.update(phase="loading", message="設定を適用中(モデル再ロード)...")
        result = settings.apply_reload_settings(runner, **fields)
        progress.update(phase="done", message="設定適用完了")
        return JSONResponse(result)
    except ValueError as e:
        progress.update(phase="error", error=str(e))
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("settings apply failed")
        progress.update(phase="error", error=str(e))
        raise HTTPException(500, f"settings apply failed: {e}")
    finally:
        _generation_lock.release()


def _run_generation(
    prompt: str,
    resolution: str,
    seconds: float,
    num_inference_steps: int,
    seed: Optional[int],
    image: Optional[Image.Image],
    last_image: Optional[Image.Image],
    upscale: int = 0,
    height: Optional[int] = None,
    width: Optional[int] = None,
    cache: Optional[str] = None,
    cache_threshold: Optional[float] = None,
    attn: Optional[str] = None,
    turbo: Optional[bool] = None,
) -> dict:
    global _current_progress

    # height/width を明示指定した場合はプリセットより優先し、H3の規則へ丸める
    # (エラーにはしない)。片方だけの指定は誤用なので 400。
    if (height is None) != (width is None):
        raise HTTPException(400, "height と width は両方指定してください(片方だけの指定は不可)")
    if height is not None:
        height = round_canvas_value(height)
        width = round_canvas_value(width)
    else:
        if resolution not in RESOLUTION_PRESETS:
            raise HTTPException(400, f"unknown resolution preset: {resolution}")
        height, width = RESOLUTION_PRESETS[resolution]

    # 秒数もモデルの許容範囲へ丸める(エラーにしない)。実際のフレーム数は
    # seconds_to_num_frames() が 17n+5 へ切り上げるため、生成尺は要求秒数より
    # わずかに長くなることがある(レスポンスの num_frames / duration_s が実値)。
    seconds = max(MIN_SECONDS, min(MAX_SECONDS, float(seconds)))

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
            cache=cache,
            cache_threshold=cache_threshold,
            attn=attn,
            turbo=turbo,
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
    num_inference_steps: int = Form(DEFAULT_NUM_INFERENCE_STEPS),
    seed: Optional[int] = Form(None),
    upscale: int = Form(0),
    height: Optional[int] = Form(None),
    width: Optional[int] = Form(None),
    cache: Optional[str] = Form(None),
    cache_threshold: Optional[float] = Form(None),
    attn: Optional[str] = Form(None),
    turbo: Optional[bool] = Form(None),
):
    """height/width を指定すると resolution プリセットより優先され、32の倍数へ丸められる。

    cache/cache_threshold/attn/turbo は即反映(再ロード不要)の任意パラメータ。
    未指定ならサーバの現在の既定(環境変数由来)のまま(core/settings.py 参照)。
    """
    result = _run_generation(
        prompt=prompt,
        resolution=resolution,
        seconds=seconds,
        num_inference_steps=num_inference_steps,
        seed=seed,
        image=None,
        last_image=None,
        upscale=upscale,
        height=height,
        width=width,
        cache=cache,
        cache_threshold=cache_threshold,
        attn=attn,
        turbo=turbo,
    )
    return JSONResponse(result)


@app.post("/api/fl2va")
def api_fl2va(
    prompt: str = Form(...),
    resolution: str = Form("768x768"),
    seconds: float = Form(5.0),
    num_inference_steps: int = Form(DEFAULT_NUM_INFERENCE_STEPS),
    seed: Optional[int] = Form(None),
    image: Optional[UploadFile] = File(None),
    last_image: Optional[UploadFile] = File(None),
    height: Optional[int] = Form(None),
    width: Optional[int] = Form(None),
    cache: Optional[str] = Form(None),
    cache_threshold: Optional[float] = Form(None),
    attn: Optional[str] = Form(None),
    turbo: Optional[bool] = Form(None),
):
    """height/width を指定すると resolution プリセットより優先され、32の倍数へ丸められる。

    cache/cache_threshold/attn/turbo は即反映(再ロード不要)の任意パラメータ。
    """
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
        height=height,
        width=width,
        cache=cache,
        cache_threshold=cache_threshold,
        attn=attn,
        turbo=turbo,
    )
    return JSONResponse(result)


# Content-type / extension -> reference kind ("image" | "video" | "audio"). Video/audio
# containers are decoded by PyAV inside MiniMaxH3Reference itself (it accepts a path and
# decodes it when built, per packing_ref2va.py's module docstring) -- this app only needs
# to classify the upload and hand it a path, never touching pixels/samples itself.
_REF_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_REF_VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}
_REF_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def _detect_reference_kind(upload: UploadFile) -> str:
    content_type = (upload.content_type or "").lower()
    if content_type.startswith("image/"):
        return "image"
    if content_type.startswith("video/"):
        return "video"
    if content_type.startswith("audio/"):
        return "audio"
    ext = Path(upload.filename or "").suffix.lower()
    if ext in _REF_IMAGE_EXTS:
        return "image"
    if ext in _REF_VIDEO_EXTS:
        return "video"
    if ext in _REF_AUDIO_EXTS:
        return "audio"
    raise HTTPException(
        400,
        f"references の種別を判定できません(filename={upload.filename!r}, content_type={upload.content_type!r})。"
        "画像/動画/音声いずれかの一般的な拡張子・content-typeにしてください。",
    )


@app.post("/api/ref2va")
def api_ref2va(
    prompt: str = Form(...),
    references: list[UploadFile] = File(...),
    height: Optional[int] = Form(None),
    width: Optional[int] = Form(None),
    seconds: Optional[float] = Form(None),
    num_inference_steps: int = Form(30),
    seed: Optional[int] = Form(None),
    cache: Optional[str] = Form(None),
    cache_threshold: Optional[float] = Form(None),
    attn: Optional[str] = Form(None),
    turbo: Optional[bool] = Form(None),
):
    """ref2va: 画像最大9・動画最大3・音声最大3(計12参照)からの動画+音声生成。

    references の送信順が参照順(プロンプト内ラベル・rotary配置に反映される)。種別は
    content-type/拡張子から自動判定する。`seconds` を省略できるのは references に音声を
    持つ参照(音声単体 or 音声付き動画)がちょうど1本のときのみ(その音声長が生成尺になる、
    MiniMaxH3Ref2VASetupStep.prepare_references の仕様どおり) -- それ以外で省略すると
    ValueError を 400 に変換して返す。

    cache/cache_threshold/attn/turbo は即反映(再ロード不要)の任意パラメータ
    (transformer_ref へ適用される)。
    """
    global _current_progress

    if not prompt or not prompt.strip():
        raise HTTPException(400, "prompt is required")
    if not references:
        raise HTTPException(400, "references is required (at least one image/video/audio file)")
    if (height is None) != (width is None):
        raise HTTPException(400, "height と width は両方指定するか、両方省略してください")
    # 32の倍数でない値はエラーにせず丸める(t2va/fl2va と同じ方針。省略時はサーバが
    # H3 自身の 16:9 キャンバスを解決するので触らない)。
    if height is not None:
        height = round_canvas_value(height)
        width = round_canvas_value(width)
    # 秒数も許容範囲へ丸める(省略時は音声参照1本からサーバが導出する)。
    if seconds is not None:
        seconds = max(MIN_SECONDS, min(MAX_SECONDS, float(seconds)))

    # Each upload is spooled to a real temp file: MiniMaxH3Reference(image=path) /
    # (video=path) / (audio=path) decodes a path itself (PyAV for video/audio, PIL for
    # image), and this app never needs the pixels/samples directly. Cleaned up in
    # `finally`, after generate_ref2va() has already decoded everything into in-memory
    # MiniMaxH3Reference objects (construction happens before the try block below, so
    # the temp files must outlive that construction).
    tmp_paths: list[Path] = []
    built_references = []
    try:
        for upload in references:
            kind = _detect_reference_kind(upload)
            suffix = Path(upload.filename or "").suffix or {"image": ".png", "video": ".mp4", "audio": ".wav"}[kind]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp") as tmp:
                tmp.write(upload.file.read())
                tmp_path = Path(tmp.name)
            tmp_paths.append(tmp_path)
            try:
                if kind == "image":
                    built_references.append(MiniMaxH3Reference(image=str(tmp_path)))
                elif kind == "video":
                    built_references.append(MiniMaxH3Reference(video=str(tmp_path)))
                else:
                    built_references.append(MiniMaxH3Reference(audio=str(tmp_path)))
            except Exception as e:
                raise HTTPException(400, f"references[{len(built_references)}] ({upload.filename}) の読み込みに失敗: {e}")

        acquired = _generation_lock.acquire(blocking=False)
        if not acquired:
            raise HTTPException(409, "別の生成が進行中です。しばらく待ってから再試行してください。")

        job_id = uuid.uuid4().hex[:12]
        progress = ProgressState(job_id=job_id, phase="starting", started_at=time.time())
        with _progress_guard:
            _current_progress = progress

        try:
            result = runner.generate_ref2va(
                prompt=prompt.strip(),
                references=built_references,
                height=height,
                width=width,
                seconds=seconds,
                num_inference_steps=num_inference_steps,
                seed=seed,
                progress=progress,
                cache=cache,
                cache_threshold=cache_threshold,
                attn=attn,
                turbo=turbo,
            )
            result["job_id"] = job_id
            result["video_url"] = f"/outputs/{Path(result['mp4_path']).name}"
            return JSONResponse(result)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            logger.exception("ref2va generation failed")
            progress.update(phase="error", error=str(e))
            raise HTTPException(500, f"ref2va generation failed: {e}")
        finally:
            _generation_lock.release()
    finally:
        for tmp_path in tmp_paths:
            tmp_path.unlink(missing_ok=True)


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


@app.get("/api/outputs")
def api_outputs_list():
    """outputs/ 直下(サブディレクトリ除く)の *.mp4 を新しい順で返す。

    解像度・尺は ffprobe で取得し、mtime+size をキーにメモリキャッシュする
    (core/gallery.py の _probe_cached)。
    """
    return {"items": gallery.list_outputs(OUTPUT_DIR)}


@app.post("/api/outputs/delete")
def api_outputs_delete(names: list[str] = Body(..., embed=True)):
    """names (outputs/ 直下のファイル名の配列) を削除する。

    パストラバーサル対策は core.gallery.safe_output_path で行う(`/`・`..`を含む名前を
    拒否し、解決後の親ディレクトリが OUTPUT_DIR 自身であることを確認する)。
    サブディレクトリ(outputs/ab_* 等)には安全対策の副作用としても一切触れられない。
    """
    try:
        result = gallery.delete_outputs(OUTPUT_DIR, names)
    except gallery.GalleryError as e:
        raise HTTPException(400, str(e))
    return result


@app.post("/api/outputs/concat")
def api_outputs_concat(names: list[str] = Body(..., embed=True)):
    """names の順に動画を連結して outputs/concat_<timestamp>.mp4 を作る。

    GPUを使わないため generation_lock は取らない。連結同士の同時実行だけを
    専用ロック(core.gallery.concat_lock)で防ぎ、取得できなければ409。
    """
    acquired = gallery.concat_lock.acquire(blocking=False)
    if not acquired:
        raise HTTPException(409, "別の連結処理が進行中です。しばらく待ってから再試行してください。")
    try:
        result = gallery.concat_outputs(OUTPUT_DIR, names)
        return JSONResponse(result)
    except gallery.GalleryError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("concat failed")
        raise HTTPException(500, f"concat failed: {e}")
    finally:
        gallery.concat_lock.release()


@app.on_event("startup")
def on_startup():
    logger.info("startup: preloading transformer/vae/audio_vae (text_encoder loaded/freed per request)")
    try:
        runner.preload_all()
        logger.info("preload done: %s", runner.status())
    except Exception:
        logger.exception("preload failed -- components will load lazily on first request instead")

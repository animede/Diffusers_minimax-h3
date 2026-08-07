"""生成済み動画・静止画ギャラリー (outputs/ 直下の *.mp4/*.png 一覧・削除、*.mp4 の連結)。

app.py を薄く保つため、一覧/削除/連結のロジックをここへ切り出す。GPUには一切触れない
(ffprobe/ffmpeg をサブプロセスで呼ぶだけ)ため、core.runner の generation_lock とは
独立に動く。連結だけは同時実行を避けるための専用ロックを持つ(下記 concat_lock)。

パストラバーサル対策: 受け取るファイル名は必ず `safe_output_path()` を通す。
`/` や `..` を含む名前・OUTPUT_DIR 直下でない解決結果(シンボリックリンクで外へ
逃げるケースを含む)を拒否する。サブディレクトリ(outputs/ab_* 等、A/B検証の資料)は
一覧にも削除にも一切含めない(直下の *.mp4 のみを対象にする)。
"""
from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import threading
import time
from pathlib import Path

logger = logging.getLogger("minimax_h3.gallery")

# mtime(ns) + size をキーにした ffprobe 結果のメモリキャッシュ。ファイルが置き換われば
# キーが変わるので自然に無効化される(TTLや明示invalidateは不要)。
_probe_cache: dict[str, dict] = {}
_probe_cache_lock = threading.Lock()

# 連結の同時実行を防ぐ専用ロック(GPUを使わないので generation_lock とは別)。
concat_lock = threading.Lock()


class GalleryError(ValueError):
    """400 相当のユーザー起因エラー(パストラバーサル・不正なファイル名等)。"""


def safe_output_path(output_dir: Path, name: str) -> Path:
    """name を outputs/ 直下のファイルパスとして安全に解決する。

    - `/` を含む名前 (サブディレクトリ指定・絶対パス) は拒否
    - `..` を含む名前は拒否
    - 解決後の親ディレクトリが OUTPUT_DIR そのものであることを確認
      (symlinkでOUTPUT_DIR外を指すファイル名だった場合も resolve() 後の parent 比較で弾く)
    """
    if not name or "/" in name or "\\" in name or ".." in name:
        raise GalleryError(f"不正なファイル名です: {name!r}")
    output_dir_resolved = output_dir.resolve()
    candidate = (output_dir / name).resolve()
    if candidate.parent != output_dir_resolved:
        raise GalleryError(f"不正なファイル名です(outputs/直下ではありません): {name!r}")
    return candidate


def _run_ffprobe(path: Path) -> dict:
    """ffprobe で duration/width/height/has_audio を取得する。失敗時は空 dict。"""
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if proc.returncode != 0:
            logger.warning("ffprobe failed for %s: %s", path, proc.stderr.strip()[:300])
            return {}
        data = json.loads(proc.stdout)
    except Exception:
        logger.exception("ffprobe error for %s", path)
        return {}

    fmt = data.get("format", {})
    streams = data.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = None
    try:
        duration = float(fmt.get("duration")) if fmt.get("duration") is not None else None
    except (TypeError, ValueError):
        duration = None
    if duration is None and video_stream is not None:
        try:
            duration = float(video_stream.get("duration"))
        except (TypeError, ValueError):
            duration = None

    fps = None
    if video_stream is not None:
        rate = video_stream.get("r_frame_rate") or video_stream.get("avg_frame_rate")
        if rate and "/" in rate:
            num, _, den = rate.partition("/")
            try:
                num_f, den_f = float(num), float(den)
                fps = num_f / den_f if den_f else None
            except ValueError:
                fps = None

    return {
        "duration": round(duration, 2) if duration is not None else None,
        "width": video_stream.get("width") if video_stream else None,
        "height": video_stream.get("height") if video_stream else None,
        "fps": round(fps, 3) if fps else None,
        "has_audio": audio_stream is not None,
        "audio_sample_rate": int(audio_stream["sample_rate"]) if audio_stream and audio_stream.get("sample_rate") else None,
        "audio_channels": audio_stream.get("channels") if audio_stream else None,
    }


def _probe_cached(path: Path) -> dict:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {}
    key = f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}"
    with _probe_cache_lock:
        cached = _probe_cache.get(key)
        if cached is not None:
            return cached
    probed = _run_ffprobe(path)
    with _probe_cache_lock:
        # 古いキー(同名・別mtimeの残骸)を掃除してから格納
        stale = [k for k in _probe_cache if k.startswith(f"{path.name}:") and k != key]
        for k in stale:
            _probe_cache.pop(k, None)
        _probe_cache[key] = probed
    return probed


def list_outputs(output_dir: Path) -> list[dict]:
    """outputs/ 直下(サブディレクトリ除く)の *.mp4 と *.png を新しい順で返す。

    PNG は静止画モード (/api/t2i) の生成物。type フィールド ("video"|"image") で
    区別する(UI はこれで <video>/<img> を出し分ける)。PNG も ffprobe で width/height
    が取れる(PNG は1フレームの video stream として扱われる)ため同じキャッシュ経路を使う。
    """
    entries = []
    for pattern, kind in (("*.mp4", "video"), ("*.png", "image")):
        for path in output_dir.glob(pattern):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            probed = _probe_cached(path)
            entries.append({
                "name": path.name,
                "url": f"/outputs/{path.name}",
                "type": kind,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "duration": probed.get("duration") if kind == "video" else None,
                "width": probed.get("width"),
                "height": probed.get("height"),
                "fps": probed.get("fps") if kind == "video" else None,
                "has_audio": probed.get("has_audio") if kind == "video" else False,
            })
    entries.sort(key=lambda e: e["mtime"], reverse=True)
    return entries


def delete_outputs(output_dir: Path, names: list[str]) -> dict:
    """names のファイルを outputs/ 直下から削除する。存在しないものは静かにスキップ。"""
    if not names:
        raise GalleryError("削除するファイル名を指定してください")
    deleted = []
    missing = []
    for name in names:
        path = safe_output_path(output_dir, name)
        if not path.exists() or not path.is_file():
            missing.append(name)
            continue
        # サブディレクトリ混入や symlink 経由の脱出は safe_output_path で既に弾いている。
        path.unlink()
        deleted.append(name)
    return {"deleted": deleted, "missing": missing}


def _probe_streams_for_concat(path: Path) -> dict:
    """連結可否判定用の詳細プロファイル(width/height/fps/audio有無/sr/ch/duration)。"""
    probed = _run_ffprobe(path)
    return {
        "width": probed.get("width"),
        "height": probed.get("height"),
        "fps": probed.get("fps"),
        "has_audio": bool(probed.get("has_audio")),
        "audio_sample_rate": probed.get("audio_sample_rate"),
        "audio_channels": probed.get("audio_channels"),
        "duration": probed.get("duration"),
    }


def _profiles_match(profiles: list[dict]) -> bool:
    first = profiles[0]
    for p in profiles[1:]:
        if p["width"] != first["width"] or p["height"] != first["height"]:
            return False
        if p["fps"] is None or first["fps"] is None or abs(p["fps"] - first["fps"]) > 1e-3:
            return False
        if p["has_audio"] != first["has_audio"]:
            return False
        if first["has_audio"]:
            if p["audio_sample_rate"] != first["audio_sample_rate"]:
                return False
            if p["audio_channels"] != first["audio_channels"]:
                return False
    return True


def concat_outputs(output_dir: Path, names: list[str]) -> dict:
    """names の順に動画を連結して outputs/concat_<timestamp>.mp4 を作る。

    全動画の (width, height, fps, 音声有無/sr/ch) が完全一致すれば concat demuxer +
    `-c copy` (高速・再エンコードなし)、そうでなければ filter_complex concat で
    先頭動画の解像度に合わせて再エンコードする(音声が無い動画には無音を合成)。
    """
    if len(names) < 2:
        raise GalleryError("連結には2本以上のファイルが必要です")

    paths = []
    for name in names:
        # 静止画 (PNG) は連結対象外。ffprobe は PNG も video stream として読めてしまう
        # ため、拡張子で明示的に弾く(黙って1フレーム動画として混ぜない)。
        if not name.lower().endswith(".mp4"):
            raise GalleryError(f"連結できるのは動画 (.mp4) のみです: {name}")
        path = safe_output_path(output_dir, name)
        if not path.exists() or not path.is_file():
            raise GalleryError(f"ファイルが見つかりません: {name}")
        paths.append(path)

    profiles = [_probe_streams_for_concat(p) for p in paths]
    for name, prof in zip(names, profiles):
        if prof["width"] is None or prof["height"] is None:
            raise GalleryError(f"動画ストリームを読み取れません: {name}")

    use_copy = _profiles_match(profiles)
    ts = int(time.time())
    out_name = f"concat_{ts}.mp4"
    out_path = output_dir / out_name

    if use_copy:
        method = "concat_demuxer_copy"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir=str(output_dir)
        ) as list_file:
            for p in paths:
                # concat demuxer のリストファイルはシングルクォートで囲みエスケープする
                escaped = str(p.resolve()).replace("'", "'\\''")
                list_file.write(f"file '{escaped}'\n")
            list_path = Path(list_file.name)
        try:
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(list_path),
                "-c", "copy",
                str(out_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode != 0:
                # copy が失敗したら再エンコードにフォールバック(コンテナ非互換等の保険)
                logger.warning("concat copy failed, falling back to re-encode: %s", proc.stderr[-2000:])
                use_copy = False
        finally:
            list_path.unlink(missing_ok=True)

    if not use_copy:
        method = "filter_complex_reencode"
        target_w, target_h = profiles[0]["width"], profiles[0]["height"]
        target_fps = profiles[0]["fps"] or 24
        any_audio = any(p["has_audio"] for p in profiles)
        target_sr = next((p["audio_sample_rate"] for p in profiles if p["has_audio"]), 32000) or 32000
        target_ch = next((p["audio_channels"] for p in profiles if p["has_audio"]), 2) or 2

        cmd = ["ffmpeg", "-y"]
        for p in paths:
            cmd += ["-i", str(p)]

        filter_parts = []
        concat_inputs = []
        for i, prof in enumerate(profiles):
            filter_parts.append(
                f"[{i}:v:0]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={target_fps}[v{i}]"
            )
            concat_inputs.append(f"[v{i}]")
            if any_audio:
                if prof["has_audio"]:
                    filter_parts.append(
                        f"[{i}:a:0]aresample={target_sr},aformat=channel_layouts="
                        f"{'stereo' if target_ch == 2 else 'mono'}[a{i}]"
                    )
                else:
                    # 音声トラックが無い動画には、その動画の長さぶんの無音を合成する
                    # (duration はこの動画自身の ffprobe 結果、無ければ ffmpeg 側で
                    # 動画の長さに自動で切り詰められるよう長めの無音を生成してから
                    # 対応する動画長でトリムする)
                    dur = profiles[i].get("duration") or _run_ffprobe(paths[i]).get("duration") or 5.0
                    filter_parts.append(
                        f"anullsrc=channel_layout={'stereo' if target_ch == 2 else 'mono'}:"
                        f"sample_rate={target_sr}:duration={dur}[a{i}]"
                    )
                concat_inputs.append(f"[a{i}]")

        n = len(paths)
        concat_filter = "".join(concat_inputs) + f"concat=n={n}:v=1:a={1 if any_audio else 0}[outv]" + ("[outa]" if any_audio else "")
        filter_complex = ";".join(filter_parts) + ";" + concat_filter

        cmd += ["-filter_complex", filter_complex, "-map", "[outv]"]
        if any_audio:
            cmd += ["-map", "[outa]"]
        cmd += [
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p",
        ]
        if any_audio:
            cmd += ["-c:a", "aac", "-b:a", "192k"]
        cmd += [str(out_path)]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            out_path.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg concat (re-encode) failed: {proc.stderr[-3000:]}")

    if not out_path.exists():
        raise RuntimeError("連結処理は成功と報告されましたが出力ファイルが見つかりません")

    result_probe = _run_ffprobe(out_path)
    return {
        "name": out_name,
        "url": f"/outputs/{out_name}",
        "method": method,
        "inputs": names,
        "duration": result_probe.get("duration"),
        "width": result_probe.get("width"),
        "height": result_probe.get("height"),
        "has_audio": result_probe.get("has_audio"),
    }

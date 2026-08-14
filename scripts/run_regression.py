#!/usr/bin/env python3
"""dev96 (RTX PRO 6000 96GB, CUDA_VISIBLE_DEVICES=0) 向け回帰基準ランナー。

docs/internal/regression_baselines_dev96.json のグループ (server_env が異なる単位) ごとに
このスクリプト自身がサーバのライフサイクルを管理する:
既存サーバを pkill -> setsid でグループの env で起動 -> startup 完了待ち ->
グループ内の各ケースを HTTP で実行 -> MD5 (と任意 checks) を比較 (--record なら採取して
JSON を更新) -> 全グループ終了後にサーバを停止し、最後にグループ A の env で起動し直す。

ケース定義・パラメータは JSON 側に持たせてあり、本スクリプトはそれを解釈するだけの
汎用実装(ケース追加は JSON 編集のみで足りる)。

使い方:
  venv/bin/python scripts/run_regression.py
  venv/bin/python scripts/run_regression.py --group A,B
  venv/bin/python scripts/run_regression.py --record
  venv/bin/python scripts/run_regression.py --file docs/internal/regression_baselines_dev96.json --group C

終了コード: 全PASS(またはRECORDED)なら0、1件でもFAILがあれば1。
"""
import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_JSON = os.path.join(REPO_ROOT, "docs", "internal", "regression_baselines_dev96.json")
PORT = 8611
BASE_URL = f"http://localhost:{PORT}"
LOG_DIR = os.path.join(REPO_ROOT, "logs")
SERVER_LOG = os.path.join(LOG_DIR, "run_regression_server.log")
STARTUP_TIMEOUT_S = 600
STARTUP_POLL_S = 3
GROUP_A_ENV_FALLBACK = (
    "CUDA_VISIBLE_DEVICES=0 H3_TRANSFORMER_QUANT=int8 H3_KEEP_TRANSFORMER=1 "
    "H3_VIDEO_VAE_FP16=1 H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_TURBO_LORA=1"
)

# ---- ファイル引数として扱うフィールド名 (multipart で File として送る) ----
_FILE_FIELDS_SINGLE = {"image", "last_image"}
_FILE_FIELDS_LIST = {"references"}


def log(msg: str) -> None:
    print(msg, flush=True)


def md5_of(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


# ---------------------------------------------------------------------------
# サーバのライフサイクル管理
# ---------------------------------------------------------------------------

def stop_server() -> None:
    subprocess.run(
        ["pkill", "-f", r"[u]vicorn app.*" + str(PORT)],
        cwd=REPO_ROOT,
    )
    time.sleep(3)


def parse_env_string(env_str: str) -> dict:
    """`KEY=VAL KEY2=VAL2 ...` 形式 (シェルのスペース区切り、値にスペース無し前提) を dict へ。"""
    env = {}
    for tok in env_str.strip().split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        env[k] = v
    return env


def start_server(env_str: str) -> tuple:
    """サーバを起動し (proc, log_offset) を返す。log_offset はこの起動のヘッダ書き込み
    **後**の SERVER_LOG のバイト位置 -- wait_for_startup はこの位置以降だけを見る。
    ログは追記式なので、ファイル全体を検索すると前のグループの
    "Application startup complete" にマッチして**まだ起動していないサーバを起動済みと
    誤認する**(2026-08-14 の初回採取でグループB/Cが実際にこれで connection refused に
    なった)。"""
    os.makedirs(LOG_DIR, exist_ok=True)
    env = os.environ.copy()
    env.update(parse_env_string(env_str))
    log_f = open(SERVER_LOG, "ab")
    log_f.write(f"\n=== starting server with env: {env_str} ===\n".encode())
    log_f.flush()
    log_offset = log_f.tell()
    proc = subprocess.Popen(
        [os.path.join(REPO_ROOT, "venv", "bin", "python"), "-m", "uvicorn", "app:app",
         "--host", "0.0.0.0", "--port", str(PORT)],
        cwd=REPO_ROOT,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,  # setsid相当: このスクリプトのタイムアウト/終了に道連れにしない
    )
    return proc, log_offset


def wait_for_startup(log_offset: int = 0, timeout_s: int = STARTUP_TIMEOUT_S) -> bool:
    """log_offset (start_server が返す、この起動のログ開始位置) 以降にだけ
    "Application startup complete" を探す。HTTP フォールバックは意図的に持たない --
    pkill 直後のレースで別プロセスに当たる余地を残さず、ログの文字列のみを信じる。"""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with open(SERVER_LOG, "rb") as f:
                f.seek(log_offset)
                tail = f.read()
                if b"Application startup complete" in tail:
                    return True
        except FileNotFoundError:
            pass
        time.sleep(STARTUP_POLL_S)
    return False


# ---------------------------------------------------------------------------
# multipart リクエスト構築 (依存ライブラリを増やさないため urllib + 自前 multipart)
# ---------------------------------------------------------------------------

def _multipart_body(fields: list, boundary: str) -> bytes:
    buf = io.BytesIO()
    for item in fields:
        buf.write(f"--{boundary}\r\n".encode())
        if item["type"] == "field":
            buf.write(f'Content-Disposition: form-data; name="{item["name"]}"\r\n\r\n'.encode())
            buf.write(str(item["value"]).encode())
            buf.write(b"\r\n")
        else:  # file
            filename = os.path.basename(item["path"])
            buf.write(
                f'Content-Disposition: form-data; name="{item["name"]}"; filename="{filename}"\r\n'.encode()
            )
            buf.write(b"Content-Type: application/octet-stream\r\n\r\n")
            with open(item["path"], "rb") as f:
                buf.write(f.read())
            buf.write(b"\r\n")
    buf.write(f"--{boundary}--\r\n".encode())
    return buf.getvalue()


def post_multipart(endpoint: str, params: dict, timeout_s: int = 1800):
    """params の値を multipart/form-data で送る。image/last_image/references はファイル
    パスとして扱う (references はリスト)。それ以外はテキストフィールド(リストなら同名で
    複数回送る = FastAPI の list[str] Form と同じ)。"""
    fields = []
    for key, value in params.items():
        if key in _FILE_FIELDS_SINGLE:
            path = os.path.join(REPO_ROOT, value) if not os.path.isabs(value) else value
            fields.append({"type": "file", "name": key, "path": path})
        elif key in _FILE_FIELDS_LIST:
            for p in value:
                path = os.path.join(REPO_ROOT, p) if not os.path.isabs(p) else p
                fields.append({"type": "file", "name": key, "path": path})
        elif isinstance(value, list):
            for v in value:
                fields.append({"type": "field", "name": key, "value": v})
        elif isinstance(value, bool):
            fields.append({"type": "field", "name": key, "value": "1" if value else "0"})
        else:
            fields.append({"type": "field", "name": key, "value": value})

    boundary = "----regressionboundary" + hashlib.md5(str(time.time()).encode()).hexdigest()[:16]
    body = _multipart_body(fields, boundary)
    req = urllib.request.Request(
        f"{BASE_URL}{endpoint}",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body_bytes = e.read()
        try:
            payload = json.loads(body_bytes)
        except Exception:
            payload = {"detail": body_bytes.decode(errors="replace")}
        return e.code, payload


def fetch_bytes(url_path: str) -> bytes:
    with urllib.request.urlopen(f"{BASE_URL}{url_path}", timeout=120) as resp:
        return resp.read()


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def count_audio_streams(mp4_bytes: bytes) -> int:
    import av
    with av.open(io.BytesIO(mp4_bytes)) as container:
        return len(container.streams.audio)


def get_num_frames(mp4_bytes: bytes) -> int:
    import av
    with av.open(io.BytesIO(mp4_bytes)) as container:
        stream = container.streams.video[0]
        n = 0
        for _ in container.decode(stream):
            n += 1
        return n


# ---------------------------------------------------------------------------
# 1ケース実行
# ---------------------------------------------------------------------------

def run_case(name: str, case: dict, record: bool, results_cache: dict) -> tuple:
    """戻り値: (status: 'PASS'|'FAIL'|'RECORDED', detail: str, updated_case: dict|None)"""
    endpoint = case["endpoint"]
    params = dict(case["params"])
    expect_status = case.get("expect_status", 200)

    t0 = time.time()
    status, payload = post_multipart(endpoint, params)
    runtime_s = round(time.time() - t0, 2)

    if expect_status != 200:
        if status == expect_status:
            contains = case.get("checks", {}).get("http_400_contains")
            detail_text = payload.get("detail", "") if isinstance(payload, dict) else str(payload)
            if contains and contains not in detail_text:
                return "FAIL", f"status={status} だが detail に '{contains}' が無い: {detail_text[:200]}", None
            return "PASS", f"status={status} (期待通り拒否)", None
        return "FAIL", f"expected status={expect_status}, got {status}: {payload}", None

    if status != 200:
        return "FAIL", f"HTTP {status}: {payload}", None

    updated_case = json.loads(json.dumps(case))  # deep copy
    updated_case["runtime_s"] = runtime_s

    # ---- 成果物の取得と MD5 ----
    actual_md5 = {}
    if "scenes" in payload:
        # t2i_batch / ref2i_batch: 複数場面
        png_md5s = []
        for scene in payload["scenes"]:
            png_bytes = fetch_bytes(scene["image_url"])
            png_md5s.append(md5_of(png_bytes))
        actual_md5["scenes_png"] = png_md5s
    else:
        if payload.get("image_url"):
            png_bytes = fetch_bytes(payload["image_url"])
            actual_md5["png"] = md5_of(png_bytes)
            results_cache[f"{name}::png_bytes"] = png_bytes
        if payload.get("video_url"):
            mp4_bytes = fetch_bytes(payload["video_url"])
            actual_md5["mp4"] = md5_of(mp4_bytes)
            results_cache[f"{name}::mp4_bytes"] = mp4_bytes

    # ---- checks ----
    checks = case.get("checks", {})
    check_msgs = []
    check_failed = False

    if "audio_streams" in checks:
        mp4_bytes = results_cache.get(f"{name}::mp4_bytes")
        if mp4_bytes is None:
            mp4_bytes = fetch_bytes(payload["video_url"])
        n = count_audio_streams(mp4_bytes)
        if n != checks["audio_streams"]:
            check_failed = True
            check_msgs.append(f"audio_streams={n} (期待 {checks['audio_streams']})")
        else:
            check_msgs.append(f"audio_streams={n} OK")

    if "num_frames" in checks:
        actual_nf = payload.get("num_frames")
        if actual_nf != checks["num_frames"]:
            check_failed = True
            check_msgs.append(f"num_frames={actual_nf} (期待 {checks['num_frames']})")
        else:
            check_msgs.append(f"num_frames={actual_nf} OK")

    if "same_png_as" in checks:
        other_name = checks["same_png_as"]
        other_png = results_cache.get(f"{other_name}::png_bytes")
        this_png = results_cache.get(f"{name}::png_bytes")
        if other_png is None or this_png is None:
            check_failed = True
            check_msgs.append(f"same_png_as={other_name}: 比較対象のPNGが未取得")
        elif md5_of(other_png) != md5_of(this_png):
            check_failed = True
            check_msgs.append(f"same_png_as={other_name}: PNG不一致")
        else:
            check_msgs.append(f"same_png_as={other_name} OK")

    if "known_anchor_png" in checks:
        expected_anchor = checks["known_anchor_png"]
        got = actual_md5.get("png")
        if got != expected_anchor:
            check_failed = True
            check_msgs.append(f"known_anchor_png 不一致: got={got} expected={expected_anchor}")
        else:
            check_msgs.append(f"known_anchor_png OK ({expected_anchor[:8]})")

    check_summary = "; ".join(check_msgs)

    # ---- MD5 比較 or 記録 ----
    if record:
        updated_case["md5"] = actual_md5 if actual_md5 else case.get("md5")
        detail = f"runtime={runtime_s}s md5={actual_md5} {check_summary}".strip()
        if check_failed:
            return "FAIL", f"[RECORDED but checks failed] {detail}", updated_case
        return "RECORDED", detail, updated_case

    expected_md5 = case.get("md5", {})
    if case.get("nondeterministic"):
        # md5 は比較せず、成果物の存在(200応答+成果物取得できたこと)と checks のみ判定
        if check_failed:
            return "FAIL", f"(nondeterministic, checks failed) {check_summary}", updated_case
        return "PASS", f"(nondeterministic, md5比較スキップ) runtime={runtime_s}s {check_summary}".strip(), updated_case

    mismatches = []
    for k, v in actual_md5.items():
        exp = expected_md5.get(k) if expected_md5 else None
        if exp is None:
            continue
        if v != exp:
            mismatches.append(f"{k}: got={v} expected={exp}")

    if mismatches or check_failed:
        detail = "; ".join(mismatches + ([check_summary] if check_summary else []))
        return "FAIL", f"runtime={runtime_s}s {detail}", updated_case

    return "PASS", f"runtime={runtime_s}s {check_summary}".strip(), updated_case


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=DEFAULT_JSON)
    ap.add_argument("--group", default=None, help="カンマ区切りのグループ名 (例: A,B). 省略時は全グループ")
    ap.add_argument("--record", action="store_true", help="MD5を採取してJSONを更新する")
    args = ap.parse_args()

    with open(args.file) as f:
        data = json.load(f)

    groups = data["groups"]
    target_groups = list(groups.keys()) if args.group is None else args.group.split(",")

    all_results = []
    results_cache = {}

    for gname in target_groups:
        if gname not in groups:
            log(f"[SKIP] unknown group: {gname}")
            continue
        group = groups[gname]
        env_str = group["server_env"]
        log(f"\n=== group {gname}: stopping any existing server ===")
        stop_server()
        log(f"=== group {gname}: starting server with env: {env_str} ===")
        _, log_offset = start_server(env_str)
        if not wait_for_startup(log_offset):
            log(f"[FATAL] group {gname}: server did not start within {STARTUP_TIMEOUT_S}s")
            all_results.append((gname, "SERVER_START", "FAIL", "startup timeout"))
            continue
        log(f"=== group {gname}: server up, running cases ===")

        for cname, case in group["cases"].items():
            try:
                status, detail, updated_case = run_case(cname, case, args.record, results_cache)
            except Exception as e:
                status, detail, updated_case = "FAIL", f"exception: {e}", None
            all_results.append((gname, cname, status, detail))
            log(f"[{status}] {gname}/{cname}: {detail}")
            if args.record and updated_case is not None:
                groups[gname]["cases"][cname] = updated_case
                with open(args.file, "w") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    f.write("\n")

    log("\n=== stopping server ===")
    stop_server()
    log(f"=== restarting server with group A env (default) ===")
    default_env = groups.get("A", {}).get("server_env", GROUP_A_ENV_FALLBACK)
    _, log_offset = start_server(default_env)
    if wait_for_startup(log_offset):
        log("=== default server (group A env) is up ===")
    else:
        log("[WARN] default server did not confirm startup within timeout")

    log("\n=== summary ===")
    n_pass = sum(1 for *_, s, _ in [(g, c, s, d) for g, c, s, d in all_results] if s in ("PASS", "RECORDED"))
    n_fail = sum(1 for g, c, s, d in all_results if s == "FAIL")
    for g, c, s, d in all_results:
        log(f"  [{s}] {g}/{c}")
    log(f"total={len(all_results)} pass_or_recorded={n_pass} fail={n_fail}")

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()

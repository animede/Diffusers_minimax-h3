"""内蔵 LLM(Gemma 4 E4B、GGUF Q4_0、CPU)によるフォールバック — ビルド済み llama-server を子プロセスで起動する。

外部の OpenAI 互換 LLM(H3_LLM_URL)に接続できないとき、core/llm.py の chat_completion() がここへ落ちる。
考え方は ACE-Step-1.5-fast の easy-music(外部失敗 → ローカル GGUF)と同じだが、Python バインディング
(llama-cpp-python、ソースビルドが必要)ではなく **llama.cpp の GitHub Releases にあるビルド済み `llama-server`**
を使う。コンパイラ不要で、Windows でも zip を展開するだけで済む。

- バイナリ: `H3_LLAMA_SERVER=<パス>` → `tools/llama/<tag>/llama-server[.exe]` → PATH の llama-server →
  無ければ Releases の固定タグ(`H3_LLAMA_CPP_TAG`、既定 b12475)から CPU 版(約 25MB)を `tools/llama/` へ自動取得。
- モデル: `H3_LLM_LOCAL_GGUF=<パス>` → `models/llm/gemma-4-E4B-it-Q4_0.gguf`(無ければ HF から約 4.4GB を自動取得)。
- 起動: `llama-server -m <gguf> --host 127.0.0.1 --port <H3_LLM_LOCAL_PORT、既定 64661> -c 16384 -t <threads>
  --reasoning off`(thinking は常に OFF)。/health が 200 を返すまで待ってから、core/llm.py が
  `http://127.0.0.1:<port>/v1/chat/completions` に普通の OpenAI 互換リクエストを送る。
- そのポートで既に何かが /health に応えていれば(利用者が自分で llama-server を立てている等)、それを使う。
- 子プロセスはプロセス終了時(atexit)に止める。親が強制終了されて atexit が走らない場合に備えて、Windows では
  Job Object(KILL_ON_JOB_CLOSE)、Linux では prctl(PR_SET_PDEATHSIG) で親の死に道連れにする(孤児化防止)。
  GPU は使わない(動画生成に使い切っているため)。

事前取得は `scripts/fetch_llama_server.py`(`scripts/install.sh --local-llm` が呼ぶ)。
"""
from __future__ import annotations

import atexit
import io
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger("minimax_h3.local_llm")

MODEL_REPO = "ggml-org/gemma-4-E4B-it-GGUF"
MODEL_FILE = "gemma-4-E4B-it-Q4_0.gguf"
MODEL_SIZE_HINT = "約 4.4GB"
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models" / "llm"
TOOLS_DIR = ROOT / "tools" / "llama"
LOGS_DIR = ROOT / "logs"
DEFAULT_TAG = "b12475"          # 2026-11-17 に Gemma 4 E4B Q4_0 で動作確認(Linux x64 / Windows x64 の CPU 版)
DEFAULT_PORT = 64661
N_CTX = 16384
HEALTH_TIMEOUT_S = 300          # 初回はモデルのページキャッシュが無く読み込みに時間がかかる

# 念のため: 思考チャネルが混ざったら除去(--reasoning off なので通常は出ない)
THOUGHT_RE = re.compile(r"<\|channel>thought\n.*?<channel\|>\s*|<think>.*?</think>\s*", flags=re.DOTALL)


def _tag() -> str:
    return os.environ.get("H3_LLAMA_CPP_TAG", "").strip() or DEFAULT_TAG


def _port() -> int:
    return int(os.environ.get("H3_LLM_LOCAL_PORT", "").strip() or DEFAULT_PORT)


def release_asset() -> tuple[str, str] | None:
    """(アーカイブ名, アーカイブ内の llama-server の相対パス)。未対応プラットフォームは None。"""
    tag = _tag()
    machine = platform.machine().lower()
    x64 = machine in ("x86_64", "amd64")
    arm = machine in ("aarch64", "arm64")
    if sys.platform.startswith("linux"):
        if x64:
            return f"llama-{tag}-bin-ubuntu-x64.tar.gz", f"llama-{tag}/llama-server"
        if arm:
            return f"llama-{tag}-bin-ubuntu-arm64.tar.gz", f"llama-{tag}/llama-server"
    elif sys.platform == "win32":
        if x64:
            return f"llama-{tag}-bin-win-cpu-x64.zip", "llama-server.exe"
        if arm:
            return f"llama-{tag}-bin-win-cpu-arm64.zip", "llama-server.exe"
    elif sys.platform == "darwin":
        if arm:
            return f"llama-{tag}-bin-macos-arm64.tar.gz", f"llama-{tag}/llama-server"
        if x64:
            return f"llama-{tag}-bin-macos-x64.tar.gz", f"llama-{tag}/llama-server"
    return None


def _exe_name() -> str:
    return "llama-server.exe" if sys.platform == "win32" else "llama-server"


def find_binary() -> Path | None:
    """起動せずに llama-server の場所だけ解決する(無ければ None)。"""
    override = os.environ.get("H3_LLAMA_SERVER", "").strip()
    if override:
        p = Path(override).expanduser()
        return p if p.exists() else None
    local = TOOLS_DIR / _tag() / _exe_name()
    if local.exists():
        return local
    which = shutil.which("llama-server")
    return Path(which) if which else None


def download_binary() -> Path:
    """Releases の固定タグから CPU 版を取得して tools/llama/<tag>/ に展開し、llama-server のパスを返す。"""
    asset = release_asset()
    if asset is None:
        raise RuntimeError(f"このプラットフォーム({sys.platform}/{platform.machine()})向けの llama.cpp ビルド済み"
                           "バイナリが無い。H3_LLAMA_SERVER=<パス> で指定するか、H3_LLM_LOCAL=0 で無効にしてください")
    name, inner = asset
    url = f"https://github.com/ggml-org/llama.cpp/releases/download/{_tag()}/{name}"
    dest_dir = TOOLS_DIR / _tag()
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("llama-server を取得中: %s", url)
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(dest_dir)
        exe = dest_dir / inner
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
            t.extractall(dest_dir, filter="data")
        # tar は llama-<tag>/ の下に入っているので 1 段持ち上げる(同じ場所に .so も並ぶ必要がある)
        sub = dest_dir / inner
        for item in sub.parent.iterdir():
            target = dest_dir / item.name
            if not target.exists():
                shutil.move(str(item), str(target))
        shutil.rmtree(sub.parent, ignore_errors=True)
        exe = dest_dir / _exe_name()
        exe.chmod(exe.stat().st_mode | 0o111)
    if not exe.exists():
        raise RuntimeError(f"展開後に {exe} が見つからない(アーカイブ構成が変わった?)")
    logger.info("llama-server 取得完了: %s(%.1f MB)", exe, len(data) / 1e6)
    return exe


def model_path() -> Path:
    override = os.environ.get("H3_LLM_LOCAL_GGUF", "").strip()
    return Path(override).expanduser() if override else MODELS_DIR / MODEL_FILE


def download_model() -> Path:
    path = model_path()
    if path.exists():
        return path
    if os.environ.get("H3_LLM_LOCAL_GGUF", "").strip():
        raise FileNotFoundError(f"H3_LLM_LOCAL_GGUF のファイルがありません: {path}")
    logger.info("内蔵 LLM のモデルをダウンロード中: %s/%s(%s)", MODEL_REPO, MODEL_FILE, MODEL_SIZE_HINT)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, local_dir=str(MODELS_DIR))
    logger.info("ダウンロード完了: %s", downloaded)
    return Path(downloaded)


def _linux_die_with_parent():
    """子プロセス側(fork 直後)で実行: 親が死んだら SIGTERM を受け取る。prctl より前に親が死んでいた場合の
    取りこぼしは、呼び出し元が記録した親 PID と比較して検出する。"""
    import ctypes
    import signal

    PR_SET_PDEATHSIG = 1
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except OSError:
        return
    if os.getppid() != _PARENT_PID:
        os._exit(0)


_PARENT_PID = os.getpid()


def _popen_kwargs() -> dict:
    """OS ごとの Popen 引数: Windows はコンソール窓なし、Linux は親の死に道連れ。"""
    if sys.platform == "win32":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    if sys.platform.startswith("linux"):
        return {"preexec_fn": _linux_die_with_parent}
    return {}


def _windows_job_object(proc: subprocess.Popen):
    """Windows: 子を Job Object に入れ、親(= この Python プロセス)が終了すると Job ごと kill されるようにする。
    成功すれば Job のハンドルを返す(呼び出し側が保持する。閉じると子が kill される)。失敗しても致命的ではない。"""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOperationCount", "WriteOperationCount",
                                                       "OtherOperationCount", "ReadTransferCount",
                                                       "WriteTransferCount", "OtherTransferCount")]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION), ("IoInfo", IO_COUNTERS),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        JobObjectExtendedLimitInformation = 9
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise OSError("CreateJobObjectW failed")
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(info),
                                                ctypes.sizeof(info)):
            raise OSError("SetInformationJobObject failed")
        if not kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(int(proc._handle))):
            raise OSError("AssignProcessToJobObject failed")
        return job
    except Exception as e:  # noqa: BLE001
        logger.warning("Job Object を設定できなかったため、親の強制終了時に llama-server が残る可能性があります: %s", e)
        return None


def _health_ok(port: int, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


class LocalLLM:
    """llama-server の子プロセスを管理する。シングルトン `local_llm` を使う。"""

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._job = None   # Windows の Job Object ハンドル(閉じると子が kill される)
        self._lock = threading.Lock()
        self._unavailable_reason = ""
        self._external_on_port = False   # 自分で起動したのではなく、既にそのポートで動いていたものを使っている
        atexit.register(self.stop)

    def is_available(self) -> bool:
        if release_asset() is None and find_binary() is None:
            self._unavailable_reason = (f"このプラットフォーム({sys.platform}/{platform.machine()})向けのビルド済み "
                                        "llama-server が無い(H3_LLAMA_SERVER で指定可)")
            return False
        return not self._unavailable_reason

    def base_url(self) -> str:
        return f"http://127.0.0.1:{_port()}"

    def running(self) -> bool:
        if self._external_on_port:
            return _health_ok(_port())
        return self._proc is not None and self._proc.poll() is None

    def ensure_running(self) -> str:
        """llama-server を(必要なら取得して)起動し、OpenAI 互換の base URL を返す。"""
        with self._lock:
            port = _port()
            if self.running() and _health_ok(port):
                return self.base_url()
            if _health_ok(port):
                logger.info("ポート %d で既に llama-server(または互換サーバ)が動いているのでそれを使います", port)
                self._external_on_port = True
                return self.base_url()
            self._external_on_port = False
            try:
                exe = find_binary() or download_binary()
                gguf = download_model()
            except Exception as e:  # noqa: BLE001
                self._unavailable_reason = f"準備に失敗: {e}"
                raise RuntimeError(self._unavailable_reason) from e
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            log = open(LOGS_DIR / "llama-server.log", "ab")
            cmd = [str(exe), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(port),
                   "-c", str(N_CTX), "-t", str(max(4, (os.cpu_count() or 8) // 2)), "--reasoning", "off"]
            logger.info("内蔵 LLM を起動: %s", " ".join(cmd))
            self._proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                          cwd=str(exe.parent), **_popen_kwargs())
            self._job = _windows_job_object(self._proc)   # Windows 以外は None
            t0 = time.time()
            while time.time() - t0 < HEALTH_TIMEOUT_S:
                if self._proc.poll() is not None:
                    raise RuntimeError(f"llama-server が終了コード {self._proc.returncode} で落ちました"
                                       f"(logs/llama-server.log 参照)")
                if _health_ok(port):
                    logger.info("内蔵 LLM 起動完了(%.1fs、%s、port %d)", time.time() - t0, gguf.name, port)
                    return self.base_url()
                time.sleep(0.5)
            self.stop()
            raise RuntimeError(f"llama-server が {HEALTH_TIMEOUT_S}s 以内に /health を返しませんでした")

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        job, self._job = self._job, None
        if job is not None and sys.platform == "win32":
            import ctypes

            ctypes.windll.kernel32.CloseHandle(job)

    def status(self) -> dict:
        exe = find_binary()
        path = model_path()
        asset = release_asset()
        return {
            "available": self.is_available(),
            "reason": self._unavailable_reason or None,
            "running": self.running(),
            "port": _port(),
            "model": MODEL_FILE,
            "model_downloaded": path.exists(),
            "model_path": str(path),
            "binary": str(exe) if exe else None,
            "binary_source": (f"llama.cpp {_tag()} {asset[0]}" if asset else None),
        }


local_llm = LocalLLM()

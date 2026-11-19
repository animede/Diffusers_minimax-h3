"""ブロックの重みを RAM に常駐させず、safetensors から直接 GPU へ流す(`H3_GROUP_STREAM_HOST=file`、2026-11-15)。

`core/blockstream.py`(pinned)は int8 transformer 全体(30GiB)をピン留めしたホスト RAM に置くので、
RAM 48GB 級が前提になる。こちらは minimax-h3-image-server の `app/filestream.py` と同じ方式で、単位(ブロック、
または adaln_proj / attn / ff の 1/3 ブロック)ごとに

  量子化済みキャッシュ(safetensors) --(8 スレッドで 32MB ずつ並列 read)--> ピン留めした受け皿(2 本で交互)
    --(copy stream で非同期 H2D)--> GPU

と流す。読み込みは次の単位を先に始め(ディスクと GPU 計算を重ねる)、使い終わった GPU 側は参照を切るだけ
(重みは不変なので書き戻しは無い)。モデル本体は `from_pretrained` の mmap のまま(RAM に展開しない)で、
流す単位のテンソルは 0 要素のプレースホルダに差し替えて手放す。

ページキャッシュの使い方: 読んだ範囲は OS のページキャッシュに残り、次のステップの read はそこから返る(速い)。
全体がキャッシュに収まらない RAM では、巡回アクセスを LRU に任せると毎ステップ全部が押し出されて全量をディスクから
読み直すことになるので、末尾の `drop` 単位だけは読み終えるたびに `posix_fadvise(DONTNEED)` で捨て、残りをキャッシュに
留める(image server で実測済みの方式)。drop の数は auto で「原本 + 余裕 − 使える RAM」から決める。
Windows には fadvise が無いので drop は効かない(`FILE_FLAG_SEQUENTIAL_SCAN` で開くだけ)。

出力は pinned / 全常駐と同じ(同じ int8 の重みを同じ場所に置くだけ。MD5 一致を確認)。
"""

from __future__ import annotations

import json
import logging
import math
import os
import struct
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import torch

from core.blockstream import _inner_names, _set, collect_slots

logger = logging.getLogger("minimax_h3.filestream")

_ALIGN = 4096            # 受け皿内のテンソル境界(dtype の view が取れるように整列)
_CHUNK = 32 << 20        # 並列読みの単位(1 スレッド逐次だと 1.6GB/s、8 スレッド × 32MB で NVMe の上限まで出る、実測)
_READ_THREADS = 8
# ファイルに無いテンソル(turbo LoRA の lora_a / lora_b など)のうち、このサイズ以上のものは GPU に常駐させず
# ホスト RAM(ページャブル)に置いて単位ごとに受け皿へ memcpy して流す(2026-11-18)。turbo の LoRA は計 1.24GiB
# あり、6GB のカードではこれが常駐すると Windows は共有メモリへ退避して極端に遅くなる(RTX 3050 6GB で実測)。
# 小さいもの(norm など)は従来どおり GPU 常駐。0 で無効(全部 GPU 常駐)。
_HOST_MIN_BYTES = int(float(os.environ.get("H3_FILE_STREAM_HOST_MIN_MB", "1").strip() or "1") * 2**20)
_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "U8": torch.uint8, "I8": torch.int8,
           "I32": torch.int32, "I64": torch.int64, "F8_E4M3": torch.float8_e4m3fn, "BOOL": torch.bool}


class FileRegion:
    """ファイル内の 1 テンソル分の領域と GPU 側の形。`host` が入っているときはファイルではなくホスト RAM 上の
    テンソル(ファイルに無いもの)を受け皿へ写す。"""
    __slots__ = ("path", "offset", "nbytes", "dtype", "shape", "host")

    def __init__(self, path: Path | None, offset: int, nbytes: int, dtype: torch.dtype, shape: tuple[int, ...],
                 host: torch.Tensor | None = None):
        self.path, self.offset, self.nbytes, self.dtype, self.shape, self.host = path, offset, nbytes, dtype, shape, host


def safetensors_index(paths: list[Path]) -> dict[str, tuple[Path, int, int, torch.dtype | None, tuple]]:
    """各ファイルのヘッダから {名前: (path, ファイル内オフセット, バイト数, dtype, shape)} を作る。"""
    index = {}
    for path in paths:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        data0 = 8 + n
        for k, v in header.items():
            if k != "__metadata__":
                index[k] = (path, data0 + v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0],
                            _DTYPES.get(v["dtype"]), tuple(v["shape"]))
    return index


def slot_names(unit: torch.nn.Module, prefix: str) -> list[str | None]:
    """`collect_slots(unit)` と同じ順序で、各スロットの safetensors 上の名前。torchao の wrapper の内側は
    diffusers の直列化規則(`<module>.weight` → `<module>._weight_<inner>`)。名前が付けられないものは None。"""
    names, seen = [], set()
    # turbo LoRA の wrapper(core/runner.py の _TurboLoRALinear)は元の Linear を `.base` に抱えるので、パラメータ名が
    # `<mod>.base.weight` になる。ファイル側の名前は巻く前の `<mod>.weight` なので `.base` を外して引く
    # (lora_a / lora_b はファイルに無いので GPU 常駐になる。2026-11-18)。
    wrappers = {n for n, m in unit.named_modules() if type(m).__name__ == "_TurboLoRALinear"}

    def canon(pname: str) -> str:
        head, _, leaf = pname.rpartition(".")
        if head.endswith(".base") and head[:-5] in wrappers:
            return f"{head[:-5]}.{leaf}"
        if head == "base" and "" in wrappers:
            return leaf
        return pname

    for pname, p in unit.named_parameters(recurse=True):
        if id(p) in seen:
            continue
        seen.add(id(p))
        full = prefix + canon(pname)
        inner = _inner_names(p)
        if inner:
            head, _, leaf = full.rpartition(".")
            names += [f"{head}._{leaf}_{attr}" if head else f"_{leaf}_{attr}" for attr in inner]
        else:
            names.append(full)
        qs = getattr(p, "quant_state", None)
        if qs is not None:
            # bitsandbytes の Params4bit: safetensors では `<weight>.absmax` / `<weight>.quant_map`。offset と
            # 二重量子化の state2 には名前が無い(小さいので GPU に常駐)
            for owner, nested in ((qs, False), (getattr(qs, "state2", None), True)):
                if owner is None:
                    continue
                for attr in ("absmax", "code", "offset"):
                    if isinstance(getattr(owner, attr, None), torch.Tensor):
                        names.append(None if nested else {"absmax": full + ".absmax", "code": full + ".quant_map"}.get(attr))
    for bname, _ in unit.named_buffers(recurse=True):
        names.append(prefix + bname)
    return names


def physical_free_gib(device) -> float | None:
    """GPU の物理的な空き VRAM(GiB)を nvidia-smi で見る(総量 − 全プロセスの使用量)。None なら取れなかった。

    Windows(WDDM)では `torch.cuda.mem_get_info` の空きに他プロセスの退避可能な分が含まれることがあり、それを信じて
    載せると物理 VRAM を越えて共有メモリへ退避し、極端に遅くなる(RTX 3050 6GB: ドライバの空き 2.2GB と出ていたのに
    タスクマネージャは専用 5.8 / 6.0GB、デコード 708 秒)。nvidia-smi の memory.used はタスクマネージャの専用 GPU メモリと
    同じ、全プロセス込みの実使用量。CUDA の番号と nvidia-smi の並びは違いうるので UUID で突き合わせる。"""
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if exe is None and sys.platform == "win32":
        for cand in (r"C:\Windows\System32\nvidia-smi.exe", r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"):
            if os.path.exists(cand):
                exe = cand
                break
    if exe is None:
        return None
    try:
        dev = torch.device(device)
        uuid = str(getattr(torch.cuda.get_device_properties(dev), "uuid", "")).lower()
        out = subprocess.run([exe, "--query-gpu=uuid,memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        rows = [[c.strip() for c in line.split(",")] for line in out.strip().splitlines() if line.strip()]
        for u, used, total in rows:
            if uuid and uuid in u.lower():
                return (float(total) - float(used)) / 1024
        if len(rows) == 1:
            return (float(rows[0][2]) - float(rows[0][1])) / 1024
    except Exception:  # noqa: BLE001
        return None
    return None


class ResidentLoader:
    """`FileBlockStreamer` と同じ名前解決でファイルの重みを読み、**全単位を GPU に載せたまま**にする(フック無し)。
    VAE デコーダ(4.5GB)のデコード中の置き場所として使う(2026-11-18): タイルごとに 36 ブロックを流す方式だと
    768²・5 秒で約 150 回 × 4.5GB をホストから送り直すため、RAM 16GB の PC ではこの転送(ページキャッシュ → 受け皿 →
    GPU)がデコード時間の大半を占めていた(RTX 4060 Laptop で 1 パス 0.77 秒のうち 0.68 秒が読み待ち)。VRAM に
    収まるなら 1 回読んで置いたままにするほうが速く、出力は同じ。`release()` でプレースホルダに戻して GPU を空ける。"""

    def __init__(self, units, device, *, names_root: torch.nn.Module, files: list[Path], label: str,
                 min_driver_free_gib: float = 0.0):
        """min_driver_free_gib > 0 なら、単位を載せる前にドライバが見る空き VRAM(mem_get_info)を確かめ、下回ったら
        そこで止める(載せた単位数は `n_units_loaded`)。Windows は expandable_segments が無く、torch が抱えたままの
        領域(reserved − allocated)が断片化で使い回せないことがあり、見積もりだけで載せると物理 VRAM を越えて
        共有メモリへ退避して極端に遅くなる(RTX 3050 6GB / RAM 32GB で実測: 512² のデコード 824 秒)。"""
        self.device = torch.device(device)
        self.label = label
        index = safetensors_index(files)
        mod_names = {id(m): nm for nm, m in names_root.named_modules()}
        self.slots: list[tuple] = []          # (owner, attr, placeholder)
        self.n_units_loaded = 0
        self.stopped_free_gib: float | None = None
        reader = _Reader(sequential=False)
        total = 0
        t0 = time.time()
        try:
            for unit in units:
                if min_driver_free_gib > 0:
                    torch.cuda.synchronize(self.device)
                    free = torch.cuda.mem_get_info(self.device)[0] / 2**30
                    phys = physical_free_gib(self.device)
                    if phys is not None:
                        free = min(free, phys)      # 物理の空き(全プロセス込み)のほうが小さければそちらを信じる
                    if free < min_driver_free_gib:
                        self.stopped_free_gib = free
                        break
                prefix = mod_names[id(unit)] + "."
                plan, pos = [], 0
                for (owner, attr, t), name in zip(collect_slots(unit), slot_names(unit, prefix)):
                    ent = index.get(name) if name else None
                    nb = t.numel() * t.element_size()
                    if ent is not None and t.numel() == 0:
                        region = FileRegion(ent[0], ent[1], ent[2], ent[3], ent[4])
                    elif ent is None or ent[2] != nb or ent[3] != t.dtype:
                        _set(owner, attr, t.to(self.device))
                        continue
                    else:
                        region = FileRegion(ent[0], ent[1], nb, t.dtype, tuple(t.shape))
                    plan.append(((owner, attr), region, pos))
                    pos += (region.nbytes + _ALIGN - 1) // _ALIGN * _ALIGN
                if not plan:
                    continue
                buf = torch.empty(max(pos, _ALIGN), dtype=torch.uint8)
                reader.read_unit(buf, plan, drop=False)
                for (owner, attr), r, p in plan:
                    g = buf[p:p + r.nbytes].view(r.dtype).view(r.shape).to(self.device)
                    self.slots.append((owner, attr, torch.empty(0, dtype=r.dtype)))
                    _set(owner, attr, g)
                    total += r.nbytes
                del buf
                self.n_units_loaded += 1
        finally:
            reader.close()
        self.total_gb = total / 2**30
        logger.info("resident (%s): %d units, %d tensors (%.2f GiB) read from %d files to the GPU in %.1fs%s",
                    label, self.n_units_loaded, len(self.slots), self.total_gb, len(files), time.time() - t0,
                    "" if self.stopped_free_gib is None else
                    f" (stopped: driver free VRAM {self.stopped_free_gib:.1f} GiB < {min_driver_free_gib:.1f} floor)")

    def release(self) -> float:
        for owner, attr, placeholder in self.slots:
            _set(owner, attr, placeholder)
        self.slots = []
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        return self.total_gb

    def describe(self) -> dict:
        return {"host": "gpu-resident", "weights_gib": round(self.total_gb, 2)}


class SplitDecoder:
    """先頭の K ブロックを GPU に置き切り(ResidentLoader)、残りをファイルから流す(FileBlockStreamer)の組。
    VRAM に全部は入らないカード(8GB に 768² など)向け: 置いた分だけ毎パスの転送が減り、流す側の重みも小さくなって
    ページキャッシュに残りやすくなる(2026-11-18)。"""

    def __init__(self, resident: "ResidentLoader", streamer: "FileBlockStreamer"):
        self.resident, self.streamer = resident, streamer

    def release(self) -> float:
        return self.resident.release() + self.streamer.release()

    def describe(self) -> dict:
        return {"host": "gpu-resident+file", "resident_gib": round(self.resident.total_gb, 2),
                "resident_tensors": len(self.resident.slots), "streamed": self.streamer.describe()}


def host_cache_budget_gb() -> float:
    """この後ページキャッシュに使える見込みのホスト RAM(GB)。MemAvailable と、cgroup の上限があれば
    「上限 − このプロセスの匿名メモリ」の小さい方(cgroup の使用量はキャッシュ込みなので使わない)。"""
    avail = _mem_available_gb()
    try:
        anon = _process_anon_gb()
        rel = open("/proc/self/cgroup").read().strip().split(":", 2)[-1]
        node = Path("/sys/fs/cgroup" + rel)
        while True:
            mx = node / "memory.max"
            if mx.exists():
                v = mx.read_text().strip()
                if v != "max":
                    avail = min(avail, int(v) / 2**30 - anon)
            if node == Path("/sys/fs/cgroup") or node == node.parent:
                break
            node = node.parent
    except (OSError, ValueError):
        pass
    return max(avail, 0.0)


def _mem_available_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 2**30
    except Exception:  # noqa: BLE001
        try:
            for line in open("/proc/meminfo"):
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2**20
        except OSError:
            pass
    return 0.0


def _process_anon_gb() -> float:
    try:
        for line in open("/proc/self/status"):
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return 0.0


class _Reader:
    """ファイルの指定範囲をピン留めバッファへ読む。32MB ずつ複数スレッドで並列に読む。

    sequential(Windows のみ意味がある): True なら FILE_FLAG_SEQUENTIAL_SCAN で開き、読んだページをキャッシュに
    残さない(30GB の transformer のように RAM に収まらないもの向け。LRU の巡回で他を押し出さない)。False なら
    通常のキャッシュ付きで開く(VAE デコーダ 4.5GB や投影 TE 1.9GB のように RAM に収まるもの向け。タイルごと・
    ステップごとの再読込がページキャッシュから返るので NVMe を毎回読まない。2026-11-18、RTX 4060 Laptop で
    デコード中に NVMe が張り付いていた件の対策)。"""

    def __init__(self, sequential: bool = True):
        self.sequential = sequential
        self.pool = ThreadPoolExecutor(_READ_THREADS, thread_name_prefix="filestream-read")
        self.local = threading.local()      # pread の無い OS(Windows)はスレッドごとに fd
        self.shared: dict[Path, int] = {}
        self.lock = threading.Lock()
        self.has_pread = hasattr(os, "preadv")

    def _open(self, path: Path) -> int:
        flags = os.O_RDONLY
        if sys.platform == "win32":
            flags |= os.O_BINARY | (getattr(os, "O_SEQUENTIAL", 0) if self.sequential else 0)
        return os.open(str(path), flags)

    def _fd(self, path: Path) -> int:
        if self.has_pread:
            with self.lock:
                fd = self.shared.get(path)
                if fd is None:
                    fd = self.shared[path] = self._open(path)
            return fd
        return self._file(path).fileno()

    def _file(self, path: Path):
        files = getattr(self.local, "files", None)
        if files is None:
            files = self.local.files = {}
        f = files.get(path)
        if f is None:
            f = files[path] = open(self._open(path), "rb", buffering=0)
        return f

    def _read_chunk(self, path: Path, mv: memoryview, offset: int) -> None:
        got, n = 0, len(mv)
        if self.has_pread:
            fd = self._fd(path)
        else:
            f = self._file(path)
        while got < n:
            if self.has_pread:
                k = os.preadv(fd, [mv[got:]], offset + got)
            else:
                f.seek(offset + got)
                k = f.readinto(mv[got:])
            if not k:
                raise OSError(f"short read: {path.name} offset={offset} got={got}/{n}")
            got += k

    def read_unit(self, buf: torch.Tensor, plan: list, drop: bool) -> None:
        """plan = [(slot, FileRegion, 受け皿内オフセット)] をまとめて並列に読む。drop なら読んだ範囲をキャッシュから捨てる。"""
        mv = memoryview(buf.numpy())
        futs = [self.pool.submit(self._read_chunk, r.path, mv[pos + c:pos + min(c + _CHUNK, r.nbytes)], r.offset + c)
                for _, r, pos in plan if r.host is None for c in range(0, r.nbytes, _CHUNK)]
        for _, r, pos in plan:
            if r.host is not None:      # ホスト RAM 上のテンソル: 受け皿へ memcpy(読みと並行)
                buf[pos:pos + r.nbytes].copy_(r.host.view(-1).view(torch.uint8))
        for f in futs:
            f.result()
        if drop and hasattr(os, "posix_fadvise"):
            for _, r, _ in plan:
                if r.host is None:
                    os.posix_fadvise(self._fd(r.path), r.offset, r.nbytes, os.POSIX_FADV_DONTNEED)

    def close(self) -> None:
        self.pool.shutdown(wait=True)
        for fd in self.shared.values():
            os.close(fd)
        self.shared.clear()


class FileBlockStreamer:
    """`units`(順に呼ばれるモジュール列)のテンソルを safetensors から受け皿経由で GPU へ流す。
    `core/blockstream.BlockStreamer` と同じフック構成(root の forward 入口で先読み開始、各単位の pre/post hook)。

    - root: forward の入口になるモジュール(transformer、または token_refiner)
    - names_root: safetensors 上の名前の基準になるモジュール(transformer。単位のモジュールパスが名前の接頭辞)
    - drop_units: 末尾の何単位を読むたびにキャッシュから捨てるか("auto" = 使える RAM からはみ出す分)
    - reserve_gb: この後ホスト RAM に乗る見込みのもの(TE / VAE のピン留め等)。auto の予算から先に引く
    """

    def __init__(self, units, device, *, root: torch.nn.Module, names_root: torch.nn.Module, files: list[Path],
                 label: str, drop_units: str | int = "auto", reserve_gb: float = 0.0, ring: int = 2):
        self.units = list(units)
        self.device = torch.device(device)
        self.label = label
        self.n = len(self.units)
        index = safetensors_index(files)
        mod_names = {id(m): nm for nm, m in names_root.named_modules()}
        self.plans, self.masters, resident_bytes, missing, total = [], [], 0, [], 0
        staged_bytes, staged = 0, []
        for unit in self.units:
            prefix = mod_names[id(unit)] + "."
            plan, masters, pos = [], [], 0
            for (owner, attr, t), name in zip(collect_slots(unit), slot_names(unit, prefix)):
                ent = index.get(name) if name else None
                nb = t.numel() * t.element_size()
                if ent is not None and t.numel() == 0:
                    # 既にプレースホルダ(vaecache.detach_blocks など): 形と dtype はファイルの方を信じる
                    region = FileRegion(ent[0], ent[1], ent[2], ent[3], ent[4])
                elif ent is None or ent[2] != nb or ent[3] != t.dtype:
                    if _HOST_MIN_BYTES and nb >= _HOST_MIN_BYTES and t.numel() > 0:
                        # ファイルに無い大きめのもの(turbo LoRA): ホスト RAM に置いて単位ごとに流す
                        region = FileRegion(None, 0, nb, t.dtype, tuple(t.shape),
                                            host=t.detach().to("cpu").contiguous())
                        staged_bytes += nb
                        staged.append(name or attr)
                    else:
                        _set(owner, attr, t.to(self.device))      # ファイルに無い小さいものは GPU に常駐
                        resident_bytes += nb
                        if name:
                            missing.append(name)
                        continue
                else:
                    region = FileRegion(ent[0], ent[1], nb, t.dtype, tuple(t.shape))
                nb = region.nbytes
                plan.append(((owner, attr), region, pos))
                masters.append(((owner, attr), torch.empty(0, dtype=region.dtype)))
                _set(owner, attr, masters[-1][1])                  # mmap の原本を手放す
                pos += (nb + _ALIGN - 1) // _ALIGN * _ALIGN
            self.plans.append(plan)
            self.masters.append(masters)
            total += pos
        if staged:
            logger.info("file streaming (%s): %d tensors not in the files are staged from host RAM per unit (%.2f GiB), "
                        "e.g. %s", label, len(staged), staged_bytes / 2**30, staged[:2])
        if missing:
            logger.warning("file streaming (%s): %d tensors not found in the files, kept on the GPU (%.2f GiB), e.g. %s",
                           label, len(missing), resident_bytes / 2**30, missing[:3])
        self.host_staged_gb = staged_bytes / 2**30
        self.unit_bytes = max((plan[-1][2] + plan[-1][1].nbytes if plan else 0) for plan in self.plans) if self.plans else 0
        self.unit_bytes = (self.unit_bytes + _ALIGN - 1) // _ALIGN * _ALIGN
        self.total_gb = total / 2**30
        self.ring = max(2, int(ring))
        self.drop_from = self.n - self._decide_drop(drop_units, reserve_gb)
        # ページキャッシュに丸ごと収まる(重み + 余裕 < 使える RAM)なら Windows でもキャッシュさせる(再読込が速い)。
        # 収まらないものは sequential(キャッシュに残さない)。Linux では open のフラグに影響しない
        self.cached_reads = (self.total_gb + 1.0 + float(reserve_gb)) <= host_cache_budget_gb()
        self.reader = _Reader(sequential=not self.cached_reads)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"filestream-{label}")
        self.bufs = []
        self.buf_free: list[torch.cuda.Event | None] = [None] * self.ring
        self.pending: dict[int, tuple[int, Future]] = {}
        self.copy_stream = torch.cuda.Stream(self.device)
        self.pinned_bytes = 0
        self._ensure_bufs()
        self._handles = []
        for i, unit in enumerate(self.units):
            self._handles.append(unit.register_forward_pre_hook(self._make_pre(i)))
            self._handles.append(unit.register_forward_hook(self._make_post(i)))
        self._handles.append(root.register_forward_pre_hook(self._begin))
        self._wait = self._rt = self._t0 = 0.0
        self._passes = 0
        logger.info("file streaming (%s): %d units (%.2f GiB) read from %d files, staging %d x %.2f GiB pinned, "
                    "drop %d units after use, resident %.2f GiB, page cache %s",
                    label, self.n, self.total_gb, len(files), self.ring, self.unit_bytes / 2**30,
                    self.n - self.drop_from, resident_bytes / 2**30,
                    "kept (fits in RAM)" if self.cached_reads else "not kept (sequential)")

    def _decide_drop(self, drop_units, reserve_gb: float) -> int:
        if self.n == 0:
            return 0
        per_unit = self.total_gb / self.n
        if isinstance(drop_units, str) and drop_units.strip().lower() == "auto":
            budget = host_cache_budget_gb()
            margin = 1.0 + float(reserve_gb) + self.ring * self.unit_bytes / 2**30
            k = int(math.ceil(max(0.0, self.total_gb + margin - budget) / max(per_unit, 1e-9)))
            logger.info("file streaming (%s): page-cache budget %.1f GiB for %.1f GiB of weights (+%.1f GiB reserved) "
                        "-> drop %d/%d units after use", self.label, budget, self.total_gb, margin, min(k, self.n), self.n)
        else:
            k = int(drop_units)
        return min(max(k, 0), self.n)

    # -- 読み込み ------------------------------------------------------------------
    def _read_unit(self, i: int, b: int) -> None:
        ev = self.buf_free[b]
        if ev is not None:
            ev.synchronize()           # この受け皿からの前回の H2D が終わるまで待つ
        t = time.time()
        self.reader.read_unit(self.bufs[b], self.plans[i], drop=i >= self.drop_from)
        self._rt += time.time() - t

    def prefetch(self, i: int) -> None:
        if 0 <= i < self.n and i not in self.pending:
            b = i % self.ring
            self.pending[i] = (b, self.pool.submit(self._read_unit, i, b))

    def _reset(self) -> None:
        """新しい forward の入口: 前回の forward で消費されなかった先読み(FBC で飛んだ単位)を捨てる。"""
        for _, fut in self.pending.values():
            fut.result()
        self.pending.clear()

    # -- GPU へ ----------------------------------------------------------------------
    def _load_to_gpu(self, i: int) -> None:
        if i not in self.pending:
            self.prefetch(i)
        b, fut = self.pending.pop(i)
        t = time.time()
        fut.result()
        self._wait += time.time() - t
        if i == 0:
            self._t0 = t
        buf = self.bufs[b]
        compute = torch.cuda.current_stream(self.device)
        with torch.cuda.stream(self.copy_stream):
            for (owner, attr), r, pos in self.plans[i]:
                g = buf[pos:pos + r.nbytes].view(r.dtype).view(r.shape).to(self.device, non_blocking=True)
                g.record_stream(compute)
                _set(owner, attr, g)
            ev = torch.cuda.Event()
            ev.record(self.copy_stream)
        self.buf_free[b] = ev
        compute.wait_event(ev)

    def _ensure_bufs(self) -> None:
        """受け皿(ring 本)を用意する。torch.empty(匿名メモリ)を cudaHostRegister でピン留めする(pin_memory() の
        cudaHostAlloc は cgroup に file として計上され、ページキャッシュの予算計算から漏れる、image server で実測)。"""
        if self.bufs:
            return
        for _ in range(self.ring):
            buf = torch.empty(max(self.unit_bytes, _ALIGN), dtype=torch.uint8)
            err = torch.cuda.cudart().cudaHostRegister(buf.data_ptr(), buf.numel(), 0)
            if int(err) != 0:
                raise RuntimeError(f"cudaHostRegister failed ({err}): not enough lockable host memory?")
            self.bufs.append(buf)
        self.buf_free = [None] * self.ring
        self.pinned_bytes = self.ring * self.unit_bytes

    def park(self) -> float:
        """受け皿を一時的に返す(デコードの間など、この streamer を使わない区間でホスト RAM を空ける。次の forward の
        入口で作り直す)。返した GiB を返す。2026-11-19: RAM 16GB の PC ではデコード中の空き RAM がページキャッシュの
        成否を決めるので、transformer / token_refiner / TE の受け皿(計 1.6GiB 前後)を返して VAE デコーダの流す側を
        キャッシュに残りやすくする。"""
        if not self.bufs:
            return 0.0
        self._reset()
        torch.cuda.synchronize(self.device)
        cudart = torch.cuda.cudart()
        for buf in self.bufs:
            cudart.cudaHostUnregister(buf.data_ptr())
        freed = self.pinned_bytes / 2**30
        self.bufs = []
        self.pinned_bytes = 0
        return freed

    def _begin(self, _module, _args) -> None:
        self._ensure_bufs()
        self._reset()
        for i in range(min(self.ring - 1, self.n)):
            self.prefetch(i)

    def _make_pre(self, i: int):
        def pre(_m, _args):
            self._load_to_gpu(i)
            self.prefetch(i + self.ring - 1)
        return pre

    def _make_post(self, i: int):
        def post(_m, _args, _out):
            for (owner, attr), placeholder in self.masters[i]:
                _set(owner, attr, placeholder)      # GPU 側は参照が切れて解放(record_stream が再利用を守る)
            if i == self.n - 1:
                self._passes += 1
                if self._passes <= 3 or self._passes % 50 == 0:
                    logger.info("file streaming (%s) pass %d: %.2fs, %.2fs waiting for reads, %.0f ms/unit read",
                                self.label, self._passes, time.time() - self._t0, self._wait,
                                1000 * self._rt / max(self.n, 1))
                self._wait = self._rt = 0.0
        return post

    def release(self) -> float:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.pool.shutdown(wait=True)
        self.reader.close()
        return self.park()

    def describe(self) -> dict:
        return {"blocks": self.n, "host": "file", "weights_gib": round(self.total_gb, 2),
                "staging_gib": round(self.ring * self.unit_bytes / 2**30, 2), "drop_units": self.n - self.drop_from,
                "host_staged_gib": round(self.host_staged_gb, 2), "page_cache": "kept" if self.cached_reads else "sequential"}

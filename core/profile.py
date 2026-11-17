"""H3_PROFILE: 計算用 GPU の総 VRAM から主プロファイル(環境変数一式)を自動で選ぶ。

core/runner.py が `os.environ.get("H3_...")` を読み始める**前**に `apply()` を呼ぶ(runner.py の
DEVICE 定義の直後)。入れ方は `os.environ.setdefault` なので、明示指定された変数は常にそちらが勝つ。

- `H3_PROFILE=auto`(既定): 主プロファイル系の変数(`MAIN_KNOBS`)が**ひとつも**指定されていない
  ときだけ、総 VRAM に応じた級(`TIERS`)の既定値一式を入れる。ひとつでも指定があれば「手動」と
  みなして何も入れない(従来の運用・検証済み経路をバイト単位で変えないため)。
- `H3_PROFILE=96gb|80gb|48gb|32gb|16gb|8gb`: その級を強制(指定済みの変数はそのまま)。
  `H3_VRAM_LIMIT_GB` で小さいカードを模擬するときは総 VRAM が変わらないので、こちらで級を指定する。
- `H3_PROFILE=off`: 何もしない。

級の中身は README の「VRAM 級別の起動例」と同じ(group 級は turbo LoRA も既定で ON、下の _TURBO 参照)。別途、`H3_ATTN_BACKEND` が未指定で既定の sage が
使えない(sm_120 以外、または sageattention 未インストール)ときは `default`(SDPA)に落とす
(これはプロファイルの手動/自動に関わらず、未指定のときだけ)。
"""
from __future__ import annotations

import importlib.util
import logging
import os

import torch

logger = logging.getLogger("minimax_h3.profile")

# これらのどれかが環境にあれば「手動」(auto は何も入れない)
MAIN_KNOBS = (
    "H3_LOWVRAM", "H3_TRANSFORMER_QUANT", "H3_TE_PROJ", "H3_TE_QUANT", "H3_TE_PRUNE",
    "H3_VIDEO_VAE_FP16", "H3_TE_STREAM", "H3_TE_DIET", "H3_TE_DEVICE", "H3_PRUNED", "H3_KEEP_TRANSFORMER",
)

# group 級(32GB 以下)は turbo LoRA(4 ステップ蒸留)を既定で ON にする。group では起動時に巻かないと
# リクエスト単位では巻けず(ブロックストリーマの転送対象が構築時に固定される)、GUI のチェックからも使えないため。
# 不要なら H3_TURBO_LORA=0 を明示する(GUI で turbo を外せば 30 ステップで走る)。ファイルは ref2va 用(ref2v)と
# t2va / fl2va 用(fl2v)を分ける(2026-11-17 に A4000 で実測した組み合わせ)。48GB 以上は起動後にリクエスト単位で
# 巻けるので、GUI の turbo チェックに任せる。
_TURBO = {
    "H3_TURBO_LORA": "1",
    "H3_TURBO_LORA_FILE": "minimax_h3_ref2v_turbo_4step_v1.0_768p_bf16.safetensors",
    "H3_TURBO_LORA_FILE_BASE": "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors",
}
_16GB = {"H3_LOWVRAM": "group", "H3_TE_PROJ": "NicoLab28/ClipProj-MiniMax-H3", "H3_VIDEO_VAE_FP16": "1", **_TURBO}

# (名前, 総 VRAM の下限 GB(10 進), 既定値一式)。上から順に最初に当てはまった級を使う。
TIERS = (
    ("96gb", 90.0, {}),
    ("80gb", 70.0, {"H3_TRANSFORMER_QUANT": "int8"}),
    ("48gb", 40.0, {"H3_LOWVRAM": "1", "H3_TE_PRUNE": "1", "H3_VIDEO_VAE_FP16": "1"}),
    ("32gb", 20.0, {"H3_LOWVRAM": "group", "H3_TE_PRUNE": "1", **_TURBO}),
    ("16gb", 11.0, dict(_16GB)),
    ("8gb", 0.0, {**_16GB, "H3_TE_STREAM": "1", "H3_TE_DIET": "1"}),
)

INFO: dict = {"mode": "unapplied"}


# group の原本をファイルから流す(H3_GROUP_STREAM_HOST=file)判定と同じ閾値(core/runner.py の
# H3_GROUP_STREAM_FILE_BELOW_GB の既定)。file は LoRA バッファを流せないので turbo と併用できない。
FILE_STREAM_BELOW_GB = 48.0


def _host_ram_gb() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().total / 1e9
    except Exception:  # noqa: BLE001
        try:
            for line in open("/proc/meminfo"):
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1e6
        except OSError:
            pass
    return None


def tier_env(total_vram_gb: float, host_ram_gb: float | None, group_host: str = "auto") -> tuple[str, dict, str | None]:
    """(級, 既定値一式, turbo を外した理由 or None)。純粋関数(テスト用)。"""
    name, env = TIERS[-1][0], TIERS[-1][2]
    for n, floor, e in TIERS:
        if total_vram_gb >= floor:
            name, env = n, e
            break
    env = dict(env)
    note = None
    if "H3_TURBO_LORA" in env:
        below = float(os.environ.get("H3_GROUP_STREAM_FILE_BELOW_GB", "").strip() or FILE_STREAM_BELOW_GB)
        uses_file = group_host == "file" or (group_host == "auto" and host_ram_gb is not None and host_ram_gb < below)
        if uses_file:
            for k in list(_TURBO):
                env.pop(k, None)
            why = ("H3_GROUP_STREAM_HOST=file が指定されている" if group_host == "file"
                   else f"host RAM {host_ram_gb:.0f}GB が {below:.0f}GB 未満な")
            note = (f"{why}ので重みをファイルから流す(file)構成になり、"
                    "turbo LoRA は併用できないため入れない(GUI の turbo も無効)")
    return name, env, note


def _tier_for(total_gb: float) -> tuple[str, dict]:
    name, env, _ = tier_env(total_gb, _host_ram_gb(), os.environ.get("H3_GROUP_STREAM_HOST", "auto").strip().lower() or "auto")
    return name, env


def apply() -> dict:
    """環境変数を埋めて、選択結果(status 用)を返す。冪等。"""
    global INFO
    requested = os.environ.get("H3_PROFILE", "auto").strip().lower() or "auto"
    explicit = [k for k in MAIN_KNOBS if k in os.environ]
    info: dict = {"requested": requested, "explicit": explicit, "applied": {}, "tier": None, "total_vram_gb": None}

    if not torch.cuda.is_available():
        info["mode"] = "no-cuda"
        INFO = info
        return info

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    info["total_vram_gb"] = round(total_gb, 1)
    names = {name for name, _, _ in TIERS}

    if requested in ("off", "0", "manual", "none"):
        info["mode"] = "off"
    elif requested == "auto" and explicit:
        info["mode"] = "manual"
    else:
        host_ram = _host_ram_gb()
        group_host = os.environ.get("H3_GROUP_STREAM_HOST", "auto").strip().lower() or "auto"
        info["host_ram_gb"] = round(host_ram, 1) if host_ram else None
        if requested in names:
            floor = next(f for n, f, _ in TIERS if n == requested)
            tier, env, note = tier_env(floor, host_ram, group_host)
            info["mode"] = "forced"
        else:
            if requested != "auto":
                logger.warning("H3_PROFILE=%r は不明です(auto / off / %s)。auto として扱います",
                               requested, " / ".join(sorted(names)))
            tier, env, note = tier_env(total_gb, host_ram, group_host)
            info["mode"] = "auto"
        if note:
            info["note"] = note
        info["tier"] = tier
        for k, v in env.items():
            if k not in os.environ:
                os.environ[k] = v
                info["applied"][k] = v

    # 既定の sage は sm_120 専用ビルド。使えない環境で未指定なら SDPA に落とす(手動/自動を問わない)。
    if "H3_ATTN_BACKEND" not in os.environ:
        cap = torch.cuda.get_device_capability(0)
        has_sage = importlib.util.find_spec("sageattention") is not None
        if cap != (12, 0) or not has_sage:
            os.environ["H3_ATTN_BACKEND"] = "default"
            info["applied"]["H3_ATTN_BACKEND"] = "default"
            info["attn_reason"] = (f"sm_{cap[0]}{cap[1]} は sage の対象外" if cap != (12, 0)
                                   else "sageattention 未インストール")

    if info["mode"] in ("auto", "forced"):
        logger.info("H3_PROFILE=%s: 総 VRAM %.1fGB -> %s 級、設定 %s%s%s", requested, total_gb, info["tier"],
                    info["applied"] or "(なし = 既定)",
                    f"(H3_ATTN_BACKEND=default: {info['attn_reason']})" if "attn_reason" in info else "",
                    f"。{info['note']}" if info.get("note") else "")
    elif info["mode"] == "manual":
        logger.info("H3_PROFILE=auto: %s が明示指定されているので手動扱い(自動設定なし)%s", explicit,
                    f"。H3_ATTN_BACKEND=default({info['attn_reason']})" if "attn_reason" in info else "")
    INFO = info
    return info

"""H3 text_encoder (Qwen3-VL-32B, bnb-4bit) の窓付き層ストリーミング (`H3_TE_STREAM=1`)。

LTX-2.5 の `backends/ltx2_5/app/testream.py` (LTX25_TE_STREAM) からの移植。
言語モデル本体 (`text_encoder.model.language_model.layers`、NF4 量子化済み decoder 層。
H3_TE_PRUNE=1 の 51 層で ~13GiB) を pinned host memory に常駐させ、エンコード中だけ
層単位で GPU へ流す。denoise / decode の間は TE の LM 層は使われないので、その間の
GPU 常駐を ~13GiB 減らすのが狙い。

方式 (LTX-2.5 と同一。層ごとの移動なので diffusers-server CLAUDE.md #33 の
「モジュール丸ごとの CPU⇔GPU 往復」には当たらず、block-level group offload と同種):
- セットアップ時: 各層の param.data / buffer / bnb quant_state テンソル
  (absmax / code / offset、double-quant 時は state2 も) を pinned CPU へ移す。
- エンコード開始時 (`language_model.forward` の入口): 先頭 window 層分のコピーを
  コピー専用 stream へ投入する。
- 各層の forward_pre_hook: 自層のコピー完了 event を計算 stream が wait し、GPU
  テンソルに record_stream(計算 stream) を打つ。
- forward_hook (層完了後): その層を CPU ポインタへ復元して GPU コピーを手放し、
  window 先の層のコピーを投入する。
- エンコード終了時 (finally): 全層を CPU ポインタへ復元する (重みは不変なので
  D2H コピーは発生せず、ポインタを差し替えるだけ)。

LTX-2.5 版との差分 (H3 の TE 構造に合わせた変更点):
- 包む対象: LTX-2.5 は `text_encoder.forward` を包むが、H3 は `text_encoder.model(...)`
  (`Qwen3VLModel`) を直接呼ぶ (lm_head を通さない、`_encode_ref2va_prompt*` /
  diffusers の `get_qwen3vl_prompt_embeds`) ので `text_encoder.forward` は呼ばれない。
  そこで `text_encoder.model.language_model.forward` を包む (どの経路も必ずここを通る。
  vision tower は GPU 常駐のまま、LM 層だけをストリームする)。
- 層の取り方は同じ (`text_encoder.model.language_model.layers`)。embed_tokens は
  H3_TE_DIET が CPU ブリッジ化済み (ストリーム対象外)、norm / rotary は小さいので GPU 常駐。
- ホスト RAM ガード: pinned 確保前に MemAvailable を確認し、足りなければ何も動かさずに
  RuntimeError (33番の事故の教訓。黙ってスワップへ突入させない)。

制約: TE が計算用 GPU に直接載っている bnb-4bit 構成専用 (`H3_TE_DEVICE` 外部常駐・
`H3_TE_QUANT=none`・`H3_TE_PROJ` は対象外)。サーバはジョブを直列実行するため
begin/end の非再入は考慮不要。出力は全常駐と bit 一致 (値は不変、置き場所だけ変える)。
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger("minimax_h3.testream")


def _read_mem_available_gib() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024**2
    return float("inf")


def _collect_movable(module: torch.nn.Module):
    """層内のデバイス移動対象テンソルを (owner, attr, tensor) で列挙する。

    nn.Parameter の .data、named_buffers、および bnb Params4bit の
    quant_state 内テンソル (absmax / code / offset、double-quant 時は state2 も)。
    """
    items = []
    for _, p in module.named_parameters(recurse=True):
        items.append((p, "data", p.data))
        qs = getattr(p, "quant_state", None)
        if qs is not None:
            for attr in ("absmax", "code", "offset"):
                t = getattr(qs, attr, None)
                if isinstance(t, torch.Tensor):
                    items.append((qs, attr, t))
            nested = getattr(qs, "state2", None)
            if nested is not None:
                for attr in ("absmax", "code", "offset"):
                    t = getattr(nested, attr, None)
                    if isinstance(t, torch.Tensor):
                        items.append((nested, attr, t))
    for name, b in module.named_buffers(recurse=True):
        owner = module
        parts = name.split(".")
        for part in parts[:-1]:
            owner = getattr(owner, part)
        items.append((owner, parts[-1], b))
    return items


class WindowedLayerStreamer:
    def __init__(self, layers, window: int = 2):
        self.layers = list(layers)
        self.window = max(1, int(window))
        self.copy_stream = torch.cuda.Stream()
        self.events = [torch.cuda.Event() for _ in self.layers]
        self.cpu_snap = []
        self.gpu_hold = [None] * len(self.layers)
        pinned_gib = 0.0
        for layer in self.layers:
            snap = []
            for owner, attr, t in _collect_movable(layer):
                cpu = t.detach().to("cpu")
                if not cpu.is_pinned():
                    cpu = cpu.pin_memory()
                pinned_gib += cpu.numel() * cpu.element_size() / 1024**3
                snap.append((owner, attr, cpu))
                self._set(owner, attr, cpu)
            self.cpu_snap.append(snap)
        self.pinned_gib = pinned_gib
        for i, layer in enumerate(self.layers):
            layer.register_forward_pre_hook(self._pre(i))
            layer.register_forward_hook(self._post(i))

    @staticmethod
    def _set(owner, attr, tensor):
        if isinstance(owner, torch.nn.Parameter):
            owner.data = tensor
        else:
            setattr(owner, attr, tensor)

    def _onload(self, i: int) -> None:
        gpu_list = []
        with torch.cuda.stream(self.copy_stream):
            for owner, attr, cpu in self.cpu_snap[i]:
                g = cpu.to("cuda", non_blocking=True)
                self._set(owner, attr, g)
                gpu_list.append(g)
            self.events[i].record(self.copy_stream)
        self.gpu_hold[i] = gpu_list

    def _pre(self, i: int):
        def hook(_m, _args):
            cur = torch.cuda.current_stream()
            cur.wait_event(self.events[i])
            for g in self.gpu_hold[i] or []:
                g.record_stream(cur)

        return hook

    def _post(self, i: int):
        def hook(_m, _args, _out):
            for owner, attr, cpu in self.cpu_snap[i]:
                self._set(owner, attr, cpu)
            self.gpu_hold[i] = None
            nxt = i + self.window
            if nxt < len(self.layers):
                self._onload(nxt)

        return hook

    def begin(self) -> None:
        for i in range(min(self.window, len(self.layers))):
            self._onload(i)

    def end(self) -> None:
        # 計算完了を待ってから全層を pinned CPU ポインタへ戻す (重みは不変なので
        # D2H コピーは発生しない。GPU コピーは参照が切れて解放される)。
        torch.cuda.current_stream().synchronize()
        for i, snap in enumerate(self.cpu_snap):
            for owner, attr, cpu in snap:
                self._set(owner, attr, cpu)
            self.gpu_hold[i] = None


def estimate_pinned_gib(text_encoder) -> float:
    """ストリーム対象 (LM 層) のテンソル合計サイズ (GiB)。pinned 確保前のガード用。"""
    layers = text_encoder.model.language_model.layers
    total = 0
    for layer in layers:
        for _o, _a, t in _collect_movable(layer):
            total += t.numel() * t.element_size()
    return total / 1024**3


def apply_te_stream(text_encoder, window: int = 2, min_free_ram_gb: float = 10.0) -> float:
    """TE の言語モデル層を窓付きストリーミング化し、`language_model.forward` を begin/end で包む。

    H3_TE_DIET の前でも後でも動く (diet は embed_tokens / lm_head のみを触る)。
    `_install_qwen3vl_submodule_timing` (H3_PHASE_TIMING) が language_model.forward を
    差し替え済みでも、現在の forward をそのまま内側に取り込む。

    ホスト RAM ガード: 推定 pinned 量 (power-of-2 丸めの余裕 +20%) + `min_free_ram_gb`
    が MemAvailable を超えていたら、何も動かさずに RuntimeError。

    Returns: pinned host へ移した重みサイズ (GiB)。冪等 (2回目以降は 0.0)。
    """
    if getattr(text_encoder, "_te_stream_applied", False):
        return 0.0

    lm = text_encoder.model.language_model
    need_gib = estimate_pinned_gib(text_encoder) * 1.2 + float(min_free_ram_gb)
    avail_gib = _read_mem_available_gib()
    if avail_gib < need_gib:
        raise RuntimeError(
            f"H3_TE_STREAM=1: ホスト RAM の空きが足りません (MemAvailable {avail_gib:.1f}GiB < "
            f"必要 {need_gib:.1f}GiB = pinned 推定 x1.2 + 余裕 {min_free_ram_gb:.0f}GiB)。"
            "H3_TE_STREAM を外すか、他プロセスを止めてください。"
        )

    streamer = WindowedLayerStreamer(lm.layers, window=window)
    inner_forward = lm.forward

    def streamed_forward(*args, **kwargs):
        streamer.begin()
        try:
            return inner_forward(*args, **kwargs)
        finally:
            streamer.end()

    lm.forward = streamed_forward
    text_encoder._te_stream_applied = True
    text_encoder._te_streamer = streamer
    logger.info(
        "[H3_TE_STREAM] %d LM layers -> pinned host (%.2fGiB, window=%d)",
        len(streamer.layers), streamer.pinned_gib, streamer.window,
    )
    return streamer.pinned_gib

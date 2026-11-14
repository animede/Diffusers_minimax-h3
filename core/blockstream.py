"""transformer のブロックストリーミング (`H3_LOWVRAM=group` の `H3_GROUP_STREAM=own`、2026-11-15)。

diffusers の `enable_group_offload(block_level, use_stream=True)` の置き換え。狙いは同じ
(50 ブロックの int8 transformer をホスト RAM に常駐させ、計算中のブロックとその先読み分だけを GPU に置く)
だが、既製品は「グループごとのフック固定費」と「CPU が計算に先回りして GPU 側の解放が遅れる」
(diffusers-movie-server の `docs/lowvram-techniques.md` 4.2 節、flux3-klein-studio の `flux3_stream.py` でも
同じ結論) ため、`core/testream.py`(TE の窓付き層ストリーミング)と同じ設計をブロックに当てる。

- セットアップ: 各ブロックのテンソルを 1 本のピン留めバッファ(`cudaHostRegister`)へ写し、パラメータを
  その view に向ける。torchao の `Int8Tensor` は外側の wrapper ではなく内側の `qdata` / `scale` / `zero_point`
  を動かす(diffusers の `_restore_torchao_tensor` と同じ `setattr` 方式。`.data` 代入は wrapper しか
  替わらないので使えない)。元の CPU テンソルは写した端から手放すので RAM の一時増加は 1 ブロック分。
- forward の入口(transformer の forward_pre_hook)で先頭 window 個のコピーをコピー専用 stream に投入、
  各ブロックの forward_pre_hook で自ブロックの event を計算 stream が wait し、GPU テンソルに
  `record_stream(計算 stream)` を打つ(これが無いと先読みの再利用で壊れる)。forward_hook で
  ピン留め側へポインタを戻し(重みは不変なので書き戻しなし)、window 先のブロックのコピーを投入する。
- 先読みされずに呼ばれたブロック(FBC で途中のブロックが飛んだ次のステップなど)は pre hook が
  その場でコピーして待つ。常に正しく、遅くなるだけ。
- 解放: `release()` でフックを外しピン留めを解除する。torch のホスト側キャッシングアロケータを
  通らないので `_host_emptyCache` は不要(呼んでも害は無い)。

出力は group offload / 全常駐と同じ(値は不変、置き場所だけ変える)。
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger("minimax_h3.blockstream")

_ALIGN = 256


def _inner_names(tensor) -> list[str] | None:
    """torchao の wrapper subclass なら内側テンソルの属性名、普通のテンソルなら None。"""
    cls = type(tensor)
    names = list(getattr(cls, "tensor_data_names", []) or [])
    if not names:
        return None
    for attr in getattr(cls, "optional_tensor_data_names", []) or []:
        if getattr(tensor, attr, None) is not None:
            names.append(attr)
    return names


def collect_slots(module: torch.nn.Module) -> list[tuple[object, str, torch.Tensor]]:
    """ブロック内で GPU へ動かすテンソルの置き場所 (owner, attr, tensor)。

    - 普通の nn.Parameter: (param, "data", param.data)
    - torchao Int8Tensor などの wrapper subclass: (param, "qdata"/"scale"/..., 内側テンソル)
    - bnb Params4bit の quant_state: (quant_state, "absmax"/..., テンソル)
    - buffer: (親モジュール, 名前, テンソル)
    """
    slots, seen = [], set()
    for _, p in module.named_parameters(recurse=True):
        if id(p) in seen:
            continue
        seen.add(id(p))
        inner = _inner_names(p)
        if inner:
            for attr in inner:
                slots.append((p, attr, getattr(p, attr)))
        else:
            slots.append((p, "data", p.data))
        qs = getattr(p, "quant_state", None)
        if qs is not None:
            for owner in (qs, getattr(qs, "state2", None)):
                if owner is None:
                    continue
                for attr in ("absmax", "code", "offset"):
                    t = getattr(owner, attr, None)
                    if isinstance(t, torch.Tensor):
                        slots.append((owner, attr, t))
    for name, b in module.named_buffers(recurse=True):
        owner = module
        parts = name.split(".")
        for part in parts[:-1]:
            owner = getattr(owner, part)
        slots.append((owner, parts[-1], b))
    return slots


def _set(owner, attr: str, tensor: torch.Tensor) -> None:
    if attr == "data" and isinstance(owner, torch.nn.Parameter):
        owner.data = tensor
    else:
        setattr(owner, attr, tensor)


def slots_bytes(module: torch.nn.Module) -> int:
    return sum(t.numel() * t.element_size() for _, _, t in collect_slots(module))


class BlockStreamer:
    """`blocks`(順に呼ばれる nn.ModuleList)をピン留めしたホスト RAM に置き、window 個先読みで GPU へ流す。"""

    def __init__(self, blocks, device, *, window: int = 2, root: torch.nn.Module | None = None):
        self.blocks = list(blocks)
        self.device = torch.device(device)
        self.window = max(1, int(window))
        self.copy_stream = torch.cuda.Stream(self.device)
        self.events: list[torch.cuda.Event | None] = [None] * len(self.blocks)
        self.masters: list[list[tuple[object, str, torch.Tensor]]] = []
        self._pinned: list[torch.Tensor] = []
        self._handles = []
        self.pinned_bytes = 0
        for block in self.blocks:
            self.masters.append(self._pin_block(block))
        for i, block in enumerate(self.blocks):
            self._handles.append(block.register_forward_pre_hook(self._make_pre(i)))
            self._handles.append(block.register_forward_hook(self._make_post(i)))
        if root is not None:
            self._handles.append(root.register_forward_pre_hook(self._begin))
        logger.info("block streamer: %d blocks, %.2f GiB pinned, window %d, device %s",
                    len(self.blocks), self.pinned_bytes / 1024**3, self.window, self.device)

    # -- host memory ----------------------------------------------------------------------
    def _pin_block(self, block: torch.nn.Module):
        slots = collect_slots(block)
        sizes = [t.numel() * t.element_size() for _, _, t in slots]
        offsets, total = [], 0
        for size in sizes:
            offsets.append(total)
            total += (size + _ALIGN - 1) // _ALIGN * _ALIGN
        buffer = torch.empty(max(total, _ALIGN), dtype=torch.uint8)
        err = torch.cuda.cudart().cudaHostRegister(buffer.data_ptr(), buffer.numel(), 0)
        if int(err) != 0:
            raise RuntimeError(f"cudaHostRegister failed ({err}): not enough lockable host memory?")
        self._pinned.append(buffer)
        self.pinned_bytes += buffer.numel()
        masters = []
        for (owner, attr, t), offset, size in zip(slots, offsets, sizes):
            src = t.detach()
            if src.device.type != "cpu":
                src = src.cpu()
            view = buffer[offset:offset + size].view(src.dtype).view(src.shape)
            view.copy_(src.contiguous())
            _set(owner, attr, view)          # 元の CPU テンソルはここで参照が切れる
            masters.append((owner, attr, view))
        return masters

    def release(self) -> float:
        """フックを外し、ピン留めを解除する。解除した GiB を返す。"""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        cudart = torch.cuda.cudart()
        for buffer in self._pinned:
            cudart.cudaHostUnregister(buffer.data_ptr())
        freed = self.pinned_bytes / 1024**3
        self._pinned.clear()
        self.masters.clear()
        self.pinned_bytes = 0
        return freed

    # -- streaming ----------------------------------------------------------------------------
    def _prefetch(self, i: int) -> None:
        compute = torch.cuda.current_stream(self.device)
        with torch.cuda.stream(self.copy_stream):
            for owner, attr, pinned in self.masters[i]:
                g = pinned.to(self.device, non_blocking=True)
                g.record_stream(compute)
                _set(owner, attr, g)
            ev = torch.cuda.Event()
            ev.record(self.copy_stream)
            self.events[i] = ev

    def _begin(self, _module, _args) -> None:
        for i in range(min(self.window, len(self.blocks))):
            if self.events[i] is None:
                self._prefetch(i)

    def _make_pre(self, i: int):
        def pre(_m, _args):
            if self.events[i] is None:
                self._prefetch(i)      # 先読みされていない(FBC で前が飛んだ等): その場でコピー
            torch.cuda.current_stream(self.device).wait_event(self.events[i])
            self.events[i] = None
        return pre

    def _make_post(self, i: int):
        def post(_m, _args, _out):
            for owner, attr, pinned in self.masters[i]:
                _set(owner, attr, pinned)   # GPU 側は参照が切れて解放(record_stream が再利用を守る)
            nxt = i + self.window
            if nxt < len(self.blocks) and self.events[nxt] is None:
                self._prefetch(nxt)
        return post

    def describe(self) -> dict:
        return {"blocks": len(self.blocks), "window": self.window,
                "pinned_gib": round(self.pinned_bytes / 1024**3, 2)}


def move_non_block_children(root: torch.nn.Module, blocks_attr, device) -> float:
    """ブロック以外(埋め込み・出力層など)を GPU に置く。`blocks_attr` は名前か名前の集合(流す側に残す子)。
    移した GiB を返す。"""
    skip = {blocks_attr} if isinstance(blocks_attr, str) else set(blocks_attr)
    moved = 0
    for name, child in root.named_children():
        if name in skip:
            continue
        child.to(device)
        moved += sum(p.numel() * p.element_size() for p in child.parameters())
        moved += sum(b.numel() * b.element_size() for b in child.buffers())
    for name, p in root.named_parameters(recurse=False):
        p.data = p.data.to(device)
        moved += p.numel() * p.element_size()
    return moved / 1024**3

"""【probe 結果: 不採用】sage attention の CUDA 拡張 (_fused / _qattn_sm89) が PyTorch の current stream ではなく
default stream に launch するため capture 対象から漏れ (CUDA Graph is empty 警告)、replay が NaN になる。native attention
なら capture は eager と bit 一致するが、GPU 律速のため denoise は 5.37s → 5.35s (141f) と短縮ゼロ。既定 OFF のまま残置。

transformer_ref.forward 全体の CUDA Graph capture/replay(H3_DENOISE_CUDAGRAPH=1, 2026-10-05 probe)。

`backends/ltx2_5/app/cudagraph.py` の ForwardGraphRunner を H3 向けに移植したもの。方式は同じ:
side stream で warmup → clearCublasWorkspaces → 共有 mempool で capture → 以後は静的入力バッファへ
copy_ して replay。

H3 の forward は CPU 依存の分岐・同期(.item() 等)を持たず(diffusers transformer_minimax_h3.py)、
入力はすべてテンソル(hidden_states / audio_hidden_states / encoder_hidden_states / timestep /
timestep_indices / token_tags / position_ids / *_indices)+ attention_kwargs(dict, 非テンソル)なので
そのまま capture できる。capture キーは (テンソル引数の shape/dtype) + (非テンソル引数の repr)。

- `inspect.signature(transformer.forward)` を denoise ループ側が参照する
  (MiniMaxH3LoopDenoiser: layout_kwargs の絞り込み)ため、ラッパーに元 forward の `__signature__` を持たせる。
- capture_error_mode は "thread_local": decode (別 GPU, 別スレッド) が capture 中も走る可能性があり、
  "global" だと他スレッドの cudaMalloc 等で capture が無効化される。
"""
from __future__ import annotations

import inspect
import os
import logging
import time
from typing import Any

import torch

logger = logging.getLogger("minimax_h3.cudagraph")

_EAGER = object()


class ForwardGraphRunner:
    def __init__(self, module: torch.nn.Module, warmup_iters: int = 3, max_captures: int = 4):
        self._module = module
        self._orig_forward = module.forward
        self.__signature__ = inspect.signature(self._orig_forward)
        self._warmup_iters = warmup_iters
        self._max_captures = max_captures
        self._pool = None
        self._captures: dict[tuple, Any] = {}
        self.enabled = True
        self.replays = 0
        self.stats: list[dict] = []

    def install(self) -> None:
        self._module.forward = self

    def uninstall(self) -> None:
        if self._module.forward is self:
            del self._module.forward  # type: ignore[attr-defined]

    def reset(self) -> None:
        self._captures.clear()
        torch.cuda.synchronize()

    @staticmethod
    def _key(kwargs: dict):
        parts = []
        for k in sorted(kwargs):
            v = kwargs[k]
            if isinstance(v, torch.Tensor):
                parts.append((k, "T", tuple(v.shape), str(v.dtype), str(v.device)))
            elif isinstance(v, (int, float, str, bool, type(None))):
                parts.append((k, "S", v))
            elif isinstance(v, dict):
                if any(isinstance(x, torch.Tensor) for x in v.values()):
                    return None
                parts.append((k, "D", tuple(sorted((dk, repr(dv)) for dk, dv in v.items()))))
            else:
                return None
        return tuple(parts)

    def __call__(self, *args, **kwargs):
        if not self.enabled or args:
            return self._orig_forward(*args, **kwargs)
        key = self._key(kwargs)
        if key is None:
            return self._orig_forward(**kwargs)
        cap = self._captures.get(key)
        if cap is _EAGER:
            return self._orig_forward(**kwargs)
        if cap is None:
            if sum(1 for v in self._captures.values() if v is not _EAGER) >= self._max_captures:
                self._captures[key] = _EAGER
                return self._orig_forward(**kwargs)
            try:
                cap = self._capture(kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("H3 cudagraph: capture failed; this shape stays eager: %r", exc)
                torch.cuda.synchronize()
                self._captures[key] = _EAGER
                return self._orig_forward(**kwargs)
            self._captures[key] = cap
            if os.environ.get("H3_CG_VERIFY", "0") == "1":
                cap["graph"].replay()
                torch.cuda.synchronize()
                with torch.no_grad():
                    ref = self._orig_forward(**{k: (cap["inputs"][k] if k in cap["inputs"] else v) for k, v in kwargs.items()})
                torch.cuda.synchronize()
                for name, a, b in zip(("video", "audio"), cap["outputs"], ref):
                    a, b = a.float(), b.float()
                    logger.info("H3 cudagraph VERIFY %s: graph nan=%s eager nan=%s maxabs=%.4g meanabs_eager=%.4g mean_diff=%.4g",
                                name, bool(torch.isnan(a).any()), bool(torch.isnan(b).any()),
                                (a - b).abs().max().item(), b.abs().mean().item(), (a - b).abs().mean().item())
        else:
            for k, static in cap["inputs"].items():
                static.copy_(kwargs[k])
        cap["graph"].replay()
        self.replays += 1
        return cap["outputs"]

    def _capture(self, kwargs: dict) -> dict:
        statics: dict[str, torch.Tensor] = {}
        static_kwargs: dict[str, Any] = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                s = v.clone()
                statics[k] = s
                static_kwargs[k] = s
            else:
                static_kwargs[k] = v
        torch.cuda.synchronize()
        mem0 = torch.cuda.memory_allocated()
        res0 = torch.cuda.memory_reserved()
        t0 = time.time()
        with torch.no_grad():
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(self._warmup_iters):
                    self._orig_forward(**static_kwargs)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            t_warm = time.time() - t0
            torch._C._cuda_clearCublasWorkspaces()
            for k, s in statics.items():
                s.copy_(kwargs[k])
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool, capture_error_mode="thread_local"):
                outputs = self._orig_forward(**static_kwargs)
        torch.cuda.synchronize()
        st = {
            "n_inputs": len(statics), "total_s": round(time.time() - t0, 2), "warmup_s": round(t_warm, 2),
            "alloc_delta_gb": round((torch.cuda.memory_allocated() - mem0) / 1e9, 3),
            "reserved_delta_gb": round((torch.cuda.memory_reserved() - res0) / 1e9, 3),
            "shape": {k: tuple(v.shape) for k, v in statics.items() if k in ("hidden_states", "audio_hidden_states", "encoder_hidden_states")},
        }
        self.stats.append(st)
        logger.info("H3 cudagraph: captured shape #%d %s", len(self.stats), st)
        return {"graph": graph, "inputs": statics, "outputs": outputs}

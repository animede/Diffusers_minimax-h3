"""クエリ分割 attention(`H3_ATTN_Q_CHUNK`、2026-11-15)。

feed-forward の分割(H3_FF_CHUNK)のあと、デノイズの活性で残る最大のものは attention の Q・K・V・出力
(1024²×5 秒の 12.6 万トークンで各 1.35GB)。attention は行(クエリ)ごとに独立なので、K・V は丸ごと作り、
Q の射影・QK ノルム・RoPE・attention カーネル・出力射影を `chunk` トークンずつ回して、出力を書き込み先の
1 本に集める。同時に存在するのは K・V(2 本)+ 出力(1 本)+ チャンク分だけになり、Q 全体と attention 出力全体の
2 本(1024² で 2.7GB)が消える。トークンごとの演算と行ごとの softmax なので結果は同じ(chunk は 128 の倍数にして
sage attention のブロック境界と揃える)。`set_attention_backend()` が書く `_attention_backend` はそのまま使う。
"""

from __future__ import annotations

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3Attention,
    MiniMaxH3AttnProcessor,
    _apply_rotary_emb,
)


class QueryChunkedAttnProcessor(MiniMaxH3AttnProcessor):
    def __init__(self, chunk: int):
        self.chunk = int(chunk)

    def __call__(self, attn: MiniMaxH3Attention, hidden_states, rotary_emb=None, attention_mask=None):
        n = hidden_states.shape[1]
        chunk = self.chunk
        if chunk <= 0 or n <= chunk or getattr(attn, "fused_projections", False):
            return super().__call__(attn, hidden_states, rotary_emb, attention_mask)
        key = attn.norm_k(attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1)))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        if rotary_emb is not None:
            cos, sin = rotary_emb
            key = _apply_rotary_emb(key, cos, sin)
        out = torch.empty_like(hidden_states)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            query = attn.norm_q(attn.to_q(hidden_states[:, start:end]).unflatten(-1, (attn.heads, -1)))
            if rotary_emb is not None:
                query = _apply_rotary_emb(query, cos[start:end], sin[start:end])
            mask = attention_mask
            if mask is not None and mask.dim() >= 2 and mask.shape[-2] == n:
                mask = mask[..., start:end, :]
            part = dispatch_attention_fn(
                query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False,
                backend=self._attention_backend, parallel_config=self._parallel_config,
            )
            part = part.flatten(2, 3).type_as(query)
            out[:, start:end] = attn.to_out[1](attn.to_out[0](part))
        return out


def enable(module, chunk: int) -> int:
    """transformer_blocks の attention にクエリ分割プロセッサを入れる(token_refiner は小さいので対象外)。"""
    if chunk <= 0:
        return 0
    count = 0
    for block in module.transformer_blocks:
        attn = block.attn
        if not isinstance(attn, MiniMaxH3Attention) or isinstance(attn.processor, QueryChunkedAttnProcessor):
            continue
        old = attn.processor
        new = QueryChunkedAttnProcessor(chunk)
        new._attention_backend = getattr(old, "_attention_backend", None)
        new._parallel_config = getattr(old, "_parallel_config", None)
        attn.set_processor(new)
        count += 1
    return count

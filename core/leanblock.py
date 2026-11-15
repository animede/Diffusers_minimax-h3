"""メモリ節約版のブロック forward(`H3_LEAN_BLOCK`、2026-11-15)。

MiniMaxH3TransformerBlock の forward は、norm1 の出力・attention の出力・norm2 の出力・feed-forward の出力を
それぞれ全トークン分(1024²×5 秒で各 1.35GB)作ってから足し合わせる。ここでは同じ計算を `chunk` トークンずつ
「正規化 → 変調 → 射影/attention/FF → ゲート → 残差に加算」まで通し、全トークン分の中間を作らない。
attention の K・V だけは全トークン分が要る(行ごとの attention は全列を見る)ので、チャンクから組み立てた
2 本のバッファに置く。同時に存在する大きなテンソルは、入力(呼び出し元が保持)・出力・K・V の 4 本。

トークンごとの演算と行ごとの attention なので結果は変わらない(MD5 一致を確認)。フック(ブロック
ストリーミング・FBC)は `block.forward` / `attn.__call__` / `ff.__call__` の境界をそのまま通る:
attention 部は専用プロセッサ(`attn(...)` を 1 回呼ぶ)、FF 部は `ff.forward` の差し替え(`ff(...)` を 1 回呼ぶ)で、
それぞれ呼び出しの中でチャンクを回す。
"""

from __future__ import annotations

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3Attention,
    MiniMaxH3AttnProcessor,
    MiniMaxH3TransformerBlock,
    _apply_rotary_emb,
)


def _modulate(norm, x, scale, shift, idx):
    """norm(x) * (1 + scale[idx]) + shift[idx] — ブロックの元の式そのまま(チャンク単位)。"""
    return norm(x) * (1.0 + scale.index_select(0, idx)) + shift.index_select(0, idx)


class LeanAttnProcessor(MiniMaxH3AttnProcessor):
    """`attn._lean_ctx` が設定されていれば、残差の入力を受け取り、残差を加えた新しい hidden_states を返す。"""

    def __init__(self, chunk: int):
        self.chunk = int(chunk)

    def __call__(self, attn: MiniMaxH3Attention, hidden_states, rotary_emb=None, attention_mask=None):
        ctx = getattr(attn, "_lean_ctx", None)
        if ctx is None:
            return super().__call__(attn, hidden_states, rotary_emb, attention_mask)
        norm1, scale, shift, gate, idx = ctx
        bsz, n, _ = hidden_states.shape
        heads, chunk = attn.heads, self.chunk
        cos, sin = rotary_emb if rotary_emb is not None else (None, None)
        key = hidden_states.new_empty((bsz, n, heads, attn.head_dim))
        value = hidden_states.new_empty((bsz, n, heads, attn.head_dim))
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            rows = idx[start:end]
            nh = _modulate(norm1, hidden_states[:, start:end], scale, shift, rows)
            k = attn.norm_k(attn.to_k(nh).unflatten(-1, (heads, -1)))
            if cos is not None:
                k = _apply_rotary_emb(k, cos[start:end], sin[start:end])
            key[:, start:end] = k
            value[:, start:end] = attn.to_v(nh).unflatten(-1, (heads, -1))
            del nh, k
        out = torch.empty_like(hidden_states)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            rows = idx[start:end]
            nh = _modulate(norm1, hidden_states[:, start:end], scale, shift, rows)
            q = attn.norm_q(attn.to_q(nh).unflatten(-1, (heads, -1)))
            if cos is not None:
                q = _apply_rotary_emb(q, cos[start:end], sin[start:end])
            mask = attention_mask
            if mask is not None and mask.dim() >= 2 and mask.shape[-2] == n:
                mask = mask[..., start:end, :]
            o = dispatch_attention_fn(q, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False,
                                      backend=self._attention_backend, parallel_config=self._parallel_config)
            o = attn.to_out[1](attn.to_out[0](o.flatten(2, 3).type_as(q)))
            out[:, start:end] = hidden_states[:, start:end] + gate.index_select(0, rows) * o
            del nh, q, o
        del key, value
        return out


def _lean_ff_forward(ff, chunk: int):
    orig = ff.forward

    def forward(hidden_states, *args, **kwargs):
        ctx = getattr(ff, "_lean_ctx", None)
        if ctx is None:
            return orig(hidden_states, *args, **kwargs)
        norm2, scale, shift, gate, idx = ctx
        n = hidden_states.shape[1]
        out = hidden_states  # 自前のバッファ(LeanAttnProcessor の出力)を行ごとに更新する
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            rows = idx[start:end]
            nh = _modulate(norm2, out[:, start:end], scale, shift, rows)
            out[:, start:end] = out[:, start:end] + gate.index_select(0, rows) * orig(nh)
            del nh
        return out

    return forward


def _lean_block_forward(block: MiniMaxH3TransformerBlock, chunk: int):
    orig = block.forward

    def forward(hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
        if (attention_mask is not None or hidden_states.shape[1] <= chunk
                or getattr(block.attn, "fused_projections", False)
                or not isinstance(block.attn.processor, LeanAttnProcessor)):
            return orig(hidden_states, temb, adaln_indices, rotary_emb, attention_mask)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(temb)
        block.attn._lean_ctx = (block.norm1, scale_msa, shift_msa, gate_msa, adaln_indices)
        try:
            hidden_states = block.attn(hidden_states, rotary_emb, attention_mask)
        finally:
            block.attn._lean_ctx = None
        block.ff._lean_ctx = (block.norm2, scale_mlp, shift_mlp, gate_mlp, adaln_indices)
        try:
            hidden_states = block.ff(hidden_states)
        finally:
            block.ff._lean_ctx = None
        return hidden_states

    return forward


def enable(transformer, chunk: int) -> int:
    """全 transformer_blocks にメモリ節約版 forward を入れる。適用したブロック数を返す。冪等。"""
    if chunk <= 0:
        return 0
    count = 0
    for block in transformer.transformer_blocks:
        if getattr(block, "_h3_lean", False):
            continue
        old = block.attn.processor
        new = LeanAttnProcessor(chunk)
        new._attention_backend = getattr(old, "_attention_backend", None)
        new._parallel_config = getattr(old, "_parallel_config", None)
        block.attn.set_processor(new)
        block.attn._lean_ctx = None
        block.ff._lean_ctx = None
        block.ff.forward = _lean_ff_forward(block.ff, chunk)
        block.forward = _lean_block_forward(block, chunk)
        block._h3_lean = True
        count += 1
    return count

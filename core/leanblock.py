"""メモリ節約版のブロック forward(`H3_LEAN_BLOCK`、2026-11-15)。

MiniMaxH3TransformerBlock の forward は、norm1 の出力・attention の出力・norm2 の出力・feed-forward の出力を
それぞれ全トークン分作ってから足し合わせる。ここでは同じ計算を `chunk` トークンずつ
「正規化 → 変調 → 射影/attention/FF → ゲート → 残差に加算」まで通し、全トークン分の中間を作らない。
attention の K・V だけは全トークン分が要る(行ごとの attention は全列を見る)。

さらに 2 つの節約(いずれも結果はビット一致。トークンごとの演算と行・ヘッドごとの attention なので):
- 残差の in-place 更新: ブロック 2 以降は入力バッファに直接書き込む(各チャンクは自分の行を読んでから書く)。
  ブロック 0・1 は FBC(first block cache)がブロック 0 の入力と出力を参照するので新しいバッファに書く。
  ブロック 0 に diffusers のフックが無いときは全ブロックを in-place にする。
- ヘッド分割の attention(`H3_ATTN_HEAD_GROUPS`): K・V・Q をヘッド G 分割で順に作る(to_q/to_k/to_v の重みの
  行スライス。torchao の Int8Tensor は中身をスライスして組み直す)。各グループの attention 出力は全ヘッド分の 1 本
  (attn_out、K と同じ大きさ)に書き、出力射影はそこから全ヘッドまとめて行う。同時に存在する大きなバッファは
  K・V の 2 本から attn_out + K/G + V/G になり、カーネル内部の K・V の写し(sage は int8 の写しを作る)も 1/G になる。
  G=1 のときは attn_out を作らず従来どおり。auto は空き VRAM に収まる最小の G を選ぶ(1 → 4 → 7 → 8 → 14 → 28)。
  正規化と変調を G 回やり直すぶん遅くなる(512² 相当で G=4 が +13%、G=8 が +35%、G=28 が +140% 程度)。
  数学的には同じ計算だが、ビット一致は保証しない: GEMM(行数 7168/G)と attention カーネルのカーネル選択・
  加算順序がヘッド数とトークン数で変わり、bf16 の最下位ビットが動くことがある(native では G≤14 で一致、
  sage ではヘッド数とトークン数の組み合わせで一致したりしなかったりする)。ブロック単位で一致する組み合わせでも
  30 ステップの実機出力は MD5 が変わった(1024² の G=4・8、768² の G=8)。差は別のサンプル相当の動画になる程度
  (基準との PSNR 1024² 34.4dB、768² 38.7dB。FBC あり/なしの差 34.2dB と同程度)。品質の低下ではないが同一ではないので、
  auto は G=1 で収まるときは必ず G=1 を選ぶ。

フック(ブロックストリーミング・FBC)は `block.forward` / `attn.__call__` / `ff.__call__` の境界をそのまま通る:
attention 部は専用プロセッサ(`attn(...)` を 1 回呼ぶ)、FF 部は `ff.forward` の差し替え(`ff(...)` を 1 回呼ぶ)で、
それぞれ呼び出しの中でチャンクを回す。
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn.functional as F
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3Attention,
    MiniMaxH3AttnProcessor,
    MiniMaxH3TransformerBlock,
    _apply_rotary_emb,
)

logger = logging.getLogger("minimax_h3.leanblock")

# auto が試す順。2 は 1 と同じメモリで遅いだけ、56 は GEMM が小さすぎる(行 128 では一致もしない)ので外す。
HEAD_GROUP_CHOICES = (1, 4, 7, 8, 14, 28)
_AUTO_MARGIN = 512 << 20  # auto の見積もりに残す余裕(アロケータの断片化・小物)


def _modulate(norm, x, scale, shift, idx):
    """norm(x) * (1 + scale[idx]) + shift[idx] — ブロックの元の式そのまま(チャンク単位)。"""
    return norm(x) * (1.0 + scale.index_select(0, idx)) + shift.index_select(0, idx)


def _divisor_at_least(n: int, g: int) -> int:
    """n の約数で g 以上の最小のもの。"""
    g = max(1, min(int(g), n))
    while n % g:
        g += 1
    return g


def _slice_rows(weight, rows: slice):
    """重みの行(出力特徴)スライス。ブロックストリーマは torchao の Int8Tensor の中身(qdata・scale)だけを GPU に
    差し替えるので、外側の device は CPU のまま。そのため通常のスライス(return_and_correct_aliasing)が device 不一致で
    失敗する。そのときは中身をスライスして同じ型で組み直す(torchao の aten.slice 実装と同じ手順)。"""
    try:
        return weight[rows]
    except RuntimeError:
        if not hasattr(weight, "qdata") or not hasattr(weight, "scale"):
            raise
    qdata = weight.qdata[rows]
    scale = weight.scale[rows] if weight.scale.numel() > 1 else weight.scale
    zero_point = weight.zero_point
    if zero_point is not None and zero_point.numel() > 1:
        zero_point = zero_point[rows]
    block_size = [min(b, n) for b, n in zip(weight.block_size, qdata.shape)]
    return type(weight)(qdata, scale, block_size, weight.dtype, zero_point=zero_point,
                        act_quant_kwargs=weight.act_quant_kwargs, act_quant_scale=weight.act_quant_scale,
                        act_quant_zero_point=weight.act_quant_zero_point, act_pre_scale=weight.act_pre_scale)


class LeanState:
    """transformer 1 本分の共有状態。ブロック 0 の forward ごとに G と in-place の可否を決める。"""

    def __init__(self, chunk: int, groups: int, ff_inner: int):
        self.chunk = int(chunk)
        self.groups = int(groups)          # 0 = auto
        self.ff_inner = int(ff_inner)
        self.groups_now = 1
        self.inplace_all = False
        self.sliceable = True
        self.last: dict = {}
        self.debug = os.environ.get("H3_LEAN_DEBUG", "0").strip() == "1"
        self._warned = False

    def begin_forward(self, block, hidden_states) -> None:
        attn = block.attn
        registry = getattr(block, "_diffusers_hook", None)
        self.inplace_all = registry is None or not getattr(registry, "hooks", None)
        groups = self.groups if self.groups > 0 else self._auto_groups(hidden_states, attn)
        groups = _divisor_at_least(attn.heads, groups)
        if groups > 1 and not self._can_slice(attn):
            groups = 1
        self.groups_now = groups
        info = {"tokens": int(hidden_states.shape[1]), "batch": int(hidden_states.shape[0]), "head_groups": groups,
                "inplace_from_block": 0 if self.inplace_all else 2}
        if info != self.last:
            self.last = info
            logger.info("lean block: %s", info)
        if self.debug and hidden_states.is_cuda:
            free, _ = torch.cuda.mem_get_info(hidden_states.device)
            logger.info("lean block: block 0 begins, free %.2f GiB, allocated %.2f GiB, peak %.2f GiB",
                        free / 2**30, torch.cuda.memory_allocated(hidden_states.device) / 2**30,
                        torch.cuda.max_memory_allocated(hidden_states.device) / 2**30)

    def _can_slice(self, attn) -> bool:
        """ヘッド分割は to_q/to_k/to_v が bias 無しの素の nn.Linear(LoRA 等が被っていない)で、重みを行スライス
        できるときだけ。できないときは G=1 に落として一度だけ警告する。"""
        try:
            for module in (attn.to_q, attn.to_k, attn.to_v):
                if type(module) is not torch.nn.Linear or module.bias is not None:
                    raise TypeError(f"{type(module).__name__} is not a plain nn.Linear")
                _slice_rows(module.weight, slice(0, attn.head_dim))
            return True
        except Exception as error:  # noqa: BLE001 - 分割できない重み型は従来の経路で動かす
            if not self._warned:
                self._warned = True
                logger.warning("lean block: head groups disabled (%s: %s)", type(error).__name__, error)
            return False

    def _auto_groups(self, hidden_states, attn) -> int:
        """空き VRAM に収まる最小の G。attention 段で同時に要るのは K・V(とカーネル内部の写し)、G>1 なら attn_out、
        ブロック 0・1 の出力バッファ、チャンク分の一時メモリ。FBC が後で持つ残差 1 本と余裕を先に引く。"""
        if not hidden_states.is_cuda:
            return 1
        device = hidden_states.device
        bsz, n, dim = hidden_states.shape
        elem = hidden_states.element_size()
        kv = bsz * n * attn.inner_dim * elem                 # K・V・attn_out の 1 本
        hidden = bsz * n * dim * elem                        # 残差ストリームの 1 本
        chunk = min(self.chunk, n) * elem
        internal = 1.5  # カーネル内部の K・V の写し(sage は int8 の K と V の写しを作る)、K・V 1 本に対する比
        free, total = torch.cuda.mem_get_info(device)
        reserved, allocated = torch.cuda.memory_reserved(device), torch.cuda.memory_allocated(device)
        fraction = getattr(torch.cuda, "get_per_process_memory_fraction", lambda *_: 1.0)(device)
        if fraction < 1.0:
            free = min(free, int(total * fraction) - reserved)
        free += reserved - allocated
        budget = free - hidden - _AUTO_MARGIN
        choice = HEAD_GROUP_CHOICES[-1]
        for g in HEAD_GROUP_CHOICES:
            g = _divisor_at_least(attn.heads, g)
            need = ((2 + internal) * kv) // g + (kv if g > 1 else 0) + hidden
            need += chunk * (4 * dim + 2 * attn.inner_dim // g)   # norm 出力・出力射影・ゲート積・q・o のチャンク
            if need <= budget:
                choice = g
                break
        if self.debug:
            logger.info("lean block: auto head groups -> %d (free %.2f GiB, budget %.2f GiB, K+V %.2f GiB, tokens %d)",
                        choice, free / 2**30, budget / 2**30, 2 * kv / 2**30, n)
        return choice


class LeanAttnProcessor(MiniMaxH3AttnProcessor):
    """`attn._lean_ctx` が設定されていれば、残差の入力を受け取り、残差を加えた新しい hidden_states を返す。"""

    def __init__(self, chunk: int):
        self.chunk = int(chunk)

    def __call__(self, attn: MiniMaxH3Attention, hidden_states, rotary_emb=None, attention_mask=None):
        ctx = getattr(attn, "_lean_ctx", None)
        if ctx is None:
            return super().__call__(attn, hidden_states, rotary_emb, attention_mask)
        norm1, scale, shift, gate, idx, inplace, groups = ctx
        bsz, n, _ = hidden_states.shape
        heads, head_dim, chunk = attn.heads, attn.head_dim, self.chunk
        cos, sin = rotary_emb if rotary_emb is not None else (None, None)

        def modulated(start, end):
            return _modulate(norm1, hidden_states[:, start:end], scale, shift, idx[start:end])

        def project(nh, linear, weight, h, start, end, norm):
            x = linear(nh) if weight is None else F.linear(nh, weight)
            x = norm(x.unflatten(-1, (h, -1)))
            return _apply_rotary_emb(x, cos[start:end], sin[start:end]) if cos is not None else x

        def build_kv(h, wk, wv):
            key = hidden_states.new_empty((bsz, n, h, head_dim))
            value = hidden_states.new_empty((bsz, n, h, head_dim))
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                nh = modulated(start, end)
                key[:, start:end] = project(nh, attn.to_k, wk, h, start, end, attn.norm_k)
                v = attn.to_v(nh) if wv is None else F.linear(nh, wv)
                value[:, start:end] = v.unflatten(-1, (h, -1))
                del nh, v
            return key, value

        def attend(q, key, value, start, end):
            mask = attention_mask
            if mask is not None and mask.dim() >= 2 and mask.shape[-2] == n:
                mask = mask[..., start:end, :]
            return dispatch_attention_fn(q, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False,
                                         backend=self._attention_backend, parallel_config=self._parallel_config)

        # in-place のときも各チャンクは自分の行を読み終えてから書くので、結果は新しいバッファに書く場合と同じ
        out = hidden_states if inplace else torch.empty_like(hidden_states)
        if groups <= 1:
            key, value = build_kv(heads, None, None)
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                nh = modulated(start, end)
                q = project(nh, attn.to_q, None, heads, start, end, attn.norm_q)
                o = attend(q, key, value, start, end)
                o = attn.to_out[1](attn.to_out[0](o.flatten(2, 3).type_as(q)))
                out[:, start:end] = hidden_states[:, start:end] + gate.index_select(0, idx[start:end]) * o
                del nh, q, o
            del key, value
            return out

        hpg = heads // groups
        wq, wk, wv = attn.to_q.weight, attn.to_k.weight, attn.to_v.weight
        attn_out = hidden_states.new_empty((bsz, n, heads, head_dim))
        for g in range(groups):
            rows = slice(g * hpg * head_dim, (g + 1) * hpg * head_dim)
            key, value = build_kv(hpg, _slice_rows(wk, rows), _slice_rows(wv, rows))
            wq_g = _slice_rows(wq, rows)
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                nh = modulated(start, end)
                q = project(nh, None, wq_g, hpg, start, end, attn.norm_q)
                attn_out[:, start:end, g * hpg:(g + 1) * hpg] = attend(q, key, value, start, end).type_as(q)
                del nh, q
            del key, value
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            o = attn_out[:, start:end].flatten(2, 3)
            o = attn.to_out[1](attn.to_out[0](o if o.is_contiguous() else o.contiguous()))
            out[:, start:end] = hidden_states[:, start:end] + gate.index_select(0, idx[start:end]) * o
            del o
        del attn_out
        return out


def _lean_ff_forward(ff, chunk: int):
    orig = ff.forward

    def forward(hidden_states, *args, **kwargs):
        ctx = getattr(ff, "_lean_ctx", None)
        if ctx is None:
            return orig(hidden_states, *args, **kwargs)
        norm2, scale, shift, gate, idx = ctx
        n = hidden_states.shape[1]
        out = hidden_states  # LeanAttnProcessor の出力(自前または in-place のバッファ)を行ごとに更新する
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            rows = idx[start:end]
            nh = _modulate(norm2, out[:, start:end], scale, shift, rows)
            out[:, start:end] = out[:, start:end] + gate.index_select(0, rows) * orig(nh)
            del nh
        return out

    return forward


def _lean_block_forward(block: MiniMaxH3TransformerBlock, index: int, state: LeanState):
    orig = block.forward

    def forward(hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
        if (attention_mask is not None or hidden_states.shape[1] <= state.chunk
                or getattr(block.attn, "fused_projections", False)
                or not isinstance(block.attn.processor, LeanAttnProcessor)):
            return orig(hidden_states, temb, adaln_indices, rotary_emb, attention_mask)
        if index == 0:
            state.begin_forward(block, hidden_states)
        inplace = (index >= 2 or state.inplace_all) and not hidden_states.requires_grad
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(temb)
        block.attn._lean_ctx = (block.norm1, scale_msa, shift_msa, gate_msa, adaln_indices, inplace, state.groups_now)
        try:
            hidden_states = block.attn(hidden_states, rotary_emb, attention_mask)
        finally:
            block.attn._lean_ctx = None
        block.ff._lean_ctx = (block.norm2, scale_mlp, shift_mlp, gate_mlp, adaln_indices)
        try:
            hidden_states = block.ff(hidden_states)
        finally:
            block.ff._lean_ctx = None
        if state.debug and hidden_states.is_cuda and index in (0, 1, 2):
            logger.info("lean block: block %d done, allocated %.2f GiB, peak %.2f GiB", index,
                        torch.cuda.memory_allocated(hidden_states.device) / 2**30,
                        torch.cuda.max_memory_allocated(hidden_states.device) / 2**30)
        return hidden_states

    return forward


def shrink_prologue(transformer) -> list[str]:
    """forward の前処理で作る全トークン分の fp32 埋め込み(`proj_in` / `audio_proj_in` の出力。1024²×5 秒で 0.68GB)は、
    ブロックのループ中もローカル変数として生き残る。使う側は直後に bf16(`context_embedder` の dtype)へ丸めるだけ
    なので、射影の出力をその dtype に丸めて返す。結果は同じで、保持が半分になる。適用したモジュール名を返す。冪等。"""
    names = []
    for name in ("proj_in", "audio_proj_in"):
        module = getattr(transformer, name, None)
        if module is None or getattr(module, "_h3_lean_rounded", False):
            continue
        orig = module.forward

        def forward(x, _orig=orig):
            return _orig(x).to(transformer.context_embedder.weight.dtype)

        module.forward = forward
        module._h3_lean_rounded = True
        names.append(name)
    return names


def enable(transformer, chunk: int, groups: int = 0) -> int:
    """全 transformer_blocks にメモリ節約版 forward を入れる。適用したブロック数を返す。冪等。
    `groups` はヘッド分割数(0 = auto)。状態は `transformer._h3_lean_state`(status 用に `.last` を持つ)。"""
    if chunk <= 0:
        return 0
    blocks = list(transformer.transformer_blocks)
    state = getattr(transformer, "_h3_lean_state", None)
    if state is None:
        ff_inner = getattr(getattr(blocks[0].ff.net[0], "proj", None), "out_features", 0) // 2 if blocks else 0
        state = LeanState(chunk, groups, ff_inner)
        transformer._h3_lean_state = state
    count = 0
    for index, block in enumerate(blocks):
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
        block.forward = _lean_block_forward(block, index, state)
        block._h3_lean = True
        count += 1
    return count

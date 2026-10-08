"""AdaLN modulation precompute for the MiniMax-H3 transformer (bf16 path only).

Ported from NVIDIA Sol-Engine's `sana-sol-engine` repo
(`models/minimax_h3/GB200/adaln.py`, Apache-2.0) with the mechanism kept verbatim --
only the call sites (this project's `MiniMaxH3Runner._ensure_transformer(_ref)`
instead of Sol-Engine's own GB200 loader script) and the module-level env-var/guard
wiring are new.

Why this works (Sol-Engine's own reasoning, cross-checked against this project's
pinned diffusers commit -- see `enable_adaln_precompute()`'s docstring below for the
exact places this was verified against `transformer_minimax_h3.py` /
`modular_pipelines/minimax_h3/denoise.py` in THIS venv, not assumed from upstream):

Roughly 13B of MiniMax-H3's 33B parameters sit in the per-block `adaln_proj`, a
`Linear(2688 -> 6 * 5376 * 3)` that every one of the 50 blocks evaluates on every
denoising step. Its input is the timestep embedding alone: a `(num_timesteps, 2688)`
tensor with a handful of rows that depends on nothing but the sampling schedule, which
this project's own `MiniMaxH3SetTimestepsStep` call fixes before the denoise loop
starts (`row_timestep_plan` is fully built by then -- see `core/runner.py`'s
`generate()`, right before `MiniMaxH3DenoiseStep`/`MiniMaxH3LoopDenoiser` run). So the
whole modulation table for the whole trajectory is knowable up front, and diffusers
recomputes it once per (block, step) anyway -- this just does that computation once,
up front, into a small table, and drops the weights that would otherwise sit
GPU-resident doing nothing else for the rest of the request.

What this buys (Sol-Engine's own measurement, in order of size):

* ~24 GB of GPU memory. ~26 GB of `adaln_proj` weights (50 blocks) are replaced by a
  ~1.5 GB table (50 steps x 50 blocks x up to 9 rows x 32256 values, bfloat16 -- the
  row count in THIS project's case is capped by `MINIMAX_H3_MODALITY_NUM * distinct
  timesteps per step`, see `precompute()`'s own docstring). The bf16 transformer's
  resident footprint drops from ~66.3GB to roughly ~42GB.
* ~26 GB of HBM reads per step eliminated. Each block used to stream ~520MB of weights
  to produce nine rows of output -- a pure bandwidth tax on an operation with
  essentially no arithmetic intensity relative to attention/FFN.

The precompute runs one GEMM per (block, step) at exactly the shapes the reference
`MiniMaxH3AdaLayerNormModulation.forward()` uses, rather than one batched GEMM per
block over all steps -- slower by a few tens of milliseconds, once, in exchange for
values that are bitwise identical to what the unmodified model would have computed.
This technique therefore needs no quality gate: the verification this task ran
(`docs/h3-adaln-precompute-20260826.md`) is an `ffmpeg framemd5` exact-match check,
not an A/B similarity threshold.

v1 scope (see `core/runner.py`'s `H3_ADALN_PRECOMP` block for the enforced guards):
bf16 transformer only (`H3_TRANSFORMER_QUANT=none`), no `H3_LOWVRAM`/`H3_LOWVRAM_GROUP`,
and rejected outright against turbo (`H3_TURBO_LORA` env default, and any per-request
`turbo=True` override -- see `core/settings.py`'s `validate_instant_settings()`). The
turbo-incompatibility was written up as "a real structural conflict, not just an
unverified combination", reasoning from `core/runner.py`'s `apply_turbo_lora()`
(comfy-format LoRA): that function wraps every block's `adaln_proj.linear` in a
`_TurboLoRALinear` (see `_turbo_lora_key_map()`'s docstring, `blocks.N.adaln_proj.linear`
key), toggled per-REQUEST via `_TurboLoRALinear.enabled` on an already-resident
transformer instance (`set_turbo_lora_enabled()`), not fixed at load time -- true for
that checkpoint format, but this v1 analysis generalized from the comfy-format LoRA to
"turbo" as a whole without checking whether the *other* supported format touches the
same submodule.

v2 (coexistence, this module's current scope -- see `docs/h3-adaln-precompute-
20260826.md`'s dated coexistence section for the full verification writeup): it does
not. This project's actual default/production turbo LoRA
(`H3_TURBO_LORA_REPO=lightx2v/Minimax-h3-Turbo`, the diffusers-native/DMD-distilled
format `apply_diffusers_turbo_lora()` applies) was checked key-by-key against all 5 of
its non-comfyui-mirror checkpoint files on HF Hub (t2va/fl2va 4step/8step v0.1/v1.0/v1.1
variants) -- every one of them adapts only `attn.{to_q,to_k,to_v,to_out.0}` and
`ff.net.{0.proj,2}` (312 wrapped Linears each, matching `apply_diffusers_turbo_lora()`'s
own docstring count), and NONE of them has an `adaln_proj`- or `norm_out`-prefixed key
(0 out of 312 paths, for all 5 files). `apply_diffusers_turbo_lora()`'s own key
derivation (`paths = sorted({k.rsplit(".lora_", 1)[0] ...})`, reading straight off
whatever keys the checkpoint actually has) confirms this is not an artifact of a
hand-maintained key map that might be missing an entry the way the comfy-format
`_turbo_lora_key_map()` is (that one *is* hand-maintained and explicitly includes
`adaln_proj.linear`/`norm_out.linear`) -- the diffusers-native LoRA genuinely never
learned an AdaLN delta, so wrapping the rest of the transformer with it changes nothing
about what `block.adaln_proj`/`self.norm_out` compute. That makes the two techniques
independent for this format: a table built by running the trajectory GEMMs through
`block.adaln_proj` (turbo-wrapped elsewhere in the transformer, or not -- irrelevant,
since turbo never wraps `adaln_proj` itself under this format) is bit-exact for both
turbo=True and turbo=False requests off the exact same table, with no rebuild needed
on toggle. `enable_adaln_precompute()`'s own docstring below documents the remaining
ordering requirement this relies on (precompute must not fire before the request's
`apply_instant_settings()` turbo wrap has already run, purely so precompute doesn't
observe half-wrapped state mid-request -- not because the wrap could change what
`adaln_proj` computes).

The comfy-format LoRA (`larryvrh/MiniMax-H3-Turbo-Lora`, `_TURBO_COMFY_REPOS`) is a
genuine exception: its checkpoint DOES carry 51 `adaln_proj`/`norm_out`-prefixed
keys (verified directly against all 3 cached snapshots of the un-pruned original,
259 total paths per file, 51 of them AdaLN: 50 blocks' `adaln_proj.linear` + 1
`final_layer.adaln_proj.linear` -> `norm_out.linear`, per `_turbo_lora_key_map()`'s own
docstring) and `apply_turbo_lora()` wraps them the same way as everything else.
`families/qwen_image` CLAUDE.md #44 in the sibling diffusers-server project hit the
identical fused-attn-plus-AdaLN LoRA shape (IC-LoRA) and the fix there was the same one
this module uses: never let precompute silently observe a turbo-wrapped
`adaln_proj.linear` and treat it as the plain projection. `precompute()` (below) now
raises loudly if it finds any `block.adaln_proj.linear` (the specific submodule it
reads and then deletes -- `self.norm_out.linear`/`final_layer` is a separate module
`precompute()` has never touched, in v1 or v2, so a comfy-format wrap surviving there
is harmless and intentionally left alone) already wrapped in a `_TurboLoRALinear` when
it runs -- this can only happen with the comfy-format LoRA (the only format that ever
wraps that specific submodule), so this guard is v2's enforcement point for keeping the
comfy format out of the precompute path, replacing v1's blanket "reject all turbo" rule
at the settings-validation layer with a narrower, format-specific one (see
`core/settings.py`'s `validate_instant_settings()` and `core/runner.py`'s
`H3_ADALN_PRECOMP` import-time block, both updated to match).
"""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger("minimax_h3.adaln_precompute")


class _StepCursor:
    """Which denoising step the block stack is currently evaluating.

    The index lives in a device tensor rather than a Python int so that a future
    `torch.compile` pass over the block stack would see one graph for the whole
    trajectory instead of specializing (and recompiling) on every step's Python int.
    This project does not currently `torch.compile` the H3 transformer, but keeping
    this device-tensor shape costs nothing and keeps the port faithful to Sol-Engine's
    own reasoning (see its `_StepCursor` docstring) in case that changes later.
    """

    __slots__ = ("step",)

    def __init__(self, device: torch.device) -> None:
        self.step = torch.zeros((), dtype=torch.long, device=device)

    def set(self, index: int) -> None:
        self.step.fill_(index)


class PrecomputedModulation(nn.Module):
    """Drop-in replacement for `MiniMaxH3AdaLayerNormModulation` that indexes a table.

    Holds `(num_steps, num_rows, 6 * hidden_size)` and returns the six chunks of the
    row block belonging to the current step, matching
    `MiniMaxH3AdaLayerNormModulation.forward()`'s own return shape exactly (see
    `transformer_minimax_h3.py`: `temb.chunk(6, dim=-1)`).
    """

    def __init__(self, table: torch.Tensor, cursor: _StepCursor) -> None:
        super().__init__()
        self.register_buffer("table", table, persistent=False)
        self.cursor = cursor

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # `table[step_tensor]` would be advanced indexing on a data-dependent value,
        # which `torch.compile(fullgraph=True)` rejects (cannot prove the result's
        # static shape). `index_select` with a length-1 index tensor has a statically
        # known output shape, so the step index can stay a runtime value without
        # breaking a future compiled graph. Not exercised today (this project does not
        # compile the H3 transformer), kept for parity with the reference and because
        # it costs nothing over the advanced-indexing form.
        rows = self.table.index_select(0, self.cursor.step.reshape(1))[0]
        return rows.chunk(6, dim=-1)


def _timestep_embedding(transformer, timestep: torch.Tensor) -> torch.Tensor:
    """The `temb` the block stack would have been handed for this step.

    Mirrors `MiniMaxH3Transformer3DModel.forward()`'s own two lines exactly (see
    `transformer_minimax_h3.py`, around `temb = self.time_proj(timestep); temb =
    self.time_embedder(temb.to(self.time_embedder.linear_1.weight.dtype))`) -- the cast
    to `time_embedder`'s own weight dtype matters because this checkpoint's
    `time_embedder` is a float32 module inside an otherwise-bf16 transformer (see that
    method's own comment), and reproducing that cast exactly is part of what makes the
    precomputed table bitwise identical to the uncached path.
    """
    temb = transformer.time_proj(timestep)
    return transformer.time_embedder(temb.to(transformer.time_embedder.linear_1.weight.dtype))


def _reject_turbo_wrapped_adaln(transformer) -> None:
    """Refuse to precompute if any `block.adaln_proj.linear` is already a
    `_TurboLoRALinear` wrapper.

    This is the coexistence guard described in this module's own docstring (v2
    section): the default/production turbo LoRA (diffusers-native format,
    `apply_diffusers_turbo_lora()`) never wraps `adaln_proj` at all, verified against
    every non-comfyui-mirror checkpoint file lightx2v ships -- so in the normal case
    this loop finds nothing and returns immediately. The only checkpoint format that
    *does* wrap `adaln_proj.linear` is the comfy-format LoRA
    (`apply_turbo_lora()`/`_turbo_lora_key_map()`, `_TURBO_COMFY_REPOS`), which this
    project already keeps out of int8/lowvram mode for an unrelated reason (`aten.cat`
    on `Int8Tensor`, see `core/settings.py`) but does NOT otherwise block from the bf16
    path this module targets. If that format ever reaches here, `precompute()`'s own
    `for parameter in projection.linear.parameters()` walk would silently read
    `_TurboLoRALinear.base`'s parameters (the property aliasing at the top of this
    file's docstring-referenced `_TurboLoRALinear.weight`/`.bias` makes that walk
    succeed without error) and then `del projection` would throw away the LoRA delta
    entirely -- a silent, wrong-answer failure mode, not a crash, so it needs an
    explicit check rather than relying on some other line to fail first.

    Imports `_TurboLoRALinear` from `core.runner` lazily (not at this module's own
    import time) to avoid a circular import: `core/runner.py` imports
    `core.adaln_precompute` lazily too (inside `_ensure_transformer`/
    `enable_adaln_precompute`'s own call sites), specifically so the two modules never
    need to import each other at load time.
    """
    from core.runner import _TurboLoRALinear

    wrapped_blocks = [
        i for i, block in enumerate(transformer.transformer_blocks)
        if isinstance(block.adaln_proj.linear, _TurboLoRALinear)
    ]
    if wrapped_blocks:
        raise RuntimeError(
            f"AdaLN precompute cannot run: {len(wrapped_blocks)} block(s) "
            f"(e.g. block {wrapped_blocks[0]}) have a turbo-LoRA-wrapped "
            "adaln_proj.linear (_TurboLoRALinear). This only happens with the "
            "comfy-format turbo LoRA checkpoint (_TURBO_COMFY_REPOS, e.g. "
            "larryvrh/MiniMax-H3-Turbo-Lora) -- unlike the default diffusers-native "
            "format (lightx2v/Minimax-h3-Turbo), the comfy format's LoRA genuinely "
            "adapts adaln_proj, so a table baked from these wrapped modules would "
            "silently discard the LoRA delta once `del projection` runs (the walk "
            "over `projection.linear.parameters()` reads the wrapped base's "
            "parameters just fine -- this cannot be caught by a shape/attribute "
            "error, only by this explicit check). Use the default diffusers-native "
            "turbo LoRA (H3_TURBO_LORA_REPO=lightx2v/Minimax-h3-Turbo, or leave "
            "H3_TURBO_LORA_REPO/H3_TURBO_LORA_FILE unset) with H3_ADALN_PRECOMP=1."
        )


@torch.no_grad()
def precompute(transformer, row_timestep_plan: list) -> dict:
    """Build every block's modulation table for the whole trajectory and free the
    `adaln_proj` weights that produced it.

    `row_timestep_plan` is the pipeline's own `[(timestep, timestep_indices), ...]`
    (`block_state.row_timestep_plan`, built by `MiniMaxH3SetTimestepsStep` before the
    denoise loop starts -- see this module's own docstring). Only the timestep *values*
    are read here; the per-row indices stay with the caller/transformer forward
    unchanged (`adaln_indices` in `MiniMaxH3TransformerBlock.forward()` still indexes
    into this table's rows exactly as it always indexed into the uncached projection's
    output rows -- the row layout is unchanged, only how those rows are produced is).

    Coexistence with turbo (v2, see module docstring): runs AFTER
    `apply_instant_settings()`'s turbo wrap for this request (`enable_adaln_precompute()`
    arms the transformer at load time, but the actual table build is deferred to the
    denoise loop's first step -- see that function's docstring -- which is always after
    `generate()`/`generate_ref2va()` have already called `apply_instant_settings()`).
    For the default diffusers-native turbo LoRA this is inconsequential either way (that
    format never touches `adaln_proj`, verified in this module's docstring), but running
    after keeps the ordering correct in case a future non-adaln-touching LoRA format
    someday DOES wrap some other module `_timestep_embedding()` reads through (it does
    not today -- `time_proj`/`time_embedder` are never turbo-wrapped by either format).
    `_reject_turbo_wrapped_adaln()` (above) is the actual enforcement point for the one
    format that DOES conflict (comfy-format), not this ordering.

    Raises if this transformer already has a precompute table installed (call
    `is_precomputed()` first, or just check `H3_ADALN_PRECOMP`'s own `_wanted` flag --
    see `enable_adaln_precompute()`) -- a repeat call would try to read
    `block.adaln_proj.linear` off a block that no longer has a `.linear` submodule
    (`PrecomputedModulation` does not expose one), so failing loudly here beats a
    confusing `AttributeError` deep inside the loop.
    """
    if is_precomputed(transformer):
        raise RuntimeError("AdaLN precompute is already installed on this transformer.")
    _reject_turbo_wrapped_adaln(transformer)

    device = next(transformer.parameters()).device
    embeddings = [_timestep_embedding(transformer, ts.to(device)) for ts, _ in row_timestep_plan]

    cursor = _StepCursor(device)
    freed_bytes = 0
    table_bytes = 0

    # A step's table has one row per (timestep, modality) pair, and the number of
    # *distinct* timesteps a step carries is not constant: the video and audio
    # schedules run at different shifts (see `core/runner.py`'s `H3_VIDEO_SHIFT`/
    # `H3_AUDIO_SHIFT` module comment) and coincide on some steps, so a plan mixes
    # 1-row and 2-row steps (x `MINIMAX_H3_MODALITY_NUM` = 3 modalities each). Pad every
    # step out to the widest one. A block reads row `timestep_indices *
    # MINIMAX_H3_MODALITY_NUM + token_tags`, which never reaches past that step's own
    # rows (`timestep_indices` only ever indexes into that step's *own* distinct-
    # timestep count), so the padding is written and never read.
    from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM

    max_rows = max(int(timestep.numel()) for timestep, _ in row_timestep_plan) * MINIMAX_H3_MODALITY_NUM

    def padded(rows: torch.Tensor) -> torch.Tensor:
        if rows.shape[0] == max_rows:
            return rows
        return torch.cat([rows, rows.new_zeros(max_rows - rows.shape[0], rows.shape[1])])

    for block in transformer.transformer_blocks:
        projection = block.adaln_proj
        # One GEMM per step at the reference's own shape keeps the result bitwise
        # identical to the uncached path (a single batched GEMM over all steps would
        # differ in floating-point summation order/kernel selection for a
        # differently-shaped input -- not assumed safe, not used).
        table = torch.stack([padded(torch.cat(projection(temb), dim=-1)) for temb in embeddings])
        table_bytes += table.numel() * table.element_size()

        for parameter in projection.linear.parameters():
            freed_bytes += parameter.numel() * parameter.element_size()

        block.adaln_proj = PrecomputedModulation(table, cursor)
        del projection

    transformer._h3opt_adaln_cursor = cursor
    # Records which turbo state this table was built while observing, purely for
    # `MiniMaxH3Runner.status()` / `_adaln_precompute_status()` reporting (v2
    # coexistence) -- NOT used by any correctness-affecting logic. The table is valid
    # for both turbo=True and turbo=False requests regardless of this value (see this
    # module's own docstring: the default LoRA format never touches `adaln_proj`, so
    # this flag is purely informational, not a "which requests can use this table"
    # gate). Read via `_TurboLoRALinear` presence anywhere else in the transformer
    # (not restricted to `adaln_proj`, which `_reject_turbo_wrapped_adaln()` already
    # confirmed is unwrapped by this point) -- a cheap proxy for "was turbo active for
    # this request" without threading an extra argument through
    # `call_with_precomputed_adaln()`.
    from core.runner import _TurboLoRALinear

    transformer._h3opt_adaln_built_with_turbo = any(
        isinstance(module, _TurboLoRALinear) and module.enabled for module in transformer.modules()
    )
    torch.cuda.empty_cache()

    stats = {
        "steps": len(row_timestep_plan),
        "blocks": len(transformer.transformer_blocks),
        "table_gb": table_bytes / 1024**3,
        "freed_gb": freed_bytes / 1024**3,
        "built_with_turbo": transformer._h3opt_adaln_built_with_turbo,
    }
    logger.info(
        "[h3opt.adaln] cached %d blocks x %d steps: table %.2f GB, freed %.2f GB of weights "
        "(built_with_turbo=%s)",
        stats["blocks"], stats["steps"], stats["table_gb"], stats["freed_gb"], stats["built_with_turbo"],
    )
    return stats


def is_precomputed(transformer) -> bool:
    """Whether `precompute()` has already replaced this transformer's `adaln_proj`
    modules with `PrecomputedModulation`. Used both by `precompute()`'s own guard and by
    `core/runner.py` to decide whether a fresh load needs (re-)arming (see
    `enable_adaln_precompute()`'s docstring: every full free+reload of the transformer,
    which this project's default `H3_TE_QUANT=bnb-4bit` steady state does around every
    request's decode window, drops this attribute along with the rest of the module and
    needs to re-arm from scratch on the next load).
    """
    return getattr(transformer, "_h3opt_adaln_cursor", None) is not None


def built_with_turbo(transformer) -> bool | None:
    """Which turbo state the currently-installed precompute table was built while
    observing (see `precompute()`'s own comment on `_h3opt_adaln_built_with_turbo` --
    informational only, both turbo states are served correctly off the same table for
    the default LoRA format). `None` if no table is installed yet.
    """
    if not is_precomputed(transformer):
        return None
    return bool(getattr(transformer, "_h3opt_adaln_built_with_turbo", False))


def enable_adaln_precompute(transformer) -> None:
    """Arm the precompute on `transformer`: it fires on the first denoise step of the
    next request that drives this transformer instance through
    `MiniMaxH3LoopDenoiser`/`MiniMaxH3Ref2VALoopDenoiser`.

    The schedule is not known until the pipeline has built `row_timestep_plan` (inside
    the denoise loop's own step 0, not at load time -- see this module's own docstring),
    so the actual precompute work is hung off the first iteration of the loop denoiser
    rather than done here. The cursor is advanced from that same patched call, since it
    is the only place that knows the current step index `i`.

    Coexistence with turbo (v2): deferring to step 0 of the denoise loop also happens to
    guarantee precompute always runs AFTER `MiniMaxH3Runner.apply_instant_settings()`'s
    turbo wrap for this request -- `generate()`/`generate_ref2va()` (and their hires-fix
    counterparts) both call `apply_instant_settings()` once the transformer is confirmed
    resident and strictly before `denoise_step(pipe, state)` (the call that drives
    `MiniMaxH3LoopDenoiser.__call__` internally), for every code path (verified by
    reading all 4 call sites in `core/runner.py`). This ordering is not required for
    correctness against the default diffusers-native turbo LoRA (it never touches
    `adaln_proj`, see this module's own docstring, so precompute would give the same
    table whether it ran before or after that wrap), but it is what makes
    `_reject_turbo_wrapped_adaln()` (`precompute()`'s own guard) able to see a
    turbo-wrapped `adaln_proj.linear` at all when the comfy-format LoRA IS in play,
    instead of racing it.

    Idempotent per-process for the *class*-level monkeypatch (`MiniMaxH3LoopDenoiser
    .__call__` is only ever wrapped once, tracked via `_h3opt_patched` on the class
    itself -- verified against this project's pinned diffusers commit,
    `diffusers.modular_pipelines.minimax_h3.denoise.MiniMaxH3LoopDenoiser.__call__` and
    `MiniMaxH3Ref2VALoopDenoiser` share the exact same bound method by inheritance,
    neither subclass overrides `__call__` -- so wrapping the base class's `__call__`
    once covers both `transformer` (t2va/fl2va) and `transformer_ref` (ref2va) call
    sites without a second patch). NOT idempotent per-transformer-instance: call this
    again on the same (already-armed-or-precomputed) `transformer` object only after a
    fresh load has replaced it with a brand new instance (`core/runner.py`'s
    `_ensure_transformer()`/`_ensure_transformer_ref()` call this once per fresh load,
    right after the load succeeds and before the request reaches the denoise loop --
    every full free+reload gets a fresh instance with no `_h3opt_adaln_wanted`/
    `_h3opt_adaln_cursor` attributes yet, so re-arming here is always correct, never a
    double-arm of the same instance).
    """
    from diffusers.modular_pipelines.minimax_h3 import denoise as h3_denoise

    if not getattr(h3_denoise.MiniMaxH3LoopDenoiser, "_h3opt_patched", False):
        original_call = h3_denoise.MiniMaxH3LoopDenoiser.__call__

        @torch.no_grad()
        def call_with_precomputed_adaln(self, components, block_state, i: int, t):
            transformer_component = getattr(components, self.transformer_name)
            if getattr(transformer_component, "_h3opt_adaln_wanted", False) and i == 0:
                precompute(transformer_component, block_state.row_timestep_plan)
                transformer_component._h3opt_adaln_wanted = False
            cursor = getattr(transformer_component, "_h3opt_adaln_cursor", None)
            if cursor is not None:
                cursor.set(i)
            return original_call(self, components, block_state, i, t)

        h3_denoise.MiniMaxH3LoopDenoiser.__call__ = call_with_precomputed_adaln
        h3_denoise.MiniMaxH3LoopDenoiser._h3opt_patched = True
        logger.info("MiniMaxH3LoopDenoiser.__call__ patched for AdaLN precompute (covers t2va/fl2va and ref2va)")

    transformer._h3opt_adaln_wanted = True

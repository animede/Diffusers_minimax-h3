"""
Instant-apply per-request settings resolution + reload-group settings application.

Split out of core/runner.py to keep the two concerns (huge, already-dense VRAM
choreography file vs. small, mostly-validation glue) apart. Imports runner module
globals lazily (inside functions) rather than at module load time, since several of
them (H3_CACHE, H3_ATTN_BACKEND, H3_TURBO_LORA, and the whole reload-group set) are
mutated in place by `apply_reload_settings()` below -- a `from core.runner import
H3_CACHE` at this module's own import time would bind a stale local name that never
sees later reassignments (Python name binding, not a reference into the other
module's namespace). Every read in this file goes through `core.runner.<NAME>` (module
attribute access) instead, exactly like `runner.py`'s own request-time code already
does implicitly for its top-level globals -- see `apply_reload_settings()`'s docstring
for why this matters for the reload group specifically.

Two independent groups (see the task brief this file implements):

- Instant (per-request, no reload): FirstBlockCache on/off + threshold, attention
  backend, turbo LoRA on/off. Resolved per-request by `resolve_instant_settings()`
  (falls back to whatever the process-wide env-var defaults are when a request field is
  left unset) and applied by `MiniMaxH3Runner.apply_instant_settings()`.
- Reload (process-wide, needs every big model dropped and reloaded under the new
  config): transformer int8, TE quant, TE layer prune, lowvram mode, video VAE fp16.
  Changed via `apply_reload_settings()`, which validates the new combination using the
  exact same rules `core/runner.py`'s own import-time validation uses (kept in sync by
  hand -- see that function's docstring), then unloads everything and calls
  `runner.preload_all()` under the new config.
"""
from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger("minimax_h3.settings")

# Guards apply_reload_settings() so two concurrent /api/settings/apply calls cannot
# interleave their unload/reload sequences (the app-level generation lock already
# keeps this from overlapping with an actual generate() call -- see app.py -- but two
# settings-apply calls racing each other is a separate, narrower race this lock alone
# closes cheaply without complicating app.py's own lock semantics).
_reload_lock = threading.Lock()

INSTANT_CACHE_CHOICES = ("fbc", "none")
RELOAD_TRANSFORMER_QUANT_CHOICES = ("none", "int8")
RELOAD_TE_QUANT_CHOICES = ("none", "bnb-4bit")
RELOAD_LOWVRAM_CHOICES = ("0", "1", "group")


def resolve_instant_settings(
    cache: str | None,
    cache_threshold: float | None,
    attn: str | None,
    turbo: bool | None,
) -> dict:
    """Fill in any unset instant-apply field from the current process-wide default
    (whatever core.runner's own H3_CACHE/H3_CACHE_THRESHOLD/H3_ATTN_BACKEND/
    H3_TURBO_LORA env vars resolved to, possibly since changed by a prior
    apply_reload_settings() call for cache_threshold's H3_CACHE_THRESHOLD -- that one is
    actually instant-group by nature but is read from the same module attribute either
    way). A request that omits every one of these fields gets back exactly today's
    server-default behaviour, unchanged.

    Validates `cache`/`attn` against known choices and raises `ValueError` (mapped to
    400 by the caller) rather than silently passing through an unrecognised string to
    `enable_cache`/`set_attention_backend`, where the failure would be much harder to
    attribute to the request.
    """
    import core.runner as runner

    if cache is None:
        cache = runner.H3_CACHE if not runner.H3_TURBO_LORA else "none"
    cache = cache.strip().lower()
    if cache not in INSTANT_CACHE_CHOICES:
        raise ValueError(f"cache must be one of {INSTANT_CACHE_CHOICES}, got {cache!r}")

    if cache_threshold is None:
        cache_threshold = runner.H3_CACHE_THRESHOLD
    cache_threshold = float(cache_threshold)
    if not (0.0 <= cache_threshold <= 1.0):
        raise ValueError(f"cache_threshold must be between 0.0 and 1.0, got {cache_threshold}")

    if attn is None:
        attn = runner.H3_ATTN_BACKEND or "default"
    attn = attn.strip().lower()
    if attn in ("", "none"):
        attn = "default"

    if turbo is None:
        turbo = runner.H3_TURBO_LORA
    turbo = bool(turbo)

    # A handful of turbo steps (4-8) leaves no redundant-computation window for FBC's
    # residual-similarity skip to safely exploit, and caching on top of an
    # already-short trajectory risks compounding drift for no measured benefit -- same
    # reasoning `H3_TURBO_LORA`'s own module comment in core/runner.py gives for why
    # the old load-time-only wiring force-disabled FBC whenever turbo was on. Preserved
    # here for the instant-apply path: `cache` is force-downgraded to "none" whenever
    # `turbo` resolves True, regardless of what the request asked for, and the
    # `effective_cache`/`cache_forced_off_by_turbo` fields let callers report this
    # back to the client instead of silently ignoring their `cache` request.
    cache_forced_off_by_turbo = turbo and cache == "fbc"
    effective_cache = "none" if turbo else cache

    resolved = {
        "cache": cache,
        "cache_threshold": cache_threshold,
        "attn": attn,
        "turbo": turbo,
        "effective_cache": effective_cache,
        "cache_forced_off_by_turbo": cache_forced_off_by_turbo,
    }
    validate_instant_settings(resolved)
    return resolved


def validate_instant_settings(resolved: dict) -> None:
    """Reject combinations confirmed NOT to work together, mirroring the exact same
    rules `core/runner.py` enforces at import time for the equivalent env vars
    (H3_TURBO_LORA vs H3_LOWVRAM_ANY/H3_TRANSFORMER_BOTH_RESIDENT, see that module's own
    comment on the `H3_TURBO_LORA` block) -- except checked per-request here, since
    turbo is now an instant (not process-wide-fixed) setting and the *current* reload
    group config can be anything by the time a turbo=True request arrives.

    Actually verified (not just "unverified") by this task's own A/B run, 2026-08-06:
    turbo=1 + (lowvram=1 or lowvram=group or transformer_both_resident, i.e. any path
    where `H3_TRANSFORMER_QUANT=int8`) reproducibly fails with `NotImplementedError:
    Int8Tensor dispatch: attempting to run unimplemented operator/function:
    func=<OpOverload(op='aten.cat', overload='default')>` -- `apply_turbo_lora()`'s
    call to `attn.fuse_projections()` (`core/runner.py`) does `torch.cat([to_q.weight,
    to_k.weight, to_v.weight])` to build the fused `to_qkv` Linear the checkpoint's LoRA
    targets, and torchao's `Int8Tensor` (`H3_INT8_MODULES_TO_NOT_CONVERT` does not skip
    `to_q`/`to_k`/`to_v`, so they ARE quantized under `H3_TRANSFORMER_QUANT=int8`) has no
    registered `aten.cat` kernel (grepped `torchao/quantization/quantize_/workflows/
    int8/int8_tensor.py`'s `@implements(...)` list -- no `aten.cat` entry). This is not
    an offload-hook-ordering problem (group mode's "wrap LoRA before enable_group_
    offload()" fix that helped a sibling project, diffusers-server CLAUDE.md #44, does
    NOT apply here): the failure happens inside `fuse_projections()` itself, before any
    group offload hook is ever consulted, so reordering the two calls cannot help --
    the base weights themselves are the wrong tensor subclass for `torch.cat`. Confirmed
    identical for both `H3_LOWVRAM=1` and `H3_LOWVRAM=group` (both force
    `H3_TRANSFORMER_QUANT=int8`, same `modules_to_not_convert` recipe as plain
    `transformer_both_resident` int8 mode) -- one 500 response with the exact same
    `Int8Tensor`/`aten.cat` error text from each, request rejected cleanly with no VRAM
    leak or lasting corruption (confirmed by a follow-up plain, non-turbo generation
    succeeding right after on the same still-running server).
    """
    import core.runner as runner

    # 2026-08-08 更新: この拒否が必要なのは **comfy 形式 (融合QKV、Ostris 版) の LoRA の
    # ときだけ**になった。既定になった diffusers ネイティブ形式 (lightx2v 版) は
    # fuse_projections() を呼ばないため上記の torch.cat 問題を踏まず、int8/低VRAM でも
    # 適用できる (スパイク実測済み、README「lightx2v 版」節)。形式は
    # `runner.turbo_lora_expected_format()` (キャッシュ済みなら実ファイルのキー、
    # 未DLならリポジトリ名) で判定する。予備判定が外れた場合も
    # `_apply_turbo_lora_checkpoint()` の実ファイルチェックが 400 で拒否する。
    # group offload は LoRA の形式を問わず拒否: enable_group_offload() の
    # cpu_param_dict は有効化時点で固定され、後から追加される _TurboLoRALinear の
    # lora_a/lora_b バッファが offload サイクルから欠落するリスクがある (未実測の
    # まま解禁しない。runner.py の import 時ガードと同じ理由 -- レビュー指摘)。
    if resolved["turbo"] and runner.H3_LOWVRAM_GROUP:
        raise ValueError(
            "turbo=1 と lowvram=group は併用できません (group offload の cpu_param_dict "
            "は有効化時点で固定されるため、後から追加される LoRA バッファが offload "
            "サイクルから欠落するリスクがある -- 未検証)。lowvram=1 なら turbo を使えます。"
        )
    if (
        resolved["turbo"]
        and (runner.H3_LOWVRAM_ANY or runner.H3_TRANSFORMER_BOTH_RESIDENT)
        and runner.turbo_lora_expected_format() == "comfy"
    ):
        raise ValueError(
            "turbo=1 with the comfy-format (fused-QKV) LoRA is only supported against "
            "the default transformer path (transformer_quant=none, lowvram=0): "
            "apply_turbo_lora()'s fuse_projections() call does torch.cat() on the "
            "transformer's to_q/to_k/to_v weights, and those are torchao Int8Tensor "
            "under transformer_quant=int8 -- Int8Tensor has no aten.cat kernel, so this "
            "reproducibly raises NotImplementedError. Use the default diffusers-native "
            "LoRA (lightx2v/Minimax-h3-Turbo) or change the reload-group settings."
        )


def validate_instant_settings_for_upscale(resolved: dict, do_upscale: bool) -> None:
    """`upscale=1` (hires-fix) + turbo=1 works and is the recommended way to run
    hires-fix: verified by this task's own A/B run, 2026-08-06 (RTX PRO 6000 96GB,
    768->1536 hires-fix, seed=12345). 8-step turbo: 210.3s total (82.6s pass1+pass2
    denoise + 24.4s decode + ~103s model load/preload overhead already counted in
    total_elapsed_s), pass1=5/pass2=2 steps (`H3_HIRES_DENOISE=0.35` default split of
    the 7 scheduler timesteps an 8-step turbo run produces), peak VRAM 88.09GB -- vs
    the non-turbo 30-step baseline's 645s (README's own upscale=1 row). 16-step turbo:
    331.8s, pass1=10/pass2=5, peak VRAM 88.35GB, visibly sharper than 8-step. Both
    frame-sampled through the full clip (start/mid/end) with no checkerboard corruption
    or color-space breakage (the exact failure mode an earlier hires-fix iteration hit
    per `_upscale_block_state_2x`'s own docstring) and non-silent, non-clipped audio
    (RMS/peak matched the API's own reported values via an independent wav decode).
    `settings.py`'s no-FBC-with-turbo rule (see `resolve_instant_settings()`) already
    covers the "FBC bookkeeping under hires-fix is turbo-unaware" concern the previous
    version of this function's docstring raised: `effective_cache` is force-downgraded
    to "none" before this is ever reached, so `_fbc_last_step_was_skip()`'s calls in the
    hires-fix branch are harmless no-ops (try/except-wrapped, return 0) whenever turbo
    is on, exactly like the non-hires-fix path.
    """


def current_settings_snapshot() -> dict:
    """Everything `GET /api/settings` reports: current values for both groups, plus the
    choice lists the UI needs to render selects/checkboxes without hardcoding them
    twice (once server-side, once in static/index.html).
    """
    import core.runner as runner

    return {
        "instant": {
            "cache": runner.H3_CACHE if not runner.H3_TURBO_LORA else "none",
            "cache_threshold": runner.H3_CACHE_THRESHOLD,
            "attn": runner.H3_ATTN_BACKEND or "default",
            "turbo": runner.H3_TURBO_LORA,
        },
        "reload": {
            "transformer_quant": runner.H3_TRANSFORMER_QUANT,
            "te_quant": runner.TE_QUANT,
            "te_prune": runner.H3_TE_PRUNE,
            "lowvram": runner.H3_LOWVRAM_RAW,
            "video_vae_fp16": runner.H3_VIDEO_VAE_FP16,
            # 投影TE は現状 env 専用 (H3_TE_PROJ / H3_TE_PROJ_QUANT)。この API からは
            # 変更できない**読み取り専用**の状態として載せる -- 有効時は te_quant /
            # te_prune の変更が apply 側で 400 拒否されるため、UI がその理由を
            # 表示できるように状態自体は見えている必要がある。
            "te_proj": bool(runner.H3_TE_PROJ),
            "te_proj_quant": runner.H3_TE_PROJ_QUANT if runner.H3_TE_PROJ else None,
        },
        "choices": {
            "cache": list(INSTANT_CACHE_CHOICES),
            "transformer_quant": list(RELOAD_TRANSFORMER_QUANT_CHOICES),
            "te_quant": list(RELOAD_TE_QUANT_CHOICES),
            "lowvram": list(RELOAD_LOWVRAM_CHOICES),
        },
        "constraints": {
            # Mirrors the exact incompatibilities core/runner.py enforces at import
            # time / generate()-time, exposed so the UI can grey out turbo instead of
            # only finding out via a 400 after clicking generate. turbo+upscale is
            # verified to work together (see validate_instant_settings_for_upscale's
            # own docstring, 2026-08-06 A/B) and is no longer greyed out.
            # int8/lowvram との非互換は **comfy 形式 (融合QKV) の LoRA のときだけ**
            # (validate_instant_settings のコメント参照) なので動的に判定する --
            # 既定の diffusers ネイティブ LoRA (lightx2v) では False になり、UI は
            # 低VRAMモードでも turbo を有効にできる。
            "turbo_incompatible_with_lowvram": runner.turbo_lora_expected_format() == "comfy",
            # group offload は LoRA 形式を問わず不可 (validate_instant_settings 参照)
            "turbo_incompatible_with_lowvram_group": True,
            "turbo_incompatible_with_transformer_both_resident": runner.turbo_lora_expected_format() == "comfy",
            "turbo_incompatible_with_upscale": False,
            "transformer_both_resident": runner.H3_TRANSFORMER_BOTH_RESIDENT,
        },
        "reload_eta_s": RELOAD_ETA_S,
    }


# Rough reload time estimates (see the task brief's own numbers / this task's own
# measurements, logged in README once verified) -- purely informational, shown in the
# UI's "apply" button before the user commits to a ~30-100s wait. Keyed by what is
# actually changing costs the most: TE swap and int8 transformer quantize are the two
# slow steps; lowvram=group additionally pays a one-time CPU-resident pin cost.
RELOAD_ETA_S = {
    "transformer_quant": 36,
    "te_quant": 40,
    "te_prune": 5,
    "lowvram": 90,
    "video_vae_fp16": 5,
}


def estimate_reload_seconds(changed_fields: list[str]) -> int:
    if not changed_fields:
        return 0
    return max(RELOAD_ETA_S.get(f, 30) for f in changed_fields)


def apply_reload_settings(runner_instance, **fields) -> dict:
    """Validate + apply a new reload-group configuration, unloading every big model and
    reloading under the new settings. Never restarts the process (CLAUDE.md-style rule
    for this project, per the task brief: only `core.runner`'s existing unload/preload
    machinery is used -- no os.execv, no self-kill).

    `fields` may contain any of: transformer_quant, te_quant, te_prune, lowvram,
    video_vae_fp16. Fields not present keep their current value (partial updates are
    fine -- the caller, `POST /api/settings/apply`, only sends what the user actually
    changed via its diff-against-current-snapshot logic, but this function itself does
    not require that; passing every field with its unchanged value is equally valid).

    Validation mirrors `core/runner.py`'s own import-time rules for the equivalent env
    vars (H3_LOWVRAM/H3_TRANSFORMER_QUANT/H3_TE_QUANT interactions -- see that module's
    big comment block above H3_LOWVRAM_RAW) as closely as possible; kept as a hand
    port rather than a shared function because the source rules run at *import* time
    (raise-and-crash-the-process semantics: wrong env vars should never start the
    server) while this runs at *request* time (raise-and-report-400 semantics: a bad
    apply must never leave the process in a half-reconfigured state). The two therefore
    cannot trivially share one function body without a mode flag threaded through every
    branch -- duplicated instead, with this docstring as the pointer to keep both in
    sync if the source rules ever change.

    Returns the new settings snapshot (see `current_settings_snapshot()`) plus timing.
    """
    import core.runner as runner

    with _reload_lock:
        t0 = time.time()

        # ---- 投影TE (H3_TE_PROJ) 有効時は TE 系の変更を拒否する ----
        # runner 側の import 時ガード (H3_TE_PROJ と H3_TE_QUANT/H3_TE_PRUNE の排他) は
        # env の組み合わせしか見ていないため、この API から te_quant/te_prune を送ると
        # **ガードを素通り**する。しかも実際のロードは `_load_text_encoder` が投影経路へ
        # 先に分岐するので値は適用されず、スナップショットだけが「変わった」と報告する
        # -- 静かな嘘になる。適用されない変更は受け取った時点で 400 で返すのが正しい。
        # (2026-08-10 のレビューで発見。投影TE 自体をこの API から切り替えられるように
        # するのは別作業 -- 現状は env 専用。)
        if runner.H3_TE_PROJ:
            _te_fields = [k for k in ("te_quant", "te_prune") if k in fields]
            if _te_fields:
                raise ValueError(
                    "投影TE (H3_TE_PROJ) が有効なため、te_quant / te_prune はこの API から"
                    f"変更できません (指定されたフィールド: {', '.join(_te_fields)})。"
                    "TE の構成は環境変数 H3_TE_PROJ / H3_TE_PROJ_QUANT で管理されています。"
                    "32B TE に戻すにはサーバーを H3_TE_PROJ なしで再起動してください。"
                )

        # ---- resolve new values (unset fields keep the current one) ----
        new_transformer_quant = fields.get("transformer_quant", runner.H3_TRANSFORMER_QUANT)
        new_te_quant = fields.get("te_quant", runner.TE_QUANT)
        new_te_prune = fields.get("te_prune", runner.H3_TE_PRUNE)
        new_lowvram_raw = fields.get("lowvram", runner.H3_LOWVRAM_RAW)
        new_video_vae_fp16 = fields.get("video_vae_fp16", runner.H3_VIDEO_VAE_FP16)

        new_transformer_quant = str(new_transformer_quant).strip().lower()
        new_te_quant = str(new_te_quant).strip().lower()
        new_lowvram_raw = str(new_lowvram_raw).strip().lower()
        new_te_prune = bool(new_te_prune)
        new_video_vae_fp16 = bool(new_video_vae_fp16)

        if new_transformer_quant not in RELOAD_TRANSFORMER_QUANT_CHOICES:
            raise ValueError(
                f"transformer_quant must be one of {RELOAD_TRANSFORMER_QUANT_CHOICES}, "
                f"got {new_transformer_quant!r}"
            )
        if new_te_quant not in RELOAD_TE_QUANT_CHOICES:
            raise ValueError(f"te_quant must be one of {RELOAD_TE_QUANT_CHOICES}, got {new_te_quant!r}")
        if new_lowvram_raw not in RELOAD_LOWVRAM_CHOICES:
            raise ValueError(f"lowvram must be one of {RELOAD_LOWVRAM_CHOICES}, got {new_lowvram_raw!r}")

        new_lowvram = new_lowvram_raw == "1"
        new_lowvram_group = new_lowvram_raw == "group"
        new_lowvram_any = new_lowvram or new_lowvram_group

        # ---- validation, mirrors core/runner.py's import-time rules ----
        if new_lowvram_any:
            if new_transformer_quant == "none" and "transformer_quant" in fields:
                # Only reject an *explicit* none -- if the caller did not touch
                # transformer_quant at all, silently auto-upgrade it to int8 below,
                # exactly like core/runner.py's own H3_LOWVRAM_ANY block does for the
                # env-var case (distinguishing "left alone" from "explicitly chosen"
                # the same way, via `fields.get(...)` presence rather than a value
                # comparison against the default).
                raise ValueError(
                    f"lowvram={new_lowvram_raw!r} requires an int8 transformer (bf16's "
                    "66.3GB does not fit a 48GB-class card even alone) but "
                    "transformer_quant=none was explicitly requested. Drop "
                    "transformer_quant from the request (it will default to int8 under "
                    "lowvram) or set transformer_quant=int8 explicitly."
                )
            new_transformer_quant = "int8"
            if new_te_quant != "bnb-4bit":
                raise ValueError(
                    f"lowvram={new_lowvram_raw!r} requires te_quant=bnb-4bit, got "
                    f"te_quant={new_te_quant!r}. bf16 TE (~66.3GB) cannot coexist with "
                    "anything else on a 24-48GB-class card."
                )

        new_transformer_both_resident = new_transformer_quant == "int8" and not new_lowvram_any

        # turbo (instant-group) compatibility with the *new* reload config: reject the
        # apply outright if turbo is currently on and would become a confirmed-broken
        # combo under the new settings (see validate_instant_settings()'s own docstring
        # for the Int8Tensor/aten.cat root cause), rather than silently leaving a
        # now-invalid turbo=on state for the next request to trip over. Mirrors
        # validate_instant_settings()'s own rule, evaluated against the *proposed* new
        # state instead of the current one.
        if runner.H3_TURBO_LORA and (new_lowvram_any or new_transformer_both_resident):
            raise ValueError(
                "Cannot apply this reload configuration while the turbo LoRA default "
                "(H3_TURBO_LORA env var) is on: lowvram/transformer_both_resident are "
                "confirmed incompatible with turbo (Int8Tensor has no aten.cat kernel, "
                "see validate_instant_settings()'s docstring). Note this only refers to "
                "the startup env var default, not any single request's instant turbo=1 "
                "override (those are validated independently, per request, by "
                "validate_instant_settings())."
            )

        changed_fields = [
            name
            for name, old, new in (
                ("transformer_quant", runner.H3_TRANSFORMER_QUANT, new_transformer_quant),
                ("te_quant", runner.TE_QUANT, new_te_quant),
                ("te_prune", runner.H3_TE_PRUNE, new_te_prune),
                ("lowvram", runner.H3_LOWVRAM_RAW, new_lowvram_raw),
                ("video_vae_fp16", runner.H3_VIDEO_VAE_FP16, new_video_vae_fp16),
            )
            if old != new
        ]
        if not changed_fields:
            logger.info("apply_reload_settings: no changes, skipping unload/reload")
            return {**current_settings_snapshot(), "changed_fields": [], "reload_time_s": 0.0}

        logger.info(
            "apply_reload_settings: changed=%s -> transformer_quant=%s te_quant=%s "
            "te_prune=%s lowvram=%s video_vae_fp16=%s",
            changed_fields, new_transformer_quant, new_te_quant, new_te_prune,
            new_lowvram_raw, new_video_vae_fp16,
        )

        # ---- unload everything (models only -- pipe shells are cheap/idempotent to
        # rebuild and are left alone) ----
        runner_instance.unload_all()

        # ---- commit new globals (module attribute assignment -- every other function
        # in core/runner.py reads these as plain module-level names, which in Python
        # resolve through the module's __dict__ at call time, so this is visible to all
        # of them immediately, no re-import needed) ----
        runner.H3_TRANSFORMER_QUANT = new_transformer_quant
        runner.TE_QUANT = new_te_quant
        runner.H3_TE_PRUNE = new_te_prune
        runner.H3_LOWVRAM_RAW = new_lowvram_raw
        runner.H3_LOWVRAM = new_lowvram
        runner.H3_LOWVRAM_GROUP = new_lowvram_group
        runner.H3_LOWVRAM_ANY = new_lowvram_any
        runner.H3_TRANSFORMER_BOTH_RESIDENT = new_transformer_both_resident
        runner.H3_VIDEO_VAE_FP16 = new_video_vae_fp16

        # ---- reload steady-state residents under the new config ----
        runner_instance.preload_all()

        elapsed = time.time() - t0
        logger.info("apply_reload_settings: done in %.1fs. status=%s", elapsed, runner_instance.status())
        return {**current_settings_snapshot(), "changed_fields": changed_fields, "reload_time_s": round(elapsed, 1)}

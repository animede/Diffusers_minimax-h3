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
    """Reject combinations that are not verified to work together, mirroring the exact
    same rules `core/runner.py` enforces at import time for the equivalent env vars
    (H3_TURBO_LORA vs H3_LOWVRAM_ANY/H3_TRANSFORMER_BOTH_RESIDENT, see that module's own
    comment on the `H3_TURBO_LORA` block) -- except checked per-request here, since
    turbo is now an instant (not process-wide-fixed) setting and the *current* reload
    group config can be anything by the time a turbo=True request arrives.
    """
    import core.runner as runner

    if resolved["turbo"] and (runner.H3_LOWVRAM_ANY or runner.H3_TRANSFORMER_BOTH_RESIDENT):
        raise ValueError(
            "turbo=1 is only verified against the default transformer path "
            "(transformer_quant=none, lowvram=0). It has not been checked against "
            "lowvram (int8/group offload) or transformer_both_resident (int8 "
            "both-resident) -- refusing to silently combine an unverified "
            "quantize/offload order with a freshly-applied LoRA delta. Drop turbo or "
            "change the reload-group settings first."
        )


def validate_instant_settings_for_upscale(resolved: dict, do_upscale: bool) -> None:
    """`upscale=1` (hires-fix) has its own pre-existing incompatibility with turbo (see
    `generate()`'s own do_upscale/H3_TURBO_LORA check) -- re-checked here against the
    *resolved* (possibly request-overridden) turbo value, since turbo is no longer
    necessarily equal to the process-wide H3_TURBO_LORA default.
    """
    if do_upscale and resolved["turbo"]:
        raise ValueError("upscale=1 (hires-fix) is not supported with turbo=1.")


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
        },
        "choices": {
            "cache": list(INSTANT_CACHE_CHOICES),
            "transformer_quant": list(RELOAD_TRANSFORMER_QUANT_CHOICES),
            "te_quant": list(RELOAD_TE_QUANT_CHOICES),
            "lowvram": list(RELOAD_LOWVRAM_CHOICES),
        },
        "constraints": {
            # Mirrors the exact incompatibilities core/runner.py enforces at import
            # time / generate()-time, exposed so the UI can grey out turbo/upscale
            # instead of only finding out via a 400 after clicking generate.
            "turbo_incompatible_with_lowvram": True,
            "turbo_incompatible_with_transformer_both_resident": True,
            "turbo_incompatible_with_upscale": True,
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
        # apply outright if turbo is currently on and would become unverified-combo
        # under the new settings, rather than silently leaving a now-invalid turbo=on
        # state for the next request to trip over. Mirrors validate_instant_settings()'s
        # own rule, evaluated against the *proposed* new state instead of the current one.
        if runner.H3_TURBO_LORA and (new_lowvram_any or new_transformer_both_resident):
            raise ValueError(
                "Cannot apply this reload configuration while the turbo LoRA default "
                "(H3_TURBO_LORA env var) is on: lowvram/transformer_both_resident are "
                "not verified together with turbo. Note this only refers to the "
                "startup env var default, not any single request's instant turbo=1 "
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

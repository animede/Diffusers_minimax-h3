"""参照プレフィックスの KV キャッシュ共有プローブ (ref2i/ref2va バッチの ~65s/場面 削減の可否判定)。

問い: ref2va のテキストエンコード (参照ラベル+ビジョン 4104 トークン + プロンプト 14-22
トークン、参照が前置・プロンプトは末尾 verbatim) を「プレフィックス1回 (use_cache=True)
+ 場面ごとにプロンプトだけキャッシュ継続」に分割したとき、hidden_states[50] が
フル計算と一致するか。因果 LM なのでプレフィックス部分は数学的に同一のはず。継続部分は
カーネル経路の違いによる丸め差が出うるので、bitwise 一致と max abs diff の両方を測る。

呼び出しレシピは transformers 5.14.1 の modeling_qwen3_vl.py を読んだ調査結果に基づく:
- プレフィックス: DynamicCache(config=model.config) を渡して use_cache=True。
  この呼び出しが model.rope_deltas (インスタンス状態) を設定する
- 継続: input_ids は新トークンのみ / attention_mask=None (cache から kv_length を
  取らせる & compute_3d_position_ids の arange 分岐 = past_key_values_length からの
  連番 + rope_deltas 加算に乗せる) / mm_token_type_ids・pixel_values 等は全て None
  (image_grid_thw を渡すと rope_deltas が再計算・上書きされる罠がある) /
  output_hidden_states=True → hidden_states は新トークン分 [1, new_len, 5120] のみ
- 場面間のキャッシュ再利用: 継続がキャッシュに suffix を追記するので、各場面の後に
  DynamicCache.crop(prefix_len) で切り戻す (直列運用。deepcopy 不要)
- rope_deltas はプレフィックス呼び出し後に触らない (この分割運用ではプレフィックス→
  N 回の継続の間に他の TE 呼び出しが無いので、インスタンス状態のままで正しい)

実行 (サーバ停止中に):
  CUDA_VISIBLE_DEVICES=0 venv/bin/python scripts/probe_ref_prefix_cache.py
"""
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

OUT_DIR = BASE_DIR / "outputs" / "ab_ref_prefix_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# H3_PROBE_REF_IMAGE で差し替え可能 (eager 対照実験はフル系列だと注意行列の実体化が
# 48GB に収まらないため、縮小参照で系列を短くして行う — ロジックの正しさはサイズ非依存)。
import os as _os

REFERENCE_PNG = Path(_os.environ.get("H3_PROBE_REF_IMAGE", str(BASE_DIR / "outputs" / "t2i_1786110542_s1.png")))
PROMPTS = [
    "The girl in the red cloak from <Picture 1> walks slowly through the misty forest, lantern swinging gently, birds chirping and leaves rustling",
    "The girl in the red cloak from <Picture 1> kneels by a stream and fills a small flask with water, gentle water sounds, wind in the trees",
    "Scene with unusual words: the girl rides a zephyr-driven sled across quantum snow at midnight",
]


def main():
    import os

    import torch

    from core.runner import DEVICE, MiniMaxH3Reference, MiniMaxH3Runner

    # 切り分け用: H3_PROBE_ATTN=eager で TE の attention 実装を固定して比較する
    # (sdpa の「フル計算 vs KVキャッシュ継続」はカーネルのタイル分割が変わるため
    # 丸め差が出うる。eager で diff が消えれば分割ロジック自体は正しいと判定できる)。
    probe_attn = os.environ.get("H3_PROBE_ATTN", "").strip()
    n_prompts = int(os.environ.get("H3_PROBE_NPROMPTS", "0")) or len(PROMPTS)
    prompts = PROMPTS[:n_prompts]

    # 参照は setup で 2048px 短辺へ強制リサイズされる (packing_ref2va の
    # MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE)。eager 対照実験はフル系列 (4104) だと
    # 注意行列の実体化で OOM するため、プローブ内 patch で短辺を縮めて系列を短くする。
    short_edge = int(os.environ.get("H3_PROBE_REF_SHORT_EDGE", "0"))
    if short_edge:
        import diffusers.modular_pipelines.minimax_h3.packing_ref2va as _pr

        _pr.MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = short_edge
        logging.info("[patch] MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE -> %d", short_edge)

    runner = MiniMaxH3Runner(OUT_DIR)
    runner._ensure_pipe_shell()
    runner._ensure_pipe_ref_shell()
    logging.info("loading text encoder...")
    runner._load_text_encoder(None)
    runner._sync_shared_components_to_ref()
    pipe = runner._pipe_ref
    if probe_attn:
        pipe.text_encoder.set_attn_implementation(probe_attn)
        logging.info("TE attn implementation forced to %s", probe_attn)

    from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep
    from diffusers.modular_pipelines.minimax_h3.packing import MINIMAX_H3_TEXT_ENCODER_LAYER
    from diffusers.modular_pipelines.minimax_h3.packing_ref2va import build_ref2va_presentation
    from diffusers.modular_pipelines.modular_pipeline import PipelineState
    from transformers.cache_utils import DynamicCache

    ref = MiniMaxH3Reference(image=str(REFERENCE_PNG))
    state = PipelineState()
    for k, v in [("prompt", "x"), ("references", [ref]), ("height", 768), ("width", 768),
                 ("num_frames", 124), ("num_inference_steps", 30), ("generator", None),
                 ("output_type", "pt"), ("attention_kwargs", None), ("latents", None), ("audio_latents", None)]:
        state.set(k, v)
    _, state = MiniMaxH3Ref2VASetupStep()(pipe, state)
    prepared = state.get("prepared_references")

    tok, proc = pipe.tokenizer, pipe.processor
    model = pipe.text_encoder.model
    logging.info("TE attn implementation: %s", getattr(model.config, "_attn_implementation", "?"))

    merge = proc.image_processor.merge_size ** 2
    images = [r.image for r in prepared if r.kind == "image"]
    vision = proc.image_processor(images=images, return_tensors="pt")
    pixel_values, image_grid_thw = vision["pixel_values"], vision["image_grid_thw"]
    counts = [int(g.prod()) // merge for g in image_grid_thw]

    # --- ground truth: 既存のフル計算 (encode_prompt そのまま) ---
    full_h50 = {}
    full_times = []
    for p in prompts:
        t0 = time.time()
        with torch.no_grad():
            emb, tags = MiniMaxH3Ref2VATextEncoderStep.encode_prompt(
                pipe, p, prepared, device=DEVICE, dtype=torch.bfloat16
            )
        full_times.append(time.time() - t0)
        full_h50[p] = emb  # [1, L, 5120] bf16
    logging.info("full encode times: %s", [round(t, 2) for t in full_times])

    # --- 分割: プレフィックス1回 + 継続 ---
    prefix_ids, prefix_tags = build_ref2va_presentation(tok, "", prepared, counts, [])
    prefix_input = torch.tensor([prefix_ids], dtype=torch.long, device=DEVICE)
    mm_ids = torch.tensor(proc.create_mm_token_type_ids([prefix_ids]), dtype=torch.long, device=DEVICE)

    cache = DynamicCache(config=model.config)
    t0 = time.time()
    with torch.no_grad():
        prefix_out = model(
            input_ids=prefix_input,
            attention_mask=torch.ones_like(prefix_input),
            mm_token_type_ids=mm_ids,
            pixel_values=pixel_values.to(DEVICE, pipe.text_encoder.dtype),
            image_grid_thw=image_grid_thw.to(DEVICE),
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
        )
    prefix_time = time.time() - t0
    prefix_h50 = prefix_out.hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER].to(torch.bfloat16)
    prefix_len = cache.get_seq_length()
    logging.info("prefix encode: %.2fs, prefix_len=%d, rope_deltas=%s",
                 prefix_time, prefix_len, model.rope_deltas)

    results = []
    for p in prompts:
        suffix_ids = tok(p, add_special_tokens=False)["input_ids"]
        suffix_input = torch.tensor([suffix_ids], dtype=torch.long, device=DEVICE)
        t0 = time.time()
        with torch.no_grad():
            suffix_out = model(
                input_ids=suffix_input,
                attention_mask=None,
                mm_token_type_ids=None,
                pixel_values=None,
                image_grid_thw=None,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
            )
        suffix_time = time.time() - t0
        suffix_h50 = suffix_out.hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER].to(torch.bfloat16)
        cache.crop(prefix_len)  # 次の場面のためにプレフィックスだけ残す

        # ネガティブコントロール (H3_PROBE_NEG_CONTROL=1): わざと位置オフセットを 0 に
        # 壊した継続。正しい継続の diff (~1-3% rel RMS) が「正しい計算の丸めノイズ」で
        # あることを確定させるための対照 — 位置バグならこの水準まで diff が跳ねるはず。
        neg_rel = None
        if os.environ.get("H3_PROBE_NEG_CONTROL") == "1":
            new_len = suffix_input.shape[1]
            bad_pos = torch.arange(0, new_len, device=DEVICE).view(1, 1, -1).expand(3, 1, -1).clone()
            bad_pos = bad_pos + model.rope_deltas.to(DEVICE).view(1, -1, 1)
            with torch.no_grad():
                bad_out = model(
                    input_ids=suffix_input,
                    attention_mask=None,
                    position_ids=bad_pos,
                    mm_token_type_ids=None,
                    pixel_values=None,
                    image_grid_thw=None,
                    past_key_values=cache,
                    use_cache=True,
                    output_hidden_states=True,
                )
            bad_h50 = bad_out.hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER].to(torch.bfloat16)
            cache.crop(prefix_len)
            fullf = full_h50[p][:, prefix_len:].float()
            neg_rel = float((fullf - bad_h50.float()).pow(2).mean().sqrt() / (fullf.pow(2).mean().sqrt() + 1e-9))

        split_h50 = torch.cat([prefix_h50, suffix_h50], dim=1)
        full = full_h50[p]
        assert split_h50.shape == full.shape, (split_h50.shape, full.shape)

        pre_f, pre_s = full[:, :prefix_len].float(), split_h50[:, :prefix_len].float()
        suf_f, suf_s = full[:, prefix_len:].float(), split_h50[:, prefix_len:].float()
        rec = {
            "prompt": p[:50],
            "suffix_len": len(suffix_ids),
            "suffix_time_s": round(suffix_time, 3),
            "prefix_bitwise_equal": bool(torch.equal(full[:, :prefix_len], split_h50[:, :prefix_len])),
            "suffix_bitwise_equal": bool(torch.equal(full[:, prefix_len:], split_h50[:, prefix_len:])),
            "prefix_max_abs_diff": float((pre_f - pre_s).abs().max()),
            "suffix_max_abs_diff": float((suf_f - suf_s).abs().max()),
            "suffix_rms": float(suf_f.pow(2).mean().sqrt()),
            "suffix_rel_rms_diff": float((suf_f - suf_s).pow(2).mean().sqrt() / (suf_f.pow(2).mean().sqrt() + 1e-9)),
            "neg_control_rel_rms_diff": neg_rel,
        }
        results.append(rec)
        logging.info("%s", json.dumps(rec, ensure_ascii=False))

    summary = {
        "probe_attn": probe_attn or "default",
        "prefix_len": prefix_len,
        "prefix_time_s": round(prefix_time, 2),
        "full_times_s": [round(t, 2) for t in full_times],
        "attn_impl": getattr(model.config, "_attn_implementation", "?"),
        "results": results,
    }
    (OUT_DIR / (f"probe_results_{probe_attn}.json" if probe_attn else "probe_results.json")).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logging.info("=== wrote %s ===", OUT_DIR / "probe_results.json")


if __name__ == "__main__":
    main()

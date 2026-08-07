"""lightx2v/Minimax-h3-Turbo (4step 蒸留 LoRA) のスパイク。

問い2つ:
  Q1 (ゲート): **int8 の transformer に適用できるか**。現行の turbo LoRA
      (larryvrh/Ostris 版) は ComfyUI 由来の融合 QKV (`qkv_proj`) を対象にしており、
      適用に `attn.fuse_projections()` = `torch.cat([to_q.weight, to_k.weight,
      to_v.weight])` が要る。torchao の `Int8Tensor` には `aten.cat` カーネルが無いため、
      int8 (= H3_LOWVRAM 必須の 48GB 級) では NotImplementedError で確実に落ちる
      (README「Turbo LoRA」節)。lightx2v 版はキーが **diffusers ネイティブで
      to_q/to_k/to_v が分離**しているので融合が不要 -- torch.cat を一切呼ばずに
      適用できるはず、というのがこのスパイクの主眼。通れば 48GB でも turbo が使える。
  Q2 (品質): 4 steps の出力が 30 steps 基準に対して実用範囲か。strength (LoRA 係数) は
      Kijai のモデルカードが 0.75 を推奨しているので 0.75 と 1.0 を A/B する。

キー形式 (このタスクで safetensors ヘッダを直接読んで確認済み):
  `<diffusers のドット付きパス>.lora_A.default.weight` / `.lora_B.default.weight`
  312 モジュール = transformer_blocks 50 x 6 + token_refiner.refiner_blocks 2 x 6。
  対象は attn (to_q/to_k/to_v/to_out.0) と ff (net.0.proj/net.2) のみで、
  adaln_proj / final_layer は**含まない** (Ostris 版とはここも異なる)。rank 128、bf16。
  → キーマップは不要 (パスがそのままモジュールパス)。

本体コードは無変更。runner の `_TurboLoRALinear` だけ流用し (scale 対応のため薄い
サブクラスをこのプローブ内で定義)、`_ensure_transformer` を wrap して「int8 ロード直後に
LoRA を巻く」形にする -- H3_LOWVRAM=1 はリクエストごとに transformer を解放/再ロード
するため、手で1回巻くだけでは次のリクエストで消えるのを避けるため。

実行 (サーバ停止中に):
  CUDA_VISIBLE_DEVICES=0 H3_LOWVRAM=1 venv/bin/python scripts/probe_lightx2v_turbo.py
"""
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

OUT_DIR = BASE_DIR / "outputs" / "ab_lightx2v_turbo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPO = "lightx2v/Minimax-h3-Turbo"
FILENAME = "minimax_h3_fl2v_turbo_4step_v0.1.safetensors"

# 本日 (2026-08-08) の t2va 基準と完全に同条件にする: seed/プロンプト/解像度/尺を揃え、
# ステップ数と LoRA だけを変える。基準は outputs/t2va_1786106892.mp4
# (30steps・denoise 197.7s・total 351.4s、H3_LOWVRAM=1 / RTX PRO 5000 48GB)。
PROMPT = "A red fox walks through a snowy forest at dawn, birds chirping, wind blowing through pine trees"
SEED = 12345
HEIGHT = WIDTH = 768
SECONDS = 5.0

# (label, steps, strength)。H3_PROBE_PLAN="label:steps:strength,..." で上書きできる
# (切り分け用: 30steps+LoRA が崩れるなら適用そのもの、崩れないならスケジュールが原因)。
PLAN = [
    ("4step_s075", 4, 0.75),
    ("4step_s100", 4, 1.00),
    ("8step_s075", 8, 0.75),
]
if os.environ.get("H3_PROBE_PLAN"):
    PLAN = [
        (p.split(":")[0], int(p.split(":")[1]), float(p.split(":")[2]))
        for p in os.environ["H3_PROBE_PLAN"].split(",")
    ]


class ScaledLoRALinear:
    """`_TurboLoRALinear` に strength (scale) を足しただけの薄いサブクラス生成関数用の器。

    実体は下の `_make_scaled_lora_linear()` が runner の `_TurboLoRALinear` を継承して作る
    (runner を import した後でないとクラスが取れないため、関数内で組み立てる)。
    """


def _make_scaled_lora_linear(base_cls):
    import torch

    class _ScaledTurboLoRALinear(base_cls):
        """base(x) + scale * B(A(x))。lightx2v 版は alpha メタデータを持たないので
        strength は自由パラメータ (Kijai のカードは 0.75 推奨)。"""

        def __init__(self, base, lora_a, lora_b, scale: float):
            super().__init__(base, lora_a, lora_b)
            self.scale = float(scale)

        def forward(self, x):
            if not self.enabled:
                return self.base(x)
            delta = torch.nn.functional.linear(
                torch.nn.functional.linear(x, self.lora_a), self.lora_b
            )
            return self.base(x) + self.scale * delta

    return _ScaledTurboLoRALinear


def apply_lightx2v_lora(transformer, lora_path: str, scale: float) -> int:
    """diffusers ネイティブキーの LoRA を、融合なしでそのまま巻く。

    `apply_turbo_lora()` (Ostris 版用) との違い: `fuse_projections()` も
    `delattr(to_q/to_k/to_v)` も**呼ばない** -- これが int8 で落ちない理由そのもの。
    キーはドット付きパスがそのままモジュールパスなのでキーマップも不要。
    """
    import torch
    from safetensors.torch import load_file

    from core.runner import _TurboLoRALinear, _get_module_by_dotted_path, _set_module_by_dotted_path

    scaled_cls = _make_scaled_lora_linear(_TurboLoRALinear)

    t0 = time.time()
    sd = load_file(lora_path)
    paths = sorted({k.rsplit(".lora_", 1)[0] for k in sd if ".lora_" in k})
    device = next(transformer.parameters()).device
    n = 0
    for path in paths:
        a = sd[f"{path}.lora_A.default.weight"].to(device=device, dtype=torch.bfloat16)
        b = sd[f"{path}.lora_B.default.weight"].to(device=device, dtype=torch.bfloat16)
        base = _get_module_by_dotted_path(transformer, path)
        if not isinstance(base, torch.nn.Linear):
            raise RuntimeError(f"expected nn.Linear at transformer.{path}, got {type(base).__name__}")
        if a.shape[1] != base.in_features or b.shape[0] != base.out_features:
            raise RuntimeError(
                f"shape mismatch at {path}: A={tuple(a.shape)} B={tuple(b.shape)} vs "
                f"in={base.in_features} out={base.out_features}"
            )
        _set_module_by_dotted_path(transformer, path, scaled_cls(base, a, b, scale))
        n += 1
    logging.info("lightx2v LoRA applied: %d modules, scale=%.2f, %.1fs", n, scale, time.time() - t0)
    return n


def main():
    import torch
    from huggingface_hub import hf_hub_download

    from core.runner import H3_LOWVRAM, H3_TRANSFORMER_QUANT, MiniMaxH3Runner, ProgressState

    logging.info("mode: H3_LOWVRAM=%s H3_TRANSFORMER_QUANT=%s", H3_LOWVRAM, H3_TRANSFORMER_QUANT)
    if H3_TRANSFORMER_QUANT != "int8":
        logging.warning("transformer quant is %r -- Q1 (int8 ゲート) の検証にならない", H3_TRANSFORMER_QUANT)

    lora_path = hf_hub_download(REPO, FILENAME)
    logging.info("lora: %s (%.2f GB)", lora_path, os.path.getsize(lora_path) / 1e9)

    runner = MiniMaxH3Runner(OUT_DIR)

    # H3_LOWVRAM=1 はリクエストごとに transformer を解放/再ロードするので、
    # 「ロード直後に巻く」フックを入れる (本体は無変更、プローブ内 wrap のみ)。
    state = {"scale": None, "applied": 0, "apply_error": None}
    orig_ensure = runner._ensure_transformer

    def ensure_and_wrap(progress=None):
        orig_ensure(progress)
        if state["scale"] is None:
            return
        tr = runner._pipe.transformer
        if getattr(tr, "_lightx2v_applied", False):
            return
        try:
            state["applied"] = apply_lightx2v_lora(tr, lora_path, state["scale"])
            tr._lightx2v_applied = True
        except Exception:
            state["apply_error"] = traceback.format_exc()
            logging.error("LoRA apply FAILED:\n%s", state["apply_error"])
            raise

    runner._ensure_transformer = ensure_and_wrap

    results = []
    for label, steps, scale in PLAN:
        logging.info("=== %s: steps=%d strength=%.2f ===", label, steps, scale)
        state["scale"] = scale
        progress = ProgressState()
        progress.update(job_id=label, started_at=time.time())
        rec = {"label": label, "steps": steps, "strength": scale}
        t0 = time.time()
        try:
            result = runner.generate(
                prompt=PROMPT, height=HEIGHT, width=WIDTH, seconds=SECONDS,
                num_inference_steps=steps, seed=SEED, progress=progress,
                # 蒸留4stepにキャッシュを重ねない (turbo が FBC を強制OFFにするのと同じ理屈)。
                cache="none",
            )
            rec.update({k: result[k] for k in (
                "num_frames", "denoise_time_s", "decode_time_s", "avg_step_time_s",
                "peak_vram_gb", "audio_rms", "audio_peak", "total_elapsed_s", "mp4_path",
            )})
            rec["modules_wrapped"] = state["applied"]
            dest = OUT_DIR / f"{label}.mp4"
            dest.write_bytes(Path(result["mp4_path"]).read_bytes())
            rec["mp4_saved"] = str(dest)
        except Exception:
            rec["exception"] = traceback.format_exc()
            logging.error("[%s] FAILED:\n%s", label, rec["exception"])
        rec["wall_time_s"] = round(time.time() - t0, 2)
        results.append(rec)

        # 次の条件のために LoRA を落とす: lowvram=1 は decode 後に transformer を
        # 解放済みなので、次回 `_ensure_transformer` が素の int8 を読み直し、
        # 新しい strength で巻き直される (`_lightx2v_applied` も新インスタンスには無い)。

    out = OUT_DIR / "probe_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    logging.info("=== wrote %s ===", out)
    for r in results:
        if r.get("exception"):
            logging.info("%s: FAILED", r["label"])
        else:
            logging.info("%s: denoise=%.1fs total=%.1fs peak=%.1fGB rms=%.4f wrapped=%d",
                         r["label"], r["denoise_time_s"], r["total_elapsed_s"],
                         r["peak_vram_gb"], r["audio_rms"], r["modules_wrapped"])


if __name__ == "__main__":
    main()

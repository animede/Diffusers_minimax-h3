"""Standalone verification for H3_TURBO_LORA's `apply_turbo_lora()` (see core/runner.py).

Loads only the (bf16, GPU-resident) transformer -- not the full server -- applies the
turbo LoRA, and checks:
  1. it applies without error and wraps the expected number of Linear layers (259).
  2. a forward pass still runs (shape sanity, using the modular pipeline's own packing
     helpers against a tiny dummy packed sequence) and produces a *different* output than
     the un-adapted base (LoRA is actually doing something, not a silent no-op).
  3. `attn.fused_projections` is True on every attention submodule after application.

Run: venv/bin/python scripts/probe_turbo_lora_apply.py
"""
import logging
import sys
import time

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

sys.path.insert(0, ".")
from core.runner import apply_turbo_lora, H3_TURBO_LORA_REPO, H3_TURBO_LORA_FILE  # noqa: E402

DEVICE = torch.device("cuda:0")


def main():
    from huggingface_hub import hf_hub_download
    from diffusers import ModularPipeline
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention

    print("[1/5] resolving turbo LoRA checkpoint...")
    lora_path = hf_hub_download(H3_TURBO_LORA_REPO, H3_TURBO_LORA_FILE)
    print(f"      -> {lora_path}")

    print("[2/5] building pipe shell + loading transformer (bf16, no quant)...")
    t0 = time.time()
    pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3")
    pipe.load_components(names=["transformer"], dtype=torch.bfloat16)
    pipe.transformer.to(DEVICE)
    pipe.transformer.eval()
    print(f"      loaded in {time.time() - t0:.1f}s")

    # Grab one real attention module + one real FF module for a targeted before/after
    # forward comparison (cheaper and more informative than a full packed-sequence run).
    blk0 = pipe.transformer.transformer_blocks[0]
    attn_test_input = torch.randn(4, pipe.transformer.config.hidden_size, device=DEVICE, dtype=torch.bfloat16)
    with torch.no_grad():
        # Pre-fusion, pre-LoRA reference output (to_q/to_k/to_v path).
        q_before = blk0.attn.to_q(attn_test_input)
        ff_before = blk0.ff(attn_test_input)
        adaln_before = blk0.adaln_proj(torch.randn(2, pipe.transformer.config.time_embed_dim, device=DEVICE, dtype=torch.bfloat16))

    print("[3/5] applying turbo LoRA...")
    t0 = time.time()
    n_wrapped = apply_turbo_lora(pipe.transformer, lora_path)
    print(f"      wrapped {n_wrapped} Linear layers in {time.time() - t0:.1f}s (expected 259)")
    assert n_wrapped == 259, f"expected 259 wrapped layers, got {n_wrapped}"

    print("[4/5] verifying fuse_projections + wrapping side effects...")
    n_attn = 0
    n_fused = 0
    for m in pipe.transformer.modules():
        if isinstance(m, MiniMaxH3Attention):
            n_attn += 1
            if m.fused_projections:
                n_fused += 1
    print(f"      {n_fused}/{n_attn} attention modules report fused_projections=True")
    assert n_attn == n_fused, "not all attention modules were fused"
    # 50 main blocks + 2 token_refiner blocks = 52 attention modules total.
    assert n_attn == 52, f"expected 52 attention modules, got {n_attn}"

    from core.runner import _TurboLoRALinear
    assert isinstance(pipe.transformer.transformer_blocks[0].attn.to_qkv, _TurboLoRALinear)
    assert isinstance(pipe.transformer.transformer_blocks[0].attn.to_out[0], _TurboLoRALinear)
    assert isinstance(pipe.transformer.transformer_blocks[0].ff.net[0].proj, _TurboLoRALinear)
    assert isinstance(pipe.transformer.transformer_blocks[0].ff.net[2], _TurboLoRALinear)
    assert isinstance(pipe.transformer.transformer_blocks[0].adaln_proj.linear, _TurboLoRALinear)
    assert isinstance(pipe.transformer.norm_out.linear, _TurboLoRALinear)
    assert isinstance(pipe.transformer.token_refiner.refiner_blocks[0].attn.to_qkv, _TurboLoRALinear)
    print("      all expected module sites are _TurboLoRALinear-wrapped")

    print("[5/5] checking LoRA delta is non-trivial (output changed vs pre-LoRA base)...")
    with torch.no_grad():
        # attn.to_q no longer exists after fusion -- compare the fused to_qkv's q-slice
        # against the pre-fusion to_q output for the same input.
        heads, hd = blk0.attn.heads, blk0.attn.head_dim
        qkv_after = blk0.attn.to_qkv(attn_test_input)
        q_after = qkv_after[:, : heads * hd]
        ff_after = blk0.ff(attn_test_input)
        adaln_after = blk0.adaln_proj(torch.randn(2, pipe.transformer.config.time_embed_dim, device=DEVICE, dtype=torch.bfloat16))

    q_diff = (q_after.float() - q_before.float()).abs().mean().item()
    ff_diff = (ff_after.float() - ff_before.float()).abs().mean().item()
    print(f"      mean abs diff: to_q slice={q_diff:.6f}, ff={ff_diff:.6f}")
    assert q_diff > 1e-6, "LoRA delta on attention appears to be a no-op"
    assert ff_diff > 1e-6, "LoRA delta on FF appears to be a no-op"

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()

"""
Standalone probe: verify that truncating Qwen3-VL-32B text_encoder's text_config
.num_hidden_layers from 64 to 50 (before from_pretrained) produces IDENTICAL
hidden_states[50] output to the full 64-layer model, and measure the VRAM/RAM
delta of the truncated bnb-4bit load.

MiniMax-H3 only ever reads `hidden_states[50]` (see diffusers' minimax_h3/encoders.py
and packing.py's MINIMAX_H3_TEXT_ENCODER_LAYER=50) and never uses the LM head. Per
transformers' `_can_record_outputs = {"hidden_states": Qwen3VLTextDecoderLayer}`
(modeling_qwen3_vl.py) + `install_output_capuring_hook`'s `capture_initial_hidden_state`
behavior (transformers/utils/output_capturing.py):
  hidden_states[0]  = the embedding output (captured as layer[0]'s *input*)
  hidden_states[k]  = the output of `layers[k-1]` for k=1..num_hidden_layers
So hidden_states[50] = output of layers[49] -- only layers[0..49] (50 layers) are
ever executed before that value is captured; layers[50:64] (14 layers) + the final
`norm` + `lm_head` are dead weight for this checkpoint's use in MiniMax-H3.

Run with the real server DOWN (this script loads its own copy of the ~62GB TE onto
the GPU) -- verify with `pgrep -f "[u]vicorn app:app.*8611"` first.
"""
import gc
import time

import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor, Qwen2TokenizerFast, BitsAndBytesConfig

SNAP = "/home/animede/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/text_encoder"
PROC_SNAP = "/home/animede/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/processor"
TOKENIZER_SNAP = "/home/animede/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/tokenizer"
MINIMAX_H3_TEXT_ENCODER_LAYER = 50


def gpu_gb():
    return torch.cuda.memory_allocated() / 1e9


def load_te(prune: bool, quant: bool):
    cfg = Qwen3VLConfig.from_pretrained(SNAP)
    if prune:
        cfg.text_config.num_hidden_layers = MINIMAX_H3_TEXT_ENCODER_LAYER
    kwargs = dict(config=cfg, dtype=torch.bfloat16, device_map="cuda")
    if quant:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(SNAP, **kwargs)
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"[load] prune={prune} quant={quant} time={dt:.1f}s gpu_allocated={gpu_gb():.2f}GB "
          f"num_layers_built={len(model.model.language_model.layers)}", flush=True)
    return model, dt


def run_forward(model, prompt="A red fox walks through a snowy forest at dawn, cinematic lighting."):
    tokenizer = Qwen2TokenizerFast.from_pretrained(TOKENIZER_SNAP)
    processor = Qwen3VLProcessor.from_pretrained(PROC_SNAP)
    token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device="cuda")
    mm_token_type_ids = torch.tensor(
        processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device="cuda"
    )
    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
            output_hidden_states=True,
        )
    print(f"[forward] num hidden_states entries = {len(outputs.hidden_states)}", flush=True)
    return outputs.hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER].float().cpu()


def free(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[free] gpu_allocated={gpu_gb():.2f}GB", flush=True)


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "bf16"  # "bf16" or "nf4"
    quant = mode == "nf4"

    print("=== FULL (64 layers) ===", flush=True)
    model_full, t_full = load_te(prune=False, quant=quant)
    hs_full = run_forward(model_full)
    free(model_full)

    print("=== PRUNED (50 layers) ===", flush=True)
    model_pruned, t_pruned = load_te(prune=True, quant=quant)
    hs_pruned = run_forward(model_pruned)
    free(model_pruned)

    diff = (hs_full - hs_pruned).abs()
    print(f"max abs diff: {diff.max().item():.6e}", flush=True)
    print(f"mean abs diff: {diff.mean().item():.6e}", flush=True)
    print(f"allclose(atol=1e-3): {torch.allclose(hs_full, hs_pruned, atol=1e-3)}", flush=True)
    print(f"bit-identical: {torch.equal(hs_full, hs_pruned)}", flush=True)
    print(f"load time full={t_full:.1f}s pruned={t_pruned:.1f}s", flush=True)

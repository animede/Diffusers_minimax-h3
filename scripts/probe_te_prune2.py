"""Isolated single-model probe (one config per process invocation) to avoid any
state leakage between two loads in the same process (see probe_te_prune.py's
suspicious VRAM growth on the second load)."""
import sys
import time

import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor, Qwen2TokenizerFast, BitsAndBytesConfig

SNAP = "/home/animede/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/text_encoder"
PROC_SNAP = "/home/animede/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/processor"
TOKENIZER_SNAP = "/home/animede/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/tokenizer"
MINIMAX_H3_TEXT_ENCODER_LAYER = 50


def gpu_gb():
    return torch.cuda.memory_allocated() / 1e9


def main():
    prune = sys.argv[1] == "prune"
    quant = sys.argv[2] == "nf4"
    out_path = sys.argv[3]

    cfg = Qwen3VLConfig.from_pretrained(SNAP)
    if prune:
        # +1 vs the read index: `Qwen3VLTextModel.forward`'s `@capture_outputs(tie_last_hidden_states=True)`
        # overwrites the LAST hidden_states entry with the post-final-norm `last_hidden_state`.
        # Truncating to exactly 50 layers makes index 50 the last entry, so it silently becomes
        # post-norm garbage instead of raw layer-49 output. 51 layers keeps index 50 mid-stack.
        cfg.text_config.num_hidden_layers = MINIMAX_H3_TEXT_ENCODER_LAYER + 1
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

    tokenizer = Qwen2TokenizerFast.from_pretrained(TOKENIZER_SNAP)
    processor = Qwen3VLProcessor.from_pretrained(PROC_SNAP)
    prompt = "A red fox walks through a snowy forest at dawn, cinematic lighting."
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
    print(f"[forward] num hidden_states entries = {len(outputs.hidden_states)} gpu_allocated={gpu_gb():.2f}GB", flush=True)
    hs = outputs.hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER].float().cpu()
    torch.save(hs, out_path)
    print(f"saved to {out_path}, shape={hs.shape}", flush=True)


if __name__ == "__main__":
    main()

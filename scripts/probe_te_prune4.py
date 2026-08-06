"""Debug: save ALL hidden_states (not just [50]) to find where full vs pruned diverge."""
import sys
import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor, Qwen2TokenizerFast

SNAP = "/home/animede/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/text_encoder"
PROC_SNAP = "/home/animede/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/processor"
TOKENIZER_SNAP = "/home/animede/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/tokenizer"
MINIMAX_H3_TEXT_ENCODER_LAYER = 50


def main():
    prune = sys.argv[1] == "prune"
    out_path = sys.argv[2]

    cfg = Qwen3VLConfig.from_pretrained(SNAP)
    if prune:
        cfg.text_config.num_hidden_layers = MINIMAX_H3_TEXT_ENCODER_LAYER
    model = Qwen3VLForConditionalGeneration.from_pretrained(SNAP, config=cfg, dtype=torch.bfloat16, device_map="cuda")

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
    all_hs = [h.float().cpu() for h in outputs.hidden_states]
    print(f"prune={prune} num_hidden_states={len(all_hs)}", flush=True)
    torch.save(all_hs, out_path)


if __name__ == "__main__":
    main()

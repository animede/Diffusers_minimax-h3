"""Debug: check whether embed_tokens weight, and layer 0 weights, are identical
between full and pruned config loads (should be byte-identical -- same checkpoint,
same shard, only the *number of layers built* differs)."""
import sys
import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

SNAP = "/home/animede/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/text_encoder"
MINIMAX_H3_TEXT_ENCODER_LAYER = 50


def main():
    prune = sys.argv[1] == "prune"
    out_prefix = sys.argv[2]

    cfg = Qwen3VLConfig.from_pretrained(SNAP)
    if prune:
        cfg.text_config.num_hidden_layers = MINIMAX_H3_TEXT_ENCODER_LAYER
    model = Qwen3VLForConditionalGeneration.from_pretrained(SNAP, config=cfg, dtype=torch.bfloat16, device_map="cuda")

    lm = model.model.language_model
    embed = lm.embed_tokens.weight.detach().float().cpu()
    layer0_q = lm.layers[0].self_attn.q_proj.weight.detach().float().cpu()
    layer49_q = lm.layers[49].self_attn.q_proj.weight.detach().float().cpu()
    print(f"prune={prune} embed sum={embed.sum().item():.6f} layer0_q sum={layer0_q.sum().item():.6f} "
          f"layer49_q sum={layer49_q.sum().item():.6f}", flush=True)
    torch.save({"embed": embed, "layer0_q": layer0_q, "layer49_q": layer49_q}, out_prefix + ".pt")


if __name__ == "__main__":
    main()

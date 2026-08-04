import sys, time
from huggingface_hub import snapshot_download

t0 = time.time()
patterns = [
    "modular_model_index.json",
    "text_encoder/*",
    "transformer/*",
    "vae/*",
    "audio_vae/*",
    "scheduler/*",
    "audio_scheduler/*",
    "tokenizer/*",
    "processor/*",
]
print(f"[{time.time()-t0:.0f}s] starting download, patterns={patterns}", flush=True)
path = snapshot_download(
    "MiniMaxAI/MiniMax-H3",
    allow_patterns=patterns,
)
print(f"[{time.time()-t0:.0f}s] DONE -> {path}", flush=True)

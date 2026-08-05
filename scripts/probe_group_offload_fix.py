"""
Confirms the fix for the `use_stream=True + low_cpu_mem_usage=True` torchao Int8Tensor
pin_memory() bug found via scripts/probe_group_offload_forward.py, against the REAL
MiniMax-H3 transformer (not the dummy stack), and measures the practical cost of each
candidate fix:
  B) use_stream=False, low_cpu_mem_usage=True  (no eager pinning, no double-buffer prefetch)
  C) use_stream=True,  low_cpu_mem_usage=False (eagerly pins ~34GB of CPU RAM at
     enable_group_offload() time)

Runs one tiny forward-adjacent check (just a few real transformer_blocks[i](...) calls
would need full H3 kwargs, which is a lot of setup) -- instead this drives the *actual*
group offloading onload/offload cycle directly via `group.onload_()` /
`.offload_()` on a couple of the hook-attached blocks, which is exactly the code path
that failed in the server run and is enough to prove correctness without needing the
full pipeline's forward() kwargs.
"""
import time

import torch


def ram_gb():
    with open("/proc/meminfo") as f:
        meminfo = {}
        for line in f:
            parts = line.split()
            meminfo[parts[0].rstrip(":")] = int(parts[1])
    return {
        "avail_gb": round(meminfo["MemAvailable"] / 1e6, 1),
        "total_gb": round(meminfo["MemTotal"] / 1e6, 1),
    }


def gpu_gb():
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
    }


def load_transformer():
    from diffusers import ModularPipeline, TorchAoConfig
    from torchao.quantization import Int8WeightOnlyConfig

    H3_INT8_MODULES_TO_NOT_CONVERT = [
        "proj_in", "audio_proj_in", "context_embedder",
        "time_embedder", "time_proj", "token_refiner",
        "norm_out", "proj_out", "audio_proj_out",
    ]
    pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3")
    quant_config = TorchAoConfig(
        Int8WeightOnlyConfig(version=2),
        modules_to_not_convert=H3_INT8_MODULES_TO_NOT_CONVERT,
    )
    pipe.load_components(
        names=["transformer"],
        dtype=torch.bfloat16,
        quantization_config={"transformer": quant_config},
        device_map={"transformer": "cpu"},
    )
    return pipe.transformer


def try_variant(transformer, use_stream: bool, low_cpu_mem_usage: bool, label: str):
    print(f"\n=== {label}: use_stream={use_stream} low_cpu_mem_usage={low_cpu_mem_usage} ===", flush=True)
    print(f"  ram_before_enable={ram_gb()}", flush=True)
    t0 = time.time()
    transformer.enable_group_offload(
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=1,
        non_blocking=True,
        use_stream=use_stream,
        record_stream=False,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    t1 = time.time()
    print(f"  enable_group_offload() done in {t1 - t0:.1f}s. ram_after_enable={ram_gb()} gpu={gpu_gb()}", flush=True)

    # Directly exercise onload/offload cycles on a few blocks (mirrors what a real
    # forward pass would trigger via the pre_forward/post_forward hooks) without needing
    # the full transformer's forward() kwargs (rotary embeds, attention masks, timestep
    # embeddings, etc. -- out of scope for this isolated check).
    from diffusers.hooks.hooks import HookRegistry
    from diffusers.hooks.group_offloading import _GROUP_OFFLOADING

    ok = True
    try:
        for i in [0, 1, 2, 25, 49]:
            block = transformer.transformer_blocks[i]
            registry = HookRegistry.check_if_exists_or_initialize(block)
            hook = registry.get_hook(_GROUP_OFFLOADING)
            if hook is None:
                print(f"  block {i}: no group_offloading hook found (unexpected)", flush=True)
                ok = False
                continue
            t2 = time.time()
            hook.group.onload_()
            torch.cuda.synchronize()
            t3 = time.time()
            hook.group.offload_()
            torch.cuda.synchronize()
            t4 = time.time()
            print(f"  block {i}: onload {t3-t2:.3f}s, offload {t4-t3:.3f}s", flush=True)
        print(f"  [{label}] onload/offload cycles OK. gpu={gpu_gb()} ram={ram_gb()}", flush=True)
    except Exception as e:
        print(f"  [{label}] FAILED: {e!r}", flush=True)
        import traceback
        traceback.print_exc()
        ok = False

    return ok, t1 - t0


def main():
    print(f"[start] ram={ram_gb()}", flush=True)
    transformer = load_transformer()
    print(f"[loaded] ram={ram_gb()}", flush=True)

    # Test B first (use_stream=False) -- cheaper to attach, no eager pinning.
    ok_b, enable_time_b = try_variant(transformer, use_stream=False, low_cpu_mem_usage=True, label="B")

    print("\n[main] B done, this process will now exit (each variant needs a fresh "
          "transformer since enable_group_offload() hooks accumulate) -- re-run with "
          "--variant=c to test C separately.", flush=True)


if __name__ == "__main__":
    import sys
    if "--variant=c" in sys.argv:
        print(f"[start] ram={ram_gb()}", flush=True)
        transformer = load_transformer()
        print(f"[loaded] ram={ram_gb()}", flush=True)
        try_variant(transformer, use_stream=True, low_cpu_mem_usage=False, label="C")
    else:
        main()

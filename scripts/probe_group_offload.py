"""
Probe: can the MiniMax-H3 transformer be loaded with device_map="cpu" + torchao int8
load-time quantization, then have enable_group_offload() applied on top?

This checks (in order, aborting early on failure so later steps don't waste time):
  1. Does `device_map="cpu"` (a plain string -> {"" : torch.device("cpu")} per
     modeling_utils.py) actually trigger torchao's `self.offload = True` skip-quantize
     path, or does the model quantize normally? (see the reading of torchao_quantizer.py:
     `"cpu" in device_map.values()` compares against the *string* "cpu", but a plain
     string device_map becomes {"" : torch.device("cpu")} -- a torch.device object, not
     the string "cpu" -- so the `in` check should be False and quantization SHOULD
     proceed normally even though every weight lands on CPU).
  2. Measure host RAM before/after (quantized int8 on CPU should be ~34GB, same as the
     already-measured GPU int8 size -- CPU tensors have no VRAM-specific reason to differ
     in size from their GPU counterparts).
  3. Call `enable_group_offload(onload_device=cuda, offload_device=cpu, ...)` on the
     resulting int8-quantized-on-CPU module and confirm no exception.
  4. Run a tiny dummy forward (if easy) or at minimum confirm hooks were attached and
     params report as still on CPU (group offloading keeps CPU as the storage device,
     onloading only during forward).

Not wired into runner.py yet -- this is purely exploratory, run standalone:
    venv/bin/python scripts/probe_group_offload.py
"""
import gc
import time

import torch

MODEL_ID = "MiniMaxAI/MiniMax-H3"


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


def main():
    print(f"[probe] start. ram={ram_gb()} gpu={gpu_gb()}", flush=True)

    from diffusers import ModularPipeline, TorchAoConfig
    from torchao.quantization import Int8WeightOnlyConfig

    H3_INT8_MODULES_TO_NOT_CONVERT = [
        "proj_in", "audio_proj_in", "context_embedder",
        "time_embedder", "time_proj", "token_refiner",
        "norm_out", "proj_out", "audio_proj_out",
    ]

    print("[probe] building ModularPipeline shell...", flush=True)
    pipe = ModularPipeline.from_pretrained(MODEL_ID)

    quant_config = TorchAoConfig(
        Int8WeightOnlyConfig(version=2),
        modules_to_not_convert=H3_INT8_MODULES_TO_NOT_CONVERT,
    )

    t0 = time.time()
    print(f"[probe] loading transformer with device_map='cpu' + int8 quant config... ram_before={ram_gb()}", flush=True)
    pipe.load_components(
        names=["transformer"],
        dtype=torch.bfloat16,
        quantization_config={"transformer": quant_config},
        device_map={"transformer": "cpu"},
    )
    t1 = time.time()
    print(f"[probe] load done in {t1 - t0:.1f}s. ram={ram_gb()} gpu={gpu_gb()}", flush=True)

    transformer = pipe.transformer
    if transformer is None:
        print("[probe] FAILED: pipe.transformer is None after load_components()", flush=True)
        return

    # Inspect a linear layer's weight class to confirm it was actually quantized
    # (torchao replaces nn.Linear.weight with an AffineQuantizedTensor subclass on
    # success; if `self.offload` skip-quantize kicked in it would stay a plain
    # bf16 torch.Tensor instead).
    found_quantized = False
    found_plain = []
    sample_count = 0
    for name, module in transformer.named_modules():
        if isinstance(module, torch.nn.Linear):
            sample_count += 1
            weight = module.weight
            weight_cls = weight.__class__.__name__
            is_plain_tensor = weight_cls == "Tensor" or weight_cls == "Parameter"
            if any((key + ".") in name or key == name.split(".")[-1] for key in H3_INT8_MODULES_TO_NOT_CONVERT):
                # expected to be skipped (modules_to_not_convert)
                continue
            if is_plain_tensor:
                found_plain.append((name, weight_cls, str(weight.dtype)))
            else:
                found_quantized = True
            if sample_count <= 5 or (found_quantized and len(found_plain) == 0 and sample_count % 50 == 0):
                print(f"[probe]   sample linear: {name} weight_cls={weight_cls} dtype={weight.dtype} device={weight.device}", flush=True)

    print(f"[probe] scan done: sample_count={sample_count} found_quantized={found_quantized} found_plain_count={len(found_plain)}", flush=True)
    if found_plain[:5]:
        print(f"[probe]   first few plain (non-quantized, unexpected) layers: {found_plain[:5]}", flush=True)

    # confirm device
    first_param = next(transformer.parameters())
    print(f"[probe] first_param.device={first_param.device}", flush=True)

    print(f"[probe] ram_after_load={ram_gb()}", flush=True)

    if not found_quantized:
        print("[probe] FAILED: no quantized linear layers found -- device_map='cpu' likely triggered the offload skip-quantize path after all.", flush=True)
        return

    print("[probe] SUCCESS: transformer loaded on CPU AND quantized to int8.", flush=True)

    # --- now try enable_group_offload ---
    print("[probe] calling enable_group_offload...", flush=True)
    t2 = time.time()
    try:
        transformer.enable_group_offload(
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type="block_level",
            num_blocks_per_group=1,
            non_blocking=True,
            use_stream=True,
            record_stream=False,
            low_cpu_mem_usage=True,
        )
        t3 = time.time()
        print(f"[probe] enable_group_offload() succeeded in {t3 - t2:.1f}s. gpu={gpu_gb()} ram={ram_gb()}", flush=True)
    except Exception as e:
        print(f"[probe] FAILED: enable_group_offload() raised: {e!r}", flush=True)
        import traceback
        traceback.print_exc()
        return

    print("[probe] DONE. group offload attached to CPU-resident int8 transformer without error.", flush=True)


if __name__ == "__main__":
    main()

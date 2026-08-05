"""
Follow-up probe to scripts/probe_group_offload.py: probe_group_offload.py only confirmed
`enable_group_offload()` attaches without error. The real server run failed inside the
first actual *forward* pass with:

    RuntimeError: cannot pin 'torch.cuda.CharTensor' only dense CPU tensors can be pinned

(from `hooks/group_offloading.py::_pinned_memory_tensors` -> `Int8Tensor.qdata.pin_memory()`).

This script reproduces that failure in isolation (small dummy Linear stack, not the full
H3 transformer) to characterize it faster than a full server round-trip, and to test the
two candidate workarounds:
  (a) `use_stream=False` (drop the double-buffered prefetch, forgo the pin_memory path
      entirely -- `_onload_from_memory` only calls `_pinned_memory_tensors()` when
      `self.stream is not None`)
  (b) manually pre-pinning cpu_param_dict tensors is not exposed at the enable_group_offload()
      API level, so (a) is the only practical mitigation without patching diffusers.
"""
import torch

MODEL_ID = "MiniMaxAI/MiniMax-H3"


def gpu_gb():
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
    }


def build_dummy_int8_stack(num_blocks=4, dim=256):
    from diffusers import TorchAoConfig
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = torch.nn.Linear(dim, dim * 2, bias=False, dtype=torch.bfloat16)
            self.lin2 = torch.nn.Linear(dim * 2, dim, bias=False, dtype=torch.bfloat16)

        def forward(self, x):
            return self.lin2(torch.nn.functional.gelu(self.lin1(x)))

    class Stack(torch.nn.Module):
        _supports_group_offloading = True

        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block() for _ in range(num_blocks)])

        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x

    stack = Stack()
    quant_config = Int8WeightOnlyConfig(version=2)
    for block in stack.blocks:
        quantize_(block, quant_config)
    return stack


def try_forward(use_stream: bool, low_cpu_mem_usage: bool, label: str):
    print(f"\n=== {label}: use_stream={use_stream} low_cpu_mem_usage={low_cpu_mem_usage} ===", flush=True)
    from diffusers.models.modeling_utils import ModelMixin

    stack = build_dummy_int8_stack()
    # enable_group_offload is a ModelMixin method; patch it onto our dummy stack the same
    # way diffusers does internally (apply_group_offloading works on any nn.Module with
    # _supports_group_offloading, but the convenience wrapper lives on ModelMixin).
    from diffusers.hooks import apply_group_offloading

    apply_group_offloading(
        module=stack,
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=1,
        non_blocking=True,
        use_stream=use_stream,
        record_stream=False,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )

    x = torch.randn(2, 256, dtype=torch.bfloat16)
    try:
        with torch.no_grad():
            for i in range(3):
                out = stack(x)
                torch.cuda.synchronize()
        print(f"[{label}] forward x3 OK. out.shape={out.shape} gpu={gpu_gb()}", flush=True)
        return True
    except Exception as e:
        print(f"[{label}] FAILED: {e!r}", flush=True)
        import traceback
        traceback.print_exc()
        return False


def main():
    results = {}
    results["stream=True,low_cpu=True"] = try_forward(True, True, "A")
    results["stream=False,low_cpu=True"] = try_forward(False, True, "B")
    results["stream=True,low_cpu=False"] = try_forward(True, False, "C")
    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k}: {'OK' if v else 'FAILED'}")


if __name__ == "__main__":
    main()

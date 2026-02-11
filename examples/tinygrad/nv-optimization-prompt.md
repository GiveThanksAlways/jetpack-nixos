# NV Backend Optimization — AI Agent Prompt

You are working on the tinygrad NV backend (TegraIface/HCQ) running on a **Jetson Orin AGX 64GB** (JetPack 6, L4T r36.4.4, SM 8.7, CUDA 12.6). The NV backend bypasses the CUDA driver API and talks directly to the `/dev/nvgpu-gpu` and `/dev/nvmap` kernel interfaces for lower overhead.

## Context

Read these files first to understand the full context:

1. **`nv-optimization-plan.md`** — The prioritized optimization roadmap (P0-P5) with benchmarks showing where NV=1 wins and where it falls flat
2. **`robust-testing-and-performance.md`** — Master tracking doc with all Phase A-E results, known bugs, and code locations
3. **`tinygrad/tinygrad/runtime/ops_nv.py`** — The NV backend implementation (TegraIface class starts ~L575, NVDevice ~L1100, NVAllocator ~L330)
4. **`tinygrad/tinygrad/renderer/ptx.py`** — PTX code generation (vectorized load/store patterns at ~L107-119)
5. **`tinygrad/tinygrad/nn/state.py`** — GGUF model loading and dequantization (~L310-360)

## Your Tasks (in priority order)

### Task 1: P0 — Fix Misaligned Access Crash (CRITICAL)

**Problem:** NV=1 crashes with `MISALIGNED_ADDR` GPU SM exception when running any quantized model (LLaMA 3.2 1B Q6_K). CUDA=1 works fine on the same model.

**Root Cause:** Q6_K uses 210-byte blocks. The dequant code in `nn/state.py` reshapes tensors into `(-1, 210)` rows, then slices at offsets like 128, 192. For row `i`, the slice starts at `base + i*210 + offset` — which is not aligned to 4/8/16 bytes when `i > 0` (since 210 is not a power of 2). The PTX renderer emits `ld.global.v4.u32` (requires 16-byte alignment) for these accesses. The CUDA driver handles misalignment via trap-and-emulate; the NV/Tegra HW enforces strict alignment and crashes.

**Evidence from dmesg:**
```
nvgpu: sm machine check err. gpc_id(0), tpc_id(0)
hww_global_esr 4 (MULTIPLE_WARP_ERRORS)
hww_warp_esr 0x10 (MISALIGNED_ADDR)
hww_warp_esr_pc 0xff55a2a830
```

**Recommended Fix (Option A):** Modify the PTX renderer to emit scalar loads (`ld.global.u8`, `ld.global.u32`) instead of vectorized loads (`ld.global.v4.u32`) when the underlying data stride is not a power of 2. This requires:

1. Detecting in the renderer or scheduler when a tensor's innermost stride results in non-power-of-2 aligned accesses
2. Falling back to scalar loads for those cases
3. Only applying this on the NV/Tegra backend (CUDA handles it fine)

**Alternative approaches:**
- **Option B:** Pad Q6_K blocks from 210 to 256 bytes in `nn/state.py` before dequant. Wastes 22% memory but ensures alignment.
- **Option C:** Add alignment metadata to the scheduler/linearizer to prevent vectorized loads when alignment can't be proven.

**How to test:**
```bash
cd /home/agent/jetpack-nixos/examples/tinygrad
nix develop -c bash -c 'cd tinygrad && NV=1 python3 examples/llama3.py --size 1B --no_api --benchmark --timing 2>&1 | tee tests/results_llama3_nv_fixed.log'
```

Success = no crash, tok/s output printed.

Also run the full test suite to ensure no regressions:
```bash
nix develop -c bash -c 'cd tinygrad && NV=1 python3 -m pytest test/test_ops.py -x -v --tb=short 2>&1 | tail -20'
```

### Task 2: P1 — Enable Huge Pages in TegraIface

**Problem:** `TegraIface.alloc()` at `ops_nv.py` ~L1166 hardcodes `page_size = mmap.PAGESIZE` (4 KB). The desktop `NVKIface.alloc()` at ~L494 uses 2 MB huge pages for allocations ≥ 8 MB. This means large weight tensors (4.9 GB for LLaMA 1B) use 1.2M page table entries instead of 2,450.

**Fix:** Change line ~1166 in `TegraIface.alloc()`:
```python
# Before:
page_size = mmap.PAGESIZE

# After:
page_size = (2 << 20) if size >= (8 << 20) else mmap.PAGESIZE
```

**Caveats to verify:**
- Does `nvgpu_as_map_buffer_ex` support 2MB pages on Tegra? Check if `page_size` field accepts large values.
- Does `nvmap ALLOC` with `align = 2MB` work? It might need specific heap flags.
- Check the L4T kernel source at `l4t-sources/nvgpu/` for page size support.

**How to test:**
```bash
# Quick sanity: allocate and use a buffer
nix develop -c bash -c 'cd tinygrad && NV=1 python3 -c "
from tinygrad import Tensor, Device
Device[\"NV\"]
a = Tensor.randn(1024, 1024).realize()
print(a.numpy().mean())
print(\"OK\")
"'

# Benchmark memory bandwidth:
nix develop -c bash -c 'cd tinygrad && NV=1 python3 tests/benchmark_nv_vs_cuda.py --copyout --d2d'

# Run full model:
nix develop -c bash -c 'cd tinygrad && NV=1 python3 examples/gpt2.py --model_size gpt2 --count 20 --temperature 0 --timing'
```

### Task 3: P2 — Benchmark fp16 GPT-2 (Quick Win)

After P0 and P1, run these benchmarks to find the first scenario where NV=1 clearly beats CUDA=1:

```bash
# fp16 GPT-2 — NV=1
nix develop -c bash -c 'cd tinygrad && NV=1 HALF=1 python3 examples/gpt2.py --model_size gpt2 --count 50 --temperature 0 --timing 2>&1 | tee tests/results_gpt2_nv_half.log'

# fp16 GPT-2 — CUDA=1
nix develop -c bash -c 'cd tinygrad && CUDA=1 HALF=1 python3 examples/gpt2.py --model_size gpt2 --count 50 --temperature 0 --timing 2>&1 | tee tests/results_gpt2_cuda_half.log'
```

NV=1 is 26-50% faster on fp16 matmul in isolation. If GPT-2 fp16 is compute-bound enough, this should translate to a measurable win.

## Environment Setup

All commands run inside the tinygrad nix dev shell:
```bash
cd /home/agent/jetpack-nixos/examples/tinygrad
nix develop
# Now you're in the shell with CUDA 12.6, Python 3.13, etc.
cd tinygrad
```

The `NV=1` env var selects the NV/Tegra backend. `CUDA=1` selects the standard CUDA backend. Only set one at a time.

## Key Code Architecture

- **TegraIface** (`ops_nv.py` ~L575-1100): Tegra-specific implementation of the NV interface. Handles allocation via nvmap, GPU VA mapping, channel setup, RM control emulation.
- **NVDevice** (`ops_nv.py` ~L1100+): Device initialization, GPFIFO setup, compute/copy/DMA queues.
- **NVComputeQueue** (`ops_nv.py`): Builds QMD (Queue Meta Data) for kernel dispatch, submits to GPFIFO.
- **NVAllocator** (`ops_nv.py` ~L330-370): Memory allocator with Tegra-optimized direct memcpy.
- **PTXRenderer** (`ptx.py`): Generates PTX assembly from tinygrad IR. The vectorized load/store patterns are the key area for P0.
- **HCQ framework** (`support/hcq.py`): Shared infrastructure for signals, command queues, kernargs bump allocator.

## Important Notes

- The tinygrad repo is at `tinygrad/` inside the workspace (i.e., `examples/tinygrad/tinygrad/`)
- Test with `python3 -m pytest test/test_ops.py` for correctness
- Test with `python3 -m pytest test/test_hcq.py` for HCQ-specific tests
- Always check `dmesg | tail -20` after runs for GPU errors
- The `_tegra_signal` workaround (P5 item 1) is currently required for correctness — don't remove it without fixing the underlying QMD reuse race
- All tinygrad patches should be minimal and targeted — this is upstream code we want to eventually contribute back

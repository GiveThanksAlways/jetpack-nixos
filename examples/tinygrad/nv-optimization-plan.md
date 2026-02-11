# NV Backend Optimization Plan — Jetson Orin AGX 64GB

**Goal:** Make `NV=1` (TegraIface/HCQ) decisively faster than `CUDA=1` on real LLM inference, and unblock quantized model support.

**Device:** Jetson Orin AGX 64GB, JetPack 6, L4T r36.4.4, ga10b iGPU, SM 8.7, 64 GB LPDDR5 (~204 GB/s theoretical)

---

## Current State (Phase E Benchmarks, 2026-02-11)

### Where NV=1 Already Wins (Micro-Benchmarks)

These are individual operation benchmarks where the NV/Tegra backend's lower overhead shines:

| Benchmark | NV=1 | CUDA=1 | NV Advantage | Why NV Wins |
|-----------|------|--------|--------------|-------------|
| **fp16 matmul 2048×2048** | 932 GFLOPS | 622 GFLOPS | **+50%** | Direct QMD dispatch skips CUDA driver overhead; same tensor core HW |
| **fp16 matmul 1024×1024** | 388 GFLOPS | 307 GFLOPS | **+26%** | Lower launch overhead is proportionally larger for smaller problems |
| **fp32 matmul 1024×1024** | 134 GFLOPS | 101 GFLOPS | **+32%** | Same story — NV has less per-kernel overhead |
| **Element-wise 10M (add)** | 47% faster | baseline | **+47%** | Near-zero launch overhead via direct GPFIFO/QMD |
| **Element-wise 10M (mul)** | 27% faster | baseline | **+27%** | Same |
| **Copyout (D→H) 16MB** | 6.35 GB/s | 2.58 GB/s | **+146%** | Direct memmove on unified memory (custom optimization) |
| **Copyout (D→H) 1MB** | 4.38 GB/s | 0.75 GB/s | **+484%** | Skips DMA staging buffer entirely |
| **Copyin (H→D) 256MB** | 5.17 GB/s | 3.78 GB/s | **+37%** | Direct memmove vs DMA staging |
| **D2D copy 256MB** | 19.1 GB/s | 15.8 GB/s | **+21%** | Lower DMA command overhead |
| **Kernel launch p99** | 1199 µs | 2637 µs | **2.2× better** | More deterministic — no CUDA driver jitter |
| **Large alloc (256MB)** | 6.6 ms | 11.0 ms | **+40% faster** | nvmap direct vs CUDA driver alloc path |

**Key insight:** NV=1 is 26-50% faster on fp16 matmul, 2-5× faster on memory copies, and 2× more deterministic on kernel launch. These are real, significant wins.

### Where It Falls Flat (End-to-End LLM)

| Model | NV=1 tok/s | CUDA=1 tok/s | Δ | Why |
|-------|-----------|-------------|---|-----|
| GPT-2 124M (fp32) | 38.3 | 38.8 | **Tied** | Memory-bandwidth-bound at batch=1 |
| LLaMA 3.2 1B Q6_K | **CRASH** | 3.28-3.30 | **Blocked** | Misaligned addr in dequant kernel |

### The Gap: Why Micro-Benchmark Wins Vanish

Batch=1 autoregressive decode has arithmetic intensity of ~0.5 FLOP/byte — every token reads the **entire** weight matrix but does minimal compute per weight. At this ratio:

- Memory bandwidth is the bottleneck, not compute
- Both backends hit the same ~29 GB/s effective bandwidth wall
- The 50% matmul speedup is irrelevant because matmul is not the bottleneck
- Kernel launch savings (~1.4 ms/token) are only 5% of the ~26 ms/token total

**The NV wins would show up at batch ≥ 8-16** where compute becomes the bottleneck, or on **fp16 models** where NV's tensor core scheduling advantage matters.

---

## Priority Roadmap

### P0 — CRITICAL: Fix Misaligned Access Crash (Unblocks Quantized Models)

**Impact:** Currently NV=1 **cannot run any quantized model at all** — LLaMA Q6_K crashes with GPU SM exception.

**Root Cause:** Q6_K uses 210-byte blocks. The dequantization code in `nn/state.py` slices these blocks at odd offsets:

```python
# nn/state.py line ~340 (ggml_type == 14, Q6_K)
blocks = t[:(n//256)*210].reshape((-1, 210))  # 210-byte rows
xl, xh = q_to_uint8(blocks[:,:128].reshape((-1, 2, 64)), 4), \
         q_to_uint8(blocks[:,128:192].reshape((-1, 2, 32)), 2).lshift(4)
scales = blocks[:,192:208].bitcast(dtypes.int8)
d = blocks[:,-2:].bitcast(dtypes.float16)
```

When `blocks` starts at an aligned address, block `i` starts at `base + i*210`. For `i=1`, that's `base+210` — **not aligned to any power of 2**. The PTX renderer then emits vectorized loads like `ld.global.v4.u32` which require 16-byte alignment. CUDA driver silently handles misaligned vectorized loads via trap-and-emulate; the NV/Tegra backend enforces strict hardware alignment and crashes.

**dmesg evidence:**
```
nvgpu: sm machine check err. gpc_id(0), tpc_id(0)
hww_warp_esr 0x10 (MISALIGNED_ADDR)
hww_warp_esr_pc 0xff55a2a830
```

**Three Fix Options:**

| Option | Approach | Pros | Cons |
|--------|----------|------|------|
| **A (Recommended)** | Force scalar loads in PTX renderer when stride is not power-of-2 | Minimal code change, correct by construction | Slightly slower dequant (~5-10%) |
| **B** | Pad Q6_K blocks to 256 bytes before dequant | Aligned access, vectorized loads work | Uses 22% more memory, changes `nn/state.py` |
| **C** | Alignment check in tinygrad scheduler/linearizer | Prevents vectorized loads when alignment can't be proven | Complex, may miss cases |

**Code locations:**
- PTX vectorized load emission: `tinygrad/renderer/ptx.py` lines ~107-119 (the `ld.{mem_type}.v{count}` patterns)
- Q6_K dequant: `tinygrad/nn/state.py` lines ~325-342
- Q4_K dequant: `tinygrad/nn/state.py` lines ~333-339 (144-byte blocks, same class of problem)

**Verification:** After fix, run:
```bash
cd /home/agent/jetpack-nixos/examples/tinygrad
nix develop -c bash -c 'cd tinygrad && NV=1 python3 examples/llama3.py --size 1B --no_api --benchmark --timing 2>&1 | tee tests/results_llama3_nv_fixed.log'
```

---

### P1 — HIGH: Enable Huge Pages in TegraIface (10-30% Memory BW Improvement)

**Impact:** Reduce TLB misses for large allocations, improving memory bandwidth.

**Current State:** `TegraIface.alloc()` hardcodes `page_size = mmap.PAGESIZE` (4 KB) for all allocations. The desktop `NVKIface.alloc()` uses 2 MB huge pages for allocations ≥ 8 MB:

```python
# NVKIface.alloc() (ops_nv.py ~L494) — does this:
page_size = (2 << 20) if size >= (8 << 20) else mmap.PAGESIZE

# TegraIface.alloc() (ops_nv.py ~L1166) — always does:
page_size = mmap.PAGESIZE  # 4 KB, always
```

For a 1B parameter model with ~4.9 GB of weight tensors, that's **1.2 million 4KB pages** vs **2,450 huge pages**. The TLB on SM 8.7 has ~32 entries per SM — with 4KB pages, nearly every weight access is a TLB miss.

**Fix:** Mirror NVKIface's logic in TegraIface:
```python
# In TegraIface.alloc():
page_size = (2 << 20) if size >= (8 << 20) else mmap.PAGESIZE
```

Also need to verify that `nvgpu_as_map_buffer_ex` and `nvmap` support 2MB pages on Tegra — check `_NVGPU_AS_MAP_BUFFER_FLAGS_LARGE_PAGES` flag.

**Code location:** `tinygrad/runtime/ops_nv.py` line ~1166, `TegraIface.alloc()`

**Expected impact:** 10-30% improvement on memory-bandwidth-bound workloads (which includes all batch=1 LLM decode).

---

### P2 — QUICK WIN: Run GPT-2 with HALF=1 (fp16 Where NV Wins 50%)

**Impact:** Test whether NV=1's proven 50% fp16 matmul advantage shows up in end-to-end fp16 inference.

**Rationale:** GPT-2 supports `HALF=1` mode which loads weights in fp16. Since NV=1 is 26-50% faster on fp16 matmul, this should be the first scenario where NV=1 **clearly beats** CUDA=1.

**Command:**
```bash
# NV=1 fp16
NV=1 HALF=1 python3 examples/gpt2.py --model_size gpt2 --count 50 --temperature 0 --timing

# CUDA=1 fp16
CUDA=1 HALF=1 python3 examples/gpt2.py --model_size gpt2 --count 50 --temperature 0 --timing
```

**Expected:** If matmul dominates at fp16 (tensor cores engaged), NV=1 should win by 20-40%. If still tied, it confirms the model is purely memory-bandwidth-bound even at fp16.

---

### P3 — MEDIUM: Kernel Optimization with BEAM Search

**Impact:** Better kernel codegen can improve both backends, but NV=1 benefits more from optimized kernels due to lower dispatch overhead.

**Commands:**
```bash
# Try BEAM=2 (optimizes kernel selection)
NV=1 BEAM=2 python3 examples/gpt2.py --model_size gpt2 --count 50 --temperature 0 --timing

# Try JITBEAM=2 (JIT-time beam search)
NV=1 JITBEAM=2 python3 examples/gpt2.py --model_size gpt2 --count 50 --temperature 0 --timing
```

**Rationale:** tinygrad's default code generation may not produce optimal kernels for Orin's SM 8.7 architecture. BEAM search explores alternative schedules and picks the fastest. If NV=1's raw dispatch is faster, better kernels amplify the advantage.

---

### P4 — MEDIUM: Batch > 1 Inference (Where Compute Matters)

**Impact:** The NV=1 matmul wins (26-50%) only matter when inference is compute-bound. That happens at batch ≥ 8-16.

**Approach:** Modify the benchmark to process multiple prompts simultaneously:
```python
# In examples/llama3.py, change batch size for prefill phase
# Or use the server mode which naturally batches requests
```

Alternatively, benchmark the **prefill** phase (processing the prompt) rather than decode — prefill is compute-bound even at batch=1 for long prompts.

**Expected:** 20-50% NV=1 advantage at batch=8+ where matmul dominates.

---

### P5 — LOW: Advanced TegraIface Optimizations

These are smaller optimizations identified in the codebase:

| # | Optimization | Code Location | Expected Impact |
|---|-------------|---------------|-----------------|
| 1 | **Remove `_tegra_signal` workaround** | `ops_nv.py` L148-149, L183-193, L1329 | Enables QMD chaining for lower dispatch overhead. Requires fixing the QMD reuse race at the scheduler level instead of the current pushbuffer workaround. |
| 2 | **GPFIFO entry count tuning** | `NVDevice._new_gpu_fifo()` — `tegra_entries=1024` | Try 2048/4096 entries to reduce ring wrap overhead |
| 3 | **Doorbell write coalescing** | `NVCommandQueue._submit_to_gpfifo()` | Batch N GPFIFO entries, single doorbell write |
| 4 | **QMD field tuning** | `NVComputeQueue.exec()` | Compare QMD fields against CUDA's QMD for same kernel |
| 5 | **VA recycling** | `TegraIface` — `_alloc_gpu_vaddr` not implemented | Prevent VA space exhaustion in long-running workloads |

---

## Scenarios Where NV=1 Should Win Big

Based on the micro-benchmark data, these are the workloads where NV=1's advantages should translate to real speedups:

### 1. fp16 Models (P2)
- NV=1 is 26-50% faster on fp16 matmul
- GPT-2 with `HALF=1` or any fp16-native model should show this

### 2. Batched Inference (P4)
- At batch ≥ 8, inference becomes compute-bound
- NV=1's 32-50% matmul advantage should manifest as 20-40% throughput gain
- Server workloads (multiple concurrent requests) naturally batch

### 3. Small-to-Medium Model Inference
- NV=1's element-wise advantage (27-47%) helps most in models with many small operations
- Models with many attention heads, small FFN blocks, or complex routing (MoE) benefit

### 4. Latency-Sensitive Applications
- NV=1's 2.2× better p99 kernel launch makes it ideal for real-time applications
- Consistent timing matters for robotics, autonomous driving, voice assistants

### 5. Memory-Heavy Workloads (After P1 Huge Pages)
- With huge pages, NV=1 should reduce TLB misses and improve effective bandwidth
- Expected 10-30% improvement on all memory-bound workloads
- This could tip the GPT-2 benchmark from "tied" to "NV wins"

### 6. After P0 Fix — Quantized Model Comparison
- Once NV=1 can run quantized models, the comparison on LLaMA 3.2 1B will be possible
- Combined with huge pages (P1), NV=1 could outperform CUDA=1 on the most important real-world workload

---

## Success Criteria

| Milestone | What It Proves | Priority |
|-----------|---------------|----------|
| NV=1 runs LLaMA Q6_K without crash | P0 fix works | **Must have** |
| NV=1 beats CUDA=1 on GPT-2 HALF=1 | fp16 advantage translates | High |
| NV=1 beats CUDA=1 on LLaMA Q6_K decode | Quantized model perf win | High |
| NV=1 beats CUDA=1 at batch=8+ | Compute-bound advantage translates | Medium |
| NV=1 shows better p99 in sustained serving | Latency advantage translates | Medium |

---

## File References

| File | What | Key Lines |
|------|------|-----------|
| `tinygrad/tinygrad/runtime/ops_nv.py` | NV backend (TegraIface, NVDevice, NVAllocator) | L1155-1220 (alloc), L148-193 (tegra_signal), L343-362 (direct memcpy) |
| `tinygrad/tinygrad/renderer/ptx.py` | PTX code generation | L107-119 (vectorized load/store emission) |
| `tinygrad/tinygrad/nn/state.py` | GGUF loading / dequantization | L325-342 (Q6_K/Q4_K block layout) |
| `tinygrad/tinygrad/runtime/support/hcq.py` | HCQ framework (signals, queues, kernargs) | L361 (kernargs_size), L382-383 (BumpAllocator) |
| `tests/benchmark_nv_vs_cuda.py` | All micro-benchmarks | Full file |
| `robust-testing-and-performance.md` | Master tracking doc | Phase E section (~L498) |

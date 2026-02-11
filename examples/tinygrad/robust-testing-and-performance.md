# Robust Testing & Performance: NV=1 vs CUDA=1 on Jetson Orin AGX 64GB

**Date:** 2026-02-10
**Device:** NVIDIA Jetson Orin AGX 64GB Developer Kit
**JetPack:** 6 (L4T r36.4.4), Kernel 5.15.148
**GPU:** ga10b iGPU, Ampere arch, SM 8.7, compute class 0xc7c0
**CUDA:** 12.6
**tinygrad:** v0.12.0 (commit cc9bf8cc)
**Backend under test:** `TegraIface` in `ops_nv.py` (`NV=1`)
**Reference backend:** CUDA (`CUDA=1`)

---

## Purpose

Validate the NV/Tegra backend is correct and robust, then benchmark it against CUDA=1 to find performance gaps and optimization opportunities. All 4 implementation phases are **COMPLETE** — this document drives the hardening and optimization loop.

**Two goals:**
1. **Correctness:** Comprehensive testing (edge cases, stress tests, model inference) — NV=1 outputs must match CUDA=1 within tolerance (`atol=1e-4` for float32).
2. **Performance:** Benchmark NV=1 vs CUDA=1 across matmul, memory bandwidth, kernel launch overhead, element-wise ops, and model inference. Identify optimization opportunities in our Tegra ioctl/driver usage.

---

## Iteration Loop

```
┌─────────────────────────────────────────────────┐
│  1. Run correctness tests (B1→B6)               │
│  2. Fix any failures in ops_nv.py TegraIface    │
│  3. Re-run to confirm fix, no regressions       │
│  4. Once all green → run benchmarks (C1→C5)     │
│  5. Analyze results, identify bottlenecks       │
│  6. Optimize (see Phase D notes)                │
│  7. Re-benchmark to measure improvement         │
│  8. Update this doc with results                │
│  9. Repeat 6-8 until satisfied                  │
└─────────────────────────────────────────────────┘
```

> note: don't forget to look at the kernel logs (dmesg, journalctl, etc) for any warnings or errors that can help us in our building/debugging iteration loops. I have seen a few warnings/ errors on the UART myself, so always good to keep an eye on that
> for example this helped: dmesg | grep -i nvgpu or dmesg | grep ga10b

---

## Phase A: Test Infrastructure

### Files

| File | Purpose | Status |
|------|---------|--------|
| `tests/dmesg_checker.py` | Kernel log (dmesg) checker — detects GPU errors/warnings automatically | ✅ DONE |
| `tests/conftest.py` | Shared test harness: backend detection, output comparison helpers, timing utilities, memory tracking | ⬜ TODO |
| `tests/tegra_helpers.py` | Low-level ioctl helpers (already exists — extend as needed) | ✅ EXISTS |

### Dev Shell (Nix Flake)

All testing **must** use the tinygrad flake dev shell at `examples/tinygrad/flake.nix`. It provides the correct `pythonEnv` with all dependencies (numpy, tqdm, pillow, tiktoken, **pytest**, **hypothesis**, **torch**), CUDA libraries, and env vars for NixOS library discovery.

> **PyTorch note:** We use a **CPU-only aarch64 wheel** (`torch 2.9.1+cpu`) to avoid a multi-hour source build. PyTorch is only used as a **reference implementation** for correctness comparison in tinygrad's test suite (`test_ops.py`, `test_nn.py`). It is NOT used for benchmarking — all performance comparisons are NV=1 vs CUDA=1 (both tinygrad backends).

```bash
cd /home/agent/jetpack-nixos/examples/tinygrad && nix develop
cd tinygrad  # tinygrad source tree

# Run tests with pytest (available in the flake):
NV=1 python3 -m pytest test/test_ops.py -v --tb=short
CUDA=1 python3 -m pytest test/test_ops.py -v --tb=short

# Or use unittest:
NV=1 python3 -m unittest test.test_ops -v
```

> **Note:** pytest was added to the flake's `pythonEnv`. Both `python3 -m pytest` and `python3 -m unittest` work. Prefer pytest for its better output formatting and `--tb=short` option.

### Kernel Log Checking (`dmesg_checker.py`)

The dmesg checker is a critical part of our iteration loop. It automatically classifies GPU kernel messages:

| Category | Examples | Action |
|----------|----------|--------|
| **ERROR** | sked exception, MMU fault, CE not idle, PBDMA interrupt | Test FAILS — investigate immediately |
| **WARNING** | nvmap tag missing | Fix if possible, track otherwise |
| **KNOWN_HARMLESS** | `tu104_gr_init_commit_rtv_cb` (RTV not available on ga10b) | Suppressed — fires on every GR context init |
| **INFO** | Module stack traces, general nvgpu messages | Logged for context |

**Usage in tests:**
```python
from dmesg_checker import DmesgChecker, check_dmesg

# Context manager (recommended)
with DmesgChecker() as dc:
    run_my_test()
assert dc.report.is_clean, dc.report.summary()

# Decorator
@check_dmesg
def test_something():
    ...

# Manual
checker = DmesgChecker()
checker.clear()
run_test()
report = checker.check()
print(report.summary())
```

**Command-line usage:**
```bash
python3 tests/dmesg_checker.py              # Show recent GPU messages
python3 tests/dmesg_checker.py --watch      # Continuous monitoring
python3 tests/dmesg_checker.py --count 20   # Last 20 GPU messages
```

### Shared Helpers Needed

- **`compare_backends(fn, atol=1e-4)`**: Run `fn` under NV=1 and CUDA=1, assert numpy outputs are `allclose`.
- **`timed(fn, warmup=10, iters=90)`**: Return median/mean/p99 wall-clock time in ms.
- **`count_fds()` / `count_maps()`**: Check `/proc/self/fd` and `/proc/self/maps` counts for leak detection.
- **`get_backend()`**: Return `"NV"` or `"CUDA"` based on `Device.DEFAULT`.

---

## Phase B: Correctness Testing

### B1. Tinygrad's `test_hcq.py` (29 tests) — HCQ Framework on Tegra

**Why first:** HCQ is the lowest-level abstraction. If signal/wait/exec/copy don't work, nothing above will.

**Command:**
```bash
cd /home/agent/jetpack-nixos/examples/tinygrad/tinygrad
NV=1 python3 -m pytest test/device/test_hcq.py -v 2>&1 | tee ../tests/results_hcq_nv.log
```

**Expected skips** (single iGPU, no debugger):
- `test_multidevice_signal_wait` — single GPU
- `test_multidevice` — single GPU
- `test_multidevice_p2p` — single GPU, no P2P
- `test_speed_cross_device_copy_bandwidth` — single GPU
- `test_on_device_hang` — no `GT200_DEBUGGER` on Tegra (stubbed to raise generic error)

**Critical tests to watch:**
- `test_copy`, `test_copy_long`, `test_copy_64bit` — DMA copy queue never independently verified on Tegra
- `test_timeline_signal_rollover` — tests signal value wrapping (edge case)
- `test_memory_barrier`, `test_memory_barrier_before_copy` — cache coherence
- `test_map_cpu_buffer_to_device` — CPU↔GPU mapping correctness
- `test_small_copies_from_host_buf*` — small transfer edge cases
- `test_bind` — kernel argument binding

**Results:**
| Test | NV=1 | Notes |
|------|------|-------|
| test_bind | ✅ OK | |
| test_copy | ✅ OK | |
| test_copy_long | ✅ OK | |
| test_copy_64bit | ⏭ SKIP | `RUN_SLOW=1` required |
| test_exec_one_kernel | ✅ OK | |
| test_exec_2_kernels_100_times | ✅ OK | Fixed via pushbuffer signal (was val=198 due to QMD reuse race) |
| test_exec_update | ✅ OK | |
| test_exec_update_fuzz | ✅ OK | |
| test_map_cpu_buffer_to_device | ❌ FAIL | Pre-existing: `TegraAllocator.map()` is a no-op — CPU buffers not mapped to GPU AS |
| test_memory_barrier | ✅ OK | |
| test_memory_barrier_before_copy | ✅ OK | |
| test_multidevice | ⏭ SKIP | Single GPU |
| test_multidevice_p2p | ⏭ SKIP | Single GPU |
| test_multidevice_signal_wait | ⏭ SKIP | Single GPU |
| test_on_device_hang | ⏭ SKIP | MOCKGPU only |
| test_signal | ✅ OK | |
| test_signal_update | ✅ OK | |
| test_small_copies_from_host_buf | ✅ OK | |
| test_small_copies_from_host_buf_intercopy | ✅ OK | |
| test_small_copies_from_host_buf_transfer | ✅ OK | (1 skip) |
| test_speed_copy_bandwidth | ✅ OK | |
| test_speed_cross_device_copy_bandwidth | ⏭ SKIP | Single GPU |
| test_speed_exec_time | ✅ OK | |
| test_timeline_signal_rollover | ✅ OK | |
| test_update_copy | ✅ OK | |
| test_update_copy_long | ✅ OK | |
| test_wait | ✅ OK | |
| test_wait_late_set | ✅ OK | |
| test_wait_update | ✅ OK | |

**Summary:** 20/20 applicable tests pass. 5 expected skips (multidevice, MOCKGPU, slow). 1 known failure (map_cpu_buffer — needs `TegraAllocator.map()` implementation).

**Status:** ✅ PASSING (2026-02-11)

---

### B2. Tinygrad's `test_ops.py` — Tensor Operations

**Why:** Validates every mathematical operation the backend supports. Hundreds of tests covering matmul, conv2d, reduce, cast, unary, binary, ternary ops.

**Commands:**
```bash
# NV=1 baseline
cd /home/agent/jetpack-nixos/examples/tinygrad/tinygrad
NV=1 python3 -m pytest test/test_ops.py -v --tb=short 2>&1 | tee ../tests/results_ops_nv.log

# CUDA=1 reference (for comparison)
CUDA=1 python3 -m pytest test/test_ops.py -v --tb=short 2>&1 | tee ../tests/results_ops_cuda.log

# Quick diff
diff <(grep -E "PASSED|FAILED|ERROR" ../tests/results_ops_nv.log) \
     <(grep -E "PASSED|FAILED|ERROR" ../tests/results_ops_cuda.log)
```

**Results:**
| Backend | Passed | Failed | Errors | Skipped |
|---------|--------|--------|--------|---------|
| NV=1    |        |        |        |         |
| CUDA=1  |        |        |        |         |

**Status:** ⬜ NOT RUN

---

### B3. Tinygrad's `test_jit.py` — JIT Fused Kernels

**Why:** Tests kernel fusion and caching — a higher-level code path not yet exercised on Tegra.

**Command:**
```bash
cd /home/agent/jetpack-nixos/examples/tinygrad/tinygrad
NV=1 python3 -m pytest test/test_jit.py -v --tb=short 2>&1 | tee ../tests/results_jit_nv.log
```

**Results:**
| Backend | Passed | Failed | Errors | Skipped |
|---------|--------|--------|--------|---------|
| NV=1    |        |        |        |         |
| CUDA=1  |        |        |        |         |

**Status:** ⬜ NOT RUN

---

### B4. Custom Tegra Edge-Case Tests

**File:** `tests/test_tegra_edge_cases.py`

| Test | What it validates | Status |
|------|-------------------|--------|
| `test_40bit_va_boundary` | All GPU VAs < 2^40 (0x10000000000). Allocate many buffers, assert `va < (1 << 40)`. | ⬜ |
| `test_memory_pressure_progressive` | Allocate 1MB → 10MB → 100MB → 1GB → 4GB → 8GB. Record max successful size and failure mode. | ⬜ |
| `test_alloc_free_cycle_leak_check` | 1000 iterations of alloc(1MB)+free(). Check fd count and `/proc/self/maps` line count don't grow. | ⬜ |
| `test_dma_copy_small` | `NVCopyQueue.copy()` for 1B, 4B, 16B, 64B. Byte-exact verification via CPU readback. | ⬜ |
| `test_dma_copy_medium` | Copy 4KB, 64KB. Verify contents. | ⬜ |
| `test_dma_copy_large` | Copy 1MB, 16MB, 256MB. Verify contents. | ⬜ |
| `test_cacheability_correctness` | Run same compute kernel on INNER_CACHEABLE vs WRITE_COMBINE buffers. Assert identical numerical output. | ⬜ |
| `test_zero_element_tensor` | `Tensor([]).reshape(0, 3)` operations. Should not crash. | ⬜ |
| `test_one_element_tensor` | Scalar tensor ops (`Tensor(42.0) + Tensor(1.0)`). | ⬜ |
| `test_non_power_of_2_shapes` | Shapes: (7,), (13, 17), (127, 127), (1023,), (4097,). Matmul with non-aligned dims. | ⬜ |
| `test_all_dtypes` | float32, float16, int32, int8, bool — basic ops on each dtype. Compare NV=1 vs numpy. | ⬜ |
| `test_gpfifo_ring_wraparound` | Submit > 1024 commands (ring size) to force wraparound. Verify no corruption. | ⬜ |
| `test_free_correctness` | Audit `TegraIface.free()`: verify GPU VA unmapped, dmabuf fd closed, nvmap handle freed. Check `/proc/self/maps` and fd count before/after. | ⬜ |
| `test_cache_invalidation_pattern` | Write→compute→readback→modify→re-compute→readback. Tests whether NOP'd `invalidate_caches()` causes stale data. | ⬜ |
| `test_large_grid_launch` | Launch kernel with max grid dimensions. Tests QMD CTA/grid field limits. | ⬜ |

**Known potential bugs to test:**
1. **`TegraIface.free()` contradictory None check** — `if mem.view is not None: ... FileIOInterface.munmap(int(mem.va_addr) if mem.view is None else ...)` — the inner condition is always False. Test whether `munmap` actually fires correctly.
2. **`invalidate_caches()` NOP** — `NV2080_CTRL_CMD_FB_FLUSH_GPU_CACHE` is NOP'd in `rm_control()`. Determine if this causes stale reads.
3. **`num_sm_per_tpc` hardcoded to 2** — verify against actual GA10B hardware characteristics.

**Status:** ⬜ NOT WRITTEN

---

### B5. Custom Stress Tests

**File:** `tests/test_tegra_stress.py`

| Test | What it validates | Duration | Status |
|------|-------------------|----------|--------|
| `test_rapid_kernel_launches_10k` | Submit 10,000 trivial kernels back-to-back. Verify all complete via timeline signal. | ~10s | ⬜ |
| `test_signal_chain_pipeline` | Build compute→signal→wait→compute→signal pipeline, 1000 iterations. Verify final result. | ~5s | ⬜ |
| `test_sustained_matmul_60s` | 1024×1024 matmul in loop for 60 seconds. No hangs, leaks, or numerical drift. | 60s | ⬜ |
| `test_mixed_compute_copy` | Interleave compute and DMA copy operations, 1000 iterations. | ~10s | ⬜ |
| `test_backpressure` | Submit commands faster than GPU can execute. Verify GPFIFO handles backpressure (no data loss). | ~5s | ⬜ |
| `test_shared_memory_kernel` | Launch kernels that use shared memory. Verify `shared_mem_bytes` set correctly in QMD. | ~1s | ⬜ |
| `test_concurrent_tensor_ops` | Multiple tensor operations scheduled rapidly (like a real training step). | ~5s | ⬜ |
| `test_memory_churn` | Rapidly create and destroy tensors of varying sizes for 30s. Monitor for VA fragmentation / fd exhaustion. | 30s | ⬜ |

**Status:** ⬜ NOT WRITTEN

---

### B6. End-to-End Model Tests

**File:** `tests/test_tegra_models.py`

| Test | Model | What it validates | Status |
|------|-------|-------------------|--------|
| `test_simple_mlp` | 2-layer MLP (784→128→10) | Forward pass. Compare NV=1 vs CUDA=1 output logits (`allclose`). | ⬜ |
| `test_cnn_forward` | Simple CNN (2 conv + 2 FC) | Conv2d + pooling + FC. Compare outputs. | ⬜ |
| `test_transformer_block` | Single attention+FFN block | Self-attention + feedforward. Compare outputs. | ⬜ |
| `test_gpt2_small` | GPT-2 124M | 1 token generation. Compare logits. Record peak memory. | ⬜ |

**Status:** ⬜ NOT WRITTEN

---

## Phase C: Performance Benchmarking

### File: `tests/benchmark_nv_vs_cuda.py`

Only run after **all Phase B tests pass**.

### C1. Matmul (Compute Throughput)

| Size | dtype | NV=1 GFLOPS | CUDA=1 GFLOPS | NV/CUDA % | Notes |
|------|-------|-------------|---------------|-----------|-------|
| 256×256 | f32 | | | | |
| 512×512 | f32 | | | | |
| 1024×1024 | f32 | | | | Prior result: ~65 GFLOPS NV |
| 2048×2048 | f32 | | | | |
| 4096×4096 | f32 | | | | |
| 1024×1024 | f16 | | | | |
| 2048×2048 | f16 | | | | |
| 4096×4096 | f16 | | | | |

**Method:** 10 warmup + 90 timed iterations per size. Report: GFLOPS = 2N³ / time_seconds / 1e9.

### C2. Memory Bandwidth

| Operation | Size | NV=1 GB/s | CUDA=1 GB/s | NV/CUDA % |
|-----------|------|-----------|-------------|-----------|
| Host→Device (copyin) | 1MB | | | |
| Host→Device (copyin) | 16MB | | | |
| Host→Device (copyin) | 256MB | | | |
| Device→Host (copyout) | 1MB | | | |
| Device→Host (copyout) | 16MB | | | |
| Device→Host (copyout) | 256MB | | | |
| Device→Device (DMA copy) | 1MB | | | |
| Device→Device (DMA copy) | 16MB | | | |
| Device→Device (DMA copy) | 256MB | | | |
| Allocation latency | 1MB | ms: | ms: | |
| Allocation latency | 16MB | ms: | ms: | |
| Allocation latency | 256MB | ms: | ms: | |

### C3. Kernel Launch Overhead

| Metric | NV=1 | CUDA=1 | Notes |
|--------|------|--------|-------|
| Trivial kernel median latency (µs) | | | 1-element kernel, 1000 iters |
| Trivial kernel p99 latency (µs) | | | |
| Doorbell→completion median (µs) | | | NV-specific: ioctl + doorbell + signal |

### C4. Element-wise Ops (Bandwidth-Limited)

| Op | Size | NV=1 GB/s | CUDA=1 GB/s | NV/CUDA % |
|----|------|-----------|-------------|-----------|
| add | 1M elements | | | |
| add | 10M elements | | | |
| mul | 1M elements | | | |
| mul | 10M elements | | | |
| exp | 1M elements | | | |
| exp | 10M elements | | | |
| relu | 1M elements | | | |
| relu | 10M elements | | | |

### C5. Model Inference

| Model | Metric | NV=1 | CUDA=1 | NV/CUDA % |
|-------|--------|------|--------|-----------|
| GPT-2 small | tokens/sec (32 tokens) | | | |
| GPT-2 small | per-token latency (ms) | | | |
| GPT-2 small | peak memory (MB) | | | |

### Benchmark Runner

**File:** `tests/run_all_benchmarks.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== NV=1 Benchmarks ==="
NV=1 python3 tests/benchmark_nv_vs_cuda.py --output tests/results_nv.json

echo "=== CUDA=1 Benchmarks ==="
CUDA=1 python3 tests/benchmark_nv_vs_cuda.py --output tests/results_cuda.json

echo "=== Comparison ==="
python3 tests/generate_comparison.py tests/results_nv.json tests/results_cuda.json
```

---

## Phase D: Optimization Opportunities

After benchmarking, investigate these areas to close the NV-vs-CUDA gap. Each links to the relevant code in `ops_nv.py`.

### D1. Memory Allocation Strategy — `INNER_CACHEABLE` vs `WRITE_COMBINE`

**Context:** Phase 2 testing showed `INNER_CACHEABLE` reads at ~11.8 GB/s vs `WRITE_COMBINE` at ~691 MB/s (17× slower). The current `TegraIface.alloc()` uses `INNER_CACHEABLE` by default.

**Opportunity:** Verify CUDA backend's allocation cacheability policy. If CUDA uses different policies for different buffer types (e.g., `WRITE_COMBINE` for output buffers that GPU writes and CPU reads), we may benefit from matching that strategy.

**Code:** `TegraIface.alloc()` in `ops_nv.py` — the `NVMAP_HANDLE_INNER_CACHEABLE` flag.

### D2. GPFIFO Entry Count Tuning

**Context:** Currently using `tegra_entries=1024` per channel. Larger rings reduce the frequency of ring wraparound and may reduce submission latency.

**Opportunity:** Experiment with 2048, 4096, 8192 entries. Measure kernel launch overhead change.

**Code:** `NVDevice._new_gpu_fifo()` — `tegra_entries` parameter.

### D3. Doorbell Write Coalescing

**Context:** Each GPFIFO submission does: write entry → write GPPut → barrier → write doorbell. For batched submissions, we could write multiple entries before ringing the doorbell once.

**Opportunity:** Batch N GPFIFO entries, then single doorbell write. May reduce PCIe/interconnect overhead.

**Code:** `NVCommandQueue._submit_to_gpfifo()` in `ops_nv.py`.

### D4. QMD Field Tuning

**Context:** QMD v03 for Ampere has many tunable fields: CTA raster width, shared memory allocation granularity, register count, etc.

**Opportunity:** Profile the generated QMD against CUDA's QMD for the same kernel. Diff the fields.

**Code:** `NVComputeQueue.exec()` in `ops_nv.py`.

### D5. Cache Invalidation (Currently NOP'd)

**Context:** `NV2080_CTRL_CMD_FB_FLUSH_GPU_CACHE` is NOP'd in `rm_control()`. The `invalidate_caches()` method calls this, so cache invalidation doesn't actually happen on Tegra.

**Opportunity:** Determine if the Jetson's cache-coherent memory (IO coherence via CVM/SysRAM) makes this unnecessary, or if we need a Tegra-native cache flush ioctl. This could affect correctness under memory-intensive workloads.

**Code:** `TegraIface.rm_control()` — the `FB_FLUSH_GPU_CACHE` case.

### D6. SM Per TPC Hardcode

**Context:** `num_sm_per_tpc` is hardcoded to 2 in the `NV2080_CTRL_CMD_GR_GET_INFO` translation. If incorrect, this affects occupancy calculations and kernel launch grid sizing.

**Opportunity:** Query actual value from GPU characteristics or kernel driver. The `_nvgpu_gpu_characteristics` struct has `num_tpc_per_gpc` and `sm_arch_sm_version` but not SM-per-TPC directly. Check if ga10b HAL code in L4T sources confirms `sm_per_tpc=2`.

**Code:** `TegraIface.rm_control()` — the `GR_GET_INFO` hardcoded dict.

### D7. Deterministic Channel Flags

**Context:** `NVGPU_GPU_FLAGS_SUPPORT_DETERMINISTIC` is used in `SETUP_BIND`. This enables a lower-overhead submission path in the kernel where the driver skips some synchronization.

**Opportunity:** Verify this flag is being used correctly and that we're not accidentally falling through to the non-deterministic path. Check if `DETERMINISTIC` mode has measurable latency benefits.

**Code:** `TegraIface.rm_alloc()` — channel setup flags.

### D8. Shared Memory Window Placement

**Context:** `shared_mem_window=0xFE00000000`, `local_mem_window=0xFD00000000` — these are within the 40-bit VA space but their placement wasn't tuned.

**Opportunity:** Check if the kernel driver has preferences for these window locations. Suboptimal placement could cause TLB thrashing.

**Code:** `NVDevice._setup_gpfifos()` — window address constants.

---

## Known Bugs / Issues to Investigate

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | ~~`test_exec_2_kernels_100_times` gets val=198 instead of 200~~ **FIXED**: QMD reuse race — CPU overwrites QMD release_payload while GPU reads dependent QMD chain. On Tegra, fast MMIO doorbell outpaces GPU QMD reads (desktop masked by PCIe latency). Fix: force pushbuffer-based signal release on Tegra (`NVComputeQueue._tegra_signal = True`). | **Critical** | ✅ Fixed |
| 2 | ~~nvmap `allocation tag` kernel WARNING~~ **FIXED**: bits [31:16] of `_nvmap_alloc_handle.flags` must contain a nonzero tag. Added `_NVMAP_TAG_TINYGRAD = 0x0900` to all 3 alloc sites. | Medium | ✅ Fixed |
| 3 | `test_map_cpu_buffer_to_device` fails — `TegraAllocator.map()` is a no-op, so CPU buffers can't be copied via GPU DMA | Medium | ⬜ TODO |
| 4 | `TegraIface.free()` has contradictory `None` check: inner `if mem.view is None` inside block guarded by `if mem.view is not None` — always evaluates to `mem.view._addr`. May be correct by accident but logic is confusing. | Low | ⬜ Audit in B4 |
| 5 | `invalidate_caches()` is NOP'd — could cause stale data under certain access patterns | Medium | ⬜ Test in B4 |
| 6 | `num_sm_per_tpc` hardcoded to 2 — may affect occupancy/grid calculations | Low | ⬜ Verify in B4 |
| 7 | `viddec_class=None` — video decode unavailable, `NVVideoQueue` methods will fail | Info | N/A (expected on Tegra) |
| 8 | `pma_enabled=False` — no hardware profiling counters | Info | N/A (expected on Tegra) |
| 9 | No VA recycling in Tegra path — `_alloc_gpu_vaddr` not implemented | Medium | ⬜ Test in B4 (memory pressure) |

---

## Results Log

*(Update this section as tests are run)*

### Run 1: 2026-02-11

**Phase B Results:**
- B1 (test_hcq): 20 / 20 passed, 5 expected skips, 1 known failure (map_cpu_buffer)
- B2 (test_ops): ⬜ NOT RUN
- B3 (test_jit): ⬜ NOT RUN
- B4 (edge cases): ⬜ NOT WRITTEN
- B5 (stress): ⬜ NOT WRITTEN
- B6 (models): ⬜ NOT WRITTEN

**Kernel logs (dmesg):** ✅ Clean after all B1 tests — no sked exceptions, no nvmap tag warnings, no CE engine errors.

**Failures fixed this session:**
1. **QMD reuse race** (`test_exec_2_kernels_100_times` val=198): Forced pushbuffer-based signal release on Tegra via `NVComputeQueue._tegra_signal`. Root cause: fast MMIO doorbell lets CPU overwrite QMD release_payload before GPU reads dependent QMD chain. Pushbuffer signal values are bump-allocated per submit (immutable), eliminating the race.
2. **nvmap tag warnings**: Added `_NVMAP_TAG_TINYGRAD = 0x0900` to `_nvmap_alloc_handle.flags` at all 3 allocation sites.
3. **Reverted unnecessary WC change**: The `NVAllocator._alloc` WC-for-Tegra-cpu_access change was based on wrong root cause analysis (cache coherence). Reverted — the real fix is pushbuffer signal.

**Phase C Highlights:**
- Not yet run — waiting for all B phases to complete.

---

## File Inventory

| File | Purpose |
|------|---------|
| `tests/dmesg_checker.py` | Kernel log checker (dmesg) — GPU error/warning detection |
| `robust-testing-and-performance.md` | This guide (master tracking doc) |
| `tests/conftest.py` | Shared test infrastructure |
| `tests/tegra_helpers.py` | Low-level ioctl helpers (existing) |
| `tests/test_tegra_edge_cases.py` | B4: Edge case tests |
| `tests/test_tegra_stress.py` | B5: Stress tests |
| `tests/test_tegra_models.py` | B6: Model tests |
| `tests/benchmark_nv_vs_cuda.py` | C: All performance benchmarks |
| `tests/run_all_benchmarks.sh` | C: Benchmark runner script |
| `tests/generate_comparison.py` | C: Results comparison reporter |

## Related Docs

| Doc | What it covers |
|-----|---------------|
| [nv-attempt.md](nv-attempt.md) | Full investigation & build report (Phases 1-4) |
| [phase1.md](phase1.md) | Reverse-engineering nvgpu ioctls |
| [phase2.md](phase2.md) | Memory management via nvmap |
| [phase3.md](phase3.md) | Command submission (GPFIFO/QMD) |
| [phase4.md](phase4.md) | TegraIface integration into tinygrad |
| [Learning-Phase1.md](Learning-Phase1.md) | Methodology walkthrough (Phase 1) |
| [Learning-Phase2.md](Learning-Phase2.md) | Methodology walkthrough (Phase 2) |
| [Learning-Phase3.md](Learning-Phase3.md) | Methodology walkthrough (Phase 3) |
| [Learning-Phase4.md](Learning-Phase4.md) | Methodology walkthrough (Phase 4) |

## Key Code Locations

| What | File | Notes |
|------|------|-------|
| TegraIface class | `tinygrad/tinygrad/runtime/ops_nv.py` ~L575-1100 | The main backend under test |
| HCQ framework | `tinygrad/tinygrad/runtime/support/hcq.py` | Shared infra (signals, queues, allocator) |
| NVDevice | `tinygrad/tinygrad/runtime/ops_nv.py` ~L1100+ | Device init, GPFIFO setup |
| NVComputeQueue | `tinygrad/tinygrad/runtime/ops_nv.py` | QMD build, kernel exec |
| NVCopyQueue | `tinygrad/tinygrad/runtime/ops_nv.py` | DMA copy engine |
| PTX renderer | `tinygrad/tinygrad/renderer/ptx.py` | Shader code generation |
| NV autogen constants | `tinygrad/tinygrad/runtime/autogen/nv_570.py` | RM API constants, QMD field defs |
| L4T kernel headers | `l4t-sources/nvgpu/include/uapi/linux/` | ioctl struct definitions |

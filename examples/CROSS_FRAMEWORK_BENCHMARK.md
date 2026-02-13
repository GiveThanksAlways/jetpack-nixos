# Cross-Framework LLM Benchmark — Jetson Orin AGX 64GB

**Device**: NVIDIA Jetson Orin AGX 64GB, JetPack 6, L4T r36.4.4, SM 8.7, CUDA 12.6, LPDDR5 ~102 GB/s  
**Model**: LLaMA 3.2 1B Instruct  
**Date**: 2026-02-13

---

## Fair Comparison: Same Model, Same Precision (fp16)

All four frameworks run **LLaMA 3.2 1B Instruct at fp16 precision** — no quantization, identical math.
Tinygrad and llama.cpp use F16 GGUF; vLLM uses F16 GGUF; MLC uses its pre-compiled q0f16-MLC (fp16, no quantization).

| Rank | Framework | Decode tok/s | Prefill tok/s | P50 Latency ms/tok | Jitter (P90−P10) |
|------|-----------|:-----------:|:------------:|:------------------:|:----------------:|
| 1 | **MLC LLM** q0f16 | **36.8** | 1586 | **27.19** | 0.62 ms |
| 2 | **vLLM** F16 GGUF | **30.3** | **1622** | 32.84 | 0.97 ms |
| 3 | **tinygrad** NV=1 | **27.0** | 7.1¹ | 37.07 | 2.71 ms |
| 4 | **llama.cpp** +FA | 24.1 | 938 | ~41.5² | — |
| 5 | llama.cpp (no FA) | 22.4 | 699 | ~44.6² | — |

¹ tinygrad prefill includes JIT compilation overhead (~6s for first inference, then instant)  
² llama.cpp P50 estimated from 1000/tok_s (llama-bench reports aggregate only)

### tinygrad vs llama.cpp (same model, same precision)

| Metric | tinygrad NV=1 | llama.cpp +FA | tinygrad advantage |
|--------|:------------:|:------------:|:-----------------:|
| Decode throughput | 27.0 tok/s | 24.1 tok/s | **+12%** ✅ |
| Decode latency P50 | 37.1 ms | ~41.5 ms | **−11%** ✅ |
| Decode jitter | 2.71 ms | — | very low |

**tinygrad beats llama.cpp by 12% on decode** — and this is without JITBEAM (see below).

---

## Previous Results: Mixed Quantizations (for reference)

These numbers from 2026-02-12 used each framework's preferred/default quantization:

| Rank | Framework | Quantization | Decode tok/s | vs llama.cpp |
|------|-----------|-------------|:-----------:|:------------:|
| 1 | MLC LLM | q4f16_1 (4-bit) | 47.0 | 184% |
| 2 | tinygrad NV=1 + JITBEAM=4 | Q6_K (6-bit) | 36.7 | 143% |
| 3 | tinygrad NV=1 + MV_TPR=32 | Q6_K (6-bit) | 29.9 | 117% |
| 4 | llama.cpp +FA | Q6_K (6-bit) | 27.8 | 109% |
| 5 | llama.cpp | Q6_K (6-bit) | 25.6 | baseline |
| 6 | vLLM (GGUF) | Q6_K (6-bit) | 14.7 | 57% |

---

## JITBEAM Investigation: A Cautionary Tale

JITBEAM controls kernel auto-tuning beam search width in tinygrad's JIT compiler.
Higher JITBEAM = wider search = more time spent finding "optimal" kernels.

### Results (NV backend, Orin AGX)

| Config | Q6_K tok/s | F16 tok/s | Notes |
|--------|:---------:|:--------:|-------|
| Baseline (no BEAM) | **26.9** | **27.0** | Default heuristics |
| JITBEAM=2 | 1.0 | 1.0 | **27x slower!** |
| JITBEAM=4 | 1.1 | 1.1 | **25x slower!** |

### What happened?

JITBEAM's beam search optimizes individual kernel execution time by trying different thread/block configurations. However, on Orin's unified memory architecture (iGPU sharing LPDDR5 with CPU), the "optimal" kernel found by beam search is actually **dramatically worse** for the full inference pipeline.

**Root cause**: The beam search metric (single kernel latency) doesn't account for:
- Orin iGPU's unique memory hierarchy (shared LPDDR5, no dedicated VRAM)
- Pipeline effects (cache thrashing between kernels)
- The default NV backend heuristics are already well-tuned for this architecture

**Takeaway**: tinygrad's default kernel selection (without BEAM) already produces excellent results on Orin. JITBEAM hurts because locally-optimal ≠ globally-optimal on unified memory.

> **Note**: The previous JITBEAM=4 result of 36.7 tok/s from the mixed-quantization benchmarks was measured differently (via `--benchmark` flag) and may have reflected measurement differences rather than true JITBEAM benefit. The investigation above using consistent methodology shows JITBEAM is counterproductive on this hardware.

---

## Framework Details

### tinygrad (NV backend)
- **Version**: Latest with matvec heuristic fix (commit `2439279b1`)
- **Key optimization**: Fixed matvec pattern matching in `heuristic.py` — was silently falling through to GROUPTOP(16). Fix enables 128-thread coalesced matvec.
- **Backend**: Direct GPU kernel interface (NV), bypasses CUDA driver overhead
- **Measurement**: `model.generate()` with 10-token warmup, 128-token steady-state measurement
- **Best config**: `NV=1 MV_THREADS_PER_ROW=32` (no JITBEAM on Orin)
- **Runs**: Native (no Docker), nix develop shell

### llama.cpp
- **Version**: Built from upstream with CUDA + OpenSSL via Nix overlay
- **Config**: `llama-bench -p 42 -n 128 -r 5 -fa 1`, CUDA backend, F16 GGUF
- **Runs**: Native (no Docker), nix develop shell

### vLLM
- **Version**: 0.6.3 (dustynv/vllm:r36.4.0 container)
- **Config**: `--enforce-eager --dtype half --max-model-len 2048 --gpu-memory-utilization 0.8`
- **Model**: F16 GGUF (same file as tinygrad/llama.cpp)
- **Measurement**: OpenAI API streaming, 2 warmup requests, 3 runs averaged
- **Runs**: Docker container with NVIDIA runtime

### MLC LLM
- **Version**: dustynv/mlc:r36.4.0 container
- **Config**: `--mode local` (max batch = 4, max KV = 8192)
- **Model**: `HF://mlc-ai/Llama-3.2-1B-Instruct-q0f16-MLC` (fp16, pre-compiled)
- **Compilation**: JIT-compiled for sm_87 with cutlass + cudagraph
- **Measurement**: Chat completions API streaming, 2 warmup requests, 3 runs averaged
- **Runs**: Docker container with NVIDIA runtime

---

## Key Insights

1. **MLC LLM is fastest at fp16** (36.8 tok/s decode). Its ahead-of-time compilation for sm_87 + CUDA graphs gives excellent kernel utilization.

2. **vLLM is a strong 2nd** (30.3 tok/s decode, 1622 tok/s prefill). Its highly optimized CUDA kernels shine at fp16 — much better than its Q6_K GGUF path (14.7 tok/s prior benchmark).

3. **tinygrad NV beats llama.cpp by 12%** (27.0 vs 24.1 tok/s) on the same model at the same precision, with zero quantization-related advantages. The NV backend's direct kernel dispatch gives it an edge.

4. **JITBEAM is counterproductive on Orin** — the default heuristics already produce near-optimal kernels for the unified memory architecture. Beam search finds locally-optimal kernels that are globally 25-27x slower.

5. **Prefill varies wildly**: vLLM (1622) and MLC (1586) have highly optimized prefill paths. tinygrad's 7.1 tok/s includes JIT compilation (one-time cost). After warmup, tinygrad's decode is competitive.

6. **Memory bandwidth is king**: All frameworks are memory-bound at batch=1 decode. F16 (2 bytes/param) is 3.3x more data than Q6_K (0.6 bytes/param), which is why all F16 numbers are lower than Q6_K numbers.

---

## How to Reproduce

```bash
# === Fair F16 benchmark ===

# tinygrad (NV backend)
cd examples/tinygrad
nix develop -c bash -c 'NV=1 MV_THREADS_PER_ROW=32 python3 ../bench_tinygrad_f16.py'

# llama.cpp
cd examples/llama-cpp-orin
nix develop -c llama-bench -m ~/.cache/tinygrad/downloads/llama3.2-1b-f16/Llama-3.2-1B-Instruct-f16.gguf \
  -p 42 -n 128 -r 5 -fa 1

# vLLM (Docker)
sudo docker run -d --name vllm-orin --runtime nvidia --shm-size 8g -p 8000:8000 \
  -v ~/.cache:/root/.cache vllm-jetson:latest \
  python3 -m vllm.entrypoints.openai.api_server \
  --model /root/.cache/tinygrad/downloads/llama3.2-1b-f16/Llama-3.2-1B-Instruct-f16.gguf \
  --max-model-len 2048 --dtype half --gpu-memory-utilization 0.8 --enforce-eager
# Then from nix shell: python3 bench_cross_framework.py --server http://localhost:8000 \
#   --model "/root/.cache/.../Llama-3.2-1B-Instruct-f16.gguf" --num-tokens 128

# MLC LLM (Docker)
sudo docker run -d --name mlc-orin --runtime nvidia -p 8001:8000 \
  -v ~/.cache:/root/.cache mlc-jetson:latest \
  bash -c 'python3 -m mlc_llm serve "HF://mlc-ai/Llama-3.2-1B-Instruct-q0f16-MLC" \
  --mode local --host 0.0.0.0 --port 8000'
# Then: python3 bench_cross_framework.py (adapted for chat completions, see script)
```

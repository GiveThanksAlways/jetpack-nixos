# Cross-Framework LLM Benchmark — Jetson Orin AGX 64GB

**Device**: NVIDIA Jetson Orin AGX 64GB, JetPack 6, L4T r36.4.4, SM 8.7, CUDA 12.6, LPDDR5 ~102 GB/s  
**Model**: LLaMA 3.2 1B Instruct  
**Metric**: Steady-state decode tokens/second (batch=1, autoregressive)  
**Date**: 2026-02-12

## Results — LLaMA 3.2 1B Instruct Decode

| Rank | Framework | Quantization | tok/s | vs llama.cpp | Notes |
|------|-----------|-------------|-------|-------------|-------|
| 1 | **MLC LLM** | q4f16_1 (4-bit) | **47.0** | **184%** | JIT-compiled for sm_87, CUDA graphs |
| 2 | **tinygrad NV=1 + JITBEAM=4** | Q6_K (6-bit) | **36.7** | **143%** | Matvec heuristic fix + beam search |
| 3 | tinygrad CUDA=1 + JITBEAM=4 | Q6_K (6-bit) | 31.8 | 124% | Same fix, CUDA backend |
| 4 | tinygrad NV=1 + MV_TPR=32 | Q6_K (6-bit) | 29.9 | 117% | Without beam search |
| 5 | llama.cpp +FA | Q6_K (6-bit) | 27.8 | 109% | Flash Attention enabled |
| 6 | **llama.cpp** | Q6_K (6-bit) | **25.6** | **baseline** | llama-bench tg128 |
| 7 | vLLM (fp16 Qwen 1.5B) | fp16 (16-bit) | 21.1 | 82% | Different model, full precision |
| 8 | vLLM (GGUF) | Q6_K (6-bit) | 14.7 | 57% | "gguf quantization not fully optimized" |

### Apples-to-Apples: Same Quantization (Q6_K, 6-bit)

| Framework | tok/s | vs llama.cpp |
|-----------|-------|-------------|
| **tinygrad NV=1 + JITBEAM=4** | **36.7** | **+43%** ✅ |
| tinygrad CUDA=1 + JITBEAM=4 | 31.8 | +24% |
| tinygrad NV=1 + MV_TPR=32 | 29.9 | +17% |
| llama.cpp +FA | 27.8 | +9% |
| llama.cpp | 25.6 | baseline |
| vLLM (GGUF) | 14.7 | -43% |

**Winner (same quant)**: tinygrad with NV backend + matvec fix + JITBEAM=4, at **36.7 tok/s** — 43% faster than llama.cpp.

### All Frameworks Including Different Quantizations

MLC LLM achieves 47.0 tok/s but uses 4-bit quantization (q4f16_1), which requires ~40% less memory bandwidth than Q6_K. A fairer comparison would need MLC with Q6_K or tinygrad with Q4, which is not currently available.

## Framework Details

### tinygrad (NV backend)
- **Version**: Latest with matvec heuristic fix (commit `2439279b1`)
- **Key optimization**: Fixed matvec pattern matching in `heuristic.py` — was silently falling through to GROUPTOP(16) (half a warp, non-coalesced). Fix enables 128-thread coalesced matvec with GROUP reduction.
- **JITBEAM=4**: Kernel auto-tuning via beam search. NV=1 benefits much more than CUDA=1 (+23% vs +6%) because lower dispatch overhead amplifies kernel optimization.
- **Backend**: Direct GPU kernel interface (NV), bypasses CUDA driver overhead.
- **Runs**: Native (no Docker), nix develop shell, GGUF Q6_K format.

### llama.cpp
- **Version**: Built from upstream with CUDA + OpenSSL via Nix overlay
- **Config**: llama-bench tg128, CUDA backend, Q6_K GGUF
- **FA**: Flash Attention enabled via `-fa 1`
- **Runs**: Native (no Docker), nix develop shell

### vLLM
- **Version**: 0.6.3 (dustynv/vllm:r36.4.0 container)
- **Config**: `--enforce-eager --dtype half --max-model-len 2048 --gpu-memory-utilization 0.8`
- **GGUF warning**: "gguf quantization is not fully optimized yet. The speed can be slower than non-quantized models."
- **fp16 result**: With Qwen2.5-1.5B-Instruct (fp16, no quantization), achieves 21.1 tok/s — better than its own GGUF path but slower than optimized quantized approaches.
- **Runs**: Docker container with NVIDIA runtime

### MLC LLM
- **Version**: dustynv/mlc:r36.4.0 container
- **Config**: `--mode local` (max batch = 4, max KV = 8192)
- **Compilation**: JIT-compiled for sm_87 (Orin) with cutlass + cudagraph
- **Quantization**: q4f16_1 (4-bit weights, fp16 arithmetic, group_size=32)
- **Memory**: 663 MB params + 386 MB KV cache + 1480 MB temp = 2529 MB total
- **Runs**: Docker container with NVIDIA runtime

## Key Insights

1. **tinygrad NV beats llama.cpp by 43%** on the same model (LLaMA 1B Q6_K) with our matvec fix + JITBEAM=4. This validates the NV backend optimization campaign.

2. **MLC LLM is fastest overall** at 47 tok/s, but uses 4-bit quantization (less memory to read per token). Its JIT compilation for sm_87 + CUDA graphs gives it excellent kernel utilization.

3. **vLLM's GGUF path is slow** (14.7 tok/s) — vLLM warns it's not optimized. vLLM is designed for high-throughput server workloads (batched requests), not single-stream decode. Its fp16 path is reasonable at 21.1 tok/s.

4. **NV backend advantage**: tinygrad NV=1 consistently outperforms CUDA=1 on Orin, especially with JITBEAM (36.7 vs 31.8 tok/s). The direct kernel interface reduces dispatch latency, which beam search then amplifies.

5. **Quantization matters**: The spread from 4-bit (47 tok/s) to fp16 (21 tok/s) shows that memory bandwidth is the primary bottleneck on Orin AGX for batch=1 decode. Lower-precision quantization directly translates to higher throughput.

## How to Reproduce

```bash
# tinygrad (NV backend, requires matvec fix)
cd examples/tinygrad
nix develop -c bash -c '
  NV=1 MV_THREADS_PER_ROW=32 JITBEAM=4 python3 tinygrad/examples/llama3.py \
    --model llama3.2:1b --count 128 --prompt "Write a detailed explanation of how neural networks work"
'

# llama.cpp
cd examples/llama-cpp-orin-nix-overlay
nix develop -c llama-bench -m ~/.cache/tinygrad/downloads/llama3-1b-instruct/Llama-3.2-1B-Instruct-Q6_K.gguf -p 128 -n 128

# vLLM (Docker, GGUF)
sudo nixos-rebuild switch --flake examples/nixos#nixos-docker-bench
sudo docker run -d --name vllm-orin --runtime nvidia --shm-size 8g -p 8000:8000 \
  -v ~/.cache:/root/.cache dustynv/vllm:r36.4.0 \
  python3 -m vllm.entrypoints.openai.api_server \
  --model /root/.cache/tinygrad/downloads/llama3-1b-instruct/Llama-3.2-1B-Instruct-Q6_K.gguf \
  --max-model-len 2048 --dtype half --gpu-memory-utilization 0.8 --enforce-eager
# Then: python3 examples/vllm/bench_vllm.py --server http://localhost:8000 \
#   --model "/root/.cache/..." --num-tokens 128

# MLC LLM (Docker)
sudo docker run -d --name mlc-orin --runtime nvidia --shm-size 8g -p 8001:8000 \
  -v ~/.cache:/root/.cache dustynv/mlc:r36.4.0 \
  bash -c 'python3 -m mlc_llm serve "HF://mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC" --mode local --host 0.0.0.0 --port 8000'
# Then: python3 examples/vllm/bench_vllm.py --server http://localhost:8001 \
#   --model "HF://mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC" --num-tokens 128
```

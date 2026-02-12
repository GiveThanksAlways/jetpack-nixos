# vLLM on Jetson Orin AGX

High-throughput LLM inference engine. Supports native install (if available) and Docker.

## Quick Start (Docker — Recommended)

vLLM native builds often fail on NixOS/Jetson due to complex C++ dependencies.
Docker provides a reliable path:

```bash
cd /home/agent/jetpack-nixos/examples/vllm

# Start vLLM server (downloads model automatically)
./run-vllm-docker.sh

# Or with a specific model:
./run-vllm-docker.sh meta-llama/Llama-3.2-3B-Instruct 8000

# Benchmark against running server
python3 bench_vllm.py --server http://localhost:8000 --model meta-llama/Llama-3.2-1B-Instruct

# Stop
docker stop vllm-orin
```

## Quick Start (Native — If It Works)

```bash
cd /home/agent/jetpack-nixos/examples/vllm
nix develop

# If pip install succeeds:
python3 -c 'import vllm; print(vllm.__version__)'

# Start server
python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --max-model-len 2048 --dtype half

# Benchmark
python3 bench_vllm.py
```

## Architecture

vLLM serves models via an OpenAI-compatible API:
```
Client (bench_vllm.py) → HTTP → vLLM Server → CUDA → GPU
```

The benchmark script measures:
- **TTFT** (Time to First Token): Prefill latency
- **Decode throughput**: Tokens per second (streaming)
- **Total throughput**: End-to-end tokens per second

## Available Models

| Model | Quantization | Size | Expected tok/s |
|-------|-------------|------|----------------|
| meta-llama/Llama-3.2-1B-Instruct | fp16 | ~2.4 GB | ~35-45 |
| meta-llama/Llama-3.2-3B-Instruct | fp16 | ~6.0 GB | ~15-25 |
| meta-llama/Meta-Llama-3.1-8B-Instruct | fp16 | ~16 GB | ~5-12 |
| Qwen/Qwen3-0.6B | fp16 | ~1.2 GB | ~45-60 |

## Benchmarking

```bash
# Default (LLaMA 1B, 25 tokens)
python3 bench_vllm.py

# Custom model
python3 bench_vllm.py --model meta-llama/Llama-3.2-3B-Instruct --num-tokens 50

# Different server
python3 bench_vllm.py --server http://localhost:8001 --model Qwen/Qwen3-0.6B
```

## Docker vs Native Performance

Docker on Jetson has minimal overhead (~1-3% throughput reduction):
- GPU compute: Zero overhead (direct CUDA passthrough via --runtime nvidia)
- Memory: Shared with host (LPDDR5, same bandwidth)
- CPU: Negligible container overhead
- I/O: Docker overlay filesystem adds ~1ms per model load (one-time)

For benchmarking purposes, Docker results are directly comparable to native.

## Comparison Matrix

| Engine | Backend | Batch=1 Dec | Batch>1 | Quantization |
|--------|---------|-------------|---------|--------------|
| tinygrad NV | Tegra/HCQ | ~37 tok/s | TBD | GGUF Q6_K |
| tinygrad CUDA | cuLaunchKernel | ~32 tok/s | TBD | GGUF Q6_K |
| llama.cpp | CUDA | ~26 tok/s | TBD | GGUF Q6_K |
| vLLM | CUDA/PagedAttn | ~35 tok/s* | Yes | fp16/AWQ/GPTQ |
| MLC LLM | TVM/CUDA | TBD | TBD | q4f16/q4f32 |

*Estimated, to be measured.

## Troubleshooting

**Docker build fails:**
Try pulling a pre-built Jetson image:
```bash
docker pull dustynv/vllm:0.6.6-r36.4.0
```

**Out of memory:**
Reduce `--gpu-memory-utilization` (default 0.8):
```bash
./run-vllm-docker.sh  # Edit the script to lower to 0.6
```

**Model download slow:**
Mount HF cache: The Docker script already mounts `~/.cache/huggingface`.

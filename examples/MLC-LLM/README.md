# MLC LLM on Jetson Orin AGX

Native CUDA inference engine for benchmarking against tinygrad and llama.cpp.

## Quick Start

```bash
cd /home/agent/jetpack-nixos/examples/MLC-LLM
nix develop

# MLC LLM should auto-install in the venv on first entry
# Test it:
python3 -c 'import mlc_llm; print(mlc_llm.__version__)'

# Chat with a model
mlc_llm chat HF://mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC

# Benchmark
python3 bench_mlc_llm.py
```

## Available Models

MLC LLM provides pre-compiled models on HuggingFace:

| Model | MLC Path | Size |
|-------|----------|------|
| LLaMA 3.2 1B (q4f16) | `HF://mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC` | ~0.7 GB |
| LLaMA 3.2 3B (q4f16) | `HF://mlc-ai/Llama-3.2-3B-Instruct-q4f16_1-MLC` | ~1.8 GB |
| Qwen 2.5 0.5B (q4f16) | `HF://mlc-ai/Qwen2.5-0.5B-Instruct-q4f16_1-MLC` | ~0.4 GB |
| Qwen 2.5 1.5B (q4f16) | `HF://mlc-ai/Qwen2.5-1.5B-Instruct-q4f16_1-MLC` | ~1.0 GB |

## Benchmarking

```bash
# LLaMA 1B
python3 bench_mlc_llm.py HF://mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC 25

# LLaMA 3B
python3 bench_mlc_llm.py HF://mlc-ai/Llama-3.2-3B-Instruct-q4f16_1-MLC 25

# Custom prompt + token count
python3 bench_mlc_llm.py HF://mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC 50
```

## Comparison with tinygrad

MLC LLM uses TVM-compiled CUDA kernels, while tinygrad uses JIT-compiled PTX.
Key differences:
- MLC LLM: Pre-compiled model-specific kernels (fast first-token, no JIT overhead)
- tinygrad: JIT-compiled kernels with BEAM search optimization (more flexible)
- Both: CUDA backend, same hardware (Orin AGX 64GB)

## Troubleshooting

**pip install fails:**
```bash
# Try nightly wheels directly
pip install mlc-ai-nightly -f https://mlc.ai/wheels
```

**CUDA not found:**
Make sure you're in the nix develop shell which sets LD_LIBRARY_PATH.

**Model download fails:**
Check internet connection. Models are cached in `~/.cache/` after first download.

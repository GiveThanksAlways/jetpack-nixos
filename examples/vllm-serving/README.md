# vLLM + GGUF Serving on Jetson Orin AGX

> TL;DR: Serve quantized GGUF models (Q3/Q4/Q5) fast on Orin AGX via vLLM, llama-cpp-server, or TabbyAPI — all NixOS-native.

## Quick Start

```bash
# 1. Add this to your NixOS config imports
imports = [ ./examples/vllm-serving/module.nix ];

# 2. Enable your preferred backend
services.vllm-serving.enable = true;          # vLLM (best throughput)
# OR
services.llama-cpp-server.enable = true;      # llama.cpp (lowest overhead for GGUF)
# OR
services.tabby-api.enable = true;             # TabbyAPI (OpenAI-compatible, good for OpenCode)

# 3. Point to your GGUF model
services.vllm-serving.model = "/models/Qwen3-Coder-Next-Q4_K_M.gguf";

# 4. Rebuild
sudo nixos-rebuild switch
```

## Which Backend?

| Backend | Best For | GGUF? | OpenAI API? | Throughput |
|---------|----------|-------|-------------|------------|
| **vLLM** | Batch inference, high throughput | ✅ (via `--tokenizer`) | ✅ | ★★★★★ |
| **llama.cpp** | Single-user, low RAM, GGUF-native | ✅ native | ✅ | ★★★☆☆ |
| **TabbyAPI** | Code completion, OpenCode/IDE | ✅ (via ExLlamaV2) | ✅ | ★★★★☆ |

## Performance Tips (Orin AGX 64GB)

- **Use Q4_K_M** — best quality/speed tradeoff on Orin
- **Set `gpu-memory-utilization: 0.90`** — leave 10% for system
- **Enable JetPack power mode 3 (MAXN)**: `sudo nvpmodel -m 0 && sudo jetson_clocks`
- **Pin all layers to GPU** (`-ngl 99`) — Orin unified memory makes this viable
- **Use mmap** for faster model loading: `--use-mmap`
- **Flash Attention** is auto-enabled in vLLM on Orin (SM 8.7)

## File Layout

```
examples/vllm-serving/
├── README.md              # This file
├── flake.nix              # Standalone flake (nix develop / nix build)
├── module.nix             # NixOS module: services.vllm-serving
├── llama-cpp-server.nix   # NixOS module: services.llama-cpp-server
├── tabby-api.nix          # NixOS module: services.tabby-api
├── performance.nix        # Orin AGX perf tuning (power, clocks, memory)
└── configuration.nix      # Example full NixOS config
```

## Connecting to OpenCode

All backends expose an OpenAI-compatible endpoint. Point OpenCode at:

```
http://localhost:8000/v1    # vLLM
http://localhost:8080/v1    # llama.cpp
http://localhost:5000/v1    # TabbyAPI
```

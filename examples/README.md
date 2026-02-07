# examples

Standalone Nix flakes and NixOS modules for Jetson Orin AGX.

Each subfolder is a self-contained flake. Pick the one that fits your use case.

## Layout

| Folder | What it does |
|---|---|
| `nixos/` | Base NixOS system config with composable modules (performance, llama.cpp, TabbyAPI, telemetry). Pick a configuration at build time. |
| `llama-cpp-orin/` | Minimal flake -- builds llama.cpp from upstream with CUDA/OpenSSL for Orin. Dev shell only. |
| `llama-cpp-orin-nix-overlay/` | Full-featured overlay with wrapper scripts (`qwen3-coder`, `qwen3-server`, `llama-benchmark`). Reusable via `overlays.default`. |
| `vLLM/` | vLLM dev shell + NixOS module (`services.vllm-serving`). Import the module into your system flake for production serving. |

## Quick reference

```bash
# NixOS system build (pick a config)
nixos-rebuild switch --flake ./nixos#nixos
nixos-rebuild switch --flake ./nixos#nixos-perf
nixos-rebuild switch --flake ./nixos#nixos-llama-cpp

# llama.cpp dev shell (simple)
cd llama-cpp-orin && nix develop

# llama.cpp dev shell (overlay + wrappers)
cd llama-cpp-orin-nix-overlay && nix develop

# vLLM dev shell
cd vLLM && nix develop
```

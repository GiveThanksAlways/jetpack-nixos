# llama.cpp on Jetson Orin AGX

## Quick Start

```bash
# Enter dev shell (downloads llama.cpp with CUDA)
nix develop

# Run Qwen3-Coder-Next (auto-downloads model)
llama-cli -hf unsloth/Qwen3-Coder-Next-GGUF:Q5_K_XL
```

## NixOS Installation

Copy `flake.nix` to your Orin AGX, then:

```bash
sudo nixos-rebuild switch --flake .#orin-agx-llama
```

Edit `flake.nix` to match your filesystem/bootloader setup.

## Model: Qwen3-Coder-Next Q5_K_XL

- Size: ~57GB
- Quant: Dynamic 2.0 (unsloth)
- Source: https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF

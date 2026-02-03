# llama.cpp Advanced - Jetson Orin AGX

## Quick Start

```bash
nix develop
qwen3-coder  # Start chatting
```

## Commands

| Command | Description |
|---------|-------------|
| `qwen3-coder` | Interactive chat |
| `qwen3-coder-server` | API server on :8080 |
| `llama-benchmark` | Performance test |

## NixOS with Systemd Service

```nix
# In your configuration.nix
services.llama-cpp = {
  enable = true;
  model = "unsloth/Qwen3-Coder-Next-GGUF:Q5_K_XL";
  port = 8080;
};
```

## Using the Overlay

```nix
# In another flake
{
  inputs.llama-orin.url = "path:./examples/llama-cpp-orin-advanced";

  outputs = { self, nixpkgs, llama-orin, ... }: {
    # Use llama-orin.overlays.default
  };
}
```

## Quant Options

`Q4_K_M` (fastest) → `Q5_K_M` → `Q5_K_XL` (recommended) → `Q6_K` → `Q8_0` (highest quality)

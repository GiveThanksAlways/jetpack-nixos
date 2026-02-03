# llama.cpp Advanced - Jetson Orin AGX

## Prerequisites (Global NixOS Setup)

Your Orin AGX must have jetpack-nixos configured globally in `/etc/nixos/`.

**`/etc/nixos/flake.nix`:**
```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    jetpack-nixos.url = "github:anduril/jetpack-nixos/master";
    jetpack-nixos.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { nixpkgs, jetpack-nixos, ... }: {
    nixosConfigurations.orin = nixpkgs.lib.nixosSystem {
      system = "aarch64-linux";
      modules = [
        jetpack-nixos.nixosModules.default
        ./configuration.nix
      ];
    };
  };
}
```

**`/etc/nixos/configuration.nix`:**
```nix
{
  hardware.nvidia-jetpack = {
    enable = true;
    som = "orin-agx";
    carrierBoard = "devkit";
  };
  hardware.graphics.enable = true;
  nixpkgs.config.allowUnfree = true;
}
```

Apply: `sudo nixos-rebuild switch --flake /etc/nixos#orin`

---

## Quick Start (Local Dev Shell)

```bash
cd /path/to/this/flake
nix develop

# Just type this to start chatting!
qwen3-coder
```

## Commands

| Command | Description |
|---------|-------------|
| `qwen3-coder` | Interactive chat (Q5_K_XL, ~57GB) |
| `qwen3-coder Q4_K_M` | Use smaller quant (~35GB) |
| `qwen3-server` | Start API server on :8080 |
| `qwen3-server Q4_K_M 3000` | Server with custom quant/port |
| `llama-benchmark` | Run performance benchmark |

## API Server Usage

```bash
# Start server
qwen3-server

# In another terminal, test it:
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Write hello world in Python"}]}'
```

## Using the Overlay in Other Flakes

```nix
{
  inputs.llama-orin.url = "path:./examples/llama-cpp-orin-advanced";

  outputs = { nixpkgs, llama-orin, ... }: {
    # Apply overlay to get: llama-cpp-orin, qwen3-coder, qwen3-server
    packages.aarch64-linux = let
      pkgs = import nixpkgs {
        system = "aarch64-linux";
        overlays = [ llama-orin.overlays.default ];
      };
    in {
      inherit (pkgs) llama-cpp-orin qwen3-coder;
    };
  };
}
```

## Model Options

| Quant | Size | Memory | Quality |
|-------|------|--------|---------|
| Q4_K_M | ~35GB | ~40GB | Good |
| Q5_K_M | ~45GB | ~50GB | Better |
| Q5_K_XL | ~57GB | ~64GB | Best (default) |

## Troubleshooting

**Out of VRAM:** Use smaller quant: `qwen3-coder Q4_K_M`

**CUDA not working:** Run `nvidia-smi` - if it fails, check global NixOS config

**Model download slow:** Downloads to `~/.cache/huggingface/` - ~57GB for Q5_K_XL

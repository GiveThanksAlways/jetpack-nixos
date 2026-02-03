# llama.cpp on Jetson Orin AGX

## Prerequisites (Global NixOS)

Your `/etc/nixos/flake.nix` must include jetpack-nixos:

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

Your `/etc/nixos/configuration.nix` needs:

```nix
{
  hardware.nvidia-jetpack = {
    enable = true;
    som = "orin-agx";
    carrierBoard = "devkit";
  };
  hardware.graphics.enable = true;
}
```

Apply with: `sudo nixos-rebuild switch --flake /etc/nixos#orin`

---

## Quick Start (Local Dev Shell)

```bash
# On your Jetson (after global NixOS is configured)
cd /path/to/this/flake
nix develop

# Run Qwen3-Coder-Next (~57GB, downloads automatically)
llama-cli -hf unsloth/Qwen3-Coder-Next-GGUF:Q5_K_XL --gpu-layers 999
```

## Model Options

| Quant | Size | Command |
|-------|------|---------|
| Q4_K_M | ~35GB | `llama-cli -hf unsloth/Qwen3-Coder-Next-GGUF:Q4_K_M -ngl 999` |
| Q5_K_M | ~45GB | `llama-cli -hf unsloth/Qwen3-Coder-Next-GGUF:Q5_K_M -ngl 999` |
| Q5_K_XL | ~57GB | `llama-cli -hf unsloth/Qwen3-Coder-Next-GGUF:Q5_K_XL -ngl 999` |

## API Server

```bash
llama-server -hf unsloth/Qwen3-Coder-Next-GGUF:Q5_K_XL --gpu-layers 999 --host 0.0.0.0 --port 8080
```

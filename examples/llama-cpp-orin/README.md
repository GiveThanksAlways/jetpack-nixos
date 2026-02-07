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

```bash
sudo nixos-rebuild switch --flake /etc/nixos#nixos

sudo nixos-rebuild switch --flake /etc/nixos#nixos-static-ip

# firewall default blocked everything in configuration.nix
nc -l 5000
nc 192.168.0.131 5000
```

---

## Quick Start (Local Dev Shell)

```bash
# On your Jetson (after global NixOS is configured)
cd /path/to/this/flake
nix develop  # Builds latest llama.cpp from source with CUDA support

# Run a supported model (llama.cpp now has qwen3next support)
llama-cli -hf unsloth/Qwen3-Coder-Next-GGUF:Q5_K_XL --gpu-layers 999

# Or use other models
llama-cli -hf mistralai/Mistral-7B-Instruct-v0.2 --gpu-layers 999
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

# note: NixOS default had ip tables blocking everything. so open up port 5000

llama-cli -m ~/.cache/llama.cpp/unsloth_Qwen3-Coder-Next-GGUF_UD-Q5_K_XL_Qwen3-Coder-Next-UD-Q5_K_XL-00001-of-00002.gguf --gpu-layers 999

llama-cli -m ~/.cache/llama.cpp/unsloth_Qwen3-Coder-Next-GGUF_Qwen3-Coder-Next-MXFP4_MOE.gguf --gpu-layers 999

llama-cli -m ~/.cache/llama.cpp/unsloth_Qwen3-Coder-Next-GGUF_Qwen3-Coder-Next-MXFP4_MOE.gguf --gpu-layers 999 --flash-attn on

llama-server -m ~/.cache/llama.cpp/unsloth_Qwen3-Coder-Next-GGUF_Qwen3-Coder-Next-MXFP4_MOE.gguf --gpu-layers 999 --host 0.0.0.0 --port 5000 -c 65536

llama-server -m ~/.cache/llama.cpp/unsloth_Qwen3-Coder-Next-GGUF_Qwen3-Coder-Next-MXFP4_MOE.gguf --gpu-layers 999 --host 0.0.0.0 --port 5000

llama-server -m ~/.cache/llama.cpp/unsloth_Qwen3-Coder-Next-GGUF_Qwen3-Coder-Next-MXFP4_MOE.gguf --gpu-layers 999 --host 0.0.0.0 --port 5000 --flash-attn on

llama-server -m ~/.cache/llama.cpp/unsloth_Qwen3-Coder-Next-GGUF_Qwen3-Coder-Next-MXFP4_MOE.gguf --gpu-layers 999 --host 0.0.0.0 --port 5000 --ctx-size 174080 --flash-attn on
# For 128K sweet spot:
llama-server -m ~/.cache/llama.cpp/unsloth_Qwen3-Coder-Next-GGUF_Qwen3-Coder-Next-MXFP4_MOE.gguf --gpu-layers 999 --host 0.0.0.0 --port 5000 --ctx-size 131072 --flash-attn on

llama-server -m ~/.cache/llama.cpp/unsloth_Qwen3-Coder-Next-GGUF_Qwen3-Coder-Next-MXFP4_MOE.gguf \
  --fit on \
  --host 0.0.0.0 \
  --port 5000 \
  --ctx-size 131072 \
  --flash-attn on

llama-cli -m ~/.cache/llama.cpp/unsloth_Qwen3-Coder-Next-GGUF_Qwen3-Coder-Next-MXFP4_MOE.gguf \
  --fit on \
  --ctx-size 131072 \
  --flash-attn on

llama-cli -m ~/.cache/llama.cpp/unsloth_GLM-4.7-Flash-GGUF_GLM-4.7-Flash-UD-IQ3_XXS.gguf \
  --fit on \
  --ctx-size 131072 \
  --flash-attn on



# fastest
llama-server -m ~/.cache/llama.cpp/unsloth_Qwen3-Coder-Next-GGUF_Qwen3-Coder-Next-MXFP4_MOE.gguf \
  --fit on \
  --host 0.0.0.0 \
  --port 5000 \
  --ctx-size 131072 \
  --flash-attn on

llama-server -m ~/.cache/llama.cpp/unsloth_Qwen3-Coder-Next-GGUF_Qwen3-Coder-Next-MXFP4_MOE.gguf \
  --fit on \
  --host 0.0.0.0 \
  --port 5000 \
  --ctx-size 131072 \
  --flash-attn on \
  --parallel 1 \
  --threads 1 \
  --threads-http 1 \
  --no-cont-batching \
  --no-slots \
  --no-webui

# this did task.n_tokens = 43275 in 6 mins and 9 seconds so about 117 toks/sec input (same tok/s out)
llama-server -m ~/.cache/llama.cpp/unsloth_Qwen3-Coder-Next-GGUF_Qwen3-Coder-Next-MXFP4_MOE.gguf \
  --gpu-layers 99 \
  --host 0.0.0.0 --port 5000 \
  --ctx-size 131072 \
  --flash-attn on \
  --parallel 1 \
  --no-cont-batching \
  --no-webui \
  --prio 2

### crazy try (still avereege like 70 ish tok/s input but with full ctx window)
llama-server -m ~/.cache/llama.cpp/unsloth_Qwen3-Coder-Next-GGUF_Qwen3-Coder-Next-MXFP4_MOE.gguf \
  --gpu-layers all \
  --host 0.0.0.0 --port 5000 \
  --flash-attn on \
  --threads 8 \
  --threads-batch 8 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --cpu-strict 1 \
  --cpu-strict-batch 1 \
  --prio 2 \
  --prio-batch 2 \
  --poll 100 \
  --poll-batch 1 \
  --parallel 1 \
  --no-cont-batching \
  --no-webui \
  --no-slots
```

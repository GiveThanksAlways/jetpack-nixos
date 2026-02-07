# vLLM on Jetson Orin AGX

Standalone flake for running [vLLM](https://github.com/vllm-project/vllm) on
Jetson Orin AGX with CUDA support.

## Dev Shell

```bash
nix develop .
vllm serve <model>
```

## NixOS Module

This flake exports a NixOS module at `nixosModules.default`.  Import it from
your main system flake to get `services.vllm-serving`:

```nix
# In your system flake.nix inputs:
vllm.url = "path:../vLLM";   # or point at a remote repo

# In modules:
modules = [
  vllm.nixosModules.default
  ({ ... }: {
    services.vllm-serving = {
      enable = true;
      model = "/models/Qwen3-Coder-Next-Q4_K_M.gguf";
      tokenizer = "Qwen/Qwen3-Coder-Next";
      quantization = "gguf";
      gpuMemoryUtilization = 0.90;
      maxModelLen = 4096;
    };
  })
];
```

## Options

| Option | Type | Default | Description |
|---|---|---|---|
| `enable` | bool | `false` | Enable vLLM serving |
| `model` | string | — | Path to GGUF or HuggingFace model ID |
| `tokenizer` | string | `""` | HuggingFace tokenizer (needed for GGUF) |
| `host` | string | `"0.0.0.0"` | Bind address |
| `port` | port | `8000` | API port |
| `gpuMemoryUtilization` | float | `0.90` | GPU memory fraction |
| `maxModelLen` | int | `4096` | Max sequence length |
| `quantization` | string | `null` | Quantization method (gguf, awq, gptq) |
| `extraArgs` | list | `[]` | Extra CLI flags |
| `environment` | attrs | `{}` | Extra env vars |

The API is OpenAI-compatible at `http://<host>:<port>/v1`.

# LLM on Jetson Orin AGX

## TL;DR - Run LLMs NOW

**On your Jetson (via UART/SSH):**

```bash
# 1. Clone repo directly on Jetson
git clone git@github.com:GiveThanksAlways/jetpack-nixos.git
cd jetpack-nixos/llm

# 2. Enter dev shell (downloads deps via nix)
nix develop

# 3a. TinyGrad + Llama 3.2 (smaller, fast to start)
pip install -e ../vendor/tinygrad
CUDA=1 python ../vendor/tinygrad/examples/llama3.py --download_model --size 1B --no_api

# 3b. GLM-4.7-Flash (bigger, smarter)
mkdir -p ~/.cache/glm
curl -L -o ~/.cache/glm/GLM-4.7-Flash-Q4_K_M.gguf \
  "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf"
# Then: glm-chat (once llama.cpp builds)
```

## Workflow

**Best approach:** Clone repo directly on Jetson, work there.

```
[Your PC] --UART/SSH--> [Jetson with NixOS]
                              |
                              +-- git clone jetpack-nixos
                              +-- cd llm && nix develop
                              +-- run models
```

The Jetson downloads everything itself (deps via Nix, models via curl).

## Models

| Model | Download | VRAM | Command |
|-------|----------|------|---------|
| Llama 3.2 1B | Auto (~700MB) | ~1GB | `python llama3.py --download_model --size 1B` |
| Llama 3.2 8B | Auto (~4GB) | ~8GB | `python llama3.py --download_model --size 8B` |
| GLM-4.7-Flash Q4 | Manual (~17GB) | ~17GB | `glm-chat` |

## Quick Reference

```bash
# TinyGrad Llama (after pip install)
CUDA=1 python ../vendor/tinygrad/examples/llama3.py \
  --download_model --size 1B --quantize int8 --no_api

# GLM (after model download)
llama-cli -m ~/.cache/glm/GLM-4.7-Flash-Q4_K_M.gguf \
  -ngl 999 -c 4096 --interactive

# Check GPU
nvidia-smi
```

## Files

```
llm/
├── flake.nix       # Main flake (nix develop)
├── tinygrad/       # TinyGrad-specific flakes  
└── glm/            # GLM-specific flake
```

## Troubleshooting

**CUDA not working:** `nvidia-smi` should show your GPU. If not, check jetpack-nixos config.

**Out of memory:** Use `--size 1B` or `Q4_K_M` quantization.

**Download interrupted:** `curl -L -C -` resumes partial downloads.

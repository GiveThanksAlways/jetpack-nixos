# LLM Inference on Jetson Orin AGX

Run large language models on your NVIDIA Jetson Orin AGX 64GB using NixOS.

## Quick Start

```bash
# Enter the development environment
cd llm
nix develop

# Option A: TinyGrad + Llama 3.2
tinygrad-llama 1B int8

# Option B: GLM-4.7-Flash
glm-download Q4_K_M
glm-chat
```

## Available Approaches

### 1. TinyGrad + Llama 3.2 (Research-Friendly)

TinyGrad is a minimalist deep learning framework that's perfect for understanding and hacking on LLMs.

```bash
# Dev shell (manual pip install)
cd llm/tinygrad
nix develop

# Install tinygrad in editable mode
pip install -e ../../vendor/tinygrad

# Run Llama
CUDA=1 python vendor/tinygrad/examples/llama3.py --download_model --size 1B --no_api
```

**Models available:**
- `1B` - Llama 3.2 1B Instruct (fast, ~1GB VRAM)
- `8B` - Llama 3.2 8B Instruct (better quality, ~8GB VRAM)

**Quantization options:** `int8`, `nf4`, `float16`, `fp8`

### 2. GLM-4.7-Flash (Production-Ready)

GLM-4.7-Flash is a 30B MoE model (3B active parameters) with exceptional reasoning capabilities.

```bash
cd llm/glm
nix develop

# Download model (choose quantization)
glm-download Q4_K_M    # ~17GB, good balance
glm-download Q6_K      # ~24GB, higher quality
glm-download Q8_0      # ~32GB, near-original quality

# Interactive chat
glm-chat

# Or start an API server
glm-server
# Then in another terminal:
glm-client "What is the capital of France?"
```

**API Usage:**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="x")
response = client.chat.completions.create(
    model="glm-4.7-flash",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Flake Structure

```
llm/
├── flake.nix                 # Unified flake (recommended)
├── README.md                 # This file
├── tinygrad/
│   ├── flake.nix            # Simple dev shell
│   └── flake-advanced.nix   # Full reproducible build
└── glm/
    └── flake.nix            # GLM-4.7-Flash setup
```

## Usage Patterns

### Pattern 1: Simple Dev Shell

```bash
nix develop ./llm
# or
nix develop ./llm#tinygrad
nix develop ./llm#glm
```

### Pattern 2: Run Directly

```bash
# Run Llama via TinyGrad
nix run ./llm#llama -- 1B int8

# Run GLM chat
nix run ./llm#glm

# Start GLM server
nix run ./llm#glm-server
```

### Pattern 3: Build and Deploy

```bash
# Build the TinyGrad package
nix build ./llm#tinygrad

# Build GLM tools
nix build ./llm#glm-chat
```

## Performance Tips

### Jetson Orin AGX 64GB

| Model | Quantization | VRAM | Speed |
|-------|--------------|------|-------|
| Llama 1B | int8 | ~1GB | ~100 tok/s |
| Llama 8B | int8 | ~8GB | ~25 tok/s |
| GLM-4.7-Flash | Q4_K_M | ~17GB | ~15 tok/s |
| GLM-4.7-Flash | Q8_0 | ~32GB | ~10 tok/s |

### Optimization Flags

```bash
# TinyGrad
export CUDA=1              # Use CUDA backend
export DEBUG=0             # Disable debug output
export BEAM=4              # Enable beam search optimization

# llama.cpp (GLM)
-ngl 999                   # Offload all layers to GPU
-c 8192                    # Context size
-t 8                       # CPU threads (for non-GPU ops)
--mlock                    # Lock memory (prevent swapping)
```

## Monitoring

```bash
# GPU usage
nvtop

# CPU/Memory
htop

# TinyGrad profiling
CUDA=1 DEBUG=2 python examples/llama3.py --profile
```

## Troubleshooting

### CUDA not detected

```bash
# Check CUDA is available
python -c "from tinygrad import Device; print(Device.DEFAULT)"

# Should show "CUDA" or "NV"
# If not, ensure nvidia driver is loaded:
nvidia-smi
```

### Out of memory

- Use smaller model (`1B` instead of `8B`)
- Use more aggressive quantization (`Q4_K_M` instead of `Q8_0`)
- Reduce context size (`-c 4096` instead of `8192`)

### Model download fails

```bash
# Manual download with resume
curl -L -C - -o ~/.cache/glm/GLM-4.7-Flash-Q4_K_M.gguf \
  https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf
```

## Integration with jetpack-nixos

These flakes are designed to work alongside the main jetpack-nixos configuration:

```nix
# In your NixOS configuration
{
  imports = [
    ./path/to/jetpack-nixos/modules/default.nix
  ];

  # The LLM tools are available in dev shells, not system packages
  # Run: nix develop ./llm
}
```

## References

- [TinyGrad](https://github.com/tinygrad/tinygrad) - Simple neural network framework
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - Efficient LLM inference
- [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) - State-of-the-art MoE model
- [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) - Meta's compact LLM

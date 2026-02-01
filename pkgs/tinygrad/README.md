# TinyGrad for Jetson

This directory contains the TinyGrad integration for NVIDIA Jetson devices running JetPack NixOS.

## Overview

[TinyGrad](https://github.com/tinygrad/tinygrad) is a simple, hackable deep learning framework that's perfect for experimenting on Jetson devices. It provides:

- **Minimal dependencies** - Just Python and NumPy
- **CUDA support** - Runs natively on Jetson's GPU
- **Hackable design** - Easy to modify and extend
- **Modern ML models** - Includes LLaMA, Stable Diffusion, Whisper, and more

## Installation

### Development Installation (Recommended)

For local development and testing, install TinyGrad in editable mode:

```bash
# Navigate to the vendor/tinygrad submodule
cd /path/to/jetpack-nixos/vendor/tinygrad

# Install in editable mode with development dependencies
pip install -e .

# Or with extra dependencies for examples
pip install -e ".[testing]"
```

### NixOS Installation

TinyGrad can be included in your NixOS configuration:

```nix
{ pkgs, ... }:
{
  environment.systemPackages = with pkgs; [
    (python3.withPackages (ps: with ps; [
      # TinyGrad will be available through the jetpack overlay
      pkgs.nvidia-jetpack.tinygrad
      numpy
      pillow
    ]))
  ];
}
```

## Quick Start

### Basic MNIST Example

```bash
# Train a CNN on MNIST using Jetson's GPU
CUDA=1 python examples/jetson_mnist.py

# Use Fashion MNIST variant
FASHION=1 CUDA=1 python examples/jetson_mnist.py

# Customize training
BS=256 STEPS=100 CUDA=1 python examples/jetson_mnist.py
```

### LLaMA Inference

```bash
# Check what models will fit in memory
python examples/jetson_llama.py --check-memory

# See recommended models for Jetson 64GB
python examples/jetson_llama.py --recommended

# Run inference (using TinyGrad's llama.py directly)
cd vendor/tinygrad/examples
CUDA=1 python llama.py --model /path/to/model --prompt "Hello, I am"
```

### Stable Diffusion

```bash
# Check memory requirements
python examples/jetson_stable_diffusion.py --check-memory

# Generate images (using TinyGrad's stable_diffusion.py)
cd vendor/tinygrad/examples
CUDA=1 python stable_diffusion.py --prompt "a beautiful sunset"
```

## Memory Guidelines for Jetson AGX Orin 64GB

With 64GB of unified memory, you can run:

| Model | Memory Required | Status |
|-------|----------------|--------|
| LLaMA 3.2 1B | ~2GB | ✅ Excellent |
| LLaMA 3.2 3B | ~6GB | ✅ Excellent |
| LLaMA 3 8B | ~16GB | ✅ Good |
| LLaMA 2 13B | ~26GB | ✅ Good (FP16) |
| LLaMA 2 70B (INT4) | ~35GB | ⚠️ Tight |
| Stable Diffusion 1.5 | ~4GB | ✅ Excellent |
| Stable Diffusion 2.1 | ~5.5GB | ✅ Excellent |
| SDXL | ~12GB | ✅ Good |

### Memory Optimization Tips

1. **Use FP16**: Set `HALF=1` to use half precision
2. **Enable JIT**: Set `JIT=1` for faster execution
3. **Batch wisely**: Smaller batch sizes use less memory
4. **Limit context**: For LLMs, reduce `MAX_CONTEXT` if needed

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA=1` | Use CUDA/GPU acceleration | CPU |
| `HALF=1` | Use FP16 for lower memory | FP32 |
| `JIT=1` | Enable JIT compilation | Off |
| `BS=512` | Batch size for training | 512 |
| `STEPS=70` | Training steps | 70 |
| `MAX_CONTEXT=4096` | LLM context window | 4096 |
| `DEBUG=1-4` | Debug output level | Off |

## TinyGrad Examples Reference

The vendored TinyGrad includes many examples you can run directly:

| Example | Description | Command |
|---------|-------------|---------|
| `beautiful_mnist.py` | MNIST training | `CUDA=1 python beautiful_mnist.py` |
| `llama.py` | LLaMA 1/2 inference | `CUDA=1 python llama.py --model PATH` |
| `llama3.py` | LLaMA 3 inference | `CUDA=1 python llama3.py --model PATH` |
| `stable_diffusion.py` | SD 1.5 image gen | `CUDA=1 python stable_diffusion.py` |
| `sdxl.py` | SDXL image gen | `CUDA=1 python sdxl.py` |
| `whisper.py` | Speech recognition | `CUDA=1 python whisper.py` |
| `yolov8.py` | Object detection | `CUDA=1 python yolov8.py` |
| `gpt2.py` | GPT-2 inference | `CUDA=1 python gpt2.py` |

## Development Workflow

### Local Development

1. Clone and set up:
```bash
cd /path/to/jetpack-nixos
git submodule update --init --recursive

# Install tinygrad in editable mode
cd vendor/tinygrad
pip install -e .
```

2. Make changes to TinyGrad:
```bash
# Edit TinyGrad source
vim tinygrad/tensor.py

# Test your changes immediately (no reinstall needed)
python -c "from tinygrad import Tensor; print(Tensor([1,2,3]))"
```

3. Commit your changes:
```bash
cd vendor/tinygrad
git add .
git commit -m "Your changes"
git push origin your-branch
```

### Testing on Jetson

1. Deploy to Jetson via NixOS rebuild
2. Or copy files and use editable install on device:
```bash
rsync -avz vendor/tinygrad jetson:/home/user/
ssh jetson "cd /home/user/tinygrad && pip install -e ."
```

## Troubleshooting

### CUDA not found
```bash
# Make sure CUDA is in your path
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# On JetPack NixOS, CUDA should be available through the system configuration
```

### Out of Memory
```bash
# Use FP16
HALF=1 CUDA=1 python your_script.py

# Reduce batch size
BS=64 CUDA=1 python your_script.py

# For LLMs, reduce context window
MAX_CONTEXT=2048 CUDA=1 python llama.py ...
```

### Performance Issues
```bash
# Enable JIT compilation
JIT=1 CUDA=1 python your_script.py

# Profile execution
DEBUG=2 CUDA=1 python your_script.py
```

## Resources

- [TinyGrad Documentation](https://docs.tinygrad.org/)
- [TinyGrad GitHub](https://github.com/tinygrad/tinygrad)
- [TinyGrad Discord](https://discord.gg/tinygrad)
- [JetPack NixOS Documentation](../README.md)
- [NVIDIA Jetson Documentation](https://developer.nvidia.com/embedded/jetson-modules)

## License

TinyGrad is licensed under the MIT License.

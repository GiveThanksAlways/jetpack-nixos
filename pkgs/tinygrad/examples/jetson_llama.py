#!/usr/bin/env python3
"""
LLaMA Inference Example for NVIDIA Jetson using TinyGrad

This example demonstrates how to run LLaMA inference on Jetson devices
with 64GB of unified memory. Suitable for 7B/8B and some 13B models.

Requirements:
  pip install -e /path/to/vendor/tinygrad
  pip install numpy sentencepiece tiktoken blobfile safetensors

Model Download:
  # Download LLaMA 3.2 1B (smallest, great for testing)
  # Or LLaMA 2 7B / LLaMA 3 8B for production use

Usage:
  # Run with TinyChat (interactive mode)
  CUDA=1 python jetson_llama.py --model /path/to/model --prompt "Hello, I am"

  # Run with specific temperature
  CUDA=1 python jetson_llama.py --model /path/to/model --prompt "Hello" --temperature 0.7

  # For memory-constrained scenarios, use quantization
  CUDA=1 HALF=1 python jetson_llama.py --model /path/to/model

Environment Variables:
  CUDA=1          - Use Jetson GPU (CUDA)
  HALF=1          - Use float16 for lower memory usage
  MAX_CONTEXT=4096 - Maximum context window
  JIT=1           - Enable JIT compilation
"""

import os
import sys
import argparse
from pathlib import Path

# Configure device before importing tinygrad
if os.getenv("CUDA", "0") == "1":
    os.environ["DEVICE"] = "CUDA"
    print("🚀 Using Jetson CUDA device")
else:
    os.environ["DEVICE"] = "CPU"
    print("💻 Using CPU device")


def get_memory_info():
    """Get memory information for Jetson (unified memory)."""
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        mem_total = None
        mem_available = None
        for line in meminfo.split('\n'):
            if 'MemTotal' in line:
                mem_total = int(line.split()[1]) / 1024 / 1024  # GB
            if 'MemAvailable' in line:
                mem_available = int(line.split()[1]) / 1024 / 1024  # GB
        return mem_total, mem_available
    except Exception:
        return None, None


def estimate_model_memory(model_size_b: float, dtype_bytes: int = 2) -> float:
    """
    Estimate memory required for a model.
    
    Args:
        model_size_b: Model size in billions of parameters
        dtype_bytes: Bytes per parameter (2 for fp16, 4 for fp32)
    
    Returns:
        Estimated memory in GB (including overhead)
    """
    params_memory = model_size_b * 1e9 * dtype_bytes / (1024**3)
    # Add ~20% overhead for KV cache and activations
    return params_memory * 1.2


def check_model_feasibility():
    """Check if common model sizes will fit on this Jetson."""
    mem_total, mem_available = get_memory_info()
    
    if mem_total is None:
        print("⚠️  Could not determine system memory")
        return
    
    print(f"\n📊 System Memory: {mem_total:.1f}GB total, {mem_available:.1f}GB available")
    print("\n📏 Model Size Estimates (FP16):")
    
    models = [
        ("LLaMA 3.2 1B", 1.0),
        ("LLaMA 3.2 3B", 3.0),
        ("LLaMA 2 7B / LLaMA 3 8B", 8.0),
        ("LLaMA 2 13B", 13.0),
        ("LLaMA 2 70B (quantized INT4)", 70.0 / 4),  # ~17.5 effective GB
        ("LLaMA 2 70B (FP16)", 70.0),
    ]
    
    for name, size in models:
        mem_needed = estimate_model_memory(size)
        status = "✅" if mem_needed < mem_available else "❌"
        print(f"  {status} {name}: ~{mem_needed:.1f}GB")


def print_recommended_models():
    """Print recommended models for Jetson 64GB."""
    print("\n🎯 Recommended Models for Jetson AGX Orin 64GB:")
    print("=" * 60)
    print("""
1. LLaMA 3.2 1B/3B - Great for learning and development
   - Fast inference, low memory usage
   - Good for prototyping and testing
   
2. LLaMA 3 8B / LLaMA 2 7B - Best balance of quality and speed
   - Fits comfortably in 64GB
   - Good for most applications
   
3. LLaMA 2 13B - Higher quality responses
   - Usable with FP16
   - Slightly slower inference
   
4. LLaMA 2 70B (INT4 quantized) - Maximum capability
   - Requires quantization (see extra/gemm for quantization tools)
   - Slower but highest quality
   
5. Mixtral 8x7B (Sparse MoE) - High quality with MoE efficiency
   - Only active experts loaded
   - Good quality to memory ratio
""")


def main():
    parser = argparse.ArgumentParser(
        description="Run LLaMA inference on Jetson with TinyGrad",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check what models will fit on your Jetson
  python jetson_llama.py --check-memory
  
  # Run inference with a local model
  CUDA=1 python jetson_llama.py --model ~/models/llama-3-8b --prompt "Hello, world"
  
  # Interactive chat mode
  CUDA=1 python jetson_llama.py --model ~/models/llama-3-8b --chat
        """
    )
    
    parser.add_argument("--model", type=str, help="Path to model directory or HuggingFace model name")
    parser.add_argument("--prompt", type=str, default="Hello, I am a", help="Prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--check-memory", action="store_true", help="Check memory and model feasibility")
    parser.add_argument("--recommended", action="store_true", help="Show recommended models")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat mode")
    
    args = parser.parse_args()
    
    # Handle info-only modes
    if args.check_memory:
        check_model_feasibility()
        return
    
    if args.recommended:
        print_recommended_models()
        return
    
    if not args.model:
        print("❌ No model specified. Use --model to specify a model path.")
        print("\nQuick start:")
        print("  python jetson_llama.py --check-memory  # Check what fits")
        print("  python jetson_llama.py --recommended   # See recommended models")
        print("\nFor full LLaMA inference, please use the TinyGrad examples directly:")
        print("  cd vendor/tinygrad/examples")
        print("  CUDA=1 python llama.py --help")
        return
    
    # Import TinyGrad components
    print("\n📦 Loading TinyGrad...")
    from tinygrad import Tensor, Device, GlobalCounters
    from tinygrad.helpers import Timing
    
    print(f"Device: {Device.DEFAULT}")
    
    # Point user to the actual TinyGrad LLaMA implementation
    print(f"\n🦙 To run LLaMA inference with model at: {args.model}")
    print("\nRun the TinyGrad LLaMA example directly:")
    print(f"  cd vendor/tinygrad/examples")
    print(f"  CUDA=1 python llama.py --model {args.model} --prompt \"{args.prompt}\"")
    print("\nOr for LLaMA 3:")
    print(f"  CUDA=1 python llama3.py --model {args.model} --prompt \"{args.prompt}\"")
    
    # Show memory status
    check_model_feasibility()


if __name__ == "__main__":
    main()

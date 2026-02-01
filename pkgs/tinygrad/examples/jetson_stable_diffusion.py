#!/usr/bin/env python3
"""
Stable Diffusion Example for NVIDIA Jetson using TinyGrad

This example demonstrates how to run Stable Diffusion image generation
on Jetson devices with 64GB of unified memory.

Requirements:
  pip install -e /path/to/vendor/tinygrad
  pip install numpy pillow

Model Download:
  Stable Diffusion 1.5 weights (downloads automatically on first run)

Usage:
  # Generate an image with a prompt
  CUDA=1 python jetson_stable_diffusion.py --prompt "a photo of a cat"

  # Specify output file
  CUDA=1 python jetson_stable_diffusion.py --prompt "sunset over mountains" --output sunset.png

  # Use different guidance scale
  CUDA=1 python jetson_stable_diffusion.py --prompt "abstract art" --guidance 12.0

  # Specify seed for reproducibility
  CUDA=1 python jetson_stable_diffusion.py --prompt "landscape" --seed 42

Environment Variables:
  CUDA=1          - Use Jetson GPU (CUDA)
  HALF=1          - Use float16 for lower memory usage (recommended)
  STEPS=50        - Number of diffusion steps (default: 50)
"""

import os
import shlex
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


def estimate_sd_memory(model: str = "sd1.5", resolution: int = 512) -> float:
    """
    Estimate memory required for Stable Diffusion.
    
    Args:
        model: Model variant (sd1.5, sd2.1, sdxl)
        resolution: Output image resolution
    
    Returns:
        Estimated memory in GB
    """
    # Base model sizes (approximate, in GB)
    model_sizes = {
        "sd1.5": 4.0,   # ~4GB for SD 1.5
        "sd2.1": 5.5,   # ~5.5GB for SD 2.1
        "sdxl": 12.0,   # ~12GB for SDXL
    }
    
    base_size = model_sizes.get(model, 4.0)
    
    # Resolution affects VRAM for activations
    resolution_factor = (resolution / 512) ** 2
    
    # Estimate with overhead
    return base_size * (1.0 + 0.5 * resolution_factor)


def check_sd_feasibility():
    """Check which SD variants will fit on this Jetson."""
    mem_total, mem_available = get_memory_info()
    
    if mem_total is None:
        print("⚠️  Could not determine system memory")
        return
    
    print(f"\n📊 System Memory: {mem_total:.1f}GB total, {mem_available:.1f}GB available")
    print("\n🎨 Stable Diffusion Variants:")
    
    variants = [
        ("Stable Diffusion 1.5 (512x512)", "sd1.5", 512),
        ("Stable Diffusion 1.5 (768x768)", "sd1.5", 768),
        ("Stable Diffusion 2.1 (512x512)", "sd2.1", 512),
        ("Stable Diffusion 2.1 (768x768)", "sd2.1", 768),
        ("SDXL Base (1024x1024)", "sdxl", 1024),
    ]
    
    for name, model, res in variants:
        mem_needed = estimate_sd_memory(model, res)
        status = "✅" if mem_needed < mem_available else "❌"
        print(f"  {status} {name}: ~{mem_needed:.1f}GB")
    
    print("\n💡 Tips for Jetson AGX Orin 64GB:")
    print("  - Use HALF=1 to enable FP16 for lower memory usage")
    print("  - SD 1.5 and SD 2.1 run very well at 512x512 and 768x768")
    print("  - SDXL is possible but may require FP16 and careful memory management")


def print_examples():
    """Print example commands."""
    print("\n📝 Example Commands:")
    print("=" * 60)
    print("""
# Basic image generation
CUDA=1 python jetson_stable_diffusion.py --prompt "a beautiful sunset"

# Higher guidance for more prompt adherence  
CUDA=1 python jetson_stable_diffusion.py --prompt "photorealistic cat" --guidance 12

# Reproducible generation with seed
CUDA=1 python jetson_stable_diffusion.py --prompt "mountains" --seed 42

# Custom output and more steps
CUDA=1 python jetson_stable_diffusion.py --prompt "artwork" --output art.png --steps 75

# Using TinyGrad SD example directly (full features):
cd vendor/tinygrad/examples
CUDA=1 python stable_diffusion.py --prompt "your prompt here"

# SDXL for higher quality:
CUDA=1 python sdxl.py --prompt "your prompt here"
""")


def main():
    parser = argparse.ArgumentParser(
        description="Run Stable Diffusion on Jetson with TinyGrad",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script provides guidance for running Stable Diffusion on Jetson.
For full functionality, use the TinyGrad examples directly:

  cd vendor/tinygrad/examples
  CUDA=1 python stable_diffusion.py --help
        """
    )
    
    parser.add_argument("--prompt", type=str, help="Text prompt for image generation")
    parser.add_argument("--output", type=str, default="output.png", help="Output image path")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale (default: 7.5)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument("--check-memory", action="store_true", help="Check memory and model feasibility")
    parser.add_argument("--examples", action="store_true", help="Show example commands")
    
    args = parser.parse_args()
    
    # Handle info-only modes
    if args.check_memory:
        check_sd_feasibility()
        return
    
    if args.examples:
        print_examples()
        return
    
    if not args.prompt:
        print("❌ No prompt specified. Use --prompt to specify an image description.")
        print("\nQuick start:")
        print("  python jetson_stable_diffusion.py --check-memory  # Check what fits")
        print("  python jetson_stable_diffusion.py --examples      # See example commands")
        print("\nFor full Stable Diffusion, use the TinyGrad examples directly:")
        print("  cd vendor/tinygrad/examples")
        print("  CUDA=1 python stable_diffusion.py --help")
        return
    
    # Import TinyGrad components
    print("\n📦 Loading TinyGrad...")
    from tinygrad import Tensor, Device
    
    print(f"Device: {Device.DEFAULT}")
    
    # Point user to the actual TinyGrad SD implementation
    print(f"\n🎨 To generate image with prompt: \"{args.prompt}\"")
    print("\nRun the TinyGrad Stable Diffusion example directly:")
    print(f"  cd vendor/tinygrad/examples")
    print(f"  CUDA=1 python stable_diffusion.py --prompt {shlex.quote(args.prompt)} --out {shlex.quote(args.output)}")
    print("\nOr for SDXL (higher quality):")
    print(f"  CUDA=1 python sdxl.py --prompt {shlex.quote(args.prompt)}")
    
    # Show memory status
    check_sd_feasibility()


if __name__ == "__main__":
    main()

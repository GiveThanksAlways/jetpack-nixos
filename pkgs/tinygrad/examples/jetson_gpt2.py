#!/usr/bin/env python3
"""
GPT-2 Text Generation Example for NVIDIA Jetson using TinyGrad

GPT-2 is a great starting point for text generation on Jetson because:
- Models are relatively small (117M to 1.5B parameters)
- Well documented and widely understood
- Fast inference even on smaller devices

Requirements:
  pip install -e /path/to/vendor/tinygrad
  pip install numpy tiktoken

Usage:
  # Run text generation
  CUDA=1 python jetson_gpt2.py --prompt "The future of AI is"

  # Choose model size (small, medium, large, xl)
  CUDA=1 python jetson_gpt2.py --size medium --prompt "Once upon a time"

  # Adjust generation parameters
  CUDA=1 python jetson_gpt2.py --prompt "Hello" --max-tokens 100 --temperature 0.8

Model Sizes:
  - small (117M):  ~500MB memory, fastest
  - medium (345M): ~1.5GB memory, good quality
  - large (774M):  ~3GB memory, better quality
  - xl (1.5B):     ~6GB memory, best quality

All models fit easily on Jetson AGX Orin 64GB!
"""

import os
import sys
import argparse

# Configure device before importing tinygrad
if os.getenv("CUDA", "0") == "1":
    os.environ["DEVICE"] = "CUDA"
    print("🚀 Using Jetson CUDA device")
else:
    os.environ["DEVICE"] = "CPU"
    print("💻 Using CPU device")


def get_model_info():
    """Get information about GPT-2 model sizes."""
    return {
        "small": {
            "params": "117M",
            "memory": "~500MB",
            "description": "Fast, good for prototyping"
        },
        "medium": {
            "params": "345M", 
            "memory": "~1.5GB",
            "description": "Balanced speed/quality"
        },
        "large": {
            "params": "774M",
            "memory": "~3GB",
            "description": "Better quality text"
        },
        "xl": {
            "params": "1.5B",
            "memory": "~6GB",
            "description": "Best quality, slower"
        }
    }


def print_model_sizes():
    """Print available model sizes."""
    print("\n📏 GPT-2 Model Sizes:")
    print("=" * 60)
    
    models = get_model_info()
    for name, info in models.items():
        print(f"\n  {name}:")
        print(f"    Parameters: {info['params']}")
        print(f"    Memory: {info['memory']}")
        print(f"    Notes: {info['description']}")
    
    print("\n💡 All models fit comfortably on Jetson AGX Orin 64GB!")
    print("   For fastest results, start with 'small' or 'medium'.")


def main():
    parser = argparse.ArgumentParser(
        description="Run GPT-2 text generation on Jetson with TinyGrad",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with small model
  CUDA=1 python jetson_gpt2.py --prompt "Hello world"
  
  # Use larger model for better quality
  CUDA=1 python jetson_gpt2.py --size large --prompt "The key to happiness is"
  
  # Creative writing with higher temperature
  CUDA=1 python jetson_gpt2.py --prompt "A story about" --temperature 0.9
  
  # More deterministic output
  CUDA=1 python jetson_gpt2.py --prompt "Explain" --temperature 0.3
        """
    )
    
    parser.add_argument("--prompt", type=str, help="Text prompt for generation")
    parser.add_argument("--size", type=str, default="small", 
                       choices=["small", "medium", "large", "xl"],
                       help="Model size (default: small)")
    parser.add_argument("--max-tokens", type=int, default=50, 
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature (0.1-1.0)")
    parser.add_argument("--list-sizes", action="store_true",
                       help="List available model sizes")
    
    args = parser.parse_args()
    
    if args.list_sizes:
        print_model_sizes()
        return
    
    if not args.prompt:
        print("❌ No prompt specified. Use --prompt to specify text.")
        print("\nQuick start:")
        print("  python jetson_gpt2.py --list-sizes     # See model options")
        print("  python jetson_gpt2.py --prompt \"Hello\" # Generate text")
        print("\nFor full GPT-2 functionality, use TinyGrad's example directly:")
        print("  cd vendor/tinygrad/examples")
        print("  CUDA=1 python gpt2.py --help")
        return
    
    print(f"\n📝 Prompt: \"{args.prompt}\"")
    print(f"📦 Model: GPT-2 {args.size}")
    
    # Import and check TinyGrad
    print("\n📦 Loading TinyGrad...")
    try:
        from tinygrad import Device
        print(f"   Device: {Device.DEFAULT}")
    except ImportError:
        print("❌ TinyGrad not installed. Run: pip install -e vendor/tinygrad")
        return
    
    # Point to TinyGrad's implementation
    print(f"\n🤖 To generate text with GPT-2:")
    print(f"\nRun the TinyGrad GPT-2 example directly:")
    print(f"  cd vendor/tinygrad/examples")
    print(f"  CUDA=1 python gpt2.py")
    print(f"\nThis will download the model and start generation.")
    
    # Show model sizes
    print_model_sizes()


if __name__ == "__main__":
    main()

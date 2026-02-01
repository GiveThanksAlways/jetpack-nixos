#!/usr/bin/env python3
"""
TinyGrad Device Check for NVIDIA Jetson

This utility checks if TinyGrad is properly configured and can use the Jetson GPU.

Usage:
  python jetson_device_check.py
"""

import os
import sys


def check_python_version():
    """Check Python version compatibility."""
    print("📌 Python Version Check")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    if version >= (3, 11):
        print("   ✅ Python version is compatible")
        return True
    elif version >= (3, 10):
        print("   ⚠️  Python 3.10 works but 3.11+ is recommended")
        return True
    else:
        print("   ❌ Python 3.10+ is required")
        return False


def check_tinygrad_import():
    """Check if TinyGrad can be imported."""
    print("\n📌 TinyGrad Import Check")
    try:
        from tinygrad import Tensor, Device
        print(f"   TinyGrad imported successfully")
        print(f"   ✅ Default device: {Device.DEFAULT}")
        return True
    except ImportError as e:
        print(f"   ❌ Failed to import TinyGrad: {e}")
        print("   Install with: pip install -e vendor/tinygrad")
        return False


def check_cuda_available():
    """Check if CUDA device is available."""
    print("\n📌 CUDA Device Check")
    try:
        from tinygrad import Device
        
        # Try to check for CUDA
        if "CUDA" in Device._devices or "GPU" in Device._devices:
            print("   ✅ CUDA device appears to be available")
            return True
        else:
            print("   ⚠️  CUDA device not detected in Device._devices")
            print("   Available devices:", list(Device._devices.keys()))
            return False
    except Exception as e:
        print(f"   ⚠️  Could not check CUDA availability: {e}")
        return False


def check_cuda_tensor():
    """Try to create a tensor on CUDA."""
    print("\n📌 CUDA Tensor Check")
    try:
        os.environ["CUDA"] = "1"
        from tinygrad import Tensor, Device
        
        # Try to use CUDA
        Device.DEFAULT = "CUDA"
        x = Tensor([1, 2, 3, 4, 5])
        result = (x * 2).numpy()
        
        if list(result) == [2, 4, 6, 8, 10]:
            print("   ✅ CUDA tensor operations work correctly!")
            print(f"   Test result: {result}")
            return True
        else:
            print(f"   ❌ Unexpected result: {result}")
            return False
    except Exception as e:
        print(f"   ⚠️  CUDA tensor test failed: {e}")
        print("   This is expected if not running on a Jetson with GPU access")
        return False


def check_memory():
    """Check system memory."""
    print("\n📌 System Memory Check")
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        
        for line in meminfo.split('\n'):
            if 'MemTotal' in line:
                mem_kb = int(line.split()[1])
                mem_gb = mem_kb / 1024 / 1024
                print(f"   Total memory: {mem_gb:.1f} GB")
                if mem_gb >= 60:
                    print("   ✅ 64GB Jetson detected - can run large models!")
                elif mem_gb >= 30:
                    print("   ✅ 32GB detected - good for medium models")
                elif mem_gb >= 14:
                    print("   ⚠️  16GB detected - suitable for smaller models")
                else:
                    print("   ⚠️  Limited memory - use small models and FP16")
                break
            
        for line in meminfo.split('\n'):
            if 'MemAvailable' in line:
                mem_kb = int(line.split()[1])
                mem_gb = mem_kb / 1024 / 1024
                print(f"   Available memory: {mem_gb:.1f} GB")
                return True
    except Exception as e:
        print(f"   ⚠️  Could not check memory: {e}")
        return False


def check_jetson_info():
    """Check Jetson-specific information."""
    print("\n📌 Jetson Device Info")
    
    # Check for Jetson model
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip().replace('\x00', '')
            print(f"   Device: {model}")
            return True
    except FileNotFoundError:
        print("   ⚠️  Not running on a Jetson device (or device tree not accessible)")
        return False
    except Exception as e:
        print(f"   ⚠️  Could not detect Jetson model: {e}")
        return False


def main():
    print("=" * 60)
    print("🔍 TinyGrad for Jetson - Device Check")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("TinyGrad Import", check_tinygrad_import),
        ("System Memory", check_memory),
        ("Jetson Device", check_jetson_info),
        ("CUDA Available", check_cuda_available),
        ("CUDA Tensor", check_cuda_tensor),
    ]
    
    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"   ❌ Check failed with error: {e}")
            results[name] = False
    
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    
    all_passed = all(results.values())
    critical_passed = results.get("Python Version", False) and results.get("TinyGrad Import", False)
    
    for name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {name}")
    
    print()
    if critical_passed:
        if all_passed:
            print("🎉 All checks passed! TinyGrad is ready to use on your Jetson.")
        else:
            print("✅ TinyGrad is installed and basic functionality works.")
            print("   Some optional features may not be available.")
    else:
        print("❌ Critical checks failed. Please fix the issues above.")
    
    print("\n💡 Quick Test:")
    print("   CUDA=1 python -c \"from tinygrad import Tensor; print(Tensor([1,2,3]) * 2)\"")
    print()


if __name__ == "__main__":
    main()

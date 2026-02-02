#!/usr/bin/env bash
# Quick test to verify LLM environment is working on Jetson
set -e

echo "🔍 Jetson LLM Environment Test"
echo "=============================="
echo ""

# Check CUDA
echo "1. Checking NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo "   ✅ GPU detected"
else
    echo "   ⚠️  nvidia-smi not found (may be OK in dev shell)"
fi
echo ""

# Check Python
echo "2. Checking Python..."
python3 --version
echo "   ✅ Python available"
echo ""

# Check TinyGrad
echo "3. Checking TinyGrad..."
if python3 -c "import tinygrad" 2>/dev/null; then
    python3 -c "
from tinygrad import Device
print(f'   Default device: {Device.DEFAULT}')
"
    echo "   ✅ TinyGrad installed"
else
    echo "   ⚠️  TinyGrad not installed (run: pip install -e vendor/tinygrad)"
fi
echo ""

# Check llama.cpp
echo "4. Checking llama.cpp..."
if command -v llama-cli &> /dev/null; then
    echo "   ✅ llama-cli available"
elif command -v llama-cpp &> /dev/null; then
    echo "   ✅ llama-cpp available"
else
    echo "   ⚠️  llama.cpp not found (needed for GLM)"
fi
echo ""

# Check model cache
echo "5. Checking model cache..."
if [ -d "$HOME/.cache/glm" ]; then
    echo "   GLM models:"
    ls -lh "$HOME/.cache/glm"/*.gguf 2>/dev/null || echo "   (no models downloaded)"
else
    echo "   No GLM models downloaded yet"
fi

if [ -d "$HOME/.cache/tinygrad" ]; then
    echo "   TinyGrad cache exists"
fi
echo ""

# Quick compute test
echo "6. Quick CUDA compute test..."
CUDA=1 python3 -c "
try:
    from tinygrad import Tensor
    import time
    x = Tensor.rand(512, 512)
    y = Tensor.rand(512, 512)
    start = time.time()
    z = (x @ y).numpy()
    elapsed = (time.time() - start) * 1000
    print(f'   512x512 matmul: {elapsed:.1f}ms')
    print('   ✅ CUDA compute working')
except Exception as e:
    print(f'   ⚠️  Test failed: {e}')
" 2>/dev/null || echo "   ⚠️  TinyGrad CUDA test skipped"
echo ""

echo "=============================="
echo "✅ Environment check complete!"
echo ""
echo "Next steps:"
echo "  TinyGrad: tinygrad-llama 1B int8"
echo "  GLM:      glm-download Q4_K_M && glm-chat"

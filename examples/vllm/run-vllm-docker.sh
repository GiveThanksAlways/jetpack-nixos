#!/bin/bash
# Run vLLM in Docker on Jetson Orin AGX
# Uses NVIDIA L4T base image with JetPack 6 CUDA 12.6
#
# Usage:
#   ./run-vllm-docker.sh [model] [port]
#
# Examples:
#   ./run-vllm-docker.sh                                     # Default: Llama 3.2 1B
#   ./run-vllm-docker.sh meta-llama/Llama-3.2-3B-Instruct    # 3B model
#   ./run-vllm-docker.sh Qwen/Qwen3-0.6B 8001                # Custom port

set -euo pipefail

MODEL="${1:-meta-llama/Llama-3.2-1B-Instruct}"
PORT="${2:-8000}"
CONTAINER_NAME="vllm-orin"
HF_CACHE="${HOME}/.cache/huggingface"

echo "=== vLLM Docker Launcher ==="
echo "Model: $MODEL"
echo "Port: $PORT"
echo "HF cache: $HF_CACHE"
echo ""

# Build if Dockerfile exists and image doesn't
if ! docker image inspect vllm-jetson:latest >/dev/null 2>&1; then
    if [ -f Dockerfile.jetson ]; then
        echo "Building vLLM Jetson Docker image..."
        docker build -t vllm-jetson:latest -f Dockerfile.jetson .
    else
        echo "Using upstream vLLM image..."
        # Try NVIDIA's Jetson-optimized image first
        VLLM_IMAGE="dustynv/vllm:0.6.6-r36.4.0"
        echo "Pulling $VLLM_IMAGE ..."
        docker pull "$VLLM_IMAGE" || {
            echo "Jetson image not available, trying generic aarch64..."
            VLLM_IMAGE="vllm/vllm-openai:latest"
            docker pull "$VLLM_IMAGE"
        }
        docker tag "$VLLM_IMAGE" vllm-jetson:latest
    fi
fi

# Stop existing container
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo ""
echo "Starting vLLM server..."
echo "  Model: $MODEL"
echo "  Port: localhost:$PORT"
echo ""

docker run -d \
    --name "$CONTAINER_NAME" \
    --runtime nvidia \
    --gpus all \
    -p "$PORT:8000" \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    -e CUDA_VISIBLE_DEVICES=0 \
    vllm-jetson:latest \
    --model "$MODEL" \
    --max-model-len 2048 \
    --dtype half \
    --gpu-memory-utilization 0.8 \
    --enforce-eager

echo ""
echo "Container started: $CONTAINER_NAME"
echo "Waiting for server to be ready..."

# Wait for health check
for i in $(seq 1 60); do
    if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo "vLLM server is ready at http://localhost:$PORT"
        echo ""
        echo "Test with:"
        echo "  curl http://localhost:$PORT/v1/completions -H 'Content-Type: application/json' -d '{\"model\": \"$MODEL\", \"prompt\": \"Hello\", \"max_tokens\": 10}'"
        echo ""
        echo "Benchmark:"
        echo "  python3 bench_vllm.py --server http://localhost:$PORT --model $MODEL"
        echo ""
        echo "Stop:"
        echo "  docker stop $CONTAINER_NAME"
        exit 0
    fi
    sleep 2
    echo "  Waiting... ($i/60)"
done

echo "ERROR: Server did not become ready in 120s"
echo "Check logs: docker logs $CONTAINER_NAME"
exit 1

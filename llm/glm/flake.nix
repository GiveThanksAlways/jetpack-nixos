{
  description = "GLM-4.7-Flash on Jetson Orin AGX - High Performance MoE LLM";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "aarch64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        # llama.cpp with CUDA support for Jetson
        # This is the most efficient way to run GGUF models on Jetson
        llama-cpp = pkgs.stdenv.mkDerivation rec {
          pname = "llama-cpp";
          version = "b5070";

          src = pkgs.fetchFromGitHub {
            owner = "ggml-org";
            repo = "llama.cpp";
            rev = version;
            sha256 = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="; # Update on first build
          };

          nativeBuildInputs = with pkgs; [
            cmake
            ninja
            pkg-config
          ];

          buildInputs = with pkgs; [
            cudaPackages.cudatoolkit
            cudaPackages.cudnn
          ];

          cmakeFlags = [
            "-DGGML_CUDA=ON"
            "-DCMAKE_CUDA_ARCHITECTURES=87"  # Orin is SM 8.7
            "-DGGML_NATIVE=OFF"
            "-DLLAMA_BUILD_SERVER=ON"
          ];

          meta = with pkgs.lib; {
            description = "LLM inference in C/C++ with CUDA support";
            homepage = "https://github.com/ggml-org/llama.cpp";
            license = licenses.mit;
            platforms = [ "aarch64-linux" ];
          };
        };

        # Model download helper
        # GLM-4.7-Flash GGUF from unsloth (quantized versions)
        download-model = pkgs.writeShellApplication {
          name = "glm-download";
          runtimeInputs = with pkgs; [ curl jq ];
          text = ''
            set -e

            MODEL_DIR="''${GLM_MODEL_DIR:-$HOME/.cache/glm}"
            mkdir -p "$MODEL_DIR"

            QUANT="''${1:-Q4_K_M}"
            MODEL_NAME="GLM-4.7-Flash-$QUANT.gguf"
            MODEL_PATH="$MODEL_DIR/$MODEL_NAME"

            echo "📥 Downloading GLM-4.7-Flash ($QUANT quantization)"
            echo "   Target: $MODEL_PATH"
            echo ""

            if [ -f "$MODEL_PATH" ]; then
                echo "✅ Model already exists at $MODEL_PATH"
                exit 0
            fi

            # Available quantizations from unsloth/GLM-4.7-Flash-GGUF:
            # Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16
            BASE_URL="https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main"

            echo "⬇️  Downloading from HuggingFace..."
            curl -L -o "$MODEL_PATH" \
                "$BASE_URL/GLM-4.7-Flash-$QUANT.gguf" \
                --progress-bar

            echo ""
            echo "✅ Download complete!"
            echo "   Size: $(du -h "$MODEL_PATH" | cut -f1)"
            echo "   Path: $MODEL_PATH"
          '';
        };

        # Main GLM chat interface
        glm-chat = pkgs.writeShellApplication {
          name = "glm-chat";
          runtimeInputs = [ llama-cpp ];
          text = ''
            set -e

            MODEL_DIR="''${GLM_MODEL_DIR:-$HOME/.cache/glm}"
            QUANT="''${1:-Q4_K_M}"
            MODEL_PATH="$MODEL_DIR/GLM-4.7-Flash-$QUANT.gguf"

            if [ ! -f "$MODEL_PATH" ]; then
                echo "❌ Model not found at $MODEL_PATH"
                echo "   Run: glm-download $QUANT"
                exit 1
            fi

            echo "💬 GLM-4.7-Flash Chat (Jetson Orin)"
            echo "   Model: $MODEL_PATH"
            echo "   GPU Layers: 999 (full offload)"
            echo ""
            echo "   Type your message and press Enter. Ctrl+C to exit."
            echo ""

            llama-cli \
                --model "$MODEL_PATH" \
                --n-gpu-layers 999 \
                --ctx-size 8192 \
                --threads 8 \
                --interactive \
                --color \
                --temp 0.7 \
                --repeat-penalty 1.1 \
                -p "You are GLM-4.7-Flash, a helpful AI assistant. Answer concisely and accurately."
          '';
        };

        # OpenAI-compatible API server
        glm-server = pkgs.writeShellApplication {
          name = "glm-server";
          runtimeInputs = [ llama-cpp ];
          text = ''
            set -e

            MODEL_DIR="''${GLM_MODEL_DIR:-$HOME/.cache/glm}"
            QUANT="''${1:-Q4_K_M}"
            MODEL_PATH="$MODEL_DIR/GLM-4.7-Flash-$QUANT.gguf"
            PORT="''${2:-8080}"

            if [ ! -f "$MODEL_PATH" ]; then
                echo "❌ Model not found at $MODEL_PATH"
                echo "   Run: glm-download $QUANT"
                exit 1
            fi

            echo "🚀 GLM-4.7-Flash API Server"
            echo "   Model: $MODEL_PATH"
            echo "   Port: $PORT"
            echo "   GPU Layers: 999"
            echo ""
            echo "   OpenAI-compatible endpoint: http://localhost:$PORT/v1"
            echo ""

            llama-server \
                --model "$MODEL_PATH" \
                --n-gpu-layers 999 \
                --ctx-size 8192 \
                --host 0.0.0.0 \
                --port "$PORT" \
                --threads 8 \
                --parallel 2
          '';
        };

        # Benchmark tool
        glm-benchmark = pkgs.writeShellApplication {
          name = "glm-benchmark";
          runtimeInputs = [ llama-cpp ];
          text = ''
            set -e

            MODEL_DIR="''${GLM_MODEL_DIR:-$HOME/.cache/glm}"
            QUANT="''${1:-Q4_K_M}"
            MODEL_PATH="$MODEL_DIR/GLM-4.7-Flash-$QUANT.gguf"

            if [ ! -f "$MODEL_PATH" ]; then
                echo "❌ Model not found at $MODEL_PATH"
                exit 1
            fi

            echo "📊 GLM-4.7-Flash Benchmark"
            echo "   Model: $QUANT"
            echo "   Running on Jetson Orin CUDA..."
            echo ""

            llama-bench \
                --model "$MODEL_PATH" \
                --n-gpu-layers 999 \
                --threads 8 \
                -p 512 \
                -n 128
          '';
        };

        # Python SDK for programmatic access
        pythonEnv = pkgs.python311.withPackages (ps: with ps; [
          requests
          openai
          httpx
          rich
        ]);

        # Example client script
        glm-client = pkgs.writeShellApplication {
          name = "glm-client";
          runtimeInputs = [ pythonEnv ];
          text = ''
            python << 'EOF'
import sys
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello! What can you help me with?"

print(f"🗣️  You: {prompt}")
print("🤖 GLM: ", end="", flush=True)

stream = client.chat.completions.create(
    model="glm-4.7-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    stream=True,
    max_tokens=512,
    temperature=0.7
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
EOF
          '';
        };

      in {
        packages = {
          default = glm-chat;
          inherit llama-cpp download-model glm-chat glm-server glm-benchmark glm-client;
        };

        apps = {
          default = { type = "app"; program = "${glm-chat}/bin/glm-chat"; };
          chat = { type = "app"; program = "${glm-chat}/bin/glm-chat"; };
          server = { type = "app"; program = "${glm-server}/bin/glm-server"; };
          download = { type = "app"; program = "${download-model}/bin/glm-download"; };
          benchmark = { type = "app"; program = "${glm-benchmark}/bin/glm-benchmark"; };
          client = { type = "app"; program = "${glm-client}/bin/glm-client"; };
        };

        devShells.default = pkgs.mkShell {
          name = "glm-dev";
          buildInputs = [
            llama-cpp
            download-model
            glm-chat
            glm-server
            glm-benchmark
            glm-client
            pythonEnv
            pkgs.nvtop
            pkgs.htop
            pkgs.curl
            pkgs.jq
          ];

          shellHook = ''
            echo ""
            echo "╔══════════════════════════════════════════════════════════════╗"
            echo "║  🌟 GLM-4.7-Flash on Jetson Orin AGX                         ║"
            echo "╠══════════════════════════════════════════════════════════════╣"
            echo "║                                                              ║"
            echo "║  Quick Start:                                                ║"
            echo "║    1. glm-download Q4_K_M    # Download model (~17GB)        ║"
            echo "║    2. glm-chat               # Interactive chat              ║"
            echo "║                                                              ║"
            echo "║  Server Mode:                                                ║"
            echo "║    glm-server                # Start OpenAI-compatible API   ║"
            echo "║    glm-client \"Hello!\"       # Query the server              ║"
            echo "║                                                              ║"
            echo "║  Quantization options: Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K    ║"
            echo "║  Recommended: Q4_K_M (balance of quality/speed)              ║"
            echo "╚══════════════════════════════════════════════════════════════╝"
            echo ""

            export GLM_MODEL_DIR="$HOME/.cache/glm"
          '';
        };

        # Overlay for integration with other flakes
        overlays.default = final: prev: {
          glm-flash = glm-chat;
          glm-server = glm-server;
          inherit llama-cpp;
        };
      }
    );
}

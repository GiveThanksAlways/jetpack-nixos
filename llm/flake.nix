{
  description = "Jetson Orin AGX LLM Inference - TinyGrad & GLM-4.7-Flash";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";

    # Pin tinygrad for reproducibility
    tinygrad-src = {
      url = "github:tinygrad/tinygrad";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, tinygrad-src }:
    flake-utils.lib.eachSystem [ "aarch64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        python = pkgs.python311;

        #############################################
        # TINYGRAD
        #############################################

        tinygrad = python.pkgs.buildPythonPackage {
          pname = "tinygrad";
          version = "0.10.0-git";
          format = "pyproject";
          src = tinygrad-src;

          nativeBuildInputs = with python.pkgs; [ setuptools wheel ];
          propagatedBuildInputs = with python.pkgs; [
            numpy pillow tqdm requests tiktoken sentencepiece
          ];

          doCheck = false;
          pythonImportsCheck = [ "tinygrad" ];
        };

        pythonWithTinygrad = python.withPackages (ps: [
          tinygrad ps.numpy ps.pillow ps.tqdm ps.requests
          ps.tiktoken ps.sentencepiece ps.safetensors
        ]);

        tinygrad-llama = pkgs.writeShellApplication {
          name = "tinygrad-llama";
          runtimeInputs = [ pythonWithTinygrad ];
          text = ''
            set -e
            export CUDA=1
            cd "${tinygrad-src}"

            SIZE="''${1:-1B}"
            QUANT="''${2:-int8}"

            echo "🦙 TinyGrad Llama 3.2 ($SIZE, $QUANT)"
            python examples/llama3.py \
              --download_model \
              --size "$SIZE" \
              --quantize "$QUANT" \
              --no_api \
              --temperature 0.7 \
              "''${@:3}"
          '';
        };

        tinygrad-benchmark = pkgs.writeShellApplication {
          name = "tinygrad-bench";
          runtimeInputs = [ pythonWithTinygrad ];
          text = ''
            export CUDA=1
            python -c "
from tinygrad import Tensor, Device
from tinygrad.helpers import getenv
import time

print(f'Device: {Device.DEFAULT}')
print()

for size in [1024, 2048, 4096]:
    a = Tensor.rand(size, size).realize()
    b = Tensor.rand(size, size).realize()

    # Warmup
    (a @ b).realize()

    start = time.perf_counter()
    for _ in range(5):
        (a @ b).realize()
    elapsed = (time.perf_counter() - start) / 5

    tflops = (2 * size**3) / elapsed / 1e12
    print(f'{size}x{size} matmul: {elapsed*1000:.2f}ms ({tflops:.2f} TFLOPS)')
"
          '';
        };

        #############################################
        # GLM-4.7-Flash (via llama.cpp)
        #############################################

        # Using system llama.cpp if available, or building from source
        # Note: For Jetson, you may need to build with CUDA SM 8.7
        llama-cpp = pkgs.llamaPackages.llama-cpp.override {
          cudaSupport = true;
        } or pkgs.stdenv.mkDerivation {
          pname = "llama-cpp-jetson";
          version = "latest";
          src = pkgs.fetchFromGitHub {
            owner = "ggml-org";
            repo = "llama.cpp";
            rev = "master";
            sha256 = pkgs.lib.fakeSha256;
          };
          nativeBuildInputs = [ pkgs.cmake pkgs.ninja ];
          cmakeFlags = [ "-DGGML_CUDA=ON" "-DCMAKE_CUDA_ARCHITECTURES=87" ];
        };

        glm-download = pkgs.writeShellApplication {
          name = "glm-download";
          runtimeInputs = [ pkgs.curl ];
          text = ''
            MODEL_DIR="''${HOME}/.cache/glm"
            mkdir -p "$MODEL_DIR"
            QUANT="''${1:-Q4_K_M}"
            FILE="GLM-4.7-Flash-$QUANT.gguf"

            if [ -f "$MODEL_DIR/$FILE" ]; then
              echo "✅ Model exists: $MODEL_DIR/$FILE"
              exit 0
            fi

            echo "⬇️  Downloading GLM-4.7-Flash $QUANT..."
            curl -L -o "$MODEL_DIR/$FILE" \
              "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/$FILE" \
              --progress-bar

            echo "✅ Done: $MODEL_DIR/$FILE ($(du -h "$MODEL_DIR/$FILE" | cut -f1))"
          '';
        };

        glm-chat = pkgs.writeShellApplication {
          name = "glm-chat";
          runtimeInputs = [ pkgs.llama-cpp or llama-cpp ];
          text = ''
            MODEL="''${HOME}/.cache/glm/GLM-4.7-Flash-''${1:-Q4_K_M}.gguf"
            [ -f "$MODEL" ] || { echo "Run: glm-download ''${1:-Q4_K_M}"; exit 1; }

            echo "💬 GLM-4.7-Flash Chat"
            llama-cli -m "$MODEL" -ngl 999 -c 8192 -t 8 --interactive --color \
              -p "You are GLM-4.7-Flash, a helpful AI assistant."
          '';
        };

        glm-server = pkgs.writeShellApplication {
          name = "glm-server";
          runtimeInputs = [ pkgs.llama-cpp or llama-cpp ];
          text = ''
            MODEL="''${HOME}/.cache/glm/GLM-4.7-Flash-''${1:-Q4_K_M}.gguf"
            PORT="''${2:-8080}"
            [ -f "$MODEL" ] || { echo "Run: glm-download ''${1:-Q4_K_M}"; exit 1; }

            echo "🚀 GLM Server at http://localhost:$PORT/v1"
            llama-server -m "$MODEL" -ngl 999 -c 8192 --host 0.0.0.0 --port "$PORT" -t 8
          '';
        };

        pythonClient = python.withPackages (ps: [ ps.openai ps.rich ]);

        glm-client = pkgs.writeShellApplication {
          name = "glm-client";
          runtimeInputs = [ pythonClient ];
          text = ''
            python << 'EOF'
import sys
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="x")
msg = " ".join(sys.argv[1:]) or "Hello!"
print(f"You: {msg}\nGLM: ", end="", flush=True)
for c in client.chat.completions.create(
    model="glm", messages=[{"role":"user","content":msg}], stream=True, max_tokens=256
):
    if c.choices[0].delta.content: print(c.choices[0].delta.content, end="", flush=True)
print()
EOF
          '';
        };

      in {
        #############################################
        # PACKAGES
        #############################################
        packages = {
          # TinyGrad
          tinygrad = tinygrad;
          tinygrad-llama = tinygrad-llama;
          tinygrad-benchmark = tinygrad-benchmark;

          # GLM
          glm-download = glm-download;
          glm-chat = glm-chat;
          glm-server = glm-server;
          glm-client = glm-client;

          default = tinygrad-llama;
        };

        #############################################
        # APPS
        #############################################
        apps = {
          # TinyGrad
          llama = { type = "app"; program = "${tinygrad-llama}/bin/tinygrad-llama"; };
          tinygrad-bench = { type = "app"; program = "${tinygrad-benchmark}/bin/tinygrad-bench"; };

          # GLM
          glm = { type = "app"; program = "${glm-chat}/bin/glm-chat"; };
          glm-server = { type = "app"; program = "${glm-server}/bin/glm-server"; };
          glm-download = { type = "app"; program = "${glm-download}/bin/glm-download"; };
          glm-client = { type = "app"; program = "${glm-client}/bin/glm-client"; };

          default = { type = "app"; program = "${tinygrad-llama}/bin/tinygrad-llama"; };
        };

        #############################################
        # DEV SHELLS
        #############################################
        devShells = {
          # Quick development shell
          default = pkgs.mkShell {
            name = "llm-dev";
            buildInputs = [
              pythonWithTinygrad
              tinygrad-llama
              tinygrad-benchmark
              glm-download
              glm-chat
              glm-server
              glm-client
              pkgs.nvtop
              pkgs.htop
              pkgs.curl
            ];

            shellHook = ''
              echo ""
              echo "╔══════════════════════════════════════════════════════════════════╗"
              echo "║           🧠 Jetson LLM Inference Environment                    ║"
              echo "╠══════════════════════════════════════════════════════════════════╣"
              echo "║                                                                  ║"
              echo "║  TinyGrad (Llama 3.2):                                           ║"
              echo "║    tinygrad-llama 1B int8     # Run Llama 3.2 1B                 ║"
              echo "║    tinygrad-bench             # GPU benchmark                    ║"
              echo "║                                                                  ║"
              echo "║  GLM-4.7-Flash:                                                  ║"
              echo "║    glm-download Q4_K_M        # Download model                   ║"
              echo "║    glm-chat                   # Interactive chat                 ║"
              echo "║    glm-server                 # OpenAI-compatible API            ║"
              echo "║    glm-client \"Hi!\"           # Query the API                    ║"
              echo "║                                                                  ║"
              echo "╚══════════════════════════════════════════════════════════════════╝"
              echo ""
              export CUDA=1
            '';
          };

          # TinyGrad focused
          tinygrad = pkgs.mkShell {
            name = "tinygrad-shell";
            buildInputs = [ pythonWithTinygrad tinygrad-llama tinygrad-benchmark pkgs.nvtop ];
            shellHook = ''
              echo "🔶 TinyGrad Dev Shell"
              echo "   pip install -e vendor/tinygrad (for editable install)"
              export CUDA=1
              export TINYGRAD_DIR="${tinygrad-src}"
            '';
          };

          # GLM focused
          glm = pkgs.mkShell {
            name = "glm-shell";
            buildInputs = [ glm-download glm-chat glm-server glm-client pythonClient pkgs.nvtop ];
            shellHook = ''
              echo "🌟 GLM-4.7-Flash Shell"
              echo "   glm-download Q4_K_M && glm-chat"
              export GLM_MODEL_DIR="$HOME/.cache/glm"
            '';
          };
        };

        #############################################
        # OVERLAY
        #############################################
        overlays.default = final: prev: {
          jetson-llm = {
            inherit tinygrad tinygrad-llama tinygrad-benchmark;
            inherit glm-download glm-chat glm-server glm-client;
          };
        };
      }
    )

    // {
      # Cross-system templates and documentation
      templates = {
        default = {
          path = ./.;
          description = "Jetson LLM inference with TinyGrad and GLM-4.7-Flash";
        };
      };
    };
}

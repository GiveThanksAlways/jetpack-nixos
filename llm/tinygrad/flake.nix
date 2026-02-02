{
  description = "TinyGrad LLM Development Environment for Jetson Orin AGX";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "aarch64-linux" "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        pythonEnv = pkgs.python311.withPackages (ps: with ps; [
          # Core dependencies for tinygrad
          numpy
          pillow
          tqdm
          requests

          # LLM dependencies
          tiktoken
          sentencepiece
          safetensors

          # Development tools
          pip
          setuptools
          wheel
        ]);

        # Convenience script to run llama3
        llama-runner = pkgs.writeShellScriptBin "llama-run" ''
          set -e
          export CUDA=1
          export DEBUG=''${DEBUG:-0}

          TINYGRAD_DIR="''${TINYGRAD_DIR:-$HOME/jetpack-nixos/vendor/tinygrad}"
          MODEL_SIZE="''${1:-1B}"
          QUANTIZE="''${2:-int8}"

          cd "$TINYGRAD_DIR"

          echo "🦙 Running Llama 3.2 $MODEL_SIZE (quantize: $QUANTIZE)"
          echo "   Using CUDA backend on Jetson"
          echo ""

          python examples/llama3.py \
            --download_model \
            --size "$MODEL_SIZE" \
            --quantize "$QUANTIZE" \
            --no_api \
            --temperature 0.7 \
            "''${@:3}"
        '';

        # Quick device check
        device-check = pkgs.writeShellScriptBin "tinygrad-check" ''
          set -e
          export CUDA=1

          TINYGRAD_DIR="''${TINYGRAD_DIR:-$HOME/jetpack-nixos/vendor/tinygrad}"
          cd "$TINYGRAD_DIR"

          echo "🔍 Checking TinyGrad CUDA device..."
          python -c "
from tinygrad import Device
from tinygrad.helpers import getenv
print(f'Default device: {Device.DEFAULT}')
print(f'CUDA available: {\"CUDA\" in Device._devices or \"NV\" in Device._devices}')

# Quick tensor test
from tinygrad import Tensor
x = Tensor([1, 2, 3]).cuda() if hasattr(Tensor, 'cuda') else Tensor([1, 2, 3])
print(f'Tensor test: {x.numpy()}')
print('✅ TinyGrad ready!')
"
        '';

      in {
        # === DEV SHELL (Approach 1) ===
        # Simple dev environment for manual exploration
        # Usage: nix develop
        devShells.default = pkgs.mkShell {
          name = "tinygrad-dev";

          buildInputs = [
            pythonEnv
            llama-runner
            device-check

            # Build tools
            pkgs.gcc
            pkgs.gnumake
            pkgs.cmake
            pkgs.git

            # Debugging
            pkgs.htop
            pkgs.nvtop
          ];

          shellHook = ''
            echo ""
            echo "╔════════════════════════════════════════════════════════════╗"
            echo "║  🧠 TinyGrad LLM Dev Environment                           ║"
            echo "╠════════════════════════════════════════════════════════════╣"
            echo "║                                                            ║"
            echo "║  Quick Start:                                              ║"
            echo "║    1. pip install -e vendor/tinygrad                       ║"
            echo "║    2. tinygrad-check          # Verify CUDA works          ║"
            echo "║    3. llama-run 1B int8       # Run Llama 3.2 1B           ║"
            echo "║                                                            ║"
            echo "║  Manual:                                                   ║"
            echo "║    cd vendor/tinygrad                                      ║"
            echo "║    CUDA=1 python examples/llama3.py --download_model       ║"
            echo "║                                                            ║"
            echo "║  Models: 1B, 8B (70B/405B need more VRAM)                  ║"
            echo "║  Quantize: int8, nf4, float16, fp8                         ║"
            echo "╚════════════════════════════════════════════════════════════╝"
            echo ""

            export CUDA=1
            export TINYGRAD_DIR="$PWD/vendor/tinygrad"
            export PYTHONPATH="$TINYGRAD_DIR:$PYTHONPATH"
          '';
        };

        # Basic packages
        packages = {
          llama-runner = llama-runner;
          device-check = device-check;
        };
      }
    );
}

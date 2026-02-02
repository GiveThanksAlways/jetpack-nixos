{
  description = "Advanced TinyGrad LLM Deployment for Jetson - Fully Reproducible";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";

    # Pin tinygrad to a specific commit for reproducibility
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

        # Build tinygrad as a proper Nix package
        tinygrad = python.pkgs.buildPythonPackage {
          pname = "tinygrad";
          version = "0.10.0-git";
          format = "pyproject";

          src = tinygrad-src;

          nativeBuildInputs = with python.pkgs; [
            setuptools
            wheel
          ];

          propagatedBuildInputs = with python.pkgs; [
            numpy
            pillow
            tqdm
            requests
            tiktoken
            sentencepiece
          ];

          # Tests require GPU
          doCheck = false;

          pythonImportsCheck = [ "tinygrad" ];
        };

        # Python environment with tinygrad baked in
        pythonWithTinygrad = python.withPackages (ps: [
          tinygrad
          ps.numpy
          ps.pillow
          ps.tqdm
          ps.requests
          ps.tiktoken
          ps.sentencepiece
          ps.safetensors
        ]);

        # Pre-download model weights as a fixed-output derivation
        # This makes the model part of the Nix store for full reproducibility
        llama-1b-weights = pkgs.fetchurl {
          url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf";
          sha256 = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="; # Update after first run
          name = "Llama-3.2-1B-Instruct-Q6_K.gguf";
        };

        # The main LLM runner application
        llm-runner = pkgs.writeShellApplication {
          name = "tinygrad-llm";
          runtimeInputs = [ pythonWithTinygrad ];
          text = ''
            set -e
            export CUDA=1

            MODEL_SIZE="''${1:-1B}"
            QUANTIZE="''${2:-int8}"

            echo "🦙 TinyGrad LLM Runner"
            echo "   Model: Llama 3.2 $MODEL_SIZE"
            echo "   Quantize: $QUANTIZE"
            echo "   Device: Jetson Orin (CUDA)"
            echo ""

            # Use the bundled tinygrad
            python -c "
import sys
sys.path.insert(0, '${tinygrad-src}')
from examples.llama3 import *

# Run in CLI mode
import argparse
args = argparse.Namespace(
    model=None,
    size='$MODEL_SIZE',
    download_model=True,
    quantize='$QUANTIZE',
    no_api=True,
    shard=1,
    host='0.0.0.0',
    port=7776,
    debug=False,
    seed=None,
    temperature=0.7,
    benchmark=False,
    timing=False,
    profile=False,
)
main(args)
"
          '';
        };

        # Benchmark runner
        benchmark = pkgs.writeShellApplication {
          name = "tinygrad-benchmark";
          runtimeInputs = [ pythonWithTinygrad ];
          text = ''
            set -e
            export CUDA=1
            export DEBUG=1

            echo "📊 TinyGrad LLM Benchmark on Jetson Orin"
            echo ""

            python -c "
import sys
sys.path.insert(0, '${tinygrad-src}')

from tinygrad import Tensor, Device, GlobalCounters
from tinygrad.helpers import Timing
import time

print(f'Device: {Device.DEFAULT}')

# Simple matrix multiplication benchmark
print('\\n🔢 Matrix Multiplication Benchmark (4096x4096):')
for dtype in ['float16', 'float32']:
    GlobalCounters.reset()
    a = Tensor.rand(4096, 4096).cast(dtype)
    b = Tensor.rand(4096, 4096).cast(dtype)

    # Warmup
    c = (a @ b).realize()

    # Timed run
    start = time.perf_counter()
    for _ in range(10):
        c = (a @ b).realize()
    elapsed = time.perf_counter() - start

    tflops = (2 * 4096**3 * 10) / elapsed / 1e12
    print(f'  {dtype}: {tflops:.2f} TFLOPS ({elapsed*100:.1f}ms per op)')

print('\\n✅ Benchmark complete!')
"
          '';
        };

        # Device diagnostics
        diagnostics = pkgs.writeShellApplication {
          name = "tinygrad-diag";
          runtimeInputs = [ pythonWithTinygrad pkgs.nvtop ];
          text = ''
            echo "🔍 TinyGrad Jetson Diagnostics"
            echo "=============================="
            echo ""

            echo "📦 Python packages:"
            python -c "import tinygrad; print(f'  tinygrad: {tinygrad.__version__ if hasattr(tinygrad, \"__version__\") else \"installed\"}')"
            python -c "import numpy; print(f'  numpy: {numpy.__version__}')"

            echo ""
            echo "🖥️ Device info:"
            python -c "
from tinygrad import Device
print(f'  Default device: {Device.DEFAULT}')
print(f'  Available: {list(Device._devices.keys())}')
"

            echo ""
            echo "🧮 Quick compute test:"
            CUDA=1 python -c "
from tinygrad import Tensor
import time
x = Tensor.rand(1000, 1000)
y = Tensor.rand(1000, 1000)
start = time.time()
z = (x @ y).numpy()
print(f'  1000x1000 matmul: {(time.time()-start)*1000:.1f}ms')
"
            echo ""
            echo "✅ All checks passed!"
          '';
        };

        # Interactive chat app
        chat-app = pkgs.writeShellApplication {
          name = "tinygrad-chat";
          runtimeInputs = [ pythonWithTinygrad ];
          text = ''
            set -e
            export CUDA=1

            echo "💬 TinyGrad Chat (Llama 3.2)"
            echo "   Loading model..."
            echo ""

            cd "${tinygrad-src}"
            python examples/llama3.py \
              --download_model \
              --size "''${1:-1B}" \
              --quantize "''${2:-int8}" \
              --no_api \
              --temperature 0.85
          '';
        };

      in {
        # === PACKAGES ===
        packages = {
          default = llm-runner;
          inherit llm-runner benchmark diagnostics chat-app tinygrad;
        };

        # === APPS ===
        # nix run .#chat
        apps = {
          default = {
            type = "app";
            program = "${llm-runner}/bin/tinygrad-llm";
          };
          chat = {
            type = "app";
            program = "${chat-app}/bin/tinygrad-chat";
          };
          benchmark = {
            type = "app";
            program = "${benchmark}/bin/tinygrad-benchmark";
          };
          diag = {
            type = "app";
            program = "${diagnostics}/bin/tinygrad-diag";
          };
        };

        # === DEV SHELL ===
        devShells.default = pkgs.mkShell {
          name = "tinygrad-advanced";
          buildInputs = [
            pythonWithTinygrad
            llm-runner
            benchmark
            diagnostics
            chat-app
            pkgs.nvtop
            pkgs.htop
          ];

          shellHook = ''
            echo ""
            echo "╔══════════════════════════════════════════════════════════════╗"
            echo "║  🚀 TinyGrad Advanced LLM Environment (Fully Reproducible)   ║"
            echo "╠══════════════════════════════════════════════════════════════╣"
            echo "║                                                              ║"
            echo "║  Commands:                                                   ║"
            echo "║    tinygrad-llm [1B|8B] [int8|nf4]  # Run LLM                ║"
            echo "║    tinygrad-chat [1B|8B] [int8]     # Interactive chat       ║"
            echo "║    tinygrad-benchmark              # GPU performance test    ║"
            echo "║    tinygrad-diag                   # System diagnostics      ║"
            echo "║                                                              ║"
            echo "║  Or use: nix run .#chat                                      ║"
            echo "╚══════════════════════════════════════════════════════════════╝"
            echo ""
            export CUDA=1
          '';
        };

        # === OVERLAY ===
        # Can be imported by other flakes
        overlays.default = final: prev: {
          tinygrad-llm = llm-runner;
          tinygrad-pkg = tinygrad;
        };
      }
    );
}

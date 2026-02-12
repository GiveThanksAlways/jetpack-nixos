{
  description = "MLC LLM dev shell for Jetson Orin AGX (native CUDA inference)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    jetpack-nixos.url = "github:anduril/jetpack-nixos";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, jetpack-nixos, flake-utils }:
    flake-utils.lib.eachSystem [ "aarch64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          overlays = [ jetpack-nixos.overlays.default ];
        };
        jetpack = pkgs.nvidia-jetpack6;
        cuda = jetpack.cudaPackages;
      in
      {
        devShells.default = pkgs.mkShell {
          name = "mlc-llm-orin";

          buildInputs = [
            pkgs.python3
            pkgs.python3Packages.pip
            pkgs.python3Packages.virtualenv
            pkgs.python3Packages.numpy
            pkgs.cmake
            pkgs.ninja
            pkgs.git
            pkgs.curl
            pkgs.wget
            pkgs.pkg-config
            pkgs.zlib
            pkgs.openssl

            # CUDA toolchain
            (pkgs.lib.getLib cuda.cuda_cudart)
            (pkgs.lib.getLib cuda.libcublas)
            (pkgs.lib.getLib cuda.libcusparse)
            (pkgs.lib.getLib cuda.libcusolver)
            (pkgs.lib.getLib cuda.libcufft)
            (pkgs.lib.getLib cuda.libcurand)
            (pkgs.lib.getLib cuda.cuda_nvrtc)
            (pkgs.lib.getLib cuda.libnvjitlink)
            (pkgs.lib.getLib cuda.cudnn)
          ];

          CUDA_HOME = "${pkgs.lib.getDev cuda.cuda_cudart}";
          CUDA_PATH = "${pkgs.lib.getDev cuda.cuda_cudart}";

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            (pkgs.lib.getLib cuda.cuda_cudart)
            (pkgs.lib.getLib cuda.libcublas)
            (pkgs.lib.getLib cuda.libcusparse)
            (pkgs.lib.getLib cuda.libcusolver)
            (pkgs.lib.getLib cuda.libcufft)
            (pkgs.lib.getLib cuda.libcurand)
            (pkgs.lib.getLib cuda.cuda_nvrtc)
            (pkgs.lib.getLib cuda.libnvjitlink)
            (pkgs.lib.getLib cuda.cudnn)
            jetpack.l4t-cuda
            jetpack.l4t-core
            pkgs.stdenv.cc.cc
            pkgs.zlib
          ];

          shellHook = ''
            echo ""
            echo "=== MLC LLM dev shell (Orin AGX / CUDA 12.6) ==="
            echo ""

            # Setup Python venv for pip-installed packages
            VENV_DIR="$PWD/.venv"
            if [ ! -d "$VENV_DIR" ]; then
              echo "Creating Python venv in .venv/ ..."
              python3 -m venv "$VENV_DIR" --system-site-packages
              source "$VENV_DIR/bin/activate"
              echo "Installing MLC LLM..."
              # MLC LLM pre-built wheels for aarch64 + CUDA
              pip install --upgrade pip
              pip install mlc-llm mlc-ai-nightly 2>/dev/null || {
                echo ""
                echo "NOTE: Pre-built wheel not available. Installing from source..."
                echo "  pip install mlc-ai-nightly -f https://mlc.ai/wheels"
                echo ""
                pip install mlc-ai-nightly -f https://mlc.ai/wheels || {
                  echo "WARNING: MLC LLM installation failed."
                  echo "Try manually: pip install mlc-ai-nightly -f https://mlc.ai/wheels"
                }
              }
            else
              source "$VENV_DIR/bin/activate"
            fi

            echo ""
            echo "Quick test:"
            echo "  python3 -c 'import mlc_llm; print(mlc_llm.__version__)'"
            echo ""
            echo "Run inference:"
            echo "  mlc_llm chat HF://mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC"
            echo ""
            echo "Benchmark (decode throughput):"
            echo "  python3 bench_mlc_llm.py"
            echo ""
          '';
        };
      }
    );
}

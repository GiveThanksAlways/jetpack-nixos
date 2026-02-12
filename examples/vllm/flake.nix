{
  description = "vLLM inference engine for Jetson Orin AGX (native + Docker fallback)";

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
        # Native dev shell (tries pip install first, falls back to Docker instructions)
        devShells.default = pkgs.mkShell {
          name = "vllm-orin";

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
            (pkgs.lib.getDev cuda.cuda_cudart)
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
            echo "=== vLLM dev shell (Orin AGX / CUDA 12.6) ==="
            echo ""

            # Setup Python venv
            VENV_DIR="$PWD/.venv-vllm"
            if [ ! -d "$VENV_DIR" ]; then
              echo "Creating Python venv in .venv-vllm/ ..."
              python3 -m venv "$VENV_DIR" --system-site-packages
              source "$VENV_DIR/bin/activate"
              echo "Attempting native vLLM installation..."
              pip install --upgrade pip
              
              # Try pre-built wheel first (Jetson-specific)
              pip install vllm 2>/dev/null && {
                echo "vLLM installed successfully (native wheel)!"
              } || {
                echo ""
                echo "=========================================="
                echo "Native vLLM install failed (expected on Jetson NixOS)."
                echo ""
                echo "OPTIONS:"
                echo ""
                echo "  1. Docker (recommended):"
                echo "     ./run-vllm-docker.sh"
                echo ""
                echo "  2. Build from source (slow, may fail):"  
                echo "     pip install vllm --no-build-isolation"
                echo ""
                echo "  3. NVIDIA L4T container:"
                echo "     See Dockerfile.jetson in this directory"
                echo "=========================================="
              }
            else
              source "$VENV_DIR/bin/activate"
            fi

            echo ""
            echo "Quick test:"
            echo "  python3 -c 'import vllm; print(vllm.__version__)'"
            echo ""
            echo "Run server:"
            echo "  python3 -m vllm.entrypoints.openai.api_server \\"
            echo "    --model meta-llama/Llama-3.2-1B-Instruct \\"
            echo "    --max-model-len 2048 --dtype half"
            echo ""
            echo "Benchmark:"
            echo "  python3 bench_vllm.py"
            echo ""
            echo "Docker alternative:"
            echo "  ./run-vllm-docker.sh"
            echo ""
          '';
        };

        # Docker-based shell (for when native doesn't work)
        devShells.docker = pkgs.mkShell {
          name = "vllm-docker";
          buildInputs = [
            pkgs.docker
            pkgs.docker-compose
            pkgs.curl
            pkgs.jq
          ];
          shellHook = ''
            echo ""
            echo "=== vLLM Docker shell ==="
            echo ""
            echo "Start vLLM server:"
            echo "  ./run-vllm-docker.sh"
            echo ""
            echo "Benchmark against running server:"
            echo "  python3 bench_vllm.py --server http://localhost:8000"
            echo ""
          '';
        };
      }
    );
}

{
  description = "Tinygrad dev shell for Jetson Orin AGX (CUDA + NV backends)";

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

        pythonEnv = pkgs.python3.withPackages (ps: [
          ps.numpy
          ps.tqdm
          ps.requests
          ps.pillow
          ps.sentencepiece
          ps.tiktoken
          ps.safetensors
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          name = "tinygrad-orin";

          buildInputs = [
            pythonEnv
            pkgs.git
          ];

          # CUDA libs needed by tinygrad CUDA and NV backends
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            cuda.cuda_cudart
            cuda.libcublas
            cuda.libcusparse
            cuda.libcusolver
            cuda.libcufft
            cuda.libcurand
            cuda.cuda_nvrtc
            cuda.cudnn
            pkgs.stdenv.cc.cc
            pkgs.linuxPackages.nvidia_x11
          ];

          shellHook = ''
            echo ""
            echo "=== tinygrad dev shell (Orin AGX / CUDA 12.6) ==="
            echo ""

            # Clone tinygrad if not present
            if [ ! -d "$PWD/tinygrad" ]; then
              echo "Cloning tinygrad..."
              git clone --depth 1 https://github.com/tinygrad/tinygrad.git
            fi

            # pip-install tinygrad in editable mode inside a venv
            VENV_DIR="$PWD/.venv"
            if [ ! -d "$VENV_DIR" ]; then
              python3 -m venv "$VENV_DIR"
            fi
            source "$VENV_DIR/bin/activate"

            if ! python3 -c "import tinygrad" 2>/dev/null; then
              echo "Installing tinygrad (editable)..."
              pip install -e ./tinygrad 2>/dev/null
            fi

            echo ""
            echo "Quick test:"
            echo "  CUDA=1 python3 -c 'from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())'"
            echo ""
            echo "LLM examples (from tinygrad/examples/):"
            echo "  # GPT-2 (small, good first test)"
            echo "  CUDA=1 python3 tinygrad/examples/gpt2.py --count 20"
            echo ""
            echo "  # LLaMA (needs model weights -- see tinygrad docs)"
            echo "  CUDA=1 python3 tinygrad/examples/llama.py --prompt 'Hello world'"
            echo ""
            echo "NV backend (experimental, may not work yet on Orin):"
            echo "  NV=1 python3 -c 'from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())'"
            echo ""
          '';
        };
      }
    );
}

{
  description = "LLM dev shell for Jetson Orin AGX";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "aarch64-linux" "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        pythonEnv = pkgs.python311.withPackages (ps: with ps; [
          numpy pillow tqdm requests pip setuptools wheel
          tiktoken sentencepiece safetensors
        ]);

      in {
        devShells.default = pkgs.mkShell {
          name = "llm";
          buildInputs = with pkgs; [
            pythonEnv
            git curl wget
            cmake gcc gnumake ninja  # for llama.cpp
            htop
          ];

          shellHook = ''
            echo ""
            echo "=== LLM Dev Shell ==="
            echo ""
            echo "Llama 3.2 (TinyGrad):"
            echo "  pip install -e ../vendor/tinygrad"
            echo "  CUDA=1 python ../vendor/tinygrad/examples/llama3.py --download_model --size 1B --no_api"
            echo ""
            echo "GLM-4.7-Flash (manual llama.cpp):"
            echo "  # Download model (~17GB)"
            echo "  curl -L -C - -o ~/.cache/glm/GLM-Q4.gguf \\"
            echo "    'https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf'"
            echo "  # Build llama.cpp with CUDA"
            echo "  git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp"
            echo "  cmake -B build -DGGML_CUDA=ON && cmake --build build -j"
            echo "  ./build/bin/llama-cli -m ~/.cache/glm/GLM-Q4.gguf -ngl 999 --interactive"
            echo ""
            export CUDA=1
            mkdir -p ~/.cache/glm
          '';
        };
      }
    );
}

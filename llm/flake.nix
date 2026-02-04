{
  description = "Minimal LLM dev shell for Jetson";

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

        # Minimal python - just the basics, install rest via pip
        pythonEnv = pkgs.python311.withPackages (ps: with ps; [
          pip setuptools wheel
          numpy
        ]);

      in {
        devShells.default = pkgs.mkShell {
          name = "llm";
          buildInputs = with pkgs; [
            pythonEnv
            git curl wget
            cmake gcc gnumake ninja  # for llama.cpp
          ];

          shellHook = ''
            echo ""
            echo "=== LLM Dev Shell (Minimal) ==="
            echo ""
            echo "First time setup:"
            echo "  pip install tqdm requests pillow tiktoken"
            echo ""
            echo "Llama 3.2 (TinyGrad):"
            echo "  pip install -e ../vendor/tinygrad"
            echo "  CUDA=1 python ../vendor/tinygrad/examples/llama3.py --download_model --size 1B --no_api"
            echo ""
            export CUDA=1
          '';
        };
      }
    );
}

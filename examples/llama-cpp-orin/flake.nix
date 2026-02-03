{
  description = "llama.cpp dev shell for Jetson Orin AGX (local use)";

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
            cudaCapabilities = [ "8.7" ]; # Orin AGX
          };
        };

        # llama.cpp with CUDA - uses system CUDA from jetpack-nixos
        llama-cpp-cuda = pkgs.llama-cpp.override {
          cudaSupport = true;
        };
      in
      {
        # Main dev shell - just run: nix develop
        devShells.default = pkgs.mkShell {
          name = "llama-cpp-orin";

          buildInputs = [
            llama-cpp-cuda
            pkgs.curl  # For downloading models
            pkgs.htop  # For monitoring
          ];

          shellHook = ''
            echo ""
            echo "=== llama.cpp for Orin AGX ==="
            echo ""
            echo "Qwen3-Coder-Next Q5_K_XL (~57GB):"
            echo "  llama-cli -hf unsloth/Qwen3-Coder-Next-GGUF:Q5_K_XL --gpu-layers 999"
            echo ""
            echo "Other commands:"
            echo "  llama-server  - Start HTTP API server"
            echo "  llama-bench   - Run benchmarks"
            echo ""
            echo "Smaller quants: Q4_K_M (~35GB), Q5_K_M (~45GB)"
            echo ""

            # Ensure HuggingFace cache dir exists
            mkdir -p ~/.cache/huggingface
          '';
        };

        # Package export
        packages = {
          default = llama-cpp-cuda;
          llama-cpp = llama-cpp-cuda;
        };

        # Direct app execution
        apps.default = {
          type = "app";
          program = "${llama-cpp-cuda}/bin/llama-cli";
        };
      }
    );
}

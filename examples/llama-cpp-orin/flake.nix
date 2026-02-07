{
  description = "llama.cpp for Jetson Orin AGX";

  inputs = {
    llama-cpp.url = "github:ggml-org/llama.cpp";
  };

  outputs = { self, llama-cpp, ... }:
    let
      system = "aarch64-linux";
      # Use llama-cpp's nixpkgs to avoid derivation hash mismatch
      pkgs = llama-cpp.inputs.nixpkgs.legacyPackages.${system};
      llamaCpp = (llama-cpp.packages.${system}.jetson-orin).overrideAttrs (old: {
        buildInputs = old.buildInputs ++ [ pkgs.openssl ];
        cmakeFlags = old.cmakeFlags ++ [ "-DLLAMA_OPENSSL=ON" ];
      });
    in
    {
      packages.${system}.default = llamaCpp;

      devShells.${system}.default = pkgs.mkShell {
        name = "llama-cpp-orin";
        packages = [ llamaCpp ];
        shellHook = ''
          echo ""
          echo "=== llama.cpp for Jetson Orin AGX ==="
          echo "Version: $(llama-cli --version 2>&1 | head -1)"
          echo ""
          echo "Run Qwen3-Coder-Next (with HuggingFace download):"
          echo "  llama-cli -hf unsloth/Qwen3-Coder-Next-GGUF:Q5_K_XL --gpu-layers 999"
          echo ""
          echo "Run Qwen3-Coder-Next (local cached model):"
          echo "  llama-cli -m ~/.cache/llama.cpp/unsloth_Qwen3-Coder-Next-GGUF_UD-Q5_K_XL_Qwen3-Coder-Next-UD-Q5_K_XL-00001-of-00002.gguf --gpu-layers 999"
          echo ""
          echo "Start API server (OpenCode / vscode):"
          echo "  llama-server -m ~/.cache/llama.cpp/unsloth_Qwen3-Coder-Next-GGUF_UD-Q5_K_XL_Qwen3-Coder-Next-UD-Q5_K_XL-00001-of-00002.gguf --gpu-layers 999 --host 0.0.0.0 --port 5000"
          echo ""
          echo "From same Jetson: http://localhost:5000"
          echo "From remote PC: http://<JETSON_IP>:5000 (e.g., http://192.168.1.100:5000)"
          echo ""
          echo "Other commands: llama-bench, llama-quantize, llama-embedding"
          echo ""
        '';
      };
    };
}

{
  description = "llama.cpp with CUDA support for NVIDIA Jetson Orin AGX";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    jetpack-nixos = {
      url = "github:anduril/jetpack-nixos/master";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, jetpack-nixos, ... }:
    let
      # Orin AGX configuration
      system = "aarch64-linux";

      # Import nixpkgs with jetpack-nixos overlay and CUDA configuration
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
          cudaCapabilities = [ "8.7" ]; # Orin AGX CUDA capability
        };
        overlays = [ jetpack-nixos.overlays.default ];
      };

      # llama-cpp with CUDA backend enabled
      llama-cpp-cuda = pkgs.llama-cpp.override {
        cudaSupport = true;
      };
    in
    {
      # NixOS configuration module for Orin AGX
      nixosConfigurations.orin-agx-llama = nixpkgs.lib.nixosSystem {
        inherit system;
        modules = [
          jetpack-nixos.nixosModules.default
          ({ pkgs, ... }: {
            # Jetson Orin AGX configuration
            hardware.nvidia-jetpack = {
              enable = true;
              som = "orin-agx";
              carrierBoard = "devkit";
            };

            # Enable GPU support - required for CUDA
            hardware.graphics.enable = true;

            # Include llama.cpp with CUDA support
            environment.systemPackages = [
              llama-cpp-cuda
            ];

            # Basic system configuration
            nixpkgs.config.allowUnfree = true;

            # Minimal required filesystems (adjust for your setup)
            fileSystems."/" = {
              device = "/dev/disk/by-label/nixos";
              fsType = "ext4";
            };

            boot.loader.systemd-boot.enable = true;
            boot.loader.efi.canTouchEfiVariables = true;

            # System stateVersion - adjust as needed
            system.stateVersion = "25.11";
          })
        ];
      };

      # Development shell with llama-cpp and CUDA tools
      devShells.${system}.default = pkgs.mkShell {
        name = "llama-cpp-orin-shell";
        buildInputs = [
          llama-cpp-cuda
          pkgs.nvidia-jetpack.cudaPackages.cuda_nvcc
        ];

        shellHook = ''
          echo "llama.cpp with CUDA support for Orin AGX"
          echo ""
          echo "Run inference with:"
          echo "  llama-cli -hf unsloth/Qwen3-Coder-Next-GGUF:Q5_K_XL"
          echo ""
          echo "Available commands:"
          echo "  llama-cli      - Main CLI for inference"
          echo "  llama-server   - HTTP API server"
          echo "  llama-bench    - Benchmarking tool"
        '';
      };

      # Standalone package
      packages.${system} = {
        default = llama-cpp-cuda;
        llama-cpp = llama-cpp-cuda;
      };

      # App for easy running
      apps.${system}.default = {
        type = "app";
        program = "${llama-cpp-cuda}/bin/llama-cli";
      };
    };
}

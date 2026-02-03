{
  description = "Advanced llama.cpp configuration for NVIDIA Jetson Orin AGX with overlay-based customization";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    jetpack-nixos = {
      url = "github:anduril/jetpack-nixos/master";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, jetpack-nixos, ... }:
    let
      system = "aarch64-linux";

      # Custom overlay for llama.cpp optimizations
      llamaOverlay = final: prev: {
        # Custom llama.cpp build with CUDA and optimizations for Orin
        llama-cpp-orin = prev.llama-cpp.override {
          cudaSupport = true;
          # Build optimizations for Orin AGX (sm_87)
        };

        # Convenience wrapper for running Qwen3-Coder-Next
        qwen3-coder-runner = final.writeShellScriptBin "qwen3-coder" ''
          set -euo pipefail

          MODEL_REPO="unsloth/Qwen3-Coder-Next-GGUF"
          MODEL_QUANT="''${1:-Q5_K_XL}"

          echo "Starting Qwen3-Coder-Next ($MODEL_QUANT)..."
          echo "Model: $MODEL_REPO:$MODEL_QUANT"
          echo ""

          exec ${final.llama-cpp-orin}/bin/llama-cli \
            -hf "$MODEL_REPO:$MODEL_QUANT" \
            --gpu-layers 999 \
            --ctx-size 32768 \
            "''${@:2}"
        '';

        # Server wrapper with sensible defaults for Orin
        qwen3-coder-server = final.writeShellScriptBin "qwen3-coder-server" ''
          set -euo pipefail

          MODEL_REPO="unsloth/Qwen3-Coder-Next-GGUF"
          MODEL_QUANT="''${1:-Q5_K_XL}"
          PORT="''${2:-8080}"

          echo "Starting Qwen3-Coder-Next API server..."
          echo "Model: $MODEL_REPO:$MODEL_QUANT"
          echo "Port: $PORT"
          echo ""

          exec ${final.llama-cpp-orin}/bin/llama-server \
            -hf "$MODEL_REPO:$MODEL_QUANT" \
            --gpu-layers 999 \
            --ctx-size 32768 \
            --host 0.0.0.0 \
            --port "$PORT" \
            "''${@:3}"
        '';

        # Benchmark utility
        llama-benchmark = final.writeShellScriptBin "llama-benchmark" ''
          set -euo pipefail

          MODEL_REPO="unsloth/Qwen3-Coder-Next-GGUF"
          MODEL_QUANT="''${1:-Q5_K_XL}"

          echo "Benchmarking Qwen3-Coder-Next ($MODEL_QUANT) on Orin AGX..."
          echo ""

          exec ${final.llama-cpp-orin}/bin/llama-bench \
            -hf "$MODEL_REPO:$MODEL_QUANT" \
            --gpu-layers 999 \
            -n 128 \
            -p 512 \
            "''${@:2}"
        '';
      };

      # Compose all overlays
      overlays = [
        jetpack-nixos.overlays.default
        llamaOverlay
      ];

      pkgs = import nixpkgs {
        inherit system overlays;
        config = {
          allowUnfree = true;
          cudaSupport = true;
          cudaCapabilities = [ "8.7" ];
        };
      };
    in
    {
      inherit overlays;

      # Reusable overlay for other flakes
      overlays.default = llamaOverlay;

      # NixOS module for Orin AGX with llama.cpp
      nixosModules.default = { config, lib, pkgs, ... }: {
        options.services.llama-cpp = {
          enable = lib.mkEnableOption "llama.cpp server for Orin AGX";

          model = lib.mkOption {
            type = lib.types.str;
            default = "unsloth/Qwen3-Coder-Next-GGUF:Q5_K_XL";
            description = "HuggingFace model to serve";
          };

          port = lib.mkOption {
            type = lib.types.port;
            default = 8080;
            description = "Port for the API server";
          };

          host = lib.mkOption {
            type = lib.types.str;
            default = "0.0.0.0";
            description = "Host to bind to";
          };

          gpuLayers = lib.mkOption {
            type = lib.types.int;
            default = 999;
            description = "Number of layers to offload to GPU (999 = all)";
          };

          contextSize = lib.mkOption {
            type = lib.types.int;
            default = 32768;
            description = "Context size for inference";
          };

          extraArgs = lib.mkOption {
            type = lib.types.listOf lib.types.str;
            default = [ ];
            description = "Extra arguments to pass to llama-server";
          };
        };

        config = lib.mkIf config.services.llama-cpp.enable {
          systemd.services.llama-cpp = {
            description = "llama.cpp API Server";
            after = [ "network.target" ];
            wantedBy = [ "multi-user.target" ];

            serviceConfig = {
              Type = "simple";
              Restart = "on-failure";
              RestartSec = "10s";
              ExecStart = lib.concatStringsSep " " ([
                "${pkgs.llama-cpp-orin}/bin/llama-server"
                "-hf ${config.services.llama-cpp.model}"
                "--gpu-layers ${toString config.services.llama-cpp.gpuLayers}"
                "--ctx-size ${toString config.services.llama-cpp.contextSize}"
                "--host ${config.services.llama-cpp.host}"
                "--port ${toString config.services.llama-cpp.port}"
              ] ++ config.services.llama-cpp.extraArgs);

              # Security hardening
              NoNewPrivileges = true;
              ProtectSystem = "strict";
              ProtectHome = true;
              PrivateTmp = true;

              # Allow network access
              PrivateNetwork = false;
            };
          };
        };
      };

      # Full NixOS configuration for Orin AGX
      nixosConfigurations.orin-agx-llama = nixpkgs.lib.nixosSystem {
        inherit system;
        modules = [
          jetpack-nixos.nixosModules.default
          self.nixosModules.default
          ({ pkgs, ... }: {
            # Apply custom overlays
            nixpkgs.overlays = [ llamaOverlay ];

            # Jetson Orin AGX configuration
            hardware.nvidia-jetpack = {
              enable = true;
              som = "orin-agx";
              carrierBoard = "devkit";
            };

            # Enable GPU support
            hardware.graphics.enable = true;

            # Enable llama.cpp server (optional - set to true for automatic startup)
            services.llama-cpp.enable = false;

            # Include all llama tools
            environment.systemPackages = with pkgs; [
              llama-cpp-orin
              qwen3-coder-runner
              qwen3-coder-server
              llama-benchmark
            ];

            nixpkgs.config.allowUnfree = true;

            fileSystems."/" = {
              device = "/dev/disk/by-label/nixos";
              fsType = "ext4";
            };

            boot.loader.systemd-boot.enable = true;
            boot.loader.efi.canTouchEfiVariables = true;

            system.stateVersion = "25.11";
          })
        ];
      };

      # Development shell with all tools
      devShells.${system}.default = pkgs.mkShell {
        name = "llama-cpp-orin-advanced-shell";
        buildInputs = with pkgs; [
          llama-cpp-orin
          qwen3-coder-runner
          qwen3-coder-server
          llama-benchmark
          nvidia-jetpack.cudaPackages.cuda_nvcc
        ];

        shellHook = ''
          echo "╔════════════════════════════════════════════════════════════════╗"
          echo "║     llama.cpp Advanced Environment for Orin AGX                ║"
          echo "╚════════════════════════════════════════════════════════════════╝"
          echo ""
          echo "Available commands:"
          echo "  qwen3-coder [QUANT]      - Interactive chat (default: Q5_K_XL)"
          echo "  qwen3-coder-server [QUANT] [PORT] - Start API server"
          echo "  llama-benchmark [QUANT]  - Run performance benchmark"
          echo "  llama-cli                - Raw llama.cpp CLI"
          echo "  llama-server             - Raw llama.cpp server"
          echo ""
          echo "Quant options: Q4_K_M, Q5_K_M, Q5_K_XL, Q6_K, Q8_0"
          echo ""
          echo "Quick start:"
          echo "  qwen3-coder              # Start chatting!"
          echo ""
        '';
      };

      # Packages
      packages.${system} = {
        default = pkgs.llama-cpp-orin;
        llama-cpp = pkgs.llama-cpp-orin;
        qwen3-coder = pkgs.qwen3-coder-runner;
        qwen3-coder-server = pkgs.qwen3-coder-server;
        benchmark = pkgs.llama-benchmark;
      };

      # Apps
      apps.${system} = {
        default = {
          type = "app";
          program = "${pkgs.qwen3-coder-runner}/bin/qwen3-coder";
        };
        server = {
          type = "app";
          program = "${pkgs.qwen3-coder-server}/bin/qwen3-coder-server";
        };
        benchmark = {
          type = "app";
          program = "${pkgs.llama-benchmark}/bin/llama-benchmark";
        };
        llama-cli = {
          type = "app";
          program = "${pkgs.llama-cpp-orin}/bin/llama-cli";
        };
      };
    };
}

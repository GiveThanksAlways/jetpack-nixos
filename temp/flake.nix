# Flake configuration for Jetson AGX Orin Developer Kit
# =====================================================
#
# This file handles dependency fetching and version pinning.
# Use this WITH configuration.nix for the flake-based approach.
#
# INSTALLATION:
#   1. Copy BOTH this file AND configuration.nix to /mnt/etc/nixos/
#   2. Run: nixos-install --flake /mnt/etc/nixos#jetson
#
# POST-INSTALL UPDATES:
#   cd /etc/nixos
#   sudo nix flake update              # Get latest versions
#   sudo nixos-rebuild switch --flake .#jetson

{
  description = "NixOS configuration for Jetson AGX Orin";

  inputs = {
    # NixOS 25.11 (matches jetpack-nixos)
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";

    # Jetpack NixOS - NVIDIA Jetson support
    jetpack-nixos.url = "github:anduril/jetpack-nixos/master";
    jetpack-nixos.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, jetpack-nixos, ... }: {
    nixosConfigurations.jetson = nixpkgs.lib.nixosSystem {
      system = "aarch64-linux";
      modules = [
        # Jetpack NixOS module (provides hardware.nvidia-jetpack options)
        jetpack-nixos.nixosModules.default

        # Your system configuration
        ./configuration.nix
      ];
    };
  };
}

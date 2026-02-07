{
  description = "vLLM / llama.cpp / TabbyAPI serving on Jetson Orin AGX (NixOS)";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.11";
    jetpack-nixos.url = "github:GiveThanksAlways/jetpack-nixos";
  };

  outputs = { self, nixpkgs, jetpack-nixos, ... }:
    let
      system = "aarch64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
          cudaCapabilities = [ "8.7" ]; # Orin
        };
        overlays = [ jetpack-nixos.overlays.default ];
      };
    in
    {
      # Full NixOS system config for Orin AGX
      nixosConfigurations.orin-agx-vllm = nixpkgs.lib.nixosSystem {
        inherit system;
        modules = [
          jetpack-nixos.nixosModules.default
          ./configuration.nix
        ];
      };

      # Dev shell with vLLM + llama.cpp tools
      devShells.${system}.default = pkgs.mkShell {
        name = "vllm-serving-dev";
        packages = with pkgs; [
          python3
          python3Packages.pip
          curl
          jq
        ];
        shellHook = ''
          echo "vLLM Serving Dev Shell — Orin AGX"
          echo "Run: python -m vllm.entrypoints.openai.api_server --help"
        '';
      };
    };
}

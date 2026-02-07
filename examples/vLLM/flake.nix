{
  description = "vLLM for Jetson Orin AGX";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    jetpack-nixos = {
      url = "github:anduril/jetpack-nixos";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, jetpack-nixos, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
          overlays = [ jetpack-nixos.overlays.default ];
        };
      in
      {
        packages.default = pkgs.python3Packages.vllm;

        devShells.default = pkgs.mkShell {
          name = "vllm-jetson";
          buildInputs = with pkgs; [
            python3
            python3Packages.vllm
            cuda
          ];
          shellHook = ''
            echo "vLLM for Jetson Orin AGX"
            echo "Run: vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602"
          '';
        };
      }
    );
}
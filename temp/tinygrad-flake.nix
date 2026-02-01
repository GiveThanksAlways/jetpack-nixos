{
  description = "TinyGrad dev environment for Jetson AGX Orin";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = nixpkgs.legacyPackages.${system}; in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python311
            python311Packages.pip
            python311Packages.numpy
            gcc
            git
            cmake
            gnumake
          ];
          shellHook = ''echo "🔶 TinyGrad Dev (pip install -e ~/vendor/tinygrad)"'';
        };
      }
    );
}

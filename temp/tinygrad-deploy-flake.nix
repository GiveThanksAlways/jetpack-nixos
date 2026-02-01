{
  description = "TinyGrad build & deploy - compiles in Nix, runs via Python";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = nixpkgs.legacyPackages.${system};
          tinygrad = pkgs.buildPythonPackage rec {
            pname = "tinygrad";
            version = "0.1";
            src = ./vendor/tinygrad;
            propagatedBuildInputs = with pkgs.python311Packages; [
              numpy
              pillow
            ];
            buildPhase = "true";
            installPhase = ''
              mkdir -p $out/${pkgs.python311.sitePackages}
              cp -r . $out/${pkgs.python311.sitePackages}/tinygrad
            '';
          };
      in
      {
        packages.default = tinygrad;
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python311
            python311Packages.pip
            python311Packages.numpy
            gcc
            git
            cmake
            gnumake
          ] ++ [ tinygrad ];
          shellHook = ''echo "🚀 TinyGrad Deploy: CUDA=1 python examples/jetson_device_check.py"'';
        };
      }
    );
}

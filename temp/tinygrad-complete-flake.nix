{
  description = "TinyGrad complete - builds everything, runs everything";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = nixpkgs.legacyPackages.${system};
          tinygrad = pkgs.buildPythonApplication rec {
            pname = "tinygrad";
            version = "0.1";
            src = ./vendor/tinygrad;
            propagatedBuildInputs = with pkgs.python311Packages; [
              numpy
              pillow
              requests
              tqdm
            ];
            postInstall = ''
              mkdir -p $out/bin
              cat > $out/bin/tinygrad-device-check << 'SCRIPT'
              #!/bin/sh
              export CUDA=1
              exec python ${self}/vendor/tinygrad/examples/jetson_device_check.py "$@"
              SCRIPT
              chmod +x $out/bin/tinygrad-device-check
            '';
            meta.mainProgram = "tinygrad-device-check";
          };
      in
      {
        packages.default = tinygrad;
        apps.default = {
          type = "app";
          program = "${tinygrad}/bin/tinygrad-device-check";
        };
        devShells.default = pkgs.mkShell {
          buildInputs = [ tinygrad ];
          shellHook = ''echo "✅ TinyGrad ready: nix run to start"'';
        };
      }
    );
}

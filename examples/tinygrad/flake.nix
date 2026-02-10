{
  description = "Tinygrad dev shell for Jetson Orin AGX (CUDA + NV backends)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    jetpack-nixos.url = "github:anduril/jetpack-nixos";
    flake-utils.url = "github:numtide/flake-utils";
    tinygrad-src = {
      url = "github:tinygrad/tinygrad/cc9bf8ccbc0b7eb0e3b8510d475fa56263ef8cab";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, jetpack-nixos, flake-utils, tinygrad-src }:
    flake-utils.lib.eachSystem [ "aarch64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          overlays = [ jetpack-nixos.overlays.default ];
        };
        jetpack = pkgs.nvidia-jetpack6;
        cuda = jetpack.cudaPackages;

        # Build tinygrad from source using Nix
        tinygrad = pkgs.python3Packages.buildPythonPackage {
          pname = "tinygrad";
          version = "0.12.0";
          src = tinygrad-src;
          format = "pyproject";

          nativeBuildInputs = [
            pkgs.python3Packages.setuptools
          ];

          propagatedBuildInputs = [
            pkgs.python3Packages.numpy
            pkgs.python3Packages.tqdm
            pkgs.python3Packages.requests
          ];

          # Skip tests during build
          doCheck = false;

          pythonImportsCheck = [ "tinygrad" ];
        };

        pythonEnv = pkgs.python3.withPackages (ps: [
          tinygrad
          ps.numpy
          ps.tqdm
          ps.requests
          ps.pillow
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          name = "tinygrad-orin";

          buildInputs = [
            pythonEnv
            pkgs.git
            pkgs.clang  # tinygrad needs clang for CPU JIT compilation
          ];

          # Force clang as CC so tinygrad CPU backend works (gcc doesn't support --target)
          CC = "${pkgs.clang}/bin/clang";
          CXX = "${pkgs.clang}/bin/clang++";

          # tinygrad uses its own library finder (DLL.findlib) that searches hardcoded
          # FHS paths (/usr/lib, /lib64, etc.) which don't exist on NixOS.
          # It checks <NAME>_PATH env vars first, so we point those directly at the .so files.
          # lib.getLib returns the `lib` output if it exists, otherwise the default output.
          CUDA_PATH = "${jetpack.l4t-cuda}/lib/libcuda.so.1";
          NVRTC_PATH = "${pkgs.lib.getLib cuda.cuda_nvrtc}/lib/libnvrtc.so";
          NVJITLINK_PATH = "${pkgs.lib.getLib cuda.libnvjitlink}/lib/libnvJitLink.so";

          # LD_LIBRARY_PATH is still needed so the dynamic linker can resolve
          # transitive dependencies (e.g. libcudart, libcublas, libstdc++).
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            (pkgs.lib.getLib cuda.cuda_cudart)   # libcudart.so
            (pkgs.lib.getLib cuda.libcublas)      # libcublas.so
            (pkgs.lib.getLib cuda.libcusparse)    # libcusparse.so
            (pkgs.lib.getLib cuda.libcusolver)    # libcusolver.so
            (pkgs.lib.getLib cuda.libcufft)       # libcufft.so
            (pkgs.lib.getLib cuda.libcurand)      # libcurand.so
            (pkgs.lib.getLib cuda.cuda_nvrtc)     # libnvrtc.so
            (pkgs.lib.getLib cuda.libnvjitlink)   # libnvJitLink.so
            (pkgs.lib.getLib cuda.cudnn)          # libcudnn.so
            jetpack.l4t-cuda                      # libcuda.so (Jetson CUDA driver)
            jetpack.l4t-core                      # libnvcucompat.so
            pkgs.stdenv.cc.cc                     # libstdc++.so
          ];

          shellHook = ''
            echo ""
            echo "=== tinygrad dev shell (Orin AGX / CUDA 12.6) ==="
            echo ""
            echo "Quick test (CPU):"
            echo "  python3 -c 'from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())'"
            echo ""
            echo "Quick test (CUDA):"
            echo "  CUDA=1 python3 -c 'from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())'"
            echo ""
            echo "LLM examples (need to clone tinygrad repo for examples):"
            echo "  git clone --depth 1 https://github.com/tinygrad/tinygrad.git tinygrad-examples"
            echo "  CUDA=1 python3 tinygrad-examples/examples/gpt2.py --count 20"
            echo ""
          '';
        };
      }
    );
}

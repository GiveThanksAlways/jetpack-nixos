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

        # Pre-built PyTorch CPU wheel from PyTorch's official aarch64 builds.
        # This avoids the 1-4 hour source build that ps.torch triggers.
        # CPU-only is fine — torch is only used as a reference implementation
        # for correctness comparison in tinygrad's test suite (test_ops.py, test_nn.py).
        torch-bin = pkgs.python3Packages.buildPythonPackage {
          pname = "torch";
          version = "2.9.1+cpu";
          format = "wheel";

          src = pkgs.fetchurl {
            name = "torch-2.9.1-cp313-cp313-linux_aarch64.whl";
            url = "https://download.pytorch.org/whl/cpu/torch-2.9.1%2Bcpu-cp313-cp313-manylinux_2_28_aarch64.whl";
            hash = "sha256-PlMuVTs37oWSBamy0ceXf9aSL1O7sbm/3VvcANGmDtQ=";
          };

          nativeBuildInputs = [
            pkgs.autoPatchelfHook
          ];

          buildInputs = [
            pkgs.stdenv.cc.cc.lib  # libstdc++.so
            pkgs.zlib              # libz.so.1
          ];

          dependencies = with pkgs.python3Packages; [
            filelock
            typing-extensions
            setuptools
            sympy
            networkx
            jinja2
            fsspec
          ];

          pythonImportsCheck = [ "torch" ];
          doCheck = false;
        };

        pythonEnv = pkgs.python3.withPackages (ps: [
          tinygrad
          torch-bin
          ps.numpy
          ps.tqdm
          ps.requests
          ps.pillow
          ps.tiktoken  # for GPT-2 tokenization
          ps.pytest  # for running tinygrad's test suite
          ps.hypothesis  # for property-based tests (test_jit.py, test_tensor.py)
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

          # tinygrad uses its own library finder (DLL.findlib in runtime/support/c.py)
          # that searches hardcoded FHS paths (/usr/lib, /lib64, etc.) which don't exist on NixOS.
          # It checks <NAME>_PATH env vars first, so we point those directly at the .so files.
          # See: grep -rn 'c.DLL' tinygrad/runtime/ for the full list of libraries.
          #
          # CUDA backend libs (required):
          CUDA_PATH = "${jetpack.l4t-cuda}/lib/libcuda.so.1";
          CUDA_INCLUDE_PATH = "${pkgs.lib.getDev cuda.cuda_cudart}/include";
          NVRTC_PATH = "${pkgs.lib.getLib cuda.cuda_nvrtc}/lib/libnvrtc.so";
          NVJITLINK_PATH = "${pkgs.lib.getLib cuda.libnvjitlink}/lib/libnvJitLink.so";
          # System libs (tinygrad loads libc via ctypes for io_uring/mmap/etc.):
          LIBC_PATH = "${pkgs.glibc}/lib/libc.so.6";

          # LD_LIBRARY_PATH is still needed so the dynamic linker can resolve
          # transitive dependencies (e.g. libcudart, libcublas, libstdc++).
          # Also includes glibc so tinygrad's CPU backend can find libgcc_s.so.1.
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
            # Add nested tinygrad folder to PYTHONPATH so 'extra' module can be found
            # This allows 'from extra.bench_log import ...' to work in tinygrad examples
            export PYTHONPATH="$PWD/tinygrad:$PYTHONPATH"
            
            echo ""
            echo "=== tinygrad dev shell (Orin AGX / CUDA 12.6) ==="
            echo ""
            echo "Quick test (CPU):"
            echo "  python3 -c 'from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())'"
            echo ""
            echo "Quick test (CUDA):"
            echo "  CUDA=1 python3 -c 'from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())'"
            echo ""
            echo "LLM examples (clone tinygrad repo first):"
            echo "  git clone --depth 1 https://github.com/tinygrad/tinygrad.git"
            echo "  cd tinygrad && CUDA=1 python3 examples/gpt2.py --count 20"
            echo ""
            echo "PYTHONPATH includes $PWD/tinygrad for the 'extra' module."
            echo ""
          '';
        };

        devShells.detective = pkgs.mkShell {
          packages = with pkgs; [
            binutils  # objdump, etc.
            gcc
            git
            wget
            unzip
            gnumake
            ctags  # For source navigation
            strace  # For ioctl tracing
            gdb
          ];
        };
      }
    );
}

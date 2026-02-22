{
  description = "Control-loop benchmark: tinygrad NV=1 vs PyTorch CUDA on Jetson AGX Orin";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.11";
    jetpack-nixos.url = "path:../..";
    jetpack-nixos.inputs.nixpkgs.follows = "nixpkgs";

    tinygrad = {
      # This is the user's custom branch with the NV=1 port for Orin.
      # See the problem statement: "control-loop-benchmarks-NV-tinygrad"
      url = "github:tinygrad/tinygrad/control-loop-benchmarks-NV-tinygrad";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, jetpack-nixos, tinygrad }:
    let
      system = "aarch64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
          cudaCapabilities = [ "8.7" ]; # Orin AGX is sm_87
        };
        overlays = [ jetpack-nixos.overlays.default ];
      };

      # Use the JetPack 6 package set (L4T 36.x, CUDA 12.6, Python 3.10).
      jp6 = pkgs.pkgsForCudaArch.sm_87.nvidia-jetpack6;

      # tinygrad installed as an editable Python package from the pinned branch.
      tinygradPkg = pkgs.python310.pkgs.buildPythonPackage {
        pname = "tinygrad";
        version = "git";
        format = "pyproject";
        src = tinygrad;
        build-system = with pkgs.python310.pkgs; [ setuptools ];
        doCheck = false;
      };

      benchmarkEnv = pkgs.python310.withPackages (ps: [
        # PyTorch with CUDA from the JetPack 6 scope
        jp6.torch-jetson
        # tinygrad NV=1 branch
        tinygradPkg
        # Benchmark utilities
        ps.numpy
        ps.matplotlib
      ]);
    in
    {
      packages.${system} = {
        benchmark-env = benchmarkEnv;
        default = benchmarkEnv;
      };

      # `nix develop` shell to run the benchmark interactively.
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          benchmarkEnv
          pkgs.python310
        ];

        shellHook = ''
          echo "Control-loop benchmark environment ready."
          echo "Run: python benchmark.py --help"
          echo ""
          echo "To benchmark all backends for 60 s each:"
          echo "  python benchmark.py --duration 60"
        '';
      };
    };
}

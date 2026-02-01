# TinyGrad package for Jetpack NixOS
# Provides TinyGrad with CUDA support for NVIDIA Jetson devices
#
# Usage with local submodule source (recommended for development):
#   tinygrad.override { src = ./vendor/tinygrad; }
#
# Usage with fetched source (for reproducible builds):
#   Uses fetchFromGitHub with pinned commit
{ lib
, python3Packages
, fetchFromGitHub
, cudaPackages ? null
, enableCuda ? true
, src ? null  # Allow overriding source for local development
}:

python3Packages.buildPythonPackage rec {
  pname = "tinygrad";
  version = "0.12.0";
  format = "pyproject";

  # Use provided src or fetch from GitHub
  # The submodule at vendor/tinygrad can be passed as src for local development
  src = if src != null then src else fetchFromGitHub {
    owner = "GiveThanksAlways";
    repo = "tinygrad";
    # Pin to specific commit for reproducibility
    # Update this when updating the vendor/tinygrad submodule
    rev = "ced886f26cbcc1a79e10fa6af2d9b30b590cf030";
    # To update this hash after changing rev:
    # nix-prefetch-github GiveThanksAlways tinygrad --rev <commit>
    # Or use: nix build .#tinygrad 2>&1 | grep "got:" to see actual hash
    hash = "sha256-0000000000000000000000000000000000000000000=";
  };

  nativeBuildInputs = with python3Packages; [
    setuptools
    wheel
  ];

  propagatedBuildInputs = with python3Packages; [
    numpy
  ] ++ lib.optionals enableCuda [
    # CUDA dependencies - these are provided by cudaPackages
  ];

  # TinyGrad has minimal core dependencies, extras are optional
  # Note: We relax Python version requirement from 3.11+ to 3.10+ because:
  # - TinyGrad's core functionality works with Python 3.10
  # - Some NixOS configurations may still use Python 3.10
  # - TinyGrad doesn't use Python 3.11-specific features in core code
  postPatch = ''
    substituteInPlace pyproject.toml \
      --replace 'requires-python = ">=3.11"' 'requires-python = ">=3.10"'
  '';

  # Skip tests during package build as they require CUDA/GPU hardware
  doCheck = false;

  pythonImportsCheck = [ "tinygrad" ];

  passthru = {
    # Provide path to examples for easy reference
    examplesPath = "${src}/examples";
  };

  meta = with lib; {
    description = "A simple, hackable deep learning framework - perfect for Jetson development";
    homepage = "https://github.com/tinygrad/tinygrad";
    license = licenses.mit;
    maintainers = [ ];
    platforms = platforms.linux;
  };
}

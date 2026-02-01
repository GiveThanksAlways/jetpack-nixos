# Tinygrad - A simple and powerful neural network framework
# Note: Tinygrad is best installed via pip in a user environment:
#   pip install tinygrad
#
# This package definition is provided as a reference but may not build
# without the correct source hash. To build from source, update the hash
# by running: nix-prefetch-github tinygrad tinygrad --rev v0.9.2

{ lib
, python3
, fetchFromGitHub
, cudaSupport ? true
}:

python3.pkgs.buildPythonPackage rec {
  pname = "tinygrad";
  version = "0.9.2";
  pyproject = true;

  # Note: Hash needs to be updated for actual builds
  # Run: nix-prefetch-github tinygrad tinygrad --rev v0.9.2
  src = fetchFromGitHub {
    owner = "tinygrad";
    repo = "tinygrad";
    rev = "v${version}";
    hash = "";  # Update with actual hash when building
  };

  nativeBuildInputs = with python3.pkgs; [
    setuptools
    wheel
  ];

  propagatedBuildInputs = with python3.pkgs; [
    numpy
    pillow
    tqdm
    pyopencl
  ];

  # Skip tests as they require GPU
  doCheck = false;

  pythonImportsCheck = [ "tinygrad" ];

  meta = with lib; {
    description = "Simple and powerful neural network framework for Jetson";
    longDescription = ''
      Tinygrad is a lightweight machine learning framework that supports CUDA,
      OpenCL, and other backends. Perfect for running ML models on Jetson devices.
      
      For production use, it's recommended to install via pip:
        pip install tinygrad
    '';
    homepage = "https://github.com/tinygrad/tinygrad";
    license = licenses.mit;
    platforms = platforms.linux;
  };
}

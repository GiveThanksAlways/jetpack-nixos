# Tinygrad - A simple and powerful neural network framework
# Optimized for Jetson devices with CUDA support

{ lib
, python3
, fetchFromGitHub
, cudaSupport ? true
}:

python3.pkgs.buildPythonPackage rec {
  pname = "tinygrad";
  version = "0.9.2";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "tinygrad";
    repo = "tinygrad";
    rev = "v${version}";
    hash = "sha256-8Y8VZqkF7V7xH0r7S3xBFDGV3xCx9F+3xq3vZz9z9z9=";  # Placeholder
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
    '';
    homepage = "https://github.com/tinygrad/tinygrad";
    license = licenses.mit;
    platforms = platforms.linux;
  };
}

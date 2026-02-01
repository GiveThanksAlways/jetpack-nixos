# TinyGrad package for Jetpack NixOS
# Provides TinyGrad with CUDA support for NVIDIA Jetson devices
{ lib
, python3Packages
, fetchFromGitHub
, cudaPackages ? null
, enableCuda ? true
}:

python3Packages.buildPythonPackage rec {
  pname = "tinygrad";
  version = "0.12.0";
  format = "pyproject";

  src = fetchFromGitHub {
    owner = "GiveThanksAlways";
    repo = "tinygrad";
    rev = "master";
    hash = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="; # Will need to be updated
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
  postPatch = ''
    substituteInPlace pyproject.toml \
      --replace 'requires-python = ">=3.11"' 'requires-python = ">=3.10"'
  '';

  # Skip tests during package build as they require CUDA/GPU hardware
  doCheck = false;

  pythonImportsCheck = [ "tinygrad" ];

  meta = with lib; {
    description = "A simple, hackable deep learning framework - perfect for Jetson development";
    homepage = "https://github.com/tinygrad/tinygrad";
    license = licenses.mit;
    maintainers = [ ];
    platforms = platforms.linux;
  };
}

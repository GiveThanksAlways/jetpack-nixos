# TinyGrad - Minimalist deep learning framework
#
# RECOMMENDED: Use the flakes in llm/tinygrad/ for LLM inference:
#   cd llm && nix develop
#   tinygrad-llama 1B int8
#
# This package definition uses the vendor/tinygrad submodule for reproducibility.
# For standalone builds, update the hash below.

{ lib
, python3
, callPackage
, cudaSupport ? true
}:

let
  # Use vendored tinygrad if available, otherwise fetch from GitHub
  tinygradSrc = if builtins.pathExists ../../vendor/tinygrad
    then ../../vendor/tinygrad
    else builtins.fetchGit {
      url = "https://github.com/tinygrad/tinygrad";
      ref = "master";
    };
in
python3.pkgs.buildPythonPackage {
  pname = "tinygrad";
  version = "0.10.0";
  format = "pyproject";

  src = tinygradSrc;

  nativeBuildInputs = with python3.pkgs; [
    setuptools
    wheel
  ];

  propagatedBuildInputs = with python3.pkgs; [
    numpy
    pillow
    tqdm
    requests
    # LLM dependencies
    tiktoken
    sentencepiece
  ];

  # Tests require GPU
  doCheck = false;

  pythonImportsCheck = [ "tinygrad" ];

  meta = with lib; {
    description = "TinyGrad - minimalist deep learning framework for Jetson";
    longDescription = ''
      TinyGrad is a lightweight ML framework supporting CUDA, OpenCL, and Metal.
      Perfect for running LLMs on Jetson devices.

      Quick start:
        cd llm && nix develop
        tinygrad-llama 1B int8
    '';
    homepage = "https://github.com/tinygrad/tinygrad";
    license = licenses.mit;
    platforms = platforms.linux;
  };
}

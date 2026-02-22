{ lib
, python310
, fetchurl
, l4t-core
, l4t-cuda
, openblas
, glib
, stdenv
  # CUDA packages from the JetPack 6 scope
, cuda_cudart
, libcublas
, libcufft
, cudnn
}:

python310.pkgs.buildPythonPackage {
  pname = "torch-jetson";
  version = "2.6.0";
  format = "wheel";

  # PyTorch for JetPack 6 (CUDA 12.6) pre-built wheel from the Jetson AI Lab index.
  # The SHA256 hash is embedded in the URL path segment (/+f/<hash-prefix>/<hash-suffix>/).
  src = fetchurl {
    url = "https://pypi.jetson-ai-lab.dev/jp6/cu126/+f/6cc/6ecfe8a5994fd/torch-2.6.0-cp310-cp310-linux_aarch64.whl";
    hash = "sha256-bMbs/opZlP1tWPttbrc/8kN0KLtJU/PrqkCfg6X025k=";
  };

  # Only supported on aarch64-linux; the wheel is a native binary.
  disabled = !stdenv.hostPlatform.isAarch64;

  # CUDA and L4T libraries are dlopened at runtime by PyTorch.
  # Including them as buildInputs lets autoPatchelfHook fix up RPATHs so
  # the libraries are found without requiring LD_LIBRARY_PATH tweaks.
  buildInputs = [
    l4t-core # libnvrm_gpu.so, libnvrm_mem.so
    l4t-cuda # libcuda.so
    openblas # libopenblas.so
    glib # libglib-2.0.so
    cuda_cudart # libcudart.so
    libcublas # libcublas.so
    libcufft # libcufft.so
    cudnn # libcudnn.so
  ];

  meta = with lib; {
    description = "PyTorch 2.6 pre-built for NVIDIA Jetson (JetPack 6, CUDA 12.6, Python 3.10)";
    homepage = "https://developer.nvidia.com/embedded/jetpack";
    sourceProvenance = with sourceTypes; [ binaryNativeCode ];
    license = licenses.unfree;
    platforms = [ "aarch64-linux" ];
    maintainers = [ ];
  };
}

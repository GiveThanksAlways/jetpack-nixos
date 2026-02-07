# configuration.nix — Example full NixOS config for Orin AGX LLM serving
#
# This config enables vLLM + llama.cpp + TabbyAPI + performance tuning.
# Copy and adapt to your needs. Only enable the backend(s) you want.

{ config, lib, pkgs, ... }:

{
  imports = [
    ./module.nix
    ./llama-cpp-server.nix
    ./tabby-api.nix
    ./performance.nix
  ];

  # ── Jetson Orin AGX basics ──
  hardware.nvidia-jetpack = {
    enable = true;
    som = "orin-agx";
    carrierBoard = "devkit";
  };
  hardware.graphics.enable = true;
  nixpkgs.config.allowUnfree = true;

  # ── Performance tuning ──
  services.orin-perf = {
    enable = true;
    powerMode = "MAXN";
    lockClocks = true;
  };

  # ── Option A: vLLM (best throughput for batch inference) ──
  # services.vllm-serving = {
  #   enable = true;
  #   model = "/models/Qwen3-Coder-Next-Q4_K_M.gguf";
  #   tokenizer = "Qwen/Qwen3-Coder-Next";
  #   quantization = "gguf";
  #   gpuMemoryUtilization = 0.90;
  #   maxModelLen = 4096;
  # };

  # ── Option B: llama.cpp (lowest overhead, GGUF-native) ──
  services.llama-cpp-server = {
    enable = true;
    model = "/models/Qwen3-Coder-Next-Q4_K_M.gguf";
    nGpuLayers = 99; # Offload everything to GPU
    contextSize = 4096;
    threads = 8;
    batchSize = 512;
    flashAttn = true;
    useMmap = true;
    extraArgs = [
      "--cont-batching" # Enable continuous batching
      "--parallel"
      "2" # Serve 2 concurrent requests
    ];
  };

  # ── Option C: TabbyAPI (best for IDE/OpenCode integration) ──
  # services.tabby-api = {
  #   enable = true;
  #   modelDir = "/models";
  #   modelName = "Qwen3-Coder-Next-Q4_K_M.gguf";
  #   maxSeqLen = 4096;
  #   cacheMode = "Q4";  # Q4 KV cache saves VRAM
  # };

  # ── Networking ──
  networking.hostName = "orin-agx-llm";
  networking.firewall.enable = true;

  # ── System ──
  system.stateVersion = "25.11";
}

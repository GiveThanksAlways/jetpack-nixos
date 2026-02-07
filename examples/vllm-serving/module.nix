{ config, lib, pkgs, ... }:

let
  cfg = config.services.vllm-serving;
in
{
  options.services.vllm-serving = {
    enable = lib.mkEnableOption "vLLM model serving for Jetson Orin";

    model = lib.mkOption {
      type = lib.types.str;
      description = "Path to GGUF model file or HuggingFace model ID.";
      example = "/models/Qwen3-Coder-Next-Q4_K_M.gguf";
    };

    tokenizer = lib.mkOption {
      type = lib.types.str;
      default = "";
      description = "HuggingFace tokenizer name (required for GGUF models).";
      example = "Qwen/Qwen3-Coder-Next";
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "0.0.0.0";
      description = "Host to bind the API server to.";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 8000;
      description = "Port for the OpenAI-compatible API.";
    };

    gpuMemoryUtilization = lib.mkOption {
      type = lib.types.float;
      default = 0.90;
      description = "Fraction of GPU memory to use (0.0-1.0).";
    };

    maxModelLen = lib.mkOption {
      type = lib.types.int;
      default = 4096;
      description = "Maximum sequence length. Lower = less VRAM, faster.";
    };

    quantization = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "Quantization method (gguf, awq, gptq, etc).";
      example = "gguf";
    };

    extraArgs = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      description = "Extra CLI arguments for vllm serve.";
    };

    package = lib.mkOption {
      type = lib.types.package;
      default = pkgs.python3Packages.vllm or pkgs.python3.pkgs.vllm or (
        throw "vLLM package not found in nixpkgs. Install vllm via pip in a Python env or add it to your overlay."
      );
      description = "vLLM package to use.";
    };

    environment = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = { };
      description = "Extra environment variables for the vLLM process.";
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.services.vllm-serving = {
      description = "vLLM OpenAI-compatible API Server";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];

      environment = {
        CUDA_VISIBLE_DEVICES = "0";
        # Orin unified memory — let vLLM use it
        PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True";
        VLLM_ATTENTION_BACKEND = "FLASH_ATTN";
      } // cfg.environment;

      serviceConfig = {
        Type = "simple";
        Restart = "on-failure";
        RestartSec = 10;

        # Run as a dedicated user
        DynamicUser = true;
        StateDirectory = "vllm";

        ExecStart =
          let
            tokenizerArgs = lib.optionals (cfg.tokenizer != "") [
              "--tokenizer"
              cfg.tokenizer
            ];
            quantArgs = lib.optionals (cfg.quantization != null) [
              "--quantization"
              cfg.quantization
            ];
            args = [
              "${cfg.package}/bin/python"
              "-m"
              "vllm.entrypoints.openai.api_server"
              "--model"
              cfg.model
              "--host"
              cfg.host
              "--port"
              (toString cfg.port)
              "--gpu-memory-utilization"
              (toString cfg.gpuMemoryUtilization)
              "--max-model-len"
              (toString cfg.maxModelLen)
            ] ++ tokenizerArgs ++ quantArgs ++ cfg.extraArgs;
          in
          lib.concatStringsSep " " args;
      };
    };

    # Open firewall port
    networking.firewall.allowedTCPPorts = [ cfg.port ];
  };
}

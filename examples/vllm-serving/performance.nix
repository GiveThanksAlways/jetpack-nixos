# performance.nix — Orin AGX performance tuning for LLM inference
#
# Import this module to max out inference speed on Jetson Orin AGX.
# Sets power mode, clocks, memory, and kernel params.

{ config, lib, pkgs, ... }:

let
  cfg = config.services.orin-perf;
in
{
  options.services.orin-perf = {
    enable = lib.mkEnableOption "Orin AGX performance tuning for LLM inference";

    powerMode = lib.mkOption {
      type = lib.types.enum [ "MAXN" "30W" "50W" ];
      default = "MAXN";
      description = "NVPModel power mode. MAXN = max performance.";
    };

    lockClocks = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Lock GPU/CPU clocks to max (jetson_clocks).";
    };

    hugepages = lib.mkOption {
      type = lib.types.int;
      default = 4096;
      description = "Number of 2MB hugepages to allocate for model loading.";
    };

    zramSwap = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Enable zram swap for memory pressure during model loading.";
    };
  };

  config = lib.mkIf cfg.enable {
    # Set NVPModel power mode at boot
    systemd.services.orin-power-mode = {
      description = "Set Orin AGX power mode";
      wantedBy = [ "multi-user.target" ];
      after = [ "multi-user.target" ];

      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        ExecStart =
          let
            modeId = {
              "MAXN" = "0";
              "30W" = "1";
              "50W" = "2";
            }.${cfg.powerMode};
          in
          pkgs.writeShellScript "set-power-mode" ''
            # nvpmodel and jetson_clocks are provided by the JetPack system.
            # They live in /usr/sbin on the Jetson; adjust PATH if needed.
            export PATH="/usr/sbin:/usr/bin:$PATH"
            nvpmodel -m ${modeId} || true
            ${lib.optionalString cfg.lockClocks ''
              sleep 2
              jetson_clocks || true
            ''}
          '';
      };
    };

    # Kernel params for max inference performance
    boot.kernel.sysctl = {
      # Reduce swap tendency — prefer keeping model in RAM
      "vm.swappiness" = 10;
      # Allow overcommit for large model mmap
      "vm.overcommit_memory" = 1;
      # Increase max map count for mmap-heavy workloads
      "vm.max_map_count" = 1048576;
    };

    # Hugepages for model loading
    boot.kernelParams = [
      "hugepagesz=2M"
      "hugepages=${toString cfg.hugepages}"
      "transparent_hugepage=always"
    ];

    # ZRAM swap — compressed in-memory swap for model loading pressure
    zramSwap = lib.mkIf cfg.zramSwap {
      enable = true;
      memoryPercent = 50;
      algorithm = "zstd";
    };

    # tmpfs for model scratch space
    fileSystems."/tmp/llm-scratch" = {
      device = "tmpfs";
      fsType = "tmpfs";
      options = [ "size=4G" "mode=1777" ];
    };
  };
}

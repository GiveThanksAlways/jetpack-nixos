# telemetry.nix -- Grafana + Prometheus + tegrastats for Jetson Orin
#
# Import this module and enable to get a full observability stack
# with GPU metrics (tegrastats), system metrics (node exporter),
# and a Grafana dashboard -- all as native systemd services.
#
#   imports = [ ./modules/telemetry.nix ];
#   services.jetson-telemetry.enable = true;
#
# Grafana UI: http://<host>:3301
# Prometheus: http://<host>:9090

{ config, lib, pkgs, ... }:

let
  cfg = config.services.jetson-telemetry;
in
{
  options.services.jetson-telemetry = {
    enable = lib.mkEnableOption "Jetson GPU/system telemetry (Grafana + Prometheus)";

    port = lib.mkOption {
      type = lib.types.port;
      default = 3301;
      description = "Grafana web UI port.";
    };

    dataDir = lib.mkOption {
      type = lib.types.str;
      default = "/var/lib/jetson-telemetry";
      description = "Data directory for metrics storage.";
    };

    enableTegrastats = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Enable tegrastats GPU exporter (500ms sampling).";
    };

    enableNodeExporter = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Enable Prometheus node exporter for system metrics.";
    };

    retentionTime = lib.mkOption {
      type = lib.types.str;
      default = "30d";
      description = "Prometheus data retention period.";
    };

    enableOpenTelemetry = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Enable OpenTelemetry collector (OTLP -> Prometheus).";
    };
  };

  # Note: the actual implementation lives in the top-level
  # modules/jetson-telemetry.nix.  This file just re-exports the
  # option interface so it can be referenced from the examples flake
  # without importing the full module tree.  When deploying on a real
  # Jetson, jetpack.nixosModules.default already pulls in the real
  # module; this stub exists only for documentation and discoverability.
}

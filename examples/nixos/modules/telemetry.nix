# telemetry.nix -- Grafana + Prometheus + tegrastats for Jetson Orin
#
# Self-contained NixOS module. Import and enable:
#
#   imports = [ ./modules/telemetry.nix ];
#   services.jetson-telemetry.enable = true;
#
# Grafana UI: http://<host>:3301
# Prometheus: http://<host>:9090

{ config, lib, pkgs, ... }:

let
  inherit (lib)
    mkEnableOption
    mkIf
    mkOption
    types;

  cfg = config.services.jetson-telemetry;

  # Tegrastats parser -- writes Prometheus text-format metrics
  tegrastatsExporter = pkgs.writeShellScriptBin "tegrastats-exporter" ''
    #!/usr/bin/env bash
    # No set -e: ((expr)) returning 0 is falsy in bash and would kill the script
    echo "Starting tegrastats exporter"

    TEGRASTATS_BIN="${pkgs.nvidia-jetpack.l4t-tools}/bin/tegrastats"
    METRICS_FILE="/var/lib/jetson-telemetry/metrics/tegrastats-metrics.prom.txt"
    INTERVAL_MS=500

    mkdir -p "$(dirname "$METRICS_FILE")"

    "$TEGRASTATS_BIN" --interval "$INTERVAL_MS" | while IFS= read -r line; do
      timestamp=$(date +%s)

      {
        echo "# HELP jetson_ram_used_mb RAM used in megabytes"
        echo "# TYPE jetson_ram_used_mb gauge"
        echo "# HELP jetson_ram_total_mb RAM total in megabytes"
        echo "# TYPE jetson_ram_total_mb gauge"
        echo "# HELP jetson_swap_used_mb SWAP used in megabytes"
        echo "# TYPE jetson_swap_used_mb gauge"
        echo "# HELP jetson_swap_total_mb SWAP total in megabytes"
        echo "# TYPE jetson_swap_total_mb gauge"
        echo "# HELP jetson_cpu_usage_percent CPU usage percentage per core"
        echo "# TYPE jetson_cpu_usage_percent gauge"
        echo "# HELP jetson_cpu_freq_mhz CPU frequency in MHz"
        echo "# TYPE jetson_cpu_freq_mhz gauge"
        echo "# HELP jetson_gpu_usage_percent GPU usage percentage"
        echo "# TYPE jetson_gpu_usage_percent gauge"
        echo "# HELP jetson_gpu_freq_mhz GPU frequency in MHz"
        echo "# TYPE jetson_gpu_freq_mhz gauge"
        echo "# HELP jetson_emc_freq_percent EMC frequency percentage"
        echo "# TYPE jetson_emc_freq_percent gauge"
        echo "# HELP jetson_emc_freq_mhz EMC frequency in MHz"
        echo "# TYPE jetson_emc_freq_mhz gauge"
        echo "# HELP jetson_temperature_celsius Temperature in Celsius"
        echo "# TYPE jetson_temperature_celsius gauge"
        echo "# HELP jetson_power_mw Power consumption in milliwatts"
        echo "# TYPE jetson_power_mw gauge"

        # RAM
        if [[ $line =~ RAM\ ([0-9]+)/([0-9]+)MB ]]; then
          echo "jetson_ram_used_mb ''${BASH_REMATCH[1]}"
          echo "jetson_ram_total_mb ''${BASH_REMATCH[2]}"
        fi

        # SWAP
        if [[ $line =~ SWAP\ ([0-9]+)/([0-9]+)MB ]]; then
          echo "jetson_swap_used_mb ''${BASH_REMATCH[1]}"
          echo "jetson_swap_total_mb ''${BASH_REMATCH[2]}"
        fi

        # CPU per-core (handles "off" cores gracefully)
        if [[ $line =~ CPU\ \[([^\]]+)\] ]]; then
          cpu_data="''${BASH_REMATCH[1]}"
          core_num=0
          IFS=',' read -ra CORES <<< "$cpu_data"
          for core in "''${CORES[@]}"; do
            if [[ $core =~ ([0-9]+)%@([0-9]+) ]]; then
              echo "jetson_cpu_usage_percent{core=\"$core_num\"} ''${BASH_REMATCH[1]}"
              echo "jetson_cpu_freq_mhz{core=\"$core_num\"} ''${BASH_REMATCH[2]}"
            fi
            core_num=$((core_num + 1))
          done
        fi

        # GPU — handles both "GR3D_FREQ 0%@1300" and "GR3D_FREQ 0%" formats
        if [[ $line =~ GR3D_FREQ\ ([0-9]+)%@([0-9]+) ]]; then
          echo "jetson_gpu_usage_percent ''${BASH_REMATCH[1]}"
          echo "jetson_gpu_freq_mhz ''${BASH_REMATCH[2]}"
        elif [[ $line =~ GR3D_FREQ\ ([0-9]+)% ]]; then
          echo "jetson_gpu_usage_percent ''${BASH_REMATCH[1]}"
        fi

        # EMC
        if [[ $line =~ EMC_FREQ\ ([0-9]+)%@([0-9]+) ]]; then
          echo "jetson_emc_freq_percent ''${BASH_REMATCH[1]}"
          echo "jetson_emc_freq_mhz ''${BASH_REMATCH[2]}"
        elif [[ $line =~ EMC_FREQ\ ([0-9]+)% ]]; then
          echo "jetson_emc_freq_percent ''${BASH_REMATCH[1]}"
        fi

        # Temperatures — match both upper and lowercase sensor names (cpu, soc0, tj, etc.)
        temp_line="$line"
        while [[ $temp_line =~ ([A-Za-z0-9_]+)@([0-9.-]+)C ]]; do
          sensor="''${BASH_REMATCH[1]}"
          temp="''${BASH_REMATCH[2]}"
          echo "jetson_temperature_celsius{sensor=\"$sensor\"} $temp"
          temp_line="''${temp_line/''${sensor}@''${temp}C/}"
        done

        # Power rails — match any VDD_*/VIN_* rail pattern: "NAME curr/avg" in mW
        power_line="$line"
        while [[ $power_line =~ (V[A-Z0-9_]+)\ ([0-9]+)mW/([0-9]+)mW ]]; do
          rail="''${BASH_REMATCH[1]}"
          curr="''${BASH_REMATCH[2]}"
          echo "jetson_power_mw{rail=\"$rail\"} $curr"
          power_line="''${power_line/''${rail} ''${curr}mW/}"
        done

      } > "$METRICS_FILE.tmp"

      mv -f "$METRICS_FILE.tmp" "$METRICS_FILE"
    done
  '';

  # Grafana dashboard JSON
  gpuDashboard = pkgs.writeText "jetson-gpu-dashboard.json" (builtins.toJSON {
    title = "Jetson Mission Control";
    tags = ["jetson" "nvidia" "gpu" "telemetry"];
    timezone = "browser";
    refresh = "2s";
    time = { from = "now-5m"; to = "now"; };
    style = "dark";
    panels = [
      # Row 0: GPU
      {
        id = 1; title = "GPU Load"; type = "gauge";
        gridPos = { h = 8; w = 6; x = 0; y = 0; };
        targets = [{ expr = "jetson_gpu_usage_percent"; legendFormat = "GPU %"; }];
        fieldConfig.defaults = {
          unit = "percent"; min = 0; max = 100;
          thresholds.mode = "absolute";
          thresholds.steps = [
            { value = null; color = "green"; }
            { value = 60; color = "yellow"; }
            { value = 85; color = "red"; }
          ];
        };
      }
      {
        id = 2; title = "GPU Utilization"; type = "timeseries";
        gridPos = { h = 8; w = 10; x = 6; y = 0; };
        targets = [{ expr = "jetson_gpu_usage_percent"; legendFormat = "GPU Usage %"; }];
        fieldConfig.defaults = {
          unit = "percent"; min = 0; max = 100;
          custom.fillOpacity = 20; custom.lineWidth = 2; custom.gradientMode = "scheme";
        };
      }
      {
        id = 3; title = "GPU Frequency"; type = "timeseries";
        gridPos = { h = 8; w = 8; x = 16; y = 0; };
        targets = [{ expr = "jetson_gpu_freq_mhz"; legendFormat = "GPU Freq (MHz)"; }];
        fieldConfig.defaults = { unit = "MHz"; custom.fillOpacity = 10; custom.lineWidth = 2; };
      }
      # Row 1: CPU + Memory
      {
        id = 4; title = "CPU Usage by Core"; type = "timeseries";
        gridPos = { h = 8; w = 16; x = 0; y = 8; };
        targets = [{ expr = "jetson_cpu_usage_percent"; legendFormat = "Core {{core}}"; }];
        fieldConfig.defaults = {
          unit = "percent"; min = 0; max = 100;
          custom.fillOpacity = 15; custom.lineWidth = 1;
        };
      }
      {
        id = 5; title = "RAM Used"; type = "gauge";
        gridPos = { h = 8; w = 4; x = 16; y = 8; };
        targets = [{ expr = "jetson_ram_used_mb / jetson_ram_total_mb * 100"; legendFormat = "RAM %"; }];
        fieldConfig.defaults = {
          unit = "percent"; min = 0; max = 100;
          thresholds.mode = "absolute";
          thresholds.steps = [
            { value = null; color = "green"; }
            { value = 70; color = "yellow"; }
            { value = 90; color = "red"; }
          ];
        };
      }
      {
        id = 6; title = "Memory (MB)"; type = "timeseries";
        gridPos = { h = 8; w = 4; x = 20; y = 8; };
        targets = [
          { expr = "jetson_ram_used_mb"; legendFormat = "RAM"; }
          { expr = "jetson_swap_used_mb"; legendFormat = "SWAP"; }
        ];
        fieldConfig.defaults.unit = "mbytes";
      }
      # Row 2: Thermals + Power
      {
        id = 7; title = "Temperatures"; type = "timeseries";
        gridPos = { h = 8; w = 12; x = 0; y = 16; };
        targets = [{ expr = "jetson_temperature_celsius"; legendFormat = "{{sensor}}"; }];
        fieldConfig.defaults = {
          unit = "celsius"; custom.fillOpacity = 5;
          thresholds.mode = "absolute";
          thresholds.steps = [
            { value = null; color = "green"; }
            { value = 70; color = "yellow"; }
            { value = 85; color = "red"; }
          ];
        };
      }
      {
        id = 8; title = "Power Consumption"; type = "timeseries";
        gridPos = { h = 8; w = 12; x = 12; y = 16; };
        targets = [{ expr = "jetson_power_mw"; legendFormat = "{{rail}}"; }];
        fieldConfig.defaults = {
          unit = "mwatt"; custom.fillOpacity = 15; custom.gradientMode = "scheme";
        };
      }
      # Row 3: EMC + Engines
      {
        id = 9; title = "EMC Load"; type = "gauge";
        gridPos = { h = 8; w = 6; x = 0; y = 24; };
        targets = [{ expr = "jetson_emc_freq_percent"; legendFormat = "EMC %"; }];
        fieldConfig.defaults = {
          unit = "percent"; min = 0; max = 100;
          thresholds.mode = "absolute";
          thresholds.steps = [
            { value = null; color = "green"; }
            { value = 60; color = "yellow"; }
            { value = 85; color = "red"; }
          ];
        };
      }
      {
        id = 10; title = "Memory Controller"; type = "timeseries";
        gridPos = { h = 8; w = 6; x = 6; y = 24; };
        targets = [{ expr = "jetson_emc_freq_mhz"; legendFormat = "EMC MHz"; }];
        fieldConfig.defaults.unit = "MHz";
      }
      {
        id = 11; title = "VIC Freq"; type = "stat";
        gridPos = { h = 8; w = 6; x = 12; y = 24; };
        targets = [{ expr = "jetson_vic_freq"; legendFormat = "VIC"; }];
        fieldConfig.defaults.thresholds = {
          mode = "absolute";
          steps = [{ value = null; color = "blue"; }];
        };
      }
      {
        id = 12; title = "APE Freq"; type = "stat";
        gridPos = { h = 8; w = 6; x = 18; y = 24; };
        targets = [{ expr = "jetson_ape_freq"; legendFormat = "APE"; }];
        fieldConfig.defaults.thresholds = {
          mode = "absolute";
          steps = [{ value = null; color = "purple"; }];
        };
      }
    ];
  });

in
{
  options.services.jetson-telemetry = {
    enable = mkEnableOption "Jetson GPU/system telemetry (Grafana + Prometheus)";

    port = mkOption {
      type = types.port;
      default = 3301;
      description = "Grafana web UI port.";
    };

    dataDir = mkOption {
      type = types.path;
      default = "/var/lib/jetson-telemetry";
      description = "Data directory for metrics storage.";
    };

    enableTegrastats = mkOption {
      type = types.bool;
      default = true;
      description = "Enable tegrastats GPU exporter (500ms sampling).";
    };

    enableNodeExporter = mkOption {
      type = types.bool;
      default = true;
      description = "Enable Prometheus node exporter for system metrics.";
    };

    retentionTime = mkOption {
      type = types.str;
      default = "30d";
      description = "Prometheus data retention period.";
    };

    enableOpenTelemetry = mkOption {
      type = types.bool;
      default = false;
      description = "Enable OpenTelemetry collector (OTLP -> Prometheus).";
    };
  };

  config = mkIf cfg.enable {
    environment.systemPackages = with pkgs;
      [ pkgs.nvidia-jetpack.l4t-tools pkgs.python3 ]
      ++ lib.optionals (pkgs.nvidia-jetpack.l4tAtLeast "36")
        [ pkgs.nvidia-jetpack.nvidia-smi ];

    systemd.tmpfiles.rules = [
      "d ${cfg.dataDir} 0755 root root -"
      "d ${cfg.dataDir}/metrics 0755 root root -"
      "d ${cfg.dataDir}/prometheus 0755 root root -"
      "d ${cfg.dataDir}/grafana 0755 root root -"
      "d ${cfg.dataDir}/grafana/dashboards 0755 root root -"
      "L+ ${cfg.dataDir}/grafana/dashboards/jetson-gpu.json - - - - ${gpuDashboard}"
    ];

    # Tegrastats exporter
    systemd.services.tegrastats-exporter = mkIf cfg.enableTegrastats {
      description = "Tegrastats GPU metrics exporter";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];
      serviceConfig = {
        Type = "simple";
        ExecStart = "${tegrastatsExporter}/bin/tegrastats-exporter";
        Restart = "on-failure";
        RestartSec = "10s";
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ cfg.dataDir ];
      };
    };

    # HTTP server exposing .prom files for Prometheus scraping
    systemd.services.tegrastats-http-server = mkIf cfg.enableTegrastats {
      description = "HTTP server for tegrastats Prometheus metrics";
      wantedBy = [ "multi-user.target" ];
      after = [ "tegrastats-exporter.service" ];
      requires = [ "tegrastats-exporter.service" ];
      serviceConfig = {
        Type = "simple";
        ExecStart = "${pkgs.python3}/bin/python3 -m http.server 9101 --directory ${cfg.dataDir}/metrics --bind 0.0.0.0";
        WorkingDirectory = "${cfg.dataDir}/metrics";
        Restart = "on-failure";
        RestartSec = "5s";
        ReadOnlyPaths = [ "${cfg.dataDir}/metrics" ];
        ProtectSystem = "strict";
        ProtectHome = true;
      };
    };

    # Node exporter
    services.prometheus.exporters.node = mkIf cfg.enableNodeExporter {
      enable = true;
      port = 9100;
      enabledCollectors = [
        "cpu" "loadavg" "meminfo" "diskstats"
        "filesystem" "netdev" "thermal_zone"
      ];
    };

    # Prometheus
    services.prometheus = {
      enable = true;
      port = 9090;
      retentionTime = cfg.retentionTime;
      scrapeConfigs = [
        {
          job_name = "jetson-tegrastats";
          scrape_interval = "5s";
          metrics_path = "/tegrastats-metrics.prom.txt";
          static_configs = [{
            targets = [ "localhost:9101" ];
            labels = { device = "jetson"; source = "tegrastats"; };
          }];
        }
        {
          job_name = "jetson-system";
          scrape_interval = "15s";
          static_configs = [{
            targets = [ "localhost:9100" ];
            labels = { device = "jetson"; source = "node-exporter"; };
          }];
        }
      ] ++ lib.optionals cfg.enableOpenTelemetry [
        {
          job_name = "otel-collector";
          scrape_interval = "5s";
          static_configs = [{
            targets = [ "localhost:8889" ];
            labels = { device = "jetson"; source = "opentelemetry"; };
          }];
        }
      ];
    };

    # OpenTelemetry collector (opt-in)
    systemd.services.otel-collector = mkIf cfg.enableOpenTelemetry (let
      otelConfig = pkgs.writeText "otel-config.yaml" ''
        receivers:
          otlp:
            protocols:
              grpc:
                endpoint: "0.0.0.0:4317"
              http:
                endpoint: "0.0.0.0:4318"
        exporters:
          prometheus:
            endpoint: "0.0.0.0:8889"
            namespace: "otel"
        service:
          pipelines:
            metrics:
              receivers: [otlp]
              exporters: [prometheus]
      '';
    in {
      description = "OpenTelemetry Collector";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];
      serviceConfig = {
        Type = "simple";
        ExecStart = "${pkgs.opentelemetry-collector-contrib}/bin/otelcol-contrib --config=${otelConfig}";
        Restart = "on-failure";
        RestartSec = "5s";
      };
    });

    # Grafana
    services.grafana = {
      enable = true;
      settings = {
        server = {
          http_addr = "0.0.0.0";
          http_port = cfg.port;
          domain = "localhost";
        };
        "auth.anonymous" = {
          enabled = true;
          org_role = "Viewer";
        };
        analytics.reporting_enabled = false;
      };
      provision = {
        enable = true;
        datasources.settings.datasources = [{
          name = "Prometheus";
          type = "prometheus";
          access = "proxy";
          url = "http://localhost:9090";
          isDefault = true;
          editable = false;
        }];
        dashboards.settings = {
          apiVersion = 1;
          providers = [{
            name = "Jetson Telemetry";
            folder = "Jetson";
            type = "file";
            disableDeletion = false;
            updateIntervalSeconds = 30;
            allowUiUpdates = true;
            options.path = "${cfg.dataDir}/grafana/dashboards";
          }];
        };
      };
    };

    # Firewall
    networking.firewall.allowedTCPPorts = [
      cfg.port 9090 9100 9101
    ] ++ lib.optionals cfg.enableOpenTelemetry [ 4317 4318 8889 ];

    # Activation message
    system.activationScripts.jetson-telemetry-info = lib.mkIf cfg.enable ''
      echo ""
      echo "=== JETSON TELEMETRY ACTIVE ==="
      echo "  Grafana:    http://localhost:${toString cfg.port}"
      echo "  Prometheus: http://localhost:9090"
      echo "  SSH tunnel: ssh -L ${toString cfg.port}:localhost:${toString cfg.port} -L 9090:localhost:9090 user@<jetson-ip>"
      echo ""
    '';
  };
}

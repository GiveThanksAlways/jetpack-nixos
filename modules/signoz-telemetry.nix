{ config, lib, pkgs, ... }:

let
  inherit (lib)
    mkEnableOption
    mkIf
    mkOption
    types;

  cfg = config.services.signoz-telemetry;

  # Enhanced tegrastats parser with fine-grained GPU metrics
  tegrastatsExporter = pkgs.writeShellScriptBin "tegrastats-exporter" ''
    #!${pkgs.bash}/bin/bash
    set -euo pipefail

    # Configuration
    TEGRASTATS_BIN="${pkgs.nvidia-jetpack.l4t-tools}/bin/tegrastats"
    METRICS_FILE="/var/lib/signoz-telemetry/metrics/tegrastats-metrics.prom"
    INTERVAL_MS=500  # Sample every 500ms for fine-grained metrics

    # Ensure metrics directory exists
    mkdir -p "$(dirname "$METRICS_FILE")"

    # Run tegrastats and parse output
    $TEGRASTATS_BIN --interval $INTERVAL_MS | while IFS= read -r line; do
      timestamp=$(date +%s)
      
      # Write metric metadata once at the beginning
      cat > "$METRICS_FILE.tmp" << 'EOF_METRICS'
# HELP jetson_ram_used_mb RAM used in megabytes
# TYPE jetson_ram_used_mb gauge
# HELP jetson_ram_total_mb RAM total in megabytes
# TYPE jetson_ram_total_mb gauge
# HELP jetson_swap_used_mb SWAP used in megabytes
# TYPE jetson_swap_used_mb gauge
# HELP jetson_swap_total_mb SWAP total in megabytes
# TYPE jetson_swap_total_mb gauge
# HELP jetson_cpu_usage_percent CPU usage percentage per core
# TYPE jetson_cpu_usage_percent gauge
# HELP jetson_cpu_freq_mhz CPU frequency in MHz
# TYPE jetson_cpu_freq_mhz gauge
# HELP jetson_gpu_usage_percent GPU usage percentage
# TYPE jetson_gpu_usage_percent gauge
# HELP jetson_gpu_freq_mhz GPU frequency in MHz
# TYPE jetson_gpu_freq_mhz gauge
# HELP jetson_emc_freq_percent EMC (memory controller) frequency percentage
# TYPE jetson_emc_freq_percent gauge
# HELP jetson_emc_freq_mhz EMC (memory controller) frequency in MHz
# TYPE jetson_emc_freq_mhz gauge
# HELP jetson_vic_freq VIC (video image compositor) frequency
# TYPE jetson_vic_freq gauge
# HELP jetson_ape_freq APE (audio processing engine) frequency
# TYPE jetson_ape_freq gauge
# HELP jetson_temperature_celsius Temperature in Celsius
# TYPE jetson_temperature_celsius gauge
# HELP jetson_power_mw Power consumption in milliwatts
# TYPE jetson_power_mw gauge
EOF_METRICS
      
      # Extract RAM usage (MB)
      if [[ $line =~ RAM\ ([0-9]+)/([0-9]+)MB ]]; then
        ram_used=''${BASH_REMATCH[1]}
        ram_total=''${BASH_REMATCH[2]}
        echo "jetson_ram_used_mb $ram_used $timestamp" >> "$METRICS_FILE.tmp"
        echo "jetson_ram_total_mb $ram_total $timestamp" >> "$METRICS_FILE.tmp"
      fi
      
      # Extract SWAP usage (MB)
      if [[ $line =~ SWAP\ ([0-9]+)/([0-9]+)MB ]]; then
        swap_used=''${BASH_REMATCH[1]}
        swap_total=''${BASH_REMATCH[2]}
        echo "jetson_swap_used_mb $swap_used $timestamp" >> "$METRICS_FILE.tmp"
        echo "jetson_swap_total_mb $swap_total $timestamp" >> "$METRICS_FILE.tmp"
      fi
      
      # Extract CPU utilization per core
      if [[ $line =~ CPU\ \[([^\]]+)\] ]]; then
        cpu_data=''${BASH_REMATCH[1]}
        
        core_num=0
        IFS=',' read -ra CORES <<< "$cpu_data"
        for core in "''${CORES[@]}"; do
          if [[ $core =~ ([0-9]+)%@([0-9]+) ]]; then
            cpu_usage=''${BASH_REMATCH[1]}
            cpu_freq=''${BASH_REMATCH[2]}
            echo "jetson_cpu_usage_percent{core=\"$core_num\"} $cpu_usage $timestamp" >> "$METRICS_FILE.tmp"
            echo "jetson_cpu_freq_mhz{core=\"$core_num\"} $cpu_freq $timestamp" >> "$METRICS_FILE.tmp"
            ((core_num++))
          fi
        done
      fi
      
      # Extract GPU frequency and usage (GR3D_FREQ)
      if [[ $line =~ GR3D_FREQ\ ([0-9]+)%@([0-9]+) ]]; then
        gpu_usage=''${BASH_REMATCH[1]}
        gpu_freq=''${BASH_REMATCH[2]}
        echo "jetson_gpu_usage_percent $gpu_usage $timestamp" >> "$METRICS_FILE.tmp"
        echo "jetson_gpu_freq_mhz $gpu_freq $timestamp" >> "$METRICS_FILE.tmp"
      fi
      
      # Extract EMC (memory controller) frequency
      if [[ $line =~ EMC_FREQ\ ([0-9]+)%@([0-9]+) ]]; then
        emc_usage=''${BASH_REMATCH[1]}
        emc_freq=''${BASH_REMATCH[2]}
        echo "jetson_emc_freq_percent $emc_usage $timestamp" >> "$METRICS_FILE.tmp"
        echo "jetson_emc_freq_mhz $emc_freq $timestamp" >> "$METRICS_FILE.tmp"
      fi
      
      # Extract VIC (Video Image Compositor) frequency
      if [[ $line =~ VIC_FREQ\ ([0-9]+) ]]; then
        vic_freq=''${BASH_REMATCH[1]}
        echo "jetson_vic_freq $vic_freq $timestamp" >> "$METRICS_FILE.tmp"
      fi
      
      # Extract APE (Audio Processing Engine) frequency
      if [[ $line =~ APE\ ([0-9]+) ]]; then
        ape_freq=''${BASH_REMATCH[1]}
        echo "jetson_ape_freq $ape_freq $timestamp" >> "$METRICS_FILE.tmp"
      fi
      
      # Extract temperatures (various sensors)
      temp_pattern="([A-Z0-9_]+)@([0-9.-]+)C"
      while [[ $line =~ $temp_pattern ]]; do
        sensor=''${BASH_REMATCH[1]}
        temp=''${BASH_REMATCH[2]}
        echo "jetson_temperature_celsius{sensor=\"$sensor\"} $temp $timestamp" >> "$METRICS_FILE.tmp"
        line="''${line/$sensor@$temp''C/}"
      done
      
      # Extract power consumption
      if [[ $line =~ VDD_IN\ ([0-9]+)/([0-9]+) ]]; then
        power_in=''${BASH_REMATCH[1]}
        echo "jetson_power_mw{rail=\"VDD_IN\"} $power_in $timestamp" >> "$METRICS_FILE.tmp"
      fi
      
      if [[ $line =~ VDD_CPU_GPU_CV\ ([0-9]+)/([0-9]+) ]]; then
        power_cpu_gpu=''${BASH_REMATCH[1]}
        echo "jetson_power_mw{rail=\"VDD_CPU_GPU_CV\"} $power_cpu_gpu $timestamp" >> "$METRICS_FILE.tmp"
      fi
      
      if [[ $line =~ VDD_SOC\ ([0-9]+)/([0-9]+) ]]; then
        power_soc=''${BASH_REMATCH[1]}
        echo "jetson_power_mw{rail=\"VDD_SOC\"} $power_soc $timestamp" >> "$METRICS_FILE.tmp"
      fi
      
      # Atomically update metrics file
      mv "$METRICS_FILE.tmp" "$METRICS_FILE"
    done
  '';

  # Grafana dashboard for Jetson GPU telemetry
  gpuDashboard = pkgs.writeText "jetson-gpu-dashboard.json" (builtins.toJSON {
    title = "Jetson GPU Telemetry";
    tags = ["jetson" "nvidia" "gpu"];
    timezone = "browser";
    refresh = "5s";
    time = {
      from = "now-5m";
      to = "now";
    };
    panels = [
      {
        id = 1;
        title = "GPU Utilization";
        type = "timeseries";
        gridPos = { h = 8; w = 12; x = 0; y = 0; };
        targets = [{
          expr = "jetson_gpu_usage_percent";
          legendFormat = "GPU Usage %";
        }];
        fieldConfig.defaults = {
          unit = "percent";
          min = 0;
          max = 100;
          custom.fillOpacity = 10;
        };
      }
      {
        id = 2;
        title = "GPU Frequency";
        type = "timeseries";
        gridPos = { h = 8; w = 12; x = 12; y = 0; };
        targets = [{
          expr = "jetson_gpu_freq_mhz";
          legendFormat = "GPU Freq (MHz)";
        }];
        fieldConfig.defaults.unit = "MHz";
      }
      {
        id = 3;
        title = "CPU Usage by Core";
        type = "timeseries";
        gridPos = { h = 8; w = 24; x = 0; y = 8; };
        targets = [{
          expr = "jetson_cpu_usage_percent";
          legendFormat = "Core {{core}}";
        }];
        fieldConfig.defaults = {
          unit = "percent";
          min = 0;
          max = 100;
        };
      }
      {
        id = 4;
        title = "Memory Usage";
        type = "timeseries";
        gridPos = { h = 8; w = 12; x = 0; y = 16; };
        targets = [
          {
            expr = "jetson_ram_used_mb";
            legendFormat = "RAM Used (MB)";
          }
          {
            expr = "jetson_swap_used_mb";
            legendFormat = "SWAP Used (MB)";
          }
        ];
        fieldConfig.defaults.unit = "mbytes";
      }
      {
        id = 5;
        title = "Temperatures";
        type = "timeseries";
        gridPos = { h = 8; w = 12; x = 12; y = 16; };
        targets = [{
          expr = "jetson_temperature_celsius";
          legendFormat = "{{sensor}}";
        }];
        fieldConfig.defaults = {
          unit = "celsius";
          thresholds = {
            mode = "absolute";
            steps = [
              { value = null; color = "green"; }
              { value = 70; color = "yellow"; }
              { value = 85; color = "red"; }
            ];
          };
        };
      }
      {
        id = 6;
        title = "Power Consumption";
        type = "timeseries";
        gridPos = { h = 8; w = 24; x = 0; y = 24; };
        targets = [{
          expr = "jetson_power_mw";
          legendFormat = "{{rail}}";
        }];
        fieldConfig.defaults.unit = "mwatt";
      }
      {
        id = 7;
        title = "Memory Controller Load";
        type = "timeseries";
        gridPos = { h = 8; w = 12; x = 0; y = 32; };
        targets = [{
          expr = "jetson_emc_freq_percent";
          legendFormat = "EMC Load %";
        }];
        fieldConfig.defaults = {
          unit = "percent";
          min = 0;
          max = 100;
        };
      }
      {
        id = 8;
        title = "Video & Audio Engines";
        type = "stat";
        gridPos = { h = 8; w = 12; x = 12; y = 32; };
        targets = [
          {
            expr = "jetson_vic_freq";
            legendFormat = "VIC";
          }
          {
            expr = "jetson_ape_freq";
            legendFormat = "APE";
          }
        ];
      }
    ];
  });

in
{
  options = {
    services.signoz-telemetry = {
      enable = mkEnableOption "Native NixOS telemetry and monitoring for NVIDIA Jetson devices";

      dataDir = mkOption {
        type = types.path;
        default = "/var/lib/signoz-telemetry";
        description = "Data directory for telemetry services";
      };

      port = mkOption {
        type = types.port;
        default = 3301;
        description = "Port for Grafana web UI";
      };

      enableTegrastats = mkOption {
        type = types.bool;
        default = true;
        description = "Enable tegrastats metrics collection with fine-grained GPU metrics";
      };

      enableNodeExporter = mkOption {
        type = types.bool;
        default = true;
        description = "Enable Prometheus Node Exporter for system metrics";
      };

      retentionTime = mkOption {
        type = types.str;
        default = "30d";
        description = "Prometheus metrics retention time";
      };
    };
  };

  config = mkIf cfg.enable {
    # Ensure required packages are available
    environment.systemPackages = with pkgs; 
      [ pkgs.nvidia-jetpack.l4t-tools ]
      ++ lib.optionals (pkgs.nvidia-jetpack.l4tAtLeast "36") 
        [ pkgs.nvidia-jetpack.nvidia-smi ];

    # Create data directory structure
    systemd.tmpfiles.rules = [
      "d ${cfg.dataDir} 0755 root root -"
      "d ${cfg.dataDir}/metrics 0755 root root -"
      "d ${cfg.dataDir}/prometheus 0755 root root -"
      "d ${cfg.dataDir}/grafana 0755 root root -"
      "d ${cfg.dataDir}/grafana/dashboards 0755 root root -"
      "L+ ${cfg.dataDir}/grafana/dashboards/jetson-gpu.json - - - - ${gpuDashboard}"
    ];

    # Tegrastats metrics exporter service (enhanced for fine-grained GPU metrics)
    systemd.services.tegrastats-exporter = mkIf cfg.enableTegrastats {
      description = "Tegrastats metrics exporter with fine-grained GPU telemetry";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];
      
      serviceConfig = {
        Type = "simple";
        ExecStart = "${tegrastatsExporter}/bin/tegrastats-exporter";
        Restart = "on-failure";
        RestartSec = "10s";
        
        DynamicUser = false;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ cfg.dataDir ];
      };
    };

    # Simple HTTP server to expose tegrastats metrics (only serves .prom files)
    systemd.services.tegrastats-http-server = mkIf cfg.enableTegrastats {
      description = "HTTP server for tegrastats Prometheus metrics";
      wantedBy = [ "multi-user.target" ];
      after = [ "tegrastats-exporter.service" ];
      requires = [ "tegrastats-exporter.service" ];
      
      serviceConfig = {
        Type = "simple";
        ExecStart = ''
          ${pkgs.python3}/bin/python3 -m http.server 9101 \
            --directory ${cfg.dataDir}/metrics \
            --bind localhost
        '';
        WorkingDirectory = "${cfg.dataDir}/metrics";
        Restart = "on-failure";
        RestartSec = "5s";
        
        # Security: Only serve metrics directory
        ReadOnlyPaths = [ "${cfg.dataDir}/metrics" ];
        ProtectSystem = "strict";
        ProtectHome = true;
      };
    };

    # Prometheus Node Exporter for system metrics
    services.prometheus.exporters.node = mkIf cfg.enableNodeExporter {
      enable = true;
      port = 9100;
      enabledCollectors = [
        "cpu"
        "loadavg"
        "meminfo"
        "diskstats"
        "filesystem"
        "netdev"
        "thermal"
      ];
    };

    # Prometheus for metrics collection and storage
    services.prometheus = {
      enable = true;
      port = 9090;
      
      retentionTime = cfg.retentionTime;
      
      scrapeConfigs = [
        {
          job_name = "jetson-tegrastats";
          scrape_interval = "5s";
          static_configs = [{
            targets = [ "localhost:9101" ];
            labels = {
              device = "jetson-orin-agx";
              source = "tegrastats";
            };
          }];
        }
        {
          job_name = "jetson-system";
          scrape_interval = "15s";
          static_configs = [{
            targets = [ "localhost:9100" ];
            labels = {
              device = "jetson-orin-agx";
              source = "node-exporter";
            };
          }];
        }
      ];
    };

    # Grafana for visualization
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

    # Open firewall ports
    networking.firewall.allowedTCPPorts = [
      cfg.port      # Grafana UI
      9090          # Prometheus
      9100          # Node Exporter
      9101          # Tegrastats HTTP server
    ];

    # Informational message
    system.activationScripts.signoz-telemetry-info = lib.mkIf cfg.enable ''
      echo ""
      echo "=========================================="
      echo "Native NixOS Telemetry Stack Enabled"
      echo "=========================================="
      echo "Grafana UI: http://localhost:${toString cfg.port}"
      echo "Prometheus: http://localhost:9090"
      echo ""
      echo "Pure NixOS implementation - No Docker!"
      echo ""
      echo "Collecting fine-grained metrics:"
      echo "  - Tegrastats (500ms GPU sampling)"
      echo "  - Node Exporter (System metrics)"
      echo ""
      echo "GPU Dashboard auto-provisioned!"
      echo "=========================================="
      echo ""
    '';
  };
}

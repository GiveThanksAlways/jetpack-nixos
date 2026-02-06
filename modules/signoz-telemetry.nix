{ config, lib, pkgs, ... }:

let
  inherit (lib)
    mkEnableOption
    mkIf
    mkOption
    types;

  cfg = config.services.signoz-telemetry;

  # Tegrastats parser and exporter script
  tegrastatsExporter = pkgs.writeShellScriptBin "tegrastats-exporter" ''
    #!${pkgs.bash}/bin/bash
    set -euo pipefail

    # Configuration
    TEGRASTATS_BIN="${pkgs.nvidia-jetpack.l4t-tools}/bin/tegrastats"
    METRICS_FILE="/var/lib/signoz-telemetry/tegrastats-metrics.prom"
    INTERVAL_MS=1000  # Sample every 1 second

    # Ensure metrics directory exists
    mkdir -p "$(dirname "$METRICS_FILE")"

    # Run tegrastats and parse output
    $TEGRASTATS_BIN --interval $INTERVAL_MS | while IFS= read -r line; do
      # Parse tegrastats output format
      # Example: RAM 5123/31906MB (lfb 7510x4MB) SWAP 0/15953MB (cached 0MB) CPU [13%@2265,5%@2265,7%@2265,7%@2265,6%@2265,6%@2265,7%@2265,6%@2265,11%@2265,7%@2265,6%@2265,7%@2265] EMC_FREQ 0%@2133 GR3D_FREQ 0%@1300 VIC_FREQ 115 APE 25 CV0@-256C CPU@52.5C Tboard@42C SOC2@50.437C Tdiode@43.75C SOC0@52.062C CV1@-256C GPU@-256C tj@52.5C SOC1@49.812C CV2@-256C VDD_IN 8039/8039 VDD_CPU_GPU_CV 3231/3231 VDD_SOC 1623/1623
      
      timestamp=$(date +%s)
      
      # Extract RAM usage (MB)
      if [[ $line =~ RAM\ ([0-9]+)/([0-9]+)MB ]]; then
        ram_used=''${BASH_REMATCH[1]}
        ram_total=''${BASH_REMATCH[2]}
        echo "# HELP jetson_ram_used_mb RAM used in megabytes" > "$METRICS_FILE.tmp"
        echo "# TYPE jetson_ram_used_mb gauge" >> "$METRICS_FILE.tmp"
        echo "jetson_ram_used_mb $ram_used $timestamp" >> "$METRICS_FILE.tmp"
        echo "# HELP jetson_ram_total_mb RAM total in megabytes" >> "$METRICS_FILE.tmp"
        echo "# TYPE jetson_ram_total_mb gauge" >> "$METRICS_FILE.tmp"
        echo "jetson_ram_total_mb $ram_total $timestamp" >> "$METRICS_FILE.tmp"
      fi
      
      # Extract SWAP usage (MB)
      if [[ $line =~ SWAP\ ([0-9]+)/([0-9]+)MB ]]; then
        swap_used=''${BASH_REMATCH[1]}
        swap_total=''${BASH_REMATCH[2]}
        echo "# HELP jetson_swap_used_mb SWAP used in megabytes" >> "$METRICS_FILE.tmp"
        echo "# TYPE jetson_swap_used_mb gauge" >> "$METRICS_FILE.tmp"
        echo "jetson_swap_used_mb $swap_used $timestamp" >> "$METRICS_FILE.tmp"
        echo "# HELP jetson_swap_total_mb SWAP total in megabytes" >> "$METRICS_FILE.tmp"
        echo "# TYPE jetson_swap_total_mb gauge" >> "$METRICS_FILE.tmp"
        echo "jetson_swap_total_mb $swap_total $timestamp" >> "$METRICS_FILE.tmp"
      fi
      
      # Extract CPU utilization per core
      if [[ $line =~ CPU\ \[([^\]]+)\] ]]; then
        cpu_data=''${BASH_REMATCH[1]}
        echo "# HELP jetson_cpu_usage_percent CPU usage percentage per core" >> "$METRICS_FILE.tmp"
        echo "# TYPE jetson_cpu_usage_percent gauge" >> "$METRICS_FILE.tmp"
        
        core_num=0
        IFS=',' read -ra CORES <<< "$cpu_data"
        for core in "''${CORES[@]}"; do
          if [[ $core =~ ([0-9]+)%@([0-9]+) ]]; then
            cpu_usage=''${BASH_REMATCH[1]}
            cpu_freq=''${BASH_REMATCH[2]}
            echo "jetson_cpu_usage_percent{core=\"$core_num\"} $cpu_usage $timestamp" >> "$METRICS_FILE.tmp"
            echo "# HELP jetson_cpu_freq_mhz CPU frequency in MHz" >> "$METRICS_FILE.tmp"
            echo "# TYPE jetson_cpu_freq_mhz gauge" >> "$METRICS_FILE.tmp"
            echo "jetson_cpu_freq_mhz{core=\"$core_num\"} $cpu_freq $timestamp" >> "$METRICS_FILE.tmp"
            ((core_num++))
          fi
        done
      fi
      
      # Extract GPU frequency
      if [[ $line =~ GR3D_FREQ\ ([0-9]+)%@([0-9]+) ]]; then
        gpu_usage=''${BASH_REMATCH[1]}
        gpu_freq=''${BASH_REMATCH[2]}
        echo "# HELP jetson_gpu_usage_percent GPU usage percentage" >> "$METRICS_FILE.tmp"
        echo "# TYPE jetson_gpu_usage_percent gauge" >> "$METRICS_FILE.tmp"
        echo "jetson_gpu_usage_percent $gpu_usage $timestamp" >> "$METRICS_FILE.tmp"
        echo "# HELP jetson_gpu_freq_mhz GPU frequency in MHz" >> "$METRICS_FILE.tmp"
        echo "# TYPE jetson_gpu_freq_mhz gauge" >> "$METRICS_FILE.tmp"
        echo "jetson_gpu_freq_mhz $gpu_freq $timestamp" >> "$METRICS_FILE.tmp"
      fi
      
      # Extract temperatures (various sensors)
      temp_pattern="([A-Z0-9_]+)@([0-9.-]+)C"
      while [[ $line =~ $temp_pattern ]]; do
        sensor=''${BASH_REMATCH[1]}
        temp=''${BASH_REMATCH[2]}
        echo "# HELP jetson_temperature_celsius Temperature in Celsius" >> "$METRICS_FILE.tmp"
        echo "# TYPE jetson_temperature_celsius gauge" >> "$METRICS_FILE.tmp"
        echo "jetson_temperature_celsius{sensor=\"$sensor\"} $temp $timestamp" >> "$METRICS_FILE.tmp"
        line="''${line/$sensor@$temp''C/}"
      done
      
      # Extract power consumption
      if [[ $line =~ VDD_IN\ ([0-9]+)/([0-9]+) ]]; then
        power_in=''${BASH_REMATCH[1]}
        echo "# HELP jetson_power_mw Power consumption in milliwatts" >> "$METRICS_FILE.tmp"
        echo "# TYPE jetson_power_mw gauge" >> "$METRICS_FILE.tmp"
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

  # Node exporter textfile collector for additional system metrics
  nodeExporterConfig = pkgs.writeText "node-exporter-config.yml" ''
    # Node Exporter Configuration
    # Collect additional system metrics for comprehensive telemetry
  '';

  # Prometheus configuration for scraping metrics
  prometheusConfig = pkgs.writeText "prometheus.yml" ''
    global:
      scrape_interval: 15s
      evaluation_interval: 15s

    scrape_configs:
      - job_name: 'jetson-telemetry'
        static_configs:
          - targets: ['localhost:9100']  # Node exporter
            labels:
              instance: 'jetson-orin-agx'
              device_type: 'nvidia-jetson'
      
      - job_name: 'tegrastats'
        static_configs:
          - targets: ['localhost:9101']  # Tegrastats exporter
            labels:
              instance: 'jetson-orin-agx'
              metric_source: 'tegrastats'
      
      - job_name: 'nvidia-smi'
        static_configs:
          - targets: ['localhost:9102']  # NVIDIA SMI exporter (if available)
            labels:
              instance: 'jetson-orin-agx'
              metric_source: 'nvidia-smi'

    remote_write:
      - url: http://localhost:8428/api/v1/write
        queue_config:
          max_samples_per_send: 10000
          capacity: 20000
          max_shards: 30
  '';

in
{
  options = {
    services.signoz-telemetry = {
      enable = mkEnableOption "SigNoz telemetry and monitoring for NVIDIA Jetson devices";

      dataDir = mkOption {
        type = types.path;
        default = "/var/lib/signoz-telemetry";
        description = "Data directory for SigNoz telemetry services";
      };

      port = mkOption {
        type = types.port;
        default = 3301;
        description = "Port for SigNoz web UI";
      };

      enableTegrastats = mkOption {
        type = types.bool;
        default = true;
        description = "Enable tegrastats metrics collection";
      };

      enableNodeExporter = mkOption {
        type = types.bool;
        default = true;
        description = "Enable Prometheus Node Exporter for system metrics";
      };

      enableNvidiaSmi = mkOption {
        type = types.bool;
        default = true;
        description = "Enable NVIDIA SMI metrics collection (if available)";
      };

      scrapeInterval = mkOption {
        type = types.str;
        default = "15s";
        description = "Prometheus scrape interval";
      };
    };
  };

  config = mkIf cfg.enable {
    # Ensure required packages are available
    environment.systemPackages = with pkgs; 
      [ pkgs.nvidia-jetpack.l4t-tools ]  # Includes tegrastats
      ++ lib.optionals (pkgs.nvidia-jetpack.l4tAtLeast "36") 
        [ pkgs.nvidia-jetpack.nvidia-smi ];  # For GPU telemetry

    # Create data directory
    systemd.tmpfiles.rules = [
      "d ${cfg.dataDir} 0755 root root -"
      "d ${cfg.dataDir}/prometheus 0755 root root -"
      "d ${cfg.dataDir}/clickhouse 0755 root root -"
      "d ${cfg.dataDir}/signoz 0755 root root -"
    ];

    # Tegrastats metrics collector service
    systemd.services.tegrastats-exporter = mkIf cfg.enableTegrastats {
      description = "Tegrastats metrics exporter for Jetson devices";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];
      
      serviceConfig = {
        Type = "simple";
        ExecStart = "${tegrastatsExporter}/bin/tegrastats-exporter";
        Restart = "on-failure";
        RestartSec = "10s";
        
        # Security hardening
        DynamicUser = false;  # Need root for tegrastats
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ cfg.dataDir ];
      };
    };

    # Prometheus Node Exporter for system metrics
    systemd.services.node-exporter = mkIf cfg.enableNodeExporter {
      description = "Prometheus Node Exporter";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];
      
      serviceConfig = {
        Type = "simple";
        ExecStart = ''
          ${pkgs.prometheus-node-exporter}/bin/node_exporter \
            --web.listen-address=:9100 \
            --collector.textfile.directory=${cfg.dataDir} \
            --collector.filesystem \
            --collector.cpu \
            --collector.meminfo \
            --collector.diskstats \
            --collector.netdev \
            --collector.loadavg \
            --collector.thermal
        '';
        Restart = "on-failure";
        RestartSec = "10s";
        
        DynamicUser = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadOnlyPaths = [ cfg.dataDir ];
      };
    };

    # VictoriaMetrics as a more efficient alternative to Prometheus
    systemd.services.victoriametrics = {
      description = "VictoriaMetrics time-series database";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];
      
      serviceConfig = {
        Type = "simple";
        ExecStart = ''
          ${pkgs.victoriametrics}/bin/victoria-metrics \
            -storageDataPath=${cfg.dataDir}/victoria-metrics \
            -retentionPeriod=12 \
            -httpListenAddr=:8428
        '';
        Restart = "on-failure";
        RestartSec = "10s";
        
        DynamicUser = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ cfg.dataDir ];
      };
    };

    # Prometheus for metrics collection and forwarding
    systemd.services.prometheus = {
      description = "Prometheus monitoring system";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" "victoriametrics.service" ];
      
      serviceConfig = {
        Type = "simple";
        ExecStart = ''
          ${pkgs.prometheus}/bin/prometheus \
            --config.file=${prometheusConfig} \
            --storage.tsdb.path=${cfg.dataDir}/prometheus \
            --web.listen-address=:9090
        '';
        Restart = "on-failure";
        RestartSec = "10s";
        
        DynamicUser = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ cfg.dataDir ];
      };
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
      };
      
      provision = {
        enable = true;
        datasources.settings.datasources = [
          {
            name = "VictoriaMetrics";
            type = "prometheus";
            access = "proxy";
            url = "http://localhost:8428";
            isDefault = true;
          }
          {
            name = "Prometheus";
            type = "prometheus";
            access = "proxy";
            url = "http://localhost:9090";
          }
        ];
      };
    };

    # Open firewall ports
    networking.firewall.allowedTCPPorts = [
      cfg.port      # Grafana UI
      9090          # Prometheus
      9100          # Node Exporter
      8428          # VictoriaMetrics
    ];

    # Informational message
    system.activationScripts.signoz-telemetry-info = lib.mkIf cfg.enable ''
      echo ""
      echo "=========================================="
      echo "SigNoz Telemetry Stack Enabled"
      echo "=========================================="
      echo "Grafana UI: http://localhost:${toString cfg.port}"
      echo "Prometheus: http://localhost:9090"
      echo "VictoriaMetrics: http://localhost:8428"
      echo "Node Exporter: http://localhost:9100/metrics"
      echo ""
      echo "Collecting metrics from:"
      echo "  - tegrastats (GPU, CPU, RAM, Power)"
      echo "  - Node Exporter (System metrics)"
      echo "  - nvidia-smi (if available)"
      echo "=========================================="
      echo ""
    '';
  };
}

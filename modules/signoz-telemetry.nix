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
    METRICS_FILE="/var/lib/signoz-telemetry/tegrastats-metrics.prom"
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

  # OpenTelemetry Collector configuration
  otelCollectorConfig = pkgs.writeText "otel-collector-config.yaml" ''
    receivers:
      prometheus:
        config:
          scrape_configs:
            # Tegrastats metrics
            - job_name: 'jetson-tegrastats'
              scrape_interval: 5s
              static_configs:
                - targets: ['localhost:9101']
                  labels:
                    device: 'jetson-orin-agx'
                    source: 'tegrastats'
            
            # Node Exporter metrics
            - job_name: 'jetson-system'
              scrape_interval: 15s
              static_configs:
                - targets: ['localhost:9100']
                  labels:
                    device: 'jetson-orin-agx'
                    source: 'node-exporter'

    processors:
      batch:
        timeout: 10s
        send_batch_size: 1024
      
      # Add resource attributes
      resource:
        attributes:
          - key: service.name
            value: jetson-telemetry
            action: upsert
          - key: device.type
            value: nvidia-jetson-orin-agx
            action: upsert

    exporters:
      otlp:
        endpoint: localhost:4317
        tls:
          insecure: true

    service:
      pipelines:
        metrics:
          receivers: [prometheus]
          processors: [batch, resource]
          exporters: [otlp]
  '';

  # SigNoz Docker Compose configuration
  signozComposeFile = pkgs.writeText "signoz-docker-compose.yaml" ''
    version: "3.9"

    x-clickhouse-defaults: &clickhouse-defaults
      restart: on-failure
      image: clickhouse/clickhouse-server:23.11.1-alpine
      tty: true
      depends_on:
        - zookeeper-1
      logging:
        options:
          max-size: 50m
          max-file: "3"
      healthcheck:
        test: ['CMD', 'wget', '--spider', '-q', 'localhost:8123/ping']
        interval: 30s
        timeout: 5s
        retries: 3
      ulimits:
        nproc: 65535
        nofile:
          soft: 262144
          hard: 262144

    services:
      zookeeper-1:
        image: bitnami/zookeeper:3.7.1
        container_name: signoz-zookeeper-1
        hostname: zookeeper-1
        user: root
        ports:
          - "2181:2181"
          - "2888:2888"
          - "3888:3888"
        volumes:
          - ${cfg.dataDir}/zookeeper-1:/bitnami/zookeeper
        environment:
          - ZOO_SERVER_ID=1
          - ALLOW_ANONYMOUS_LOGIN=yes
          - ZOO_AUTOPURGE_INTERVAL=1

      clickhouse:
        <<: *clickhouse-defaults
        container_name: signoz-clickhouse
        hostname: clickhouse
        ports:
          - "9000:9000"
          - "8123:8123"
          - "9181:9181"
        volumes:
          - ${cfg.dataDir}/clickhouse:/var/lib/clickhouse/
        environment:
          - CLICKHOUSE_DB=signoz

      query-service:
        image: signoz/query-service:0.39.0
        container_name: signoz-query-service
        command:
          [
            "-config=/root/config/prometheus.yml"
          ]
        ports:
          - "6060:6060"
          - "8080:8080"
        volumes:
          - ${cfg.dataDir}/prometheus.yml:/root/config/prometheus.yml
        environment:
          - ClickHouseUrl=tcp://clickhouse:9000
          - STORAGE=clickhouse
          - GODEBUG=netdns=go
          - TELEMETRY_ENABLED=true
          - DEPLOYMENT_TYPE=docker-standalone-amd
        depends_on:
          - clickhouse

      otel-collector:
        image: signoz/signoz-otel-collector:0.88.11
        container_name: signoz-otel-collector
        command:
          [
            "--config=/etc/otel-collector-config.yaml"
          ]
        user: root
        volumes:
          - ${otelCollectorConfig}:/etc/otel-collector-config.yaml
        ports:
          - "4317:4317"     # OTLP gRPC receiver
          - "4318:4318"     # OTLP HTTP receiver
        depends_on:
          - clickhouse

      frontend:
        image: signoz/frontend:0.39.0
        container_name: signoz-frontend
        restart: on-failure
        depends_on:
          - query-service
        ports:
          - "${toString cfg.port}:3301"
        volumes:
          - ${cfg.dataDir}/nginx-config.conf:/etc/nginx/conf.d/default.conf

    networks:
      default:
        name: signoz-network
  '';

  # Prometheus config for SigNoz query service
  prometheusYml = pkgs.writeText "prometheus.yml" ''
    global:
      scrape_interval: 60s

    scrape_configs:
      - job_name: 'otel-collector'
        static_configs:
          - targets: ['otel-collector:8888']
  '';

  # Nginx config for SigNoz frontend
  nginxConfig = pkgs.writeText "nginx-config.conf" ''
    server {
      listen 3301;
      server_name _;

      location / {
        root /usr/share/nginx/html;
        index index.html;
        try_files $uri $uri/ /index.html;
      }

      location /api {
        proxy_pass http://query-service:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
      }
    }
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
        description = "Enable tegrastats metrics collection with fine-grained GPU metrics";
      };

      enableNodeExporter = mkOption {
        type = types.bool;
        default = true;
        description = "Enable Prometheus Node Exporter for system metrics";
      };
    };
  };

  config = mkIf cfg.enable {
    # Ensure required packages are available
    environment.systemPackages = with pkgs; 
      [ pkgs.nvidia-jetpack.l4t-tools ]
      ++ lib.optionals (pkgs.nvidia-jetpack.l4tAtLeast "36") 
        [ pkgs.nvidia-jetpack.nvidia-smi ];

    # Enable Docker for SigNoz stack
    virtualisation.docker = {
      enable = true;
      autoPrune.enable = true;
    };

    # Create data directory structure
    systemd.tmpfiles.rules = [
      "d ${cfg.dataDir} 0755 root root -"
      "d ${cfg.dataDir}/clickhouse 0755 root root -"
      "d ${cfg.dataDir}/zookeeper-1 0755 root root -"
      "L+ ${cfg.dataDir}/prometheus.yml - - - - ${prometheusYml}"
      "L+ ${cfg.dataDir}/nginx-config.conf - - - - ${nginxConfig}"
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

    # Simple HTTP server to expose tegrastats metrics
    systemd.services.tegrastats-http-server = mkIf cfg.enableTegrastats {
      description = "HTTP server for tegrastats Prometheus metrics";
      wantedBy = [ "multi-user.target" ];
      after = [ "tegrastats-exporter.service" ];
      requires = [ "tegrastats-exporter.service" ];
      
      serviceConfig = {
        Type = "simple";
        ExecStart = ''
          ${pkgs.python3}/bin/python3 -m http.server 9101 \
            --directory ${cfg.dataDir} \
            --bind localhost
        '';
        WorkingDirectory = cfg.dataDir;
        Restart = "on-failure";
        RestartSec = "5s";
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
      };
    };

    # SigNoz stack via Docker Compose
    systemd.services.signoz-stack = {
      description = "SigNoz Observability Platform";
      wantedBy = [ "multi-user.target" ];
      after = [ "docker.service" "network-online.target" ];
      requires = [ "docker.service" ];
      
      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        WorkingDirectory = cfg.dataDir;
        
        ExecStartPre = [
          "${pkgs.coreutils}/bin/mkdir -p ${cfg.dataDir}"
          "${pkgs.coreutils}/bin/cp ${signozComposeFile} ${cfg.dataDir}/docker-compose.yaml"
        ];
        
        ExecStart = ''
          ${pkgs.docker-compose}/bin/docker-compose \
            -f ${cfg.dataDir}/docker-compose.yaml \
            up -d
        '';
        
        ExecStop = ''
          ${pkgs.docker-compose}/bin/docker-compose \
            -f ${cfg.dataDir}/docker-compose.yaml \
            down
        '';
        
        ExecReload = ''
          ${pkgs.docker-compose}/bin/docker-compose \
            -f ${cfg.dataDir}/docker-compose.yaml \
            restart
        '';
      };
    };

    # Open firewall ports
    networking.firewall.allowedTCPPorts = [
      cfg.port      # SigNoz Frontend UI
      4317          # OTLP gRPC
      4318          # OTLP HTTP
      9100          # Node Exporter
      9101          # Tegrastats HTTP server
    ];

    # Informational message
    system.activationScripts.signoz-telemetry-info = lib.mkIf cfg.enable ''
      echo ""
      echo "=========================================="
      echo "SigNoz Telemetry Stack Enabled"
      echo "=========================================="
      echo "SigNoz UI: http://localhost:${toString cfg.port}"
      echo ""
      echo "Collecting fine-grained metrics from:"
      echo "  - Tegrastats (GPU, CPU, RAM, Power, Temps)"
      echo "  - Node Exporter (System metrics)"
      echo ""
      echo "GPU Metrics include:"
      echo "  - GPU usage % and frequency"
      echo "  - EMC (memory controller) usage and freq"
      echo "  - VIC (video) and APE (audio) frequencies"
      echo "  - Per-core CPU usage and frequencies"
      echo "  - Temperature sensors (all zones)"
      echo "  - Power rails (VDD_IN, CPU_GPU_CV, SOC)"
      echo ""
      echo "Wait 30-60 seconds for services to start"
      echo "=========================================="
      echo ""
    '';
  };
}

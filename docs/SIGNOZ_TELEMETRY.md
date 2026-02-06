# SigNoz Telemetry Module for NVIDIA Jetson Devices

This module provides comprehensive telemetry and monitoring for NVIDIA Jetson devices (Orin AGX, Xavier, etc.) with a modern web UI powered by Grafana.

## Features

- **Jetson-Specific Metrics** via `tegrastats`:
  - GPU utilization and frequency
  - CPU usage per core and frequency
  - RAM and SWAP usage
  - Temperature sensors (CPU, GPU, SOC, thermal diode, etc.)
  - Power consumption (VDD_IN, VDD_CPU_GPU_CV, VDD_SOC)
  - EMC frequency

- **System Metrics** via Prometheus Node Exporter:
  - CPU usage and load average
  - Memory and swap usage
  - Disk I/O and capacity
  - Network traffic
  - Filesystem usage
  - System temperatures

- **Modern WebUI**:
  - Real-time Grafana dashboards
  - Pre-configured data sources
  - Time-series visualization
  - Efficient metric storage with VictoriaMetrics

## Quick Start

### Using Pre-configured Profile

```bash
sudo nixos-rebuild switch --flake .#signoz
```

## Accessing the Dashboard

Access the Grafana dashboard at: http://localhost:3301

## Configuration

See examples/signoz-telemetry-example.nix for detailed configuration options.

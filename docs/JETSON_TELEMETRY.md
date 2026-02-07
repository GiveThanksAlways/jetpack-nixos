# Native NixOS Telemetry for NVIDIA Jetson

Pure NixOS observability stack - no Docker required!

## Quick Start

```bash
sudo nixos-rebuild switch --flake .#signoz
```

Access Grafana: http://localhost:3301

## Features

- ✅ Pure NixOS systemd services
- ✅ Fine-grained GPU telemetry (500ms)
- ✅ Auto-provisioned dashboard
- ✅ No Docker dependencies

## Setup

See [SETUP.md](SETUP.md) for complete guide.

## Configuration

```nix
services.jetson-telemetry = {
  enable = true;
  enableTegrastats = true;      # GPU metrics
  enableNodeExporter = true;     # System metrics
};
```


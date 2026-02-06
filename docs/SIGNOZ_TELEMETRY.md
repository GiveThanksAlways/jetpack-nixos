# SigNoz Telemetry for NVIDIA Jetson

Modern observability platform with fine-grained GPU telemetry.

## Quick Start

```bash
sudo nixos-rebuild switch --flake .#signoz
```

Access SigNoz: http://localhost:3301

## Setup Guide

See [SETUP.md](SETUP.md) for complete TL;DR guide.

## Configuration

```nix
services.signoz-telemetry = {
  enable = true;
  enableTegrastats = true;      # Fine-grained GPU metrics (500ms)
  enableNodeExporter = true;     # System metrics
};
```


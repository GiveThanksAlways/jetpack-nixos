# Telemetry Viewer (run from your PC)

View Jetson GPU/system dashboards in Chrome on your local machine.

## On the Jetson (one-time)

Enable telemetry in your NixOS config:

```nix
# configuration.nix (or flake module)
services.jetson-telemetry = {
  enable = true;
  # enableOpenTelemetry = true;  # optional: adds OTLP collector
};
```

Rebuild:

```bash
sudo nixos-rebuild switch
```

## On your PC

```bash
cd examples/telemetry-viewer
./connect-telemetry.sh <jetson-ip>
# or with a specific user:
./connect-telemetry.sh <jetson-ip> myuser
```

Open Chrome: `http://localhost:3301`

Navigate: Dashboards -> Jetson -> Jetson Mission Control

## What you see

- GPU load gauge + utilization over time
- GPU frequency
- Per-core CPU usage
- RAM/SWAP gauges and timeseries
- All thermal sensors with color thresholds
- Power draw per rail (VDD_IN, CPU_GPU_CV, SOC)
- EMC (memory controller) load gauge
- VIC / APE engine stats

## Toggle telemetry off (for max performance)

```nix
services.jetson-telemetry.enable = false;
```

Rebuild and all telemetry services stop. Zero overhead.

# Native NixOS Telemetry Setup - TL;DR

## Pure NixOS Implementation - No Docker! 🎉

All services run as native NixOS systemd services.

### Quick Start
```bash
git pull
sudo nixos-rebuild switch --flake .#jetson-telemetry
```

**Grafana UI**: http://<jetson-ip>:3301

### What You Get
- ✅ Pure NixOS (no Docker)
- ✅ Fine-grained GPU telemetry (500ms)
- ✅ Auto-provisioned GPU dashboard
- ✅ Instant startup

## Architecture

```
Jetson (Pure NixOS)
├── tegrastats-exporter (systemd)
├── node-exporter (systemd)
├── prometheus (systemd)
└── grafana (systemd) → :3301
```

## Key Metrics

- `jetson_gpu_usage_percent` - GPU load
- `jetson_gpu_freq_mhz` - GPU frequency
- `jetson_emc_freq_percent` - Memory controller
- `jetson_cpu_usage_percent{core="N"}` - Per-core CPU
- `jetson_temperature_celsius{sensor="..."}` - Temps
- `jetson_power_mw{rail="..."}` - Power

## Troubleshooting

```bash
# Check services
sudo systemctl status grafana prometheus

# View logs
sudo journalctl -u grafana -f

# Check metrics
cat /var/lib/jetson-telemetry-telemetry/metrics/tegrastats-metrics.prom
```

## Why Native NixOS?

- ✅ No Docker overhead
- ✅ Instant startup (~500MB vs ~2GB)
- ✅ Native systemd integration
- ✅ Simpler configuration

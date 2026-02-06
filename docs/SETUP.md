# SigNoz Telemetry Setup - TL;DR

## On Jetson Device (Orin AGX)

### Quick Start
```bash
# Pull and activate
git pull origin copilot/add-signoz-telemetry-webui
sudo nixos-rebuild switch --flake .#signoz

# Wait 30-60 seconds for services to start
```

### What You Get
- **SigNoz UI**: http://\<jetson-ip\>:3301
- **Fine-grained GPU telemetry** (500ms sampling)
- **Modern space-age visualizations**
- **GPU-specific dashboard**

## On Your PC

### View Telemetry
1. **Open browser**: http://\<jetson-ip\>:3301
2. **Navigate to Dashboards**
3. **Views available**:
   - **Services** → See jetson-telemetry service
   - **Metrics** → Query GPU/CPU/Power metrics
   - **Logs** → System logs (if enabled)

### GPU-Specific Metrics
In SigNoz Metrics Explorer, query:
- `jetson_gpu_usage_percent` - GPU utilization
- `jetson_gpu_freq_mhz` - GPU frequency
- `jetson_emc_freq_percent` - Memory controller load
- `jetson_vic_freq` - Video engine freq
- `jetson_temperature_celsius{sensor="GPU"}` - GPU temp
- `jetson_power_mw{rail="VDD_CPU_GPU_CV"}` - GPU power

### Create GPU Dashboard
1. Go to **Dashboards** → **New Dashboard**
2. **Add Panel**:
   - **Panel 1**: GPU Usage (Query: `jetson_gpu_usage_percent`)
   - **Panel 2**: GPU Freq (Query: `jetson_gpu_freq_mhz`)
   - **Panel 3**: GPU Temp (Query: `jetson_temperature_celsius{sensor="GPU"}`)
   - **Panel 4**: GPU Power (Query: `jetson_power_mw{rail="VDD_CPU_GPU_CV"}`)
   - **Panel 5**: Memory Controller (Query: `jetson_emc_freq_percent`)
   - **Panel 6**: Video Engine (Query: `jetson_vic_freq`)
3. **Save** as "Jetson GPU Telemetry"

## Metrics Available

### GPU Metrics
- `jetson_gpu_usage_percent` - GPU utilization %
- `jetson_gpu_freq_mhz` - GPU clock speed
- `jetson_emc_freq_percent` - Memory controller usage %
- `jetson_emc_freq_mhz` - Memory controller frequency
- `jetson_vic_freq` - Video image compositor freq
- `jetson_ape_freq` - Audio processing engine freq

### CPU Metrics
- `jetson_cpu_usage_percent{core="N"}` - Per-core CPU usage
- `jetson_cpu_freq_mhz{core="N"}` - Per-core frequency

### Memory Metrics
- `jetson_ram_used_mb` / `jetson_ram_total_mb`
- `jetson_swap_used_mb` / `jetson_swap_total_mb`

### Temperature Metrics
- `jetson_temperature_celsius{sensor="GPU|CPU|SOC0|SOC1|SOC2|..."}` 

### Power Metrics
- `jetson_power_mw{rail="VDD_IN|VDD_CPU_GPU_CV|VDD_SOC"}`

## Troubleshooting

### Services not starting?
```bash
# Check Docker
sudo systemctl status docker

# Check SigNoz containers
sudo docker ps

# Restart SigNoz
sudo systemctl restart signoz-stack
```

### Can't access UI?
```bash
# Check firewall
sudo iptables -L -n | grep 3301

# Check SigNoz frontend
sudo docker logs signoz-frontend
```

### No metrics?
```bash
# Check tegrastats exporter
sudo systemctl status tegrastats-exporter

# Check metrics file
cat /var/lib/signoz-telemetry/tegrastats-metrics.prom
```

## Advanced

### Custom Query Examples
```promql
# Average GPU usage over 5 minutes
avg_over_time(jetson_gpu_usage_percent[5m])

# Max temperature
max(jetson_temperature_celsius)

# Total power consumption
sum(jetson_power_mw)

# CPU usage per core as heatmap
jetson_cpu_usage_percent
```

### Change Sampling Rate
Edit `/var/lib/signoz-telemetry/docker-compose.yaml`:
```yaml
# Change INTERVAL_MS in tegrastats-exporter
# 500 = 500ms (default, fine-grained)
# 1000 = 1 second
# 100 = 100ms (ultra fine-grained)
```

## Architecture

```
Jetson Device
├── tegrastats → Prometheus metrics (500ms sampling)
├── Node Exporter → System metrics
├── OTel Collector → Ingests metrics
├── ClickHouse → Stores time-series data
├── Query Service → API backend
└── SigNoz Frontend → Modern WebUI (port 3301)
```

## Resources
- SigNoz Docs: https://signoz.io/docs/
- Metrics Query: https://signoz.io/docs/userguide/query-builder/
- Dashboards: https://signoz.io/docs/userguide/manage-dashboards/

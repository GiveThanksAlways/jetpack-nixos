# SigNoz Telemetry Setup - TL;DR

## On Jetson Device (Orin AGX)

### Quick Start
```bash
# Pull and activate
git pull origin copilot/add-signoz-telemetry-webui
sudo nixos-rebuild switch --flake .#signoz

# Wait 30-60 seconds for Docker containers to start
# Check status: sudo docker ps
```

### What You Get
- **SigNoz UI**: http://\<jetson-ip\>:3301
- **Fine-grained GPU telemetry** (500ms sampling - 2x more detailed than before)
- **Modern space-age visualizations**
- **Dedicated GPU dashboard capability**
- **OpenTelemetry-native** metrics pipeline

---

## On Your PC

### Access SigNoz
1. **Open browser**: http://\<jetson-ip\>:3301
2. **First time?** You'll see the SigNoz welcome screen
3. **Navigate to**:
   - **Services** → View `jetson-telemetry` service
   - **Dashboard** → Create custom dashboards
   - **Metrics Explorer** → Query and visualize metrics

### Create GPU-Specific Dashboard

#### Option 1: Quick GPU Dashboard
1. Go to **Dashboard** → **New Dashboard**
2. Click **+ Add Panel**
3. Add these panels:

**Panel 1: GPU Utilization**
- Query: `jetson_gpu_usage_percent`
- Visualization: Time Series (Line Chart)
- Color: Gradient (green → yellow → red)

**Panel 2: GPU Frequency**
- Query: `jetson_gpu_freq_mhz`
- Visualization: Time Series
- Unit: MHz

**Panel 3: GPU Temperature**
- Query: `jetson_temperature_celsius{sensor="GPU"}`
- Visualization: Gauge
- Thresholds: 0-70 (green), 70-85 (yellow), 85+ (red)

**Panel 4: GPU Power**
- Query: `jetson_power_mw{rail="VDD_CPU_GPU_CV"}`
- Visualization: Time Series (Area)
- Unit: Milliwatts

**Panel 5: Memory Controller Load**
- Query: `jetson_emc_freq_percent`
- Visualization: Time Series
- Description: Memory bandwidth usage

**Panel 6: Video Engine Activity**
- Query: `jetson_vic_freq`
- Visualization: Stat (single value)
- Description: Hardware video processing

4. **Save** as "Jetson GPU Command Center" 🚀

#### Option 2: Advanced GPU Heatmap
Create a panel with:
- Query: `jetson_cpu_usage_percent`
- Visualization: Heatmap
- Group by: `core` label
- Shows all CPU cores as color-coded heatmap

---

## All Available Metrics

### 🎮 GPU Metrics (Fine-Grained)
```promql
jetson_gpu_usage_percent          # GPU utilization 0-100%
jetson_gpu_freq_mhz               # GPU clock speed
jetson_emc_freq_percent           # Memory controller usage %
jetson_emc_freq_mhz               # Memory controller frequency
jetson_vic_freq                   # Video image compositor
jetson_ape_freq                   # Audio processing engine
```

### 💻 CPU Metrics (Per-Core)
```promql
jetson_cpu_usage_percent{core="0"}   # CPU core 0 usage
jetson_cpu_freq_mhz{core="0"}        # CPU core 0 frequency
# Cores: 0-11 for Orin AGX (12 cores total)
```

### 🧠 Memory Metrics
```promql
jetson_ram_used_mb                # RAM in use
jetson_ram_total_mb               # Total RAM
jetson_swap_used_mb               # Swap usage
jetson_swap_total_mb              # Total swap
```

### 🌡️ Temperature Metrics (All Sensors)
```promql
jetson_temperature_celsius{sensor="GPU"}     # GPU temp
jetson_temperature_celsius{sensor="CPU"}     # CPU temp
jetson_temperature_celsius{sensor="SOC0"}    # SOC zone 0
jetson_temperature_celsius{sensor="SOC1"}    # SOC zone 1
jetson_temperature_celsius{sensor="SOC2"}    # SOC zone 2
jetson_temperature_celsius{sensor="Tboard"}  # Board temp
jetson_temperature_celsius{sensor="Tdiode"}  # Thermal diode
```

### ⚡ Power Metrics
```promql
jetson_power_mw{rail="VDD_IN"}           # Total input power
jetson_power_mw{rail="VDD_CPU_GPU_CV"}   # CPU+GPU power
jetson_power_mw{rail="VDD_SOC"}          # SOC power
```

---

## Advanced Queries

### GPU Efficiency
```promql
# GPU utilization per watt
jetson_gpu_usage_percent / (jetson_power_mw{rail="VDD_CPU_GPU_CV"} / 1000)
```

### Temperature Hotspot
```promql
# Highest temperature sensor
max(jetson_temperature_celsius)
```

### Total System Power
```promql
# Sum of all power rails
sum(jetson_power_mw) / 1000  # Convert to watts
```

### CPU Core Imbalance
```promql
# Standard deviation of CPU usage across cores
stddev(jetson_cpu_usage_percent)
```

### GPU Memory Bandwidth
```promql
# EMC frequency as proxy for memory bandwidth
jetson_emc_freq_mhz * jetson_emc_freq_percent / 100
```

---

## Troubleshooting

### Services not starting?
```bash
# Check Docker
sudo systemctl status docker

# Check SigNoz containers
sudo docker ps -a

# View logs
sudo docker logs signoz-frontend
sudo docker logs signoz-query-service
sudo docker logs signoz-clickhouse

# Restart SigNoz
sudo systemctl restart signoz-stack
```

### Can't access UI?
```bash
# Check firewall
sudo iptables -L -n | grep 3301

# Check if port is listening
sudo ss -tlnp | grep 3301

# Check frontend container
sudo docker logs signoz-frontend -f
```

### No metrics showing?
```bash
# Check tegrastats exporter
sudo systemctl status tegrastats-exporter

# Check metrics file exists
ls -lh /var/lib/signoz-telemetry/tegrastats-metrics.prom

# View raw metrics
cat /var/lib/signoz-telemetry/tegrastats-metrics.prom

# Check OTel collector
sudo docker logs signoz-otel-collector -f
```

### Metrics delayed or missing?
```bash
# Check scrape interval (default 5s for GPU, 15s for system)
# Edit if needed: /var/lib/signoz-telemetry/docker-compose.yaml

# Restart services
sudo systemctl restart tegrastats-exporter
sudo systemctl restart signoz-stack
```

---

## Customization

### Change GPU Sampling Rate
Edit tegrastats interval in `/var/lib/signoz-telemetry/docker-compose.yaml`:
```yaml
# In tegrastats-exporter service:
# INTERVAL_MS=500   # 500ms (current - fine-grained)
# INTERVAL_MS=100   # 100ms (ultra fine-grained)
# INTERVAL_MS=1000  # 1 second (standard)
```

Then restart:
```bash
sudo systemctl restart tegrastats-exporter
```

### Add Custom Metrics
SigNoz ingests OpenTelemetry metrics. Add your own exporters:
1. Create Prometheus exporter
2. Add to OTel collector config
3. Restart `signoz-stack`

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│ Jetson Orin AGX Device                          │
├─────────────────────────────────────────────────┤
│                                                 │
│  tegrastats (500ms) ──► Prometheus Exporter    │
│                          (port 9101)            │
│                              │                  │
│  Node Exporter ──────────────┤                 │
│  (port 9100)                 │                  │
│                              ▼                  │
│                    ┌──────────────────┐         │
│                    │ OTel Collector   │         │
│                    │ (scrapes metrics)│         │
│                    └────────┬─────────┘         │
│                             │                   │
│                             ▼                   │
│                    ┌──────────────────┐         │
│                    │   ClickHouse     │         │
│                    │ (time-series DB) │         │
│                    └────────┬─────────┘         │
│                             │                   │
│                             ▼                   │
│                    ┌──────────────────┐         │
│                    │  Query Service   │         │
│                    │   (API Backend)  │         │
│                    └────────┬─────────┘         │
│                             │                   │
│                             ▼                   │
│                    ┌──────────────────┐         │
│                    │  SigNoz Frontend │         │
│                    │   (port 3301)    │         │
│                    └──────────────────┘         │
│                             │                   │
└─────────────────────────────┼───────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Your PC        │
                    │   Browser        │
                    │ (Modern WebUI)   │
                    └──────────────────┘
```

---

## Pro Tips

1. **Use time range selector** in top-right to view different time periods
2. **Create alerts** in SigNoz for temperature or power thresholds
3. **Export dashboards** as JSON for backup/sharing
4. **Use PromQL functions** like `rate()`, `avg_over_time()`, `predict_linear()`
5. **Group panels by category** for better organization
6. **Add annotations** to mark important events
7. **Share dashboard links** with your team

---

## Resources
- **SigNoz Docs**: https://signoz.io/docs/
- **Query Builder**: https://signoz.io/docs/userguide/query-builder/
- **Dashboards**: https://signoz.io/docs/userguide/manage-dashboards/
- **PromQL Guide**: https://prometheus.io/docs/prometheus/latest/querying/basics/


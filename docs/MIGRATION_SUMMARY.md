# Grafana → SigNoz Migration Summary

## What Changed

### Removed
- ❌ Grafana
- ❌ Prometheus (standalone)
- ❌ VictoriaMetrics
- ❌ Pre-configured Grafana dashboards (JSON)

### Added
- ✅ **SigNoz** - Native observability platform
- ✅ **ClickHouse** - High-performance time-series storage
- ✅ **SigNoz Query Service** - Backend API
- ✅ **SigNoz Frontend** - Modern React WebUI
- ✅ **OpenTelemetry Collector** - Standards-compliant metrics ingestion
- ✅ **Fine-grained GPU metrics** - 500ms sampling (was 1000ms)
- ✅ **Additional GPU metrics** - EMC, VIC, APE frequencies

## Why SigNoz?

### Modern Architecture
- **Cloud-native** - Built for containers and Kubernetes
- **OpenTelemetry-first** - Industry standard for observability
- **High performance** - ClickHouse for fast queries
- **Better UX** - Modern React UI vs legacy Grafana

### Benefits
1. **All-in-one platform** - Metrics, traces, logs in single UI
2. **Better performance** - ClickHouse faster than Prometheus
3. **Modern UI** - Space-age visualizations
4. **Easy dashboards** - No JSON editing required
5. **OpenTelemetry native** - Future-proof standards

## Configuration Comparison

### Before (Grafana)
```nix
services.signoz-telemetry = {
  enable = true;
  enableTegrastats = true;
  enableNodeExporter = true;
  scrapeInterval = "15s";
};
```

### After (SigNoz)
```nix
services.signoz-telemetry = {
  enable = true;
  enableTegrastats = true;    # Now 500ms sampling
  enableNodeExporter = true;
};
```

**Simpler configuration, more features!**

## Features Comparison

| Feature | Grafana Stack | SigNoz Stack |
|---------|--------------|--------------|
| WebUI | Grafana (legacy) | SigNoz (modern) |
| Storage | VictoriaMetrics | ClickHouse |
| Query Language | PromQL | PromQL |
| Metrics Ingestion | Prometheus | OTel Collector |
| Dashboard Creation | JSON editing | UI wizard |
| GPU Sampling | 1000ms | 500ms |
| Additional Metrics | - | EMC, VIC, APE |
| Deployment | Multiple services | Docker Compose |
| Port | 3301 | 3301 |

## Migration Path

### If you're using old config:
1. `git pull` to get latest changes
2. `sudo nixos-rebuild switch --flake .#signoz`
3. Old services will be stopped
4. SigNoz will start (wait 30-60s)
5. Access: http://\<jetson-ip\>:3301

### Data Migration
**Note**: Historical data from Grafana/VictoriaMetrics is NOT migrated.
- SigNoz starts collecting fresh metrics
- Old data remains in `/var/lib/signoz-telemetry` until manually removed
- Consider backing up old dashboards (JSON) if needed

## New Capabilities

### 1. Fine-Grained GPU Telemetry
- **500ms sampling** (2x more detailed)
- Memory controller (EMC) metrics
- Video engine (VIC) frequency
- Audio engine (APE) frequency

### 2. Modern Dashboard Creation
- **No JSON required** - Use UI wizard
- Drag-and-drop panels
- Live query builder
- Visual threshold editing

### 3. Advanced Queries
- GPU efficiency calculations
- Temperature hotspot detection
- Power consumption analysis
- CPU core imbalance metrics

### 4. Better Performance
- ClickHouse faster than Prometheus
- Efficient compression
- Faster dashboard loading
- Better query performance

## Troubleshooting

### Missing old dashboards?
Old Grafana dashboards are gone. Create new ones in SigNoz:
1. Go to Dashboards → New Dashboard
2. Add panels using UI wizard
3. Use same PromQL queries
4. Save for later use

See `docs/SETUP.md` for GPU dashboard creation guide.

### Want old data?
Old metrics in VictoriaMetrics format cannot be imported to ClickHouse.
Options:
1. Keep old system running temporarily
2. Export critical data before migration
3. Accept fresh start with better platform

### Performance issues?
SigNoz requires more resources than Grafana:
- **CPU**: ~5-10% (vs 2% for Grafana)
- **Memory**: ~1-2GB (vs 500MB for Grafana)
- **Storage**: ClickHouse more efficient long-term

Adjust if needed:
```bash
# Edit Docker Compose resources
sudo vi /var/lib/signoz-telemetry/docker-compose.yaml
sudo systemctl restart signoz-stack
```

## Resources

- **Setup Guide**: `docs/SETUP.md`
- **SigNoz Docs**: https://signoz.io/docs/
- **Dashboard Guide**: `dashboards/README.md`
- **Query Guide**: https://signoz.io/docs/userguide/query-builder/

## Summary

✅ **Better technology** - SigNoz is modern, performant, standards-based
✅ **More metrics** - Fine-grained GPU telemetry with 500ms sampling
✅ **Better UX** - Modern UI, easier dashboard creation
✅ **Future-proof** - OpenTelemetry is the industry standard

The migration delivers a superior observability platform with enhanced GPU monitoring capabilities.

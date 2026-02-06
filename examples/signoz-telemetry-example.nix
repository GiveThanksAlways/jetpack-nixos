# Example NixOS configuration for enabling SigNoz telemetry on Jetson devices
#
# This configuration enables comprehensive fine-grained telemetry collection:
# - GPU metrics (500ms sampling) via tegrastats
# - CPU, memory, storage, network via Node Exporter
# - Power consumption and temperature monitoring
# - Modern SigNoz WebUI with space-age visualizations
#
# To use this configuration:
# 1. Copy to your NixOS configuration directory
# 2. Import in configuration.nix or flake.nix
# 3. Run: sudo nixos-rebuild switch --flake .#signoz
#
# After activation, access SigNoz at:
# http://localhost:3301 (or http://<jetson-ip>:3301 from PC)

{ config, pkgs, lib, ... }:

{
  # Enable SigNoz telemetry stack with Docker-based deployment
  services.signoz-telemetry = {
    enable = true;
    
    # SigNoz WebUI port
    port = 3301;
    
    # Data directory for SigNoz components
    dataDir = "/var/lib/signoz-telemetry";
    
    # Enable fine-grained GPU telemetry (500ms sampling)
    enableTegrastats = true;
    
    # Enable comprehensive system metrics
    enableNodeExporter = true;
  };

  # Docker is required for SigNoz stack
  # This is automatically enabled by the module

  # Optional: Adjust firewall if needed
  # The module already opens necessary ports:
  # - 3301: SigNoz Frontend WebUI
  # - 4317/4318: OpenTelemetry ingestion
  # - 9100/9101: Metrics exporters
}


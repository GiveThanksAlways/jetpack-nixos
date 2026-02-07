# Example NixOS configuration for native telemetry on Jetson devices
#
# Pure NixOS implementation - no Docker required!
#
# Features:
# - Fine-grained GPU telemetry (500ms sampling) via tegrastats
# - System metrics via Node Exporter
# - Grafana WebUI with auto-provisioned GPU dashboard
# - All services run as native systemd services
#
# Usage:
# 1. Import in configuration.nix or flake.nix
# 2. Run: sudo nixos-rebuild switch --flake .#signoz
# 3. Access: http://localhost:3301 (or http://<jetson-ip>:3301 from PC)

{ config, pkgs, lib, ... }:

{
  # Enable native NixOS telemetry stack
  services.jetson-telemetry = {
    enable = true;
    
    # Grafana WebUI port
    port = 3301;
    
    # Data directory
    dataDir = "/var/lib/jetson-telemetry";
    
    # Enable fine-grained GPU telemetry (500ms sampling)
    enableTegrastats = true;
    
    # Enable system metrics
    enableNodeExporter = true;
    
    # Prometheus retention time
    retentionTime = "30d";
  };

  # All required services (Grafana, Prometheus, exporters)
  # are automatically configured by the module
  
  # No Docker required!
}



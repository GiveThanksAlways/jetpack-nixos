# Example NixOS configuration for enabling SigNoz telemetry on Jetson devices
#
# This configuration enables comprehensive telemetry collection including:
# - GPU metrics via tegrastats
# - CPU, memory, and storage metrics via Node Exporter
# - Power consumption and temperature monitoring
# - Real-time visualization via Grafana
#
# To use this configuration:
# 1. Copy this file to your NixOS configuration directory
# 2. Import it in your configuration.nix or flake.nix
# 3. Run: sudo nixos-rebuild switch --flake .#signoz
#
# After activation, access the Grafana dashboard at:
# http://localhost:3301

{ config, pkgs, lib, ... }:

{
  # Enable SigNoz telemetry stack
  services.signoz-telemetry = {
    enable = true;
    
    # Web UI port (Grafana)
    port = 3301;
    
    # Data directory for metrics storage
    dataDir = "/var/lib/signoz-telemetry";
    
    # Enable all telemetry collectors
    enableTegrastats = true;    # Jetson-specific GPU/CPU/Power metrics
    enableNodeExporter = true;   # System metrics (CPU, RAM, disk, network)
    enableNvidiaSmi = true;      # Additional NVIDIA GPU metrics
    
    # How often to collect metrics
    scrapeInterval = "15s";
  };

  # Optional: Configure Grafana further
  services.grafana = {
    # Already enabled by signoz-telemetry module
    
    settings = {
      # Additional Grafana settings can be added here
      # For example, to enable authentication:
      # "auth.anonymous".enabled = false;
      # security.admin_password = "your-secure-password";
    };
  };

  # Optional: Adjust firewall rules if needed
  # The module already opens the necessary ports by default
  # networking.firewall.allowedTCPPorts = [ 3301 9090 9100 8428 ];
}

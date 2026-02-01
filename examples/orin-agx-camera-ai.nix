# Example NixOS configuration for NVIDIA Jetson Orin AGX 64GB with camera and AI vision support
#
# This configuration enables:
# - Waveshare Binocular Camera Module Dual IMX219 (stereo camera)
# - ESP32-CAM support (WiFi streaming cameras)
# - Computer vision libraries (OpenCV, GStreamer)
# - Tinygrad for ML/AI workloads
# - Stereo 3D vision processing tools
#
# Usage:
#   1. Import this module in your configuration.nix
#   2. Or use it as a base for your own configuration

{ config, lib, pkgs, ... }:

{
  # Enable Jetson Orin AGX hardware support
  hardware.nvidia-jetpack = {
    enable = true;
    som = "orin-agx";
    carrierBoard = "devkit";
    # Use JetPack 6 (default for Orin AGX)
    # majorVersion = "6";

    # Enable IMX219 camera support (for Waveshare Binocular Camera)
    cameras.imx219 = {
      enable = true;
      enableStereo = true;  # Enable stereo vision support
      enableCalibrationTools = true;
    };
  };

  # Enable GPU/graphics support (required for CUDA and computer vision)
  hardware.graphics.enable = true;

  # System packages for camera and AI vision work
  environment.systemPackages = with pkgs; [
    # NVIDIA Jetpack tools
    nvidia-jetpack.l4t-camera        # Camera libraries and tools
    nvidia-jetpack.l4t-multimedia    # Multimedia processing
    nvidia-jetpack.l4t-tools         # Jetson utilities (jetson_clocks, etc.)
    
    # Computer vision and ML
    opencv                           # Computer vision library (CUDA-enabled)
    python3Packages.opencv4          # Python bindings for OpenCV
    
    # Camera utilities
    v4l-utils                        # V4L2 camera testing tools
    ffmpeg                           # Video processing
    gstreamer                        # Media framework
    gst_all_1.gst-plugins-base       # GStreamer plugins
    gst_all_1.gst-plugins-good       # More GStreamer plugins
    gst_all_1.gst-plugins-bad        # Even more GStreamer plugins
    
    # Python environment for ML/AI
    python3                          # Python 3
    python3Packages.pip              # Package installer
    python3Packages.numpy            # Numerical computing
    python3Packages.scipy            # Scientific computing
    python3Packages.matplotlib       # Plotting
    python3Packages.pillow           # Image processing
    python3Packages.flask            # Web server for camera streaming
    python3Packages.requests         # HTTP library for ESP32-CAM
    
    # Network tools for ESP32-CAM
    curl                             # Transfer data from/to servers
    wget                             # Network downloader
    
    # Development tools
    git                              # Version control
    vim                              # Text editor
  ] ++ lib.optionals config.hardware.nvidia-jetpack.enable [
    # Additional Jetpack samples and examples
    nvidia-jetpack.multimedia-samples  # Camera and video samples
  ];

  # Kernel modules for camera support
  boot.kernelModules = [
    "v4l2loopback"  # Virtual V4L2 device support
  ];

  # Networking for ESP32-CAM access
  networking = {
    # Enable network manager for WiFi configuration
    networkmanager.enable = lib.mkDefault true;
    
    # Enable mDNS for discovering ESP32-CAM devices
    firewall = {
      enable = true;
      allowedTCPPorts = [ 
        80   # HTTP for ESP32-CAM web interface
        81   # Alternative HTTP port
        8080 # Alternative HTTP port
        8081 # Alternative HTTP port
      ];
      allowedUDPPorts = [
        5353 # mDNS
      ];
    };
  };

  # Avahi for mDNS service discovery (find ESP32-CAM on network)
  services.avahi = {
    enable = true;
    nssmdns4 = true;
    publish = {
      enable = true;
      addresses = true;
      domain = true;
      workstation = true;
    };
  };

  # I2C access for camera configuration
  services.udev.extraRules = ''
    # Allow i2c group to access I2C devices
    KERNEL=="i2c-[0-9]*", GROUP="i2c", MODE="0660"
    
    # Video devices (cameras)
    KERNEL=="video[0-9]*", GROUP="video", MODE="0660"
  '';

  # Additional notes and documentation
  system.stateVersion = lib.mkDefault "25.11";

  # Helpful environment variables
  environment.variables = {
    # Point to camera configuration directory
    NVCAM_CONFIG_DIR = "/var/nvidia/nvcam";
  };
}

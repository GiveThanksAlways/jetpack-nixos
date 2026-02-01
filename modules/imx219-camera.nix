# NixOS module for IMX219 camera support on Jetson devices
# Enables support for Sony IMX219 camera sensors commonly used in:
# - Raspberry Pi Camera Module v2
# - Waveshare Binocular Camera Module
# - Other IMX219-based camera modules

{ config, lib, pkgs, ... }:

let
  inherit (lib)
    mkEnableOption
    mkIf
    mkOption
    types
    ;

  cfg = config.hardware.nvidia-jetpack.cameras.imx219;
in
{
  options = {
    hardware.nvidia-jetpack.cameras.imx219 = {
      enable = mkEnableOption "IMX219 camera support";
      
      enableStereo = mkOption {
        type = types.bool;
        default = false;
        description = ''
          Enable stereo camera support (e.g., for Waveshare Binocular Camera Module).
          This assumes two IMX219 sensors are connected.
        '';
      };

      enableCalibrationTools = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Include camera calibration and testing tools in system packages.
        '';
      };
    };
  };

  config = mkIf (config.hardware.nvidia-jetpack.enable && cfg.enable) {
    # Enable camera support services
    services.nvargus-daemon.enable = mkIf (lib.versionAtLeast config.hardware.nvidia-jetpack.majorVersion "5") true;

    # Camera utilities and tools
    environment.systemPackages = with pkgs; [
      v4l-utils                    # V4L2 utilities for camera testing
      nvidia-jetpack.l4t-camera    # NVIDIA camera libraries
      
      # GStreamer with NVARGUS support
      gstreamer
      gst_all_1.gst-plugins-base
      gst_all_1.gst-plugins-good
      gst_all_1.gst-plugins-bad
    ] ++ lib.optionals cfg.enableCalibrationTools [
      # OpenCV for camera calibration
      opencv
      python3Packages.opencv4
    ] ++ lib.optionals cfg.enableStereo [
      # Additional tools for stereo vision
      python3Packages.numpy
      python3Packages.scipy
    ];

    # Kernel parameters for camera support
    boot.kernelParams = [
      # Enable camera debugging (optional)
      # "camera_platform.camera_debug=1"
    ];

    # Load camera-related kernel modules
    boot.kernelModules = [
      # V4L2 modules
      "v4l2loopback"
    ];

    # Extra kernel module options
    boot.extraModprobeConfig = ''
      # V4L2 loopback for virtual cameras
      options v4l2loopback devices=4 video_nr=20,21,22,23 card_label="Virtual Camera"
    '';

    # Udev rules for camera access
    services.udev.extraRules = ''
      # IMX219 camera I2C access
      SUBSYSTEM=="i2c-dev", GROUP="i2c", MODE="0660"
      
      # Video devices
      SUBSYSTEM=="video4linux", KERNEL=="video[0-9]*", GROUP="video", MODE="0660"
      
      # Media devices
      SUBSYSTEM=="media", KERNEL=="media[0-9]*", GROUP="video", MODE="0660"
    '';

    # Ensure i2c group exists
    users.groups.i2c = { };

    # Environment variables for camera support
    environment.variables = {
      # NVIDIA camera configuration directory
      NVCAM_CONFIG_DIR = "/var/nvidia/nvcam";
    };

    # Helpful message for users
    system.activationScripts.imx219-camera-info = lib.stringAfter [ "etc" ] ''
      cat > /etc/motd.d/imx219-camera << 'MOTD'
      
      ═══════════════════════════════════════════════════════════════
                IMX219 Camera Support Enabled
      ───────────────────────────────────────────────────────────────
      
      Camera utilities available:
        - v4l2-ctl: V4L2 camera control
        - gst-launch-1.0: GStreamer pipeline testing
        - nvargus-daemon: NVIDIA Argus camera daemon
      
      Quick camera test:
        $ v4l2-ctl --list-devices
        $ gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! nvoverlaysink
      
      ${lib.optionalString cfg.enableStereo ''
      Stereo camera support enabled.
      Test both cameras:
        $ gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! autovideosink
        $ gst-launch-1.0 nvarguscamerasrc sensor-id=1 ! autovideosink
      ''}
      
      Add your user to the 'video' and 'i2c' groups:
        $ sudo usermod -a -G video,i2c $USER
      
      ═══════════════════════════════════════════════════════════════
      MOTD
    '';
  };
}

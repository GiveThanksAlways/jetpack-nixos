# Example Configurations for Jetson Devices

This directory contains example NixOS configurations for various use cases with NVIDIA Jetson devices.

## Available Examples

### `orin-agx-camera-ai.nix`

Complete configuration for computer vision and AI/ML workloads on the Jetson Orin AGX 64GB developer kit with camera support.

**Features:**
- Waveshare Binocular Camera Module (Dual IMX219) support
- ESP32-CAM network camera support
- Computer vision libraries (OpenCV with CUDA)
- Machine learning tools (TensorRT, support for Tinygrad)
- Stereo 3D vision processing
- GStreamer for video processing
- Network configuration for WiFi cameras

**Usage:**

1. Import in your `configuration.nix`:
   ```nix
   imports = [
     /path/to/jetpack-nixos/examples/orin-agx-camera-ai.nix
     # your other imports
   ];

   # Add your user to required groups
   users.users.youruser = {
     isNormalUser = true;
     extraGroups = [ "wheel" "video" "i2c" "networkmanager" ];
   };
   ```

2. Or use with flakes:
   ```nix
   {
     inputs = {
       nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
       jetpack-nixos.url = "github:anduril/jetpack-nixos";
     };

     outputs = { nixpkgs, jetpack-nixos, ... }: {
       nixosConfigurations.orin = nixpkgs.lib.nixosSystem {
         system = "aarch64-linux";
         modules = [
           jetpack-nixos.nixosModules.default
           ./examples/orin-agx-camera-ai.nix
           ./configuration.nix
         ];
       };
     };
   }
   ```

3. Build and deploy:
   ```bash
   nixos-rebuild switch --flake .#orin
   ```

See `CAMERA-AI-SETUP.md` for detailed setup instructions and usage examples.

## Creating Your Own Configuration

You can use these examples as a starting point and customize them for your needs:

```nix
{ config, pkgs, ... }:

{
  imports = [
    /path/to/jetpack-nixos/examples/orin-agx-camera-ai.nix
  ];

  # Override or extend the configuration
  hardware.nvidia-jetpack.cameras.imx219 = {
    enableStereo = false;  # Disable if using single camera
  };

  # Add your own packages
  environment.systemPackages = with pkgs; [
    # your packages here
  ];
}
```

## Module Options

### Camera Support

The `hardware.nvidia-jetpack.cameras.imx219` module provides:

- `enable`: Enable IMX219 camera support
- `enableStereo`: Enable stereo camera configuration (for dual cameras)
- `enableCalibrationTools`: Include camera calibration tools

### General Jetpack Options

See the main README.md for all available `hardware.nvidia-jetpack` options.

## Contributing

If you create useful example configurations, consider contributing them back to this directory!

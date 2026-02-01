# Quick Start Guide: Camera and AI Vision on Jetson Orin AGX

This guide gets you up and running with camera-based AI vision projects on your Jetson Orin AGX 64GB in minutes.

## Prerequisites

1. **Hardware:**
   - NVIDIA Jetson Orin AGX 64GB Developer Kit
   - Waveshare Binocular Camera Module (Dual IMX219) OR single IMX219 camera
   - (Optional) ESP32-CAM modules for wireless cameras
   - microSD card or NVMe SSD with NixOS installed

2. **Software:**
   - NixOS installed on your Jetson (see main README.md)
   - Basic familiarity with NixOS configuration

## Step 1: Hardware Setup

### IMX219 Camera Connection

1. **Power off** your Jetson Orin AGX
2. Locate the CSI camera connectors on the board (multiple connectors available)
3. For **single camera**: Connect to CSI-0 (labeled "CAM0")
4. For **stereo camera** (Waveshare Binocular):
   - Connect left camera to CSI-0
   - Connect right camera to CSI-1
5. Ensure the blue stripe on the ribbon cable faces the correct direction (toward heatsink)
6. Power on the device

### ESP32-CAM Setup (Optional)

1. Flash ESP32-CAM with CameraWebServer example from Arduino IDE
2. Configure WiFi credentials in the firmware
3. Power the ESP32-CAM (5V supply required)
4. Note the IP address shown on serial monitor

## Step 2: NixOS Configuration

### Option A: Quick Setup (Recommended)

Add to your `/etc/nixos/configuration.nix`:

```nix
{ config, pkgs, ... }:

{
  imports = [
    # Import the camera-AI example configuration
    (builtins.fetchGit {
      url = "https://github.com/anduril/jetpack-nixos";
      ref = "master";
    } + "/examples/orin-agx-camera-ai.nix")
  ];

  # Add your user to required groups
  users.users.youruser = {
    isNormalUser = true;
    extraGroups = [ "wheel" "video" "i2c" "networkmanager" ];
  };

  # Your other configuration...
}
```

### Option B: Manual Configuration

```nix
{ config, pkgs, ... }:

{
  imports = [
    # Import jetpack-nixos module
    (builtins.fetchTarball {
      url = "https://github.com/anduril/jetpack-nixos/archive/master.tar.gz";
    } + "/modules/default.nix")
  ];

  hardware.nvidia-jetpack = {
    enable = true;
    som = "orin-agx";
    carrierBoard = "devkit";
    
    # Enable camera support
    cameras.imx219 = {
      enable = true;
      enableStereo = true;  # Set to false for single camera
      enableCalibrationTools = true;
    };
  };

  hardware.graphics.enable = true;

  # Add camera and ML packages
  environment.systemPackages = with pkgs; [
    opencv
    python3Packages.opencv4
    v4l-utils
    gstreamer
    python3Packages.numpy
    python3Packages.flask
  ];

  users.users.youruser = {
    extraGroups = [ "video" "i2c" "networkmanager" ];
  };
}
```

### Rebuild System

```bash
sudo nixos-rebuild switch
```

## Step 3: Verify Camera Installation

### Run the camera test script:

```bash
# Download the test script
curl -O https://raw.githubusercontent.com/anduril/jetpack-nixos/master/examples/test-cameras.sh
chmod +x test-cameras.sh

# Run the test
./test-cameras.sh
```

### Manual verification:

```bash
# Check if cameras are detected
ls -l /dev/video*

# List available cameras
v4l2-ctl --list-devices

# Test camera with GStreamer
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \
  'video/x-raw(memory:NVMM),width=1920,height=1080' ! \
  nvvidconv ! nvoverlaysink
```

## Step 4: Run Example Applications

### 1. Stereo Vision (Depth Estimation)

```bash
# Download example scripts
cd ~
git clone https://github.com/anduril/jetpack-nixos
cd jetpack-nixos/examples

# Run stereo vision example
python3 stereo-vision-example.py
```

**Controls:**
- Press 'q' to quit
- Press 's' to save disparity map

### 2. Object Detection

```bash
python3 object-detection-camera.py --camera 0
```

**Controls:**
- Press 'd' to toggle detection overlay
- Press 'e' to show edge detection
- Press 's' to save current frame
- Press 'q' to quit

### 3. Web Camera Streaming

```bash
# Install Flask if not already available
nix-shell -p python3Packages.flask

# Start web server
python3 camera-web-stream.py --port 8080 --camera 0
```

Then open in browser: `http://your-jetson-ip:8080`

### 4. ESP32-CAM Capture

```bash
# Replace with your ESP32-CAM IP
python3 esp32-cam-capture.py 192.168.1.100
```

## Step 5: Develop Your Own Projects

### Python Template for Camera Access

```python
import cv2
import numpy as np

def gstreamer_pipeline(sensor_id=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink"
    )

# Open camera
cap = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0), cv2.CAP_GSTREAMER)

while True:
    ret, frame = cap.read()
    if ret:
        # Your processing here
        cv2.imshow('Camera', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Using CUDA-Accelerated OpenCV

```python
import cv2

# Upload to GPU
gpu_img = cv2.cuda_GpuMat()
gpu_img.upload(frame)

# Process on GPU
gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)

# Download result
result = gpu_gray.download()
```

## Performance Optimization

### Enable Maximum Performance Mode

```bash
# Check current power mode
sudo nvpmodel -q

# Set to maximum performance (mode 0)
sudo nvpmodel -m 0

# Enable maximum clocks
sudo jetson_clocks
```

### Monitor GPU Usage

```bash
# View GPU utilization
nvidia-smi

# Continuous monitoring
watch -n 1 nvidia-smi
```

## Troubleshooting

### Camera Not Detected

```bash
# Check kernel logs
dmesg | grep -i imx219
dmesg | grep -i camera

# Check I2C devices
sudo i2cdetect -y -r 9  # or 10, depending on port

# Restart nvargus daemon
sudo systemctl restart nvargus-daemon
```

### Permission Denied

```bash
# Verify group membership
groups

# Add to video and i2c groups (logout required)
sudo usermod -a -G video,i2c $USER
```

### Poor Performance

1. Enable max performance mode (see above)
2. Reduce camera resolution:
   ```python
   # Use 1280x720 instead of 1920x1080
   capture_width=1280, capture_height=720
   ```
3. Use CUDA acceleration where possible
4. Profile your code to find bottlenecks

## Next Steps

- Explore TensorRT for optimized inference
- Calibrate stereo cameras for better depth estimation
- Train custom models for your specific application
- Check out NVIDIA's Jetson AI courses and examples

## Resources

- [NVIDIA Jetson Developer Zone](https://developer.nvidia.com/embedded/jetson)
- [Full Documentation](./CAMERA-AI-SETUP.md)
- [OpenCV CUDA Documentation](https://docs.opencv.org/master/d1/dfb/intro.html)
- [Jetson AI Fundamentals](https://developer.nvidia.com/embedded/learn/jetson-ai-fundamentals)

## Getting Help

If you encounter issues:

1. Check the detailed documentation in `CAMERA-AI-SETUP.md`
2. Run the diagnostic script: `./test-cameras.sh`
3. Review kernel logs: `dmesg | grep -i camera`
4. Ask on [NVIDIA Developer Forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/)

---

**Happy building!** 🚀

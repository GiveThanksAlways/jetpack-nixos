# Camera and AI Vision Setup for Jetson Orin AGX

This guide explains how to use the Waveshare Binocular IMX219 camera and ESP32-CAM modules on the NVIDIA Jetson Orin AGX 64GB developer kit for computer vision and machine learning projects.

## Hardware Setup

### Waveshare Binocular Camera Module Dual IMX219

The Waveshare Binocular Camera uses two IMX219 8MP sensors for stereo vision applications.

#### Connection:
1. Connect the camera module to one of the CSI ports on the Jetson Orin AGX devkit
2. The devkit has multiple CSI connectors - use CSI connectors 0 and 1 for the dual camera setup
3. Make sure the blue marking on the ribbon cable faces the correct direction (usually towards the heatsink)

#### Verification:
After connecting the camera and booting your system, verify detection:
```bash
# List video devices
ls -l /dev/video*

# Check for camera detection in kernel logs
dmesg | grep -i imx219

# Test camera with v4l2
v4l2-ctl --list-devices

# Capture a test image from camera 0
v4l2-ctl --device=/dev/video0 --set-fmt-video=width=1920,height=1080,pixelformat=RG10 --stream-mmap --stream-count=1 --stream-to=test.raw
```

### ESP32-CAM Modules

The ESP32-CAM is a WiFi-enabled camera module that can stream video over the network.

#### Setup:
1. Flash the ESP32-CAM with appropriate firmware (e.g., CameraWebServer example from Arduino IDE)
2. Configure WiFi credentials in the firmware
3. Power the ESP32-CAM (requires 5V supply, usually via FTDI adapter or USB power)
4. Note the IP address displayed on serial monitor

#### Accessing ESP32-CAM:
```bash
# Once on the same network, access the web interface
curl http://<ESP32-CAM-IP>/

# Stream video using ffmpeg
ffmpeg -i http://<ESP32-CAM-IP>:81/stream -vcodec copy output.mp4

# Display stream with gstreamer
gst-launch-1.0 urisourcebin uri=http://<ESP32-CAM-IP>:81/stream ! decodebin ! autovideosink

# Or use VLC, OpenCV, or other tools to access the MJPEG stream
```

## NixOS Configuration

### Basic Setup

Import the example configuration in your `configuration.nix`:

```nix
{ config, pkgs, ... }:

{
  imports = [
    /path/to/jetpack-nixos/examples/orin-agx-camera-ai.nix
    # ... your other imports
  ];

  # Add your user to required groups
  users.users.youruser = {
    isNormalUser = true;
    extraGroups = [ "wheel" "video" "i2c" "networkmanager" ];
  };
}
```

### Using Flakes

If using flakes, add jetpack-nixos as an input and import the example:

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    jetpack.url = "github:anduril/jetpack-nixos/master";
    jetpack.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, jetpack }: {
    nixosConfigurations.orin = nixpkgs.lib.nixosSystem {
      system = "aarch64-linux";
      modules = [
        jetpack.nixosModules.default
        ./examples/orin-agx-camera-ai.nix
        ./configuration.nix
      ];
    };
  };
}
```

## Testing Cameras

### IMX219 Stereo Camera Testing

#### Using GStreamer (NVIDIA accelerated):
```bash
# Test left camera (usually video0)
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \
  'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
  nvvidconv ! autovideosink

# Test right camera (usually video1)
gst-launch-1.0 nvarguscamerasrc sensor-id=1 ! \
  'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
  nvvidconv ! autovideosink

# Capture from both cameras simultaneously
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \
  'video/x-raw(memory:NVMM),width=1920,height=1080' ! nvvidconv ! \
  'video/x-raw,format=I420' ! queue ! jpegenc ! multifilesink location=left_%d.jpg
```

#### Using Python and OpenCV:
```python
import cv2

# Open camera 0
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### NVIDIA Multimedia Samples

The configuration includes NVIDIA's multimedia samples which provide camera examples:

```bash
# List available samples
ls -la /nix/store/*multimedia-samples*/bin/

# Run camera_jpeg_capture example (if available)
# Check the samples directory for available tools
```

## Computer Vision and Machine Learning

### OpenCV with CUDA

OpenCV is configured with CUDA support for GPU acceleration:

```python
import cv2
import numpy as np

# Check if CUDA is available
print(f"CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}")

# Use GPU for image processing
img = cv2.imread('image.jpg')
gpu_img = cv2.cuda_GpuMat()
gpu_img.upload(img)

# Perform GPU-accelerated operations
gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
result = gpu_gray.download()
```

### Stereo Vision Processing

Example stereo vision code using OpenCV:

```python
import cv2
import numpy as np

# Stereo camera calibration parameters (example - calibrate your own!)
# See: https://docs.opencv.org/master/d9/d0c/group__calib3d.html

left_cap = cv2.VideoCapture(0)
right_cap = cv2.VideoCapture(1)

# Stereo matcher
stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)

while True:
    ret_l, left = left_cap.read()
    ret_r, right = right_cap.read()
    
    if ret_l and ret_r:
        # Convert to grayscale
        gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        
        # Compute disparity map
        disparity = stereo.compute(gray_l, gray_r)
        
        # Normalize for display
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        cv2.imshow('Disparity', disp_vis)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

left_cap.release()
right_cap.release()
cv2.destroyAllWindows()
```

### Using Tinygrad for ML

Tinygrad is a lightweight machine learning framework. Install and use it:

```bash
# Create a Python virtual environment
python -m venv ~/ml-env
source ~/ml-env/bin/activate

# Install tinygrad (with CUDA support)
pip install tinygrad

# Test tinygrad
python3 << 'PYTHON'
from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d
import numpy as np

# Create a simple test tensor
x = Tensor.randn(1, 3, 224, 224)
print(f"Created tensor with shape: {x.shape}")

# Simple convolution
conv = Conv2d(3, 64, kernel_size=3)
y = conv(x)
print(f"Output shape: {y.shape}")

print("Tinygrad is working!")
PYTHON
```

### TensorRT for Inference

NVIDIA TensorRT is available for optimized inference:

```python
# TensorRT is included in the CUDA packages
# Use it for optimized model inference on Jetson
import tensorrt as trt

# Check TensorRT version
print(f"TensorRT version: {trt.__version__}")
```

## Troubleshooting

### Camera Not Detected

```bash
# Check I2C devices
i2cdetect -y -r 9  # or 10, depending on which CSI port

# Check kernel logs
dmesg | grep -i camera
dmesg | grep -i imx219

# Verify nvargus daemon is running
systemctl status nvargus-daemon

# Check camera device nodes
ls -l /dev/video*
```

### Permission Issues

```bash
# Ensure your user is in the video group
groups

# If not, add yourself (requires re-login)
sudo usermod -a -G video,i2c $USER

# Check udev rules
cat /etc/udev/rules.d/*
```

### Performance Optimization

```bash
# Enable maximum performance mode
sudo jetson_clocks

# Check current power mode
sudo nvpmodel -q

# Set to maximum performance mode (mode 0)
sudo nvpmodel -m 0
```

## Additional Resources

- [NVIDIA Jetson Developer Guide](https://docs.nvidia.com/jetson/)
- [Argus Camera API Documentation](https://docs.nvidia.com/jetson/l4t-multimedia/group__LibargusAPI.html)
- [IMX219 Camera Sensor Datasheet](https://www.sony-semicon.com/files/62/pdf/p-11_IMX219PQ.pdf)
- [OpenCV CUDA Documentation](https://docs.opencv.org/master/d1/dfb/intro.html)
- [Tinygrad Documentation](https://github.com/tinygrad/tinygrad)

## Example Projects

### 1. Real-time Stereo Depth Estimation
### 2. Object Detection with ESP32-CAM
### 3. 3D Reconstruction from Stereo Images
### 4. Machine Learning Inference Pipeline

See the examples directory for sample code and projects.

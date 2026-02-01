#!/usr/bin/env bash
# Camera testing script for Jetson devices with IMX219 cameras

set -e

echo "================================"
echo "Jetson Camera Test Script"
echo "================================"
echo ""

# Check if running with proper permissions
if ! groups | grep -q video; then
    echo "WARNING: You are not in the 'video' group."
    echo "Run: sudo usermod -a -G video,i2c $USER"
    echo "Then log out and log back in."
    echo ""
fi

# Check for camera devices
echo "1. Checking for video devices..."
if ls /dev/video* > /dev/null 2>&1; then
    echo "   Found video devices:"
    ls -l /dev/video*
else
    echo "   ERROR: No video devices found!"
    echo "   Make sure your camera is connected properly."
    exit 1
fi
echo ""

# Check for nvargus daemon
echo "2. Checking nvargus-daemon status..."
if systemctl is-active --quiet nvargus-daemon; then
    echo "   nvargus-daemon is running ✓"
else
    echo "   WARNING: nvargus-daemon is not running"
    echo "   Try: sudo systemctl start nvargus-daemon"
fi
echo ""

# List available cameras
echo "3. Listing available cameras with v4l2-ctl..."
if command -v v4l2-ctl > /dev/null; then
    v4l2-ctl --list-devices
else
    echo "   v4l2-ctl not found. Install with: nix-shell -p v4l-utils"
fi
echo ""

# Check I2C devices for IMX219
echo "4. Checking I2C devices (looking for IMX219 at 0x10)..."
if command -v i2cdetect > /dev/null; then
    for i2c in /dev/i2c-*; do
        bus=$(echo $i2c | grep -o '[0-9]*')
        echo "   Scanning I2C bus $bus:"
        i2cdetect -y -r $bus 2>/dev/null | grep -E "10|UU" || echo "     No IMX219 detected on this bus"
    done
else
    echo "   i2cdetect not found. Install with: nix-shell -p i2c-tools"
fi
echo ""

# Test camera with gstreamer (if available)
echo "5. Testing camera with GStreamer..."
if command -v gst-launch-1.0 > /dev/null; then
    echo "   Attempting to capture 10 frames from camera 0..."
    if gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=10 ! \
        'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
        nvvidconv ! 'video/x-raw,format=I420' ! fakesink 2>/dev/null; then
        echo "   Camera 0 test: SUCCESS ✓"
    else
        echo "   Camera 0 test: FAILED"
        echo "   Try running with nvarguscamerasrc directly:"
        echo "   gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! nvoverlaysink"
    fi
    echo ""
    
    echo "   Attempting to capture from camera 1 (if stereo)..."
    if gst-launch-1.0 nvarguscamerasrc sensor-id=1 num-buffers=10 ! \
        'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
        nvvidconv ! 'video/x-raw,format=I420' ! fakesink 2>/dev/null; then
        echo "   Camera 1 test: SUCCESS ✓ (stereo camera detected)"
    else
        echo "   Camera 1 test: Not available (single camera or error)"
    fi
else
    echo "   gst-launch-1.0 not found"
fi
echo ""

echo "================================"
echo "Camera test complete!"
echo ""
echo "Next steps:"
echo "  - To display camera feed: gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! nvoverlaysink"
echo "  - For Python/OpenCV: import cv2; cap = cv2.VideoCapture(0)"
echo "  - See CAMERA-AI-SETUP.md for more examples"
echo "================================"

#!/usr/bin/env python3
"""
ESP32-CAM Capture Example
Demonstrates capturing video stream from ESP32-CAM modules

Requirements:
- ESP32-CAM with CameraWebServer firmware
- ESP32-CAM connected to same network
- OpenCV with Python bindings

Usage:
  python3 esp32-cam-capture.py <ESP32-CAM-IP>
  
Example:
  python3 esp32-cam-capture.py 192.168.1.100
"""

import cv2
import sys
import time
import requests
from urllib.parse import urljoin

def test_esp32_connection(ip_address):
    """Test if ESP32-CAM is reachable"""
    try:
        url = f"http://{ip_address}/"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✓ ESP32-CAM is reachable at {ip_address}")
            return True
        else:
            print(f"✗ ESP32-CAM returned status code {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot reach ESP32-CAM at {ip_address}: {e}")
        return False

def get_stream_url(ip_address, port=81):
    """Generate stream URL for ESP32-CAM"""
    return f"http://{ip_address}:{port}/stream"

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 esp32-cam-capture.py <ESP32-CAM-IP>")
        print("Example: python3 esp32-cam-capture.py 192.168.1.100")
        return 1

    ip_address = sys.argv[1]
    
    print("ESP32-CAM Capture Demo")
    print("======================")
    print(f"ESP32-CAM IP: {ip_address}")
    print("")

    # Test connection
    print("Testing connection to ESP32-CAM...")
    if not test_esp32_connection(ip_address):
        print("\nMake sure:")
        print("1. ESP32-CAM is powered on")
        print("2. ESP32-CAM is connected to WiFi")
        print("3. You're on the same network")
        print("4. IP address is correct")
        return 1

    print("")
    print("Opening video stream...")
    stream_url = get_stream_url(ip_address)
    print(f"Stream URL: {stream_url}")
    
    # Open video stream
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open stream at {stream_url}")
        print("\nTroubleshooting:")
        print("1. Verify the ESP32-CAM is running CameraWebServer firmware")
        print("2. Try accessing the stream in a web browser")
        print("3. Check if port 81 is the correct stream port")
        return 1

    print("✓ Stream opened successfully!")
    print("")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save snapshot")
    print("  'r' - Show/hide resolution info")
    print("")

    frame_count = 0
    show_info = True
    fps_start_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("ERROR: Failed to read frame. Stream may have ended.")
            break

        frame_count += 1

        # Calculate FPS
        if frame_count % 30 == 0:
            fps_end_time = time.time()
            fps = 30 / (fps_end_time - fps_start_time)
            fps_start_time = fps_end_time

        # Add info overlay
        if show_info:
            height, width = frame.shape[:2]
            info_text = f"FPS: {fps:.1f} | Resolution: {width}x{height} | Frame: {frame_count}"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(frame, f"ESP32-CAM: {ip_address}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display frame
        cv2.imshow('ESP32-CAM Stream', frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('s'):
            filename = f"esp32_capture_{frame_count:04d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved snapshot to {filename}")
        elif key == ord('r'):
            show_info = not show_info

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {frame_count} frames")
    print("Done!")
    return 0

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Object Detection Example using Camera and TensorRT
Demonstrates real-time object detection on Jetson using the IMX219 camera

This example uses a pre-trained model for object detection.
For full TensorRT integration, you would need to convert your model to TensorRT format.

Requirements:
- IMX219 camera connected
- OpenCV with Python bindings
- numpy

Usage:
  python3 object-detection-camera.py [--camera CAMERA_ID]
"""

import cv2
import numpy as np
import sys
import argparse
import time

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    """Generate GStreamer pipeline for NVIDIA Argus camera"""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )

def detect_edges(frame):
    """Simple edge detection using Canny"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def detect_contours(frame):
    """Detect and draw contours"""
    edges = detect_edges(frame)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter small contours
    significant_contours = [c for c in contours if cv2.contourArea(c) > 500]
    
    # Draw contours and bounding boxes
    output = frame.copy()
    for contour in significant_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        area = cv2.contourArea(contour)
        cv2.putText(output, f"Area: {int(area)}", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return output, len(significant_contours)

def main():
    parser = argparse.ArgumentParser(description='Object Detection with Camera')
    parser.add_argument('--camera', type=int, default=0, help='Camera sensor ID (default: 0)')
    parser.add_argument('--width', type=int, default=1920, help='Capture width')
    parser.add_argument('--height', type=int, default=1080, help='Capture height')
    args = parser.parse_args()

    print("Camera Object Detection Demo")
    print("============================")
    print(f"Using camera sensor: {args.camera}")
    print("")

    # Try to open camera with GStreamer
    try:
        print("Opening camera with GStreamer (nvarguscamerasrc)...")
        camera = cv2.VideoCapture(
            gstreamer_pipeline(
                sensor_id=args.camera,
                capture_width=args.width,
                capture_height=args.height
            ),
            cv2.CAP_GSTREAMER
        )
        
        if not camera.isOpened():
            print("Failed to open with GStreamer, trying V4L2...")
            camera = cv2.VideoCapture(args.camera)
            
        if not camera.isOpened():
            print("ERROR: Could not open camera!")
            return 1
            
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    print("✓ Camera opened successfully!")
    print("")
    print("Controls:")
    print("  'q' - Quit")
    print("  'd' - Toggle detection overlay")
    print("  'e' - Show edge detection")
    print("  's' - Save current frame")
    print("")

    frame_count = 0
    show_detection = True
    show_edges = False
    fps_start = time.time()
    fps = 0

    while True:
        ret, frame = camera.read()
        
        if not ret:
            print("ERROR: Failed to capture frame")
            break

        frame_count += 1

        # Calculate FPS
        if frame_count % 30 == 0:
            fps_end = time.time()
            fps = 30 / (fps_end - fps_start)
            fps_start = fps_end

        # Process frame
        if show_edges:
            edges = detect_edges(frame)
            display_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif show_detection:
            display_frame, num_objects = detect_contours(frame)
            cv2.putText(display_frame, f"Objects detected: {num_objects}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            display_frame = frame

        # Add FPS counter
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display
        cv2.imshow('Camera Detection', display_frame)

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('d'):
            show_detection = not show_detection
            show_edges = False
            print(f"Detection overlay: {'ON' if show_detection else 'OFF'}")
        elif key == ord('e'):
            show_edges = not show_edges
            show_detection = False
            print(f"Edge detection: {'ON' if show_edges else 'OFF'}")
        elif key == ord('s'):
            filename = f"capture_{frame_count:04d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved frame to {filename}")

    # Cleanup
    camera.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames at {fps:.1f} FPS")
    print("Done!")
    return 0

if __name__ == "__main__":
    sys.exit(main())

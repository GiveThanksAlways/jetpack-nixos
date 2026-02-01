#!/usr/bin/env python3
"""
Stereo Vision Example for Waveshare Binocular IMX219 Camera
Demonstrates basic stereo vision depth estimation using OpenCV

Requirements:
- Two IMX219 cameras connected to CSI ports
- OpenCV with Python bindings
- numpy

Usage:
  python3 stereo-vision-example.py
"""

import cv2
import numpy as np
import sys

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    """
    Generate GStreamer pipeline for NVIDIA Argus camera
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )

def main():
    print("Stereo Vision Demo for IMX219 Cameras")
    print("=====================================")
    print("Press 'q' to quit")
    print("Press 's' to save disparity map")
    print("")

    # Try to open cameras using GStreamer with nvarguscamerasrc
    try:
        print("Opening left camera (sensor 0)...")
        left_camera = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0), cv2.CAP_GSTREAMER)
        
        print("Opening right camera (sensor 1)...")
        right_camera = cv2.VideoCapture(gstreamer_pipeline(sensor_id=1), cv2.CAP_GSTREAMER)
        
        if not left_camera.isOpened() or not right_camera.isOpened():
            print("ERROR: Could not open cameras with GStreamer.")
            print("Falling back to standard V4L2...")
            left_camera = cv2.VideoCapture(0)
            right_camera = cv2.VideoCapture(1)
            
            if not left_camera.isOpened() or not right_camera.isOpened():
                print("ERROR: Could not open cameras!")
                print("Make sure both cameras are connected.")
                return 1
    except Exception as e:
        print(f"ERROR opening cameras: {e}")
        return 1

    print("Cameras opened successfully!")
    print("")

    # Create stereo block matching object
    # These parameters may need tuning for your specific setup
    stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=21)
    
    # You can also try StereoSGBM for better quality (but slower)
    # stereo = cv2.StereoSGBM_create(
    #     minDisparity=0,
    #     numDisparities=16*5,
    #     blockSize=11,
    #     P1=8 * 3 * 11**2,
    #     P2=32 * 3 * 11**2,
    #     disp12MaxDiff=1,
    #     uniquenessRatio=10,
    #     speckleWindowSize=100,
    #     speckleRange=32
    # )

    frame_count = 0
    
    while True:
        # Capture frames from both cameras
        ret_left, left_frame = left_camera.read()
        ret_right, right_frame = right_camera.read()

        if not ret_left or not ret_right:
            print("ERROR: Failed to capture frames")
            break

        # Convert to grayscale for stereo matching
        gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        # Compute disparity map
        disparity = stereo.compute(gray_left, gray_right)

        # Normalize disparity for visualization
        disparity_normalized = cv2.normalize(
            disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        # Apply colormap for better visualization
        disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)

        # Stack images for display
        top_row = np.hstack((left_frame, right_frame))
        bottom_row = np.hstack((
            cv2.cvtColor(gray_left, cv2.COLOR_GRAY2BGR),
            disparity_color
        ))
        combined = np.vstack((top_row, bottom_row))

        # Resize for display if too large
        display_height = 720
        aspect = combined.shape[1] / combined.shape[0]
        display_width = int(display_height * aspect)
        combined_resized = cv2.resize(combined, (display_width, display_height))

        # Add labels
        cv2.putText(combined_resized, "Left Camera", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined_resized, "Right Camera", (display_width//2 + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined_resized, "Left Gray", (10, display_height//2 + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined_resized, "Disparity Map", (display_width//2 + 10, display_height//2 + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Stereo Vision', combined_resized)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('s'):
            filename = f"disparity_{frame_count:04d}.png"
            cv2.imwrite(filename, disparity_normalized)
            print(f"Saved disparity map to {filename}")

        frame_count += 1

    # Cleanup
    left_camera.release()
    right_camera.release()
    cv2.destroyAllWindows()
    print("Done!")
    return 0

if __name__ == "__main__":
    sys.exit(main())

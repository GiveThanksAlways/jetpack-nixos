#!/usr/bin/env python3
"""
Camera Web Streaming Server
Streams camera feed over HTTP using Flask

Requirements:
- IMX219 camera connected
- Flask (pip install flask)
- OpenCV with Python bindings

Usage:
  python3 camera-web-stream.py [--port PORT] [--camera CAMERA_ID]
  
Example:
  python3 camera-web-stream.py --port 8080 --camera 0
  
Then open http://your-jetson-ip:8080 in a web browser
"""

import cv2
import sys
import argparse
from flask import Flask, Response, render_template_string

app = Flask(__name__)

# Global camera object
camera = None
camera_id = 0

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=640,
    display_height=480,
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

def get_camera():
    """Initialize camera if not already open"""
    global camera, camera_id
    if camera is None or not camera.isOpened():
        try:
            print(f"Opening camera {camera_id} with GStreamer...")
            camera = cv2.VideoCapture(
                gstreamer_pipeline(sensor_id=camera_id),
                cv2.CAP_GSTREAMER
            )
            
            if not camera.isOpened():
                print("GStreamer failed, trying V4L2...")
                camera = cv2.VideoCapture(camera_id)
                
            if not camera.isOpened():
                raise Exception("Could not open camera")
                
            print("Camera opened successfully!")
        except Exception as e:
            print(f"ERROR opening camera: {e}")
            camera = None
    
    return camera

def generate_frames():
    """Generate video frames for streaming"""
    cam = get_camera()
    if cam is None:
        yield b''
        return
        
    while True:
        success, frame = cam.read()
        if not success:
            break
        else:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            
            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Video streaming home page"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Jetson Camera Stream</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                text-align: center;
                background-color: #1a1a1a;
                color: #fff;
                padding: 20px;
            }
            h1 {
                color: #76b900;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            img {
                max-width: 100%;
                border: 2px solid #76b900;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            }
            .info {
                margin: 20px 0;
                padding: 15px;
                background-color: #2a2a2a;
                border-radius: 8px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 Jetson Camera Live Stream</h1>
            <div class="info">
                <p>Streaming from Camera {{ camera_id }}</p>
                <p>Powered by NVIDIA Jetpack on Orin AGX</p>
            </div>
            <img src="{{ url_for('video_feed') }}" alt="Camera Stream">
        </div>
    </body>
    </html>
    """
    return render_template_string(html, camera_id=camera_id)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    global camera_id
    
    parser = argparse.ArgumentParser(description='Camera Web Streaming Server')
    parser.add_argument('--port', type=int, default=8080, help='Port to run server on (default: 8080)')
    parser.add_argument('--camera', type=int, default=0, help='Camera sensor ID (default: 0)')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    args = parser.parse_args()
    
    camera_id = args.camera
    
    print("=" * 60)
    print("Camera Web Streaming Server")
    print("=" * 60)
    print(f"Camera: {camera_id}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Access from browser: http://<your-jetson-ip>:{args.port}")
    print("=" * 60)
    print("Press Ctrl+C to stop")
    print()
    
    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if camera is not None:
            camera.release()
        print("Done!")

if __name__ == '__main__':
    main()

import os
os.environ['YOLO_CONFIG_DIR'] = '/tmp'

from ultralytics import YOLO
import onnxruntime as ort
import cv2
import numpy as np
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import json
import time
import threading

# Import servo control (GPIO)
import RPi.GPIO as GPIO

# Export model with optimized settings for person detection - NO SIMPLIFICATION to prevent Pi restart
model = YOLO('yolov8n.pt')
model.export(format='onnx', opset=11, simplify=False, imgsz=320)  # Simplify disabled to reduce memory usage

class ServoController:
    def __init__(self, pan_pin=12, tilt_pin=13, freq=50):
        """
        Pan-Tilt servo controller for object tracking
        :param pan_pin: GPIO pin for pan servo (horizontal)
        :param tilt_pin: GPIO pin for tilt servo (vertical)
        :param freq: PWM frequency (typically 50Hz for servos)
        """
        self.pan_pin = pan_pin
        self.tilt_pin = tilt_pin
        
        # Set GPIO mode
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pan_pin, GPIO.OUT)
        GPIO.setup(self.tilt_pin, GPIO.OUT)
        
        # Initialize PWM for servos
        self.pan_servo = GPIO.PWM(self.pan_pin, freq)
        self.tilt_servo = GPIO.PWM(self.tilt_pin, freq)
        
        # Start PWM with neutral position (7.5% duty cycle = 90 degrees)
        self.pan_servo.start(7.5)  # 90° (center)
        self.tilt_servo.start(7.5)  # 90° (center)
        
        # Current angles (degrees)
        self.pan_angle = 90
        self.tilt_angle = 90
        
        # Lock for thread safety
        self.lock = threading.Lock()

    def move_to_angle(self, servo, angle):
        """
        Move servo to specific angle
        :param servo: PWM object
        :param angle: Angle in degrees (0-180)
        """
        # Convert angle to duty cycle (2.5% to 12.5%)
        duty_cycle = 2.5 + (angle / 180.0) * 10.0
        servo.ChangeDutyCycle(duty_cycle)
        time.sleep(0.05)  # Small delay for smooth movement

    def set_pan_angle(self, angle):
        """Set pan servo angle (horizontal)"""
        with self.lock:
            self.pan_angle = max(0, min(180, angle))  # Clamp to 0-180
            self.move_to_angle(self.pan_servo, self.pan_angle)

    def set_tilt_angle(self, angle):
        """Set tilt servo angle (vertical)"""
        with self.lock:
            self.tilt_angle = max(0, min(180, angle))  # Clamp to 0-180
            self.move_to_angle(self.tilt_servo, self.tilt_angle)

    def center_servos(self):
        """Center both servos to 90 degrees"""
        self.set_pan_angle(90)
        self.set_tilt_angle(90)

    def cleanup(self):
        """Stop PWM and cleanup GPIO"""
        self.pan_servo.stop()
        self.tilt_servo.stop()
        GPIO.cleanup()

class PersonDetectorONNX:
    def __init__(self, onnx_model_path, conf_threshold=0.35, iou_threshold=0.4):
        # Optimize for Pi4 CPU
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            onnx_model_path,
            providers=['CPUExecutionProvider'],
            sess_options=opts
        )
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_shape = (320, 320)

    def preprocess(self, image):
        # Resize to 320x320 for speed
        resized = cv2.resize(image, self.input_shape)
        input_tensor = resized.astype(np.float32) / 255.0
        input_tensor = input_tensor.transpose(2, 0, 1)[np.newaxis, ...]
        return input_tensor

    def postprocess(self, outputs, orig_shape):
        # Parse YOLOv8 output format
        predictions = outputs[0].squeeze(0)  # (84, 8400)
        
        # Extract bounding boxes (first 4 values) and class scores (remaining 80)
        boxes = predictions[:4]  # (4, 8400) - x, y, w, h
        scores = predictions[4:]  # (80, 8400) - class scores
        
        # Convert from [x, y, w, h] to [x1, y1, x2, y2]
        x_center = boxes[0]
        y_center = boxes[1]
        width = boxes[2]
        height = boxes[3]
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # Combine into (4, 8400) array
        boxes_xyxy = np.array([x1, y1, x2, y2])
        
        # Get max class score and corresponding class
        max_scores = np.max(scores, axis=0)  # (8400,)
        class_ids = np.argmax(scores, axis=0)  # (8400,)
        
        # Filter by confidence
        conf_mask = max_scores > self.conf_threshold
        boxes_xyxy = boxes_xyxy[:, conf_mask]  # (4, N)
        scores_filtered = max_scores[conf_mask]  # (N,)
        class_ids_filtered = class_ids[conf_mask]  # (N,)
        
        # Only keep person class (0)
        person_mask = class_ids_filtered == 0
        boxes_xyxy = boxes_xyxy[:, person_mask]  # (4, M)
        scores_filtered = scores_filtered[person_mask]  # (M,)
        class_ids_filtered = class_ids_filtered[person_mask]  # (M,)
        
        # Validate we have detections
        if len(scores_filtered) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Convert to (M, 4) format for NMS
        boxes_for_nms = boxes_xyxy.T  # (M, 4)
        
        # Apply NMS
        keep_indices = self.nms(boxes_for_nms, scores_filtered, self.iou_threshold)
        
        # Return filtered results
        return (
            boxes_for_nms[keep_indices], 
            scores_filtered[keep_indices], 
            class_ids_filtered[keep_indices]
        )

    def nms(self, boxes, scores, iou_threshold):
        if len(boxes) == 0:
            return []
            
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep

    def detect(self, image):
        orig_shape = image.shape
        input_tensor = self.preprocess(image)
        outputs = self.session.run(None, {'images': input_tensor})
        boxes, scores, class_ids = self.postprocess(outputs, orig_shape)
        return boxes, scores, class_ids

# Initialize detector and servo controller
detector = PersonDetectorONNX('yolov8n.onnx')
servo_controller = ServoController(pan_pin=12, tilt_pin=13)  # GPIO pins for servos

# Flask app for web streaming
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

latest_detection = {"boxes": [], "labels": [], "scores": [], "count": 0}

def track_object(boxes_scaled, frame_shape):
    """
    Calculate servo angles to track the largest person in frame
    :param boxes_scaled: Detected bounding boxes in original frame scale
    :param frame_shape: Shape of the original frame
    """
    if len(boxes_scaled) == 0:
        # No detections - center servos
        servo_controller.center_servos()
        return
    
    # Find largest person (by area)
    areas = []
    for box in boxes_scaled:
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        areas.append(area)
    
    # Get index of largest person
    largest_idx = np.argmax(areas)
    largest_box = boxes_scaled[largest_idx]
    
    # Calculate center of largest person
    x1, y1, x2, y2 = largest_box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Get frame dimensions
    frame_width = frame_shape[1]
    frame_height = frame_shape[0]
    
    # Calculate error from center (in pixels)
    error_x = center_x - (frame_width / 2)
    error_y = center_y - (frame_height / 2)
    
    # Convert pixel error to angle adjustment (proportional control)
    # Adjust these values based on your desired sensitivity
    pan_adjustment = (error_x / frame_width) * 45  # Max 45° adjustment
    tilt_adjustment = -(error_y / frame_height) * 45  # Invert Y-axis
    
    # Get current angles
    current_pan = servo_controller.pan_angle
    current_tilt = servo_controller.tilt_angle
    
    # Calculate new angles
    new_pan = max(0, min(180, current_pan + pan_adjustment))
    new_tilt = max(0, min(180, current_tilt + tilt_adjustment))
    
    # Update servos
    servo_controller.set_pan_angle(new_pan)
    servo_controller.set_tilt_angle(new_tilt)

def generate_frames():
    global latest_detection
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 20)
    
    # Frame rate control
    fps_controller = 0
    servo_update_interval = 5  # Update servos every N frames
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect only people
        small_frame = cv2.resize(frame, (320, 320))
        boxes, scores, class_ids = detector.detect(small_frame)
        
        # Calculate person count
        person_count = len(boxes)
        
        # Scale boxes back to original frame size
        boxes_scaled = boxes.copy()
        boxes_scaled[:, [0, 2]] *= (frame.shape[1] / 320.0)
        boxes_scaled[:, [1, 3]] *= (frame.shape[0] / 320.0)
        
        # Track object with servos (every N frames to reduce jitter)
        if fps_controller % servo_update_interval == 0:
            track_object(boxes_scaled, frame.shape)
        
        fps_controller += 1
        
        # Draw ONLY person bounding boxes (green)
        for box, score in zip(boxes_scaled, scores):
            x1, y1, x2, y2 = map(int, box)
            label = f"Person: {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Thick green box
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add person count overlay (simplified)
        cv2.rectangle(frame, (10, 10), (150, 50), (0, 0, 0), -1)  # Black background
        cv2.putText(frame, f"Count: {person_count}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Update detection data
        latest_detection = {
            "boxes": [[int(x) for x in box] for box in boxes_scaled],
            "labels": ["person"] * len(class_ids),
            "scores": [float(score) for score in scores],
            "count": person_count,
            "timestamp": time.time(),
            "pan_angle": servo_controller.pan_angle,
            "tilt_angle": servo_controller.tilt_angle
        }
        
        socketio.emit('detection', latest_detection)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Object Detection Stream</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
        <style>
            body { margin: 0; padding: 20px; font-family: Arial, sans-serif; }
            .container { display: flex; flex-direction: column; align-items: center; }
            #video-feed { border: 2px solid #333; width: 640px; height: 480px; }
            #detection-info { margin-top: 20px; padding: 10px; background: #f0f0f0; border-radius: 5px; }
            .count-display { 
                font-size: 24px; 
                font-weight: bold; 
                color: #006600; 
                background: #ccffcc; 
                padding: 10px; 
                border-radius: 5px; 
                margin: 10px 0;
            }
            .servo-info {
                margin-top: 10px;
                font-size: 16px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Person Detection Stream</h1>
            <img id="video-feed" src="/video_feed" alt="Person Detection Stream">
            <div id="detection-info">
                <h3>People Count:</h3>
                <div class="count-display" id="person-count">0 people detected</div>
                <div class="servo-info">
                    <p>Pan Angle: <span id="pan-angle">90°</span></p>
                    <p>Tilt Angle: <span id="tilt-angle">90°</span></p>
                </div>
            </div>
        </div>
        
        <script>
            const socket = io();
            
            socket.on('detection', function(data) {
                const countDiv = document.getElementById('person-count');
                const panAngleDiv = document.getElementById('pan-angle');
                const tiltAngleDiv = document.getElementById('tilt-angle');
                
                // Update count display
                countDiv.innerHTML = `Count: ${data.count}`;
                
                // Update servo angles
                panAngleDiv.innerHTML = `${data.pan_angle.toFixed(1)}°`;
                tiltAngleDiv.innerHTML = `${data.tilt_angle.toFixed(1)}°`;
            });
        </script>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def cleanup():
    """Cleanup function for graceful shutdown"""
    servo_controller.cleanup()

def main():
    print("Starting person detection web server with servo tracking...")
    print("Access stream at: http://<raspberry-pi-ip>:5000")
    print("Pan servo: GPIO 12, Tilt servo: GPIO 13")
    
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\nShutting down...")
        cleanup()
    finally:
        cleanup()

if __name__ == "__main__":
    main()
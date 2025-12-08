# Optimized person detection with simple count for drone streaming
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

# Export model with optimized settings for person detection - NO SIMPLIFICATION to prevent Pi restart
model = YOLO('yolov8n.pt')
model.export(format='onnx', opset=11, simplify=False, imgsz=320)  # Simplify disabled to reduce memory usage

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

# Initialize detector for person-only detection
detector = PersonDetectorONNX('yolov8n.onnx')

# Flask app for web streaming
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

latest_detection = {"boxes": [], "labels": [], "scores": [], "count": 0}

def generate_frames():
    global latest_detection
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 20)
    
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
            "timestamp": time.time()
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
        <title>Person Detection Stream</title>
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Person Detection Stream</h1>
            <img id="video-feed" src="/video_feed" alt="Person Detection Stream">
            <div id="detection-info">
                <h3>People Count:</h3>
                <div class="count-display" id="person-count">0 people detected</div>
            </div>
        </div>
        
        <script>
            const socket = io();
            
            socket.on('detection', function(data) {
                const countDiv = document.getElementById('person-count');
                
                // Update count display - changed to just "Count: X"
                countDiv.innerHTML = `Count: ${data.count}`;
            });
        </script>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    print("Starting person detection web server with simplified counting...")
    print("Access stream at: http://<raspberry-pi-ip>:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()

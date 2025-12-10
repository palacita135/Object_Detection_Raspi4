# Object_Detection_Raspi4
Object Detection in Raspberry Pi 4 with Yolov8n

# Install dependencies for web streaming
```bash
sudo apt update
sudo apt install python3-pip python3-venv libatlas-base-dev libhdf5-dev libhdf5-serial-dev libhdf5-103
pip3 install ultralytics onnx onnxruntime opencv-python-headless flask flask-socketio pillow numpy
```

**Wiring Diagram:**
```bash
**Raspberry Pi GPIO Layout:**
    [5V] [GPIO 12] [GPIO 13] [GND]
      |      |        |       |
      |   ┌──┴──┐  ┌──┴──┐   |
      |   │Pan  │  │Tilt │   |
      |   │Servo│  │Servo│   |
      |   └─────┘  └─────┘   |
      |______________________|
             Power Rail
```
Wiring Connections:
```bash
    Pan Servo (Horizontal):
        Red wire → 5V (Pin 2 or 4)
        Brown/Black wire → GND (Pin 6, 9, 14, 20, 25, 30, 34, or 39)
        Orange/Yellow wire → GPIO 12 (Pin 32)
    Tilt Servo (Vertical):
        Red wire → 5V (Same rail as Pan)
        Brown/Black wire → GND (Same rail as Pan)
        Orange/Yellow wire → GPIO 13 (Pin 33)
```
Key Features:
```bash
    Proportional Control: Smooth servo movement based on object position
    Jitter Reduction: Servos update every N frames to reduce oscillation
    Angle Clamping: Prevents servos from exceeding 0-180° range
    Thread Safety: Locks prevent race conditions
    Web Interface: Shows real-time servo angles
    Largest Object Tracking: Follows biggest person in frame
```
Hardware Notes:
```bash
    Use 5V power supply capable of 1A+ for servos
    Consider separate power supply for servos to avoid Pi voltage drops
    SG90 or MG90S servos work well for this application
    Adjust pan_adjustment and tilt_adjustment multipliers to change sensitivity
```

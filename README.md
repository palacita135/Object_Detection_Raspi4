# Object_Detection_Raspi4
Object Detection in Raspberry Pi 4 with Yolov8n

# Install dependencies for web streaming
```bash
sudo apt update
sudo apt install python3-pip python3-venv libatlas-base-dev libhdf5-dev libhdf5-serial-dev libhdf5-103
pip3 install ultralytics onnx onnxruntime opencv-python-headless flask flask-socketio pillow numpy
sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly libgstreamer1.0-dev libgstrtspserver-1.0-dev

# Install RPi.GPIO via pip (more reliable on Ubuntu)
pip3 install RPi.GPIO

# Install system dependencies
sudo apt update
sudo apt install python3-dev python3-pip

# For hardware PWM support (optional but recommended)
sudo apt install wiringpi

# Download and build pigpio from source
wget https://github.com/jgarff/rpi_ws281x/archive/master.zip
unzip master.zip
cd rpi_ws281x-master
make
sudo make install

# Or install via git
git clone https://github.com/jgarff/rpi_ws281x.git
cd rpi_ws281x
make
sudo make install

# Install Python wrapper
pip3 install rpi.gpio
```
For your servo application, the standard RPi.GPIO will work fine. The original code should run after installing:
```bash
pip3 install RPi.GPIO
```
Test with:
```bash
python3 -c "import RPi.GPIO as GPIO; print('GPIO OK')"
```
Then run your script:
```bash
python3 tracking_object.py
```
If you encounter permission errors later, add your user to gpio group:
```bash
sudo usermod -a -G gpio $USER
```
And log out/in or reboot.
---

**Wiring Diagram:**
```bash
**Raspberry Pi GPIO Layout:**
    [5V] [GPIO 17] [GPIO 27] [GND]
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
        Red wire → 5V (external power)
        Brown/Black wire → GND (Pin 6, 9, 14, 20, 25, 30, 34, or 39)
        Orange/Yellow wire → GPIO 17 (Pin 11)
    Tilt Servo (Vertical):
        Red wire → 5V (external power)
        Brown/Black wire → GND (Same rail as Pan)
        Orange/Yellow wire → GPIO 27 (Pin 13)
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

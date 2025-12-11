#!/usr/bin/env python3
"""
Simple Servo Test Script using Standard Range (2.5% - 12.5% duty cycle)

This script moves the servos to specific angles using the standard
PWM range without any calibration logic.
"""

import RPi.GPIO as GPIO
import time
import sys

# --- Configuration ---
PAN_PIN = 17  # Changed from 12
TILT_PIN = 27 # Changed from 13
PWM_FREQUENCY = 50  # Hz (standard for servos)

# GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

def set_servo_angle_standard(pin, angle):
    """
    Move a servo to an angle using the standard 2.5%-12.5% duty cycle range.
    This is a more common range for 0-180 degree control.
    """
    GPIO.setup(pin, GPIO.OUT)
    pwm = GPIO.PWM(pin, PWM_FREQUENCY)
    pwm.start(0) # Start with 0% duty cycle

    # Standard formula for 0-180 degrees
    # 0 degrees -> 2.5% duty cycle
    # 180 degrees -> 12.5% duty cycle
    duty_cycle = 2.5 + (angle / 180.0) * 10.0
    
    print(f"Setting {pin} to {angle}° (Duty Cycle: {duty_cycle:.2f}%)")
    pwm.ChangeDutyCycle(duty_cycle)
    
    # Hold the position for a few seconds
    time.sleep(2)
    
    # Stop the PWM signal
    pwm.ChangeDutyCycle(0)
    pwm.stop()

def main():
    print("=== Simple Servo Test (Standard Range) ===")
    print(f"Testing Pan (GPIO {PAN_PIN}) and Tilt (GPIO {TILT_PIN})") # Updated print
    print("Using standard 2.5%-12.5% duty cycle range.")
    
    try:
        # Test Pan Servo (GPIO 17)
        print(f"\nTesting Pan Servo (GPIO {PAN_PIN})...") # Updated print
        for angle in [0, 90, 180, 90]: # Move to 0, center, 180, center
            set_servo_angle_standard(PAN_PIN, angle)
            time.sleep(0.5) # Brief pause between moves

        # Test Tilt Servo (GPIO 27)
        print(f"\nTesting Tilt Servo (GPIO {TILT_PIN})...") # Updated print
        for angle in [0, 90, 180, 90]: # Move to 0, center, 180, center
            set_servo_angle_standard(TILT_PIN, angle)
            time.sleep(0.5) # Brief pause between moves

        print(f"\nTest complete. Servos should be centered.")

    except KeyboardInterrupt:
        print("\nTest interrupted.")
    finally:
        print("\nCleaning up GPIO...")
        GPIO.cleanup()
        print("Done.")

if __name__ == "__main__":
    main()
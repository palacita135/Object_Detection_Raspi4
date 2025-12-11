#!/usr/bin/env python3
"""
Servo Calibration Script for Raspberry Pi 4 (Ubuntu Server 22.04)

This script helps you find the precise minimum and maximum pulse widths
for your servos to achieve accurate 0° and 180° positions.

It uses the RPi.GPIO library and standard PWM.
"""

import RPi.GPIO as GPIO
import time
import sys

# --- Configuration ---
# Change these pins to match your wiring
PAN_PIN = 12
TILT_PIN = 13
PWM_FREQUENCY = 50  # Hz (standard for servos)

# Initial estimates for pulse widths (microseconds)
# Adjust these if your servo behaves wildly at startup
INITIAL_MIN_PULSE_US = 500  # Slightly below typical 0.5ms
INITIAL_MAX_PULSE_US = 2500 # Slightly above typical 2.5ms

# GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False) # Suppress warnings if GPIO pins are already in use

def setup_servo(pin):
    """Initialize a GPIO pin for servo control."""
    GPIO.setup(pin, GPIO.OUT)
    pwm = GPIO.PWM(pin, PWM_FREQUENCY)
    pwm.start(0) # Start with 0% duty cycle (stopped)
    return pwm

def us_to_duty_cycle(pulse_width_us, freq_hz=PWM_FREQUENCY):
    """Convert pulse width in microseconds to duty cycle percentage."""
    period_us = 1_000_000 / freq_hz
    duty_cycle = (pulse_width_us / period_us) * 100
    return duty_cycle

def angle_to_pulse_width(angle, min_pulse_us, max_pulse_us):
    """Map an angle (0-180) to a pulse width."""
    if angle < 0: angle = 0
    if angle > 180: angle = 180
    pulse_width = min_pulse_us + (angle / 180.0) * (max_pulse_us - min_pulse_us)
    return pulse_width

def set_servo_angle(pin, angle, min_pulse_us, max_pulse_us):
    """
    Move the servo connected to the given pin to a specific angle.
    This function creates its own PWM object to ensure correct pin control.
    """
    # Setup PWM for the specific pin
    pwm = GPIO.PWM(pin, PWM_FREQUENCY)
    pwm.start(0) # Start with 0% duty cycle
    
    pulse_width = angle_to_pulse_width(angle, min_pulse_us, max_pulse_us)
    duty_cycle = us_to_duty_cycle(pulse_width)
    pwm.ChangeDutyCycle(duty_cycle)
    print(f"  -> Setting {pin} to {angle}° (Pulse: {pulse_width:.1f}us, Duty: {duty_cycle:.2f}%)")
    
    # Keep the pulse active briefly for the servo to move
    time.sleep(0.1)
    
    # Stop the PWM signal to hold the position
    pwm.ChangeDutyCycle(0)
    pwm.stop() # Clean up the temporary PWM object

def calibrate_servo(pin_number, pin_name, current_min_us, current_max_us):
    """Interactive calibration loop for a single servo."""
    print(f"\n--- Calibrating {pin_name} Servo (GPIO {pin_number}) ---")
    print(f"Current Calibration: Min = {current_min_us}us, Max = {current_max_us}us")
    print("Commands:")
    print("  t <angle>  - Test an angle (e.g., 't 0', 't 90', 't 180')")
    print("  m <us>     - Set new MIN pulse width (e.g., 'm 550')")
    print("  M <us>     - Set new MAX pulse width (e.g., 'M 2450')")
    print("  s          - Sweep from 0 to 180 and back (using current calibration)")
    print("  c          - Confirm and return new values")
    print("  q          - Quit calibration for this servo")
    
    min_pulse_us = current_min_us
    max_pulse_us = current_max_us

    while True:
        try:
            user_input = input(f"\n{pin_name} Command: ").strip()
            if not user_input:
                continue

            # Split the input into command and argument more explicitly
            parts = user_input.split()
            command = parts[0].lower()
            args_str = ' '.join(parts[1:]).strip() if len(parts) > 1 else ""

            if command == 'q':
                print("Calibration cancelled for this servo.")
                return current_min_us, current_max_us # Return original values if cancelled

            elif command == 'c':
                print(f"Calibration confirmed for {pin_name}.")
                print(f"Final values - Min: {min_pulse_us}us, Max: {max_pulse_us}us")
                # Move servo to center position (90 degrees) after confirmation
                print(f"  -> Moving to center position (90°)...")
                set_servo_angle(pin_number, 90, min_pulse_us, max_pulse_us)
                return min_pulse_us, max_pulse_us

            elif command == 't':
                try:
                    angle = int(args_str)
                    if 0 <= angle <= 180:
                        set_servo_angle(pin_number, angle, min_pulse_us, max_pulse_us)
                        print(f"  -> Moved to {angle}°")
                    else:
                        print("  -> Angle must be between 0 and 180.")
                except ValueError:
                    print("  -> Invalid angle. Use 't <angle>'.")

            elif command == 'm':
                try:
                    new_min = int(args_str)
                    if 500 <= new_min <= 2500: # Reasonable limits
                        min_pulse_us = new_min
                        print(f"  -> MIN pulse width updated to {min_pulse_us}us")
                        # Move to 0° to test
                        set_servo_angle(pin_number, 0, min_pulse_us, max_pulse_us)
                        print(f"  -> Testing 0° with new MIN value...")
                    else:
                        print("  -> MIN pulse should be between 500 and 2500 microseconds.")
                except ValueError:
                    print("  -> Invalid MIN value. Use 'm <us>'.")

            elif command == 'M': # Correctly handle capital M for MAX
                try:
                    new_max = int(args_str)
                    if 500 <= new_max <= 2500: # Reasonable limits
                        max_pulse_us = new_max # ASSIGN the new value to the variable
                        print(f"  -> MAX pulse width updated to {max_pulse_us}us")
                        # Move to 180° to test
                        set_servo_angle(pin_number, 180, min_pulse_us, max_pulse_us)
                        print(f"  -> Testing 180° with new MAX value...")
                    else:
                        print("  -> MAX pulse should be between 500 and 2500 microseconds.")
                except ValueError:
                    print("  -> Invalid MAX value. Use 'M <us>'.")

            elif command == 's':
                print(f"  -> Sweeping 0° to 180° and back using Min: {min_pulse_us}us, Max: {max_pulse_us}us")
                # Sweep 0 to 180
                for a in range(0, 181, 5):
                    set_servo_angle(pin_number, a, min_pulse_us, max_pulse_us)
                    time.sleep(0.05)
                # Sweep 180 back to 0
                for a in range(180, -1, -5):
                    set_servo_angle(pin_number, a, min_pulse_us, max_pulse_us)
                    time.sleep(0.05)
                print("  -> Sweep complete.")

            else:
                print("  -> Unknown command. Use 't', 'm', 'M', 's', 'c', or 'q'.")

        except KeyboardInterrupt:
            print("\nCalibration interrupted.")
            return current_min_us, current_max_us

def main():
    print("=== Raspberry Pi Servo Calibration Tool ===")
    print(f"Using GPIO pins: Pan = GPIO {PAN_PIN}, Tilt = GPIO {TILT_PIN}")
    print("This tool helps you find the correct pulse width limits for your servos.")
    print("Ensure your servos are wired correctly before starting.")
    print("Press Ctrl+C at any time to quit safely.\n")

    try:
        print("Starting calibration sequence...")
        print("Recommended: Calibrate the pan servo first, then the tilt servo.")

        # Calibrate Pan Servo
        pan_min_us, pan_max_us = calibrate_servo(PAN_PIN, "Pan", INITIAL_MIN_PULSE_US, INITIAL_MAX_PULSE_US)
        
        # Calibrate Tilt Servo
        tilt_min_us, tilt_max_us = calibrate_servo(TILT_PIN, "Tilt", INITIAL_MIN_PULSE_US, INITIAL_MAX_PULSE_US)

        print("\n=== Calibration Complete ===")
        print(f"Pan Servo (GPIO {PAN_PIN}): Min = {pan_min_us}us, Max = {pan_max_us}us")
        print(f"Tilt Servo (GPIO {TILT_PIN}): Min = {tilt_min_us}us, Max = {tilt_max_us}us")
        print("\nYou can now use these values in your main tracking script.")
        print("For example, update the angle_to_duty_cycle function to use these calibrated values.")
        print("\nExample for Pan servo:")
        print(f"  pulse_width = {pan_min_us} + (angle / 180.0) * ({pan_max_us} - {pan_min_us})")
        print(f"  duty_cycle = (pulse_width / (1_000_000 / {PWM_FREQUENCY})) * 100")
        
        # Move both servos to a safe center position before exiting
        print("\nMoving servos to center position (90°)...")
        set_servo_angle(PAN_PIN, 90, pan_min_us, pan_max_us)
        set_servo_angle(TILT_PIN, 90, tilt_min_us, tilt_max_us)
        time.sleep(0.5) # Brief pause

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("\nCleaning up GPIO...")
        GPIO.cleanup()
        print("Done.")

if __name__ == "__main__":
    main()
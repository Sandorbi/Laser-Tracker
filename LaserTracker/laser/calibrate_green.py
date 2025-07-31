#!/usr/bin/env python3

# GREEN LASER CALIBRATION HELPER
# ------------------------------
# Interactive tool to find optimal HSV values for your specific green laser
# Run this before using ztracker_green.py for best results

import cv2
import numpy as np
import sys
import os

# Default HSV values for 532nm green laser
default_h_min, default_s_min, default_v_min = 45, 100, 100
default_h_max, default_s_max, default_v_max = 75, 255, 255

def nothing(val):
    """Callback function for trackbars (does nothing)"""
    pass

def calibrate_green_laser(camera_id=0):
    """Interactive HSV calibration for green laser detection"""
    
    print("Green Laser HSV Calibration Tool")
    print("=================================")
    print("1. Turn on your green laser and point it at a surface")
    print("2. Use the sliders to adjust HSV values until only the laser dot is white in the mask")
    print("3. Press 'q' to quit and get the optimal values")
    print("4. Press 'r' to reset to default values")
    print("5. Press 's' to save current values")
    print()
    
    # Initialize camera
    if isinstance(camera_id, str):
        cap = None
        print(f"Reading from directory: {camera_id}")
    else:
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return None
        print(f"Using camera {camera_id}")
    
    # Create windows
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('HSV Mask', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
    
    # Create trackbars
    cv2.createTrackbar('H Min', 'Controls', default_h_min, 179, nothing)
    cv2.createTrackbar('S Min', 'Controls', default_s_min, 255, nothing)
    cv2.createTrackbar('V Min', 'Controls', default_v_min, 255, nothing)
    cv2.createTrackbar('H Max', 'Controls', default_h_max, 179, nothing)
    cv2.createTrackbar('S Max', 'Controls', default_s_max, 255, nothing)
    cv2.createTrackbar('V Max', 'Controls', default_v_max, 255, nothing)
    
    # Create a black image for the controls window
    controls_img = np.zeros((200, 400, 3), np.uint8)
    cv2.putText(controls_img, 'Adjust HSV values', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(controls_img, 'Press q to quit', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(controls_img, 'Press r to reset', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(controls_img, 'Press s to save', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow('Controls', controls_img)
    
    frame_count = 0
    best_values = None
    
    while True:
        # Read frame
        if cap:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break
        else:
            # For directory input (not implemented yet)
            frame = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(frame, 'Directory input not implemented', (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get trackbar values
        h_min = cv2.getTrackbarPos('H Min', 'Controls')
        s_min = cv2.getTrackbarPos('S Min', 'Controls')
        v_min = cv2.getTrackbarPos('V Min', 'Controls')
        h_max = cv2.getTrackbarPos('H Max', 'Controls')
        s_max = cv2.getTrackbarPos('S Max', 'Controls')
        v_max = cv2.getTrackbarPos('V Max', 'Controls')
        
        # Create HSV mask
        lower_green = np.array([h_min, s_min, v_min])
        upper_green = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Create result image
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Add text overlay to original frame
        cv2.putText(frame, f'HSV: [{h_min},{s_min},{v_min}] - [{h_max},{s_max},{v_max}]', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Count white pixels in mask (laser detection strength)
        white_pixels = np.sum(mask == 255)
        total_pixels = mask.shape[0] * mask.shape[1]
        detection_ratio = white_pixels / total_pixels
        
        cv2.putText(frame, f'Detection: {detection_ratio:.4f} ({white_pixels} px)', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show images
        cv2.imshow('Original', frame)
        cv2.imshow('HSV Mask', mask)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            best_values = (lower_green, upper_green)
            break
        elif key == ord('r'):
            # Reset to default values
            cv2.setTrackbarPos('H Min', 'Controls', default_h_min)
            cv2.setTrackbarPos('S Min', 'Controls', default_s_min)
            cv2.setTrackbarPos('V Min', 'Controls', default_v_min)
            cv2.setTrackbarPos('H Max', 'Controls', default_h_max)
            cv2.setTrackbarPos('S Max', 'Controls', default_s_max)
            cv2.setTrackbarPos('V Max', 'Controls', default_v_max)
            print("Reset to default values")
        elif key == ord('s'):
            # Save current values
            print(f"Current HSV values:")
            print(f"Lower: [{h_min}, {s_min}, {v_min}]")
            print(f"Upper: [{h_max}, {s_max}, {v_max}]")
            print(f"Detection ratio: {detection_ratio:.4f}")
            
            # Save to file
            with open('green_laser_hsv.txt', 'w') as f:
                f.write(f"# Green Laser HSV Calibration Results\n")
                f.write(f"# Use these values with ztracker_green.py\n")
                f.write(f"# Lower HSV: {h_min} {s_min} {v_min}\n")
                f.write(f"# Upper HSV: {h_max} {s_max} {v_max}\n")
                f.write(f"# Detection ratio: {detection_ratio:.4f}\n")
                f.write(f"\n")
                f.write(f"--hsv-lower {h_min} {s_min} {v_min} --hsv-upper {h_max} {s_max} {v_max}\n")
            print("Saved to green_laser_hsv.txt")
        
        frame_count += 1
    
    # Cleanup
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    
    if best_values:
        lower, upper = best_values
        print("\nCalibration Results:")
        print("===================")
        print(f"Lower HSV: {lower}")
        print(f"Upper HSV: {upper}")
        print("\nTo use these values with the enhanced tracker:")
        print(f"python ztracker_green.py --camera 0 --hsv-lower {lower[0]} {lower[1]} {lower[2]} --hsv-upper {upper[0]} {upper[1]} {upper[2]}")
        return lower, upper
    
    return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calibrate HSV values for green laser detection")
    parser.add_argument("-c", "--camera", type=int, default=0, help="Camera device id (default: 0)")
    
    args = parser.parse_args()
    
    try:
        calibrate_green_laser(args.camera)
    except KeyboardInterrupt:
        print("\nCalibration interrupted by user")
    except Exception as e:
        print(f"Error during calibration: {e}")
        sys.exit(1) 
#!/usr/bin/env python3

# DEBUG LASER DETECTOR - Shows what brightness values are actually being detected
# This helps troubleshoot why the laser might not be detected

import cv2
import numpy as np
import argparse
import time

def debug_laser_detection(camera_id=1, show_all_bright_spots=True):
    """Debug function to see what the detector actually sees"""
    
    print("DEBUG LASER DETECTOR")
    print("====================")
    print("This will show you:")
    print("1. Brightest spot in each frame (even if below threshold)")
    print("2. All bright spots above threshold")
    print("3. Green mask to see if green filtering works")
    print("Press 'q' to quit, 't' to adjust threshold")
    print()
    
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    threshold = 100
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Convert to different color spaces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create green mask
        green_lower = np.array([40, 50, 50])
        green_upper = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Find brightest spot in gray image
        min_val, max_val_gray, min_loc, max_loc_gray = cv2.minMaxLoc(gray)
        
        # Find brightest spot in green-masked image
        masked_gray = cv2.bitwise_and(gray, gray, mask=green_mask)
        min_val_green, max_val_green, min_loc_green, max_loc_green = cv2.minMaxLoc(masked_gray)
        
        # Display results every 10 frames to avoid spam
        if frame_count % 10 == 0:
            print(f"Frame {frame_count}:")
            print(f"  Brightest overall: {max_loc_gray} = {max_val_gray}")
            print(f"  Brightest green:   {max_loc_green} = {max_val_green}")
            print(f"  Threshold:         {threshold}")
            print(f"  Detected:          {'YES' if max_val_green > threshold else 'NO'}")
            print()
        
        # Create display
        display = frame.copy()
        
        # Draw brightest overall spot (blue)
        cv2.circle(display, max_loc_gray, 15, (255, 0, 0), 2)
        cv2.putText(display, f"Overall: {max_val_gray}", 
                   (max_loc_gray[0], max_loc_gray[1]-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw brightest green spot (green)
        if max_val_green > 0:
            cv2.circle(display, max_loc_green, 10, (0, 255, 0), 2)
            cv2.putText(display, f"Green: {max_val_green}", 
                       (max_loc_green[0], max_loc_green[1]+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # If above threshold, draw laser detection (red)
        if max_val_green > threshold:
            cv2.circle(display, max_loc_green, 5, (0, 0, 255), -1)
            cv2.putText(display, "LASER DETECTED!", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Add info
        cv2.putText(display, f"Threshold: {threshold} (press t to change)", 
                   (10, display.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, f"Green max: {max_val_green}, Overall max: {max_val_gray}", 
                   (10, display.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show images
        cv2.imshow('Debug - Camera Feed', display)
        cv2.imshow('Debug - Green Mask', green_mask)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            new_threshold = input(f"Current threshold: {threshold}. Enter new threshold (50-255): ")
            try:
                threshold = max(50, min(255, int(new_threshold)))
                print(f"Threshold set to {threshold}")
            except:
                print("Invalid input, keeping current threshold")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug laser detection")
    parser.add_argument("-c", "--camera", type=int, default=1, help="Camera device id")
    
    args = parser.parse_args()
    debug_laser_detection(args.camera) 
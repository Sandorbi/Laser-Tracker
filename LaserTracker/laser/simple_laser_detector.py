#!/usr/bin/env python3

# SIMPLE LASER DETECTOR for MOVING CAMERA
# ========================================
# Ultra-simple approach: just find the brightest spot in each frame
# No motion detection, no background subtraction - pure brightness detection
# Perfect for moving camera + stationary laser

import cv2
import numpy as np
import argparse
import time

class SimpleLaserDetector:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        
        # Detection parameters
        self.brightness_threshold = 200  # Minimum brightness for laser detection
        self.green_weight = 1.5          # Weight green channel more for green lasers
        self.min_area = 5                # Minimum area in pixels
        self.max_area = 100              # Maximum area in pixels
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
    def open_camera(self):
        """Open camera with optimized settings"""
        print(f"Opening camera {self.camera_id}...")
        
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return False
            
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual dimensions
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera opened: {width}x{height}")
        
        return True
    
    def detect_laser_simple(self, frame):
        """Ultra-simple laser detection: find brightest spot"""
        
        # Convert to HSV for better green detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create green mask (optional - can be disabled for any color laser)
        green_lower = np.array([40, 50, 50])
        green_upper = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Convert to grayscale for brightness detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply green mask to focus on green areas (optional)
        masked_gray = cv2.bitwise_and(gray, gray, mask=green_mask)
        
        # Find brightest pixel
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(masked_gray)
        
        # Check if bright enough
        if max_val > self.brightness_threshold:
            return max_loc, max_val, True
        else:
            return None, max_val, False
    
    def detect_laser_blob(self, frame):
        """Alternative: blob-based detection for more robust results"""
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(gray, self.brightness_threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest bright area
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Check if area is reasonable for a laser dot
            if self.min_area <= area <= self.max_area:
                # Get center of contour
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Get brightness at center
                    brightness = gray[cy, cx]
                    return (cx, cy), brightness, True
        
        return None, 0, False
    
    def run(self, detection_method='simple'):
        """Main detection loop"""
        
        if not self.open_camera():
            return
        
        print("\n" + "="*60)
        print("SIMPLE LASER DETECTOR - Moving Camera Mode")
        print("="*60)
        print("Detection method:", detection_method)
        print(f"Brightness threshold: {self.brightness_threshold}")
        print("Controls:")
        print("  q - quit")
        print("  + - increase brightness threshold")
        print("  - - decrease brightness threshold")
        print("  s - switch detection method")
        print("Point your laser and move the camera around!")
        print("="*60)
        
        cv2.namedWindow('Laser Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Laser Detection', 800, 600)
        
        detection_function = self.detect_laser_simple if detection_method == 'simple' else self.detect_laser_blob
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error reading frame")
                    break
                
                # Detect laser
                start_time = time.time()
                laser_pos, brightness, detected = detection_function(frame)
                detection_time = (time.time() - start_time) * 1000  # ms
                
                # Draw results
                display_frame = frame.copy()
                
                if detected and laser_pos:
                    # Draw laser position
                    cv2.circle(display_frame, laser_pos, 10, (0, 255, 0), 2)
                    cv2.circle(display_frame, laser_pos, 3, (0, 0, 255), -1)
                    
                    # Draw crosshairs
                    h, w = display_frame.shape[:2]
                    cv2.line(display_frame, (laser_pos[0], 0), (laser_pos[0], h), (0, 255, 0), 1)
                    cv2.line(display_frame, (0, laser_pos[1]), (w, laser_pos[1]), (0, 255, 0), 1)
                    
                    # Print coordinates
                    print(f"LASER: {int(laser_pos[0]):3d},{int(laser_pos[1]):3d} | Brightness: {int(brightness):3d} | Time: {detection_time:.1f}ms")
                
                # Add info overlay
                info_text = [
                    f"Method: {detection_method}",
                    f"Threshold: {self.brightness_threshold}",
                    f"Detection: {'YES' if detected else 'NO'}",
                    f"Brightness: {brightness}",
                    f"Time: {detection_time:.1f}ms"
                ]
                
                for i, text in enumerate(info_text):
                    color = (0, 255, 0) if detected else (0, 0, 255)
                    cv2.putText(display_frame, text, (10, 30 + i*25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Calculate and display FPS
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, display_frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Laser Detection', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('+') or key == ord('='):
                    self.brightness_threshold = min(255, self.brightness_threshold + 10)
                    print(f"Brightness threshold: {self.brightness_threshold}")
                elif key == ord('-'):
                    self.brightness_threshold = max(50, self.brightness_threshold - 10)
                    print(f"Brightness threshold: {self.brightness_threshold}")
                elif key == ord('s'):
                    detection_method = 'blob' if detection_method == 'simple' else 'simple'
                    detection_function = self.detect_laser_simple if detection_method == 'simple' else self.detect_laser_blob
                    print(f"Switched to {detection_method} detection")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print(f"\nSession complete. Average FPS: {fps:.1f}")

def main():
    parser = argparse.ArgumentParser(description="Simple laser detector for moving camera")
    parser.add_argument("-c", "--camera", type=int, default=0, help="Camera device id")
    parser.add_argument("-t", "--threshold", type=int, default=200, help="Brightness threshold")
    parser.add_argument("-m", "--method", choices=['simple', 'blob'], default='simple', 
                       help="Detection method")
    
    args = parser.parse_args()
    
    detector = SimpleLaserDetector(args.camera)
    detector.brightness_threshold = args.threshold
    detector.run(args.method)

if __name__ == "__main__":
    main() 
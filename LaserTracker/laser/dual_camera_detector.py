#!/usr/bin/env python3

# DUAL CAMERA LASER DETECTOR
# ===========================
# Uses the EXACT same detection logic as simple_laser_detector.py
# but runs on two cameras simultaneously for better coverage

import cv2
import numpy as np
import argparse
import time
import threading

class DualCameraLaserDetector:
    def __init__(self, camera1_id=0, camera2_id=1):
        self.camera1_id = camera1_id
        self.camera2_id = camera2_id
        self.cap1 = None
        self.cap2 = None
        
        # Detection parameters (same as simple_laser_detector.py)
        self.brightness_threshold = 200
        self.green_weight = 1.5
        self.min_area = 5
        self.max_area = 100
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
        # Thread control
        self.running = True
        
    def open_cameras(self):
        """Open both cameras"""
        print(f"Opening camera {self.camera1_id} and {self.camera2_id}...")
        
        self.cap1 = cv2.VideoCapture(self.camera1_id, cv2.CAP_DSHOW)
        self.cap2 = cv2.VideoCapture(self.camera2_id, cv2.CAP_DSHOW)
        
        if not self.cap1.isOpened():
            print(f"Error: Could not open camera {self.camera1_id}")
            return False
            
        if not self.cap2.isOpened():
            print(f"Error: Could not open camera {self.camera2_id}")
            return False
            
        # Set camera properties for better performance
        for cap in [self.cap1, self.cap2]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual dimensions
        width1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        height2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Camera {self.camera1_id} opened: {width1}x{height1}")
        print(f"Camera {self.camera2_id} opened: {width2}x{height2}")
        
        return True
    
    def detect_laser_simple(self, frame):
        """EXACT same detection logic as simple_laser_detector.py"""
        
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
        """EXACT same blob detection as simple_laser_detector.py"""
        
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
    
    def process_camera(self, cap, camera_name, detection_method='simple'):
        """Process one camera feed"""
        
        detection_function = self.detect_laser_simple if detection_method == 'simple' else self.detect_laser_blob
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print(f"Error reading from {camera_name}")
                break
            
            # Detect laser (same logic as simple_laser_detector.py)
            start_time = time.time()
            laser_pos, brightness, detected = detection_function(frame)
            detection_time = (time.time() - start_time) * 1000  # ms
            
            # Draw results (same as simple_laser_detector.py)
            display_frame = frame.copy()
            
            if detected and laser_pos:
                # Draw laser position
                cv2.circle(display_frame, laser_pos, 10, (0, 255, 0), 2)
                cv2.circle(display_frame, laser_pos, 3, (0, 0, 255), -1)
                
                # Draw crosshairs
                h, w = display_frame.shape[:2]
                cv2.line(display_frame, (laser_pos[0], 0), (laser_pos[0], h), (0, 255, 0), 1)
                cv2.line(display_frame, (0, laser_pos[1]), (w, laser_pos[1]), (0, 255, 0), 1)
                
                # Print coordinates with camera identifier
                print(f"{camera_name} LASER: {int(laser_pos[0]):3d},{int(laser_pos[1]):3d} | Brightness: {int(brightness):3d} | Time: {detection_time:.1f}ms")
            
            # Add info overlay
            info_text = [
                f"{camera_name}",
                f"Method: {detection_method}",
                f"Threshold: {self.brightness_threshold}",
                f"Detection: {'YES' if detected else 'NO'}",
                f"Brightness: {int(brightness)}",
                f"Time: {detection_time:.1f}ms"
            ]
            
            for i, text in enumerate(info_text):
                color = (0, 255, 0) if detected else (0, 0, 255)
                cv2.putText(display_frame, text, (10, 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Calculate and display FPS
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, display_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow(f'{camera_name} Detection', display_frame)
            
            # Small delay to prevent overwhelming
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
    
    def run(self, detection_method='simple'):
        """Main detection loop for dual cameras"""
        
        if not self.open_cameras():
            return
        
        print("\n" + "="*60)
        print("DUAL CAMERA LASER DETECTOR")
        print("="*60)
        print("Detection method:", detection_method)
        print(f"Brightness threshold: {self.brightness_threshold}")
        print("Controls:")
        print("  q - quit (press in any window)")
        print("  + - increase brightness threshold")
        print("  - - decrease brightness threshold")
        print("Point your laser and move around - both cameras will track it!")
        print("="*60)
        
        # Create windows
        cv2.namedWindow(f'Camera {self.camera1_id} Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow(f'Camera {self.camera2_id} Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f'Camera {self.camera1_id} Detection', 640, 480)
        cv2.resizeWindow(f'Camera {self.camera2_id} Detection', 640, 480)
        
        try:
            # Start processing both cameras in parallel
            thread1 = threading.Thread(target=self.process_camera, 
                                      args=(self.cap1, f"CAM{self.camera1_id}", detection_method))
            thread2 = threading.Thread(target=self.process_camera, 
                                      args=(self.cap2, f"CAM{self.camera2_id}", detection_method))
            
            thread1.start()
            thread2.start()
            
            # Main loop to handle global key presses
            while self.running:
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break
                elif key == ord('+') or key == ord('='):
                    self.brightness_threshold = min(255, self.brightness_threshold + 10)
                    print(f"Brightness threshold: {self.brightness_threshold}")
                elif key == ord('-'):
                    self.brightness_threshold = max(50, self.brightness_threshold - 10)
                    print(f"Brightness threshold: {self.brightness_threshold}")
            
            # Wait for threads to finish
            thread1.join()
            thread2.join()
            
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            self.running = False
        
        finally:
            if self.cap1:
                self.cap1.release()
            if self.cap2:
                self.cap2.release()
            cv2.destroyAllWindows()
            
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            print(f"\nSession complete. Average FPS: {fps:.1f}")

def main():
    parser = argparse.ArgumentParser(description="Dual camera laser detector")
    parser.add_argument("-c1", "--camera1", type=int, default=0, help="First camera device id")
    parser.add_argument("-c2", "--camera2", type=int, default=1, help="Second camera device id")
    parser.add_argument("-t", "--threshold", type=int, default=80, help="Brightness threshold")
    parser.add_argument("-m", "--method", choices=['simple', 'blob'], default='simple', 
                       help="Detection method")
    
    args = parser.parse_args()
    
    detector = DualCameraLaserDetector(args.camera1, args.camera2)
    detector.brightness_threshold = args.threshold
    detector.run(args.method)

if __name__ == "__main__":
    main() 
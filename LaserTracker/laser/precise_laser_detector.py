#!/usr/bin/env python3

# PRECISE LASER DETECTOR - Reduces false positives
# =================================================
# Based on simple_laser_detector.py but with stricter filtering
# to reduce detection of non-laser bright objects

import cv2
import numpy as np
import argparse
import time

class PreciseLaserDetector:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        
        # Detection parameters - optimized to reduce false positives
        self.brightness_threshold = 70   # Lower threshold based on debug data
        self.green_strictness = 0.7      # How much it must be green (0.0-1.0)
        self.min_area = 3                # Minimum area in pixels
        self.max_area = 50               # Maximum area in pixels (smaller = more precise)
        
        # Temporal filtering - laser should be consistent over multiple frames
        self.min_detection_frames = 2    # Must detect for N consecutive frames
        self.detection_history = []      # Track recent detections
        self.max_history = 5             # Keep last N detections
        
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
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera opened: {width}x{height}")
        
        return True
    
    def is_green_enough(self, hsv_roi):
        """Check if a region is green enough to be a laser"""
        
        # Define green range for 532nm laser
        green_lower = np.array([45, 80, 50])   # Stricter saturation
        green_upper = np.array([75, 255, 255])
        
        # Create mask for green pixels
        green_mask = cv2.inRange(hsv_roi, green_lower, green_upper)
        
        # Calculate percentage of green pixels
        total_pixels = hsv_roi.shape[0] * hsv_roi.shape[1]
        green_pixels = np.sum(green_mask > 0)
        green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0
        
        return green_ratio >= self.green_strictness
    
    def detect_laser_precise(self, frame):
        """Precise laser detection with false positive reduction"""
        
        # Convert to different color spaces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Apply threshold to find bright areas
        _, thresh = cv2.threshold(gray, self.brightness_threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours (potential laser spots)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_candidate = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size (laser dots are small)
            if not (self.min_area <= area <= self.max_area):
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (laser dots are roughly circular)
            aspect_ratio = w / h if h > 0 else 0
            if not (0.5 <= aspect_ratio <= 2.0):  # Not too elongated
                continue
            
            # Get center point
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
                
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Extract small ROI around the spot for color analysis
            roi_size = max(5, int(max(w, h) * 1.5))
            x1 = max(0, cx - roi_size//2)
            y1 = max(0, cy - roi_size//2)
            x2 = min(frame.shape[1], cx + roi_size//2)
            y2 = min(frame.shape[0], cy + roi_size//2)
            
            hsv_roi = hsv[y1:y2, x1:x2]
            
            # Check if it's green enough
            if not self.is_green_enough(hsv_roi):
                continue
            
            # Get brightness at center
            brightness = gray[cy, cx]
            
            # Calculate score based on brightness, size, and greenness
            size_score = 1.0 - (area / self.max_area)  # Smaller is better
            brightness_score = min(1.0, brightness / 255.0)
            
            total_score = brightness_score * 0.7 + size_score * 0.3
            
            if total_score > best_score:
                best_score = total_score
                best_candidate = ((cx, cy), brightness, area, total_score)
        
        return best_candidate
    
    def add_to_history(self, detection):
        """Add detection to history for temporal filtering"""
        self.detection_history.append(detection)
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
    
    def is_temporally_consistent(self):
        """Check if recent detections are consistent"""
        if len(self.detection_history) < self.min_detection_frames:
            return False
        
        # Check if we have enough recent positive detections
        recent_detections = self.detection_history[-self.min_detection_frames:]
        return all(d is not None for d in recent_detections)
    
    def run(self):
        """Main detection loop with false positive reduction"""
        
        if not self.open_camera():
            return
        
        print("\n" + "="*60)
        print("PRECISE LASER DETECTOR - Reduced False Positives")
        print("="*60)
        print(f"Brightness threshold: {self.brightness_threshold}")
        print(f"Green strictness: {self.green_strictness}")
        print(f"Size range: {self.min_area}-{self.max_area} pixels")
        print("Controls:")
        print("  q - quit")
        print("  + - increase brightness threshold")
        print("  - - decrease brightness threshold")
        print("  g - increase green strictness")
        print("  h - decrease green strictness")
        print("Point your GREEN laser and move the camera around!")
        print("="*60)
        
        cv2.namedWindow('Precise Laser Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Precise Laser Detection', 800, 600)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error reading frame")
                    break
                
                # Detect laser
                start_time = time.time()
                detection = self.detect_laser_precise(frame)
                detection_time = (time.time() - start_time) * 1000  # ms
                
                # Add to temporal history
                self.add_to_history(detection)
                
                # Check if detection is temporally consistent
                consistent = self.is_temporally_consistent()
                
                # Draw results
                display_frame = frame.copy()
                
                if detection and consistent:
                    laser_pos, brightness, area, score = detection
                    
                    # Draw laser position with confidence indicator
                    confidence_color = (0, int(255 * score), int(255 * (1-score)))
                    cv2.circle(display_frame, laser_pos, 12, confidence_color, 2)
                    cv2.circle(display_frame, laser_pos, 3, (0, 0, 255), -1)
                    
                    # Draw crosshairs
                    h, w = display_frame.shape[:2]
                    cv2.line(display_frame, (laser_pos[0], 0), (laser_pos[0], h), confidence_color, 1)
                    cv2.line(display_frame, (0, laser_pos[1]), (w, laser_pos[1]), confidence_color, 1)
                    
                    # Print coordinates with confidence
                    print(f"LASER: {laser_pos[0]:3d},{laser_pos[1]:3d} | Brightness: {int(brightness):3d} | Area: {area:2.0f} | Score: {score:.2f} | Time: {detection_time:.1f}ms")
                
                elif detection and not consistent:
                    # Show candidate but not confirmed
                    laser_pos, brightness, area, score = detection
                    cv2.circle(display_frame, laser_pos, 8, (0, 255, 255), 1)  # Yellow = candidate
                
                # Add info overlay
                status = "CONFIRMED" if (detection and consistent) else "CANDIDATE" if detection else "SEARCHING"
                status_color = (0, 255, 0) if consistent else (0, 255, 255) if detection else (0, 0, 255)
                
                info_text = [
                    f"Status: {status}",
                    f"Brightness threshold: {self.brightness_threshold}",
                    f"Green strictness: {self.green_strictness:.1f}",
                    f"History: {len([d for d in self.detection_history if d])}/{len(self.detection_history)}",
                    f"Time: {detection_time:.1f}ms"
                ]
                
                for i, text in enumerate(info_text):
                    cv2.putText(display_frame, text, (10, 30 + i*25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                
                # Calculate and display FPS
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, display_frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Precise Laser Detection', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('+') or key == ord('='):
                    self.brightness_threshold = min(255, self.brightness_threshold + 5)
                    print(f"Brightness threshold: {self.brightness_threshold}")
                elif key == ord('-'):
                    self.brightness_threshold = max(30, self.brightness_threshold - 5)
                    print(f"Brightness threshold: {self.brightness_threshold}")
                elif key == ord('g'):
                    self.green_strictness = min(1.0, self.green_strictness + 0.1)
                    print(f"Green strictness: {self.green_strictness:.1f}")
                elif key == ord('h'):
                    self.green_strictness = max(0.1, self.green_strictness - 0.1)
                    print(f"Green strictness: {self.green_strictness:.1f}")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print(f"\nSession complete. Average FPS: {fps:.1f}")

def main():
    parser = argparse.ArgumentParser(description="Precise laser detector with false positive reduction")
    parser.add_argument("-c", "--camera", type=int, default=1, help="Camera device id")
    parser.add_argument("-t", "--threshold", type=int, default=70, help="Brightness threshold")
    parser.add_argument("-g", "--green-strictness", type=float, default=0.7, help="Green strictness (0.0-1.0)")
    
    args = parser.parse_args()
    
    detector = PreciseLaserDetector(args.camera)
    detector.brightness_threshold = args.threshold
    detector.green_strictness = args.green_strictness
    detector.run()

if __name__ == "__main__":
    main() 
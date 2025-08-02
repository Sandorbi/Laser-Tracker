#!/usr/bin/env python3

# DUAL CAMERA LASER DETECTOR - OPTIMIZED FOR SPEED
# ==================================================
# Same detection logic but optimized for higher FPS with dual cameras

import cv2
import numpy as np
import argparse
import time
import threading
from queue import Queue
import multiprocessing as mp
import math

class FastDualCameraDetector:
    def __init__(self, camera1_id=0, camera2_id=1):
        self.camera1_id = camera1_id
        self.camera2_id = camera2_id
        
        # Detection parameters
        self.brightness_threshold = 80
        
        # Performance optimizations
        self.skip_frames = 1        # Process every N frames (1 = all frames, 2 = every other frame)
        self.display_scale = 0.7    # Scale down display for better performance
        self.max_queue_size = 2     # Limit frame queue size
        
        # Thread control
        self.running = True
        self.frame_queues = [Queue(maxsize=self.max_queue_size), Queue(maxsize=self.max_queue_size)]
        self.result_queues = [Queue(maxsize=5), Queue(maxsize=5)]
        
        # Performance tracking
        self.fps_counters = [0, 0]
        self.fps_timers = [time.time(), time.time()]
        
        # 3D triangulation parameters (from user's math code)
        self.phih = 0.2639  # horizontal FOV angle in radians
        self.phiv = 0.1985  # vertical FOV angle in radians
        self.Nh = 320.0     # half horizontal resolution
        self.Nv = 240.0     # half vertical resolution
        self.B = 0.0        # baseline distance between cameras (set this to actual distance)
        
        # Current laser coordinates from both cameras
        self.cam1_coords = None  # (x, y) or None
        self.cam2_coords = None  # (x, y) or None
        
        # 3D position results
        self.laser_3d_x = 0.0
        self.laser_3d_y = 0.0
        self.laser_3d_z = 0.0
        self.laser_3d_r = 0.0
        
    def detect_laser_fast(self, frame):
        """Optimized detection - same logic but faster"""
        
        # Convert to HSV (faster than multiple conversions)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Optimized green mask
        green_mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply mask
        masked_gray = cv2.bitwise_and(gray, gray, mask=green_mask)
        
        # Find brightest pixel (fastest method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(masked_gray)
        
        if max_val > self.brightness_threshold:
            return max_loc, max_val, True
        else:
            return None, max_val, False
    
    def calculate_3d_position(self):
        """Calculate 3D position and print both cameras' coordinates at the same moment"""
        
        # Only proceed if both cameras have detected something
        if self.cam1_coords is None or self.cam2_coords is None or self.B == 0.0:
            return
        
        # Print synchronized 2D coordinates
        print(f"[SYNC COORDS] CAM1: {self.cam1_coords} | CAM2: {self.cam2_coords}")
        
        # Convert pixel coordinates to normalized coordinates from center
        n1h = (self.cam1_coords[0] - self.Nh)
        n1v = (-self.cam1_coords[1] + self.Nv)
        n2h = (self.cam2_coords[0] - self.Nh)
        n2v = (-self.cam2_coords[1] + self.Nv)
        
        # User's calculation code
        if (n1h - n2h) != 0:
            ty = self.B / (n1h - n2h)
            nv = (n1v + n2v) / 2
            self.laser_3d_y = ty * n1h
            self.laser_3d_x = ty * self.Nh / math.tan(self.phih)
            self.laser_3d_z = ty * nv * (self.Nh * math.tan(self.phiv)) / (self.Nv * math.tan(self.phih))
            self.laser_3d_r = math.sqrt(self.laser_3d_x**2 + self.laser_3d_y**2 + self.laser_3d_z**2)
            
            print(f"3D POSITION: X={self.laser_3d_x:.2f} Y={self.laser_3d_y:.2f} Z={self.laser_3d_z:.2f} R={self.laser_3d_r:.2f}")

    def camera_capture_thread(self, camera_id, frame_queue):
        """Dedicated thread for camera capture only"""
        
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Optimize camera settings for speed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
        
        print(f"Camera {camera_id} capture thread started")
        
        frame_count = 0
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame_count += 1
                
                # Skip frames for better performance
                if frame_count % (self.skip_frames + 1) != 0:
                    continue
                
                # Non-blocking queue put (drop frames if queue is full)
                try:
                    frame_queue.put(frame, block=False)
                except:
                    pass  # Drop frame if queue is full
                    
        finally:
            cap.release()
            print(f"Camera {camera_id} capture thread stopped")
    
    def detection_thread(self, camera_idx, frame_queue, result_queue):
        """Dedicated thread for laser detection"""
        
        camera_name = f"CAM{self.camera1_id if camera_idx == 0 else self.camera2_id}"
        
        while self.running:
            try:
                # Get frame from queue
                frame = frame_queue.get(timeout=0.1)
                
                # Detect laser
                start_time = time.time()
                laser_pos, brightness, detected = self.detect_laser_fast(frame)
                detection_time = (time.time() - start_time) * 1000
                
                # Update FPS counter
                self.fps_counters[camera_idx] += 1
                
                # Store coordinates for 3D calculation
                if detected and laser_pos:
                    if camera_idx == 0:  # Camera 1
                        self.cam1_coords = laser_pos
                    else:  # Camera 2
                        self.cam2_coords = laser_pos
                    
                    # Calculate 3D position if both cameras have detected laser
                    self.calculate_3d_position()
                    
                    # Print results (only when detected to reduce I/O overhead)
                    # print(f"{camera_name} LASER: {int(laser_pos[0]):3d},{int(laser_pos[1]):3d} | Brightness: {int(brightness):3d} | Time: {detection_time:.1f}ms")
                else:
                    # Clear coordinates when laser not detected
                    if camera_idx == 0:
                        self.cam1_coords = None
                    else:
                        self.cam2_coords = None
                
                # Prepare result for display
                result = {
                    'frame': frame,
                    'laser_pos': laser_pos,
                    'brightness': brightness,
                    'detected': detected,
                    'detection_time': detection_time,
                    'camera_name': camera_name
                }
                
                # Non-blocking result put
                try:
                    result_queue.put(result, block=False)
                except:
                    pass  # Drop result if queue is full
                    
            except:
                continue  # Timeout or other error, continue
    
    def display_thread(self, camera_idx, result_queue):
        """Dedicated thread for display (optional - can be disabled for max speed)"""
        
        camera_id = self.camera1_id if camera_idx == 0 else self.camera2_id
        window_name = f'Fast CAM{camera_id} Detection'
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Calculate display size
        display_width = int(640 * self.display_scale)
        display_height = int(480 * self.display_scale)
        cv2.resizeWindow(window_name, display_width, display_height)
        
        while self.running:
            try:
                result = result_queue.get(timeout=0.1)
                
                frame = result['frame']
                laser_pos = result['laser_pos']
                detected = result['detected']
                brightness = result['brightness']
                detection_time = result['detection_time']
                camera_name = result['camera_name']
                
                # Minimal drawing for speed
                if detected and laser_pos:
                    # Draw only essential elements
                    cv2.circle(frame, laser_pos, 8, (0, 255, 0), 2)
                    cv2.circle(frame, laser_pos, 2, (0, 0, 255), -1)
                    
                    # Draw crosshairs
                    h, w = frame.shape[:2]
                    cv2.line(frame, (laser_pos[0], 0), (laser_pos[0], h), (0, 255, 0), 1)
                    cv2.line(frame, (0, laser_pos[1]), (w, laser_pos[1]), (0, 255, 0), 1)
                
                # Minimal text overlay
                status_color = (0, 255, 0) if detected else (0, 0, 255)
                cv2.putText(frame, f"{camera_name} {'DETECTED' if detected else 'SEARCHING'}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                # Calculate and show FPS
                elapsed = time.time() - self.fps_timers[camera_idx]
                if elapsed >= 1.0:  # Update FPS every second
                    fps = self.fps_counters[camera_idx] / elapsed
                    self.fps_counters[camera_idx] = 0
                    self.fps_timers[camera_idx] = time.time()
                    
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Scale down frame for display if needed
                if self.display_scale != 1.0:
                    frame = cv2.resize(frame, (display_width, display_height))
                
                cv2.imshow(window_name, frame)
                
                # Quick key check
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
                    
            except:
                continue
        
        cv2.destroyWindow(window_name)
    
    def run(self, show_display=True):
        """Main optimized dual camera loop"""
        
        print("\n" + "="*60)
        print("FAST DUAL CAMERA LASER DETECTOR")
        print("="*60)
        print(f"Brightness threshold: {self.brightness_threshold}")
        print(f"Skip frames: {self.skip_frames} (lower = higher FPS)")
        print(f"Display scale: {self.display_scale} (lower = higher FPS)")
        print(f"Display enabled: {show_display}")
        print("Controls:")
        print("  q - quit (in any window or console)")
        print("Optimized for maximum FPS!")
        print("="*60)
        
        try:
            # Start capture threads
            capture_threads = []
            for i, camera_id in enumerate([self.camera1_id, self.camera2_id]):
                thread = threading.Thread(target=self.camera_capture_thread, 
                                         args=(camera_id, self.frame_queues[i]))
                thread.daemon = True
                thread.start()
                capture_threads.append(thread)
            
            # Start detection threads
            detection_threads = []
            for i in range(2):
                thread = threading.Thread(target=self.detection_thread, 
                                         args=(i, self.frame_queues[i], self.result_queues[i]))
                thread.daemon = True
                thread.start()
                detection_threads.append(thread)
            
            # Start display threads (optional)
            display_threads = []
            if show_display:
                for i in range(2):
                    thread = threading.Thread(target=self.display_thread, 
                                             args=(i, self.result_queues[i]))
                    thread.daemon = True
                    thread.start()
                    display_threads.append(thread)
            
            print("All threads started. Detecting laser...")
            
            # Main control loop
            start_time = time.time()
            try:
                while self.running:
                    time.sleep(0.1)  # Low CPU usage for main thread
                    
                    # Check for keyboard interrupt
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
            except KeyboardInterrupt:
                print("\nInterrupted by user")
            
            # Cleanup
            self.running = False
            time.sleep(0.5)  # Give threads time to stop
            
            # Calculate final stats
            total_time = time.time() - start_time
            print(f"\nSession completed in {total_time:.1f} seconds")
            
        finally:
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Fast dual camera laser detector")
    parser.add_argument("-c1", "--camera1", type=int, default=0, help="First camera device id")
    parser.add_argument("-c2", "--camera2", type=int, default=1, help="Second camera device id")
    parser.add_argument("-t", "--threshold", type=int, default=80, help="Brightness threshold")
    parser.add_argument("--skip-frames", type=int, default=1, help="Skip N frames (higher = better FPS)")
    parser.add_argument("--no-display", action="store_true", help="Disable display for maximum FPS")
    parser.add_argument("--scale", type=float, default=0.7, help="Display scale factor")
    parser.add_argument("--baseline", type=float, default=0.0, help="Distance between cameras in cm for 3D calculation")
    parser.add_argument("--fov-h", type=float, default=0.2639, help="Horizontal FOV angle in radians")
    parser.add_argument("--fov-v", type=float, default=0.1985, help="Vertical FOV angle in radians")
    
    args = parser.parse_args()
    
    detector = FastDualCameraDetector(args.camera1, args.camera2)
    detector.brightness_threshold = args.threshold
    detector.skip_frames = args.skip_frames
    detector.display_scale = args.scale
    
    # Set 3D calculation parameters
    detector.B = args.baseline
    detector.phih = args.fov_h
    detector.phiv = args.fov_v
    
    show_display = not args.no_display
    detector.run(show_display)

if __name__ == "__main__":
    main() 
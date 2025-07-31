# ZTRACKER for MOVING CAMERA + STATIONARY LASER
#
# Modified version optimized for 360-degree camera movement
# while detecting a stationary laser pointer
#
# ------------------------------------------
# Moving camera laser detection
# ------------------------------------------

import laser
import argparse
import numpy as np

def modify_for_moving_camera():
    """Modify laser detection parameters for moving camera use case"""
    
    # Monkey patch the Calibration class to use moving camera settings
    original_init = laser.Calibration.__init__
    
    def moving_camera_init(self, cam):
        # Call original init
        original_init(self, cam)
        
        # Override parameters for moving camera detection
        print("Applying MOVING CAMERA optimizations:")
        
        # DISABLE global motion detection (main culprit)
        self.globalMotionThreshold = 0.9  # Allow 90% of image to change
        print(f"  - Global motion threshold: {self.globalMotionThreshold} (was 0.001)")
        
        # Reduce motion threshold dependency, focus on absolute brightness
        self.motionThreshold = 3  # Lower threshold for motion detection
        print(f"  - Motion threshold: {self.motionThreshold} (was 7)")
        
        # Increase laser intensity threshold - focus on very bright objects
        self.laserIntensity = 100  # Higher threshold for laser brightness
        print(f"  - Laser intensity threshold: {self.laserIntensity} (was 40)")
        
        # Reduce background dependency
        self.jitterArea = 0.01  # Allow more movement tolerance
        print(f"  - Jitter area: {self.jitterArea} (was 0.003)")
        
    laser.Calibration.__init__ = moving_camera_init

def modify_background_handling():
    """Modify background handling for moving camera"""
    
    # Monkey patch the Background class to use shorter averaging
    original_background_init = laser.Background.__init__
    
    def fast_background_init(self, length):
        # Use much shorter background for moving camera
        short_length = min(3, length)  # Max 3 frames of background
        original_background_init(self, short_length)
        print(f"  - Background length: {short_length} frames (was {length})")
    
    laser.Background.__init__ = fast_background_init

def modify_oneStepTracker():
    """Modify main detection to work better with moving camera"""
    
    original_oneStepTracker = laser.oneStepTracker
    
    def moving_camera_oneStepTracker(background, img, show, clipBox, snake, cal):
        mask = []
        
        # Always update background quickly for moving camera
        background.add(img)
        
        if background.empty:
            return (mask, 0)
        
        # Use absolute brightness detection instead of just motion
        gray = laser.cv2.cvtColor(img, laser.cv2.COLOR_BGR2GRAY)
        
        # Find brightest points in image (alternative to motion detection)
        brightness_threshold = 200  # Very bright pixels only
        bright_mask = gray > brightness_threshold
        
        if np.sum(bright_mask) > 0:
            # Get coordinates of bright pixels
            y_coords, x_coords = np.where(bright_mask)
            if len(x_coords) > 0:
                # Find brightest pixel
                max_idx = np.argmax(gray[y_coords, x_coords])
                max_x, max_y = x_coords[max_idx], y_coords[max_idx]
                maxVal = int(gray[max_y, max_x])
                maxLoc = (max_x, max_y)
                
                laser.printd(f"Bright pixel detected at {maxLoc} with intensity {maxVal}")
                
                # Simple detection: if it's bright enough and in bounds
                if maxVal > cal.laserIntensity and laser.insideBox(maxLoc, clipBox):
                    # Create a simple candidate
                    best = np.array([max_x, max_y, maxVal])
                    
                    laser.printd("==> Adding bright point to snake.")
                    snake.grow(best[0:2])
                    
                    if laser.gdebug:
                        laser.plotVal(show, best, color=laser.MAXVAL_COLOR)
        
        snake.next_frame()
        return (mask, 0)
    
    laser.oneStepTracker = moving_camera_oneStepTracker

def webcamTracker(camera_id):
    print("Applying moving camera optimizations...")
    modify_for_moving_camera()
    modify_background_handling() 
    modify_oneStepTracker()
    print("Ready for 360-degree camera movement!")
    
    app = laser.sample_app
    laser.main_loop(camera_id, app)

if __name__ == "__main__":
    print("MOVING CAMERA Laser Tracker - Optimized for 360° camera movement")
    print("================================================================")
    print("This version detects STATIONARY laser with MOVING camera")
    print("Point your laser at a surface and move the camera around")
    print()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="set debug mode")
    parser.add_argument("-c", "--camera", type=int, help="set camera device id")
    parser.add_argument("-i", "--input", help="load frames from specified directory")
    
    args = parser.parse_args()
    camera_id = args.camera if args.camera is not None else 0
    
    laser.set_debug(args.debug)
    
    print(f"Starting moving camera tracker on camera {camera_id}...")
    webcamTracker(camera_id)
    print("Bye") 
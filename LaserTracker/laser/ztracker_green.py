# ZTRACKER with GREEN LASER ENHANCEMENTS
#
# Enhanced version optimized for 532nm green lasers
# Reduces false positives by combining motion detection with color filtering
#
# ------------------------------------------
# stand-alone tracker for demo and debugging
# ------------------------------------------
#
# (c) 2018-2022, San Vu Ngoc. University of Rennes 1.
# Enhanced for green lasers 2024

import laser
import laser_green  # Import our green laser enhancements
import argparse
import numpy as np

def webcamTracker(camera_id):
    # Apply green laser enhancements before starting
    laser_green.apply_green_laser_enhancements()
    
    app = laser.sample_app
    laser.main_loop(camera_id, app)

if __name__ == "__main__":
    print("Welcome to Enhanced Ztracker with GREEN LASER FILTERING by San Vu Ngoc, University of Rennes 1.")
    print("This version is optimized for 532nm green lasers and reduces false positives.")
    print("This program comes with ABSOLUTELY NO WARRANTY")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true",
                        help="set debug mode")
    parser.add_argument("-c", "--camera", type=int, help="set camera device id")
    parser.add_argument("-i", "--input", help="load frames from specified directory instead of camera (ignored if camera is specified)")
    parser.add_argument("--green-sensitivity", type=float, default=0.3,
                        help="green color sensitivity (0.1-0.9, default 0.3)")
    parser.add_argument("--hsv-lower", nargs=3, type=int, default=[45, 100, 100],
                        help="HSV lower threshold for green (H S V)")
    parser.add_argument("--hsv-upper", nargs=3, type=int, default=[75, 255, 255],
                        help="HSV upper threshold for green (H S V)")
    
    args = parser.parse_args()
    debug = args.debug
    input_dir = args.input
    camera_id = args.camera
    
    # Update green laser parameters if provided
    if args.hsv_lower != [45, 100, 100]:
        laser_green.GREEN_LASER_HSV_LOWER = np.array(args.hsv_lower)
        print(f"Updated HSV lower threshold: {args.hsv_lower}")
    
    if args.hsv_upper != [75, 255, 255]:
        laser_green.GREEN_LASER_HSV_UPPER = np.array(args.hsv_upper)
        print(f"Updated HSV upper threshold: {args.hsv_upper}")
    
    if camera_id is None:
        if input_dir is None:
            camera_id = -1
        else:
            camera_id = input_dir + "/frame_%07d.jpg"
    
    laser.set_debug(debug)
    
    print("\nGreen Laser Detection Settings:")
    print(f"- HSV Lower: {laser_green.GREEN_LASER_HSV_LOWER}")
    print(f"- HSV Upper: {laser_green.GREEN_LASER_HSV_UPPER}")
    print(f"- Green sensitivity: {args.green_sensitivity}")
    print("\nStarting enhanced tracker...")
    
    webcamTracker(camera_id)
    print("Bye") 
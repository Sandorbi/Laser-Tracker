# -*- coding: utf-8 -*

# Enhanced Laser pointer detector with GREEN LASER COLOR FILTERING
# Based on laser.py but optimized for 532nm green lasers
# ----------------------------------------------------------

import sys
import os

# Add the original laser module to path
sys.path.append(os.path.dirname(__file__))
import laser

import numpy as np
import cv2

# HSV color range for green laser (532nm ±10nm)
# These values are optimized for 532nm green lasers
GREEN_LASER_HSV_LOWER = np.array([45, 100, 100])   # Lower HSV threshold
GREEN_LASER_HSV_UPPER = np.array([75, 255, 255])   # Upper HSV threshold

def filter_green_laser(img, candidates):
    """
    Filter candidates by green color to reduce false positives.
    Only keep candidates that are predominantly green.
    """
    if len(candidates) == 0:
        return candidates
    
    # Convert image to HSV for better color filtering
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create mask for green colors
    green_mask = cv2.inRange(hsv, GREEN_LASER_HSV_LOWER, GREEN_LASER_HSV_UPPER)
    
    filtered_candidates = []
    for candidate in candidates:
        x, y, intensity = int(candidate[0]), int(candidate[1]), candidate[2]
        
        # Check a small region around the candidate point
        radius = 3
        x1, y1 = max(0, x-radius), max(0, y-radius)
        x2, y2 = min(img.shape[1], x+radius+1), min(img.shape[0], y+radius+1)
        
        # Check if the region is predominantly green
        roi_mask = green_mask[y1:y2, x1:x2]
        green_pixels = np.sum(roi_mask > 0)
        total_pixels = roi_mask.size
        
        # Keep candidate if at least 30% of the region is green
        green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0
        
        if green_ratio >= 0.3:  # At least 30% green pixels
            # Boost score for green candidates
            enhanced_candidate = candidate.copy()
            enhanced_candidate[2] = intensity * (1 + green_ratio)  # Boost intensity based on greenness
            filtered_candidates.append(enhanced_candidate)
    
    return np.array(filtered_candidates) if filtered_candidates else np.array([]).reshape(0, 3)

# Monkey-patch the original maxValPos function to include green filtering
original_maxValPos = laser.maxValPos

def enhanced_maxValPos(gray, threshold, maxCandidates, img=None):
    """Enhanced version that includes green color filtering"""
    # Get original candidates
    candidates = original_maxValPos(gray, threshold, maxCandidates)
    
    # Apply green filtering if image is provided
    if img is not None and len(candidates) > 0:
        candidates = filter_green_laser(img, candidates)
    
    return candidates

# Monkey-patch the oneStepTracker to pass the image for color filtering
original_oneStepTracker = laser.oneStepTracker

def enhanced_oneStepTracker(background, img, show, clipBox, snake, cal):
    """Enhanced tracker with green laser filtering"""
    global gdebug
    mask = []

    if background.empty:
        background.add(img)
        return (mask, 0)

    laser.printd("/------------------- new image --------------------\\")
    diff, maxVal, maxLoc = laser.diff_max(background.mean(), img, cal.laserDiameter//2)
    gm = laser.globalMotion(diff, cal.motionThreshold)
    laser.printd("Global Motion  = " + str(gm))
    laser.printd("Max Intensity  = " + str(maxVal))

    if laser.gdebug:
        (x,y) = maxLoc
        laser.plotVal(show, [x,y,maxVal], color=laser.MAXVAL_COLOR)

    newPoint = False

    # We try to detect the pointer only if there is no global motion
    if maxVal > cal.motionThreshold and gm < cal.globalMotionThreshold and laser.insideBox(maxLoc, clipBox):
        candidateThreshold = maxVal-5
        t0 = laser.timeit.default_timer()
        
        # Use enhanced maxValPos with green filtering
        candidates = enhanced_maxValPos(diff, cal.motionThreshold, 10, img)
        
        laser.print_time("Candidates (sec)", t0)
        laser.printd(candidates.shape)
        
        if laser.gdebug:
            for c in candidates:
                laser.printd(c)
                laser.plotVal(show, c)

        if not snake.empty():
            p = snake.predict()
            laser.printd("Predicted = " + str(p))
            if laser.gdebug:
                cv2.circle(show, (p[0], p[1]), 10, laser.PREDICTED_COLOR, 1)
        else:
            p = None

        # Select the best candidate
        if len(candidates) > 0:
            best, score = laser.bestPixel(candidates, snake.active, snake.size, cal.jitterDist, cal.laserIntensity, p)
            laser.printd("SCORE = " + str(score))
            
            if snake.active and p is not None:
                dd = np.linalg.norm(p - best[0:2])
                laser.printd("Deviation from prediction = " + str(dd))
                d = np.linalg.norm(best[0:2] - snake.last())
                laser.printd("Distance = " + str(d))

            if laser.gdebug:
                thr = cal.motionThreshold + (best[2] - cal.motionThreshold)/3
                mask, rect, angle = laser.laserShape(diff, (best[0],best[1]), thr,
                                                   maxRadius=int(cal.laserDiameter), debug=False)

            if score >= 0.5:
                laser.printd("==> Adding new point to snake.")
                newPoint = True

                if snake.size >= 1 and (np.linalg.norm(snake.last() - best[0:2]) > cal.laserDiameter):
                    background.add(img)
                else:
                    laser.printd("Not updating background because pointer did not move enough.")

                snake.grow(best[0:2])
                l = snake.length()
                a = snake.area(show)
                laser.printd("Length         = " + str(l))
                laser.printd("Area           = " + str(a))
            else:
                laser.printd("Score too low: " + str(score))
        else:
            laser.printd("No green candidates found after color filtering")

    snake.next_frame()
    return (mask, maxVal)

def apply_green_laser_enhancements():
    """Apply all green laser enhancements to the laser module"""
    laser.maxValPos = enhanced_maxValPos
    laser.oneStepTracker = enhanced_oneStepTracker
    print("Green laser enhancements applied!")
    print("HSV range: Lower", GREEN_LASER_HSV_LOWER, "Upper", GREEN_LASER_HSV_UPPER)

if __name__ == "__main__":
    print("Green Laser Enhancement Module")
    print("Import this module and call apply_green_laser_enhancements() before using ztracker.py") 
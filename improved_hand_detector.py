import cv2
import numpy as np
import math

class ImprovedHandDetector:
    """
    An improved hand detector using OpenCV with skin color segmentation
    and better finger counting logic.
    """
    def __init__(self):
        # Kernels for morphological operations
        self.kernel_small = np.ones((3, 3), np.uint8)
        self.kernel_medium = np.ones((5, 5), np.uint8)
        self.kernel_large = np.ones((7, 7), np.uint8)
        
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
        
        # Color ranges for skin detection in YCrCb space
        # These ranges can be adjusted based on different skin tones
        self.lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        self.upper_skin = np.array([235, 173, 127], dtype=np.uint8)
        
        # Store detection state
        self.contours = []
        self.hand_contour = None
        self.hand_hull = None
        self.finger_tips = []
        self.finger_count = 0
        self.hand_center = None
        self.roi = None
        
        # History for stabilization
        self.history_length = 5
        self.count_history = [0] * self.history_length
        self.history_index = 0
        
    def reset(self):
        """Reset the detector state"""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
        print("Background model reset. Please wait a moment for calibration...")
    
    def _get_skin_mask(self, frame):
        """Extract skin regions using color thresholding in YCrCb color space"""
        # Convert to YCrCb color space
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        
        # Create a mask for skin color
        skin_mask = cv2.inRange(ycrcb, self.lower_skin, self.upper_skin)
        
        # Apply morphological operations to clean up the mask
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, self.kernel_small)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, self.kernel_medium)
        skin_mask = cv2.dilate(skin_mask, self.kernel_small, iterations=1)
        
        return skin_mask
    
    def _get_largest_contour(self, mask, min_area=5000):
        """Find the largest contour in the mask that's likely to be a hand"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        if not large_contours:
            return None
        
        # Get the largest contour
        return max(large_contours, key=cv2.contourArea)
    
    def _get_defects_and_hull(self, contour):
        """Get convexity defects and hull for the contour"""
        if contour is None or len(contour) < 5:  # Need at least 5 points for meaningful defects
            return None, None
        
        # Get the convex hull
        hull = cv2.convexHull(contour, returnPoints=False)
        
        # Get convexity defects
        try:
            defects = cv2.convexityDefects(contour, hull)
            return defects, hull
        except:
            return None, hull
    
    def _find_fingertips(self, contour, defects, frame_shape):
        """
        Find fingertips using contour, convexity defects, and distance from center.
        Returns list of fingertip coordinates and count.
        """
        if contour is None or defects is None:
            return [], 0
        
        # Find center of hand for reference
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            return [], 0
            
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        self.hand_center = (cx, cy)
        
        # Get bounding box for scale reference
        x, y, w, h = cv2.boundingRect(contour)
        self.roi = (x, y, w, h)
        
        # Filter defects to find finger valleys
        finger_valleys = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            
            # Calculate the triangle sides
            a = math.dist(far, start)
            b = math.dist(far, end)
            c = math.dist(start, end)
            
            # Calculate angle using cosine law
            if a*b == 0:  # Avoid division by zero
                continue
                
            angle = math.degrees(math.acos((a**2 + b**2 - c**2) / (2*a*b)))
            
            # Filter based on angle and distance
            if angle < 90 and d/256.0 > 10:  # d is scaled by 256 in OpenCV
                finger_valleys.append((start, end, far))
        
        # Find potential fingertips
        # Convert to contour format for easier processing
        cnt_array = np.array(contour).reshape((-1, 2))
        
        # Get the hull in point form for fingertip detection
        hull_points = cv2.convexHull(contour, returnPoints=True)
        hull_points = np.array([p[0] for p in hull_points])
        
        # Candidate fingertips are convex points that are:
        # 1. At the top of the hand (y-coordinate smaller than center)
        # 2. Far enough from the center
        # 3. Part of the convex hull
        fingertips = []
        min_dist = math.sqrt(frame_shape[0]**2 + frame_shape[1]**2) * 0.05  # 5% of diagonal
        
        for point in hull_points:
            # Check if point is above center of hand
            if point[1] < cy:  # Y-axis points down
                # Check distance from center
                distance = math.dist(point, (cx, cy))
                if distance > min_dist:
                    fingertips.append(tuple(point))
        
        # Special case: check for thumb which might be to the side
        # Look for points to the left/right that have large difference from center
        thumb_candidates = []
        for point in hull_points:
            # Check if point is significantly to the left or right of center
            if abs(point[0] - cx) > w * 0.25:  # At least 25% of width away
                distance = math.dist(point, (cx, cy))
                if distance > min_dist:
                    thumb_candidates.append((tuple(point), distance))
        
        # Add thumb if found (point with largest distance)
        if thumb_candidates:
            thumb_candidates.sort(key=lambda x: x[1], reverse=True)
            thumb_point = thumb_candidates[0][0]
            if thumb_point not in fingertips:
                fingertips.append(thumb_point)
        
        # Cap at 5 fingers maximum (choose farthest points if more are found)
        if len(fingertips) > 5:
            fingertips_with_dist = [(p, math.dist(p, (cx, cy))) for p in fingertips]
            fingertips_with_dist.sort(key=lambda x: x[1], reverse=True)
            fingertips = [p[0] for p in fingertips_with_dist[:5]]
        
        return fingertips, len(fingertips)
    
    def _stabilize_count(self, count):
        """Stabilize the finger count using a rolling history"""
        # Update history
        self.count_history[self.history_index] = count
        self.history_index = (self.history_index + 1) % self.history_length
        
        # Use mode for stabilization
        counts = {}
        for c in self.count_history:
            counts[c] = counts.get(c, 0) + 1
        
        # Return most frequent count
        return max(counts.items(), key=lambda x: x[1])[0]
    
    def detect_hand(self, frame, draw=True):
        """
        Main method to detect hand and count fingers in a frame.
        
        Args:
            frame: Input BGR frame from camera
            draw: Whether to draw hand features on the frame
            
        Returns:
            Processed frame with visualizations if draw=True
            Updates finger_count property with current count
        """
        # Make a copy of the frame for drawing
        result_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Step 1: Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel_small)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel_medium)
        
        # Step 2: Get skin color mask
        skin_mask = self._get_skin_mask(frame)
        
        # Step 3: Combine the masks
        combined_mask = cv2.bitwise_and(fg_mask, skin_mask)
        combined_mask = cv2.dilate(combined_mask, self.kernel_medium, iterations=1)
        
        # Step 4: Find the largest contour (hand)
        self.hand_contour = self._get_largest_contour(combined_mask)
        
        # Step 5: Process the hand contour if found
        if self.hand_contour is not None:
            # Get defects and hull
            defects, hull = self._get_defects_and_hull(self.hand_contour)
            
            # Find fingertips
            self.finger_tips, raw_count = self._find_fingertips(self.hand_contour, defects, frame.shape)
            
            # Stabilize the count
            self.finger_count = self._stabilize_count(raw_count)
            
            # Draw visualizations if requested
            if draw:
                # Draw hand contour
                cv2.drawContours(result_frame, [self.hand_contour], 0, (0, 255, 0), 2)
                
                # Draw hand center
                if self.hand_center:
                    cv2.circle(result_frame, self.hand_center, 5, (0, 0, 255), -1)
                
                # Draw fingertips
                for fingertip in self.finger_tips:
                    cv2.circle(result_frame, fingertip, 8, (255, 0, 0), -1)
                
                # Draw ROI if available
                if self.roi:
                    x, y, w, h = self.roi
                    cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                
                # Draw finger count text near the hand
                if self.hand_center:
                    cx, cy = self.hand_center
                    cv2.putText(result_frame, f"Count: {self.finger_count}", 
                               (cx - 50, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        else:
            # No hand detected
            self.finger_count = 0
            self.finger_tips = []
            self.hand_center = None
            self.roi = None
        
        return result_frame
    
    def get_finger_count(self):
        """Return the current finger count"""
        return self.finger_count
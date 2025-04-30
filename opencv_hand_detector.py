import cv2
import numpy as np

class OpenCVHandDetector:
    """
    A simpler hand detector using only OpenCV.
    Uses color segmentation to detect hands.
    """
    def __init__(self, threshold_min=0, threshold_max=20):
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.kernel = np.ones((5, 5), np.uint8)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        
        # Store the background to use for detection
        self.background_set = False
        self.background = None
        
        # Store the results of the last processing
        self.contours = []
        self.biggest_contour = None
        self.hand_region = None
        self.fingers = []
        
    def set_background(self, frame):
        """Set current frame as background."""
        self.background = frame.copy()
        self.background_set = True
        print("Background set. Now show your hand.")
        
    def find_hand(self, frame, draw=True):
        """Find hand in the frame using background subtraction and skin color detection."""
        # Make a copy of the frame
        img = frame.copy()
        
        # Apply MOG2 background subtraction
        fg_mask = self.bg_subtractor.apply(img)
        
        # Apply some morphological operations to enhance the mask
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=2)
        
        # Find contours in the foreground mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process the largest contour as the hand
        self.contours = contours
        self.biggest_contour = None
        self.hand_region = None
        
        if contours:
            # Find the biggest contour by area
            self.biggest_contour = max(contours, key=cv2.contourArea)
            
            # Only process if the contour is large enough
            if cv2.contourArea(self.biggest_contour) > 5000:
                # Get the bounding rectangle
                x, y, w, h = cv2.boundingRect(self.biggest_contour)
                self.hand_region = (x, y, w, h)
                
                # Draw the contour and bounding box
                if draw:
                    cv2.drawContours(img, [self.biggest_contour], 0, (0, 255, 0), 2)
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Draw the hull and defects for finger counting
                    hull = cv2.convexHull(self.biggest_contour, returnPoints=False)
                    self.fingers = []
                    
                    try:
                        defects = cv2.convexityDefects(self.biggest_contour, hull)
                        if defects is not None:
                            finger_count = 1  # Start with 1 for the outermost finger
                            for i in range(defects.shape[0]):
                                s, e, f, d = defects[i, 0]
                                start = tuple(self.biggest_contour[s][0])
                                end = tuple(self.biggest_contour[e][0])
                                far = tuple(self.biggest_contour[f][0])
                                
                                # Calculate the triangle sides
                                a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                                b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                                c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                                
                                # Apply cosine law to find angle
                                angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180 / np.pi
                                
                                # If angle is less than 90 degrees, it's likely a finger valley
                                if angle < 90:
                                    finger_count += 1
                                    cv2.circle(img, far, 5, [0, 0, 255], -1)
                                    self.fingers.append(far)
                            
                            # Draw finger count
                            cv2.putText(img, f"Fingers: {min(finger_count, 5)}", (x, y - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    except:
                        pass
        
        return img
    
    def count_fingers(self):
        """
        Count fingers based on convexity defects.
        Returns the estimated number of extended fingers (0-5).
        """
        if not self.biggest_contour is None and len(self.fingers) > 0:
            # Adding 1 because convexity defects give valleys between fingers
            # The number of fingers is typically (number of defects + 1)
            return min(len(self.fingers) + 1, 5)
        return 0
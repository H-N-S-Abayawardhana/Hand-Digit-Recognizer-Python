import cv2
import mediapipe as mp


class HandDetector:
    """
    Class for detecting hands and extracting landmarks using MediaPipe.
    """
    
    def __init__(self, static_mode=False, max_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize the HandDetector.
        
        Args:
            static_mode (bool): If True, detection is done on every image. If False, detection is done only once,
                               and tracking is used for subsequent frames.
            max_hands (int): Maximum number of hands to detect.
            min_detection_confidence (float): Minimum confidence for hand detection.
            min_tracking_confidence (float): Minimum confidence for hand tracking.
        """
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
    
    def find_hands(self, img, draw=True):
        """
        Detect hands in an image and optionally draw landmarks.
        
        Args:
            img (numpy.ndarray): The input image (BGR format from OpenCV).
            draw (bool): If True, landmarks and connections will be drawn on the image.
            
        Returns:
            numpy.ndarray: The image with or without drawn landmarks.
        """
        # Convert BGR image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image
        self.results = self.hands.process(img_rgb)
        
        # If hands are detected
        if self.results.multi_hand_landmarks:
            # For each hand
            for hand_landmarks in self.results.multi_hand_landmarks:
                # Draw landmarks if specified
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS
                    )
        
        return img
    
    def find_positions(self, img, hand_no=0):
        """
        Get the positions of all landmarks for a specific hand.
        
        Args:
            img (numpy.ndarray): The input image.
            hand_no (int): The index of the hand (0 for the first hand detected).
            
        Returns:
            list: A list of landmark positions (id, x, y) or an empty list if no hand is detected.
        """
        landmarks = []
        
        # Check if any hands were detected
        if self.results.multi_hand_landmarks:
            # If the requested hand exists in the detected hands
            if len(self.results.multi_hand_landmarks) > hand_no:
                # Get the landmarks for the specified hand
                hand = self.results.multi_hand_landmarks[hand_no]
                
                h, w, c = img.shape
                # For each landmark in the hand
                for id, lm in enumerate(hand.landmark):
                    # Convert normalized coordinates to pixel coordinates
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append([id, cx, cy])
        
        return landmarks
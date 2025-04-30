import cv2
import time
from opencv_hand_detector import OpenCVHandDetector

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Initialize hand detector
    detector = OpenCVHandDetector()
    
    # Variables for FPS calculation
    prev_time = 0
    current_time = 0
    
    # Initial delay to set background
    print("Keep your hand out of view for background calibration...")
    time.sleep(2)  # Give the camera time to adjust
    
    # Get a frame for background setting
    for _ in range(30):  # Skip a few frames for camera to stabilize
        cap.read()
    
    # Set background from a clean frame
    success, frame = cap.read()
    if success:
        frame = cv2.flip(frame, 1)  # Flip for mirror effect
        detector.set_background(frame)
        time.sleep(1)
    
    print("Ready! Show your hand to the camera.")
    
    # Frame counter for background reset
    frame_count = 0
    
    # Main loop
    while True:
        # Capture frame-by-frame
        success, frame = cap.read()
        if not success:
            print("Error: Failed to grab frame.")
            break
        
        # Flip the frame horizontally for a more intuitive selfie-view
        frame = cv2.flip(frame, 1)
        
        # Process every frame to detect hand
        frame = detector.find_hand(frame)
        
        # Count raised fingers
        finger_count = detector.count_fingers()
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) != 0 else 0
        prev_time = current_time
        
        # Display results on the frame
        # Display count in a circle
        cv2.circle(frame, (50, 50), 40, (255, 0, 0), cv2.FILLED)
        cv2.putText(frame, str(finger_count), (40, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 120), 
                   cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Press 'r' to reset background", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 5), 
                   cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
        
        # Display the resulting frame
        cv2.imshow('Hand Digit Recognizer (OpenCV Only)', frame)
        
        # Process key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Exit on 'q' press
            break
        elif key == ord('r'):
            # Reset background on 'r' press
            print("Resetting background... Keep your hand out of view.")
            time.sleep(1)
            success, frame = cap.read()
            if success:
                frame = cv2.flip(frame, 1)
                detector.set_background(frame)
        
        # Frame counter
        frame_count += 1
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
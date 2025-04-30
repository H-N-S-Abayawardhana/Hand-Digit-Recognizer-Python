import cv2
import time
from hand_detector import HandDetector
from utils import count_fingers, get_finger_status

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 for default camera
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Initialize hand detector
    detector = HandDetector(min_detection_confidence=0.7)
    
    # Variables for FPS calculation
    prev_time = 0
    current_time = 0
    
    # Main loop
    while True:
        # Capture frame-by-frame
        success, frame = cap.read()
        if not success:
            print("Error: Failed to grab frame.")
            break
        
        # Flip the frame horizontally for a more intuitive selfie-view
        frame = cv2.flip(frame, 1)
        
        # Find hands in the frame
        frame = detector.find_hands(frame)
        
        # Find positions of hand landmarks
        landmarks = detector.find_positions(frame)
        
        # Count raised fingers
        finger_count = count_fingers(landmarks)
        
        # Get individual finger status
        finger_status = get_finger_status(landmarks)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) != 0 else 0
        prev_time = current_time
        
        # Display results on the frame
        # Display count in a circle
        cv2.circle(frame, (50, 50), 40, (255, 0, 0), cv2.FILLED)
        cv2.putText(frame, str(finger_count), (40, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display finger status
        y_offset = 150
        for finger, status in finger_status.items():
            color = (0, 255, 0) if status else (0, 0, 255)  # Green if up, red if down
            cv2.putText(frame, f"{finger}: {'Up' if status else 'Down'}", 
                       (10, y_offset), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
            y_offset += 30
        
        # Display FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                   cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        
        # Display the resulting frame
        cv2.imshow('Hand Digit Recognizer', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
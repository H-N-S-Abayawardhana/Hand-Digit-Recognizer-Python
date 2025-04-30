import cv2
import time
from improved_hand_detector import ImprovedHandDetector

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set webcam resolution (can help with processing speed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize improved hand detector
    detector = ImprovedHandDetector()
    
    # Variables for FPS calculation
    prev_time = 0
    current_time = 0
    
    # Initial delay for setup
    print("Setting up background model. Please keep your hand out of view for a few seconds...")
    
    # Warmup period - let the background model initialize
    warmup_frames = 50
    for i in range(warmup_frames):
        success, frame = cap.read()
        if success:
            frame = cv2.flip(frame, 1)  # Flip for mirror effect
            # Process but don't display during warmup
            _ = detector.detect_hand(frame, draw=False)
            
            # Show progress
            progress = int((i / warmup_frames) * 100)
            print(f"Calibrating: {progress}% complete", end='\r')
            
    print("\nCalibration complete! You can now show your hand.")
    print("Press 'r' to reset background model, 'q' to quit.")
    
    # Main loop
    while True:
        # Capture frame-by-frame
        success, frame = cap.read()
        if not success:
            print("Error: Failed to grab frame.")
            break
        
        # Flip the frame horizontally for a more intuitive selfie-view
        frame = cv2.flip(frame, 1)
        
        # Process frame to detect hand and count fingers
        processed_frame = detector.detect_hand(frame)
        
        # Get finger count
        finger_count = detector.get_finger_count()
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) != 0 else 0
        prev_time = current_time
        
        # Display results on the frame
        # Draw finger count in a circle
        cv2.circle(processed_frame, (50, 50), 40, (255, 0, 0), cv2.FILLED)
        cv2.putText(processed_frame, str(finger_count), (40, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display FPS
        cv2.putText(processed_frame, f"FPS: {int(fps)}", (10, 30), 
                   cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        
        # Display instructions
        cv2.putText(processed_frame, "Press 'r' to reset background", 
                   (10, processed_frame.shape[0] - 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
        cv2.putText(processed_frame, "Press 's' to adjust skin thresholds", 
                   (10, processed_frame.shape[0] - 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
        cv2.putText(processed_frame, "Press 'q' to quit", 
                   (10, processed_frame.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
        
        # Display the resulting frame
        cv2.imshow('Improved Hand Digit Recognizer', processed_frame)
        
        # Process key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Exit on 'q' press
            break
        elif key == ord('r'):
            # Reset background model on 'r' press
            detector.reset()
            # Warm up the new model
            print("Resetting background model. Please keep your hand out of view for a few seconds...")
            time.sleep(1)
            for _ in range(20):
                success, frame = cap.read()
                if success:
                    frame = cv2.flip(frame, 1)
                    _ = detector.detect_hand(frame, draw=False)
            print("Reset complete!")
        elif key == ord('s'):
            # Placeholder for skin threshold adjustment if needed
            # In a full application, this would open a trackbar window
            print("Skin threshold adjustment not implemented in this demo.")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
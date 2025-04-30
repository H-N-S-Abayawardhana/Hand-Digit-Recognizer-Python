# Hand Digit Recognizer

A Python application that detects hands from your webcam feed, counts how many fingers are raised, and displays that number in real-time.

## Features

- Real-time hand detection using MediaPipe
- Accurate finger counting (0-5)
- Individual finger status display (Up/Down)
- Frame rate (FPS) display
- Flipped camera view for more intuitive interaction

## Requirements

- Python 3.6+
- OpenCV
- MediaPipe

## Installation

1. Clone this repository or download the files:
   ```
   git clone https://github.com/yourusername/hand-digit-recognizer.git
   cd hand-digit-recognizer
   ```

2. Set up a virtual environment (recommended):
   ```
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main application:
```
python main.py
```

- Position your hand in front of the webcam
- Raise or lower fingers to see the count change
- Press 'q' to quit the application

## How It Works

1. **Hand Detection**: The `HandDetector` class uses MediaPipe's Hand module to detect hands and their landmarks in each video frame.

2. **Landmark Extraction**: For each detected hand, we extract the 21 landmarks representing different parts of the hand (wrist, finger joints, fingertips).

3. **Finger Counting Logic**: 
   - For the thumb, we check if the tip's x-coordinate is to the right of its joint
   - For other fingers, we check if the fingertip's y-coordinate is above its middle joint
   - We count how many fingers meet these criteria

4. **Visualization**: The results are displayed on the video feed, including:
   - Total finger count in a blue circle
   - Status of each finger (Up/Down)
   - Current FPS (frames per second)

## Project Structure

- `main.py`: Main application entry point
- `hand_detector.py`: Hand detection module using MediaPipe
- `utils.py`: Utility functions for finger counting
- `requirements.txt`: List of dependencies

## Future Improvements

- Add support for both hands simultaneously
- Implement gesture recognition
- Add sound effects for different finger counts
- Save screenshots
- Improve thumb detection logic based on hand orientation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*Note: This application requires a webcam and adequate lighting for optimal performance.*
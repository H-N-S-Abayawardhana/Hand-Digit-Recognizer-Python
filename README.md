# Hand Digit Recognizer (OpenCV-Only Version)

A Python application that uses OpenCV to detect hands from your webcam feed, counts extended fingers, and displays the count in real-time. This version does not require MediaPipe and is compatible with newer Python versions like Python 3.13.

## Features

- Real-time hand detection using OpenCV background subtraction and contour analysis
- Finger counting based on convexity defects
- Frame rate (FPS) display
- Background calibration for improved detection
- Flipped camera view for more intuitive interaction

## Requirements

- Python 3.6+ (including Python 3.13)
- OpenCV
- NumPy

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
   pip install -r opencv_requirements.txt
   ```

## Usage

Run the OpenCV-only version:
```
python opencv_main.py
```

1. When the application starts, keep your hand out of view for background calibration
2. Once calibration is complete, show your hand to the camera
3. The number of detected fingers will appear in the blue circle
4. Press 'r' to reset the background if detection becomes unreliable
5. Press 'q' to quit the application

## How It Works

1. **Background Subtraction**: The application captures a clean background frame and uses it to detect foreground objects (your hand).

2. **Contour Detection**: OpenCV finds contours in the foreground mask, with the largest contour assumed to be your hand.

3. **Convexity Defects**: The program analyses the convexity defects (valleys between fingers) to estimate how many fingers are extended.

4. **Visualization**: The results are displayed on the video feed, including:
   - Total finger count in a blue circle
   - Current FPS (frames per second)
   - Instructions for reset and exit

## Tips for Best Results

1. Ensure good lighting in your environment
2. Use a plain background for better hand segmentation
3. Position your hand with fingers spread apart
4. Reset the background (press 'r') if you change positions or lighting conditions
5. Keep your hand clearly visible and within the camera frame

## Limitations

This OpenCV-only approach is less accurate than the MediaPipe version but works with Python 3.13. The finger counting may be less reliable in challenging lighting conditions or with complex backgrounds.

## Project Structure

- `opencv_main.py`: Main application entry point
- `opencv_hand_detector.py`: Hand detection using OpenCV
- `opencv_requirements.txt`: List of dependencies

## License

This project is licensed under the MIT License.

---

*Note: For better accuracy, consider using the MediaPipe version with Python 3.10.*
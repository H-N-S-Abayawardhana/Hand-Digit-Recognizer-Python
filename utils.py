def count_fingers(landmarks):
    """
    Count the number of raised fingers based on hand landmarks.
    
    MediaPipe hand landmark indexes:
    - Wrist: 0
    - Thumb: 1 (base) to 4 (tip)
    - Index: 5 (base) to 8 (tip)
    - Middle: 9 (base) to 12 (tip)
    - Ring: 13 (base) to 16 (tip)
    - Pinky: 17 (base) to 20 (tip)
    
    Args:
        landmarks (list): List of landmark positions [id, x, y].
        
    Returns:
        int: Number of raised fingers (0-5).
    """
    # If no hand landmarks detected, return 0
    if not landmarks:
        return 0
    
    # Convert landmarks to a dictionary for easier access
    landmarks_dict = {landmark[0]: (landmark[1], landmark[2]) for landmark in landmarks}
    
    fingers = 0
    
    # Thumb (special case) - compare x-coordinates
    # Thumb is considered up if the tip (4) is to the right of the joint (3) for right hand
    # For a more robust solution, you would need to consider hand chirality (left/right hand)
    if landmarks_dict[4][0] > landmarks_dict[3][0]:
        fingers += 1
    
    # For fingers 2-5, check if the fingertip y-coordinate is above the middle joint
    # Lower y-coordinate value means higher position on the screen
    # Index finger (tip: 8, middle joint: 6)
    if landmarks_dict[8][1] < landmarks_dict[6][1]:
        fingers += 1
    
    # Middle finger (tip: 12, middle joint: 10)
    if landmarks_dict[12][1] < landmarks_dict[10][1]:
        fingers += 1
    
    # Ring finger (tip: 16, middle joint: 14)
    if landmarks_dict[16][1] < landmarks_dict[14][1]:
        fingers += 1
    
    # Pinky finger (tip: 20, middle joint: 18)
    if landmarks_dict[20][1] < landmarks_dict[18][1]:
        fingers += 1
    
    return fingers


def get_finger_status(landmarks):
    """
    Get the status of each finger (up or down).
    
    Args:
        landmarks (list): List of landmark positions [id, x, y].
        
    Returns:
        dict: Dictionary with status of each finger (True for up, False for down).
    """
    if not landmarks:
        return {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False}
    
    # Convert landmarks to a dictionary for easier access
    landmarks_dict = {landmark[0]: (landmark[1], landmark[2]) for landmark in landmarks}
    
    finger_status = {
        "thumb": landmarks_dict[4][0] > landmarks_dict[3][0],  # Thumb is up if tip is to the right of joint
        "index": landmarks_dict[8][1] < landmarks_dict[6][1],  # Other fingers are up if tip is above middle joint
        "middle": landmarks_dict[12][1] < landmarks_dict[10][1],
        "ring": landmarks_dict[16][1] < landmarks_dict[14][1],
        "pinky": landmarks_dict[20][1] < landmarks_dict[18][1]
    }
    
    return finger_status
import cv2  # OpenCV for image and video processing
import os  # For directory and file management
import numpy as np  # For numerical operations
from skimage.metrics import structural_similarity as ssim  # For similarity metrics (if needed in future)
import time  # To manage timing and delays

# Function to apply filters to enhance image quality
def apply_filters(frame):
    """
    Applies a series of filters to enhance the image quality.
    
    Steps:
    1. Applies Gaussian blur to smooth the image.
    2. Sharpens the image using a weighted combination of the blurred versions.
    3. Converts the image to LAB color space for better contrast adjustment.
    4. Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) on the 'L' channel to improve contrast.
    5. Converts the LAB image back to BGR format for display.
    
    Parameters:
        frame: The input image frame from the video feed.
        
    Returns:
        enhanced: The enhanced image after applying filters.
    """
    # Apply a Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    gaussian = cv2.GaussianBlur(blurred, (5, 5), 1.0)
    
    # Sharpen the image
    sharpened = cv2.addWeighted(blurred, 1.5, gaussian, -0.5, 0)
    
    # Convert to LAB color space for contrast adjustment
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE on the 'L' channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge channels back and convert to BGR
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced

# Function to initialize video capture with support for various backends
def get_video_capture():
    """
    Attempts to open the camera using different backends to ensure compatibility
    across different systems and operating environments.
    
    Returns:
        cam: The opened video capture object if successful, or None if failed.
    """
    # List of backends to try
    backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2]
    for backend in backends:
        cam = cv2.VideoCapture(0, backend)
        if cam.isOpened():  # Check if the camera opened successfully
            print(f"Using backend: {backend}")
            return cam
    print("Failed to open camera with all backends.")
    return None

# Function to extract the Region of Interest (ROI) of the hand from the frame
def extract_hand_roi(frame):
    """
    Extracts the ROI of the hand from the given frame using skin color segmentation.
    
    Steps:
    1. Convert the frame to HSV color space for better skin color segmentation.
    2. Create a mask for skin color using predefined HSV ranges.
    3. Apply Gaussian blur to the mask to reduce noise.
    4. Find the contours in the mask and select the largest one.
    5. Extract the bounding box around the largest contour and apply padding.
    
    Parameters:
        frame: The input image frame from the video feed.
        
    Returns:
        hand_roi: The cropped image of the hand ROI, or None if no hand is detected.
        bounding_box: The coordinates of the bounding box, or None if no hand is detected.
    """
    # Convert to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define HSV range for skin color
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)
    
    # Apply Gaussian blur to smooth the mask
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Select the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Expand the bounding box to include the whole hand
        padding = 30
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)
        
        # Crop the hand ROI
        hand_roi = frame[y:y + h, x:x + w]
        return hand_roi, (x, y, x + w, y + h)
    
    return None, None

# Main function to capture and save hand signs
def capture_signs():
    """
    Captures frames from the video feed, extracts the hand ROI, and saves it
    if a stable sign is detected.
    
    Steps:
    1. Set up the main folder to store captured signs.
    2. Open the video capture with backend support.
    3. Loop to continuously capture frames and detect hand signs.
    4. Extract and enhance the hand ROI, then save it in a structured folder.
    5. Implement a cooldown period to avoid multiple captures of the same sign.
    6. Provide a way to stop the capture using the 'q' key.
    """
    main_folder = "captured_signs"
    os.makedirs(main_folder, exist_ok=True)
    
    cam = get_video_capture()
    if cam is None:
        raise Exception("No compatible camera found.")
    
    stable_duration = 1.0  # Duration (in seconds) to consider a sign stable
    cooldown_period = 3.0  # Time (in seconds) before detecting a new sign
    
    captured_signs = 0  # Counter for the number of signs captured
    
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        
        # Extract the hand ROI from the frame
        hand_roi, bounding_box = extract_hand_roi(frame)
        
        if hand_roi is not None:
            # Increment the counter for captured signs
            captured_signs += 1
            sign_folder = os.path.join(main_folder, f"Sign_{captured_signs}")
            os.makedirs(sign_folder, exist_ok=True)
            
            # Apply filters to enhance the hand ROI and save the image
            filtered_hand_roi = apply_filters(hand_roi)
            roi_path = os.path.join(sign_folder, "hand_roi.jpg")
            cv2.imwrite(roi_path, filtered_hand_roi)
            print(f"Captured and saved hand ROI for Sign_{captured_signs}")
            
            # Wait for the cooldown period before detecting the next sign
            time.sleep(cooldown_period)
        
        # Display the video feed in a window
        cv2.imshow("Video Feed", frame)
        
        # Stop the capture process when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Capture stopped by user.")
            break
    
    # Release the video capture and close all windows
    cam.release()
    cv2.destroyAllWindows()
    print("All sign captures complete.")

# Start the sign capture process
capture_signs()

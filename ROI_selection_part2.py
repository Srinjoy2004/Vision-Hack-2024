import cv2 
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
import time

# Apply filters to enhance image quality
def apply_filters(frame):
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    gaussian = cv2.GaussianBlur(blurred, (5, 5), 1.0)
    sharpened = cv2.addWeighted(blurred, 1.5, gaussian, -0.5, 0)
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced

# Function to get video capture with backend support
def get_video_capture():
    backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2]
    for backend in backends:
        cam = cv2.VideoCapture(0, backend)
        if cam.isOpened():
            print(f"Using backend: {backend}")
            return cam
    print("Failed to open camera with all backends.")
    return None

# Function to extract ROI from the hand sign
def extract_hand_roi(frame):
    # Convert to HSV for better color segmentation
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Thresholding to create a mask for skin color
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)

    # Apply Gaussian blur to the mask
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding box for the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Expand the bounding box to ensure the whole hand is included
        padding = 30  # Increased padding for more accurate ROI
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)

        # Crop the hand ROI
        hand_roi = frame[y:y + h, x:x + w]
        return hand_roi, (x, y, x + w, y + h)
    
    return None, None

# Main logic to capture frames dynamically based on sign change detection
def capture_signs():
    main_folder = "captured_signs"
    os.makedirs(main_folder, exist_ok=True)

    cam = get_video_capture()
    if cam is None:
        raise Exception("No compatible camera found.")

    stable_duration = 1.0  # Duration to consider a sign stable
    cooldown_period = 3.0  # Time in seconds before detecting a new sign

    captured_signs = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # Extract hand ROI
        hand_roi, bounding_box = extract_hand_roi(frame)

        if hand_roi is not None:
            captured_signs += 1
            sign_folder = os.path.join(main_folder, f"Sign_{captured_signs}")
            os.makedirs(sign_folder, exist_ok=True)

            # Apply filters and save the hand ROI
            filtered_hand_roi = apply_filters(hand_roi)
            roi_path = os.path.join(sign_folder, "hand_roi.jpg")
            cv2.imwrite(roi_path, filtered_hand_roi)
            print(f"Captured and saved hand ROI for Sign_{captured_signs}")

            # Reset for the next sign with a cooldown period
            time.sleep(cooldown_period)

        # Show the video feed
        cv2.imshow("Video Feed", frame)

        # Manual stop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Capture stopped by user.")
            break

    cam.release()
    cv2.destroyAllWindows()
    print("All sign captures complete.")

# Call the capture_signs function to start the process
capture_signs()
q
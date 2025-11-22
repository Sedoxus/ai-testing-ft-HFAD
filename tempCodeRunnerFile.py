import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# Configuration
MODEL_PATH = "best_model.h5" 
class_names = ["closed_fist", "open_hand", "peace_sign"]
img_size = 128

# Smoothing settings
USE_SMOOTHING = True
SMOOTHING_FRAMES = 5
prediction_buffer = deque(maxlen=SMOOTHING_FRAMES)

# Confidence threshold
confidence_threshold = 0.50

# Region of Interest (ROI) settings
USE_ROI = True  
ROI_SIZE = 300  
TRACK_HAND = True  
ROI_PADDING = 40  

# Skin detection settings (HSV color space)
LOWER_SKIN = np.array([0, 20, 70], dtype=np.uint8)
UPPER_SKIN = np.array([20, 255, 255], dtype=np.uint8)

# Hand detection constraints
MIN_HAND_AREA = 5000  
MAX_HAND_AREA = 50000  
MIN_ASPECT_RATIO = 0.5  
MAX_ASPECT_RATIO = 2.0 

# ---------------------
# Load Model
# ---------------------
print("Loading model...")
model = load_model(MODEL_PATH)
print("✓ Model loaded!")
print(f"Class order: {class_names}\n")
print("=== Controls ===")
print("'q' - Quit")
print("'s' - Toggle smoothing")
print("'r' - Reset smoothing buffer")
print("'o' - Toggle ROI (Region of Interest)")
print("'t' - Toggle hand tracking")
print("'d' - Show debug view (skin detection)")
print("'[' - Decrease ROI size")
print("']' - Increase ROI size")
print("'+' / '=' - Increase confidence threshold")
print("'-' / '_' - Decrease confidence threshold\n")

# ---------------------
# Helper Functions
# ---------------------
def smooth_predictions(pred):
    """Average predictions over multiple frames"""
    prediction_buffer.append(pred[0])
    return np.mean(prediction_buffer, axis=0)

def detect_hand_contour(frame):
    """Detect hand using skin color detection and contours"""
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create skin mask
    skin_mask = cv2.inRange(hsv, LOWER_SKIN, UPPER_SKIN)
    
    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    
    # Blur to reduce noise
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
    
    # Find contours
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour = assumed to be hand
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h if h != 0 else 0
        
        # Filter based on area and aspect ratio to avoid tracking arm
        if (MIN_HAND_AREA < area < MAX_HAND_AREA and 
            MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO):
            return x, y, w, h, skin_mask
    
    return None, None, None, None, skin_mask

# ---------------------
# Main Loop
# ---------------------
cap = cv2.VideoCapture(0)
frame_count = 0

# For storing last known hand position
last_roi_x, last_roi_y = None, None
show_debug = False

print("✓ Starting hand gesture recognition...")
print("Tip: For best hand tracking, ensure good lighting and contrast with background\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Detect hand if tracking is enabled
    hand_detected = False
    skin_mask = None
    
    if USE_ROI and TRACK_HAND:
        x, y, w, h, skin_mask = detect_hand_contour(frame)
        
        if x is not None:
            hand_detected = True
            
            # Calculate ROI center from hand position
            hand_center_x = x + w // 2
            hand_center_y = y + h // 2
            
            # Center ROI on hand with padding
            roi_x = hand_center_x - ROI_SIZE // 2
            roi_y = hand_center_y - ROI_SIZE // 2
            
            # Store last position
            last_roi_x, last_roi_y = roi_x, roi_y
        elif last_roi_x is not None:
            # Use last known position if hand not detected
            roi_x, roi_y = last_roi_x, last_roi_y
        else:
            # Default to center if never detected
            roi_x = (width - ROI_SIZE) // 2
            roi_y = (height - ROI_SIZE) // 2

    if USE_ROI:
        if not TRACK_HAND or (not hand_detected and last_roi_x is None):
            # Fixed ROI in center (original behavior)
            roi_x = (width - ROI_SIZE) // 2
            roi_y = (height - ROI_SIZE) // 2
        
        # Make sure ROI fits within frame
        roi_x = max(0, min(roi_x, width - ROI_SIZE))
        roi_y = max(0, min(roi_y, height - ROI_SIZE))
        
        # Extract ROI
        roi_frame = frame[roi_y:roi_y+ROI_SIZE, roi_x:roi_x+ROI_SIZE]
        
        # Draw ROI rectangle on original frame
        roi_color = (0, 255, 0) if hand_detected else (255, 165, 0)  # Green if tracking, orange if not
        cv2.rectangle(frame, (roi_x, roi_y), 
                     (roi_x + ROI_SIZE, roi_y + ROI_SIZE), roi_color, 3)
        
        track_status = "Tracking" if hand_detected else ("Last Position" if last_roi_x else "Centered")
        cv2.putText(frame, f"ROI: {track_status}", (roi_x + 10, roi_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi_color, 2)
        
        # Make ROI square by center-cropping (if not already square)
        roi_h, roi_w, _ = roi_frame.shape
        min_dim = min(roi_h, roi_w)
        start_x = (roi_w - min_dim) // 2
        start_y = (roi_h - min_dim) // 2
        square_frame = roi_frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
        
        display = frame.copy()
    else:
        # Use full frame (original behavior)
        min_dim = min(height, width)
        start_x = (width - min_dim) // 2
        start_y = (height - min_dim) // 2
        square_frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
        display = square_frame.copy()

    # Preprocess (EXACTLY like training)
    img = cv2.resize(square_frame, (img_size, img_size))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img, verbose=0)
    
    # Apply smoothing if enabled
    if USE_SMOOTHING:
        final_pred = smooth_predictions(pred)
    else:
        final_pred = pred[0]
    
    label_index = np.argmax(final_pred)
    label = class_names[label_index]
    confidence = final_pred[label_index]

    # Get display dimensions
    h, w, _ = display.shape

    # Draw semi-transparent info panel
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)

    # Display all probabilities with bars
    y_pos = 30
    for i, name in enumerate(class_names):
        prob = final_pred[i]
        color = (0, 255, 0) if i == label_index else (180, 180, 180)
        
        # Probability bar
        bar_length = int((prob * 100) * 2)  # max 200px
        cv2.rectangle(display, (10, y_pos), (10 + bar_length, y_pos + 20), color, -1)
        cv2.rectangle(display, (10, y_pos), (210, y_pos + 20), (100, 100, 100), 1)
        
        # Text
        text = f"{name}: {prob*100:.1f}%"
        cv2.putText(display, text, (220, y_pos + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        y_pos += 45

    # Hand detection status
    if USE_ROI and TRACK_HAND:
        status_color = (0, 255, 0) if hand_detected else (180, 180, 180)
        status_text = "Hand: Detected" if hand_detected else "Hand: Not Detected"
        cv2.putText(display, status_text, (10, y_pos + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

    # Main prediction display at bottom
    if confidence >= confidence_threshold:
        # Green box for confident prediction
        cv2.rectangle(display, (0, h - 70), (w, h), (0, 100, 0), -1)
        cv2.putText(display, f"Gesture: {label}", (10, h - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, f"Confidence: {confidence*100:.1f}%", (10, h - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        # Red box for low confidence
        cv2.rectangle(display, (0, h - 70), (w, h), (0, 0, 100), -1)
        cv2.putText(display, "Low Confidence", (10, h - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, f"{confidence*100:.1f}% < {confidence_threshold*100:.0f}% threshold", 
                   (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Settings display (top right)
    track_status = "Track: ON" if TRACK_HAND else "Track: OFF"
    roi_text = f"ROI: {'ON' if USE_ROI else 'OFF'} ({ROI_SIZE}px)"
    settings_text = f"Smooth: {'ON' if USE_SMOOTHING else 'OFF'} | {roi_text} | {track_status} | Thresh: {confidence_threshold:.2f}"
    
    # Adjust text size if too long
    text_size = 0.45 if TRACK_HAND else 0.5
    cv2.putText(display, settings_text, (w - 650, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 255, 0), 1)

    # Show frame
    cv2.imshow("Hand Gesture Recognition", display)
    
    # Show debug view if enabled
    if show_debug and skin_mask is not None:
        cv2.imshow("Skin Detection (Debug)", skin_mask)

    # Print probabilities every 15 frames
    if frame_count % 15 == 0:
        probs = {class_names[i]: f"{final_pred[i]*100:.1f}%" for i in range(len(class_names))}
        hand_status = " [Hand Tracked]" if (TRACK_HAND and hand_detected) else ""
        print(probs, f"-> {label} ({confidence*100:.1f}%){hand_status}")

    # Handle keys
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
    elif key == ord("s"):
        USE_SMOOTHING = not USE_SMOOTHING
        prediction_buffer.clear()
        print(f"\nSmoothing: {'ON' if USE_SMOOTHING else 'OFF'}")
    elif key == ord("t"):
        TRACK_HAND = not TRACK_HAND
        last_roi_x, last_roi_y = None, None
        print(f"\nHand Tracking: {'ON' if TRACK_HAND else 'OFF'}")
    elif key == ord("d"):
        show_debug = not show_debug
        if not show_debug:
            cv2.destroyWindow("Skin Detection (Debug)")
        print(f"\nDebug View: {'ON' if show_debug else 'OFF'}")
    elif key == ord("o"):
        USE_ROI = not USE_ROI
        prediction_buffer.clear()
        last_roi_x, last_roi_y = None, None
        print(f"\nROI: {'ON' if USE_ROI else 'OFF'}")
    elif key == ord("r"):
        prediction_buffer.clear()
        last_roi_x, last_roi_y = None, None
        print("\nPrediction buffer & hand position reset")
    elif key == ord("["):
        ROI_SIZE = max(150, ROI_SIZE - 50)
        print(f"\nROI size: {ROI_SIZE}px")
    elif key == ord("]"):
        ROI_SIZE = min(600, ROI_SIZE + 50)
        print(f"\nROI size: {ROI_SIZE}px")
    elif key in [ord("+"), ord("=")]:
        confidence_threshold = min(0.95, confidence_threshold + 0.05)
        print(f"\nConfidence threshold: {confidence_threshold:.2f}")
    elif key in [ord("-"), ord("_")]:
        confidence_threshold = max(0.30, confidence_threshold - 0.05)
        print(f"\nConfidence threshold: {confidence_threshold:.2f}")

cap.release()
cv2.destroyAllWindows()
print("\n✓ Detection stopped")
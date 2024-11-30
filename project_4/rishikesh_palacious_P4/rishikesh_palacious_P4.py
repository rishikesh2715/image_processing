import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
import pickle

def read_images_from_folder(folder_path):
    image_paths = list(Path(folder_path).glob("*.*"))
    for image_path in image_paths:
        if image_path.is_file():
            print(f"Processing image: {image_path.name}")
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            _, img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
            yield img

def preprocess_image(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

def detect_edges(img):
    return img

def find_card_contour(edged):
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        MIN_CARD_AREA = 20000
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CARD_AREA]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for i, cnt in enumerate(contours):
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2)
        cnt = contours[0]
        hull = cv2.convexHull(cnt)
        return hull.reshape(-1, 2)
    return None

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def get_rotated_card(img, contour):
    rect = order_points(contour)
    (tl, tr, br, bl) = rect
    
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth -1, maxHeight -1],
        [0, maxHeight -1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    
    if maxHeight < maxWidth:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    
    return warped

def extract_number_and_suit(warped):
    """
    Extract ROIs with consistent sizing
    """
    h, w = warped.shape[:2]
    
    # Extract ROIs
    number_roi = warped[int(0.02*h):int(0.15*h), int(0.02*w):int(0.14*w)]
    suit_roi = warped[int(0.155*h):int(0.26*h), int(0.015*w):int(0.14*w)]
    
    # Make sure we have valid ROIs
    if number_roi is None or number_roi.size == 0 or suit_roi is None or suit_roi.size == 0:
        return None, None
        
    # Threshold ROIs to ensure binary images
    _, number_roi = cv2.threshold(number_roi, 128, 255, cv2.THRESH_BINARY)
    _, suit_roi = cv2.threshold(suit_roi, 128, 255, cv2.THRESH_BINARY)
    
    return number_roi, suit_roi

def load_classifiers(suit_model_path, suit_encoder_path, number_model_path, number_encoder_path):
    """Load both trained models and encoders"""
    suit_model = tf.keras.models.load_model(suit_model_path)
    number_model = tf.keras.models.load_model(number_model_path)
    
    with open(suit_encoder_path, 'rb') as f:
        suit_encoder = pickle.load(f)
    with open(number_encoder_path, 'rb') as f:
        number_encoder = pickle.load(f)
        
    return suit_model, suit_encoder, number_model, number_encoder

def classify_card(number_roi, suit_roi, number_model, number_encoder, suit_model, suit_encoder, target_size=(50, 50)):
    """Classify both number and suit using trained models"""
    if number_roi is None or suit_roi is None:
        return None, None
    
    # Classify number
    number_roi_processed = cv2.resize(number_roi, target_size)
    number_roi_processed = number_roi_processed.flatten() / 255.0
    number_prediction = number_model.predict(number_roi_processed.reshape(1, -1), verbose=0)
    
    # Get top 3 predictions for number
    top3_number_indices = np.argsort(number_prediction[0])[-3:][::-1]  # Get indices of top 3 predictions
    top3_numbers = [
        (number_encoder.inverse_transform([idx])[0], number_prediction[0][idx])
        for idx in top3_number_indices
    ]
    
    # Get best prediction for suit (keeping original behavior)
    suit_roi_processed = cv2.resize(suit_roi, target_size)
    suit_roi_processed = suit_roi_processed.flatten() / 255.0
    suit_prediction = suit_model.predict(suit_roi_processed.reshape(1, -1), verbose=0)
    predicted_suit = suit_encoder.inverse_transform([suit_prediction.argmax()])[0]
    suit_confidence = suit_prediction.max()
    
    return (top3_numbers, number_prediction.max()), (predicted_suit, suit_confidence)


def read_from_webcam():
    """Capture frame from webcam when user presses Enter"""
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS, 30.0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return None
        
    print("Press ENTER to capture frame, 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Show live feed
        cv2.imshow('Webcam Feed', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            # Convert to grayscale and threshold
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
            cap.release()
            cv2.destroyAllWindows()
            return binary
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return None

def read_single_image():
    """Read a single image file"""
    while True:
        image_path = input("Enter the image path: ")
        path = Path(image_path)
        if path.is_file():
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            _, img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
            return img
        else:
            print("Invalid file path. Please try again.")

def plot_images(original, edged, warped, number_roi, suit_roi, number_prediction=None, suit_prediction=None):
    plt.figure(figsize=(16, 8))
    plt.subplot(2,2,1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Image")

    plt.subplot(2,2,2)
    plt.imshow(warped, cmap='gray')
    plt.title("Rotated and Cropped Card")

    plt.subplot(2,2,3)
    plt.imshow(number_roi, cmap='gray')
    title = f"Classified as: {number_prediction[0][0][0]}" if number_prediction else "Number ROI"
    plt.title(title)

    plt.subplot(2,2,4)
    plt.imshow(suit_roi, cmap='gray')
    title = f"Classified as: {suit_prediction[0]}" if suit_prediction else "Suit ROI"
    plt.title(title)

    plt.tight_layout()
    plt.show()

def process_image(img, suit_model, suit_encoder, number_model, number_encoder):
    """Process a single image"""
    if img is None:
        print("No valid image to process")
        return
        
    preprocessed = preprocess_image(img)
    edged = detect_edges(preprocessed)
    card_contour = find_card_contour(edged)
    
    if card_contour is not None and len(card_contour) >= 4:
        if len(card_contour) > 4:
            peri = cv2.arcLength(card_contour, True)
            card_contour = cv2.approxPolyDP(card_contour, 0.02 * peri, True)
            if len(card_contour) != 4:
                print("Could not approximate contour to 4 points.")
                return
            card_contour = card_contour.reshape(4, 2)
        
        warped = get_rotated_card(img, card_contour)
        number_roi, suit_roi = extract_number_and_suit(warped)
        
        if number_roi is not None and suit_roi is not None:
            number_result, suit_result = classify_card(
                number_roi, suit_roi, 
                number_model, number_encoder,
                suit_model, suit_encoder
            )
            
            if number_result[1] > 0.5 and suit_result[1] > 0.5:
                print(f"Predicted: {number_result[0][0][0]} of {suit_result[0]}")
            else:
                print("Low confidence prediction - possible misdetection")
            
            plot_images(img, edged, warped, number_roi, suit_roi, number_result, suit_result)
        else:
            print("Failed to extract valid ROIs")
    else:
        print("No card contour detected or contour does not have enough points.")

def main(suit_model_path, suit_encoder_path, number_model_path, number_encoder_path):
    # Load the trained models and encoders
    suit_model, suit_encoder, number_model, number_encoder = load_classifiers(
        suit_model_path, suit_encoder_path, number_model_path, number_encoder_path
    )
    
    while True:
        print("\nChoose an option:")
        print("1. Use webcam")
        print("2. Read single image")
        print("3. Read images from folder")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            img = read_from_webcam()
            if img is not None:
                process_image(img, suit_model, suit_encoder, number_model, number_encoder)
                
        elif choice == '2':
            img = read_single_image()
            if img is not None:
                process_image(img, suit_model, suit_encoder, number_model, number_encoder)
                
        elif choice == '3':
            folder_path = input("Enter the folder path containing images: ")
            for img in read_images_from_folder(folder_path):
                process_image(img, suit_model, suit_encoder, number_model, number_encoder)
                
        elif choice == '4':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    suit_model_path = "suit_model_best.h5"
    suit_encoder_path = "suit_encoder.pkl"
    number_model_path = "number_model_best.h5"
    number_encoder_path = "number_encoder.pkl"
    
    main(suit_model_path, suit_encoder_path, number_model_path, number_encoder_path)
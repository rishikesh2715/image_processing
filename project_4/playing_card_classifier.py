import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib2 import Path
import os

def read_image():
    # enter 1 or 2 to choose between webcam video or image
    print("Enter 1 to use webcam video or 2 to use an image")
    choice = input()
    if choice == '1':
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            cv2.imshow("frame", frame)

            # wait for enter to get pressed and then store the frame in img
            if cv2.waitKey(1) & 0xFF == ord('\r'):
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                break

            # IF 'q' is pressed, break the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return img

    elif choice == '2':
        while True:
            image_name = input("Enter the image name: ")
            image_path = Path(image_name)
            if not image_path.exists():
                print(f"Cannot read image named '{image_name}'")
            else:
                img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                return img
    else:
        print("Invalid choice. Enter 1 or 2")
        return None

def gaussian_blur(img):
    gaussian_kernel = np.array([[0, 0, 1, 2, 1, 0, 0],
                                [0, 3, 13, 22, 13, 3, 0],
                                [1, 13, 59, 97, 59, 13, 1],
                                [2, 22, 97, 159, 97, 22, 2],
                                [1, 13, 59, 97, 59, 13, 1],
                                [0, 3, 13, 22, 13, 3, 0],
                                [0, 0, 1, 2, 1, 0, 0]], dtype=np.float32) / 1003

    img_blurred = cv2.filter2D(img, cv2.CV_32F, gaussian_kernel)
    return img_blurred

def sobel_kernel(img, threshold_value):
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    grad_x = cv2.filter2D(img, cv2.CV_32F, gx)
    grad_y = cv2.filter2D(img, cv2.CV_32F, gy)

    grad_magnitude = np.abs(grad_x) + np.abs(grad_y)
    threshold = np.max(grad_magnitude) * threshold_value
    edge_mask = grad_magnitude > threshold
    grad_direction = np.arctan2(grad_y, grad_x) * 180 / np.pi
    edge_directions = (grad_direction[edge_mask] - 90.0) % 180

    hist, bin_edges = np.histogram(edge_directions, bins=np.arange(1, 181, 1))
    dominant_angle_index = np.argmax(hist)
    dominant_angle = bin_edges[dominant_angle_index]
    print(f"dominant_angle: {dominant_angle}")

    return edge_mask, dominant_angle

def rotate_image(img, dominant_angle):
    rotation_angle = 90 - dominant_angle
    if rotation_angle < 0:
        rotation_angle += 180

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_image = cv2.warpAffine(img, rotation_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return rotated_image

def crop_image(img):
    crop_img_blurred = gaussian_blur(img)
    crop_edge_mask, crop_dominant_angle = sobel_kernel(img, threshold_value=0.40)

    edge_coords = np.column_stack(np.where(crop_edge_mask))

    x_min, y_min = edge_coords.min(axis=0)
    x_max, y_max = edge_coords.max(axis=0)

    cropped_img = img[x_min:x_max, y_min:y_max]

    return cropped_img

# Function to crop the ROI of the image (roi is the number and suit of the card)
def card_roi(cropped_img):
    # apply binary thresholding
    _, thresh = cv2.threshold(cropped_img, 70, 255, cv2.THRESH_BINARY_INV)

    # crop the left top corner of the image
    number_roi = thresh[0:225, 0:160]
    suit_roi = thresh[225:390, 0:160]
    return number_roi, suit_roi

def load_templates():
    numbers_path = Path('Template/numbers')
    suits_path = Path('Template/suits')

    number_templates = {}
    suit_templates = {}

    # Load number templates
    for filename in os.listdir(numbers_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            name = filename.split('.')[0]
            template = cv2.imread(str(numbers_path / filename), cv2.IMREAD_GRAYSCALE)
            if template is None:
                print(f"Failed to load number template {filename}")
                continue
            _, template = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY_INV)
            number_templates[name] = template

    # Load suit templates
    for filename in os.listdir(suits_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            name = filename.split('.')[0]
            template = cv2.imread(str(suits_path / filename), cv2.IMREAD_GRAYSCALE)
            if template is None:
                print(f"Failed to load suit template {filename}")
                continue
            _, template = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY_INV)
            suit_templates[name] = template

    return number_templates, suit_templates


def match_templates(roi, templates):
    best_match = None
    best_score = -np.inf

    for name, template in templates.items():
        # Resize ROI to match template size
        roi_resized = cv2.resize(roi, (template.shape[1], template.shape[0]))
        result = cv2.matchTemplate(roi_resized, template, cv2.TM_CCOEFF_NORMED)
        (_, score, _, _) = cv2.minMaxLoc(result)

        if score > best_score:
            best_score = score
            best_match = name

    return best_match, best_score

def plot_images(img, cropped_img, number_roi, suit_roi, number_match, suit_match):
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")

    plt.subplot(2, 3, 2)
    plt.imshow(cropped_img, cmap='gray')
    plt.title("Rotated & Cropped Image")

    plt.subplot(2, 3, 3)
    plt.imshow(number_roi, cmap='gray')
    plt.title(f"Number ROI\nMatched: {number_match}")

    plt.subplot(2, 3, 4)
    plt.imshow(suit_roi, cmap='gray')
    plt.title(f"Suit ROI\nMatched: {suit_match}")

    plt.tight_layout()
    plt.show()

def main():
    img = read_image()
    if img is None:
        return
    blurred_image = gaussian_blur(img)
    edge_mask, dominant_angle = sobel_kernel(blurred_image, threshold_value=0.1)
    print(f"Dominant Angle: {dominant_angle}")
    rotated_image = rotate_image(blurred_image, dominant_angle)
    cropped_image = crop_image(rotated_image)
    number_roi, suit_roi = card_roi(cropped_image)

    # Load templates
    number_templates, suit_templates = load_templates()

    # Match templates
    number_match, number_score = match_templates(number_roi, number_templates)
    suit_match, suit_score = match_templates(suit_roi, suit_templates)

    print(f"Detected Card: {number_match} of {suit_match}")

    plot_images(img, cropped_image, number_roi, suit_roi, number_match, suit_match)

if __name__ == "__main__":
    main()

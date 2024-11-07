import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib2 import Path


def read_image():
    # enter 1 or 2 to coose between webcame video or image
    print("Enter 1 to use webcam video or 2 to use an image")
    choice = input()
    if choice == '1':
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            cv2.imshow("frame", frame)

            # wait for enter to get presses and then store the frame in img
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
    gx = np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])
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

# Function to crop the ROI of the image ( roi is the number and suit of the card)
def card_roi(cropped_img):
    # apply binary thresholding
    _, thresh = cv2.threshold(cropped_img, 70, 255, cv2.THRESH_BINARY)

    # crop the left top corner of the image
    number_roi = thresh[0:225, 0:160]
    suit_roi = thresh[225:390, 0:160]
    return number_roi, suit_roi

def template_matching(roi, templates, template_names):
    # Ensure ROI is of the right data type (uint8)
    roi = roi.astype(np.uint8)
    
    best_match_name = None
    best_match_score = -1
    matched_template = None
    
    for template, name in zip(templates, template_names):
        # Ensure the template is also uint8
        template = template.astype(np.uint8)
        
        result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if max_val > best_match_score:
            best_match_score = max_val
            best_match_name = name
            matched_template = template

    return best_match_name, best_match_score, matched_template


def load_templates(template_paths):
    templates = []
    template_names = []
    for path in template_paths:
        template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if template is not None:
            templates.append(template)
            template_names.append(Path(path).stem)
    return templates, template_names


def plot_images(img, cropped_img, number_roi, suit_roi):
    plt.figure(figsize=(8,6))
    plt.subplot(2,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")

    plt.subplot(2,2,2)
    plt.imshow(cropped_img, cmap='gray')
    plt.title("Rotated & Cropped Image")

    plt.subplot(2,2,3)
    plt.imshow(number_roi, cmap='gray')
    plt.title("ROI of the card number")

    plt.subplot(2,2,4)
    plt.imshow(suit_roi, cmap='gray')
    plt.title("ROI of the card suit")

    plt.tight_layout()
    plt.show()

def main():
    img = read_image()
    blurred_image = gaussian_blur(img)
    edge_mask, dominant_angle = sobel_kernel(blurred_image, threshold_value=0.25)
    rotated_image = rotate_image(blurred_image, dominant_angle)
    cropped_image = crop_image(rotated_image)
    number_roi, suit_roi = card_roi(cropped_image)
    
    # Load number and suit templates
    number_template_paths = ["templates/numbers/A.jpeg", "templates/numbers/2.jpeg", "templates/numbers/3.jpeg",
                             "templates/numbers/4.jpeg", "templates/numbers/5.jpeg", "templates/numbers/6.jpeg",
                             "templates/numbers/7.jpeg", "templates/numbers/8.jpeg", "templates/numbers/9.jpeg",
                             "templates/numbers/10.jpeg", "templates/numbers/J.jpeg", "templates/numbers/Q.jpeg",
                             "templates/numbers/K.jpeg"]
    suit_template_paths = ["templates/suits/club.jpeg", "templates/suits/diamond.jpeg", "templates/suits/heart.jpeg", "templates/suits/spade.jpeg"] 
    
    number_templates, number_names = load_templates(number_template_paths)
    suit_templates, suit_names = load_templates(suit_template_paths)

    # Apply template matching to number ROI
    best_number, number_score, matched_number_template = template_matching(number_roi, number_templates, number_names)
    
    # Apply template matching to suit ROI
    best_suit, suit_score, matched_suit_template = template_matching(suit_roi, suit_templates, suit_names)

    # Display results
    print(f"Best matched number: {best_number} with score: {number_score}")
    print(f"Best matched suit: {best_suit} with score: {suit_score}")

    plot_images(img, cropped_image, number_roi, suit_roi)

    if matched_number_template is not None:
        plt.figure(figsize=(5, 5))
        plt.title(f"Matched Number Template: {best_number}")
        plt.imshow(matched_number_template, cmap='gray')
        plt.show()

    if matched_suit_template is not None:
        plt.figure(figsize=(5, 5))
        plt.title(f"Matched Suit Template: {best_suit}")
        plt.imshow(matched_suit_template, cmap='gray')
        plt.show()

if __name__ =="__main__":
    main()


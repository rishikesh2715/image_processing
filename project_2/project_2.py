import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib2 import Path


def verify_image():
    while True:
        # image_name = input("Enter the image name: ")
        image_name = "Testimage1.tif"
        image_path = Path(image_name)
        if not image_path.exists():
            print(f"Cannot read image named '{image_name}'")
        else:
            return image_path

def read_image(image_path):
    image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    image_gray_blurred = cv2.GaussianBlur(image_gray, (3,3), 9)
    ret, image_gray_blurred = cv2.threshold(image_gray_blurred, 127, 255, cv2.THRESH_BINARY)
    return image_gray_blurred

def sobel_filter(image_gray_blurred):
    # Sobel kernels
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Compute gradients
    grad_x = cv2.filter2D(image_gray_blurred, cv2.CV_32F, gx)
    grad_y = cv2.filter2D(image_gray_blurred, cv2.CV_32F, gy)

    # Compute gradient magnitude and direction
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_direction = np.arctan2(grad_y, grad_x) * 180 / np.pi

    # Threshold gradient magnitude to get significant edges
    threshold = np.max(grad_magnitude) * 0.8  # Adjust threshold factor as needed
    edge_mask = grad_magnitude > threshold

    # Extract gradient directions at significant edges
    edge_directions = grad_direction[edge_mask]

    # Handle wrapping of angles
    edge_directions = edge_directions % 180  # Convert angles to range [0, 180)

    # Create histogram of edge orientations
    bins = np.arange(0, 181, 1)  # 1-degree bins from 0 to 180 degrees
    hist, bin_edges = np.histogram(edge_directions, bins=bins)

    # Find the dominant angle(s)
    dominant_angle_index = np.argmax(hist)
    dominant_angle = bin_edges[dominant_angle_index]
    print(f"Dominant angle: {dominant_angle} degrees")

    return grad_magnitude, dominant_angle

def rotate_image(image, angle):
    # Convert angle to radians for rotation matrix
    angle_rad = -np.deg2rad(angle)

    # Get image dimensions
    h, w = image.shape

    # Calculate the center of the image
    center_x, center_y = w // 2, h // 2

    # Create rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])

    # Create an empty output image
    rotated_image = np.zeros_like(image)

    # Iterate over each pixel in the output image
    for i in range(h):
        for j in range(w):
            # Coordinates relative to center
            x = j - center_x
            y = i - center_y

            # Apply inverse rotation
            x_rot = rotation_matrix[0, 0] * x + rotation_matrix[0, 1] * y
            y_rot = rotation_matrix[1, 0] * x + rotation_matrix[1, 1] * y

            # Map back to source image coordinates
            x_src = int(round(x_rot + center_x))
            y_src = int(round(y_rot + center_y))

            # Check if coordinates are within image bounds
            if 0 <= x_src < w and 0 <= y_src < h:
                rotated_image[i, j] = image[y_src, x_src]

    return rotated_image

def crop_card(rotated_image):
    # Apply Sobel operator
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    grad_x = cv2.filter2D(rotated_image, cv2.CV_32F, gx)
    grad_y = cv2.filter2D(rotated_image, cv2.CV_32F, gy)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Threshold to create binary mask
    threshold = np.max(grad_magnitude) * 0.8
    edge_mask = grad_magnitude > threshold

    # Find coordinates where edges exist
    coords = np.column_stack(np.where(edge_mask))

    # Get bounding box coordinates
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop the image
    cropped_image = rotated_image[y_min:y_max+1, x_min:x_max+1]

    return cropped_image

def show_image(image):
    plt.figure(figsize=(8,6))
    plt.imshow(image, cmap='gray')
    plt.tight_layout()
    plt.show()

def main():
    image_path = verify_image()
    image_gray_blurred = read_image(image_path)
    grad_magnitude, dominant_angle = sobel_filter(image_gray_blurred)
    image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    rotated_image = rotate_image(image_gray, dominant_angle)
    cropped_image = crop_card(rotated_image)
    show_image(cropped_image)


if __name__ == '__main__':
    main()

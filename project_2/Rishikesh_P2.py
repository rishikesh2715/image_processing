import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib2 import Path


def read_image():
    while True:
        image_name = input("Enter the image name: ")
        image_path = Path(image_name)
        if not image_path.exists():
            print(f"Cannot read image named '{image_name}'")
        else:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            return img

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

def plot_images(img, cropped_img):
    plt.figure(figsize=(8,6))
    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")

    plt.subplot(1,2,2)
    plt.imshow(cropped_img, cmap='gray')
    plt.title("Rotated & Cropped Image")
    
    plt.tight_layout()
    plt.show()

def main():
    img = read_image()
    blurred_image = gaussian_blur(img)
    edge_mask, dominant_angle = sobel_kernel(blurred_image, threshold_value=0.25)
    rotated_image = rotate_image(blurred_image, dominant_angle)
    cropped_image = crop_image(rotated_image)
    plot_images(img, cropped_image)

if __name__ =="__main__":
    main()


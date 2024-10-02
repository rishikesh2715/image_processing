import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
from pathlib2 import Path


while True:
    image_name = input("Enter the image name: ")
    image_path = Path(image_name)
    if not image_path.exists():
        print(f"Cannot read image named '{image_name}'")
    else:
        path = image_path
        break

# path = "Testimage3.tif"
img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

gaussian_kernel = np.array([[1,4,7,4,1],
                            [4,16,26,16,4],
                            [7,26,41,26,7],
                            [4,16,26,16,4],
                            [1,4,7,4,1]]) / 273


img_blurred = cv2.filter2D(img, cv2.CV_32F, gaussian_kernel)


gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

grad_x = cv2.filter2D(img_blurred, cv2.CV_32F, gx)
grad_y = cv2.filter2D(img_blurred, cv2.CV_32F, gy)

grad_magnitude = np.abs(grad_x) + np.abs(grad_y)
grad_direction = np.arctan2(grad_y, grad_x) * 180 / np.pi
threshold = np.max(grad_magnitude) * 0.25
edge_mask = grad_magnitude > threshold

edge_directions = grad_direction[edge_mask] - 90.0

edge_directions = edge_directions % 180.0 


bins = np.arange(0, 181, 1)  
hist, bin_edges = np.histogram(edge_directions, bins=bins)

dominant_angle_index = np.argmax(hist)
dominant_angle = bin_edges[dominant_angle_index]
print(f"Dominant Angle: {dominant_angle}")

(h, w) = img.shape[:2]
center = (w // 2, h // 2)


rotation_angle = 90.0 - dominant_angle
print(f"rotation_angle: {rotation_angle}")


rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

rotated_image = cv2.warpAffine(img, rotation_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

crop_img_blurred = cv2.filter2D(rotated_image, -1, gaussian_kernel)
crop_img_blurred = cv2.filter2D(crop_img_blurred, -1, gaussian_kernel)
crop_grad_x = cv2.filter2D(crop_img_blurred, cv2.CV_32F, gx)
crop_grad_y = cv2.filter2D(crop_img_blurred, cv2.CV_32F, gy)

crop_grad_magnitude = np.abs(crop_grad_x) + np.abs(crop_grad_y)
crop_threshold = np.max(crop_grad_magnitude) * 0.25

crop_edge_mask = crop_grad_magnitude > crop_threshold


edge_coords = np.column_stack(np.where(crop_edge_mask))

x_min, y_min = edge_coords.min(axis=0)
x_max, y_max = edge_coords.max(axis=0)

cropped_image = rotated_image[x_min:x_max, y_min:y_max]



images = [img, edge_mask, rotated_image, crop_edge_mask, cropped_image]

titles = [f"{path}, Dominant Angle: {dominant_angle}",
            f"Edge Mask" ,f"Rotated Image, Rotation Angle: {rotation_angle}", 
            "Cropped Edge Mask", "Cropped Image"]

plt.figure(figsize=(12,8))
for i in range(len(titles)):
    plt.subplot(3, 2, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])

plt.subplot(3,2,6)
plt.hist(edge_directions, bins=bins)
plt.tight_layout()
plt.show()
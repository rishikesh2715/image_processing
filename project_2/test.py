import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

path = "Testimage1.tif"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

gaussian_kernel = np.array([[1,4,7,4,1],
                            [4,16,26,16,4],
                            [7,26,41,26,7],
                            [4,16,26,16,4],
                            [1,4,7,4,1]]) / 273


img_blurred = cv2.filter2D(img, -1, gaussian_kernel)


gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

grad_x = cv2.filter2D(img_blurred, cv2.CV_32F, gx)
grad_y = cv2.filter2D(img_blurred, cv2.CV_32F, gy)

grad_magnitude = np.abs(grad_x) + np.abs(grad_y)
grad_direction = np.arctan2(grad_y, grad_x) * 180 / np.pi

# print(grad_direction)
threshold = np.max(grad_magnitude) * 0.3
edge_mask = grad_magnitude > threshold

edge_directions = grad_direction[edge_mask] - 90.0
# print(edge_directions)
edge_directions = edge_directions % 180.0  
# print(edge_directions)
bins = np.arange(0, 181, 1)  
hist, bin_edges = np.histogram(edge_directions, bins=bins)

dominant_angle_index = np.argmax(hist)
dominant_angle = bin_edges[dominant_angle_index]

# Get image center for rotation
(h, w) = img.shape[:2]
center = (w // 2, h // 2)

rotation_angle = 90.0 - dominant_angle
print(rotation_angle)
r_img = ndimage.rotate(edge_mask, rotation_angle)
plt.imshow(r_img, cmap='gray')
plt.show()
# Compute the rotation matrix
# rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

# # Calculate the new dimensions of the rotated image
# cos_angle = np.abs(rotation_matrix[0, 0])
# sin_angle = np.abs(rotation_matrix[0, 1])
# new_w = int((h * sin_angle) + (w * cos_angle))
# new_h = int((h * cos_angle) + (w * sin_angle))

# # Adjust the rotation matrix to shift the image to the center
# rotation_matrix[0, 2] += (new_w / 2) - center[0]
# rotation_matrix[1, 2] += (new_h / 2) - center[1]

# # Apply the rotation using warpAffine with the adjusted dimensions
# rotated_image = cv2.warpAffine(img, rotation_matrix, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)
# Compute the rotation matrix
rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

# Apply the rotation using warpAffine
rotated_image = cv2.warpAffine(img, rotation_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
# rotated_image = cv2.warpAffine(img, rotation_matrix, (w, h))
"""
Now that rotation is working properly

Lets move on to cropping it.

"""

crop_img_blurred = cv2.filter2D(rotated_image, -1, gaussian_kernel)
crop_img_blurred = cv2.filter2D(crop_img_blurred, -1, gaussian_kernel)
crop_grad_x = cv2.filter2D(crop_img_blurred, cv2.CV_32F, gx)
crop_grad_y = cv2.filter2D(crop_img_blurred, cv2.CV_32F, gy)

crop_grad_magnitude = np.abs(crop_grad_x) + np.abs(crop_grad_y)
crop_threshold = np.max(crop_grad_magnitude) * 0.3

crop_edge_mask = crop_grad_magnitude > crop_threshold

"""How do i find the edges pixel locations now????"""
# Step 1: Find the coordinates of the edge pixels
# Step 1: Find the coordinates of the edge pixels as before
# edge_coords = np.column_stack(np.where(crop_edge_mask))

# # Step 2: Filter out edges near the borders (you might need to adjust the margin value based on your image)
# margin = 0 # This is a tolerance value to avoid border pixels; adjust this as needed
# filtered_edge_coords = edge_coords[
#     (edge_coords[:, 0] > margin) & (edge_coords[:, 0] < h - margin) &
#     (edge_coords[:, 1] > margin) & (edge_coords[:, 1] < w - margin)
# ]

# # Check if any edge pixels are left after filtering
# if len(filtered_edge_coords) == 0:
#     print("No edge pixels detected within margin range. Adjust the margin value or threshold.")
# else:
#     # Step 3: Find the minimum and maximum x and y coordinates from filtered edges
#     x_min, y_min = filtered_edge_coords.min(axis=0)
#     x_max, y_max = filtered_edge_coords.max(axis=0)

#     # Step 4: Crop the rotated image using the refined bounding box
#     cropped_image = rotated_image[x_min:x_max, y_min:y_max]

#     # Display the cropped image
#     # plt.imshow(cropped_image, cmap='gray')
#     # plt.title("Cropped Image")
#     # plt.axis('on')
#     # plt.show()


# Step 1: Find the coordinates of the edge pixels
edge_coords = np.column_stack(np.where(crop_edge_mask))

# Step 2: Find the minimum and maximum x and y coordinates to get the bounding box
x_min, y_min = edge_coords.min(axis=0)
x_max, y_max = edge_coords.max(axis=0)

# Step 3: Crop the rotated image using the bounding box
cropped_image = rotated_image[x_min:x_max, y_min:y_max]


images = [img, rotated_image, crop_edge_mask, cropped_image]
titles = [f"Original Image: {path}", "Rotated Image", "Cropped Edge Mask", "Cropped Image"]

plt.figure(figsize=(12,8))
for i in range(len(titles)):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])

plt.tight_layout()
plt.show()
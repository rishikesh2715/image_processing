import numpy as np
import cv2
import matplotlib.pyplot as plt


img = cv2.imread("Testimage2.tif", cv2.IMREAD_GRAYSCALE)


gaussian_kernel = np.array([[1,4,7,4,1],
                            [4,16,26,16,4],
                            [7,26,41,26,7],
                            [4,16,26,16,4],
                            [1,4,7,4,1]]) / 273


img_blurred = cv2.filter2D(img, -1, gaussian_kernel)


# sobel filters
# gx = np.array([[1, 0, -1],
#                 [2, 0, -2],
#                 [1, 0, -1]])

# gy = np.array([[1, 2, 1],
#                 [0, 0, 0],
#                 [-1, -2, -1]])


# grad_x = cv2.filter2D(img_blurred, cv2.CV_32F, gx)
# grad_y = cv2.filter2D(img_blurred, cv2.CV_32F, gy)

# # Gradient Magnitude
# sobel_magintitude = np.abs(grad_x) + np.abs(grad_y)
# threshold = np.max(sobel_magintitude) * 0.3
# edge_mask = sobel_magintitude > threshold

gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

grad_x = cv2.filter2D(img_blurred, cv2.CV_32F, gx)
grad_y = cv2.filter2D(img_blurred, cv2.CV_32F, gy)

# grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
grad_magnitude = np.abs(grad_x) + np.abs(grad_y)
grad_direction = np.arctan2(grad_y, grad_x) * 180 / np.pi

threshold = np.max(grad_magnitude) * 0.3 
edge_mask = grad_magnitude > threshold

edge_directions = grad_direction[edge_mask] - 90

edge_directions = edge_directions % 180  

bins = np.arange(0, 181, 1)  
hist, bin_edges = np.histogram(edge_directions, bins=bins)


dominant_angle_index = np.argmax(hist)
dominant_angle = bin_edges[dominant_angle_index]
print(f"Dominant angle: {dominant_angle} degrees")

# return grad_magnitude, dominant_angle

# Get image center for rotation
(h, w) = img.shape[:2]
center = (w // 2, h // 2)

rotation_angle = 90 - dominant_angle
print(rotation_angle)

# Compute the rotation matrix
rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

# Apply the rotation using warpAffine
rotated_image = cv2.warpAffine(img, rotation_matrix, (w, h))


"""
Now that rotation is working properly

Lets move on to cropping it.

"""

# gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
# gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

crop_grad_x = cv2.filter2D(rotated_image, cv2.CV_32F, gx)
crop_grad_y = cv2.filter2D(rotated_image, cv2.CV_32F, gy)







plt.figure(figsize=(8,6))
plt.imshow(rotated_image, cmap='gray')
plt.tight_layout()
plt.show()
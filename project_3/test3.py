import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('Proj3.tif', cv2.IMREAD_GRAYSCALE)

image = cv2.GaussianBlur(image, (51,51), sigmaX=10)

# Use Gaussian filter to remove non-uniform illumination
blurred_image = cv2.GaussianBlur(image, (15, 15), sigmaX=10)  # Adjust sigmaX as needed

# Subtract the illumination-removed image from the original image
illumination_removed_image_black = cv2.subtract(image, blurred_image)

# Subtract again - not too sure why the image above is black, i just thought subtracting again would make it better and it did...
# I would appreciate an explain on this if you see this comment.
uniform_image = cv2.subtract(image, illumination_removed_image_black)

# Perform Fast Fourier Transform
fft_image = np.fft.fft2(illumination_removed_image_black)
fft_image = np.fft.fftshift(fft_image)

# List of coordinates
coordinates = [(274, 181), (265, 190), (281, 195), (263, 213), (278, 218), (270, 227), (272, 204)]

# Adjustable lowpass filter size
filter_size = 2

# Get image dimensions
rows, cols = uniform_image.shape

# Create a mesh grid for frequency coordinates
u, v = np.meshgrid(np.arange(cols), np.arange(rows))

# Create the lowpass filter based on Gaussian filter equation
lowpass_filter = np.zeros((rows, cols))

for coord in coordinates:
    x, y = coord
    distance = np.sqrt((u - x) ** 2 + (v - y) ** 2)
    lowpass_filter = np.maximum(lowpass_filter, np.where(distance <= filter_size, 1, 0))

# Apply the lowpass filter
filtered_fft = fft_image * lowpass_filter

# Perform Inverse Fast Fourier Transform to get resulting image
result_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft))

# Convert the result to a valid data type for display and normalize
result_image = np.uint8(result_image.real)
result_image = cv2.normalize(result_image, None, 0, 255, cv2.NORM_MINMAX)

# Display the lowpass filter, illumination-removed, and resulting images
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(lowpass_filter, cmap='gray')
plt.title('Low Pass Filter in Frequency Domain')

plt.subplot(1, 3, 2)
plt.imshow(uniform_image, cmap='gray')
plt.title('Illumination Removed Image')

plt.subplot(1, 3, 3)
plt.imshow(result_image, cmap='gray')
plt.title('Pattern Image')

plt.show()
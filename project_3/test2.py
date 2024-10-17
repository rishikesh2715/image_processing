import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("Proj3.tif", cv2.IMREAD_GRAYSCALE)
# image = cv2.GaussianBlur(image, (5,5), sigmaX=10)

# Compute the Fourier Transform
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)

# Compute the magnitude spectrum
image_fft_spectrum = 20 * np.log(1 + np.abs(fshift))

# Coordinates of the peaks (x, y)
coordinates = [(274, 181), (265, 190), (281, 195), (263, 213), (278, 218), (270, 227), (272, 204)]

# Get image dimensions
rows, cols = image.shape

# Initialize mask with zeros
mask = np.zeros((rows, cols), dtype=np.uint8)

# Create the mask based on the peak coordinates
for coord in coordinates:
    x, y = coord  # x is column index, y is row index
    # Ensure indices are within bounds
    y_min = max(y - 1, 0)
    y_max = min(y + 1, rows)
    x_min = max(x - 1, 0)
    x_max = min(x + 1, cols)
    mask[y_min:y_max, x_min:x_max] = 1


# Apply the corrected mask
fshift_filtered = fshift * mask

# fshift_filtered = cv2.filter2D(fshift, cv2.CV_32F,  mask)

# Inverse shift and inverse Fourier Transform
f_ishift = np.fft.ifftshift(fshift_filtered)
img_back = np.fft.ifft2(f_ishift)

# Take the magnitude
img_extracted = np.abs(img_back)

# Normalize the image
img_extracted = cv2.normalize(img_extracted, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)



# Estimate the background
background = cv2.GaussianBlur(image, (101, 101), 0)

# Subtract background to get uniformly illuminated image
img_uniform = cv2.subtract(image, background)

# Normalize the result
img_uniform = cv2.normalize(img_uniform, None, 0, 255, cv2.NORM_MINMAX)


# Display all images together
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(mask, cmap='gray')
plt.title('Frequency-Domain Filter')

plt.subplot(2, 2, 3)
plt.imshow(img_extracted, cmap='gray')
plt.title('Extracted Periodic Pattern')

plt.subplot(2, 2, 4)
plt.imshow(img_uniform, cmap='gray')
plt.title('Uniformly Illuminated Image')

plt.tight_layout()
plt.show()

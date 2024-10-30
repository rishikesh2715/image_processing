import numpy as np
import matplotlib.pyplot as plt
import cv2

" read the image and convert it to float32 to avoid clipping issues "
img = cv2.imread("Proj3.tif", cv2.IMREAD_GRAYSCALE)

img_float = img.astype(np.float32)

blurred = cv2.GaussianBlur(img_float, (51, 51), 0)

corrected_image = img_float - blurred
corrected_image = cv2.normalize(corrected_image, None, 0, 255, cv2.NORM_MINMAX)
corrected_image = corrected_image.astype(np.uint8)

corrected_image_fourier = np.fft.fft2(corrected_image)
corrected_image_fourier_shifted = np.fft.fftshift(corrected_image_fourier)


"""
cooridnatess are the 6 peaks in the fourier transform spectrum of the image

"""

coordinates = [(274, 181), (265, 190), (281, 195), (263, 213), (278, 218), (270, 227), (272, 204)]


# meshgrid for the coordinates

rows, cols = corrected_image_fourier_shifted.shape
u, v = np.meshgrid(np.arange(cols), np.arange(rows))
combined_filter = np.ones((rows, cols))

for coord in coordinates:
    crow, ccol = coord

    distance = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)

    butterworth_filter = 1 / (1 + (distance / 1)**(2 * 2))

    combined_filter += butterworth_filter

combined_filter = cv2.normalize(combined_filter, None, 0, 1, cv2.NORM_MINMAX)

filtered_fourier = corrected_image_fourier_shifted * combined_filter

filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fourier))
filtered_image = np.real(filtered_image)
filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)
filtered_image = filtered_image.astype(np.uint8)

# plt.figure(figsize=(12, 6))

# # plt.subplot(2, 2, 1)
# plt.figure(figsize=(12, 6))
# plt.title('Original Image')
# plt.imshow(img, cmap='gray')
# plt.show()

# plt.subplot(2, 2, 2)
plt.figure(figsize=(12, 6))
plt.title('Frequency Domain Filter')
plt.imshow(combined_filter, cmap='gray')
plt.show()

# plt.subplot(2, 2, 3)
plt.figure(figsize=(12, 6))
plt.title('Extracted Periodic Pattern')
plt.imshow(filtered_image, cmap='gray')
plt.show()

# plt.subplot(2, 2, 4)
plt.figure(figsize=(12, 6))
plt.title('Uniformly Illuminated Image')
plt.imshow(corrected_image, cmap='gray')
plt.show()



# plt.tight_layout()
plt.show()

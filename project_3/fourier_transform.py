import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread('Proj3.tif', cv2.IMREAD_GRAYSCALE)

blurred_image = cv2.GaussianBlur(image, (15, 15), sigmaX=10)

illumination_removed_image_black = cv2.subtract(image, blurred_image)

plt.imshow(illumination_removed_image_black, cmap='gray')
plt.show()

uniform_image = cv2.subtract(image, illumination_removed_image_black)

fft_image = np.fft.fft2(illumination_removed_image_black)

fft_image = np.fft.fftshift(fft_image)

fft_image =cv2.normalize(fft_image, None, 0, 255, cv2.NORM_MINMAX)

plt.imshow(fft_image, cmap='gray')
plt.show()

coordinates = [(274, 181), (265, 190), (281, 195), (263, 213), (278, 218), (270, 227), (272, 204)]




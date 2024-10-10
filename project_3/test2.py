import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("Proj3.tif", cv2.IMREAD_GRAYSCALE)

image_float = image.astype(np.float32)

image_blurred = cv2.GaussianBlur(image_float, (51,51), 0)

illuminated_image = image_float - image_blurred

illuminated_image = cv2.normalize(illuminated_image, None, 0, 255, cv2.NORM_MINMAX)

illuminated_image = illuminated_image.astype(np.uint8)

image_fft_spectrum = np.log(1 + np.abs(np.fft.fftshift(np.fft.fft2(image))))
illuminated_image_fft_spectrum = np.log(1 + np.abs(np.fft.fftshift(np.fft.fft2(illuminated_image))))



orignal_min_val = np.min(image_fft_spectrum)
orignal_max_val = np.max(image_fft_spectrum)

orignal_mapped_image_fft = (image_fft_spectrum - orignal_min_val) * (255 / (orignal_max_val - orignal_min_val))
orignal_mapped_image_fft = orignal_mapped_image_fft.astype(np.uint8)


"""
illuminated image linear mapping

"""
illuminated_min_val = np.min(illuminated_image_fft_spectrum)
illuminated_max_val = np.max(illuminated_image_fft_spectrum)

illuminated_mapped_image = (illuminated_image_fft_spectrum - illuminated_min_val) * (255 / (illuminated_max_val - illuminated_min_val))
illuminated_mapped_image = illuminated_mapped_image.astype(np.uint8)

"""
Done
"""





images = [image, image_blurred, illuminated_image,orignal_mapped_image_fft, illuminated_mapped_image]
titles = ["Original Image", "Blurred Image", "Illuminated Image", "FFT Spectrum Orginal", "FFT Spectrum"]

for i in range(len(titles)):
    plt.subplot(2,3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])

plt.tight_layout()
plt.show()
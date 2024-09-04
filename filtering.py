""""

3. **Filtering**
   - Convolution and kernels
   - Low-pass filters (e.g., mean filter, Gaussian filter)
   - High-pass filters (e.g., Laplacian filter)
   - Median filter for noise reduction

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lena.png", cv2.IMREAD_COLOR)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


""" 
Adding Noise to the Picture
"""

# Adding Gaussian Noise to the image
def gaussian_noise(img):
    row, col, ch = img.shape
    mean = 10
    stdv = 30
    gauss = np.random.normal(mean, stdv, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = img + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

# Adding Salt & Paper noise to the image
def salt_pepper(img):
    row, col, ch = img.shape
    s_vs_p = 0.5  # Salt to pepper ratio
    amount = 0.04  # Amount of noise
    out = np.copy(img)

    # Salt mode
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape[:2]]
    
    # Apply salt noise (set pixel values to white)
    if ch == 1:  # Grayscale image
        out[coords[0], coords[1]] = 255
    else:  # Color image
        out[coords[0], coords[1], :] = 255

    # Pepper mode
    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape[:2]]
    
    # Apply pepper noise (set pixel values to black)
    if ch == 1:  # Grayscale image
        out[coords[0], coords[1]] = 0
    else:  # Color image
        out[coords[0], coords[1], :] = 0

    return out

def poisson_noise(img):
    # Ensure the image is in a valid range (e.g., 0 to 255 for uint8 images)
    img = np.asarray(img, dtype=float)
    
    # Apply Poisson noise
    noisy = np.random.poisson(img)
    
    # Ensure the output has the same data type as the input
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    return noisy

"""
Convolation and Kernels

"""
# Basic Blurring Technqique
def blur(img):
    blur_average = cv2.blur(img, (9,9))
    blur_gaussian = cv2.GaussianBlur(img, (9,9), 0)

    return blur_average, blur_gaussian


img_gaussian = gaussian_noise(img)
img_sp = salt_pepper(img)
img_poisson = poisson_noise(img)
img_blur, img_gaussian_blur = blur(img)

plt.figure(figsize=(12,10))

plt.subplot(2,3,1)
plt.imshow(img)
plt.title("Original Image")

plt.subplot(2,3,2)
plt.imshow(img_gaussian)
plt.title("Image with Gaussian Noise")

plt.subplot(2,3,3)
plt.imshow(img_sp)
plt.title("Image with Salt & Pepper Noise")

plt.subplot(2,3,4)
plt.imshow(img_poisson)
plt.title("Image with Poisson Noise")

plt.subplot(2,3,5)
plt.imshow(img_blur)
plt.title("Image with Average Blur (9X9 Kernel)")

plt.subplot(2,3,6)
plt.imshow(img_gaussian_blur)
plt.title("Image with Gaussian Blur (9X9 Kernel)")

plt.tight_layout()
plt.show()
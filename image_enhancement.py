"""

2. **Image Enhancement**
   - Contrast adjustment
   - Brightness adjustment
   - Histogram equalization

"""

import cv2
import matplotlib.pyplot as plt


img = cv2.imread("cat.jpg", cv2.IMREAD_GRAYSCALE)


"""
Contrast & Brightness Adjustment

"""

alpha = 10000
beta = 1

img_enhanced = cv2.convertScaleAbs(img, alpha, beta)


fig, axs = plt.subplots(1, 3, figsize=(16, 10))  

axs[0].imshow(img, cmap = 'gray')
axs[0].set_title('Orignal Grayscale Image')


axs[1].hist(img.ravel(), 256, [0, 256])
axs[1].set_title('Histogram')
axs[1].set_xlabel('Pixel Intensity')
axs[1].set_ylabel('Frequency')

axs[2].imshow(img_enhanced, cmap = 'gray')
axs[2].set_title(f'Enhanced Image with Adjusted Contrast: {alpha} \n & Brightness: {beta}')

plt.tight_layout()
plt.show()








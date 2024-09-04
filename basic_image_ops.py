"""

1. **Basic Image Operations**
   - Image resizing and scaling -- Done 
   - Image rotation and flipping
   - Cropping and padding

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# reading the image (openCV loads image in BGR format from defualt)
img = cv2.imread("cat.jpg")

# changing the image to RGB format
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# size of the image
height, width, shape = img.shape
# print(f"Height: {height}, Widht: { width}, shape: {shape} ")

""" Resizing the image """

# half size
img_half = cv2.resize(img, ((int(width/2)), int((height/2))))

# 1.5 times the size
img_bigger = cv2.resize(img, ((int(width*1.5)), int((height*1.5))))

# Scaling the images with a scaling factor of 2 (fx = scaling along x axis and fy = scaling along y axis)
img_scaled = cv2.resize(img, (0,0), fx = 2, fy = 2, interpolation = cv2.INTER_LINEAR)


""" Image Rotation and Flipping """

# Rotating the image 45 degrees
img_rotated_45 = ndimage.rotate(img, 45.0)

# Rotating the image 180 degrees (flipping)
img_rotated_180 = ndimage.rotate(img, 180.0)


""" Cropping and Padding """

# cropping the image
img_cropped = img[500:1500, 1000:2000]

# adding padding around the image. The value parameter is optional. Black(0,0,0) is default.
img_padded = cv2.copyMakeBorder(img, 300, 300, 300, 300, cv2.BORDER_CONSTANT, value=[0,0,0])


""" Plotting the images with Matplotlib"""

Title = [
    "Original", 
    "Half", 
    "Bigger", 
    "Scaled (Factor = 2)", 
    "Rotated 45 Deg", 
    "Rotated 180 Deg",
    "Cropped Image",
    "Padded image"]

images = [img, img_half, img_bigger, img_scaled, img_rotated_45, img_rotated_180, img_cropped, img_padded]

plt.figure(figsize = (12,10))

for i in range(len(images)):
    plt.subplot(3, 3, i+1)
    plt.title(Title[i])
    plt.imshow(images[i])

plt.tight_layout()
plt.savefig('final_output.png', bbox_inches = 'tight')
plt.show()
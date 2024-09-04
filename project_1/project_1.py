"""
Rishikesh

Project 1

Image Processing

python 3.9.5
"""

import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np

imgs = [cv2.imread(file) for file in glob.glob("ImageSet1/ImageSet1/*.jpg")]

imgs_list = []

day_night = []

for img in imgs:
    imgs_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgs_list.append(imgs_rgb)

for image in imgs_list:
    gray_pixel = np.sum((image[:, :, 0] == image[:, :, 1]) & (image[:, :, 1] == image[:, :, 2]))

    total_pixels = image.shape[0] * image.shape[1]

    is_night = gray_pixel >= total_pixels / 2

    if is_night:
        day_night.append('Night')
    else:
        day_night.append('Day')

    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.title('Night' if is_night else 'Day')
    plt.show()


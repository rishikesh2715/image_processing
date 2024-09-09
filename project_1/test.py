import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

path = glob.glob('ImageSet1/ImageSet1/*.JPG')
images = [cv2.imread(file) for file in path]

titles = ['RGB', 'HSV', 'HSV_FULL', 'HLS', 'HLS_FULL', 'YCrCb']
color_conversions = [
    cv2.COLOR_BGR2RGB,
    cv2.COLOR_BGR2HSV,
    cv2.COLOR_BGR2HSV_FULL,
    cv2.COLOR_BGR2HLS,
    cv2.COLOR_BGR2HLS_FULL,
    cv2.COLOR_BGR2YCrCb
]

for idx, image in enumerate(images):
    plt.figure(figsize=(15, 10))
    for i, (title, conversion) in enumerate(zip(titles, color_conversions)):
        plt.subplot(2, 3, i+1)
        plt.imshow(cv2.cvtColor(image, conversion))
        plt.title(title)
    
    plt.tight_layout()
    plt.suptitle(f'Image {idx+1}', fontsize=16)
    plt.show()
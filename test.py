import numpy as np
import cv2
import matplotlib.pyplot as plt



img = cv2.imread("logo.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_linear = cv2.resize(img, (0,0), fx = 5.0, fy = 5.0, interpolation = cv2.INTER_LINEAR)
img_nearest = cv2.resize(img, (0,0), fx = 5.0, fy = 5.0, interpolation = cv2.INTER_NEAREST)
img_area = cv2.resize(img, (0,0), fx = 5.0, fy = 5.0, interpolation = cv2.INTER_AREA)
img_cubic = cv2.resize(img, (0,0), fx = 5.0, fy = 5.0, interpolation = cv2.INTER_CUBIC)
img_lanczos = cv2.resize(img, (0,0), fx = 5.0, fy = 5.0, interpolation = cv2.INTER_LANCZOS4)


images = [img, img_linear, img_nearest, img_area, img_cubic, img_lanczos]

titles = ["Original", "Linear", "Nearest", "Area", "Cubic", "Lanczos"]

plt.figure(figsize=(16,14))
for i in range(len(images)):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i])
    plt.title(titles[i])

plt.show()

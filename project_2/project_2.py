import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("king.jpg", cv2.IMREAD_GRAYSCALE)

ret, img_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)


plt.figure(figsize=(12,10))

plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.title("Orignal Image")

plt.subplot(1,2,2)
plt.imshow(img_binary, cmap='gray')
plt.title("Binarized Image")

plt.show()

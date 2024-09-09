import numpy as np
import matplotlib.pyplot as plt
import cv2


img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(img)

def negative(image):
    return 255 - image


def log(img):
    c = 255 / np.log(1 + np.max(img))
    return c * np.log(1 + img)


img = negative(img)
plt.figure(figsize=(15, 10))
plt.imshow(img, cmap='gray')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import cv2


img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# print(img)

def negative(image):
    return 255 - image


def log(img):
    c = 255 / np.log(1 + np.max(img))
    return c * np.log(1 + img)

def gamma(img):
    g = 0.5
    c = 255 / np.max(img)**g
    return c * np.power(img, g)



original_image = img
img_gamma = gamma(img)
images = [original_image, img_gamma]
titles = ["Original Image", "Gamma corrected Image"]


plt.figure(figsize=(12,8))
for i in range(len(titles)):
    plt.subplot(1,2, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])

plt.tight_layout()
plt.show()


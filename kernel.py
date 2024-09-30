# import numpy as np
# import cv2
# import matplotlib.pyplot as plt


# blur_kernel = np.array([[1,0,-1],
#                         [2,0,-2],
#                         [1,0,-1]])

# # blur_kernel = blur_kernel.T

# print(blur_kernel)
# array = np.array([[0,1,2,3,4,5,6,7],
#                     [0,1,2,3,4,5,6,7],
#                     [0,1,2,3,4,5,6,7],
#                     [0,1,2,3,4,5,6,7],
#                     [0,1,2,3,4,5,6,7],
#                     [0,1,2,3,4,5,6,7]], dtype=np.uint8)


# new_array = cv2.filter2D(array, -1, blur_kernel)
# images = [array, new_array]

# titles = ["original", "Blurred?"]

# plt.figure(figsize=(12,8))
# for i in range(len(titles)):
#     plt.subplot(1,2,i+1)
#     plt.imshow(images[i], cmap='gray')
#     plt.title(titles[i])

# plt.tight_layout()
# plt.show()

import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

ret, img_binary = cv2.threshold(img, 100, 150, cv2.THRESH_BINARY)

img_binary = np.clip(img_binary, 50, 150)

img_equalized = cv2.equalizeHist(img_binary)

images = [img_binary, img_equalized]
titles = ["Binarized Image", "Histogram Equalized Image"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for i in range(len(titles)):
    axes[0, i].imshow(images[i], cmap='gray', vmin = 0, vmax = 255)
    axes[0, i].set_title(titles[i])

    axes[1, i].hist(images[i].ravel(), bins=256, range=[0, 256], color='black')
    axes[1, i].set_title(f"Histogram of {titles[i]}")

plt.tight_layout()
plt.show()

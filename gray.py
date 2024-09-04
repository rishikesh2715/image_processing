import numpy as np
import matplotlib.pyplot as plt


img_test = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.uint8)

img_range = np.arange(0, 256, 1, dtype=np.uint8)

# img_range = np.array([img_range])

# plt.imshow(img_test, cmap='gray', vmin = 0, vmax=255)
plt.figure(figsize=(20, 10))
plt.imshow(img_range, cmap='gray')
# img_range
plt.show()
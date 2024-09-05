import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np

# read the images from the folder
path = glob.glob("ImageSet1/ImageSet1/*.JPG")
imgs = [cv2.imread(file) for file in path]

for img in imgs:
    # convert the image to RGB from BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # calculate the number of gray pixels and the total number of pixels
    gray_pixel = np.sum((img[:, :, 0] == img[:, :, 1]) & (img[:, :, 1] == img[:, :, 2]))
    total_pixel = img.shape[0] * img.shape[1]

    # check if the image contains more than 90% gray pixels
    is_night = gray_pixel >= total_pixel * 0.9

    # if it is is_night then set the title to 'Night' else 'Day'
    if is_night:
        title = 'Night'
    else:
        title = 'Day'

    # display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.title(title)
    plt.show()


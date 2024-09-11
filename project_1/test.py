import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob 


path = glob.glob("ImageSet1/ImageSet1/*.JPG")
imgs = [cv2.imread(file) for file in path]

images = [] 
titles = ['RGB', 'HSV', 'HSV_FULL', 'HLS', 'HLS_FULL', 'LAB']

for img in imgs:
    images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    images.append(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    images.append(cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL))
    images.append(cv2.cvtColor(img, cv2.COLOR_BGR2HLS))
    images.append(cv2.cvtColor(img, cv2.COLOR_BGR2HLS_FULL))
    images.append(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))

plt.figure(figsize=(16, 12)) 
for i in range(len(titles)):
    plt.subplot(2, 3, i+1)  
    plt.imshow(images[i])   
    plt.title(titles[i]) 

plt.tight_layout()  
plt.show()



def negative_image(img):
    return 255 - img

negative_images = [negative_image(img) for img in images]

plt.figure(figsize=(16, 12)) 
for i in range(len(titles)):
    plt.subplot(2, 3, i+1)  
    plt.imshow(negative_images[i])   
    plt.title(titles[i]) 

plt.tight_layout()  
plt.show()  




def gamma_correction(img, gamma=1.0):
    return np.power(img, gamma)

gamma_images = [gamma_correction(img, gamma=1.5) for img in images]

plt.figure(figsize=(16, 12)) 
for i in range(len(titles)):
    plt.subplot(2, 3, i+1)  
    plt.imshow(gamma_images[i])   
    plt.title(titles[i])


def plot_histogram(img):
    plt.figure(figsize=(16, 12)) 
    plt.hist(img.ravel(), bins=256, range=(0, 256), color='gray')
    plt.title('Histogram')
    plt.show()
# 
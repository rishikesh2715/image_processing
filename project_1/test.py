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

import cv2
import glob
import numpy as np

"""
Please set the path to the ImageSet1 folder with the "*.JPG" extension which
would allow the script to read all the files in the folder that ends with .jpg extension
"""
path = glob.glob("ImageSet1/ImageSet1/*.JPG")

# read all the images in the folder
imgs = [cv2.imread(file) for file in path]

# image number counter
image_count = 0
total_images = len(imgs)

for img in imgs:

    # convert the image to HSV color space
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # calculate the percentage of pixels with hue 0
    hue_0 = np.sum(img_hsv[:, :, 0] == 0)
    total_pixel = img.shape[0] * img.shape[1]
    is_night = hue_0 >= total_pixel * 0.9
    
    # if the image is a night image, set the title to 'Night' else set it to 'Day'
    if is_night:
        title = 'Night'
    else:
        title = 'Day'
    
    
    # resize the image to 1080x720
    img = cv2.resize(img, (1080, 720))
    
    # create a blank image for the title and instructions
    title_img = np.zeros((100, 1080, 3), dtype=np.uint8)
    
    # put the title and instructions on the blank image
    cv2.putText(title_img, title, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) 
    
    if image_count < total_images - 1:
        msg = "Press Enter to continue or Q to close the script"
    else:
        msg = "Last image. Press any button to close the script"
    
    cv2.putText(title_img, msg, (10, 70), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    
    # combine the title image and the main image
    combined_img = np.vstack((title_img, img))
    
    # display the image
    cv2.imshow('image', combined_img)
    
    # wait for a key press
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q') and image_count < total_images - 1:
        break
    
    image_count += 1
    cv2.destroyAllWindows()

# close all the windows
cv2.destroyAllWindows()


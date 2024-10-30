import numpy as np
import matplotlib.pyplot as plt
import cv2


# ask the user if they would like to open the camera or a image file
print('Would you like to open the camera or an image file?')
print('1. Camera')
print('2. Image file')

choice = int(input('Enter your choice: '))

if choice == 1:
    # camera parameters
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FPS, 30.0)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # read video frame in while loop
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # display the frame
        cv2.imshow('Frame', frame)

        # check if enter is pressed
        if cv2.waitKey(1) & 0xFF == 13:
            image = frame
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            break

        # exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the camera and close the window
    camera.release()
    cv2.destroyAllWindows()

elif choice == 2:
    # ask the user to enter the path of the image file
    path = input('Enter the path of the image file: ')
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print('Invalid path')
        exit()

else:
    print('Invalid choice')
    exit()


# check if the image is captured
if image is None:
    print('No image captured')
    exit()


# plot the image
plt.figure(figsize=(14, 10))
plt.imshow(image, cmap='gray')
plt.show()

# sobel kernel
sobel_x = np.array([[-1, 0, 1], 
                    [-2, 0, 2], 
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

# apply the sobel kernel to the image
image_sobel_x = cv2.filter2D(image, -1, sobel_x)
image_sobel_y = cv2.filter2D(image, -1, sobel_y)

# magnitude of the gradient
magnitude = np.sqrt(image_sobel_x**2 + image_sobel_y**2)

# plot the magnitude of the gradient
plt.figure(figsize=(14, 10))
plt.imshow(magnitude, cmap='gray')
plt.show()

# apply thresholding to the magnitude of the gradient
threshold = 100
magnitude_thresholded = np.where(magnitude > threshold, 255, 0)

# plot the thresholded image
plt.figure(figsize=(14, 10))
plt.imshow(magnitude_thresholded, cmap='gray')


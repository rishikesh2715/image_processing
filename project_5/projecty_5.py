import cv2
import numpy as np
import matplotlib.pyplot as plt
import random as rng


# read video file
cap = cv2.VideoCapture("BZ33C_Chip2A_Worm06.avi")

# find the length of the video
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# read a random frame
frame_number = rng.randint(0, length)

# set the frame number
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

# read the frame
ret, frame = cap.read()

# convert to gray scale
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


#display the first frame
plt.figure(figsize=(12, 12))
plt.imshow(frame, cmap='gray')
plt.title(f"Frame Number: {frame_number}")
plt.tight_layout()
plt.show()













"""
This code works for showing the video with the correct delay between frames.
"""

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import random as rng

# # Read video file
# cap = cv2.VideoCapture("BZ33C_Chip2A_Worm06.avi")

# # Find the length of the video
# length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print(length)

# # Get the video's FPS
# fps = cap.get(cv2.CAP_PROP_FPS)
# print(f"Frames per second: {fps}")

# # Calculate delay between frames
# delay = int(1000 / fps)



# # Display the video
# while True:
#     ret, frame = cap.read()

#     desired_width = 1080

#     # Calculate the aspect ratio
#     aspect_ratio = frame.shape[1] / frame.shape[0]  # width / height

#     # Calculate new dimensions
#     new_width = desired_width
#     new_height = int(desired_width / aspect_ratio)

#     frame = cv2.resize(frame, (new_width, new_height))
#     if not ret:
#         break
#     cv2.imshow("frame", frame)
#     if cv2.waitKey(delay) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




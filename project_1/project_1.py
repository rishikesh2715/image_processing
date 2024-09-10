import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np

# read the images from the folder
path = glob.glob("ImageSet1/ImageSet1/*.JPG")
imgs = [cv2.imread(file) for file in path]
image_count = 0
total_images = len(imgs)

for img in imgs:
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue_0 = np.sum(img_hsv[:, :, 0] == 0)
    total_pixel = img.shape[0] * img.shape[1]
    is_night = hue_0 >= total_pixel * 0.9
    
    if is_night:
        title = 'Night'
    else:
        title = 'Day'
    
    img = cv2.resize(img, (1080, 720))
    
    # Create a blank image for the title and instructions
    title_img = np.zeros((100, 1080, 3), dtype=np.uint8)
    
    # Put the title and instructions on the blank image
    cv2.putText(title_img, title, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) 
    
    if image_count < total_images - 1:
        instructions = "Press Enter to continue or Q to close the script"
    else:
        instructions = "Last image. Press any button to close the script"
    
    cv2.putText(title_img, instructions, (10, 70), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Combine the title image and the main image
    combined_img = np.vstack((title_img, img))
    
    cv2.imshow('image', combined_img)
    
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q') and image_count < total_images - 1:
        break
    
    image_count += 1
    cv2.destroyAllWindows()

cv2.destroyAllWindows()



# import cv2
# import numpy as np
# import glob
# from PIL import Image, ImageDraw, ImageFont

# def create_text_image(width, height, title, instructions, font_path="arial.ttf"):
#     # Create a new image with a black background
#     image = Image.new('RGB', (width, height), color='black')
#     draw = ImageDraw.Draw(image)

#     # Load fonts
#     title_font = ImageFont.truetype(font_path, 36)
#     instructions_font = ImageFont.truetype(font_path, 24)

#     # Calculate text positions
#     title_bbox = draw.textbbox((0, 0), title, font=title_font)
#     title_position = ((width - title_bbox[2]) // 2, 10)
    
#     instructions_bbox = draw.textbbox((0, 0), instructions, font=instructions_font)
#     instructions_position = ((width - instructions_bbox[2]) // 2, 60)

#     # Draw the text
#     draw.text(title_position, title, font=title_font, fill='white')
#     draw.text(instructions_position, instructions, font=instructions_font, fill='white')

#     # Convert PIL Image to numpy array for OpenCV
#     return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# # read the images from the folder
# path = glob.glob("ImageSet1/ImageSet1/*.JPG")
# imgs = [cv2.imread(file) for file in path]
# image_count = 0
# total_images = len(imgs)

# for img in imgs:
#     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     hue_0 = np.sum(img_hsv[:, :, 0] == 0)
#     total_pixel = img.shape[0] * img.shape[1]
#     is_night = hue_0 >= total_pixel * 0.9
    
#     title = 'Night' if is_night else 'Day'
    
#     img = cv2.resize(img, (1080, 720))
    
#     if image_count < total_images - 1:
#         instructions = "Press Enter to continue or Q to close the script"
#     else:
#         instructions = "Last image. Press any button to close the script"
    
#     # Create text image
#     text_img = create_text_image(1080, 100, title, instructions)
    
#     # Combine the text image and the main image
#     combined_img = np.vstack((text_img, img))
    
#     cv2.imshow('image', combined_img)
    
#     key = cv2.waitKey(0) & 0xFF
#     if key == ord('q') and image_count < total_images - 1:
#         break
    
#     image_count += 1
#     cv2.destroyAllWindows()

# cv2.destroyAllWindows()
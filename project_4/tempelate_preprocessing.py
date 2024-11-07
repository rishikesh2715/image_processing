import cv2
import glob
import os
import matplotlib.pyplot as plt

# Make sure the output directory exists
output_dir = "Tempelate/suits_preprocessed"
os.makedirs(output_dir, exist_ok=True)

# Read every image in the 'Tempelate/numbers' folder
path = glob.glob("Tempelate/suits/*.jpeg")  # Make sure this path is correct

# Check if path is empty
if not path:
    print("No images found. Please check the folder path or image file extensions.")

for image in path:
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)  # Load the image in grayscale
    
    if img is not None:
        # Binarize the image
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # normalize the image to 128 x 128
        img = cv2.resize(img, (128, 128))

        # Save the image with the same name in the output directory
        filename = os.path.basename(image)  # Extract filename
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, img)  # Write the image to the output path

        # Display the image in a non-blocking way
        plt.imshow(img, cmap='gray')
        plt.axis('off')  # Optional: Turn off axis for better viewing
        plt.pause(0.001)  # Display image for a short time without blocking
    else:
        print(f"Unable to read image: {image}")

plt.close()  # Close all figure windows after processing

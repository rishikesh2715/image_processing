import numpy as np
import matplotlib.pyplot as plt
import cv2

def remove_non_uniform_illumination(img):
    # Convert the image to float32 to avoid clipping issues
    img_float = img.astype(np.float32)

    # Apply Gaussian blur to estimate the background illumination
    blurred = cv2.GaussianBlur(img_float, (51, 51), 0)

    # Subtract the blurred image from the original image
    illumination_corrected = img_float - blurred

    # Normalize the result to the range [0, 255]
    illumination_corrected = cv2.normalize(illumination_corrected, None, 0, 255, cv2.NORM_MINMAX)

    # Convert back to uint8
    illumination_corrected = illumination_corrected.astype(np.uint8)

    return illumination_corrected

def apply_lowpass_filter(image_fourier, cutoff, order):
    
    # Coordinates of each peak in the Fourier spectrum (peaks where you want to apply the filter)
    coordinates = [(274, 181), (265, 190), (281, 195), (263, 213), (278, 218), (270, 227), (272, 204)]

    # Get image dimensions
    rows, cols = image_fourier.shape

    # Create a meshgrid for the coordinates
    u, v = np.meshgrid(np.arange(cols), np.arange(rows))

    # Initialize the combined filter as ones (no filtering outside of the target regions)
    combined_filter = np.ones((rows, cols))

    # Loop through the coordinates and apply the Butterworth filter for each peak
    for coord in coordinates:
        crow, ccol = coord
        
        # Calculate the Euclidean distance for the Butterworth filter, centered around each peak
        distance = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)  # Align x,y correctly
        
        # Apply the Butterworth filter formula centered at each coordinate (peak)
        butterworth_filter = 1 / (1 + (distance / cutoff)**(2 * order))
        
        # Multiply the filters to combine them at each coordinate (applying Butterworth filtering only at these peaks)
        combined_filter += butterworth_filter

    # Normalize the combined filter to avoid overflow
    combined_filter = cv2.normalize(combined_filter, None, 0, 1, cv2.NORM_MINMAX)

    # Apply the combined Butterworth filter to the Fourier-transformed image
    filtered_fourier = image_fourier * combined_filter

    # Perform the inverse Fourier transform to get the filtered image
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fourier))
    filtered_image = np.real(filtered_image)

    # Normalize the filtered image to the range [0, 255]
    filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)
    filtered_image = filtered_image.astype(np.uint8)

    return filtered_image, combined_filter

def main():
    # Load the image in grayscale
    img = cv2.imread("Proj3.tif", cv2.IMREAD_GRAYSCALE)

    # Step 1: Remove non-uniform illumination
    corrected_image = remove_non_uniform_illumination(img)

    # Step 2: Perform Fourier transform on the corrected image
    corrected_image_fourier = np.fft.fft2(corrected_image)
    corrected_image_fourier_shifted = np.fft.fftshift(corrected_image_fourier)

    # Step 3: Apply the Butterworth lowpass filter in the Fourier domain
    filtered_image, combined_filter = apply_lowpass_filter(corrected_image_fourier_shifted, cutoff=1, order=2)

    # Step 4: Rotate the filtered image by 180 degrees
    # filtered_image = np.rot90(filtered_image, 2)

    # Step 5: Visualize results
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    
    # plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 2)
    plt.title('Combined Filter')
    plt.imshow(combined_filter, cmap='gray')

    # plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 3)
    plt.title('Extracted Pattern')
    plt.imshow(filtered_image, cmap='gray')

    # plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 4)
    plt.title('Illumination Corrected Image')
    plt.imshow(corrected_image, cmap='gray')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

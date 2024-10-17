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

    # Find the Fourier Transform of the image
    illumination_corrected_fourier = np.log(1 + np.abs(np.fft.fftshift(np.fft.fft2(illumination_corrected))))

    return illumination_corrected, illumination_corrected_fourier

def apply_lowpass_filter(image, image_fourier, cutoff, order):
    
    # Coordinates of each peak in the Fourier spectrum
    coordinates = [(274, 181), (265, 190), (281, 195), (263, 213), (278, 218), (270, 227), (272, 204)]

    # Create a Butterworth low-pass filter
    def butterworth_lowpass_filter(shape, cutoff, order):
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        x = np.linspace(-ccol, ccol, cols)
        y = np.linspace(-crow, crow, rows)
        X, Y = np.meshgrid(x, y)
        D = np.sqrt(X**2 + Y**2)
        H = 1 / (1 + (D / cutoff)**(2 * order))
        return H

    # Adjustable lowpass filter size
    filter_size = 1

    # Get image dimensions
    rows, cols = image.shape

    # Create a mesh grid for frequency coordinates
    u, v = np.meshgrid(np.arange(cols), np.arange(rows))

    # Create the lowpass filter based on Gaussian filter equation (manual filter)
    manual_lowpass_filter = np.zeros((rows, cols))

    for coord in coordinates:
        x, y = coord
        distance = np.sqrt((u - x) ** 2 + (v - y) ** 2)
        manual_lowpass_filter = np.maximum(manual_lowpass_filter, np.where(distance <= filter_size, 1, 0))

    # Apply the Butterworth low-pass filter
    butterworth_filter = butterworth_lowpass_filter(image.shape, cutoff, order)

    # Combine the Butterworth filter with the manual filter
    combined_filter = butterworth_filter * manual_lowpass_filter

    # Apply the combined filter to the Fourier spectrum
    filtered_fourier = image_fourier * combined_filter

    # Perform the inverse Fourier transform to get the filtered image
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fourier))
    filtered_image = np.real(filtered_image)

    # Normalize the filtered image to the range [0, 255]
    filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)
    filtered_image = filtered_image.astype(np.uint8)

    return filtered_image, filtered_fourier

def main():
    # Load the image in grayscale
    img = cv2.imread("Proj3.tif", cv2.IMREAD_GRAYSCALE)

    corrected_image, corrected_image_fourier = remove_non_uniform_illumination(img)

    # Adjust the cutoff and order if needed
    filtered_image, filtered_image_fourier = apply_lowpass_filter(corrected_image, corrected_image_fourier, cutoff=50, order=2)

    # Display the original and corrected images
    plt.figure(figsize=(12, 6))
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')

    plt.figure(figsize=(12, 6))
    plt.title('Illumination Corrected Image')
    plt.imshow(corrected_image, cmap='gray')

    plt.figure(figsize=(12, 6))
    plt.title('Spectrum of Illumination Corrected Image')
    plt.imshow(corrected_image_fourier, cmap='gray')

    plt.figure(figsize=(12, 6))
    plt.title('Filtered Image')
    plt.imshow(filtered_image, cmap='gray')
    
    plt.figure(figsize=(12, 6))
    plt.title('Filtered Image Spectrum')
    plt.imshow(np.log(1 + np.abs(filtered_image_fourier)), cmap='gray')

    plt.show()

if __name__ == "__main__":
    main()
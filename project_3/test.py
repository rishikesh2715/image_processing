import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter

# Function to create a Butterworth low pass filter
def butterworth_lowpass_filter(shape, cutoff, order):
    P, Q = shape
    u = np.arange(P)
    v = np.arange(Q)
    U, V = np.meshgrid(u - P/2, v - Q/2, sparse=False, indexing='ij')
    D = np.sqrt(U**2 + V**2)
    H = 1 / (1 + (D / cutoff)**(2 * order))
    return H

# Step 1: Read the input image in grayscale
img = cv2.imread('Proj3.tif', cv2.IMREAD_GRAYSCALE)

# Check if the image was successfully read
if img is None:
    print('Error opening image!')
    exit()

# Step 2: Compute the Fourier Transform of the image
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Compute the magnitude spectrum
magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
magnitude_spectrum = 20 * np.log(magnitude + 1)

# Step 3: Detect peaks in the magnitude spectrum to find the periodic pattern
# Normalize the magnitude spectrum for thresholding
norm_magnitude = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
norm_magnitude = np.uint8(norm_magnitude)

# Threshold to get the bright spots (peaks)
_, thresh = cv2.threshold(norm_magnitude, 180, 255, cv2.THRESH_BINARY)

# Apply a maximum filter to find local maxima
max_filt = maximum_filter(norm_magnitude, size=20)
peaks = (norm_magnitude == max_filt) & (thresh == 255)

# Get coordinates of peaks
peak_coords = np.column_stack(np.where(peaks))

# Step 4: Create a frequency-domain filter (mask) to extract the pattern
rows, cols = img.shape
mask = np.zeros((rows, cols, 2), np.float32)  # Changed data type to float32

# Create small squares around each peak in the mask
for y, x in peak_coords:
    x_start = max(0, x - 5)
    x_end = min(cols, x + 5)
    y_start = max(0, y - 5)
    y_end = min(rows, y + 5)
    mask[y_start:y_end, x_start:x_end] = 1

# Apply the mask to the shifted DFT
fshift = dft_shift * mask

# Step 5: Visualize the frequency-domain filter
mask_magnitude = 20 * np.log(cv2.magnitude(mask[:, :, 0], mask[:, :, 1]) + 1)

# Step 6: Perform inverse DFT to get the extracted periodic pattern
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# Normalize the extracted pattern for display
cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
img_back = np.uint8(img_back)

# Step 7: Implement Butterworth Low Pass Filter to estimate illumination
# Create the Butterworth low pass filter
cutoff = 30  # Adjust the cutoff frequency as needed
order = 2    # Adjust the filter order as needed
H = butterworth_lowpass_filter((rows, cols), cutoff, order)

# Since H is real and we have complex DFT, we need to expand H to two channels
H = H.astype(np.float32)
H = np.stack((H, H), axis=-1)  # Make it two-channel to match DFT shape

# Apply the Butterworth filter to the shifted DFT
low_pass_dft = dft_shift * H

# Shift back (inverse shift)
f_ishift_lp = np.fft.ifftshift(low_pass_dft)

# Inverse DFT to get the low pass filtered image (estimated illumination)
img_low_pass = cv2.idft(f_ishift_lp)
img_low_pass = cv2.magnitude(img_low_pass[:, :, 0], img_low_pass[:, :, 1])

# Normalize the low pass image for display
cv2.normalize(img_low_pass, img_low_pass, 0, 255, cv2.NORM_MINMAX)
img_low_pass = np.uint8(img_low_pass)

# Step 8: Remove non-uniform illumination using the Butterworth filter
# Correct the illumination by dividing the original image by the estimated illumination
img_float = img.astype(np.float32)
illumination = img_low_pass.astype(np.float32)
img_corrected_bw = img_float / (illumination + 1)

# Normalize the result
img_corrected_bw = cv2.normalize(img_corrected_bw, None, 0, 255, cv2.NORM_MINMAX)
img_corrected_bw = np.uint8(img_corrected_bw)

# Step 9: (Optional) Remove non-uniform illumination using Gaussian Blur for comparison
# Estimate the illumination using a large kernel Gaussian blur
blur = cv2.GaussianBlur(img, (101, 101), 0)

# Correct the illumination by dividing the original image by the estimated illumination
illumination_gauss = blur.astype(np.float32)
img_corrected_gauss = img_float / (illumination_gauss + 1)

# Normalize the result
img_corrected_gauss = cv2.normalize(img_corrected_gauss, None, 0, 255, cv2.NORM_MINMAX)
img_corrected_gauss = np.uint8(img_corrected_gauss)

# Plot all images in the same window
plt.figure(figsize=(15, 10))

# Subplot 1: Input Image
plt.subplot(3, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Input Image')
plt.axis('off')

# Subplot 2: Magnitude Spectrum
plt.subplot(3, 3, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.axis('off')

# Subplot 3: Frequency-domain Filter
plt.subplot(3, 3, 3)
plt.imshow(mask_magnitude, cmap='gray')
plt.title('Frequency-domain Filter')
plt.axis('off')

# Subplot 4: Extracted Periodic Pattern
plt.subplot(3, 3, 4)
plt.imshow(img_back, cmap='gray')
plt.title('Extracted Periodic Pattern')
plt.axis('off')

# Subplot 5: Butterworth Low Pass Filtered Image
plt.subplot(3, 3, 5)
plt.imshow(img_low_pass, cmap='gray')
plt.title('Butterworth Low Pass Filtered')
plt.axis('off')

# Subplot 6: Uniformly Illuminated Image (Butterworth)
plt.subplot(3, 3, 6)
plt.imshow(img_corrected_bw, cmap='gray')
plt.title('Uniform Illumination (Butterworth)')
plt.axis('off')

# Subplot 7: Gaussian Blurred Image
plt.subplot(3, 3, 7)
plt.imshow(blur, cmap='gray')
plt.title('Gaussian Blurred Image')
plt.axis('off')

# Subplot 8: Uniformly Illuminated Image (Gaussian)
plt.subplot(3, 3, 8)
plt.imshow(img_corrected_gauss, cmap='gray')
plt.title('Uniform Illumination (Gaussian)')
plt.axis('off')

# Hide the unused subplot (optional)
plt.subplot(3, 3, 9)
plt.axis('off')

# Adjust layout and display
plt.tight_layout()
plt.show()

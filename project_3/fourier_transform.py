import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load an image
image = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

# generate fft of the image
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)

# generate magnitude spectrum
magnitude_spectrum = 20*np.log(np.abs(fshift))

# generate phase spectrum
phase_spectrum = np.angle(fshift)

# plot the magnitude spectrum
plt.subplot(121)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')

# plot the phase spectrum
plt.subplot(122)
plt.imshow(phase_spectrum, cmap='gray')
plt.title('Phase Spectrum')

plt.tight_layout()
plt.show()



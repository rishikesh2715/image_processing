import numpy as np
import matplotlib.pyplot as plt

# Parameters
M = 64  # Length of the signal
t = np.arange(M)  # Time indices

# Signal with frequency components at +16 and -16, and a DC offset
dc_offset = 1
frequency = 16 / M
x = np.cos(2 * np.pi * frequency * t) + dc_offset

# Compute the FFT and shift it
X = np.fft.fft(x)
X_shifted = np.fft.fftshift(X)

# Frequency bins
freqs = np.fft.fftfreq(M, d=1)
# freqs_shifted = np.fft.fftshift(freqs)
freqs_shifted = freqs
freq_bins = freqs_shifted * M  # Scale to frequency bins

# Plot the shifted Fourier spectrum
plt.stem(freq_bins, np.abs(X_shifted))
plt.xlabel('Frequency Bin')
plt.ylabel('Magnitude')
plt.title('Shifted Fourier Spectrum')
plt.grid(True)
plt.show()
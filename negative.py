# import numpy as np
# import matplotlib.pyplot as plt

# # Function to compute the Hough Transform accumulator
# def hough_transform(image):
#     # Get image dimensions
#     rows, cols = image.shape
#     # Calculate the maximum possible rho value (diagonal length of the image)
#     diag_len = int(np.ceil(np.sqrt(rows**2 + cols**2)))
#     rhos = np.linspace(-diag_len, diag_len, 2 * diag_len)
#     # Theta values from -90 to 90 degrees
#     thetas = np.deg2rad(np.arange(-90, 90))
#     # Initialize the accumulator array
#     accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
#     # Find the indices of the edge (white) pixels
#     y_idxs, x_idxs = np.nonzero(image)
#     # Vote in the accumulator array
#     for i in range(len(x_idxs)):
#         x = x_idxs[i]
#         y = y_idxs[i]
#         for t_idx in range(len(thetas)):
#             theta = thetas[t_idx]
#             # Calculate rho
#             rho = x * np.cos(theta) + y * np.sin(theta)
#             # Find the closest rho index
#             rho_idx = np.argmin(np.abs(rhos - rho))
#             # Increment the accumulator
#             accumulator[rho_idx, t_idx] += 1
#     return accumulator, thetas, rhos


# image1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
#                    [0, 255, 0, 0, 0, 0, 0, 0],
#                    [0, 255, 0, 0, 0, 0, 0, 0],
#                    [0, 255, 0, 0, 0, 0, 0, 0],
#                    [0, 255, 0, 0, 0, 0, 0, 0],
#                    [0, 255, 0, 0, 0, 0, 0, 0],
#                    [0, 255, 0, 0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)


# image2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
#                    [0, 255, 0, 0, 0, 0, 0, 0],
#                    [0, 255, 0, 0, 0, 0, 0, 0],
#                    [0, 255, 0, 0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 0, 0, 0],
#                    [0, 255, 0, 0, 0, 0, 0, 0],
#                    [0, 255, 0, 0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

# # Compute the Hough Transform accumulators
# accumulator1, thetas1, rhos1 = hough_transform(image1)
# accumulator2, thetas2, rhos2 = hough_transform(image2)

# # Function to plot the image and its Hough accumulator
# def plot_hough(image1, image2,  accumulator1, accumulator2,  thetas1, thetas2, rhos1, rhos2, title1, title2):
#     plt.figure(figsize=(12, 6))

#     # Plot the original image
#     plt.subplot(2, 2, 1)
#     plt.imshow(image1, cmap='gray', origin='upper')
#     plt.title(f'{title1} - Original Image')
#     plt.axis('off')

#     # Plot the Hough accumulator
#     plt.subplot(2, 2, 2)
#     extent = [np.rad2deg(thetas1[0]), np.rad2deg(thetas1[-1]), rhos1[-1], rhos1[0]]
#     plt.imshow(accumulator1, cmap='hot', extent=extent, aspect='auto')
#     plt.title(f'{title1} - Hough Accumulator')
#     plt.xlabel('Theta (degrees)')
#     plt.ylabel('Rho (pixels)')

#     plt.subplot(2, 2, 3)
#     plt.imshow(image2, cmap='gray', origin='upper')
#     plt.title(f'{title2} - Original Image')
#     plt.axis('off')

#     # Plot the Hough accumulator
#     plt.subplot(2, 2, 4)
#     extent = [np.rad2deg(thetas2[0]), np.rad2deg(thetas2[-1]), rhos2[-1], rhos2[0]]
#     plt.imshow(accumulator2, cmap='hot', extent=extent, aspect='auto')
#     plt.title(f'{title2} - Hough Accumulator')
#     plt.xlabel('Theta (degrees)')
#     plt.ylabel('Rho (pixels)')


#     plt.tight_layout()
#     plt.show()

# # Plot the results for the first image
# plot_hough(image1, image2, accumulator1, accumulator2, thetas1, thetas2, rhos1, rhos2, 'Image 1', 'Image 2')


import numpy as np
import matplotlib.pyplot as plt

# Function to compute the Hough Transform accumulator
def hough_transform(image):
    # Get image dimensions
    rows, cols = image.shape
    # Calculate the maximum possible rho value (diagonal length of the image)
    diag_len = int(np.ceil(np.sqrt(rows**2 + cols**2)))
    rhos = np.linspace(-diag_len, diag_len, 2 * diag_len)
    # Theta values from -90 to 90 degrees
    thetas = np.deg2rad(np.arange(-90, 90))
    # Initialize the accumulator array
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    # Find the indices of the edge (white) pixels
    y_idxs, x_idxs = np.nonzero(image)
    # Vote in the accumulator array
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(len(thetas)):
            theta = thetas[t_idx]
            # Calculate rho
            rho = x * np.cos(theta) + y * np.sin(theta)
            # Find the closest rho index
            rho_idx = np.argmin(np.abs(rhos - rho))
            # Increment the accumulator
            accumulator[rho_idx, t_idx] += 1
    return accumulator, thetas, rhos

# Function to detect peaks in the accumulator
def detect_peaks(accumulator, num_peaks, threshold=0.5):
    indices = np.argwhere(accumulator > threshold * np.max(accumulator))
    peaks = sorted(indices, key=lambda x: accumulator[x[0], x[1]], reverse=True)
    return peaks[:num_peaks]

# Function to plot the image, its Hough accumulator, and detected lines
def plot_hough_with_lines(image, accumulator, thetas, rhos, title):
    plt.figure(figsize=(12, 6))

    # Plot the original image
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray', origin='upper')
    plt.title(f'{title} - Original Image')
    plt.axis('off')

    # Plot the Hough accumulator
    plt.subplot(1, 3, 2)
    extent = [np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), rhos[-1], rhos[0]]
    plt.imshow(accumulator, cmap='hot', extent=extent, aspect='auto')
    plt.title(f'{title} - Hough Accumulator')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Rho (pixels)')

    # Detect peaks
    num_peaks = 1  # Adjust based on expected number of lines
    peaks = detect_peaks(accumulator, num_peaks)

    # Plot detected lines on the image
    plt.subplot(1, 3, 3)
    plt.imshow(image, cmap='gray', origin='upper')
    plt.title(f'{title} - Detected Lines')
    plt.axis('off')

    for peak in peaks:
        rho_idx, theta_idx = peak
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]
        # Convert polar coordinates to Cartesian coordinates for plotting
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # Determine the start and end points of the line
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        # Plot the line
        plt.plot((x1, x2), (y1, y2), color='red')

    plt.tight_layout()
    plt.show()

# Create the first image (8x8)
image1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 255, 0, 0, 0, 0, 0, 0],
                   [0, 255, 0, 0, 0, 0, 0, 0],
                   [0, 255, 0, 0, 0, 0, 0, 0],
                   [0, 255, 0, 0, 0, 0, 0, 0],
                   [0, 255, 0, 0, 0, 0, 0, 0],
                   [0, 255, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

# Create the second image (8x8)
image2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 255, 0, 0, 0, 0, 0, 0],
                   [0, 255, 0, 0, 0, 0, 0, 0],
                   [0, 255, 0, 0, 0, 0, 0, 0],
                   [0, 0,   0, 0, 0, 0, 0, 0],
                   [0, 255, 0, 0, 0, 0, 0, 0],
                   [0, 255, 0, 0, 0, 0, 0, 0],
                   [0, 0,   0, 0, 0, 0, 0, 0]], dtype=np.uint8)

# Compute the Hough Transform accumulators
accumulator1, thetas1, rhos1 = hough_transform(image1)
accumulator2, thetas2, rhos2 = hough_transform(image2)

# Plot the results for the first image
plot_hough_with_lines(image1, accumulator1, thetas1, rhos1, 'Image 1')

# Plot the results for the second image
plot_hough_with_lines(image2, accumulator2, thetas2, rhos2, 'Image 2')

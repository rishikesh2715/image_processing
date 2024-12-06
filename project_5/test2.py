import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class WormSegmenter:
    # [Previous WormSegmenter class code remains exactly the same]
    # I'm omitting it here for brevity but it should stay intact
    def __init__(self):
        self.kernel = np.ones((3,3), np.uint8)
        
    def preprocess_frame(self, frame):
        """
        Preprocess frame to enhance worm visibility.
        Args:
            frame: Input frame (can be BGR or grayscale)
        Returns:
            Preprocessed binary image
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            blurred, 
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,  # Block size
            5    # C constant
        )
        
        return binary

    def remove_grid(self, binary):
        """
        Remove the microfluidic device grid pattern.
        Args:
            binary: Binary image with grid
        Returns:
            Binary image with reduced grid interference
        """
        # Use morphological operations to reduce grid
        kernel_size = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Open operation to remove small dots
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Close operation to connect worm segments
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        # plt.imshow(closed, cmap='gray')
        # plt.show()
        
        return closed

    def get_worm_mask(self, binary):
        """
        Extract the worm from binary image using connected components.
        Args:
            binary: Preprocessed binary image
        Returns:
            Binary mask containing only the worm
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        # Get areas of all components (excluding background)
        areas = stats[1:, cv2.CC_STAT_AREA]
        
        # Find the largest component (should be the worm)
        if len(areas) > 0:
            largest_label = 1 + np.argmax(areas)
            worm_mask = (labels == largest_label).astype(np.uint8) * 255
        else:
            worm_mask = np.zeros_like(binary)
            
        return worm_mask

    def clean_mask(self, mask):
        """
        Clean up the worm mask using morphological operations.
        Args:
            mask: Binary mask of the worm
        Returns:
            Cleaned binary mask
        """
        # Fill holes
        filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        
        # Remove small objects
        cleaned = cv2.morphologyEx(filled, cv2.MORPH_OPEN, self.kernel)
        
        return cleaned

    def segment_frame(self, frame):
        """
        Perform full segmentation pipeline on a single frame.
        Args:
            frame: Input video frame
        Returns:
            Binary mask of segmented worm
        """
        # Preprocess the frame
        binary = self.preprocess_frame(frame)
        
        # Remove grid pattern
        no_grid = self.remove_grid(binary)
        
        # Get worm mask
        worm_mask = self.get_worm_mask(no_grid)
        
        # Clean up the mask
        cleaned_mask = self.clean_mask(worm_mask)
        
        return cleaned_mask

def post_process_mask(mask):
    """
    Additional cleanup of the mask using erosion and dilation.
    Args:
        mask: Initial binary mask
    Returns:
        Cleaned binary mask
    """
    # Create a slightly larger kernel for more aggressive cleaning
    # kernel = np.ones((6,6), np.uint8)  # Increased from 3x3 to 5x5
    
    # # Erode to remove small artifacts
    # eroded = cv2.erode(mask, kernel, iterations=1)
    
    # # Dilate to restore worm size
    # # cleaned = eroded  # No need to dilate
    # cleaned = cv2.dilate(eroded, kernel, iterations=2)
    
    # return cleaned

    # get structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # apply morphological operations
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # close operation to connect worm segments
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(opened, cmap='gray')
    plt.title('closed')

    plt.subplot(1, 2, 2)
    plt.imshow(closed, cmap='gray')

    plt.tight_layout()
    plt.show()



    return closed
def process_single_frame(video_path, frame_number=0):
    """
    Process a single frame from the video.
    Args:
        video_path: Path to input video file
        frame_number: Which frame to process (default 0)
    Returns:
        tuple: (original frame, initial mask, cleaned mask, overlay)
    """
    # Initialize video capture
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read frame
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Could not read frame {frame_number}")
    
    # Initialize segmenter and process frame
    segmenter = WormSegmenter()
    initial_mask = segmenter.segment_frame(frame)

    
    
    # Additional cleanup
    cleaned_mask = post_process_mask(initial_mask)
    
    # Create overlay with cleaned mask
    vis = np.zeros_like(frame)
    vis[cleaned_mask > 0] = [0, 255, 0]  # Green mask
    overlay = cv2.addWeighted(frame, 0.7, vis, 0.3, 0)
    
    # Clean up
    cap.release()
    
    return frame, initial_mask, cleaned_mask, overlay

def display_results(frame, initial_mask, cleaned_mask, overlay):
    """
    Display all steps of the process using matplotlib.
    """
    plt.figure(figsize=(18, 15))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cv2.imwrite("original_frame.png", frame)
    plt.title('Original Frame')

    plt.subplot(2, 2, 2)
    plt.imshow(initial_mask, cmap='gray')
    # save the initial mask
    cv2.imwrite("initial_mask.png", initial_mask)
    plt.title('Initial Mask')

    plt.subplot(2, 2, 3)
    plt.imshow(cleaned_mask, cmap='gray')
    plt.title('Cleaned Mask')

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title('Overlay')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    video_path = "BZ33C_Chip1D_Worm27.avi"
    
    # Choose which frame to process (e.g., frame 50)
    frame_number = 14

    
    
    try:
        # Process single frame
        frame, initial_mask, cleaned_mask, overlay = process_single_frame(video_path, frame_number)
        
        # Display results
        display_results(frame, initial_mask, cleaned_mask, overlay)
        
    except Exception as e:
        print(f"Error processing video: {e}")
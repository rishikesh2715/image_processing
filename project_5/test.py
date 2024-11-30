import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class WormSegmenter:
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

def process_video(video_path, output_dir=None, display=True):
    """
    Process entire video and optionally save results.
    Args:
        video_path: Path to input video file
        output_dir: Directory to save output frames (optional)
        display: Whether to display results (default True)
    """
    # Create output directory if needed
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize video capture
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Initialize segmenter
    segmenter = WormSegmenter()
    
    frame_num = 0
    while True:
        # Read frame
        ret, frame = cap.read()
        print(frame.shape)
        if not ret:
            break
            
        # Segment frame
        mask = segmenter.segment_frame(frame)
        
        # Create visualization
        vis = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        vis[mask > 0] = [0, 255, 0]  # Green mask
        overlay = cv2.addWeighted(frame, 0.7, vis, 0.3, 0)
        
        # Save if requested
        if output_dir:
            cv2.imwrite(str(output_dir / f"frame_{frame_num:04d}.png"), overlay)
        
        # Display if requested
        if display:
            aspect_ratio = frame.shape[0] / frame.shape[1]
            desired_width = 1500
            new_height = int(desired_width * aspect_ratio)
            cv2.namedWindow('Segmentation', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Segmentation', desired_width, new_height)  # Adjust these numbers as needed
            cv2.imshow('Segmentation', overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        frame_num += 1
    
    # Cleanup
    cap.release()
    if display:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    video_path = "BZ33C_Chip2A_Worm06.avi"  # Replace with your video path
    output_dir = "output"          # Replace with desired output directory
    
    try:
        process_video(video_path, output_dir, display=True)
    except Exception as e:
        print(f"Error processing video: {e}")
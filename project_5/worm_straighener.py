import numpy as np
import cv2
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
import matplotlib.pyplot as plt

class WormStraightener:
    def __init__(self, num_control_points=5):
        self.num_control_points = num_control_points
        self.backbone = None
        self.control_points = None
        
    def initialize_control_points(self, mask):
        """Initialize control points using MST method."""
        # Randomly select points from the worm body
        y, x = np.where(mask > 0)
        indices = np.random.choice(len(x), min(100, len(x)), replace=False)
        points = np.column_stack((x[indices], y[indices]))
        
        # Calculate pairwise distances
        distances = squareform(pdist(points))
        
        # Get MST
        mst = minimum_spanning_tree(distances).toarray()
        
        # Find the diameter (longest path) in the MST
        max_dist = 0
        start_idx = end_idx = 0
        
        for i in range(len(points)):
            distances = np.zeros(len(points))
            visited = np.zeros(len(points), dtype=bool)
            self._find_longest_path(mst, i, distances, visited)
            max_i = np.argmax(distances)
            if distances[max_i] > max_dist:
                max_dist = distances[max_i]
                start_idx = i
                end_idx = max_i
        
        # Get the points along the diameter
        path = self._get_path(mst, start_idx, end_idx)
        self.control_points = points[path]
        
        return self.control_points
    
    def _find_longest_path(self, mst, start, distances, visited):
        """Helper function to find longest path in MST."""
        visited[start] = True
        neighbors = np.where(mst[start] > 0)[0]
        
        for neighbor in neighbors:
            if not visited[neighbor]:
                distances[neighbor] = distances[start] + mst[start, neighbor]
                self._find_longest_path(mst, neighbor, distances, visited)
    
    def _get_path(self, mst, start, end):
        """Get path between two points in MST."""
        path = [start]
        current = start
        visited = set([start])
        
        while current != end:
            neighbors = np.where(mst[current] > 0)[0]
            next_point = None
            min_remaining = float('inf')
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    distances = np.zeros(len(mst))
                    visited_temp = np.zeros(len(mst), dtype=bool)
                    self._find_longest_path(mst, neighbor, distances, visited_temp)
                    if distances[end] < min_remaining:
                        min_remaining = distances[end]
                        next_point = neighbor
            
            if next_point is None:
                break
                
            path.append(next_point)
            visited.add(next_point)
            current = next_point
            
        return path
    
    def refine_backbone(self, mask):
        """Refine control points based on boundary distances."""
        # Find worm boundary
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_NONE)
        boundary = contours[0].squeeze()
        
        # Refine each control point
        for i in range(len(self.control_points)):
            # Find closest boundary points
            distances = np.sqrt(np.sum((boundary - self.control_points[i])**2, axis=1))
            sorted_indices = np.argsort(distances)
            
            # Move point to midway between boundary points
            self.control_points[i] = np.mean(boundary[sorted_indices[:2]], axis=0)
    
    def generate_spine(self):
        """Generate smooth backbone using cubic spline."""
        # Sort control points by x-coordinate
        sorted_indices = np.argsort(self.control_points[:, 0])
        sorted_points = self.control_points[sorted_indices]
        
        # Fit cubic spline
        t = np.arange(len(sorted_points))
        cs_x = CubicSpline(t, sorted_points[:, 0])
        cs_y = CubicSpline(t, sorted_points[:, 1])
        
        # Generate smooth backbone
        t_fine = np.linspace(0, len(sorted_points)-1, 1000)
        self.backbone = np.column_stack((cs_x(t_fine), cs_y(t_fine)))
        
        return self.backbone
    
    def straighten(self, mask):
        """Main function to straighten the worm."""
        # Initialize control points
        self.initialize_control_points(mask)
        
        # Refine backbone
        self.refine_backbone(mask)
        
        # Generate smooth backbone
        self.generate_spine()
        
        # Calculate worm width
        width = self._estimate_worm_width(mask)
        
        # Create straightened image
        straightened = self._create_straightened_image(mask, width)
        
        return straightened
    
    def _estimate_worm_width(self, mask):
        """Estimate average worm width."""
        return int(np.sum(mask > 0) / len(self.backbone) * 1.5)
    
    def _create_straightened_image(self, mask, width):
        """Create straightened image by sampling perpendicular to backbone."""
        # Initialize output image
        length = len(self.backbone)
        straightened = np.zeros((width, length), dtype=np.uint8)
        
        # For each point along backbone
        for i in range(length-1):
            # Calculate perpendicular direction
            dx = self.backbone[i+1][0] - self.backbone[i][0]
            dy = self.backbone[i+1][1] - self.backbone[i][1]
            perp_dx = -dy
            perp_dy = dx
            norm = np.sqrt(perp_dx**2 + perp_dy**2)
            perp_dx /= norm
            perp_dy /= norm
            
            # Sample along perpendicular line
            for j in range(width):
                offset = j - width//2
                x = int(self.backbone[i][0] + offset*perp_dx)
                y = int(self.backbone[i][1] + offset*perp_dy)
                
                if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                    straightened[j, i] = mask[y, x]
        
        return straightened

def visualize_results(mask, straightened, backbone=None):
    """Visualize original and straightened worm."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Original mask with backbone
    ax1.imshow(mask, cmap='gray')
    if backbone is not None:
        ax1.plot(backbone[:, 0], backbone[:, 1], 'r-', linewidth=2)
    ax1.set_title('Original Mask with Backbone')
    ax1.axis('off')
    
    # Straightened worm
    ax2.imshow(straightened, cmap='gray')
    ax2.set_title('Straightened Worm')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()



def main():
    # Get input mask path
    input_path = "initial_mask1.png"  # Replace with your mask image path
    
    # Read the binary mask
    mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask from {input_path}")
    
    # Ensure binary
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Create straightener and process mask
    straightener = WormStraightener()
    straightened = straightener.straighten(mask)
    
    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.imshow(mask, cmap='gray')
    if straightener.backbone is not None:
        ax1.plot(straightener.backbone[:, 0], straightener.backbone[:, 1], 'r-', linewidth=2)
    ax1.set_title('Original Mask with Backbone')
    ax1.axis('off')
    
    ax2.imshow(straightened, cmap='gray')
    ax2.set_title('Straightened Worm')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Optionally save the straightened result
    output_path = "straightened_worm.png"
    cv2.imwrite(output_path, straightened)

if __name__ == "__main__":
    main()
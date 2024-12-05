
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.spatial.distance import pdist, squareform
# from scipy.sparse.csgraph import minimum_spanning_tree

# def load_and_verify_mask(image_path):
#     """Previous load_and_verify_mask function remains the same"""
#     mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if mask is None:
#         raise ValueError(f"Could not read image from {image_path}")
    
#     _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
#     return mask

# def find_longest_path(mst, points):
#     """
#     Find the longest path in the MST using two-pass approach.
#     Args:
#         mst: Minimum spanning tree adjacency matrix
#         points: Array of point coordinates
#     Returns:
#         path_indices: Indices of points forming the longest path
#     """
#     n_points = len(points)
    
#     def bfs_farthest_point(start):
#         """Find farthest point from start using BFS."""
#         distances = np.full(n_points, np.inf)
#         distances[start] = 0
#         parents = np.full(n_points, -1)
#         queue = [start]
        
#         while queue:
#             current = queue.pop(0)
#             neighbors = np.where(mst[current] > 0)[0]
            
#             for neighbor in neighbors:
#                 if distances[neighbor] == np.inf:
#                     distances[neighbor] = distances[current] + mst[current, neighbor]
#                     parents[neighbor] = current
#                     queue.append(neighbor)
        
#         # Find farthest point
#         end_point = np.argmax(distances)
        
#         # Reconstruct path
#         path = []
#         current = end_point
#         while current != -1:
#             path.append(current)
#             current = parents[current]
            
#         return end_point, path[::-1], distances[end_point]
    
#     # First pass: Find farthest point from arbitrary start (use point 0)
#     end1, _, _ = bfs_farthest_point(0)
    
#     # Second pass: Find farthest point from end1
#     _, final_path, max_dist = bfs_farthest_point(end1)
    
#     return final_path

# def process_and_visualize(mask, num_samples=50):
#     """
#     Process mask and visualize results.
#     """
#     # Get coordinates of worm pixels
#     y, x = np.where(mask > 0)
    
#     # Randomly sample points
#     indices = np.random.choice(len(x), min(num_samples, len(x)), replace=False)
#     points = np.column_stack((x[indices], y[indices]))
    
#     # Calculate MST
#     distances = squareform(pdist(points))
#     mst = minimum_spanning_tree(distances).toarray()
    
#     # Find longest path
#     path_indices = find_longest_path(mst, points)
#     path_points = points[path_indices]
    
#     # Visualize
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
#     # Original MST
#     ax1.imshow(mask, cmap='gray')
#     for i in range(len(points)):
#         for j in range(i+1, len(points)):
#             if mst[i, j] > 0:
#                 ax1.plot([points[i, 0], points[j, 0]], 
#                         [points[i, 1], points[j, 1]], 
#                         'r-', alpha=0.5)
#     ax1.scatter(points[:, 0], points[:, 1], c='r', s=30)
#     ax1.set_title('Minimum Spanning Tree')
    
#     # Longest path
#     ax2.imshow(mask, cmap='gray')
#     for i in range(len(path_points)-1):
#         ax2.plot([path_points[i, 0], path_points[i+1, 0]], 
#                 [path_points[i, 1], path_points[i+1, 1]], 
#                 'g-', linewidth=2)
#     ax2.scatter(path_points[:, 0], path_points[:, 1], c='g', s=50)
#     ax2.scatter(path_points[0, 0], path_points[0, 1], c='r', s=100, label='Start')
#     ax2.scatter(path_points[-1, 0], path_points[-1, 1], c='b', s=100, label='End')
#     ax2.set_title('Longest Path (Backbone)')
#     ax2.legend()
    
#     plt.tight_layout()
#     plt.show()
    
#     return points, mst, path_points

# if __name__ == "__main__":
#     # Replace with your mask image path
#     image_path = "initial_mask1.png"
    
#     try:
#         # Load mask
#         mask = load_and_verify_mask(image_path)
        
#         # Process and visualize
#         points, mst, path_points = process_and_visualize(mask, num_samples=50)
        
#     except Exception as e:
#         print(f"Error: {e}")



import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.spatial.distance import cdist

def find_worm_endpoints(mask):
    """
    Find the true endpoints of the worm using distance transform and contour analysis.
    Args:
        mask: Binary mask of the worm
    Returns:
        tuple: Coordinates of (head, tail)
    """
    # Get contour points
    contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                 cv2.RETR_EXTERNAL, 
                                 cv2.CHAIN_APPROX_NONE)
    contour = contours[0].squeeze()
    
    # Find points farthest apart on contour
    distances = cdist(contour, contour)
    i, j = np.unravel_index(np.argmax(distances), distances.shape)
    
    # Get these two extreme points
    point1 = contour[i]
    point2 = contour[j]
    
    # The tail is usually more pointed (has smaller area near it)
    kernel = np.ones((5,5), np.uint8)
    areas = []
    for point in [point1, point2]:
        # Create small mask around point
        temp_mask = np.zeros_like(mask)
        cv2.circle(temp_mask, tuple(point), 10, 255, -1)
        # Calculate area of worm in this region
        region = cv2.bitwise_and(mask, temp_mask)
        areas.append(np.sum(region > 0))
    
    # Point with smaller area is likely the tail
    if areas[0] < areas[1]:
        tail, head = point1, point2
    else:
        tail, head = point2, point1
        
    return head, tail

def get_skeleton_path(skeleton, start, end):
    """
    Get ordered points along skeleton between start and end points.
    Args:
        skeleton: Binary skeleton image
        start: Starting point coordinates
        end: Ending point coordinates
    Returns:
        ordered_points: List of points from start to end
    """
    # Find nearest skeleton points to start and end
    skel_points = np.column_stack(np.where(skeleton > 0))
    skel_points = np.flip(skel_points, axis=1)  # Switch to (x,y)
    
    # Find closest skeleton points to start and end
    start_idx = np.argmin(np.sum((skel_points - start)**2, axis=1))
    end_idx = np.argmin(np.sum((skel_points - end)**2, axis=1))
    
    start_point = tuple(skel_points[start_idx])
    end_point = tuple(skel_points[end_idx])
    
    # Get path using distance from start point
    visited = np.zeros_like(skeleton, dtype=bool)
    ordered_points = [start_point]
    current = start_point
    
    while current != end_point:
        y, x = current[1], current[0]
        patch = skeleton[max(0, y-1):min(y+2, skeleton.shape[0]),
                        max(0, x-1):min(x+2, skeleton.shape[1])]
        patch_points = np.column_stack(np.where(patch > 0))
        patch_points = patch_points + [max(0, y-1), max(0, x-1)]
        
        min_dist = float('inf')
        next_point = None
        
        for py, px in patch_points:
            if not visited[py, px]:
                dist = np.sqrt((px - end_point[0])**2 + (py - end_point[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    next_point = (px, py)
        
        if next_point is None:
            break
            
        visited[next_point[1], next_point[0]] = True
        ordered_points.append(next_point)
        current = next_point
    
    return ordered_points

def process_and_visualize(mask):
    """
    Process mask using skeletonization and visualize results.
    """
    # Get skeleton
    skeleton = skeletonize(mask > 0)
    skeleton = skeleton.astype(np.uint8) * 255
    
    # Find true endpoints
    head, tail = find_worm_endpoints(mask)
    
    # Get ordered points along skeleton
    skeleton_points = get_skeleton_path(skeleton, tail, head)
    skeleton_points = np.array(skeleton_points)
    
    # Visualize
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='gray')
    plt.plot(skeleton_points[:, 0], skeleton_points[:, 1], 'g-', linewidth=2)
    plt.scatter(tail[0], tail[1], c='r', s=100, label='Tail')
    plt.scatter(head[0], head[1], c='b', s=100, label='Head')
    plt.title('Skeleton Backbone')
    plt.legend()
    plt.show()
    
    return skeleton_points

if __name__ == "__main__":
    # Replace with your mask image path
    image_path = "initial_mask1.png"
    
    try:
        # Load mask
        mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Process and visualize
        skeleton_points = process_and_visualize(mask)
        
    except Exception as e:
        print(f"Error: {e}")
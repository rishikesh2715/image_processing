import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree
from skimage.morphology import skeletonize
from scipy.spatial.distance import cdist
import networkx as nx

def extract_worm_boundary(binary_img):
    # Make sure image is binary with values 0 and 255
    binary_img = (binary_img > 128).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Get the largest contour (should be the worm)
    worm_contour = max(contours, key=cv2.contourArea)
    
    # Create an empty image to draw the boundary
    boundary_img = np.zeros_like(binary_img)
    cv2.drawContours(boundary_img, [worm_contour], -1, 255, 1)
    
    # Create filled region image
    filled_img = np.zeros_like(binary_img)
    cv2.drawContours(filled_img, [worm_contour], -1, 255, -1)  # -1 for filling
    
    # Get boundary points as a list of [x,y] coordinates
    boundary_points = worm_contour.squeeze()
    
    return boundary_points, boundary_img, filled_img

def visualize_boundary_points(binary_img, boundary_points):
    """
    Visualize the boundary points to verify they're correct
    """
    vis_img = cv2.cvtColor(binary_img.copy(), cv2.COLOR_GRAY2BGR)
    
    # Draw each boundary point as a small red dot
    for point in boundary_points:
        cv2.circle(vis_img, tuple(point.astype(int)), 1, (0,0,255), -1)
    
    return vis_img

def test_boundary_extraction(image_path):
    # Read binary image
    binary_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Extract boundary and filled region
    boundary_points, boundary_img, filled_img = extract_worm_boundary(binary_img)
    
    # Visualize boundary points
    points_vis = visualize_boundary_points(boundary_img, boundary_points)
     
    return boundary_points, boundary_img, filled_img



##############################################################
def initialize_control_points(filled_img, num_points=200):
    """
    Initialize control points with constraints to ensure the path stays within the worm body
    """
    # Get worm pixels
    y_coords, x_coords = np.where(filled_img > 0)
    points = np.column_stack((x_coords, y_coords))
    
    # Create distance transform
    dist_transform = cv2.distanceTransform(filled_img, cv2.DIST_L2, 5)
    
    # Find worm tips using distance from centroid
    centroid = np.mean(points, axis=0)
    distances_from_centroid = np.sqrt(np.sum((points - centroid)**2, axis=1))
    
    # Get boundary points
    kernel = np.ones((3,3), np.uint8)
    boundary = cv2.erode(filled_img, kernel, iterations=1)
    boundary = filled_img - boundary
    boundary_points = np.column_stack(np.where(boundary > 0))
    
    # Find tip candidates (points far from centroid and on boundary)
    boundary_mask = np.zeros_like(filled_img)
    boundary_mask[boundary_points[:,1], boundary_points[:,0]] = 1
    tip_candidates = []
    
    for i in np.argsort(distances_from_centroid)[::-1]:
        point = points[i]
        if boundary_mask[int(point[1]), int(point[0])] and len(tip_candidates) < 10:
            tip_candidates.append(point)
    
    # Find the two most distant tips that have a valid path through the worm
    best_tips = None
    max_path_score = -float('inf')
    
    for i in range(len(tip_candidates)):
        for j in range(i+1, len(tip_candidates)):
            tip1, tip2 = tip_candidates[i], tip_candidates[j]
            
            # Check if path between tips stays in worm body
            line_points = np.linspace(tip1, tip2, 100)
            path_points = line_points.astype(int)
            path_valid = np.all(filled_img[path_points[:,1], path_points[:,0]])
            
            if path_valid:
                # Score based on distance and path quality
                tip_distance = np.linalg.norm(tip1 - tip2)
                path_score = np.mean([dist_transform[y,x] for x,y in path_points])
                total_score = tip_distance * path_score
                
                if total_score > max_path_score:
                    max_path_score = total_score
                    best_tips = (tip1, tip2)
    
    if best_tips is None:
        # Fallback: use the most distant tips
        tip1, tip2 = tip_candidates[0], tip_candidates[1]
        best_tips = (tip1, tip2)
    
    # Sample internal points with higher density along the centerline
    remaining_points = num_points - 2
    
    # Create sampling probability based on distance transform
    prob = dist_transform.ravel()
    prob = prob / np.sum(prob)
    
    # Sample points weighted by distance transform
    flat_indices = np.random.choice(
        len(prob),
        size=remaining_points,
        p=prob,
        replace=False
    )
    rows = flat_indices // dist_transform.shape[1]
    cols = flat_indices % dist_transform.shape[1]
    internal_points = np.column_stack([cols, rows])
    
    # Combine tips and internal points
    all_points = np.vstack([
        best_tips[0],
        best_tips[1],
        internal_points
    ])
    
    # Build graph with constraints
    G = nx.Graph()
    for i in range(len(all_points)):
        G.add_node(i)
        for j in range(i+1, len(all_points)):
            # Check if connection stays within worm body
            p1, p2 = all_points[i], all_points[j]
            line_points = np.linspace(p1, p2, 20).astype(int)
            if np.all(filled_img[line_points[:,1], line_points[:,0]]):
                dist = np.linalg.norm(p1 - p2)
                # Penalize long connections
                if dist > 20:
                    dist *= 2
                G.add_edge(i, j, weight=dist)
    
    # Find path between tips (indices 0 and 1)
    try:
        path = nx.shortest_path(G, 0, 1, weight='weight')
        path_points = all_points[path]
    except nx.NetworkXNoPath:
        # Fallback to MST if direct path fails
        MST = nx.minimum_spanning_tree(G)
        longest_path = max(nx.single_source_shortest_path(MST, 0).values(), key=len)
        path_points = all_points[longest_path]
    
    return path_points, all_points


def divide_boundary(boundary_points, first_point, last_point):
    """Divides boundary into left and right sides"""
    # Find closest boundary points to endpoints
    dist_first = np.sum((boundary_points - first_point)**2, axis=1)
    dist_last = np.sum((boundary_points - last_point)**2, axis=1)
    idx_first = np.argmin(dist_first)
    idx_last = np.argmin(dist_last)
    
    # Split boundary into two parts
    if idx_first < idx_last:
        left = boundary_points[idx_first:idx_last+1]
        right = np.vstack([boundary_points[idx_last:], boundary_points[:idx_first+1]])
    else:
        left = boundary_points[idx_last:idx_first+1]
        right = np.vstack([boundary_points[idx_first:], boundary_points[:idx_last+1]])
    
    return left, right

def refine_backbone(control_points, boundary_points):
    """Refines backbone control points using boundary information"""
    cp = control_points.copy()
    # Get left and right boundaries
    left_boundary, right_boundary = divide_boundary(boundary_points, cp[0], cp[-1])
    
    # Build KD-trees for efficient nearest neighbor search
    tree_left = KDTree(left_boundary)
    tree_right = KDTree(right_boundary)
    
    # Refine internal control points
    for _ in range(5):
        new_cp = []
        for i in range(1, len(cp)-1):
            # Find nearest boundary points
            _, idx_left = tree_left.query(cp[i], k=1)
            _, idx_right = tree_right.query(cp[i], k=1)
            
            # Calculate midpoint
            left_pt = left_boundary[idx_left]
            right_pt = right_boundary[idx_right]
            midpoint = (left_pt + right_pt) / 2
            
            # Add smoothness constraint
            smooth_pos = (cp[i-1] + cp[i+1]) / 2
            new_pos = 0.7 * midpoint + 0.3 * smooth_pos
            new_cp.append(new_pos)
        
        # Update control points
        cp[1:-1] = new_cp
    
    return cp

def generate_cutting_planes(control_points, spacing=1):
    control_points = control_points.astype(np.float64)
    
    # Calculate approximate arc length
    arc_length = np.sum(np.sqrt(np.sum(np.diff(control_points, axis=0)**2, axis=1)))
    n_points = int(arc_length)  # One point per pixel of arc length
    
    k = min(3, len(control_points) - 1)
    try:
        tck, u = splprep([control_points[:,0], control_points[:,1]], s=0, k=k)
        u_new = np.linspace(0, 1, n_points)
        backbone_points = np.column_stack(splev(u_new, tck))
        
        dx, dy = splev(u_new, tck, der=1)
        tangents = np.column_stack([dx, dy])
        norms = np.linalg.norm(tangents, axis=1)[:, np.newaxis]
        norms[norms == 0] = 1
        tangents = tangents / norms
        normals = np.column_stack([-tangents[:,1], tangents[:,0]])
        
        return backbone_points, normals
    except:
        # Fallback to linear interpolation
        distances = np.cumsum(np.sqrt(np.sum(np.diff(control_points, axis=0)**2, axis=1)))
        distances = np.insert(distances, 0, 0)
        points_spaced = np.interp(np.linspace(0, distances[-1], n_points), 
                                distances, range(len(control_points)))
        
        backbone_points = np.vstack([np.interp(points_spaced, range(len(control_points)), 
                                             control_points[:,i]) for i in range(2)]).T
        
        tangents = np.diff(backbone_points, axis=0, prepend=backbone_points[:1])
        norms = np.linalg.norm(tangents, axis=1)[:, np.newaxis]
        norms[norms == 0] = 1
        tangents = tangents / norms
        normals = np.column_stack([-tangents[:,1], tangents[:,0]])
        
        return backbone_points, normals

def straighten_worm(binary_mask, control_points, width=100):
    backbone_points, normals = generate_cutting_planes(control_points)
    straightened = np.zeros((len(backbone_points), width), dtype=np.uint8)
    half_width = width // 2
    
    for i, (point, normal) in enumerate(zip(backbone_points, normals)):
        for j in range(-half_width, half_width):
            sample_point = point + j * normal
            x, y = np.clip(sample_point.astype(int), [0, 0], 
                          [binary_mask.shape[1]-1, binary_mask.shape[0]-1])
            straightened[i, j + half_width] = binary_mask[y, x]
    
    return straightened

def visualize_backbone(binary_mask, control_points):
    """Visualizes backbone on binary mask"""
    vis_img = cv2.cvtColor(binary_mask.copy(), cv2.COLOR_GRAY2BGR)
    
    # Draw control points
    for point in control_points:
        cv2.circle(vis_img, tuple(point.astype(int)), 3, (0,0,255), -1)
    
    # Draw spline through points
    backbone_points, _ = generate_cutting_planes(control_points)
    for i in range(len(backbone_points)-1):
        pt1 = tuple(backbone_points[i].astype(int))
        pt2 = tuple(backbone_points[i+1].astype(int))
        cv2.line(vis_img, pt1, pt2, (0,255,0), 1)
        
    return vis_img

def segment_worm(video_path, frame_to_analyze_index):
    half_window = 50

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=25, detectShadows=False)

    # Train using frames around the frame of interest
    start_frame = max(frame_to_analyze_index - half_window, 0)
    end_frame = min(frame_to_analyze_index + half_window, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for f_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        fgbg.apply(frame)
    cap.release()

    # Read the frame of interest
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_analyze_index)
    ret, frame_to_analyze = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame at index {frame_to_analyze_index}.")
        exit(1)

    # Get the foreground mask for the frame of interest
    fgmask = fgbg.apply(frame_to_analyze, learningRate=0)

    # Convert to boolean mask for remove_small_objects
    worm_bool = fgmask > 0

    # Set a min_size that keeps the worm but removes small pixels.
    # Adjust this value based on the worm size and noise level.
    min_size = 55
    worm_cleaned_bool = remove_small_objects(worm_bool, min_size=min_size)

    # Convert back to uint8 mask
    worm_cleaned = (worm_cleaned_bool.astype(np.uint8) * 255)

    # Now apply closing to fill holes inside the worm
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (75,75))
    worm_closed = cv2.morphologyEx(worm_cleaned, cv2.MORPH_CLOSE, kernel_close)

    # Display results
    plt.figure(figsize=(12,5))
    plt.subplot(2,2,1)
    plt.title(f"Frame, {frame_to_analyze_index}")
    plt.imshow(frame_to_analyze, cmap='gray')

    plt.subplot(2,2,2)
    plt.title("Raw Foreground Mask")
    plt.imshow(fgmask, cmap='gray')

    plt.subplot(2,2,3)
    plt.title("After Removing Small Objects")
    plt.imshow(worm_cleaned, cmap='gray')

    plt.subplot(2,2,4)
    plt.title("After Closing Worm")
    plt.imshow(worm_closed, cmap='gray')

    plt.tight_layout()
    plt.show()

    # print(frame_to_analyze.shape)

    return frame_to_analyze, worm_closed

def visualize_processing_steps(video_path, num_points=500):
   
    # Read images
    original_img, binary_img = segment_worm(video_path, frame_to_analyze_index=14)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    # Get all processing steps
    boundary_points, boundary_img, filled_img = extract_worm_boundary(binary_img)
    sampled_points, all_sampled_points = initialize_control_points(filled_img, num_points=num_points)
    refined_cp = refine_backbone(sampled_points, boundary_points)
    backbone_points, normals = generate_cutting_planes(refined_cp)
    
    # 1. Original Images
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.imshow(binary_img, cmap='gray')
    plt.title('Binary Mask')
    plt.subplot(122)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original Image')
    plt.suptitle('Step 1: Input Images', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 2. Boundary Detection
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    boundary_vis_binary = cv2.cvtColor(binary_img.copy(), cv2.COLOR_GRAY2BGR)
    for point in boundary_points:
        cv2.circle(boundary_vis_binary, tuple(point.astype(int)), 1, (0,0,255), -1)
    plt.imshow(cv2.cvtColor(boundary_vis_binary, cv2.COLOR_BGR2RGB))
    plt.title('Boundary Points (Binary)')
    
    plt.subplot(122)
    boundary_vis_original = cv2.cvtColor(original_img.copy(), cv2.COLOR_GRAY2BGR)
    for point in boundary_points:
        cv2.circle(boundary_vis_original, tuple(point.astype(int)), 1, (0,0,255), -1)
    plt.imshow(cv2.cvtColor(boundary_vis_original, cv2.COLOR_BGR2RGB))
    plt.title('Boundary Points (Original)')
    plt.suptitle('Step 2: Boundary Detection', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 3. Filled Mask
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.imshow(filled_img, cmap='gray')
    plt.title('Filled Mask (Binary)')
    plt.subplot(122)
    filled_vis_original = cv2.bitwise_and(original_img, original_img, mask=filled_img)
    plt.imshow(filled_vis_original, cmap='gray')
    plt.title('Filled Mask (Original)')
    plt.suptitle('Step 3: Region Filling', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 4. Initial Path
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    init_path_binary = cv2.cvtColor(filled_img.copy(), cv2.COLOR_GRAY2BGR)
    for point in all_sampled_points:
        cv2.circle(init_path_binary, tuple(point.astype(int)), 2, (255,0,0), -1)
    for i in range(len(sampled_points)-1):
        pt1 = tuple(sampled_points[i].astype(int))
        pt2 = tuple(sampled_points[i+1].astype(int))
        cv2.circle(init_path_binary, pt1, 4, (0,0,255), -1)
        cv2.line(init_path_binary, pt1, pt2, (0,255,0), 2)
    plt.imshow(cv2.cvtColor(init_path_binary, cv2.COLOR_BGR2RGB))
    plt.title('Initial Path (Binary)')
    
    plt.subplot(122)
    init_path_original = cv2.cvtColor(original_img.copy(), cv2.COLOR_GRAY2BGR)
    for point in all_sampled_points:
        cv2.circle(init_path_original, tuple(point.astype(int)), 2, (255,0,0), -1)
    for i in range(len(sampled_points)-1):
        pt1 = tuple(sampled_points[i].astype(int))
        pt2 = tuple(sampled_points[i+1].astype(int))
        cv2.circle(init_path_original, pt1, 4, (0,0,255), -1)
        cv2.line(init_path_original, pt1, pt2, (0,255,0), 2)
    plt.imshow(cv2.cvtColor(init_path_original, cv2.COLOR_BGR2RGB))
    plt.title('Initial Path (Original)')
    plt.suptitle('Step 4: Initial Backbone Path', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 5. Boundary Division
    plt.figure(figsize=(12, 5))
    left_boundary, right_boundary = divide_boundary(boundary_points, refined_cp[0], refined_cp[-1])
    
    plt.subplot(121)
    boundary_div_binary = cv2.cvtColor(filled_img.copy(), cv2.COLOR_GRAY2BGR)
    for point in left_boundary:
        cv2.circle(boundary_div_binary, tuple(point.astype(int)), 1, (255,0,0), -1)
    for point in right_boundary:
        cv2.circle(boundary_div_binary, tuple(point.astype(int)), 1, (0,255,0), -1)
    plt.imshow(cv2.cvtColor(boundary_div_binary, cv2.COLOR_BGR2RGB))
    plt.title('Boundary Division (Binary)')
    
    plt.subplot(122)
    boundary_div_original = cv2.cvtColor(original_img.copy(), cv2.COLOR_GRAY2BGR)
    for point in left_boundary:
        cv2.circle(boundary_div_original, tuple(point.astype(int)), 1, (255,0,0), -1)
    for point in right_boundary:
        cv2.circle(boundary_div_original, tuple(point.astype(int)), 1, (0,255,0), -1)
    plt.imshow(cv2.cvtColor(boundary_div_original, cv2.COLOR_BGR2RGB))
    plt.title('Boundary Division (Original)')
    plt.suptitle('Step 5: Boundary Division', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 6. Refined Backbone
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    refined_binary = cv2.cvtColor(filled_img.copy(), cv2.COLOR_GRAY2BGR)
    for point in refined_cp:
        cv2.circle(refined_binary, tuple(point.astype(int)), 3, (0,0,255), -1)
    for i in range(len(backbone_points)-1):
        pt1 = tuple(backbone_points[i].astype(int))
        pt2 = tuple(backbone_points[i+1].astype(int))
        cv2.line(refined_binary, pt1, pt2, (0,255,0), 1)
    plt.imshow(cv2.cvtColor(refined_binary, cv2.COLOR_BGR2RGB))
    plt.title('Refined Backbone (Binary)')
    
    plt.subplot(122)
    refined_original = cv2.cvtColor(original_img.copy(), cv2.COLOR_GRAY2BGR)
    for point in refined_cp:
        cv2.circle(refined_original, tuple(point.astype(int)), 3, (0,0,255), -1)
    for i in range(len(backbone_points)-1):
        pt1 = tuple(backbone_points[i].astype(int))
        pt2 = tuple(backbone_points[i+1].astype(int))
        cv2.line(refined_original, pt1, pt2, (0,255,0), 1)
    plt.imshow(cv2.cvtColor(refined_original, cv2.COLOR_BGR2RGB))
    plt.title('Refined Backbone (Original)')
    plt.suptitle('Step 6: Refined Backbone', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 7. Cutting Planes
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    cutting_binary = cv2.cvtColor(filled_img.copy(), cv2.COLOR_GRAY2BGR)
    for point, normal in zip(backbone_points[::20], normals[::20]):
        pt1 = tuple((point - normal * 50).astype(int))
        pt2 = tuple((point + normal * 50).astype(int))
        cv2.line(cutting_binary, pt1, pt2, (0,0,255), 1)
    plt.imshow(cv2.cvtColor(cutting_binary, cv2.COLOR_BGR2RGB))
    plt.title('Cutting Planes (Binary)')
    
    plt.subplot(122)
    cutting_original = cv2.cvtColor(original_img.copy(), cv2.COLOR_GRAY2BGR)
    for point, normal in zip(backbone_points[::20], normals[::20]):
        pt1 = tuple((point - normal * 50).astype(int))
        pt2 = tuple((point + normal * 50).astype(int))
        cv2.line(cutting_original, pt1, pt2, (0,0,255), 1)
    plt.imshow(cv2.cvtColor(cutting_original, cv2.COLOR_BGR2RGB))
    plt.title('Cutting Planes (Original)')
    plt.suptitle('Step 7: Cutting Planes', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 8. Straightened Results
    plt.figure(figsize=(12, 5))


    # Mask straightening
    plt.subplot(121)
    straightened_mask = straighten_worm(filled_img, refined_cp, width=200)
    plt.imshow(straightened_mask, cmap='gray')
    plt.title('Straightened Mask')

    # Masked original image straightening
    plt.subplot(122)
    # Create masked original image by setting non-worm pixels to white (255)
    masked_original = original_img.copy()
    # masked_original[filled_img == 0] = 127  # Set background to white
    straightened_original = straighten_worm(masked_original, refined_cp, width=200)
    plt.imshow(straightened_original, cmap='gray')
    plt.title('Straightened Worm (Masked Original)')

    plt.suptitle('Step 8: Final Straightened Results', fontsize=14)
    plt.tight_layout()
    plt.show()

# Usage:
visualize_processing_steps("BZ33C_Chip1D_Worm27.avi")




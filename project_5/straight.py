import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree
from skimage.morphology import skeletonize  # For skeleton-based endpoint detection
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
def initialize_control_points(filled_img, num_points=1000):
    import networkx as nx
    
    # Find worm endpoints
    y_coords, x_coords = np.where(filled_img > 0)
    left_x = min(x_coords)
    right_x = max(x_coords)
    top_y = min(y_coords)
    bottom_y = max(y_coords)
    
    # Create mask for endpoints
    endpoints = []
    for x, y in zip(x_coords, y_coords):
        # Check if point is at extremes
        if (x == left_x or x == right_x or y == top_y or y == bottom_y):
            endpoints.append([x, y])
    
    # Convert to array and cluster nearby points
    endpoints = np.array(endpoints)
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=10, min_samples=1).fit(endpoints)
    cluster_centers = []
    for label in set(clustering.labels_):
        cluster_points = endpoints[clustering.labels_ == label]
        cluster_centers.append(np.mean(cluster_points, axis=0))
    
    # Sample internal points
    points = np.column_stack((x_coords, y_coords))
    internal_points = num_points - len(cluster_centers)
    indices = np.random.choice(len(points), internal_points, replace=False)
    sampled_points = np.vstack([points[indices], cluster_centers])
    
    # Build graph and find path
    G = nx.Graph()
    for i in range(len(sampled_points)):
        G.add_node(i)
        for j in range(i+1, len(sampled_points)):
            dist = np.linalg.norm(sampled_points[i] - sampled_points[j])
            G.add_edge(i, j, weight=dist)
    
    MST = nx.minimum_spanning_tree(G)
    paths = nx.single_source_shortest_path(MST, 0)
    end1 = max(paths.items(), key=lambda x: len(x[1]))[0]
    
    paths = nx.single_source_shortest_path(MST, end1)
    end2, longest_path = max(paths.items(), key=lambda x: len(x[1]))
    
    path_points = sampled_points[longest_path]
    
    return path_points, sampled_points


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


def visualize_pipeline(binary_path, original_path, num_points=200):
    # Read both images
    binary_img = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
    original_img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    
    
    boundary_points, boundary_img, filled_img = extract_worm_boundary(binary_img)
    
    
    sampled_points, all_sampled_points = initialize_control_points(filled_img, num_points=num_points)
    
    
    refined_cp = refine_backbone(sampled_points, boundary_points)
    
    
    backbone_points, normals = generate_cutting_planes(refined_cp)
    
    
    fig = plt.figure(figsize=(20, 10))
    
    
    plt.subplot(251)
    plt.imshow(filled_img, cmap='gray')
    plt.title('Binary Mask')
    
    plt.subplot(253)
    debug_img_binary = cv2.cvtColor(filled_img.copy(), cv2.COLOR_GRAY2BGR)
    
    for point in all_sampled_points:
        cv2.circle(debug_img_binary, tuple(point.astype(int)), 2, (255,0,0), -1)
    for i in range(len(sampled_points)-1):
        pt1 = tuple(sampled_points[i].astype(int))
        pt2 = tuple(sampled_points[i+1].astype(int))
        cv2.circle(debug_img_binary, pt1, 4, (0,0,255), -1)
        cv2.line(debug_img_binary, pt1, pt2, (0,255,0), 2)
    plt.imshow(cv2.cvtColor(debug_img_binary, cv2.COLOR_BGR2RGB))
    plt.title('Initial Path (Binary)')
    
    
    plt.subplot(252)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(254)
    debug_img_original = cv2.cvtColor(original_img.copy(), cv2.COLOR_GRAY2BGR)
    
    for point in all_sampled_points:
        cv2.circle(debug_img_original, tuple(point.astype(int)), 2, (255,0,0), -1)
    for i in range(len(sampled_points)-1):
        pt1 = tuple(sampled_points[i].astype(int))
        pt2 = tuple(sampled_points[i+1].astype(int))
        cv2.circle(debug_img_original, pt1, 4, (0,0,255), -1)
        cv2.line(debug_img_original, pt1, pt2, (0,255,0), 2)
    plt.imshow(cv2.cvtColor(debug_img_original, cv2.COLOR_BGR2RGB))
    plt.title('Initial Path (Original)')
    
    
    plt.subplot(255)
    left_boundary, right_boundary = divide_boundary(boundary_points, refined_cp[0], refined_cp[-1])
    boundary_vis = cv2.cvtColor(original_img.copy(), cv2.COLOR_GRAY2BGR)
    for point in left_boundary:
        cv2.circle(boundary_vis, tuple(point.astype(int)), 1, (255,0,0), -1)
    for point in right_boundary:
        cv2.circle(boundary_vis, tuple(point.astype(int)), 1, (0,255,0), -1)
    plt.imshow(cv2.cvtColor(boundary_vis, cv2.COLOR_BGR2RGB))
    plt.title('Boundary Division (Original)')
    
    
    plt.subplot(257)
    vis_refined = cv2.cvtColor(original_img.copy(), cv2.COLOR_GRAY2BGR)
    for point in refined_cp:
        cv2.circle(vis_refined, tuple(point.astype(int)), 3, (0,0,255), -1)
    for i in range(len(backbone_points)-1):
        pt1 = tuple(backbone_points[i].astype(int))
        pt2 = tuple(backbone_points[i+1].astype(int))
        cv2.line(vis_refined, pt1, pt2, (0,255,0), 1)
    plt.imshow(cv2.cvtColor(vis_refined, cv2.COLOR_BGR2RGB))
    plt.title('Refined Backbone (Original)')
    
    
    plt.subplot(258)
    vis_planes = cv2.cvtColor(original_img.copy(), cv2.COLOR_GRAY2BGR)
    for point, normal in zip(backbone_points[::20], normals[::20]):
        pt1 = tuple((point - normal * 50).astype(int))
        pt2 = tuple((point + normal * 50).astype(int))
        cv2.line(vis_planes, pt1, pt2, (0,0,255), 1)
    plt.imshow(cv2.cvtColor(vis_planes, cv2.COLOR_BGR2RGB))
    plt.title('Cutting Planes (Original)')
    
    
    plt.subplot(2,5,(9,10))
    straightened_original = straighten_worm(original_img, refined_cp, width=200)
    plt.imshow(straightened_original, cmap='gray')
    plt.title('Straightened Worm (Original)')
    
    plt.tight_layout()
    plt.show()

visualize_pipeline("initial_mask2.png", "original_frame.png")




# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# def read_images_from_folder(folder_path):
#     image_paths = list(Path(folder_path).glob("*.jpg"))  # Adjust file extension as needed
#     for image_path in image_paths:
#         if image_path.is_file():
#             print(f"Processing image: {image_path.name}")
#             img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

#             # binarize the image
#             # _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
#             yield img

# def gaussian_blur(img):
#     gaussian_kernel = np.array([[0, 0, 1, 2, 1, 0, 0],
#                                 [0, 3, 13, 22, 13, 3, 0],
#                                 [1, 13, 59, 97, 59, 13, 1],
#                                 [2, 22, 97, 159, 97, 22, 2],
#                                 [1, 13, 59, 97, 59, 13, 1],
#                                 [0, 3, 13, 22, 13, 3, 0],
#                                 [0, 0, 1, 2, 1, 0, 0]], dtype=np.float32) / 1003
#     img_blurred = cv2.filter2D(img, cv2.CV_32F, gaussian_kernel)
#     return img_blurred

# def sobel_kernel(img, threshold_value):
#     gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#     gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
#     grad_x = cv2.filter2D(img, cv2.CV_32F, gx)
#     grad_y = cv2.filter2D(img, cv2.CV_32F, gy)
#     grad_magnitude = np.abs(grad_x) + np.abs(grad_y)
#     threshold = np.max(grad_magnitude) * threshold_value
#     edge_mask = grad_magnitude > threshold
#     grad_direction = np.arctan2(grad_y, grad_x) * 180 / np.pi
#     edge_directions = (grad_direction[edge_mask] - 90.0) % 180
#     hist, bin_edges = np.histogram(edge_directions, bins=np.arange(1, 181, 1))
#     dominant_angle_index = np.argmax(hist)
#     dominant_angle = bin_edges[dominant_angle_index]
#     return edge_mask, dominant_angle

# def rotate_image(img, dominant_angle):
#     rotation_angle = 90 - dominant_angle
#     if rotation_angle < 0:
#         rotation_angle += 180
#     (h, w) = img.shape[:2]
#     center = (w // 2, h // 2)
#     rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
#     rotated_image = cv2.warpAffine(img, rotation_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
#     return rotated_image

# def crop_image(img):
#     crop_img_blurred = gaussian_blur(img)
#     crop_img_blurred = gaussian_blur(crop_img_blurred)
#     crop_img_blurred = gaussian_blur(crop_img_blurred)
#     crop_edge_mask, crop_dominant_angle = sobel_kernel(crop_img_blurred, threshold_value=0.4)
#     edge_coords = np.column_stack(np.where(crop_edge_mask))
#     x_min, y_min = edge_coords.min(axis=0)
#     x_max, y_max = edge_coords.max(axis=0)
#     cropped_img = img[x_min:x_max, y_min:y_max]
#     return cropped_img, crop_edge_mask

# def plot_images(img, cropped_img, edge_mask, crop_edge_mask):
#     plt.figure(figsize=(16, 12))
#     plt.subplot(2,2,1)
#     plt.imshow(img, cmap='gray')
#     plt.title("Original Image")
#     plt.subplot(2,2,2)
#     plt.imshow(cropped_img, cmap='gray')
#     plt.title("Rotated & Cropped Image")
#     plt.subplot(2,2,3)
#     plt.imshow(edge_mask, cmap='gray')
#     plt.title("Edge Mask")
#     plt.subplot(2,2,4)
#     plt.imshow(crop_edge_mask, cmap='gray')
#     plt.title("Cropped Edge Mask")
#     plt.tight_layout()
#     plt.show()

# def main(folder_path):
#     for img in read_images_from_folder(folder_path):
#         blurred_image = gaussian_blur(img)
#         blurred_image = gaussian_blur(blurred_image)
#         blurred_image = gaussian_blur(blurred_image)
#         edge_mask, dominant_angle = sobel_kernel(blurred_image, threshold_value=0.2)
#         rotated_image = rotate_image(blurred_image, dominant_angle)
#         cropped_image, crop_edge_mask = crop_image(rotated_image)
#         plot_images(img, cropped_image, edge_mask, crop_edge_mask)

# if __name__ == "__main__":
#     folder_path = input("Enter the folder path containing images: ")
#     main(folder_path)
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# def read_images_from_folder(folder_path):
#     image_paths = list(Path(folder_path).glob("*.jpg"))  # Adjust file extension as needed
#     for image_path in image_paths:
#         if image_path.is_file():
#             print(f"Processing image: {image_path.name}")
#             img = cv2.imread(str(image_path))
#             yield img

# def preprocess_image(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     return blurred

# def detect_edges(img):
#     edged = cv2.Canny(img, 50, 150)
#     return edged

# def find_card_contour(edged):
#     contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # Assume the largest contour is the card
#     if contours:
#         card_contour = max(contours, key=cv2.contourArea)
#         return card_contour
#     else:
#         return None

# def get_rotated_card(img, contour):
#     # Get the minimum area rectangle
#     rect = cv2.minAreaRect(contour)
#     box = cv2.boxPoints(rect)
#     box = np.int0(box)

#     # Calculate rotation matrix
#     width = int(rect[1][0])
#     height = int(rect[1][1])
#     src_pts = box.astype("float32")
#     dst_pts = np.array([[0, height-1],
#                         [0, 0],
#                         [width-1, 0],
#                         [width-1, height-1]], dtype="float32")
    
#     # Correct the order of points if necessary
#     if width < height:
#         dst_pts = np.array([[0, width-1],
#                             [0, 0],
#                             [height-1, 0],
#                             [height-1, width-1]], dtype="float32")
#         width, height = height, width

#     M = cv2.getPerspectiveTransform(src_pts, dst_pts)
#     warped = cv2.warpPerspective(img, M, (width, height))
#     return warped

# def crop_rank_suit(warped):
#     # Assuming the rank and suit are located at the top-left corner
#     h, w, _ = warped.shape
#     rank_region = warped[0:int(0.2*h), 0:int(0.2*w)]
#     suit_region = warped[int(0.8*h):h, int(0.8*w):w]
#     return rank_region, suit_region

# def plot_images(original, edged, warped, rank, suit):
#     plt.figure(figsize=(16, 12))
#     plt.subplot(2,2,1)
#     plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
#     plt.title("Original Image")
#     plt.subplot(2,2,2)
#     plt.imshow(edged, cmap='gray')
#     plt.title("Edge Detected Image")
#     plt.subplot(2,2,3)
#     plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
#     plt.title("Warped (Aligned) Image")
#     plt.subplot(2,2,4)
#     plt.imshow(cv2.cvtColor(rank, cv2.COLOR_BGR2RGB))
#     plt.title("Rank Region")
#     plt.tight_layout()
#     plt.show()

# def main(folder_path):
#     for img in read_images_from_folder(folder_path):
#         preprocessed = preprocess_image(img)
#         edged = detect_edges(preprocessed)
#         card_contour = find_card_contour(edged)
#         if card_contour is not None:
#             warped = get_rotated_card(img, card_contour)
#             rank_region, suit_region = crop_rank_suit(warped)
#             plot_images(img, edged, warped, rank_region, suit_region)
#         else:
#             print("No card contour detected.")

# if __name__ == "__main__":
#     folder_path = input("Enter the folder path containing images: ")
#     main(folder_path)
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# def read_images_from_folder(folder_path):
#     image_paths = list(Path(folder_path).glob("*.jpg"))  # Adjust file extension as needed
#     for image_path in image_paths:
#         if image_path.is_file():
#             print(f"Processing image: {image_path.name}")
#             img = cv2.imread(str(image_path))
#             yield img

# def preprocess_image(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     return blurred

# def detect_edges(img):
#     edged = cv2.Canny(img, 50, 150)
#     return edged

# def find_card_contour(edged):
#     contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # Assume the largest contour is the card
#     if contours:
#         card_contour = max(contours, key=cv2.contourArea)
#         return card_contour
#     else:
#         return None

# def get_rotated_card(img, contour):
#     # Get the minimum area rectangle
#     rect = cv2.minAreaRect(contour)
#     box = cv2.boxPoints(rect)
#     box = np.int0(box)

#     # Calculate width and height of the rectangle
#     width = int(rect[1][0])
#     height = int(rect[1][1])

#     # Correct dimensions if necessary
#     if width < height:
#         width, height = height, width

#     # Get the rotation matrix for the rectangle
#     src_pts = box.astype("float32")
#     dst_pts = np.array([[0, height-1],
#                         [0, 0],
#                         [width-1, 0],
#                         [width-1, height-1]], dtype="float32")

#     # Compute the perspective transform matrix and apply it
#     M = cv2.getPerspectiveTransform(src_pts, dst_pts)
#     warped = cv2.warpPerspective(img, M, (width, height))

#     return warped

# def plot_images(original, edged, warped):
#     plt.figure(figsize=(16, 8))
#     plt.subplot(1,3,1)
#     plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
#     plt.title("Original Image")
#     plt.subplot(1,3,2)
#     plt.imshow(edged, cmap='gray')
#     plt.title("Edge Detected Image")
#     plt.subplot(1,3,3)
#     plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
#     plt.title("Rotated and Cropped Card")
#     plt.tight_layout()
#     plt.show()

# def main(folder_path):
#     for img in read_images_from_folder(folder_path):
#         preprocessed = preprocess_image(img)
#         edged = detect_edges(preprocessed)
#         card_contour = find_card_contour(edged)
#         if card_contour is not None:
#             warped = get_rotated_card(img, card_contour)
#             plot_images(img, edged, warped)
#         else:
#             print("No card contour detected.")

# if __name__ == "__main__":
#     folder_path = input("Enter the folder path containing images: ")
#     main(folder_path)




# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# def read_images_from_folder(folder_path):
#     image_paths = list(Path(folder_path).glob("*.jpg"))  # Adjust file extension as needed
#     for image_path in image_paths:
#         if image_path.is_file():
#             print(f"Processing image: {image_path.name}")
#             img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
#             #binarize the image
#             _, img = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)
#             yield img

# def preprocess_image(img):
#     # Image is already in grayscale
#     blurred = cv2.GaussianBlur(img, (5, 5), 0)
#     return blurred

# def detect_edges(img):
#     edged = cv2.Canny(img, 50, 150)
#     return edged

# def find_card_contour(edged):
#     contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         # Sort contours by area in descending order
#         contours = sorted(contours, key=cv2.contourArea, reverse=True)
#         for contour in contours:
#             # Approximate the contour to a polygon
#             peri = cv2.arcLength(contour, True)
#             approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
#             # If the approximated contour has four points, we can assume it's the card
#             if len(approx) == 4:
#                 return approx.reshape(4, 2)
#         # If no 4-point contour is found, return the largest contour
#         return contours[0].reshape(-1, 2)
#     else:
#         return None

# def order_points(pts):
#     # Initialize a list of coordinates to hold the ordered points
#     rect = np.zeros((4, 2), dtype="float32")

#     # The top-left point will have the smallest sum, whereas the bottom-right will have the largest sum
#     s = pts.sum(axis=1)
#     rect[0] = pts[np.argmin(s)]
#     rect[2] = pts[np.argmax(s)]

#     # The top-right point will have the smallest difference, whereas the bottom-left will have the largest difference
#     diff = np.diff(pts, axis=1)
#     rect[1] = pts[np.argmin(diff)]
#     rect[3] = pts[np.argmax(diff)]

#     return rect

# def get_rotated_card(img, contour):
#     # Order the contour points
#     rect = order_points(contour)
#     (tl, tr, br, bl) = rect

#     # Compute the width of the new image
#     widthA = np.linalg.norm(br - bl)
#     widthB = np.linalg.norm(tr - tl)
#     maxWidth = int(max(widthA, widthB))

#     # Compute the height of the new image
#     heightA = np.linalg.norm(tr - br)
#     heightB = np.linalg.norm(tl - bl)
#     maxHeight = int(max(heightA, heightB))

#     # Set up the destination points for the perspective transform
#     dst = np.array([
#         [0, 0],
#         [maxWidth - 1, 0],
#         [maxWidth -1, maxHeight -1],
#         [0, maxHeight -1]
#     ], dtype="float32")

#     # Compute the perspective transform matrix and apply it
#     M = cv2.getPerspectiveTransform(rect, dst)
#     warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

#     return warped

# def plot_images(original, edged, warped):
#     plt.figure(figsize=(16, 8))
#     plt.subplot(1,3,1)
#     plt.imshow(original, cmap='gray')
#     plt.title("Original Image")
#     plt.subplot(1,3,2)
#     plt.imshow(edged, cmap='gray')
#     plt.title("Edge Detected Image")
#     plt.subplot(1,3,3)
#     plt.imshow(warped, cmap='gray')
#     plt.title("Rotated and Cropped Card")
#     plt.tight_layout()
#     plt.show()

# def main(folder_path):
#     for img in read_images_from_folder(folder_path):
#         preprocessed = preprocess_image(img)
#         edged = detect_edges(preprocessed)
#         card_contour = find_card_contour(edged)
#         if card_contour is not None and len(card_contour) == 4:
#             warped = get_rotated_card(img, card_contour)
#             plot_images(img, edged, warped)
#         else:
#             print("No card contour detected or contour does not have 4 points.")

# if __name__ == "__main__":
#     folder_path = input("Enter the folder path containing images: ")
#     main(folder_path)





import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def read_images_from_folder(folder_path):
    image_paths = list(Path(folder_path).glob("*.jpg"))  # Adjust file extension as needed
    for image_path in image_paths:
        if image_path.is_file():
            print(f"Processing image: {image_path.name}")
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (1024, 768))
            # _, img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
            yield img

def preprocess_image(img):
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    # Apply adaptive thresholding to improve edge detection
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

def detect_edges(img):
    # Use the preprocessed image directly since it's binary after thresholding
    return img

def find_card_contour(edged):
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Filter contours by area
        MIN_CARD_AREA = 50000 # Adjust based on your image size
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CARD_AREA]
        # Sort contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Use hierarchy to find the outermost contour
        for i, cnt in enumerate(contours):
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2)
        # If no 4-point contour is found, use the convex hull of the largest contour
        cnt = contours[0]
        hull = cv2.convexHull(cnt)
        return hull.reshape(-1, 2)
    else:
        return None

def order_points(pts):
    # Initialize a list of coordinates to hold the ordered points
    rect = np.zeros((4, 2), dtype="float32")
    
    # The top-left point will have the smallest sum, whereas the bottom-right will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # The top-right point will have the smallest difference, whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def get_rotated_card(img, contour):
    # Order the contour points
    rect = order_points(contour)
    (tl, tr, br, bl) = rect
    
    # Compute the width of the new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    
    # Compute the height of the new image
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    
    # Set up the destination points for the perspective transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth -1, maxHeight -1],
        [0, maxHeight -1]
    ], dtype="float32")
    
    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    
    # Optional: Rotate the image if width is less than height (to keep the card upright)
    if maxHeight < maxWidth:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    
    return warped

def extract_number_and_suit(warped):
    h, w = warped.shape[:2]
    # Define regions using relative positions
    number_roi = warped[int(0.02*h):int(0.20*h), int(0.02*w):int(0.18*w)]
    suit_roi = warped[int(0.20*h):int(0.35*h), int(0.02*w):int(0.18*w)]
    return number_roi, suit_roi




def plot_images(original, edged, warped):
    plt.figure(figsize=(16, 8))
    plt.subplot(1,3,1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Image")
    plt.subplot(1,3,2)
    plt.imshow(edged, cmap='gray')
    plt.title("Edge Detected Image")
    plt.subplot(1,3,3)
    plt.imshow(warped, cmap='gray')
    plt.title("Rotated and Cropped Card")
    plt.tight_layout()
    plt.show()

def main(folder_path):
    for img in read_images_from_folder(folder_path):
        preprocessed = preprocess_image(img)
        edged = detect_edges(preprocessed)
        card_contour = find_card_contour(edged)
        if card_contour is not None and len(card_contour) >= 4:
            # If the contour has more than 4 points, approximate to 4 points
            if len(card_contour) > 4:
                peri = cv2.arcLength(card_contour, True)
                card_contour = cv2.approxPolyDP(card_contour, 0.02 * peri, True)
                if len(card_contour) != 4:
                    print("Could not approximate contour to 4 points.")
                    continue
                card_contour = card_contour.reshape(4, 2)
            warped = get_rotated_card(img, card_contour)
            plot_images(img, edged, warped)
        else:
            print("No card contour detected or contour does not have enough points.")

if __name__ == "__main__":
    folder_path = input("Enter the folder path containing images: ")
    main(folder_path)





# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# def read_images_from_folder(folder_path):
#     image_paths = list(Path(folder_path).glob("*.jpg"))  # Adjust file extension as needed
#     for image_path in image_paths:
#         if image_path.is_file():
#             print(f"Processing image: {image_path.name}")
#             img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
#             # No need to resize unless necessary
#             # img = cv2.resize(img, (1024, 768))
#             yield img

# def preprocess_image(img):
#     # Apply Gaussian blur to reduce noise
#     blurred = cv2.GaussianBlur(img, (3, 3), 0)
#     # Apply adaptive thresholding to create a binary image
#     thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#                                    cv2.THRESH_BINARY_INV, 11, 2)
#     return thresh, blurred

# def find_card_contour(thresh):
#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     if contours and hierarchy is not None:
#         # Filter contours by area
#         MIN_CARD_AREA = 50000  # Adjust based on your image size
#         max_area = 0
#         max_contour = None
#         for cnt in contours:
#             area = cv2.contourArea(cnt)
#             if area > MIN_CARD_AREA and area > max_area:
#                 max_area = area
#                 max_contour = cnt

#         if max_contour is not None:
#             # Approximate the contour to a polygon
#             peri = cv2.arcLength(max_contour, True)
#             approx = cv2.approxPolyDP(max_contour, 0.02 * peri, True)
#             if len(approx) == 4:
#                 return approx.reshape(4, 2)
#             else:
#                 # Use the convex hull as a fallback
#                 hull = cv2.convexHull(max_contour)
#                 if len(hull) >= 4:
#                     return hull.reshape(-1, 2)
#                 else:
#                     print("Convex hull does not have enough points.")
#                     return None
#         else:
#             return None
#     else:
#         return None

# def order_points(pts):
#     rect = np.zeros((4, 2), dtype="float32")
#     # Sum and diff to order points
#     s = pts.sum(axis=1)
#     rect[0] = pts[np.argmin(s)]  # Top-left
#     rect[2] = pts[np.argmax(s)]  # Bottom-right
#     diff = np.diff(pts, axis=1)
#     rect[1] = pts[np.argmin(diff)]  # Top-right
#     rect[3] = pts[np.argmax(diff)]  # Bottom-left
#     return rect

# def refine_corners(img, corners):
#     if corners is None or len(corners) == 0:
#         print("No corners to refine.")
#         return corners
#     # Criteria for corner refinement
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
#     corners = corners.reshape(-1, 1, 2).astype(np.float32)
#     # Ensure corners are within image boundaries
#     h, w = img.shape[:2]
#     corners = np.clip(corners, [0, 0], [w - 1, h - 1])
#     # Check if corners array is valid
#     if corners.size == 0:
#         print("Corners array is empty after clipping.")
#         return corners.reshape(-1, 2)
#     try:
#         refined_corners = cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), criteria)
#         return refined_corners.reshape(-1, 2)
#     except cv2.error as e:
#         print(f"cornerSubPix failed: {e}")
#         return corners.reshape(-1, 2)

# def get_rotated_card(img, contour, original_img):
#     # Order the contour points
#     rect = order_points(contour)
#     rect = refine_corners(original_img, rect)  # Use original or blurred image
#     if rect is None or len(rect) != 4:
#         print("Refined corners are invalid.")
#         return None
#     (tl, tr, br, bl) = rect
#     # Compute the width and height of the new image
#     widthA = np.linalg.norm(br - bl)
#     widthB = np.linalg.norm(tr - tl)
#     maxWidth = int(max(widthA, widthB))
#     heightA = np.linalg.norm(tr - br)
#     heightB = np.linalg.norm(tl - bl)
#     maxHeight = int(max(heightA, heightB))
#     # Add margin
#     margin = 20  # Adjust as needed
#     dst = np.array([
#         [0 - margin, 0 - margin],
#         [maxWidth - 1 + margin, 0 - margin],
#         [maxWidth - 1 + margin, maxHeight - 1 + margin],
#         [0 - margin, maxHeight - 1 + margin]
#     ], dtype="float32")
#     # Perspective transform
#     M = cv2.getPerspectiveTransform(rect, dst)
#     warped = cv2.warpPerspective(img, M, (int(maxWidth + 2 * margin), int(maxHeight + 2 * margin)))
#     # Correct orientation
#     warped = correct_orientation(warped)
#     return warped

# def correct_orientation(warped):
#     h, w = warped.shape[:2]
#     if h < w:
#         warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
#     return warped

# def plot_images(original, thresh, warped):
#     plt.figure(figsize=(16, 8))
#     plt.subplot(1,3,1)
#     plt.imshow(original, cmap='gray')
#     plt.title("Original Image")
#     plt.subplot(1,3,2)
#     plt.imshow(thresh, cmap='gray')
#     plt.title("Thresholded Image")
#     plt.subplot(1,3,3)
#     plt.imshow(warped, cmap='gray')
#     plt.title("Rotated and Cropped Card")
#     plt.tight_layout()
#     plt.show()

# def main(folder_path):
#     for img in read_images_from_folder(folder_path):
#         thresh, blurred = preprocess_image(img)
#         card_contour = find_card_contour(thresh)
#         if card_contour is not None and len(card_contour) >= 4:
#             # If the contour has more than 4 points, approximate to 4 points
#             if len(card_contour) > 4:
#                 peri = cv2.arcLength(card_contour, True)
#                 card_contour = cv2.approxPolyDP(card_contour, 0.02 * peri, True)
#                 if len(card_contour) != 4:
#                     print("Could not approximate contour to 4 points.")
#                     continue
#                 card_contour = card_contour.reshape(4, 2)
#             warped = get_rotated_card(img, card_contour, blurred)
#             if warped is not None:
#                 plot_images(img, thresh, warped)
#             else:
#                 print("Warped image is None.")
#         else:
#             print("No card contour detected or contour does not have enough points.")

# if __name__ == "__main__":
#     folder_path = input("Enter the folder path containing images: ")
#     main(folder_path)







# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# def read_images_from_folder(folder_path):
#     image_paths = list(Path(folder_path).glob("*.jpg"))  # Adjust file extension as needed
#     for image_path in image_paths:
#         if image_path.is_file():
#             print(f"Processing image: {image_path.name}")
#             img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
#             img = cv2.resize(img, (1024, 768))
#             # _, img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
#             yield img

# def preprocess_image(img):
#     # Apply Gaussian blur to reduce noise
#     blurred = cv2.GaussianBlur(img, (11, 11), 0)
#     # Apply adaptive thresholding to improve edge detection
#     thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                    cv2.THRESH_BINARY_INV, 11, 2)
#     return thresh

# def detect_edges(img):
#     # Use the preprocessed image directly since it's binary after thresholding
#     return img

# def find_card_contour(edged):
#     contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         # Filter contours by area
#         MIN_CARD_AREA = 50000  # Adjust based on your image size
#         contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CARD_AREA]
#         # Sort contours by area in descending order
#         contours = sorted(contours, key=cv2.contourArea, reverse=True)
#         # Use hierarchy to find the outermost contour
#         for i, cnt in enumerate(contours):
#             peri = cv2.arcLength(cnt, True)
#             approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
#             if len(approx) == 4:
#                 return approx.reshape(4, 2)
#         # If no 4-point contour is found, use the convex hull of the largest contour
#         cnt = contours[0]
#         hull = cv2.convexHull(cnt)
#         return hull.reshape(-1, 2)
#     else:
#         return None

# def order_points(pts):
#     # Initialize a list of coordinates to hold the ordered points
#     rect = np.zeros((4, 2), dtype="float32")
    
#     # The top-left point will have the smallest sum, whereas the bottom-right will have the largest sum
#     s = pts.sum(axis=1)
#     rect[0] = pts[np.argmin(s)]
#     rect[2] = pts[np.argmax(s)]
    
#     # The top-right point will have the smallest difference, whereas the bottom-left will have the largest difference
#     diff = np.diff(pts, axis=1)
#     rect[1] = pts[np.argmin(diff)]
#     rect[3] = pts[np.argmax(diff)]
    
#     return rect

# def get_rotated_card(img, contour):
#     # Order the contour points
#     rect = order_points(contour)
#     (tl, tr, br, bl) = rect
    
#     # Compute the width of the new image
#     widthA = np.linalg.norm(br - bl)
#     widthB = np.linalg.norm(tr - tl)
#     maxWidth = int(max(widthA, widthB))
    
#     # Compute the height of the new image
#     heightA = np.linalg.norm(tr - br)
#     heightB = np.linalg.norm(tl - bl)
#     maxHeight = int(max(heightA, heightB))
    
#     # Set up the destination points for the perspective transform
#     dst = np.array([
#         [0, 0],
#         [maxWidth - 1, 0],
#         [maxWidth -1, maxHeight -1],
#         [0, maxHeight -1]
#     ], dtype="float32")
    
#     # Compute the perspective transform matrix and apply it
#     M = cv2.getPerspectiveTransform(rect, dst)
#     warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    
#     # Optional: Rotate the image if width is less than height (to keep the card upright)
#     if maxHeight < maxWidth:
#         warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    
#     return warped

# def extract_number_and_suit(warped):
#     h, w = warped.shape[:2]
#     # Define regions using relative positions
#     number_roi = warped[int(0.02*h):int(0.20*h), int(0.02*w):int(0.18*w)]
#     suit_roi = warped[int(0.20*h):int(0.35*h), int(0.02*w):int(0.18*w)]
#     return number_roi, suit_roi

# def preprocess_roi(roi):
#     # Apply Gaussian blur
#     roi_blurred = cv2.GaussianBlur(roi, (3, 3), 0)
#     # Apply thresholding
#     _, roi_thresh = cv2.threshold(roi_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     return roi_thresh

# def clean_roi(roi_thresh):
#     kernel = np.ones((3, 3), np.uint8)
#     # Erode to remove small white noise
#     roi_eroded = cv2.erode(roi_thresh, kernel, iterations=1)
#     # Dilate to restore the main features
#     roi_cleaned = cv2.dilate(roi_eroded, kernel, iterations=1)
#     return roi_cleaned

# def plot_images(original, edged, warped, number_roi=None, suit_roi=None):
#     plt.figure(figsize=(16, 8))
#     plt.subplot(2,3,1)
#     plt.imshow(original, cmap='gray')
#     plt.title("Original Image")
#     plt.subplot(2,3,2)
#     plt.imshow(edged, cmap='gray')
#     plt.title("Edge Detected Image")
#     plt.subplot(2,3,3)
#     plt.imshow(warped, cmap='gray')
#     plt.title("Rotated and Cropped Card")
#     if number_roi is not None and suit_roi is not None:
#         plt.subplot(2,3,4)
#         plt.imshow(number_roi, cmap='gray')
#         plt.title("Number Region")
#         plt.subplot(2,3,5)
#         plt.imshow(suit_roi, cmap='gray')
#         plt.title("Suit Region")
#     plt.tight_layout()
#     plt.show()

# def main(folder_path):
#     for img in read_images_from_folder(folder_path):
#         preprocessed = preprocess_image(img)
#         edged = detect_edges(preprocessed)
#         card_contour = find_card_contour(edged)
#         if card_contour is not None and len(card_contour) >= 4:
#             # If the contour has more than 4 points, approximate to 4 points
#             if len(card_contour) > 4:
#                 peri = cv2.arcLength(card_contour, True)
#                 card_contour = cv2.approxPolyDP(card_contour, 0.02 * peri, True)
#                 if len(card_contour) != 4:
#                     print("Could not approximate contour to 4 points.")
#                     continue
#                 card_contour = card_contour.reshape(4, 2)
#             warped = get_rotated_card(img, card_contour)
#             # Extract number and suit regions
#             number_roi, suit_roi = extract_number_and_suit(warped)
#             # Preprocess the ROIs
#             number_roi_preprocessed = preprocess_roi(number_roi)
#             suit_roi_preprocessed = preprocess_roi(suit_roi)
#             # Clean the ROIs
#             number_roi_cleaned = clean_roi(number_roi_preprocessed)
#             suit_roi_cleaned = clean_roi(suit_roi_preprocessed)
#             # Plot the results
#             plot_images(img, edged, warped, number_roi_cleaned, suit_roi_cleaned)
#             # Proceed with template matching using number_roi_cleaned and suit_roi_cleaned
#         else:
#             print("No card contour detected or contour does not have enough points.")

# if __name__ == "__main__":
#     folder_path = input("Enter the folder path containing images: ")
#     main(folder_path)

import cv2

# Load grayscale image
image = cv2.imread('project_5\\test_image.png', cv2.IMREAD_GRAYSCALE)

# Initialize MSER detector
mser = cv2.MSER_create()

# Detect MSER regions
regions, _ = mser.detectRegions(image)

# Draw MSER regions on the image
for region in regions:
    hull = cv2.convexHull(region.reshape(-1, 1, 2))
    cv2.polylines(image, [hull], isClosed=True, color=(255, 0, 0), thickness=2)

cv2.imshow('MSER Regions', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
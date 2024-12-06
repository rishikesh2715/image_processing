import cv2
import numpy as np
def main():
    
    src = cv2.imread("initial_mask.png", cv2.IMREAD_GRAYSCALE)
    
    
    gray = src
    
    
    # gray = cv2.medianBlur(gray, 5)
    
    
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=1, maxRadius=30)
    
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(src, center, radius, (255, 0, 255), 3)
    
    
    cv2.imshow("detected circles", src)
    cv2.waitKey(0)
    
    return 0
if __name__ == "__main__":
    main()

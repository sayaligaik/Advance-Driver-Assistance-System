import cv2
import numpy as np

def lane_detection(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Define the region of interest (ROI)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[(0, height), (width, height), (width // 2, height // 2)]])
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Hough transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

    # Draw lines on the original image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image

# Load video or webcam
cap = cv2.VideoCapture('input_video.mp4')  # Change to 0 for webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame
    processed_frame = lane_detection(frame)

    # Display the result
    cv2.imshow('Lane Detection', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
print("OpenCV version:", cv2.__version__)
cap = cv2.VideoCapture(0)
print("Camera available:", cap.isOpened())
cap.release()

# Test the webcam can capture the image or not
import cv2 as cv
import numpy as np

# create a VideoCapture object to cature the video, and if only have 1 camera connected, just pass 0
cap=cv.VideoCapture(0)
# If camera doesn't capture video, then exit
if not cap.isOpened():
    print("Cannot open camera")
    exit
# Keep capture the video if camera is open
while True:
    # Capture frame by frame
    ret, frame=cap.read()

    # If frame is read correctly,ret is true
    if not ret:
        print("Can't receive frame, Exiting ...")
        exit
    # Conevert to grayscale
    gray=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv.imshow('frame',frame)
    cv.imshow('gray',gray)
    # Set the key q to be the close key
    if cv.waitKey(1)==ord('q'):
        break
# When everything done release the capture
cap.release()
cv.destroyAllWindows()

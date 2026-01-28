from holistic_detector import HolisticDetector
import cv2 as cv

def main():
    # Initialize Holistic detector
    detector = HolisticDetector()
    # Open the webcam with optimized settings
    cap = cv.VideoCapture(0)
    # Set camera properties for better performance
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280) # set width to 1280
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720) # set height to 720
    cap.set(cv.CAP_PROP_FPS, 30)           # set FPS to 30
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)     # reduce buffer to minimize delay

    while cap.isOpened():
        ret, frame =cap.read()
        # If not open successfully, then exit
        if not ret:
            print("Capture failed, exiting...")
            exit()
        # Process the frame to get holistic results
        results= detector.process_frame(frame)
        # Draw landmarks on the frame
        annoted_frame = detector.draw_landmarks(frame, results)
        # show the annotated frame
        cv.imshow('Holistic Detection', cv.flip(annoted_frame, 1))
        if cv.waitKey(1) == ord('q'):
            break
    # Cloes the webcam and destroy all windows
    cap.release()
    cv.destroyAllWindows()

if __name__=='__main__':
    main()


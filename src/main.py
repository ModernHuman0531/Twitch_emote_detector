from holistic_detector import HolisticDetector
import cv2 as cv
import os


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

    # Load images once at startup instead of every frame
    images_name = {
        1: "ding1.png",
        2: "ding2.png", 
        3: "ding3.png",
        4: "ding4.png"
    }
    image_path = os.path.join('..', 'images')
    image_file = 'ding1.png'  # Default image

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv.flip(frame, 1)
        # If not open successfully, then exit
        if not ret:
            print("Capture failed, exiting...")
            break
            
        # Process the frame to get holistic results
        results = detector.process_frame(frame)
        # Draw landmarks on the frame
        annotated_frame = detector.draw_landmarks(frame, results)
        info_text = f"Currently displaying: {image_file}"
        cv.putText(annotated_frame, info_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        # Show the annotated frame
        cv.imshow('Holistic Detection', annotated_frame)
        
        # Get key press ONCE per frame
        pressed_key = cv.waitKey(1) & 0xFF
        
        # Handle key presses
        if pressed_key == ord('q'):
            break
        elif pressed_key == ord('1'):
            image_file = images_name[1]
            print(f"Switched to {image_file}")
        elif pressed_key == ord('2'):
            image_file = images_name[2]
            print(f"Switched to {image_file}")
        elif pressed_key == ord('3'):     
            image_file = images_name[3]
            print(f"Switched to {image_file}")
        elif pressed_key == ord('4'):
            image_file = images_name[4]
            print(f"Switched to {image_file}")

        # Display the current image
        if image_file:
            image_full_path = os.path.join(image_path, image_file)
            if os.path.exists(image_full_path):
                image = cv.imread(image_full_path)
                if image is not None:
                    cv.imshow("Ding Emotes", image)
                else:
                    print(f"Failed to load image: {image_full_path}")
            else:
                print(f"Image file not found: {image_full_path}")

    # Close the webcam and destroy all windows
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()


from holistic_detector import HolisticDetector
from pose_classify_model import PoseClassifyModel
from preprocessor import Preprocessor
import numpy as np
import cv2 as cv
import os

def load_image(image_idx):
    """
    Returns the image path for the given image name.
    After we get the model prediction(it'll be string like "ding1.png"), we can call this function to load the image.
    """
    image_dir = os.path.join('..', 'images')
    images_name = {
        "Proud": "ding1.png",
        "Laugh": "ding2.png",
        "Upset": "ding3.png",
        "Thumbs down": "ding4.png"
    }
    image_path = ""
    if os.path.exists(image_dir):
        image_path = os.path.join(image_dir, images_name[image_idx])
    else:
        image_path = None
        print(f"Image directory not found: {image_dir}")
    
    return image_path


def show_image(image_path, window_name="Predict emote"):
    """
    Display the image at the given path using OpenCV.
    Split into load_image and show_image is because when image display fails, 
    we can know whether it's loading or displaying issue.
    """
    if image_path is not None:
        image = cv.imread(image_path) 
        cv.imshow(window_name, image)
        print(f"Displaying image: {image_path}")
    else:
        black_blank = np.zeros((400,300,3), dtype=np.uint8)
        cv.imshow(window_name, black_blank)
        print(f"Pose not recognized, failed to load image: {image_path}")


def main():
    # Initialize Holistic detector
    detector = HolisticDetector()

    # Initialize Pose Classification model
    model = PoseClassifyModel(model_name='pose_classify_model.pkl')
    
    # Initialize Preprocessor
    preprocessor = Preprocessor()

    # Open the webcam with optimized settings
    cap = cv.VideoCapture(0)
    # Set camera properties for better performance
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280) # set width to 1280
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720) # set height to 720
    cap.set(cv.CAP_PROP_FPS, 30)           # set FPS to 30
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)     # reduce buffer to minimize delay

    # Set the confidence threshold for displaying predictions
    confidence_threshold = 0.65

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv.flip(frame, 1)
        # If not open successfully, then exit
        if not ret:
            raise RuntimeError("Failed to read from webcam.")
        # Process the frame to get landmarks
        results = detector.process_frame(frame)
        frame = detector.draw_landmarks(frame, results)

        # Extract features
        feature_vector = preprocessor.extract_landmarks(results)
        # Reshape feature vector for prediction into 2D array
        feature_vector = np.array(feature_vector).reshape(1,-1)
        # Predict the pose label and its probability
        pose_label, probability = model.predict(feature_vector)

        # If the probability is higher than the threshold, display the corresponding image, else display the black blank
        if probability >= confidence_threshold:
            image_path = load_image(pose_label)
            show_image(image_path)
        else:
            show_image(None)

        # Add some function buttom to commnunicate with user
        pressed_key = cv.waitKey(1) & 0xFF
        if pressed_key == ord('q'):
            break
        elif pressed_key == ord('s'):
            # Save current frame as image
            save_path = os.path.join('..', 'results')
            file_num = 0
            # Calculate the file numbers in save_path, and auto calculate the new file name
            for lists in os.listdir(save_path):
                sub_path = os.path.join(save_path, lists)
                if os.path.isfile(sub_path):
                    file_num += 1
            # Named the new file with incremented number
            image_name = f"capture_{file_num+1}.png"
            cv.imwrite(os.path.join(save_path, image_name),frame)
        
        # Display the webcam frame with landmarks
        cv.imshow("Webcam Feed", frame)

            
    # Close the webcam and destroy all windows
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()


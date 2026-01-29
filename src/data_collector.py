import cv2 as cv
import numpy as np
import os
import joblib
from holistic_detector import HolisticDetector
from preprocessor import Preprocessor
from pose_classify_model import PoseClassifyModel

class DataCollector:
    def __init__(self):
        # Initialize components
        self.detector = HolisticDetector()
        self.preprocessor = Preprocessor()
        self.model = PoseClassifyModel()
        # Initialize variables for collecting data
        self.current_pose = 0 # Default to "Unknown"
        self.frame_delay = 10 # Collect data every 10 frames
        self.frame_count = 0
        self.auto_collect = False
        self.collected_samples = 0 # Initialize to 0 in every run for one pose
        self.sample_per_pose = 200 # Number of samples to collect per pose
        self.data_folder = os.path.join('..', 'data')
        self.data_name = 'collected_data.npz'

        # Set webcam properties
        self.cap = cv.VideoCapture(0)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv.CAP_PROP_FPS, 30)
        self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

        self.pose_labels = {
            0: "Unknown",
            1: "Proud",
            2: "Laugh",
            3: "Upset",
            4: "Thumbs down"
        }
        
        # Initialize data storage at class level
        self.data_X = []  # 2D array, for features
        self.data_y = []  # 1D array, for labels
    
    def collect_data(self):
        """
        Main function to collect data from the webcam frame.
        Controls:
            - '1' key: Set current pose to "Proud"
            - '2' key: Set current pose to "Laugh"
            - '3' key: Set current pose to "Upset"
            - '4' key: Set current pose to "Thumbs down"
            - 'a' key: Start/stop auto data collection
            - 's' key: Save data to specified folder
            - 'l' key: Load existing data from specified folder
            - 't' key: Train the model with collected data
            - 'q' key: Quit the program
        """

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to read from webcam.")

            self.frame_count += 1

            # Process the frame to get landmarks
            # Flip the frame horizontally for a later selfie-view display
            frame = cv.flip(frame, 1)
            results = self.detector.process_frame(frame)
            frame = self.detector.draw_landmarks(frame, results)

            if (self.auto_collect and self.current_pose != 0 and self.frame_count % self.frame_delay == 0):
                # Extract features
                feature_vector = self.preprocessor.extract_landmarks(results)
                self.data_X.append(feature_vector)
                self.data_y.append(self.current_pose)
                self.collected_samples += 1
                print(f" Collected sample {self.collected_samples} for pose {self.pose_labels[self.current_pose]}")

                # Check if we have collected enough samples
                if self.collected_samples >= self.sample_per_pose:
                    print(f" Completed collecting {self.sample_per_pose} samples for pose {self.pose_labels[self.current_pose]}")
                    print(" Press another number key to collect data for a different pose, or 'a' to stop auto collection.")
                    self.current_pose = 0
                    self.collected_samples = 0
            
            # Display instructions
            auto_state = "ON" if self.auto_collect else "OFF"
            if self.auto_collect and self.current_pose != 0:
                info_text = f"Auto collecting {self.pose_labels[self.current_pose]} pose."
                cv.putText(frame, info_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            elif self.auto_collect:
                info_text = "Auto collection is ON. Press number key to select pose."
                cv.putText(frame, info_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                info_text = "Auto collection is OFF. Press 'a' to start."
                cv.putText(frame, info_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            # Show the frame
            cv.imshow('Data Collector', frame)

            # Handle key presses 
            pressed_key = cv.waitKey(1) & 0xFF
            # Quit the program
            if pressed_key == ord('q'):
                break
            # Select pose to collect data for 
            elif pressed_key>=ord('1') and pressed_key<=ord('4'):
                self.current_pose = pressed_key - ord('0')
                self.collected_samples = 0
                print(f" Switch to colecting pose: {self.pose_labels[self.current_pose]}")
            # Toggle auto collection
            elif pressed_key == ord('a'):
                self.auto_collect = not self.auto_collect
                status = "started" if self.auto_collect else "stopped"
                print(f"Auto collection is {status}.")
                if self.auto_collect:
                    print("Auto collection started. Press number key to select pose.")
            # Save collected data
            elif pressed_key == ord('s'):
                if len(self.data_X) > 0:
                    self.save_data(np.array(self.data_X), np.array(self.data_y))
                    print(f" Saved {len(self.data_X)} samples to {self.data_folder}")
                    self.data_X = []
                    self.data_y = []
                else:
                    print(" No data to save. Please collect some data first.")
            elif pressed_key == ord('l'):
                loaded_X, loaded_y = self.load_data()
                if loaded_X is not None and loaded_y is not None:
                    self.data_X = loaded_X.tolist()
                    self.data_y = loaded_y.tolist()
                    print(f" Loaded data from {self.data_folder}, total samples: {len(self.data_y)}")
            elif pressed_key == ord('t'):
                if len(self.data_X) > 0:
                    print("Training model with collected data ...")
                    self.train_model(np.array(self.data_X), np.array(self.data_y))
                else:
                    print("No data to train the model. Please collect or load the data first.")
        # Release the webcam and close windows
        self.cap.release()
        cv.destroyAllWindows()
    def train_model(self, X, y):
        """
        Train the pose classification model with collected data and save the model.
        """
        # Train the model
        self.model.train(X, y)
        # Save the trained model
        self.model.save_model('pose_classify_model.pkl')
        print("Model trained and saved as 'pose_classify_model.pkl'.")
    
    def save_data(self, X, y):
        """
        Save collected data to data folder
        """
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        
        file_path = os.path.join(self.data_folder, self.data_name)
        np.savez_compressed(file_path, X=X, y=y)
        print(f"Data saved to {file_path}.")
    
    def load_data(self):
        """
        Load the collected data from data folder. Change the data_name if needed.
        """
        file_path = os.path.join(self.data_folder, self.data_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file {self.data_name} not found in folder {self.data_folder}.")
        data = np.load(file_path)
        X = data['X']
        y = data['y']
        print(f"Data loaded from {file_path}, total samples: {len(y)}")
        return X, y
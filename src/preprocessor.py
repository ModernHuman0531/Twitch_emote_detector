import cv2 as cv
import numpy as np

class Preprocessor:
    """
    Preprocess the results from MediaPipe Holistic model to extract relevant landmarks.
    And turn it into the desired format as the input of random forest model.    
    """

    def __init__(self):
        # Define specified landmarks for face, pose and hands
        self.face_landmarks_ids = [
           1, # Nose tip
           33, 133, 159, 145, # Left eye
           362, 263, 386, 374, # Right eye
           61, 291, 11, 16 # Mouth
        ] 
        self.pose_landmarks_ids = [11, 12, 13, 14] # Shoulders and wrists
        self.handmarks_ids = list(range(21)) # All 21 hand landmarks

    def extract_landmarks(self, results):
        """
        Extract relevant landmarks from MediaPipe Holistic results.
        Args:
            results: MediaPipe Holistic results object.
        Returns:
            An array of extracted landmarks in the order of face, pose, left hand, right hand.
        """
        feature_vector = []
        # Extract face landmarks, use relative coordinates to nose tip
        # Have 13*2 = 26 values
        if results.face_landmarks:
            nose_landmark = results.face_landmarks.landmark[1]
            for idx in self.face_landmarks_ids:
                landmark = results.face_landmarks.landmark[idx]
                feature_vector.append(landmark.x - nose_landmark.x)
                feature_vector.append(landmark.y - nose_landmark.y)
        else:
            feature_vector.extend(np.zeros(len(self.face_landmarks_ids)*2).tolist())
        
        # Extract pose landmarks, use relative coordinates to the middle point of shoulders
        # Have 4*2 = 8 values
        if results.pose_landmarks:
            left_shoulder = results.pose_landmarks.landmark[11]
            right_shoulder = results.pose_landmarks.landmark[12]
            mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
            mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            for idx in self.pose_landmarks_ids:
                landmark = results.pose_landmarks.landmark[idx]
                feature_vector.append(landmark.x - mid_shoulder_x)
                feature_vector.append(landmark.y - mid_shoulder_y)
        else:
            feature_vector.extend(np.zeros(len(self.pose_landmarks_ids)*2).tolist())
        
        # Extract left hand landmarks, use relative coordinates to wrist
        # Have 21*2 + 1(handedness) = 43 values
        if results.left_hand_landmarks:
            wrist_landmark = results.left_hand_landmarks.landmark[0]
            for idx in self.handmarks_ids:
                landmark = results.left_hand_landmarks.landmark[idx]
                feature_vector.append(landmark.x - wrist_landmark.x)
                feature_vector.append(landmark.y - wrist_landmark.y)
            feature_vector.append(1) # Indicate left hand detected
        else:
            feature_vector.extend(np.zeros(len(self.handmarks_ids)*2).tolist())
            feature_vector.append(0) # Indicate left hand not detected
        
        # Extract right hand landmarks, use relative coordinates to wrist
        # Have 21*2 + 1(handedness) = 43 values
        if results.right_hand_landmarks:
            wrist_landmark = results.right_hand_landmarks.landmark[0]
            for idx in self.handmarks_ids:
                landmark = results.right_hand_landmarks.landmark[idx]
                feature_vector.append(landmark.x - wrist_landmark.x)
                feature_vector.append(landmark.y - wrist_landmark.y)
            feature_vector.append(1) # Indicate right hand detected
        else:
            feature_vector.extend(np.zeros(len(self.handmarks_ids)*2).tolist())
            feature_vector.append(0) # Indicate right hand not detected
        
        # Add four more features: relative coordnates between left and right wrists with nose
        if results.pose_landmarks:
            if results.left_hand_landmarks:
                left_wrist = results.left_hand_landmarks.landmark[0]
                nose_landmark = results.face_landmarks.landmark[1]
                feature_vector.append(left_wrist.x - nose_landmark.x)
                feature_vector.append(left_wrist.y - nose_landmark.y)
            else:
                feature_vector.extend([0,0])
            if results.right_hand_landmarks:
                right_wrist = results.right_hand_landmarks.landmark[0]
                nose_landmark = results.face_landmarks.landmark[1]
                feature_vector.append(right_wrist.x - nose_landmark.x)
                feature_vector.append(right_wrist.y - nose_landmark.y)
            else:
                feature_vector.extend([0,0])
        else:
            feature_vector.extend([0,0,0,0])
        
        return np.array(feature_vector)

                

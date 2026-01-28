import cv2 as cv
import mediapipe as mp
import numpy as np

class HolisticDetector:
    def __init__(self):
        # Initialize MediaPipe drawing utilities
        self.my_drawing = mp.solutions.drawing_utils
        self.my_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize Mediapipe holistic solution
        self.my_holistic = mp.solutions.holistic
        
        # Define specified landmarks for face, pose and hands
        self.face_landmarks_ids = [
           1, # Nose tip
           33, 133, 159, 145, # Left eye
           362, 263, 386, 374, # Right eye
           61, 291, 11, 16 # Mouth
        ] 
        self.pose_landmarks_ids = [11, 12, 13, 14] # Shoulders and wrists

        # Define holistic model with desired parameters
        self.holistic = self.my_holistic.Holistic(
             static_image_mode = False,
             model_complexity = 1, # Use full mode (0: Lite, 1: Full, 2: Heavy)
             enable_segmentation = False,
             refine_face_landmarks = False,
             min_detection_confidence = 0.5, # Higher confidence for more stable detection
             min_tracking_confidence = 0.5 # Lower tracking confidence for faster response
        )
    def process_frame(self, frame):
        # Process the input frame and return results
        frame.flags.writeable = False
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results =self.holistic.process(frame)
        return results
    
    def draw_landmarks(self, frame, results):
        # Draw landmarks annotation on the frame
        frame.flags.writeable = True
        # frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        # Set frame's height and width
        height, width = frame.shape[:2]

        # Draw landmarks
        # Draw face landmarks
        if results.face_landmarks:
            face_coords = {}
            for idx in self.face_landmarks_ids:
                landmark = results.face_landmarks.landmark[idx]
                x, y = int(landmark.x * width), int(landmark.y * height)
                face_coords[idx] = (x, y)
            # Draw selected face landmarks
            for face_coord in face_coords.values():
                cv.circle(frame, face_coord, 8, (0, 0, 255), -1)
            # Connect landmarks to form a simple face outline
            # Eyes connection (33-159-133-145-33) and (362-386-263-374-362)
            cv.line(frame, face_coords[33], face_coords[159], (0, 255, 0), 2)
            cv.line(frame, face_coords[159], face_coords[133], (0, 255, 0), 2)
            cv.line(frame, face_coords[133], face_coords[145], (0, 255, 0), 2)
            cv.line(frame, face_coords[145], face_coords[33], (0, 255, 0), 2) 
            cv.line(frame, face_coords[362], face_coords[386], (0, 255, 0), 2)
            cv.line(frame, face_coords[386], face_coords[263], (0, 255, 0), 2)
            cv.line(frame, face_coords[263], face_coords[374], (0, 255, 0), 2)
            cv.line(frame, face_coords[374], face_coords[362], (0, 255, 0), 2) 

            # Mouth connection (61-11-291-16-61)
            cv.line(frame, face_coords[61], face_coords[11], (0, 255, 0), 2)
            cv.line(frame, face_coords[11], face_coords[291], (0, 255, 0), 2)
            cv.line(frame, face_coords[291], face_coords[16], (0, 255, 0), 2)
            cv.line(frame, face_coords[16], face_coords[61], (0, 255, 0), 2)

            # Connect eyes to nose
            cv.line(frame, face_coords[1], face_coords[133], (0,255,0),2)
            cv.line(frame, face_coords[1], face_coords[362], (0,255,0),2)

        # Draw pose landmarks
        if results.pose_landmarks:
            pose_coords = {}
            for idx in self.pose_landmarks_ids:
                landmark = results.pose_landmarks.landmark[idx]
                x, y = int(landmark.x * width), int(landmark.y * height)
                pose_coords[idx] = (x,y)
            # Draw selected pose landmarks
            for pose_coord in pose_coords.values():
                cv.circle(frame, pose_coord, 8, (255, 0, 0), -1)
            # Connect shoulders and wrists
            # (11-13), (11-12), (12-14)
            cv.line(frame, pose_coords[11], pose_coords[13], (0, 0, 255), 2)
            cv.line(frame, pose_coords[11], pose_coords[12], (0, 0, 255), 2)
            cv.line(frame, pose_coords[12], pose_coords[14], (0, 0, 255), 2)
        
        # Draw hand landmarks
        # Self-defined connections and landmarks for hands
        HAND_CONNECTIONS = self.my_drawing.DrawingSpec(color=(0, 255, 0), thickness=2) 
        HAND_LANDMARKS = self.my_drawing.DrawingSpec(color=(0, 0, 255), thickness=6, circle_radius=4)
        # Draw left hand landmarks 
        if results.left_hand_landmarks:
            self.my_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.my_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec = HAND_LANDMARKS,
                connection_drawing_spec = HAND_CONNECTIONS
            )
        # Draw right hand landmarks
        if results.right_hand_landmarks:
            self.my_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.my_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec = HAND_LANDMARKS,
                connection_drawing_spec = HAND_CONNECTIONS
            )
        return frame


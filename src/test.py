import cv2 as cv
import mediapipe as mp
import numpy as np

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe solutions
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_holistic = mp.solutions.holistic

"""
Put open webcam code into the collect data script, and organize other code into classes and functions.
"""
# Open the webcam with optimized settings 
cap = cv.VideoCapture(0)
# Set camera properties for better performance
# cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)   # Reduce resolution for better performance
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv.CAP_PROP_FPS, 30)            # Set desired FPS
cap.set(cv.CAP_PROP_BUFFERSIZE, 1)      # Reduce buffer to minimize delay



# Define the holistic model with optimized settings
with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,                   # Use lighter model (0=lite, 1=full, 2=heavy)
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,         # Higher confidence for more stable detection
    min_tracking_confidence=0.5           # Lower tracking confidence for faster response
) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        # If not open successfully, then exit
        if not ret:
            print("Ignoring empty camera frame.")
            continue
        # To improve the performance, optionally mark the frame as not writeable to pass by reference
        
        frame.flags.writeable = False
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = holistic.process(frame)

        # Draw landmarks annotation on the frame
        frame.flags.writeable = True
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        # Set frame's height and width
        height, width = frame.shape[:2]

        # Draw face landmarks
        if results.face_landmarks:
            face_coords = {}
            selected_face_indices = [
                1, # Nose tip
                33, 133, 159, 145, # Left eye
                362, 263, 386, 374, # Right eye
                61, 291, 11, 16 # Mouth
            ]
            face_landmarks = results.face_landmarks
            for idx in selected_face_indices:
                landmark = face_landmarks.landmark[idx]
                x, y = int(landmark.x * width), int(landmark.y * height)
                face_coords[idx] = (x,y)
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


        # Draw pose landmarks, only need shoulders and elbows
        if results.pose_landmarks:
            key_pose = [11, 12, 13, 14] # Shoulders and wrists
            coords = {} # To store coordinates for connections
            pose_landmarks = results.pose_landmarks
            for idx in key_pose:
                landmark = pose_landmarks.landmark[idx]
                x, y = int(landmark.x * width), int(landmark.y * height)
                coords[idx] = (x, y)
            
            # Draw key landmarks
            for coord in coords.values():
                cv.circle(frame, coord, 8, (255, 0, 0), -1)
            # Connect shoulders and wrists, (11-13, 11-12, 12-14)
            cv.line(frame, coords[11], coords[13], (0, 0, 255), 2)
            cv.line(frame, coords[11], coords[12], (0, 0, 255), 2)
            cv.line(frame, coords[12], coords[14], (0, 0, 255), 2)

            
        
        # Draw left and right hand landmarks
        
        # Self-defined connections and landmarks for hands
        HAND_CONNECTIONS = mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)
        HAND_LANDMARKS = mp_drawing.DrawingSpec(color=(0,0,255), thickness=6,circle_radius=4)

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                HAND_LANDMARKS,
                HAND_CONNECTIONS
            )
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                HAND_LANDMARKS,
                HAND_CONNECTIONS
            )

        # Flip the image horizontally for a selfie-view display
        cv.imshow('Mediapipe Holistic',cv.flip(frame,1))
        if cv.waitKey(1)==ord('q'):
            break
cap.release()
cv.destroyAllWindows() 

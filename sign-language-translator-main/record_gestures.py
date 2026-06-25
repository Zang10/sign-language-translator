import cv2
import mediapipe as mp
import numpy as np
import os

# Configuration
GESTURES = ["thank_you", "hello", "help"]  # Add more signs here
SAMPLES_PER_CLASS = 200  # Number of recordings per gesture
DATA_DIR = "dataset"  # Folder to save recordings

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Create folders for each gesture
for gesture in GESTURES:
    os.makedirs(f"{DATA_DIR}/{gesture}", exist_ok=True)

# Record each gesture
for gesture_idx, gesture in enumerate(GESTURES):
    print(f"Recording {gesture}... (Press SPACE to start, Q to stop)")
    
    cap = cv2.VideoCapture(0)
    counter = 0
    
    while counter < SAMPLES_PER_CLASS:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Show instructions
        cv2.putText(frame, f"Recording {gesture} ({counter}/{SAMPLES_PER_CLASS})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press Q to stop", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        if results.multi_hand_landmarks:
            # Save landmarks as numpy array
            landmarks = []
            for landmark in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            np.save(f"{DATA_DIR}/{gesture}/{counter}.npy", np.array(landmarks))
            counter += 1
        
        cv2.imshow("Recording", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

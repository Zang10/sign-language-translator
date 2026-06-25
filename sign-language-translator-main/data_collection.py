import os
import cv2
import numpy as np
from hand_detector import detect_hands

# Create directories
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of classes (letters/gestures) and samples per class
num_classes = 5  # Start with 5 letters (A, B, C, D, E)
samples_per_class = 50

cap = cv2.VideoCapture(0)

for class_num in range(num_classes):
    class_dir = os.path.join(DATA_DIR, str(class_num))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    
    print(f'Collecting data for class {class_num}')
    done = False
    
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Ready? Press "Q" to collect class {class_num}', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Data Collection', frame)
        
        if cv2.waitKey(25) == ord('q'):
            break
    
    counter = 0
    while counter < samples_per_class:
        ret, frame = cap.read()
        frame, results = detect_hands(frame)
        
        if results.multi_hand_landmarks:
            cv2.putText(frame, f'Collecting sample {counter}', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Data Collection', frame)
            cv2.waitKey(25)
            
            # Save the normalized landmarks
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])
            
            np.save(os.path.join(class_dir, f'{counter}.npy'), np.array(landmarks))
            counter += 1

cap.release()
cv2.destroyAllWindows()

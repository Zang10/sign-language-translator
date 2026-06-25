import cv2
import numpy as np
import mediapipe as mp  # <-- This import was missing
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("sign_language_model.h5")

# Load gesture labels
with open('gesture_labels.txt') as f:
    GESTURES = [line.strip() for line in f]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, 
                      max_num_hands=1, 
                      min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Mirror the frame
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        # Extract landmarks
        landmarks = []
        for landmark in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        # Predict
        prediction = model.predict(np.expand_dims(landmarks, axis=0))
        predicted_class = GESTURES[np.argmax(prediction)]
        confidence = np.max(prediction)
        
        # Display result
        cv2.putText(frame, f"{predicted_class} ({confidence:.2f})", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Sign Language Detector", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

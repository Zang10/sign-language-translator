
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def is_thank_you_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    
    # Simple "Thank You" detection (thumb above fingers)
    return (thumb_tip.y < index_tip.y and 
            index_tip.y < middle_tip.y and 
            abs(middle_tip.x - thumb_tip.x) < 0.1)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue
    
    # Mirror the image
    image = cv2.flip(image, 1)
    
    # Process hand landmarks
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            if is_thank_you_gesture(hand_landmarks):
                # White background rectangle
                cv2.rectangle(image, (30, 30), (450, 120), (255, 255, 255), -1)
                
                # Black text (simple font)
                cv2.putText(image, "THANK YOU", (50, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    cv2.imshow('Sign Language Detector', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

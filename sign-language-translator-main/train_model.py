import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Configuration - UPDATE THESE TO MATCH YOUR PROJECT
DATA_DIR = "dataset"  # Folder where your recorded gestures are stored
GESTURES = ["thank_you", "hello", "help"]  # Must match your recorded gesture folders

# Load data
X, y = [], []
for gesture_idx, gesture in enumerate(GESTURES):
    gesture_dir = os.path.join(DATA_DIR, gesture)
    if not os.path.exists(gesture_dir):
        print(f"Error: Folder '{gesture_dir}' not found. Did you run record_gestures.py?")
        exit()

    for file in os.listdir(gesture_dir):
        if file.endswith(".npy"):
            X.append(np.load(os.path.join(gesture_dir, file)))
            y.append(gesture_idx)

if len(X) == 0:
    print("Error: No training data found. Please record gestures first.")
    exit()

# Convert to numpy arrays
X = np.array(X)
y = to_categorical(y)

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(GESTURES), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
print("Training started... (This may take 5-10 minutes)")
history = model.fit(X_train, y_train,
                    epochs=30,
                    validation_data=(X_test, y_test))

# Save model
model.save("sign_language_model.h5")
print(f"Model saved! Test accuracy: {np.max(history.history['val_accuracy']):.2f}")

# Optional: Save class labels for later use
with open('gesture_labels.txt', 'w') as f:
    f.write('\n'.join(GESTURES))

import os
import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

MODEL_FILE = "sign_language_model_custom.keras"
CLASS_INDICES_FILE = "class_indices.npy"
CONFIRM_SECONDS = 3
CONFIDENCE_THRESHOLD = 0.6

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"{MODEL_FILE} not found.")
model = load_model(MODEL_FILE)

# Load class mapping
if not os.path.exists(CLASS_INDICES_FILE):
    raise FileNotFoundError(f"{CLASS_INDICES_FILE} not found.")
class_indices = np.load(CLASS_INDICES_FILE, allow_pickle=True).item()
labels = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]
print("Loaded labels:", labels)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open webcam.")

sentence = ""
current_letter = None
start_time = None

print("✅ Webcam started. Hold a sign steady for", CONFIRM_SECONDS, "seconds to confirm.")
print("Press 'c' to clear sentence, 'q' to quit.\n")

def draw_progress_bar(frame, progress, x=10, y=80, width=300, height=20):
    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)
    fill = int(width * progress)
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + height), (0, 255, 0), -1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1) #mirror
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    predicted_char = None
    confidence = 0.0
    probs_text = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min = int(min(x_coords) * w) - 40
            y_min = int(min(y_coords) * h) - 40
            x_max = int(max(x_coords) * w) + 40
            y_max = int(max(y_coords) * h) + 40
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)

            # Extract ROI
            roi = frame[y_min:y_max, x_min:x_max]
            if roi.size > 0:
                # Preprocessing EXACTLY like dataset collection
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (28, 28))
                # Do NOT normalize (model trained on raw 0–255)
                inp = gray.reshape(1, 28, 28, 1).astype("float32")

                preds = model.predict(inp, verbose=0)[0]
                pred_idx = int(np.argmax(preds))
                confidence = float(preds[pred_idx])

                # safe index
                if pred_idx < len(labels):
                    predicted_char = labels[pred_idx]
                else:
                    predicted_char = None

                # top3 for display
                top3_idx = preds.argsort()[-3:][::-1]
                probs_text = " | ".join(f"{labels[i]}:{preds[i]:.2f}" for i in top3_idx if i < len(labels))

            # draw box and landmarks
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # display detection
    if predicted_char and confidence >= CONFIDENCE_THRESHOLD:
        cv2.putText(frame, f"Detected: {predicted_char} ({confidence:.2f})", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Detected: -", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 140, 255), 2)

    # top3 text
    if probs_text:
        y0 = 120
        for i, chunk in enumerate(probs_text.split(" | ")):
            cv2.putText(frame, chunk, (10, y0 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # confirm letter if steady for few seconds
    if predicted_char and confidence >= CONFIDENCE_THRESHOLD:
        if current_letter == predicted_char:
            elapsed = time.time() - start_time if start_time else 0
            progress = min(elapsed / CONFIRM_SECONDS, 1.0)
            draw_progress_bar(frame, progress)
            if elapsed >= CONFIRM_SECONDS:
                sentence += predicted_char
                print(f"✅ CONFIRMED: {predicted_char}")
                current_letter = None
                start_time = None
        else:
            current_letter = predicted_char
            start_time = time.time()
    else:
        current_letter = None
        start_time = None

    # show sentence
    cv2.putText(frame, f"Sentence: {sentence}", (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    cv2.imshow("Sign Language Recognition", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    if k == ord('c'):
        sentence = ""

cap.release()
cv2.destroyAllWindows()

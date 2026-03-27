import cv2
import os
import time
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Base directory to store dataset
base_dir = "dataset_custom"
os.makedirs(base_dir, exist_ok=True)

# Camera setup
cap = cv2.VideoCapture(0)
print("📷 Press a letter key (A–Z except J, Z) to capture samples for that sign.")
print("Press 'Q' to quit.\n")

current_label = None
capture_count = 0
delay = 0.2  # seconds between captures

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Detect hand
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
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)

            if current_label:
                roi = frame[y_min:y_max, x_min:x_max]
                if roi.size > 0:
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, (28,28))
                    save_dir = os.path.join(base_dir, current_label)
                    os.makedirs(save_dir, exist_ok=True)
                    filename = os.path.join(save_dir, f"{int(time.time()*1000)}.jpg")
                    cv2.imwrite(filename, gray)
                    capture_count += 1
                    time.sleep(delay)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display info
    cv2.putText(frame, f"Label: {current_label or '-'}  |  Captured: {capture_count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.imshow("Sign Data Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord('q'):
        break

    # Change current label
    elif 65 <= key <= 90 or 97 <= key <= 122:  # A–Z or a–z
        current_label = chr(key).upper()
        capture_count = 0
        print(f"⚡ Capturing images for '{current_label}'...")

cap.release()
cv2.destroyAllWindows()

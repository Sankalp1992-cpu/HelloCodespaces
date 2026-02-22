import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Zoom parameters
zoom_factor = 1.0
min_zoom = 1.0
max_zoom = 3.0

cap = cv2.VideoCapture(0)

def zoom_frame(frame, zoom):
    h, w = frame.shape[:2]
    new_w = int(w / zoom)
    new_h = int(h / zoom)

    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    x2 = x1 + new_w
    y2 = y1 + new_h

    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape

            # Get thumb tip and index tip
            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]

            x1, y1 = int(thumb.x * w), int(thumb.y * h)
            x2, y2 = int(index.x * w), int(index.y * h)

            # Draw points
            cv2.circle(frame, (x1, y1), 8, (0,255,0), -1)
            cv2.circle(frame, (x2, y2), 8, (0,255,0), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (255,0,0), 2)

            # Calculate distance
            distance = math.hypot(x2 - x1, y2 - y1)

            # Map distance to zoom
            zoom_factor = np.interp(distance, [30, 200], [1.0, 3.0])
            zoom_factor = max(min_zoom, min(max_zoom, zoom_factor))

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Apply zoom
    zoomed_frame = zoom_frame(frame, zoom_factor)

    cv2.putText(zoomed_frame, f'Zoom: {zoom_factor:.2f}x',
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,255), 2)

    cv2.imshow("Pinch to Zoom", zoomed_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

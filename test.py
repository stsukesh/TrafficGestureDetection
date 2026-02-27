import cv2
import numpy as np
import mediapipe as mp

print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")

cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✅ Camera is working! Press 'q' on the pop-up window to close.")
    while True:
        ret, frame = cap.read()
        cv2.imshow('Review Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("❌ Camera not found. Check privacy settings.")
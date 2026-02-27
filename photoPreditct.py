import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1) # static_image_mode=True is key!

image_path = 'C:\Users\vasiv\OneDrive\Desktop\TrafficSignalProject\policeman-gesturing-to-stop.webp' 
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not find image at {image_path}")
else:
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # 3. Simple Logic Example: Check if right hand is above head (Stop Signal)
        # Landmarks: 16 is Right Wrist, 0 is Nose
        right_wrist_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
        nose_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y
        
        label = "Normal"
        if right_wrist_y < nose_y:
            label = "STOP SIGNAL DETECTED"

        # Display result
        cv2.putText(image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Gesture Prediction', image)
        cv2.waitKey(0) # Waits until you press a key to close
        cv2.destroyAllWindows()
    else:
        print("No person detected in the image.")
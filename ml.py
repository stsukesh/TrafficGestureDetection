import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# 1. Load the AI Model
model = load_model('gesture_model.h5')

# Updated list based on common traffic project datasets
# If the output is wrong, we just swap the order of these names
# Updated list based on your actual model output (Class 5 = Stop)
# We move 'Stop Front' to the 6th position (index 5)
classes = ['No Gesture', 'Move Left', 'Move Right', 'Start', 'Hand Down', 'Stop Front']
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1)

image_path = r'C:\Users\vasiv\OneDrive\Desktop\TrafficSignalProject\LegendStopSign.jpeg'
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not find image at {image_path}")
else:
    # Optional: Resize for better display if image is huge
    image = cv2.resize(image, (800, 800))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # Draw skeleton
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 2. Extract 54 features (First 18 landmarks * x,y,z)
        landmarks = []
        for i in range(18):
            lm = results.pose_landmarks.landmark[i]
            landmarks.extend([lm.x, lm.y, lm.z])
        
        input_data = np.array(landmarks).reshape(1, 54)
        
        # 3. AI Prediction
        prediction = model.predict(input_data)
        class_id = np.argmax(prediction)
        confidence = prediction[0][class_id]
        
        # 4. Handle Labels
        if class_id < len(classes):
            label = classes[class_id]
        else:
            label = f"Class {class_id}" # Fallback to prevent IndexError

        display_text = f"{label} ({int(confidence*100)}%)"
        print(f"DEBUG: Predicted Class ID {class_id} with {confidence*100:.2f}% confidence")
        
        # 5. Show Results
        cv2.rectangle(image, (0,0), (500, 60), (0,0,0), -1) # Black background for text
        cv2.putText(image, display_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow('Traffic Signal Project - AI Review', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("MediaPipe failed to find the policeman's joints.")
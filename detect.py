import cv2
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from data import BodyPart
from ml import Movenet

# Load MoveNet and the model
movenet = Movenet('movenet_thunder')
model = load_model('gesture_model.h5')

# Class names and poses
class_n = ['NoPose', 'Pose1', 'Pose2', 'Pose3', 'Pose4', 'Pose5', 'Pose6', 'Pose7']
class_names = np.array(class_n)
poses = {
    "NoPose": "Unknown Pose",
    "Pose1": "Stop vehicles from left and right",
    "Pose2": "Stop from front",
    "Pose3": "Stop vehicles from behind",
    "Pose4": "Start vehicle from left",
    "Pose5": "Start vehicle from right",
    "Pose6": "Start vehicle on T point",
    "Pose7": "Stop vehicles from front and back"
}

# Process Tensor Image
def process_tensor_image(image):
    if isinstance(image, str):
        image = tf.io.read_file(image)
        image = tf.image.decode_jpeg(image)
    else:
        image = tf.convert_to_tensor(image, dtype=tf.float32)
    return image

# Detect function
def detect(input_tensor, inference_count=3):
    """Detect pose from an input tensor."""
    movenet.detect(input_tensor.numpy(), reset_crop_region=True)
    for _ in range(inference_count - 1):
        person = movenet.detect(input_tensor.numpy(), reset_crop_region=False)
    return person

# Classify pose
def classify_pose(image, face):
    image = process_tensor_image(image)
    person = detect(image)
    pose_landmarks = np.array(
        [[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
         for keypoint in person.keypoints],
        dtype=np.float32
    )
    coordinates = pose_landmarks.flatten().astype(str).tolist()
    df = pd.DataFrame([coordinates]).reset_index(drop=True)
    X = df.astype('float64').to_numpy()
    X = np.insert(X, 0, face, axis=1)
    X = np.insert(X, 0, face, axis=1)
    X = np.insert(X, 0, face, axis=1)
    y = model.predict(X)
    y_pred = [class_names[i] for i in np.argmax(y, axis=1)]
    return poses[y_pred[0]]

# Main function for real-time pose detection
def main():
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    print("Starting detection. Press 'q' to exit.")
    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Detect face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        face = 1 if len(faces) > 0 else 0

        # Resize frame for faster processing
        frame_height, frame_width, _ = frame.shape
        resized_frame = cv2.resize(frame, (int(frame_width * (320 / frame_height)), 320))

        # Classify pose
        start_time = time.time()
        pose = classify_pose(resized_frame, face)
        end_time = time.time()
        detection_time = end_time - start_time

        if face == 1:
            face_detected = "Face detected"
        else:
            face_detected = "Face not detected"

        # Display FPS and pose
        fps = int(1.0 / detection_time) if detection_time > 0 else 0
        cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'POSE: {pose}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f'{face_detected}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display frame
        cv2.imshow('Traffic Sign Gesture Detection', frame)

        # Wait for 1 second for the next frame
        time.sleep(1)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

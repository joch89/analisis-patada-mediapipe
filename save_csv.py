# save_csv.py

import sys
import os
import cv2
import mediapipe as mp
import pandas as pd

# Añadir el directorio raíz del proyecto
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from core.pose_utils import get_landmark_coords

VIDEO_PATH = 'videos/Costados_5.mp4'
OUTPUT_CSV = 'output_landmarks_data.csv'

mp_pose = mp.solutions.pose

def setup_mediapipe_detector():
    return mp.solutions.pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: No se puede abrir el archivo de video en {VIDEO_PATH}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose_detector = setup_mediapipe_detector()
    landmarks_data = []

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(rgb_frame)

        if results.pose_landmarks:
            frame_landmarks = {'frame': frame_idx}
            for i, lm in enumerate(results.pose_landmarks.landmark):
                frame_landmarks[f'x_{i}'] = lm.x
                frame_landmarks[f'y_{i}'] = lm.y
            landmarks_data.append(frame_landmarks)

        frame_idx += 1

    cap.release()

    # Guardar CSV
    df = pd.DataFrame(landmarks_data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Landmarks guardados en {OUTPUT_CSV}")

if __name__ == '__main__':
    main()

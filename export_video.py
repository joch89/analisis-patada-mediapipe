# export_video.py

import sys
import os
import cv2
import mediapipe as mp
from core.pose_utils import kick_counter_logic, draw_pose_landmarks, draw_kick_history

VIDEO_PATH = 'videos/Costados_5.mp4'
OUTPUT_VIDEO = 'output_video.mp4'
DISPLAY_WIDTH = 720  # tamaño para exportar

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

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
    fps = cap.get(cv2.CAP_PROP_FPS)

    pose_detector = setup_mediapipe_detector()
    mp_drawing = mp.solutions.drawing_utils

    # Configurar writer de video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    kick_counter = 0
    kick_stage = "LISTO"
    angle_history = []
    recorded_kick_angles = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks
            prev_kick_count = kick_counter
            kick_counter, kick_stage, knee_angle = kick_counter_logic(
                landmarks, width, height, kick_counter, kick_stage, angle_history
            )

            if kick_counter > prev_kick_count:
                recorded_kick_angles.append(knee_angle)

            frame = draw_pose_landmarks(frame, results, mp_drawing)
            frame = draw_kick_history(frame, width, recorded_kick_angles)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Video exportado en {OUTPUT_VIDEO}")

if __name__ == '__main__':
    main()

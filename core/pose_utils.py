# kick_analyzer/core/pose_utils.py

import numpy as np
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
PoseLandmark = mp.solutions.pose.PoseLandmark


def calculate_angle(a: tuple, b: tuple, c: tuple) -> float:
    """Calcula el ángulo en grados (0-180) entre tres puntos (a, b, c), con 'b' como vértice."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle


def get_landmark_coords(landmarks, landmark_index: int, width: int, height: int) -> tuple[int, int]:
    """Extrae las coordenadas X, Y de un landmark y las escala a píxeles."""
    if landmarks.landmark:
        lm = landmarks.landmark[landmark_index]
        x = int(lm.x * width)
        y = int(lm.y * height)
        return (x, y)
    return (0, 0)


def kick_counter_logic(landmarks, width, height, counter: int, stage: str, data_history: list) -> tuple[int, str, float]:
    """Lógica de conteo de patadas y cálculo de ángulo de cadera."""
    hip_angle = 180.0
    try:
        right_knee_coords = get_landmark_coords(landmarks, PoseLandmark.RIGHT_KNEE.value, width, height)
        right_hip_coords = get_landmark_coords(landmarks, PoseLandmark.RIGHT_HIP.value, width, height)
        left_knee_coords = get_landmark_coords(landmarks, PoseLandmark.LEFT_KNEE.value, width, height)

        hip_angle = calculate_angle(right_knee_coords, right_hip_coords, left_knee_coords)

        right_ankle_lm = landmarks.landmark[PoseLandmark.RIGHT_ANKLE.value]
        right_hip_lm = landmarks.landmark[PoseLandmark.RIGHT_HIP.value]

        if right_ankle_lm.y < right_hip_lm.y:
            if stage != "PATEANDO":
                counter += 1
                stage = "PATEANDO"
        else:
            if stage == "PATEANDO":
                stage = "LISTO"

    except Exception:
        pass

    return counter, stage, hip_angle


def draw_pose_landmarks(image, detection_results, mp_drawing):
    """Dibuja los landmarks de la pose en la imagen."""
    if detection_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            detection_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=5, circle_radius=5),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=5, circle_radius=5)
        )
    return image


def draw_kick_history(annotated_image, width, kick_history):
    """Dibuja el contador general y el historial de ángulos de patadas."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    color = (0, 0, 255)

    counter_text = f'Total Patadas: {len(kick_history)}'
    counter_size = cv2.getTextSize(counter_text, font, font_scale, font_thickness)[0]
    x = width - counter_size[0] - 20
    y = 40

    cv2.putText(annotated_image, counter_text, (x, y), font, font_scale, color, font_thickness, cv2.LINE_AA)

    line_spacing = 30
    for i, angle in enumerate(kick_history[-10:]):
        text = f'Patada {i + 1}: {int(angle)} grados'
        text_y = y + (i + 1) * line_spacing
        cv2.putText(annotated_image, text, (x, text_y), font, 0.7, color, 2, cv2.LINE_AA)

    return annotated_image


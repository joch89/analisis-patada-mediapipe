# main.py

import sys
import os
import cv2
import mediapipe as mp
import time

# --- CONFIGURACIÓN DE VISUALIZACIÓN ---
DISPLAY_WIDTH = 360
# --------------------------------------

# Añadir el directorio raíz del proyecto
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Importaciones
from core.pose_utils import kick_counter_logic, draw_pose_landmarks, get_landmark_coords

# Configuración del video
VIDEO_PATH = 'videos/Costados_5.mp4'
mp_drawing = mp.solutions.drawing_utils


# --- Setup de MediaPipe ---
def setup_mediapipe_detector():
    """Inicializa y retorna el detector de pose de MediaPipe."""
    return mp.solutions.pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


# --- Función auxiliar para mostrar el historial de patadas ---
def draw_kick_history(annotated_image, width, kick_history):
    """
    Dibuja el contador general y el historial de ángulos de cada patada
    en la esquina superior derecha de la pantalla.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    color = (0, 0, 255)  # Rojo (BGR)

    # Texto principal (contador total)
    counter_text = f'Total Patadas: {len(kick_history)}'
    counter_size = cv2.getTextSize(counter_text, font, font_scale, font_thickness)[0]
    x = width - counter_size[0] - 20
    y = 40  # posición inicial desde arriba

    cv2.putText(annotated_image, counter_text, (x, y),
                font, font_scale, color, font_thickness, cv2.LINE_AA)

    # Mostrar historial debajo
    line_spacing = 30
    for i, angle in enumerate(kick_history[-10:]):  # muestra las últimas 10 patadas
        text = f'Patada {i + 1}: {int(angle)}°'
        text_y = y + (i + 1) * line_spacing
        cv2.putText(annotated_image, text, (x, text_y),
                    font, 0.7, color, 2, cv2.LINE_AA)

    return annotated_image


# --- Bucle Principal ---
def main():
    kick_counter = 0
    kick_stage = "LISTO"
    angle_history = []        # guarda todos los ángulos de cada patada detectada
    recorded_kick_angles = [] # lista para mostrar historial visual

    pose_detector = setup_mediapipe_detector()
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: No se puede abrir el archivo de video en {VIDEO_PATH}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks

            # Lógica de conteo y ángulo actual
            prev_kick_count = kick_counter
            kick_counter, kick_stage, knee_angle = kick_counter_logic(
                landmarks, width, height, kick_counter, kick_stage, angle_history
            )

            # Si se detectó una nueva patada, registrar su ángulo
            if kick_counter > prev_kick_count:
                recorded_kick_angles.append(knee_angle)

            # Dibujar landmarks
            frame = draw_pose_landmarks(frame, results, mp_drawing)

            # Mostrar métricas en la izquierda
            cv2.putText(frame, f'Patadas: {kick_counter}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Etapa: {kick_stage}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Angulo Cadera: {int(knee_angle)}°', (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Mostrar historial de patadas en la derecha
            frame = draw_kick_history(frame, width, recorded_kick_angles)

        # Redimensionar para mostrar
        display_height = int(frame.shape[0] * (DISPLAY_WIDTH / frame.shape[1]))
        display_frame = cv2.resize(frame, (DISPLAY_WIDTH, display_height))

        cv2.imshow('Analisis de Patada de Costado', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

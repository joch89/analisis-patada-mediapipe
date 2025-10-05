import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
import os


# --- Obtener ruta del script ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- CSV en la carpeta raíz del proyecto ---
csv_file = os.path.join(script_dir, "output_landmarks_data.csv")

# --- Verificar existencia ---
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"No se encontró el archivo CSV: {csv_file}")

print(f"Leyendo CSV desde: {csv_file}")
landmarks_df = pd.read_csv(csv_file)

# --- Función para calcular ángulo entre tres puntos ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

# --- IDs de landmarks relevantes ---
RIGHT_ANKLE_ID = 28
RIGHT_HIP_ID = 24
RIGHT_KNEE_ID = 26
LEFT_KNEE_ID = 25

# --- Extraer coordenadas ---
combined_data = pd.DataFrame({
    'tobillo_x': landmarks_df[f'x_{RIGHT_ANKLE_ID}'],
    'tobillo_y': landmarks_df[f'y_{RIGHT_ANKLE_ID}'],
    'cadera_x': landmarks_df[f'x_{RIGHT_HIP_ID}'],
    'cadera_y': landmarks_df[f'y_{RIGHT_HIP_ID}'],
    'rodilla_derecha_x': landmarks_df[f'x_{RIGHT_KNEE_ID}'],
    'rodilla_derecha_y': landmarks_df[f'y_{RIGHT_KNEE_ID}'],
    'rodilla_izquierda_x': landmarks_df[f'x_{LEFT_KNEE_ID}'],
    'rodilla_izquierda_y': landmarks_df[f'y_{LEFT_KNEE_ID}']
})

# --- Detectar si tobillo está más arriba que cadera ---
combined_data['tobillo_mas_alto_que_cadera'] = combined_data['tobillo_y'] < combined_data['cadera_y']

# --- Detectar eventos de inicio y fin de patadas ---
combined_data['inicio_patada'] = (~combined_data['tobillo_mas_alto_que_cadera'].shift(1).fillna(False)) & (combined_data['tobillo_mas_alto_que_cadera'])
combined_data['fin_patada'] = (combined_data['tobillo_mas_alto_que_cadera'].shift(1).fillna(False)) & (~combined_data['tobillo_mas_alto_que_cadera'])

kick_starts = combined_data[combined_data['inicio_patada']].index.tolist()
kick_ends = combined_data[combined_data['fin_patada']].index.tolist()

paired_kicks = []
i = j = 0
while i < len(kick_starts) and j < len(kick_ends):
    if kick_starts[i] < kick_ends[j]:
        paired_kicks.append((kick_starts[i], kick_ends[j]))
        i += 1
        j += 1
    elif kick_starts[i] > kick_ends[j]:
        j += 1
    else:
        paired_kicks.append((kick_starts[i], kick_ends[j]))
        i += 1
        j += 1

if i < len(kick_starts):
    last_frame = len(landmarks_df) - 1
    paired_kicks.append((kick_starts[i], last_frame))

print("Patadas detectadas (método tobillo):")
for idx, (start, end) in enumerate(paired_kicks, 1):
    print(f"Patada {idx}: Inicio {start}, Fin {end}")

# --- Calcular ángulo de cadera ---
hip_angles = []
for idx, row in combined_data.iterrows():
    angle = calculate_angle(
        [row['rodilla_derecha_x'], row['rodilla_derecha_y']],
        [row['cadera_x'], row['cadera_y']],
        [row['rodilla_izquierda_x'], row['rodilla_izquierda_y']]
    )
    hip_angles.append(angle)

combined_data['angulo_cadera'] = hip_angles

# --- Derivadas ---
combined_data['velocidad_cadera'] = combined_data['angulo_cadera'].diff()
combined_data['aceleracion_cadera'] = combined_data['velocidad_cadera'].diff()
combined_data['jerk_cadera'] = combined_data['aceleracion_cadera'].diff()

# --- Suavizado para picos ---
window_size = 5
combined_data['velocidad_cadera_suave'] = combined_data['velocidad_cadera'].rolling(window=window_size, center=True).mean()
combined_data['aceleracion_cadera_suave'] = combined_data['aceleracion_cadera'].rolling(window=window_size, center=True).mean()

# --- Detección de picos de velocidad ---
pos_peaks_idx, _ = find_peaks(combined_data['velocidad_cadera_suave'].dropna(), height=5, distance=10)
neg_peaks_idx, _ = find_peaks(-combined_data['velocidad_cadera_suave'].dropna(), height=5, distance=10)
picos_positivos = combined_data.iloc[pos_peaks_idx].index.tolist()
picos_negativos = combined_data.iloc[neg_peaks_idx].index.tolist()

# --- Refinar kicks usando derivadas ---
refined_kicks = []
for start_ankle, end_ankle in paired_kicks:
    segment = combined_data.loc[start_ankle:end_ankle]
    max_vel_frame = segment['velocidad_cadera_suave'].idxmax() if segment['velocidad_cadera_suave'].notna().any() else start_ankle
    min_vel_frame = segment['velocidad_cadera_suave'].idxmin() if segment['velocidad_cadera_suave'].notna().any() else end_ankle

    start_frame_refined = max_vel_frame
    end_frame_refined = min_vel_frame

    if start_frame_refined < end_frame_refined:
        refined_kicks.append((start_frame_refined, end_frame_refined))
    else:
        refined_kicks.append((start_ankle, end_ankle))

print("\nPatadas refinadas (método derivadas):")
for idx, (start, end) in enumerate(refined_kicks, 1):
    print(f"Patada {idx}: Inicio {start}, Fin {end}")

# --- Graficar resultados ---
plt.figure(figsize=(15, 15))
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'cyan']

# 1) Ángulo de cadera
plt.subplot(4,1,1)
plt.plot(combined_data.index, combined_data['angulo_cadera'], label='Ángulo de cadera')
for idx, (start, end) in enumerate(refined_kicks):
    plt.axvspan(start, end, color=colors[idx % len(colors)], alpha=0.2)
    plt.text(start, max(combined_data['angulo_cadera']), f'Patada {idx+1}', color=colors[idx % len(colors)], fontsize=10, verticalalignment='top')
plt.ylabel('Ángulo (°)')
plt.title('Ángulo de cadera con intervalos de patadas')
plt.legend()
plt.grid(True)

# 2) Velocidad
plt.subplot(4,1,2)
plt.plot(combined_data.index, combined_data['velocidad_cadera'], alpha=0.5, label='Velocidad')
plt.plot(combined_data.index, combined_data['velocidad_cadera_suave'], color='darkorange', label='Velocidad suavizada')
plt.scatter(picos_positivos, combined_data.loc[picos_positivos, 'velocidad_cadera_suave'], color='green', marker='^', s=80, label='Peaks positivos')
plt.scatter(picos_negativos, combined_data.loc[picos_negativos, 'velocidad_cadera_suave'], color='purple', marker='v', s=80, label='Peaks negativos')
plt.ylabel('Velocidad (°/frame)')
plt.title('Velocidad angular de cadera')
plt.legend()
plt.grid(True)

# 3) Aceleración
plt.subplot(4,1,3)
plt.plot(combined_data.index, combined_data['aceleracion_cadera'], alpha=0.5, label='Aceleración')
plt.plot(combined_data.index, combined_data['aceleracion_cadera_suave'], color='darkgreen', label='Aceleración suavizada')
plt.axhline(0, color='gray', linestyle='--')
plt.ylabel('Aceleración (°/frame²)')
plt.title('Aceleración angular de cadera')
plt.legend()
plt.grid(True)

# 4) Jerk
plt.subplot(4,1,4)
plt.plot(combined_data.index, combined_data['jerk_cadera'], alpha=0.5, label='Jerk', color='red')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Frame')
plt.ylabel('Jerk (°/frame³)')
plt.title('Jerk angular de cadera')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.subplots_adjust(hspace=0.5)
plt.show()

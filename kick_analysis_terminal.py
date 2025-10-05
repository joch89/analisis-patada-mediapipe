import os
import pandas as pd

# --- Obtener ruta del script ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- CSV en la carpeta raíz del proyecto ---
csv_file = os.path.join(script_dir, "output_landmarks_data.csv")

# --- Verificar existencia ---
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"No se encontró el archivo CSV: {csv_file}")

print(f"Leyendo CSV desde: {csv_file}")

# --- Leer CSV ---
landmarks_df = pd.read_csv(csv_file)

# --- Mostrar primeras filas ---
print("Primeras filas del CSV:")
print(landmarks_df.head())

# --- Información general ---
print("\nInformación del CSV:")
print(landmarks_df.info())

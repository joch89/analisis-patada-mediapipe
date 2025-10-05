import os
import pandas as pd

# Obtener la carpeta donde está el script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construir ruta absoluta al CSV
csv_file = os.path.join(script_dir, "outputs", "landmarks_data.csv")
print(f"Leyendo CSV desde: {csv_file}")

# Leer CSV
landmarks_df = pd.read_csv(csv_file)

# Mostrar primeras filas para confirmar
print(landmarks_df.head())

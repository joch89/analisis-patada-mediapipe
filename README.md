# 🥋 Análisis de Patada de Costado con MediaPipe y OpenCV

**Autor:** José Chamorro  
**Profesión:** Kinesiólogo · Quiropráctico · Especialista en Biomecánica y entusiasta en Data Science.  
**Repositorio:** [github.com/joch89](https://github.com/joch89)

---

## 🎯 Descripción del proyecto

Este proyecto utiliza **MediaPipe**, **OpenCV** y **Python** para detectar, registrar y analizar el movimiento de una **patada de costado (yoko geri)** en taekwondo.

El objetivo es desarrollar una herramienta que permita analizar el rendimiento biomecánico del movimiento, proporcionando métricas como:

- Ángulo de cadera y rodilla  
- Velocidad, aceleración y jerk angular  
- Detección automática de inicio y fin de patadas  
- Contador de patadas y visualización en video  


## 🧩 Estructura del proyecto

- `kick_analyzer/` - Carpeta principal del proyecto  
  - `core/` - Funciones y utilidades centrales  
    - `pose_utils.py` - Funciones para procesar poses y calcular métricas  
  - `videos/` - Carpeta para videos de entrada 
  - `export_video.py` - Script para procesar video y exportar resultados visuales  
  - `save_csv.py` - Script para guardar datos de landmarks en CSV  
  - `kick_analysis.py` - Script para análisis y visualización de datos


## ⚙️ Flujo de trabajo

### 1️⃣ Extraer datos del video
Guarda los landmarks detectados en un CSV.

```bash
python save_csv.py
```

### 2️⃣ Analizar biomecánicamente la patada
Genera gráficos de ángulo, velocidad, aceleración y jerk angular.

```bash
python kick_analysis.py
```

### 3️⃣ Exportar video con resultados

Dibuja los landmarks, contador y ángulos sobre el video original.
```bash
python export_video.py
```

🎥 Salida: outputs/output_video.mp4



## 📦 Dependencias principales

Instala los requisitos con:
```bash
pip install opencv-python mediapipe pandas matplotlib scipy numpy
```


## 🧠 Futuras mejoras
- Comparador automático entre deportistas o sesiones.
- Integración con métricas de flexibilidad y estabilidad postural.



## 🩺 Aplicación en ciencias del movimiento

Este sistema permite combinar análisis biomecánico con herramientas de Data Science aplicadas a la salud y el rendimiento deportivo, facilitando la cuantificación objetiva del movimiento.




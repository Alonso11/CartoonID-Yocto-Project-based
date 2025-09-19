# 🎬 CartoonID - Sistema de Detección de Personajes de Shrek

> Sistema embebido multimedia con Yocto Project y GStreamer para detección en tiempo real de personajes de Shrek

## 📋 Descripción del Proyecto

Este proyecto implementa un **sistema operativo Linux personalizado** usando **Yocto Project** para aplicaciones multimedia embebidas. El sistema incluye un **pipeline de GStreamer** con **inteligencia artificial YOLO** para detectar personajes de las películas de Shrek en videos en tiempo real.

### 🎭 Personajes Detectados
- **Shrek**
- **Fiona**
- **Donkey**




## 📁 Estructura del Proyecto
  ```
  📦 CartoonID-Yocto-Project
  ┣ 📂 docs/
  ┣ 📂 sources/
  ┃ ┣  shrek_pipeline.py
  ┃ ┣  best.pt
  ┃ ┗  plugins/python/
  ┗ ⚙ set_gstreamer_envs.sh
```
  | Archivo | Descripción |
  |---------|-------------|
  | `shrek_pipeline.py` | Pipeline GStreamer principal |
  | `best.pt` | Modelo YOLO entrenado |
  | `yolo_batch_detector.py` | Plugin de detección IA |


## 🛠️ Tecnologías Utilizadas

- **Yocto Project** - Generación de imagen Linux personalizada
- **GStreamer** - pipeline de trabajo multimedia
- **YOLO + Ultralytics** - Detección de personajes
- **Python 3.9+** - Implementación de pipelines
- **OpenCV** - Procesamiento de imágenes

## 👥 Desarrolladores
  Este proyecto fue desarrollado por:

      [Lizzy Gonzalez Alvarado] • [Bonilla Blanco] • [Fabian Gomez Quesada]


## 🔧 Requisitos




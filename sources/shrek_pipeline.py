#!/usr/bin/env python3
"""
CartoonID Shrek Detection Pipeline
Pipeline completo de GStreamer 
"""

import os
import sys

# Configurar variables de entorno ANTES de importar GStreamer
def setup_gstreamer_environment():
    """Configurar variables de entorno necesarias para GStreamer"""
    # Obtener directorio del proyecto
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Configurar paths de plugins
    gst_plugin_path = os.environ.get('GST_PLUGIN_PATH', '')
    plugins_path = os.path.join(current_dir, 'plugins')

    if plugins_path not in gst_plugin_path:
        if gst_plugin_path:
            os.environ['GST_PLUGIN_PATH'] = f"{gst_plugin_path}:{plugins_path}"
        else:
            os.environ['GST_PLUGIN_PATH'] = plugins_path

    # Configurar PYTHONPATH
    python_path = os.environ.get('PYTHONPATH', '')
    if plugins_path not in python_path:
        if python_path:
            os.environ['PYTHONPATH'] = f"{python_path}:{plugins_path}"
        else:
            os.environ['PYTHONPATH'] = plugins_path

    print(f"🔧 GST_PLUGIN_PATH: {os.environ.get('GST_PLUGIN_PATH')}")
    print(f"🐍 PYTHONPATH: {os.environ.get('PYTHONPATH')}")

# Configurar entorno ANTES de cualquier importación de GStreamer
setup_gstreamer_environment()

# Ahora importar GStreamer
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')

from gi.repository import Gst, GObject, GLib
import argparse
import time
from pathlib import Path

class ShrekDetectionPipeline:
    """Pipeline principal para detección de personajes de Shrek"""

    def __init__(self, video_path, model_path="best.pt", show_fps=True, save_output=False, output_path="output.mp4"):
        self.video_path = video_path
        self.model_path = model_path
        self.show_fps = show_fps
        self.save_output = save_output
        self.output_path = output_path

        # Variables del pipeline
        self.pipeline = None
        self.bus = None
        self.loop = None

        # Estadísticas
        self.start_time = time.time()
        self.frame_count = 0

        # Inicializar GStreamer y refrescar plugins
        Gst.init(None)

        # Forzar registro de plugins
        registry = Gst.Registry.get()
        registry.scan_path(os.path.join(os.path.dirname(__file__), 'plugins'))

        # Verificar que el plugin python está disponible
        python_plugin = Gst.Registry.get().find_plugin("python")
        if not python_plugin:
            print("⚠️  Plugin 'python' no encontrado, intentando alternativa...")
        else:
            print("✅ Plugin 'python' encontrado correctamente")

    def create_pipeline(self):
        """Crear el pipeline de GStreamer completo"""

        # Verificar que el archivo de video existe
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        # Verificar que el modelo existe
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Pipeline base para video desde archivo
        pipeline_str = f"""
            filesrc location="{self.video_path}" !
            decodebin !
            videoscale !
            videoconvert !
            video/x-raw,format=RGB,width=640,height=480 !
            python module=yolo_batch_detector
                model-path="{self.model_path}"
                confidence=0.3
                inference-size=640
                batch-size=4
                show-fps={str(self.show_fps).lower()} !
            videoconvert !
        """

        # Salida: display o archivo
        if self.save_output:
            # Guardar a archivo MP4
            pipeline_str += f"""
                x264enc tune=zerolatency bitrate=2000 !
                mp4mux !
                filesink location="{self.output_path}"
            """
        else:
            # Mostrar en pantalla
            pipeline_str += """
                autovideosink sync=false
            """

        print(f"Pipeline GStreamer:")
        print(pipeline_str)
        print("-" * 50)

        # Crear pipeline
        self.pipeline = Gst.parse_launch(pipeline_str)

        if not self.pipeline:
            raise RuntimeError("Failed to create GStreamer pipeline")

        # Configurar bus para mensajes
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self.on_message)

        return True

    def on_message(self, bus, message):
        """Manejar mensajes del pipeline"""
        mtype = message.type

        if mtype == Gst.MessageType.EOS:
            print("\\nEnd of stream reached")
            self.stop_pipeline()

        elif mtype == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"\\nError: {err}")
            print(f"Debug: {debug}")
            self.stop_pipeline()

        elif mtype == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            print(f"Warning: {warn}")

        elif mtype == Gst.MessageType.INFO:
            info, debug = message.parse_info()
            print(f"Info: {info}")

        elif mtype == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old_state, new_state, pending_state = message.parse_state_changed()
                print(f"Pipeline state changed from {old_state.value_nick} to {new_state.value_nick}")

        return True

    def start_pipeline(self):
        """Iniciar el pipeline"""
        print("Starting Shrek detection pipeline...")
        print(f"Video: {self.video_path}")
        print(f"Model: {self.model_path}")
        print(f"Output: {'File: ' + self.output_path if self.save_output else 'Display'}")
        print("-" * 50)

        # Cambiar estado a PLAYING
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Unable to set pipeline to PLAYING state")

        # Main loop
        self.loop = GLib.MainLoop()

        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\\nInterrupted by user")
            self.stop_pipeline()

    def stop_pipeline(self):
        """Detener el pipeline"""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)

        if self.loop:
            self.loop.quit()

        # Mostrar estadísticas finales
        total_time = time.time() - self.start_time
        print(f"\\nPipeline stopped")
        print(f"Total processing time: {total_time:.2f} seconds")
        if self.save_output and os.path.exists(self.output_path):
            print(f"Output saved to: {self.output_path}")


def find_video_files(directory="."):
    """Encontrar archivos de video en el directorio"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
    video_files = []

    for ext in video_extensions:
        video_files.extend(Path(directory).glob(f"*{ext}"))
        video_files.extend(Path(directory).glob(f"*{ext.upper()}"))

    return sorted(video_files)


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="CartoonID Shrek Detection Pipeline")
    parser.add_argument("--video", "-v", type=str, help="Path to video file")
    parser.add_argument("--model", "-m", type=str, default="best.pt", help="Path to YOLO model")
    parser.add_argument("--output", "-o", type=str, help="Output video file (optional)")
    parser.add_argument("--no-fps", action="store_true", help="Hide FPS display")
    parser.add_argument("--list-videos", action="store_true", help="List available video files")

    args = parser.parse_args()

    # Listar videos disponibles
    if args.list_videos:
        videos = find_video_files()
        if videos:
            print("Available video files:")
            for i, video in enumerate(videos, 1):
                print(f"  {i}. {video}")
        else:
            print("No video files found in current directory")
        return

    # Determinar archivo de video
    video_path = args.video
    if not video_path:
        # Buscar videos automáticamente
        videos = find_video_files()
        if videos:
            video_path = str(videos[0])
            print(f"Using first video found: {video_path}")
        else:
            print("No video file specified and none found. Use --video or place a video file in the directory.")
            return

    # Configurar salida
    save_output = args.output is not None
    output_path = args.output if save_output else "shrek_detected_output.mp4"
    show_fps = not args.no_fps

    try:
        # Crear y ejecutar pipeline
        pipeline = ShrekDetectionPipeline(
            video_path=video_path,
            model_path=args.model,
            show_fps=show_fps,
            save_output=save_output,
            output_path=output_path
        )

        pipeline.create_pipeline()
        pipeline.start_pipeline()

    except KeyboardInterrupt:
        print("\\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
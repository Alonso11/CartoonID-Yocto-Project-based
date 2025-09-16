#!/usr/bin/env python
# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

# yolo_batch_detector.py
# GStreamer plugin for YOLO batch inference on video frames
# Synchronized version to avoid timestamping issues - FIXED VERSION

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('GstVideo', '1.0')

from gi.repository import Gst, GObject, GstBase, GstVideo
import numpy as np
import cv2
from ultralytics import YOLO
from collections import deque
import time

Gst.init(None)

class YoloBatchDetector(GstBase.BaseTransform):
    __gstmetadata__ = (
        'YOLO Batch Detector',
        'Transform/Video',
        'Synchronized YOLO batch inference on video frames',
        'AI Assistant'
    )

    # FIX: Definir caps más flexibles sin restricciones de formato específico
    __gsttemplates__ = (
        Gst.PadTemplate.new(
            "src",
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.from_string(
                "video/x-raw,format={RGB,BGR,RGBA,BGRA,RGBx,BGRx,xRGB,xBGR,ARGB,ABGR},"
                "width=(int)[1,MAX],height=(int)[1,MAX],framerate=(fraction)[0/1,MAX]"
            )
        ),
        Gst.PadTemplate.new(
            "sink",
            Gst.PadDirection.SINK,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.from_string(
                "video/x-raw,format={RGB,BGR,RGBA,BGRA,RGBx,BGRx,xRGB,xBGR,ARGB,ABGR},"
                "width=(int)[1,MAX],height=(int)[1,MAX],framerate=(fraction)[0/1,MAX]"
            )
        )
    )

    # Propiedades del plugin
    __gproperties__ = {
        "model-path": (
            str,
            "Model Path",
            "Path to YOLO model weights",
            "yolov8n.pt",
            GObject.ParamFlags.READWRITE
        ),
        "batch-size": (
            int,
            "Batch Size",
            "Number of frames to process in batch",
            1, 32, 8,
            GObject.ParamFlags.READWRITE
        ),
        "inference-size": (
            int,
            "Inference Size",
            "Input size for YOLO inference",
            320, 1280, 640,
            GObject.ParamFlags.READWRITE
        ),
        "confidence": (
            float,
            "Confidence Threshold",
            "Confidence threshold for detections",
            0.0, 1.0, 0.25,
            GObject.ParamFlags.READWRITE
        ),
        "show-fps": (
            bool,
            "Show FPS",
            "Display FPS and inference metrics on video",
            True,
            GObject.ParamFlags.READWRITE
        )
    }

    def __init__(self):
        GstBase.BaseTransform.__init__(self)
        
        # Configuración
        self.set_passthrough(False)
        self.set_in_place(True)
        
        # Propiedades
        self.model_path = "yolov8n.pt"
        self.batch_size = 8
        self.inference_size = 640
        self.confidence = 0.25
        self.show_fps = True
        
        # Variables internas
        self.model = None
        self.frame_buffer = deque(maxlen=32)  # Buffer para frames pendientes
        self.width = 0
        self.height = 0
        self.channels = 3
        self.format = "RGB"  # FIX: Almacenar formato actual
        self.bytes_per_pixel = 3  # FIX: Bytes por pixel según formato
        
        # Métricas
        self.frame_count = 0
        self.batch_count = 0
        self.total_inference_time = 0.0
        self.last_fps_time = time.time()
        self.fps = 0.0
        self.avg_inference_fps = 0.0
        self.detections_count = 0

    def do_get_property(self, prop):
        if prop.name == 'model-path':
            return self.model_path
        elif prop.name == 'batch-size':
            return self.batch_size
        elif prop.name == 'inference-size':
            return self.inference_size
        elif prop.name == 'confidence':
            return self.confidence
        elif prop.name == 'show-fps':
            return self.show_fps
        else:
            raise AttributeError('unknown property %s' % prop.name)

    def do_set_property(self, prop, value):
        if prop.name == 'model-path':
            self.model_path = value
            self._load_model()
        elif prop.name == 'batch-size':
            self.batch_size = value
        elif prop.name == 'inference-size':
            self.inference_size = value
        elif prop.name == 'confidence':
            self.confidence = value
        elif prop.name == 'show-fps':
            self.show_fps = value
        else:
            raise AttributeError('unknown property %s' % prop.name)

    def _load_model(self):
        """Carga el modelo YOLO"""
        try:
            Gst.info(f"Loading YOLO model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            # Test básico del modelo
            test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            test_results = self.model(test_frame, verbose=False)
            Gst.info(f"YOLO model loaded successfully - test inference completed")
            return True
        except Exception as e:
            Gst.error(f"Failed to load YOLO model: {e}")
            self.model = None
            return False

    def _get_format_info(self, format_str):
        """FIX: Obtiene información sobre el formato de pixel"""
        format_info = {
            'RGB': {'channels': 3, 'bytes_per_pixel': 3, 'opencv_code': None},
            'BGR': {'channels': 3, 'bytes_per_pixel': 3, 'opencv_code': cv2.COLOR_BGR2RGB},
            'RGBA': {'channels': 4, 'bytes_per_pixel': 4, 'opencv_code': cv2.COLOR_RGBA2RGB},
            'BGRA': {'channels': 4, 'bytes_per_pixel': 4, 'opencv_code': cv2.COLOR_BGRA2RGB},
            'RGBx': {'channels': 4, 'bytes_per_pixel': 4, 'opencv_code': cv2.COLOR_RGBA2RGB},
            'BGRx': {'channels': 4, 'bytes_per_pixel': 4, 'opencv_code': cv2.COLOR_BGRA2RGB},
            'xRGB': {'channels': 4, 'bytes_per_pixel': 4, 'opencv_code': cv2.COLOR_RGBA2RGB},
            'xBGR': {'channels': 4, 'bytes_per_pixel': 4, 'opencv_code': cv2.COLOR_BGRA2RGB},
            'ARGB': {'channels': 4, 'bytes_per_pixel': 4, 'opencv_code': cv2.COLOR_RGBA2RGB},
            'ABGR': {'channels': 4, 'bytes_per_pixel': 4, 'opencv_code': cv2.COLOR_BGRA2RGB}
        }
        return format_info.get(format_str, format_info['RGB'])

    def do_set_caps(self, incaps, outcaps):
        """Configurar caps de entrada y salida"""
        try:
            # FIX: Manejo mejorado de caps
            video_info = GstVideo.VideoInfo.new_from_caps(incaps)
            if video_info is not None:
                self.width = video_info.width
                self.height = video_info.height
                self.format = video_info.finfo.name
            else:
                # Fallback: extraer información manualmente
                caps_struct = incaps.get_structure(0)
                self.width = caps_struct.get_int('width')[1] if caps_struct.get_int('width')[0] else 640
                self.height = caps_struct.get_int('height')[1] if caps_struct.get_int('height')[0] else 480
                format_result = caps_struct.get_string('format')
                self.format = format_result if format_result else 'RGB'
            
            # FIX: Obtener información del formato
            format_info = self._get_format_info(self.format)
            self.channels = format_info['channels']
            self.bytes_per_pixel = format_info['bytes_per_pixel']
            
            Gst.info(f"Video caps: {self.width}x{self.height}, format={self.format}, "
                    f"channels={self.channels}, bpp={self.bytes_per_pixel}")
                
        except Exception as e:
            Gst.error(f"Failed to parse caps: {e}")
            # Valores por defecto seguros
            self.width = 640
            self.height = 480
            self.format = 'RGB'
            self.channels = 3
            self.bytes_per_pixel = 3
            
        # Cargar modelo si no está cargado
        if self.model is None:
            if not self._load_model():
                Gst.error("YOLO model not loaded")
                return False
            
        return True

    def _convert_frame_to_rgb(self, frame):
        """FIX: Convierte frame al formato RGB para YOLO"""
        format_info = self._get_format_info(self.format)
        
        if format_info['opencv_code'] is not None:
            # Convertir al formato RGB
            rgb_frame = cv2.cvtColor(frame, format_info['opencv_code'])
            return rgb_frame
        else:
            # Ya está en RGB o formato compatible
            return frame[:, :, :3] if frame.shape[2] > 3 else frame

    def _convert_frame_from_rgb(self, rgb_frame):
        """FIX: Convierte frame RGB de vuelta al formato original"""
        format_info = self._get_format_info(self.format)
        
        if self.format == 'BGR':
            return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        elif self.format in ['RGBA', 'RGBx']:
            # Agregar canal alfa
            alpha = np.ones((rgb_frame.shape[0], rgb_frame.shape[1], 1), dtype=rgb_frame.dtype) * 255
            return np.concatenate([rgb_frame, alpha], axis=2)
        elif self.format in ['BGRA', 'BGRx']:
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            alpha = np.ones((bgr_frame.shape[0], bgr_frame.shape[1], 1), dtype=bgr_frame.dtype) * 255
            return np.concatenate([bgr_frame, alpha], axis=2)
        else:
            # Para otros formatos, devolver RGB y ajustar canales si es necesario
            if self.channels == 4 and rgb_frame.shape[2] == 3:
                alpha = np.ones((rgb_frame.shape[0], rgb_frame.shape[1], 1), dtype=rgb_frame.dtype) * 255
                return np.concatenate([rgb_frame, alpha], axis=2)
            return rgb_frame

    def _process_batch_sync(self, frames):
        """Procesa un batch de frames síncronamente"""
        if not frames or self.model is None:
            return frames
            
        try:
            start_time = time.time()
            
            # FIX: Convertir frames a RGB para YOLO
            rgb_frames = []
            for frame in frames:
                rgb_frame = self._convert_frame_to_rgb(frame)
                rgb_frames.append(rgb_frame)
            
            # Realizar inferencia en batch
            results = self.model(
                rgb_frames,
                imgsz=self.inference_size,
                conf=self.confidence,
                verbose=False
            )
            
            inference_time = time.time() - start_time
            
            # Procesar resultados
            annotated_frames = []
            total_detections = 0
            
            for i, result in enumerate(results):
                # Anotar el frame RGB
                annotated_rgb = result.plot(
                    line_width=2,
                    font_size=0.8
                )
                
                # FIX: Convertir de vuelta al formato original
                annotated_frame = self._convert_frame_from_rgb(annotated_rgb)
                annotated_frames.append(annotated_frame)
                
                total_detections += len(result.boxes) if result.boxes is not None else 0
            
            # Actualizar métricas
            self.batch_count += 1
            self.total_inference_time += inference_time
            self.detections_count = total_detections
            self.avg_inference_fps = len(frames) / inference_time if inference_time > 0 else 0
            
            Gst.debug(f"Batch {self.batch_count}: {len(frames)} frames, "
                     f"{inference_time:.3f}s, {self.avg_inference_fps:.1f} inf_fps, "
                     f"{total_detections} detections")
                     
            return annotated_frames
            
        except Exception as e:
            Gst.error(f"Batch processing error: {e}")
            return frames

    def _add_fps_overlay(self, frame):
        """Agrega overlay con métricas de rendimiento"""
        if not self.show_fps:
            return frame
            
        # Calcular FPS de display
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
        
        # FIX: Trabajar con RGB para el overlay
        work_frame = self._convert_frame_to_rgb(frame)
        
        # Preparar texto de métricas
        fps_text = f"Display FPS: {self.fps:.1f}"
        inf_text = f"Inference FPS: {self.avg_inference_fps:.1f}"
        batch_text = f"Batch: {self.batch_size}"
        det_text = f"Detections: {self.detections_count}"
        
        # Configuración de texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (0, 255, 0)  # Verde en RGB
        bg_color = (0, 0, 0)  # Fondo negro
        
        # Posiciones para el texto
        texts = [fps_text, inf_text, batch_text, det_text]
        y_positions = [30, 60, 90, 120]
        
        # Dibujar cada línea de texto
        for i, (text, y_pos) in enumerate(zip(texts, y_positions)):
            # Obtener tamaño del texto
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Dibujar fondo
            cv2.rectangle(work_frame, 
                         (10, y_pos - text_height - 5), 
                         (10 + text_width + 10, y_pos + baseline + 5), 
                         bg_color, -1)
            
            # Dibujar texto
            cv2.putText(work_frame, text, (15, y_pos), font, font_scale, color, thickness)
        
        # FIX: Convertir de vuelta al formato original
        return self._convert_frame_from_rgb(work_frame)

    def do_transform_ip(self, buffer):
        """Procesa cada buffer de entrada"""
        if self.model is None:
            Gst.warning("YOLO model not loaded - processing frame without detection")
            return Gst.FlowReturn.OK
            
        # Mapear buffer para lectura/escritura
        success, map_info = buffer.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE)
        if not success:
            Gst.error("Failed to map buffer")
            return Gst.FlowReturn.ERROR
            
        try:
            # FIX: Calcular el tamaño esperado del buffer
            expected_size = self.width * self.height * self.bytes_per_pixel
            if len(map_info.data) < expected_size:
                Gst.error(f"Buffer too small: {len(map_info.data)} < {expected_size}")
                buffer.unmap(map_info)
                return Gst.FlowReturn.ERROR
            
            # Convertir buffer a numpy array
            frame_data = np.frombuffer(map_info.data[:expected_size], dtype=np.uint8)
            frame = frame_data.reshape((self.height, self.width, self.bytes_per_pixel)).copy()
            
            # Agregar frame al buffer
            self.frame_buffer.append(frame)
            self.frame_count += 1
            
            output_frame = frame  # Por defecto usar frame original
            
            # Procesar cuando tengamos suficientes frames para un batch
            if len(self.frame_buffer) >= self.batch_size:
                Gst.info(f"Processing batch of {self.batch_size} frames")
                
                # Extraer batch
                batch_frames = []
                for _ in range(self.batch_size):
                    batch_frames.append(self.frame_buffer.popleft())
                
                # Procesar batch síncronamente
                processed_frames = self._process_batch_sync(batch_frames)
                
                # Usar el primer frame procesado como output
                if processed_frames and len(processed_frames) > 0:
                    output_frame = processed_frames[0]
                    Gst.info(f"Using processed frame with {self.detections_count} detections")
                    
                    # Guardar el resto de frames procesados para frames futuros
                    for i in range(1, len(processed_frames)):
                        self.frame_buffer.appendleft(processed_frames[i])
                else:
                    Gst.warning("No processed frames returned from batch")
            
            # Si tenemos frames procesados pendientes, usar uno
            elif len(self.frame_buffer) > self.batch_size:
                output_frame = self.frame_buffer.popleft()
                Gst.debug("Using cached processed frame")
            
            # Agregar overlay de métricas
            output_frame = self._add_fps_overlay(output_frame)
            
            # FIX: Asegurar dimensiones y formato correctos
            if output_frame.shape != (self.height, self.width, self.bytes_per_pixel):
                if output_frame.shape[:2] != (self.height, self.width):
                    output_frame = cv2.resize(output_frame, (self.width, self.height))
                
                # Ajustar canales si es necesario
                if output_frame.shape[2] != self.bytes_per_pixel:
                    if self.bytes_per_pixel == 4 and output_frame.shape[2] == 3:
                        # Agregar canal alfa
                        alpha = np.ones((self.height, self.width, 1), dtype=output_frame.dtype) * 255
                        output_frame = np.concatenate([output_frame, alpha], axis=2)
                    elif self.bytes_per_pixel == 3 and output_frame.shape[2] == 4:
                        # Remover canal alfa
                        output_frame = output_frame[:, :, :3]
            
            # Escribir frame de vuelta al buffer
            result_bytes = output_frame.tobytes()
            if len(result_bytes) <= len(map_info.data):
                map_info.data[:len(result_bytes)] = result_bytes
            else:
                Gst.error(f"Result frame too large: {len(result_bytes)} > {len(map_info.data)}")
                buffer.unmap(map_info)
                return Gst.FlowReturn.ERROR
            
            buffer.unmap(map_info)
            return Gst.FlowReturn.OK
            
        except Exception as e:
            Gst.error(f"Transform error: {e}")
            buffer.unmap(map_info)
            return Gst.FlowReturn.ERROR

    def do_stop(self):
        """Limpieza al parar el elemento"""
        self.frame_buffer.clear()
        
        if self.batch_count > 0:
            total_frames = self.batch_count * self.batch_size
            avg_fps = total_frames / self.total_inference_time if self.total_inference_time > 0 else 0
            Gst.info(f"Processing complete. Batches: {self.batch_count}, "
                    f"Total frames: {total_frames}, Avg inference FPS: {avg_fps:.1f}")
        
        return True

# Registrar el tipo
GObject.type_register(YoloBatchDetector)

# Factory para el elemento
__gstelementfactory__ = ("yolo_batch_detector", Gst.Rank.NONE, YoloBatchDetector)
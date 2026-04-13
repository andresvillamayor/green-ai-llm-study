"""
Runner para clasificación de imágenes (Experimento B del protocolo).

Este módulo implementa la interfaz uniforme para modelos de visión,
permitiendo comparación justa de eficiencia energética entre arquitecturas.

Modelos soportados:
- vgg16: Línea base clásica (alto consumo, buena precisión)
- convnextv2_base: Arquitectura moderna de alto rendimiento
- mobilenetv4_conv_medium: Optimizada para eficiencia energética

Dataset: Imagenette 160px (subconjunto de ImageNet, reproducible)

Autor: Andrés Rubén Villamayor Ruiz Diaz
Fecha: 2026
"""

import time
import logging
import torch
from typing import Dict, Optional, List
from pathlib import Path

# Imports condicionales para evitar errores si no están instaladas las dependencias
try:
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    from PIL import Image
    import timm  # PyTorch Image Models para modelos adicionales
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

logger = logging.getLogger(__name__)


class VisionRunner:
    """
    Runner para tareas de clasificación de imágenes con métricas de energía.
    
    Implementa la misma interfaz que LLMRunner para mantener consistencia
    en el pipeline de benchmarking del protocolo Green AI.
    
    Atributos:
        model_name (str): Nombre del modelo en HuggingFace o timm
        model: Instancia del modelo cargado en memoria
        processor: Procesador de imágenes asociado al modelo
        device (str): 'cpu' o 'mps' para Apple Silicon
        batch_size (int): Tamaño de lote para inferencia (protocolo: 64)
        input_size (int): Resolución de entrada (protocolo: 160px)
    """
    
    # Configuración por defecto según protocolo
    DEFAULT_INPUT_SIZE = 160  # Imagenette 160px
    DEFAULT_BATCH_SIZE = 64
    
    def __init__(
        self,
        model_name: str,
        device: str = "mps",  # Apple Silicon M4
        batch_size: int = DEFAULT_BATCH_SIZE,
        input_size: int = DEFAULT_INPUT_SIZE,
        verbose: bool = False
    ):
        """
        Inicializa el runner de visión.
        
        Args:
            model_name: ID del modelo en HuggingFace o timm
                Ejemplos: 
                - "facebook/convnextv2-base-22k-224"
                - "timm/mobilenetv4_conv_medium.e250_r224_in1k"
                - "timm/vgg16.ra_in1k"
            device: 'cpu' o 'mps' para aceleración en Apple Silicon
            batch_size: Tamaño de lote para inferencia (por defecto: 64)
            input_size: Resolución de entrada en píxeles (por defecto: 160)
            verbose: Activar logs detallados de depuración
        """
        if not VISION_AVAILABLE:
            raise ImportError(
                "Dependencias de visión no instaladas. "
                "Ejecutar: pip install transformers timm pillow torch"
            )
        
        # Validar dispositivo para Apple Silicon
        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS no disponible en este sistema, usando CPU")
            device = "cpu"
        
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.input_size = input_size
        self.verbose = verbose
        
        logger.info("Cargando modelo de visión: %s", model_name)
        load_start = time.time()
        
        # Cargar modelo: priorizar transformers, fallback a timm
        try:
            # Intentar cargar desde HuggingFace transformers
            self.model = AutoModelForImageClassification.from_pretrained(
                model_name, 
                torch_dtype=torch.float16 if device == "mps" else torch.float32
            ).to(device)
            self.processor = AutoImageProcessor.from_pretrained(model_name)
        except Exception:
            # Fallback a timm para modelos no disponibles en transformers
            logger.info("Intentando cargar desde timm: %s", model_name)
            self.model = timm.create_model(model_name, pretrained=True, num_classes=1000).to(device)
            self.model.eval()
            # Procesador genérico para timm
            from torchvision import transforms
            self.processor = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            self._use_timm = True
        else:
            self._use_timm = False
        
        load_time = time.time() - load_start
        logger.info("Modelo cargado en %.2f segundos", load_time)
        
        # Contador de parámetros para reporte de complejidad
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info("Parámetros totales: %d (%.2f M)", total_params, total_params / 1e6)
    
    def generate(
        self,
        image_path: str,
        top_k: int = 5,
        max_tokens: int = 100,  # Placeholder por interfaz uniforme
        temperature: float = 0.1  # Placeholder por interfaz uniforme
    ) -> Dict:
        """
        Clasifica una imagen y retorna predicciones + métricas de rendimiento.
        
        Args:
            image_path: Ruta a la imagen de entrada (JPG, PNG)
            top_k: Cantidad de clases a retornar en predicciones
            max_tokens: No aplica para visión, mantenido por compatibilidad
            temperature: No aplica, mantenido por compatibilidad
        
        Returns:
            dict: Resultados con:
                - predictions (list): Lista de {label, score} para top_k clases
                - top_1_class (str): Clase con mayor probabilidad
                - top_1_score (float): Score de la clase ganadora
                - latency_ms (float): Latencia de inferencia en milisegundos
                - inference_time_s (float): Tiempo total en segundos
                - tokens (int): 1 (placeholder para compatibilidad con benchmark)
                - tokens_per_s (float): 1.0/inference_time (placeholder)
                - memory_mb (float): Memoria usada por el modelo en MB
                - input_shape (tuple): Dimensiones de entrada procesada
        """
        # Validar archivo de entrada
        if not Path(image_path).exists():
            raise FileNotFoundError("Imagen no encontrada: %s" % image_path)
        
        # Cargar y preprocesar imagen
        image = Image.open(image_path).convert("RGB")
        
        if self._use_timm:
            # Preprocesamiento para modelos timm
            inputs = self.processor(image).unsqueeze(0).to(self.device)
        else:
            # Preprocesamiento para modelos transformers
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Medir latencia de inferencia (sin incluir pre/post-procesamiento)
        torch.cuda.synchronize() if self.device == "cuda" else None  # Para timing preciso
        start_time = time.time()
        
        # Forward pass sin gradientes (modo inferencia)
        with torch.no_grad():
            if self._use_timm:
                outputs = self.model(inputs)
                # timm retorna logits directamente
                logits = outputs
            else:
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Softmax para convertir logits a probabilidades
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        
        # Sincronizar para timing preciso en MPS/CUDA
        torch.cuda.synchronize() if self.device == "cuda" else None
        latency_ms = (time.time() - start_time) * 1000
        
        # Post-procesar: obtener top_k predicciones
        top_results = torch.topk(probs, min(top_k, len(probs)))
        
        predictions = []
        for idx, prob in zip(top_results.indices, top_results.values):
            # Obtener label: transformers tiene id2label, timm no
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'id2label'):
                label = self.model.config.id2label[idx.item()]
            else:
                label = "class_%d" % idx.item()  # Fallback para timm
            
            predictions.append({
                "label": label,
                "score": round(prob.item(), 4),
                "class_id": idx.item()
            })
        
        # Calcular memoria usada (aproximada para MPS/CPU)
        memory_mb = self._estimate_memory_usage()
        
        # Preparar resultado con interfaz compatible con benchmark
        tokens_generated = 1  # Placeholder para compatibilidad
        inference_time_s = latency_ms / 1000.0
        
        return {
            "predictions": predictions,
            "top_1_class": predictions[0]["label"] if predictions else None,
            "top_1_score": predictions[0]["score"] if predictions else 0.0,
            "latency_ms": round(latency_ms, 2),
            "inference_time_s": round(inference_time_s, 3),
            "tokens": tokens_generated,
            "tokens_per_s": round(tokens_generated / inference_time_s, 2) if inference_time_s > 0 else 0.0,
            "memory_mb": round(memory_mb, 2),
            "input_shape": list(inputs.shape) if hasattr(inputs, 'shape') else None,
            "model_name": self.model_name,
            "device": self.device
        }
    
    def _estimate_memory_usage(self) -> float:
        """
        Estima el uso de memoria del modelo en MB.
        
        Returns:
            float: Memoria estimada en megabytes
        """
        # Para MPS/CPU: estimar por parámetros + buffers
        total_params = sum(p.numel() for p in self.model.parameters())
        total_buffers = sum(b.numel() for b in self.model.buffers())
        
        # Asumir float32 (4 bytes) o float16 (2 bytes) según dispositivo
        bytes_per_param = 2 if self.device == "mps" else 4
        memory_bytes = (total_params + total_buffers) * bytes_per_param
        
        # Agregar overhead estimado para activaciones (aprox. 2x parámetros)
        memory_bytes *= 3
        
        return memory_bytes / (1024 * 1024)  # Convertir a MB
    
    def get_embedding(self, image_path: str) -> List[float]:
        """
        Obtiene embedding de imagen para tareas de búsqueda/similitud.
        
        Útil para análisis de representaciones internas del modelo.
        
        Args:
            image_path: Ruta a la imagen de entrada
        
        Returns:
            list: Embedding normalizado como lista de floats
        """
        image = Image.open(image_path).convert("RGB")
        
        if self._use_timm:
            inputs = self.processor(image).unsqueeze(0).to(self.device)
        else:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            if hasattr(self.model, 'get_features'):
                # Algunos modelos tienen método explícito para features
                embedding = self.model.get_features(inputs)
            else:
                # Fallback: usar salida de la capa anterior al clasificador
                if self._use_timm:
                    embedding = self.model.forward_features(inputs)
                    # Global average pooling si es tensor 4D
                    if embedding.dim() == 4:
                        embedding = torch.nn.functional.adaptive_avg_pool2d(embedding, 1).flatten(1)
                else:
                    # Para transformers, obtener última hidden state
                    outputs = self.model(**inputs, output_hidden_states=True)
                    embedding = outputs.hidden_states[-1].mean(dim=1)  # Promedio temporal
        
        # Normalizar embedding (L2 norm)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding[0].cpu().numpy().tolist()
    
    def cleanup(self):
        """
        Libera recursos del modelo para evitar fugas de memoria.
        
        Importante cuando se ejecutan múltiples benchmarks en secuencia.
        """
        if hasattr(self, 'model'):
            # Mover a CPU y eliminar referencia para liberar memoria MPS/GPU
            if self.device == "mps" and torch.backends.mps.is_available():
                self.model = self.model.to("cpu")
            del self.model
        
        if hasattr(self, 'processor'):
            del self.processor
        
        # Forzar garbage collection para liberar memoria inmediatamente
        import gc
        gc.collect()
        
        if self.verbose:
            logger.debug("Recursos de VisionRunner liberados")
    
    def __enter__(self):
        """Soporte para contexto with (opcional)."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup automático al salir de contexto with."""
        self.cleanup()
        return False  # No suprimir excepciones
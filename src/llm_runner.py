"""
Módulo para inferencia de LLM usando llama-cpp-python con modelos cuantizados en formato GGUF.

Este módulo proporciona una interfaz de alto nivel para cargar y ejecutar inferencia
en modelos de lenguaje cuantizados usando el backend llama.cpp. Soporta aceleración
por hardware vía Metal en Apple Silicon e incluye monitoreo de rendimiento.

Clases:
    LLMRunner: Clase principal para inferencia con carga de modelo y generación de texto
"""

from llama_cpp import Llama
from pathlib import Path
from typing import Dict, Optional, List
import time
import logging

logger = logging.getLogger(__name__)


class LLMRunner:
    """
    Envoltorio de alto nivel para inferencia de LLM con backend llama.cpp.
    
    Esta clase maneja la carga del modelo, ejecución de inferencia y seguimiento
    de rendimiento para modelos de lenguaje cuantizados en formato GGUF. Soporta
    aceleración por GPU en Apple Silicon vía Metal Performance Shaders cuando está disponible.
    
    Atributos:
        model_path (Path): Ruta al archivo del modelo en formato GGUF
        llm (Llama): Instancia del modelo llama.cpp subyacente
        model_name (str): Nombre del archivo del modelo sin extensión
        n_ctx (int): Tamaño de la ventana de contexto en tokens
    """
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int = 8,
        n_gpu_layers: int = -1,
        n_batch: int = 512,
        verbose: bool = False
    ):
        """
        Inicializa el runner LLM y carga el modelo en memoria.
        
        Args:
            model_path (str): Ruta al archivo del modelo GGUF (relativa o absoluta)
            n_ctx (int, opcional): Tamaño de la ventana de contexto en tokens. Por defecto: 4096
            n_threads (int, opcional): Número de hilos de CPU a usar. Por defecto: 8
            n_gpu_layers (int, opcional): Número de capas a descargar a la GPU.
                -1 significa todas las capas (aceleración GPU completa). Por defecto: -1
            n_batch (int, opcional): Tamaño del lote para procesamiento del prompt. Por defecto: 512
            verbose (bool, opcional): Activar logging de debug de llama.cpp. Por defecto: False
        
        Raises:
            FileNotFoundError: Si model_path no existe
            
        Nota:
            El tiempo de carga del modelo varía según el tamaño y la velocidad del almacenamiento.
            Tiempos típicos: 2-10 segundos para modelos de 7B de parámetros.
            
            Descarga a GPU (n_gpu_layers):
            - -1: Descargar todas las capas (máximo uso de GPU)
            - 0: Solo CPU (sin aceleración por GPU)
            - N: Descargar N capas específicas (modo híbrido)
            
            Tamaño de lote (n_batch):
            - Valores altos (512-2048) mejoran la utilización de la GPU
            - Valores bajos (128-256) reducen el uso de memoria
        """
        self.model_path = Path(model_path)
        
        # Validar que el archivo del modelo existe antes de intentar cargarlo
        if not self.model_path.exists():
            raise FileNotFoundError(f"Archivo de modelo no encontrado: {model_path}")
        
        logger.info(f"Cargando modelo: {self.model_path.name}")
        load_start = time.time()
        
        # Inicializar modelo llama.cpp con los parámetros especificados
        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            verbose=verbose,
            use_mmap=True,    # Mapeo de memoria para carga más eficiente
            use_mlock=False   # No bloquear páginas en memoria RAM
        )
        
        load_time = time.time() - load_start
        logger.info(f"Modelo cargado exitosamente en {load_time:.2f} segundos")
        
        # Guardar metadatos del modelo para referencia posterior
        self.model_name = self.model_path.stem
        self.n_ctx = n_ctx
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None
    ) -> Dict:
        """
        Genera texto a partir de un prompt usando el modelo cargado.
        
        Utiliza muestreo por núcleo (top-p) y temperatura para controlar
        la aleatoriedad de la generación. Registra automáticamente el tiempo
        de inferencia y el throughput de tokens.
        
        Args:
            prompt (str): Texto de entrada para completar
            max_tokens (int, opcional): Máximo de tokens a generar. Por defecto: 100
            temperature (float, opcional): Temperatura de muestreo (0.0 a 1.0).
                Valores más altos aumentan la aleatoriedad. Por defecto: 0.7
            top_p (float, opcional): Umbral para muestreo por núcleo (0.0 a 1.0).
                Considera solo tokens con probabilidad acumulada <= top_p. Por defecto: 0.9
            stop (list[str], opcional): Secuencias de tokens que detienen la generación.
                Por defecto: ["</s>"] (token estándar de fin de secuencia)
        
        Returns:
            dict: Resultados de la generación con las siguientes claves:
                - text (str): Texto generado
                - tokens (int): Cantidad de tokens generados (respuesta)
                - time_s (float): Tiempo total de generación en segundos
                - tokens_per_s (float): Throughput en tokens por segundo
                - prompt_tokens (int): Tokens del prompt de entrada
                - total_tokens (int): Suma de prompt + respuesta
        
        Nota:
            Interpretación de la temperatura:
            - 0.0: Greedy (determinista, siempre elige la opción más probable)
            - 0.5: Muestreo enfocado (baja aleatoriedad)
            - 0.7: Balanceado (recomendado para la mayoría de tareas)
            - 1.0: Máxima diversidad (puede ser menos coherente)
            
            Top-p (muestreo por núcleo):
            - Considera solo tokens que componen la masa de probabilidad top_p
            - 0.9 es un buen valor por defecto para salidas coherentes pero diversas
        """
        logger.info(f"Generando respuesta (max_tokens={max_tokens})")
        
        start_time = time.time()
        
        # Llamar al motor de inferencia de llama.cpp
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or ["</s>"],
            echo=False  # No incluir el prompt en la salida
        )
        
        elapsed = time.time() - start_time
        
        # Extraer resultados de la generación desde el objeto de respuesta
        text = response['choices'][0]['text']
        tokens = response['usage']['completion_tokens']
        
        # Armar diccionario con métricas de rendimiento
        result = {
            "text": text,
            "tokens": tokens,
            "time_s": round(elapsed, 3),
            "tokens_per_s": round(tokens / elapsed, 2) if elapsed > 0 else 0.0,
            "prompt_tokens": response['usage']['prompt_tokens'],
            "total_tokens": response['usage']['total_tokens']
        }
        
        logger.info(
            f"Generados {tokens} tokens en {elapsed:.2f} segundos "
            f"({result['tokens_per_s']:.2f} tok/s)"
        )
        
        return result
    
    def get_model_info(self) -> Dict:
        """
        Obtiene metadatos del modelo cargado.
        
        Returns:
            dict: Información del modelo con las siguientes claves:
                - name (str): Nombre del archivo del modelo sin extensión
                - path (str): Ruta completa al archivo del modelo
                - size_mb (float): Tamaño del archivo en megabytes
                - context_size (int): Tamaño de la ventana de contexto en tokens
        
        Ejemplo:
            >>> info = runner.get_model_info()
            >>> print(f"Modelo: {info['name']} ({info['size_mb']:.1f} MB)")
            >>> print(f"Contexto: {info['context_size']} tokens")
        """
        return {
            "name": self.model_name,
            "path": str(self.model_path),
            "size_mb": round(self.model_path.stat().st_size / (1024 * 1024), 2),
            "context_size": self.n_ctx
        }
    
    def cleanup(self):
        """
        Libera recursos explícitamente (recomendado para benchmarks de energía).
        
        Nota:
            En Python, __del__ no garantiza liberación inmediata.
            Para mediciones precisas de consumo, llamá a cleanup()
            entre iteraciones del benchmark.
        """
        if hasattr(self, 'llm'):
            del self.llm
            logger.debug("Recursos del modelo liberados")
    
    def __del__(self):
        """
        Limpieza al destruir el objeto (fallback si no se llama cleanup()).
        
        Nota: No confiar en este método para benchmarks críticos de energía.
        """
        self.cleanup()
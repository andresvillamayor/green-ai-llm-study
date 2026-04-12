"""
Módulo para inferencia de LLM usando MLX en Apple Silicon.

Este módulo proporciona una interfaz para ejecutar inferencia usando el framework
MLX de Apple, optimizado para procesadores de la serie M con aceleración GPU vía Metal.

Clases:
    MLXRunner: Clase principal para inferencia, compatible con la interfaz de LLMRunner
"""

from pathlib import Path
from typing import Dict, Optional, List
import time
import logging
import os

logger = logging.getLogger(__name__)


class MLXRunner:
    """
    Envoltorio para inferencia de LLM usando MLX en Apple Silicon.
    
    Proporciona la misma interfaz que LLMRunner pero utiliza el backend MLX
    para aceleración optimizada con GPU Metal en chips de la serie M.
    
    Atributos:
        model_name (str): Identificador del modelo (ej: ruta o ID de HuggingFace)
        model: Instancia del modelo MLX cargado
        tokenizer: Tokenizador asociado al modelo
        simple_name (str): Nombre simplificado del modelo para logging
        context_size (int): Tamaño de la ventana de contexto en tokens
    """
    
    def __init__(
        self,
        model_name: str,
        verbose: bool = False,
        context_size: int = 4096,
        **kwargs
    ):
        """
        Inicializa el runner MLX y carga el modelo en memoria.
        
        Args:
            model_name (str): ID del modelo en HuggingFace o ruta local.
                Ejemplo: "mlx-community/Qwen2.5-7B-Instruct-4bit"
            verbose (bool, opcional): Activar logging detallado. Por defecto: False
            context_size (int, opcional): Tamaño de contexto en tokens. Por defecto: 4096
            **kwargs: Argumentos adicionales (se ignoran, para compatibilidad)
        
        Raises:
            ImportError: Si mlx_lm no está instalado
            Exception: Si falla la carga del modelo
        
        Nota:
            Los modelos se cachean en ~/.cache/huggingface/hub/
            La primera ejecución descargará el modelo si no está en caché.
            El tiempo de carga típico para modelos de 7B es de 3-8 segundos en M4.
        """
        self.model_name = model_name
        self.context_size = context_size
        
        logger.info("Cargando modelo MLX: %s", model_name)
        load_start = time.time()
        
        try:
            from mlx_lm import load
            self.model, self.tokenizer = load(model_name)
        except ImportError:
            logger.error("No se encontro mlx_lm. Instalar con: pip install mlx-lm")
            raise
        except Exception as e:
            logger.error("Error al cargar el modelo MLX: %s", e)
            raise
        
        load_time = time.time() - load_start
        logger.info("Modelo MLX cargado en %.2f segundos", load_time)
        
        # Guardar nombre simplificado para mostrar en resultados
        self.simple_name = model_name.split('/')[-1]
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None
    ) -> Dict:
        """
        Genera texto a partir de un prompt usando el modelo MLX.
        
        Utiliza muestreo por núcleo (top-p) y temperatura para controlar
        la aleatoriedad de la generación. Registra tiempo y throughput.
        
        Args:
            prompt (str): Texto de entrada para completar
            max_tokens (int, opcional): Máximo de tokens a generar. Por defecto: 100
            temperature (float, opcional): Temperatura de muestreo (0.0 a 1.0).
                Valores más altos = más creatividad/aleatoriedad. Por defecto: 0.7
            top_p (float, opcional): Umbral para muestreo por núcleo (0.0 a 1.0).
                Considera solo tokens con probabilidad acumulada <= top_p. Por defecto: 0.9
            stop (list[str], opcional): Secuencias que detienen la generación.
                Por defecto: None (no se usa en MLX actualmente)
        
        Returns:
            dict: Resultados de la generación con las siguientes claves:
                - text (str): Texto generado
                - tokens (int): Cantidad de tokens generados (respuesta)
                - time_s (float): Tiempo total de generación en segundos
                - tokens_per_s (float): Throughput en tokens por segundo
                - prompt_tokens (int): Tokens del prompt de entrada
                - total_tokens (int): Suma de prompt + respuesta
        
        Nota:
            Interpretación de temperatura:
            - 0.0: Greedy (determinista, siempre elige la opción más probable)
            - 0.5: Muestreo enfocado (baja aleatoriedad)
            - 0.7: Balanceado (recomendado para la mayoría de tareas)
            - 1.0: Máxima diversidad (puede ser menos coherente)
        """
        logger.info("Generando con MLX (max_tokens=%d)", max_tokens)
        
        start_time = time.time()
        
        try:
            from mlx_lm import generate
            
            # Generar con MLX
            # Nota: en mlx-lm 0.31.x el parámetro es 'temp', no 'temperature'
            # Se usa temp=temperature para mantener la interfaz consistente
            response_text = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=top_p,
                verbose=False
            )
            
        except TypeError as e:
            # Fallback por si cambia el nombre del parámetro en futuras versiones
            if "unexpected keyword argument" in str(e):
                logger.warning("Parametro de sampling no reconocido")
                response_text = generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    verbose=False
                )
            else:
                raise
        except Exception as e:
            logger.error("Error en generación MLX: %s", e)
            raise
        
        elapsed = time.time() - start_time
        
        # Contar tokens generados usando el tokenizador
        try:
            response_ids = self.tokenizer.encode(response_text)
            tokens_generated = len(response_ids) if hasattr(response_ids, '__len__') else 1
        except Exception:
            # Fallback: estimación conservadora
            tokens_generated = max(1, len(response_text.strip()) // 4)
            logger.warning("Conteo de tokens por fallback. Verificar tokenizador.")
        
        # Contar tokens del prompt
        try:
            prompt_ids = self.tokenizer.encode(prompt)
            prompt_tokens = len(prompt_ids) if hasattr(prompt_ids, '__len__') else len(prompt.split())
        except Exception:
            prompt_tokens = len(prompt.split())
        
        # Armar resultado con métricas de rendimiento
        result = {
            "text": response_text,
            "tokens": tokens_generated,
            "time_s": round(elapsed, 3),
            "tokens_per_s": round(tokens_generated / elapsed, 2) if elapsed > 0 else 0.0,
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens + tokens_generated
        }
        
        logger.info(
            "Generados %d tokens en %.2fs (%.2f tok/s)",
            tokens_generated, elapsed, result['tokens_per_s']
        )
        
        return result
    
    def get_model_info(self) -> Dict:
        """
        Obtiene metadatos del modelo cargado.
        
        Returns:
            dict: Información del modelo con las siguientes claves:
                - name (str): Nombre simplificado del modelo
                - path (str): Ruta estimada en caché de HuggingFace
                - framework (str): Identificador del framework ("MLX")
                - size_mb (float): Tamaño estimado en MB (0 si no disponible)
                - context_size (int): Tamaño de contexto configurado
        
        Ejemplo:
            runner = MLXRunner("mlx-community/Qwen2.5-7B-Instruct-4bit")
            info = runner.get_model_info()
            print("Modelo: %s | Contexto: %d tokens" % (info['name'], info['context_size']))
        """
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub/")
        
        return {
            "name": self.simple_name,
            "path": "%smodels--%s" % (cache_dir, self.model_name.replace('/', '--')),
            "framework": "MLX",
            "size_mb": 0.0,
            "context_size": self.context_size
        }
    
    def cleanup(self):
        """
        Libera recursos explícitamente (recomendado para benchmarks de energía).
        
        Nota:
            En Python, __del__ no garantiza liberación inmediata.
            Para mediciones precisas de consumo, llamá a cleanup()
            entre iteraciones del benchmark.
        """
        if hasattr(self, 'model'):
            del self.model
            logger.debug("Modelo MLX liberado")
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
            logger.debug("Tokenizador MLX liberado")
    
    def __del__(self):
        """
        Limpieza al destruir el objeto (fallback si no se llama cleanup()).
        
        Nota: No confiar en este método para benchmarks críticos de energía.
        """
        self.cleanup()
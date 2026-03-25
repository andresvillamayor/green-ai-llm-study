"""
Wrapper para llama-cpp-python optimizado para Apple M4
"""

from llama_cpp import Llama
from pathlib import Path
from typing import Dict, Optional, List
import time
import logging

logger = logging.getLogger(__name__)


class LLMRunner:
    """
    Runner para LLMs con llama.cpp y Metal (GPU M4)
    
    Ejemplo:
        >>> runner = LLMRunner("models/llama-2-7b.Q8_0.gguf")
        >>> result = runner.generate("¿Qué es IA?", max_tokens=100)
        >>> print(result['text'])
    """
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 8,
        n_gpu_layers: int = -1,
        verbose: bool = False
    ):
        """
        Inicializar runner
        
        Args:
            model_path: Ruta al modelo GGUF
            n_ctx: Tamaño del contexto (tokens)
            n_threads: Threads CPU
            n_gpu_layers: Capas en GPU (-1 = todas)
            verbose: Mostrar logs de llama.cpp
        """
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        
        logger.info(f"📦 Cargando modelo: {self.model_path.name}")
        load_start = time.time()
        
        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose
        )
        
        load_time = time.time() - load_start
        logger.info(f"✅ Modelo cargado en {load_time:.2f}s")
        
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
        Generar respuesta del LLM
        
        Args:
            prompt: Texto de entrada
            max_tokens: Máximo de tokens a generar
            temperature: Creatividad (0-1)
            top_p: Nucleus sampling
            stop: Tokens de parada
            
        Returns:
            Diccionario con text, tokens, time_s, tokens_per_s
        """
        logger.info(f"🔄 Generando respuesta (max_tokens={max_tokens})...")
        
        start_time = time.time()
        
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or ["</s>"],
            echo=False
        )
        
        elapsed = time.time() - start_time
        
        text = response['choices'][0]['text']
        tokens = response['usage']['completion_tokens']
        
        result = {
            "text": text,
            "tokens": tokens,
            "time_s": elapsed,
            "tokens_per_s": tokens / elapsed if elapsed > 0 else 0,
            "prompt_tokens": response['usage']['prompt_tokens'],
            "total_tokens": response['usage']['total_tokens']
        }
        
        logger.info(
            f"✅ {tokens} tokens en {elapsed:.2f}s "
            f"({result['tokens_per_s']:.2f} tok/s)"
        )
        
        return result
    
    def get_model_info(self) -> Dict:
        """Información del modelo"""
        return {
            "name": self.model_name,
            "path": str(self.model_path),
            "size_mb": self.model_path.stat().st_size / (1024 * 1024),
            "context_size": self.n_ctx
        }
    
    def __del__(self):
        """Limpiar al destruir"""
        if hasattr(self, 'llm'):
            del self.llm

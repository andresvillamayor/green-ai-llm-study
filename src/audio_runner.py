"""
Runner para transcripción de voz a texto.

Autor: Andrés Rubén Villamayor Ruiz Diaz
"""

import time
import logging
import torch
import gc
from datetime import datetime, timezone
from typing import Dict, Optional
from pathlib import Path

try:
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    import soundfile as sf
    import librosa
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

logger = logging.getLogger(__name__)


class AudioRunner:
    """Runner para ASR con métricas de energía."""
    
    DEFAULT_SAMPLE_RATE = 16000
    
    def __init__(
        self,
        model_name: str,
        device: str = "mps",
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        verbose: bool = False
    ):
        if not AUDIO_AVAILABLE:
            raise ImportError("Faltan dependencias de audio: pip install transformers torch soundfile librosa")
        
        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS no disponible, usando CPU")
            device = "cpu"
        
        self.model_name = model_name
        self.device = device
        self.sample_rate = sample_rate
        
        logger.info("Cargando modelo ASR: %s", model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, torch_dtype=torch.float16 if device == "mps" else torch.float32
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
    def generate(self, audio_path: str, max_tokens: int = 256) -> Dict:
        if not Path(audio_path).exists():
            raise FileNotFoundError("Audio no encontrado: %s" % audio_path)
        
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1: audio = audio.mean(axis=1) # Mono
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).to(self.device)
        
        # FIX MPS/FP16: Forzar que el input tenga el mismo tipo que el modelo
        inputs.input_features = inputs.input_features.to(self.model.dtype)
        
        # Medir latencia de inferencia (excluye pre/post-procesamiento)
        start_time = time.time()
       

        start = time.time()
        with torch.no_grad():
            gen_ids = self.model.generate(inputs.input_features, max_length=max_tokens)
        
        latency_ms = (time.time() - start) * 1000
        text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        
        return {
            "text": text.strip(),
            "tokens": len(text.split()),
            "latency_ms": round(latency_ms, 2),
            "inference_time_s": round(latency_ms / 1000.0, 3),
            "tokens_per_s": round(len(text.split()) / (latency_ms / 1000.0), 2) if latency_ms > 0 else 0.0,
            "model_name": self.model_name
        }
    
    def cleanup(self):
        if hasattr(self, 'model'): del self.model
        if hasattr(self, 'processor'): del self.processor
        gc.collect()

    def __enter__(self): return self
    def __exit__(self, *args): self.cleanup()
"""
GREEN-IA — Modulo de carga del modelo.

Responsabilidad unica: resolver la ruta del modelo GGUF segun config.yaml,
verificar que el archivo exista, y cargar la instancia de Llama. No conoce
nada de categorias, prompts, ni medicion de energia.
"""

import sys
import time
from pathlib import Path

import yaml
from llama_cpp import Llama


def resolver_ruta_modelo(root: Path, model_name: str, cuantizacion: str) -> Path:
    """Resuelve la ruta del archivo .gguf segun config.yaml.

    Args:
        root: directorio raiz del proyecto.
        model_name: nombre del modelo tal como aparece en config.yaml
            (ej. "qwen2.5-7b", "llama-2-7b").
        cuantizacion: "q4" o "q8".

    Returns:
        Path absoluto al archivo .gguf esperado.
    """
    with open(root / "config.yaml", encoding="utf-8") as f:
        config_raw = yaml.safe_load(f) or {}
    ruta_str = config_raw.get("models", {}).get(model_name, {}).get(
        cuantizacion, f"models/{model_name}/{model_name}.Q4_K_M.gguf")
    return root / ruta_str


def cargar_modelo(ruta_modelo: Path, cfg) -> Llama:
    """Carga el modelo GGUF en memoria. Aborta si el archivo no existe.

    El modelo se carga una unica vez por ejecucion del script; la misma
    instancia se reutiliza para todas las inferencias posteriores
    (warmup + mediciones oficiales), evitando el costo de recarga
    entre repeticiones.

    Args:
        ruta_modelo: Path al archivo .gguf ya verificado.
        cfg: ExperimentConfig con n_ctx, n_gpu_layers, n_threads, n_batch.

    Returns:
        Instancia de Llama lista para inferencia.
    """
    if not ruta_modelo.exists():
        print(f"\nERROR: modelo no encontrado: {ruta_modelo}")
        sys.exit(1)
    print(f"\n  Cargando {ruta_modelo.name}...")
    t_carga = time.perf_counter()
    llm = Llama(
        model_path=str(ruta_modelo), n_ctx=cfg.n_ctx,
        n_gpu_layers=cfg.n_gpu_layers(), n_threads=cfg.n_threads(),
        n_batch=cfg.n_batch(), verbose=False,
    )
    print(f"  Modelo cargado en {time.perf_counter() - t_carga:.1f}s")
    return llm

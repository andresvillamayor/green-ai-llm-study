#!/usr/bin/env python3
"""
Script unificado de ejecución para el protocolo Green AI.

Orquesta los tres experimentos (LLMs, Visión, Audio) según el protocolo
de la mesa de profesores, garantizando:
- 15 repeticiones por configuración para significancia estadística.
- Medición de energía y emisiones de CO2 (Paraguay/España).
- Métricas de rendimiento estilo MLPerf (latencia p50/p95, throughput).
- Limpieza de recursos y manejo robusto de errores.

Autor: Andrés Rubén Villamayor Ruiz Diaz
Fecha: 2026
"""

import sys
import os
import time
import csv
import logging
import statistics
import platform
from datetime import datetime, timezone  
from pathlib import Path
from typing import Dict, List, Optional

# Configurar paths relativos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports de módulos propios
from src.energy_tracker import EnergyTracker
from src.llm_runner import LLMRunner
from src.mlx_runner import MLXRunner
from src.vision_runner import VisionRunner
from src.audio_runner import AudioRunner

# Configurar logging
os.makedirs("results", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("results/unified_benchmark.log", encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuración del protocolo
REPETICIONES = 15
DEVICE = "mps" if platform.machine() == "arm64" else "cpu"
SAMPLE_IMAGE = "assets/sample_image.jpg"
SAMPLE_AUDIO = "assets/sample_audio.wav"
PROMPT_TEXT = "Que es la inteligencia artificial y como impacta en la sostenibilidad energetica?"

# Definición de Experimentos
EXPERIMENT_A_LLM = [
    {"name": "Qwen2.5-7B", "quant": "INT4", "runner": LLMRunner, "path": "models/qwen2.5-7b/Qwen2.5-7B-Instruct-Q4_K_M.gguf"},
    {"name": "Qwen2.5-7B", "quant": "INT8", "runner": LLMRunner, "path": "models/qwen2.5-7b/Qwen2.5-7B-Instruct-Q8_0.gguf"},
    {"name": "Qwen2.5-7B-MLX", "quant": "INT4", "runner": MLXRunner, "path": "mlx-community/Qwen2.5-7B-Instruct-4bit"},
]

EXPERIMENT_B_VISION = [
    {"name": "MobileNetV4", "quant": "FP16", "runner": VisionRunner, "path": "timm/mobilenetv4_conv_medium.e250_r224_in1k"},
    {"name": "ConvNeXt-V2", "quant": "FP16", "runner": VisionRunner, "path": "timm/convnextv2_base.fcmae_ft_in1k"},
]

EXPERIMENT_C_AUDIO = [
    {"name": "Whisper-Small", "quant": "FP16", "runner": AudioRunner, "path": "openai/whisper-small"},
    {"name": "Distil-Whisper", "quant": "FP16", "runner": AudioRunner, "path": "distil-whisper/distil-small.en"},
]


def get_timestamp_utc() -> str:
    """
    Obtiene timestamp actual en formato ISO 8601 (UTC).
    
    Returns:
        str: Timestamp en formato 'YYYY-MM-DDTHH:MM:SSZ'
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_single_inference(runner, exp_type: str, input_data: str, max_tokens: int = 100) -> Optional[Dict]:
    """
    Ejecuta una sola inferencia y retorna métricas crudas.
    
    Args:
        runner: Instancia del runner (LLM/Vision/Audio)
        exp_type: Tipo de experimento ("LLM", "Vision", "Audio")
        input_data: Prompt, ruta de imagen o ruta de audio
        max_tokens: Máximo de tokens para LLMs
    
    Returns:
        dict: Métricas de la inferencia o None si falla
    """
    tracker = EnergyTracker(project_name="unified_bench")
    tracker.start()

    try:
        if exp_type == "LLM":
            res = runner.generate(prompt=input_data, max_tokens=max_tokens)
            metric_val = res.get("tokens", 0)
        elif exp_type == "Vision":
            res = runner.generate(image_path=input_data)
            metric_val = 1  # 1 imagen procesada
        elif exp_type == "Audio":
            res = runner.generate(audio_path=input_data)
            metric_val = res.get("tokens", 0)
        else:
            raise ValueError("Tipo de experimento desconocido: %s" % exp_type)

        exec_time = res.get("inference_time_s", 0)
        latency_ms = res.get("latency_ms", exec_time * 1000)

    except FileNotFoundError as e:
        logger.error("Archivo no encontrado: %s", e)
        tracker.stop()
        return None
    except Exception as e:
        logger.error("Error en inferencia (%s): %s", exp_type, e)
        tracker.stop()
        return None

    # Detener tracker y calcular energía
    try:
        tracker.stop()
        stats = tracker.get_stats()
    except Exception as e:
        logger.warning("Error al obtener stats de energía: %s", e)
        stats = {"emissions_g_co2": 0.0, "energy_wh": 0.0}

    # Extrapolación a España: factor 245/25 = 9.8x (metodología documentada)
    co2_py_g = stats.get("emissions_g_co2", 0.0)
    co2_es_g = co2_py_g * (245.0 / 25.0)

    return {
        "inference_time_s": exec_time,
        "latency_ms": latency_ms,
        "metric_value": metric_val,
        "energy_wh": stats.get("energy_wh", 0.0),
        "co2_py_g": round(co2_py_g, 4),
        "co2_es_g": round(co2_es_g, 4),
        "timestamp": get_timestamp_utc()
    }


def run_experiment(experiment_name: str, configs: List[Dict], exp_type: str) -> List[Dict]:
    """
    Ejecuta un experimento completo con N repeticiones.
    
    Args:
        experiment_name: Nombre del experimento (A_LLM, B_Vision, C_Audio)
        configs: Lista de configuraciones de modelos a evaluar
        exp_type: Tipo de experimento ("LLM", "Vision", "Audio")
    
    Returns:
        list: Lista de resultados agregados por configuración
    """
    all_results = []

    for cfg in configs:
        logger.info("=" * 80)
        logger.info("Experimento %s: %s (%s)", experiment_name, cfg["name"], cfg["quant"])
        logger.info("=" * 80)

        runner = None
        try:
            # Inicializar runner según tipo
            runner_kwargs = {"device": DEVICE, "verbose": False}
            if exp_type == "LLM":
                runner_kwargs.update({"n_ctx": 2048, "n_gpu_layers": -1})
            
            logger.info("Inicializando runner: %s", cfg["runner"].__name__)
            runner = cfg["runner"](cfg["path"], **runner_kwargs)

            # Determinar input según modalidad
            if exp_type == "LLM":
                input_data = PROMPT_TEXT
            elif exp_type == "Vision":
                input_data = SAMPLE_IMAGE
            elif exp_type == "Audio":
                input_data = SAMPLE_AUDIO
            else:
                raise ValueError("Modalidad no soportada: %s" % exp_type)

            run_metrics = []
            for i in range(1, REPETICIONES + 1):
                logger.info("Repeticion %d/%d...", i, REPETICIONES)
                res = run_single_inference(runner, exp_type, input_data)
                if res:
                    run_metrics.append(res)
                else:
                    logger.warning("Repeticion %d fallida, continuando...", i)

            if not run_metrics:
                logger.warning("Sin resultados validos para %s, saltando...", cfg["name"])
                continue

            # Calcular estadísticas descriptivas
            times = [r["inference_time_s"] for r in run_metrics]
            latencies = [r["latency_ms"] for r in run_metrics]
            energies = [r["energy_wh"] for r in run_metrics]
            co2_py = [r["co2_py_g"] for r in run_metrics]
            co2_es = [r["co2_es_g"] for r in run_metrics]

            avg_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0.0
            p50_lat = statistics.median(latencies)
            p95_lat = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 20 else max(latencies)
            
            avg_energy = statistics.mean(energies)
            avg_co2_py = statistics.mean(co2_py)
            avg_co2_es = statistics.mean(co2_es)

            # Calcular throughput global (métricas totales / tiempo total)
            total_metric = sum(r["metric_value"] for r in run_metrics)
            total_time = sum(times)
            throughput = total_metric / total_time if total_time > 0 else 0.0

            result_row = {
                "Experiment": experiment_name,
                "Model": cfg["name"],
                "Quantization": cfg["quant"],
                "Framework": cfg["runner"].__name__,
                "Device": DEVICE,
                "Avg_Inference_Time_s": round(avg_time, 3),
                "Std_Inference_Time_s": round(std_time, 3),
                "Latency_P50_ms": round(p50_lat, 2),
                "Latency_P95_ms": round(p95_lat, 2),
                "Avg_Energy_Wh": round(avg_energy, 4),
                "Avg_CO2_PY_g": round(avg_co2_py, 4),
                "Avg_CO2_ES_g": round(avg_co2_es, 4),
                "Throughput": round(throughput, 2),
                "Repetitions": REPETICIONES,
                "Timestamp_First": run_metrics[0]["timestamp"],
                "Timestamp_Last": run_metrics[-1]["timestamp"]
            }
            all_results.append(result_row)
            logger.info("Resultados agregados para %s (%s)", cfg["name"], cfg["quant"])

        except ImportError as e:
            logger.error("Error de importación en %s: %s", cfg["name"], e)
            logger.info("Sugerencia: pip install transformers timm librosa soundfile")
        except Exception as e:
            logger.error("Fallo crítico en experimento %s - %s: %s", experiment_name, cfg["name"], e)
            import traceback
            logger.debug("Traceback: %s", traceback.format_exc())
        finally:
            if runner and hasattr(runner, "cleanup"):
                try:
                    runner.cleanup()
                    logger.debug("Cleanup completado para %s", cfg["name"])
                except Exception as e:
                    logger.warning("Error en cleanup de %s: %s", cfg["name"], e)

    return all_results


def save_results(results: List[Dict], filename: str = "results/final_comparison_table.csv") -> bool:
    """
    Guarda tabla comparativa unificada en CSV.
    
    Args:
        results: Lista de diccionarios con resultados
        filename: Ruta del archivo de salida
    
    Returns:
        bool: True si se guardó exitosamente, False si no hay datos
    """
    if not results:
        logger.warning("No hay resultados para guardar.")
        return False

    try:
        fieldnames = list(results[0].keys())
        with open(filename, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        logger.info("Tabla unificada guardada en: %s", filename)
        logger.info("Total configuraciones evaluadas: %d", len(results))
        return True
    except Exception as e:
        logger.error("Error al guardar resultados: %s", e)
        return False


def main():
    """Punto de entrada principal del protocolo unificado."""
    logger.info("INICIO PROTOCOLO GREEN AI UNIFICADO")
    logger.info("Plataforma: %s | Device: %s", platform.platform(), DEVICE)
    logger.info("Repeticiones por config: %d", REPETICIONES)
    logger.info("Timestamp inicio: %s", get_timestamp_utc())

    all_results = []

    # Ejecutar los 3 experimentos en secuencia
    logger.info("Ejecutando Experimento A: LLMs")
    all_results.extend(run_experiment("A_LLM", EXPERIMENT_A_LLM, "LLM"))
    
    logger.info("Ejecutando Experimento B: Vision")
    all_results.extend(run_experiment("B_Vision", EXPERIMENT_B_VISION, "Vision"))
    
    logger.info("Ejecutando Experimento C: Audio")
    all_results.extend(run_experiment("C_Audio", EXPERIMENT_C_AUDIO, "Audio"))

    # Guardar y mostrar resumen final
    save_results(all_results)

    logger.info("=" * 80)
    logger.info("PROTOCOLO FINALIZADO: %s", get_timestamp_utc())
    logger.info("Resultados: results/final_comparison_table.csv")
    logger.info("Log completo: results/unified_benchmark.log")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
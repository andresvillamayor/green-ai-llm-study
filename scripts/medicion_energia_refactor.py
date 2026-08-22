"""
GREEN-IA — Modulo de medicion de energia.

Responsabilidad unica: instrumentar una inferencia con CodeCarbon para
medir el consumo energetico (CPU, GPU, RAM) de esa operacion especifica.
Delega la generacion de texto a inferencia_refactor.generar_respuesta(),
sin conocer los parametros de generacion ni el contenido del prompt mas
alla de pasarlo hacia adelante.

Metodologia de medicion respaldada en:
  Lacoste, A., Luccioni, A., Schmidt, V., & Dandres, T. (2019).
  Quantifying the Carbon Emissions of Machine Learning.
  arXiv:1910.09700.
"""

import sys
import time

from inferencia_refactor import generar_respuesta


def medir_energia(llm, prompt: str, cfg) -> dict:
    """Ejecuta una inferencia y mide su consumo energetico con CodeCarbon.

    El tracker envuelve exactamente la operacion de inferencia (ni antes
    ni despues de la llamada al modelo), para que la medicion corresponda
    unicamente al costo energetico de esa generacion especifica.

    Tras la medicion, se limpia el KV-cache del modelo (llm.reset())
    para que cada repeticion sea independiente de la anterior. Sin este
    reset, se observo empiricamente que el coeficiente de variacion (CV)
    entre repeticiones subia de 0.31% a 33% — evidencia de que el estado
    interno del modelo contaminaba mediciones subsiguientes si no se
    limpiaba explicitamente.

    Ejecutado con privilegios de administrador (sudo), CodeCarbon utiliza
    powermetrics de macOS para obtener lecturas reales de potencia de
    CPU y GPU por separado, en lugar de estimaciones basadas en TDP.

    Args:
        llm: instancia de Llama ya cargada.
        prompt: texto ya formateado con la plantilla de chat del modelo.
        cfg: ExperimentConfig con los hiperparametros de generacion.

    Returns:
        dict con tokens generados, tiempo de inferencia, y energia total
        y desglosada por componente (CPU/GPU/RAM) en kWh y mWh.
    """
    try:
        from codecarbon import EmissionsTracker
    except ImportError:
        print("ERROR: codecarbon no instalado. pip install codecarbon")
        sys.exit(1)

    tracker = EmissionsTracker(
        project_name="green_ai_analisis_categoria",
        measure_power_secs=1,
        save_to_file=False,
        log_level="error",
        allow_multiple_runs=True,
    )
    tracker.start()
    t_inicio = time.perf_counter()
    resultado = generar_respuesta(llm, prompt, cfg)
    t_fin = time.perf_counter()
    tracker.stop()

    try:
        llm.reset()
    except Exception:
        pass

    def _kwh(nombre) -> float:
        obj = getattr(tracker, nombre, None)
        if obj is not None and hasattr(obj, "kWh"):
            try:
                return float(obj.kWh)
            except Exception:
                pass
        return 0.0

    kwh_total = _kwh("_total_energy")
    if kwh_total == 0:
        kwh_total = _kwh("_total_cpu_energy") + _kwh("_total_gpu_energy") + _kwh("_total_ram_energy")

    uso = resultado.get("usage", {})
    return {
        "tokens_generados"    : int(uso.get("completion_tokens", 0)),
        "tiempo_inferencia_s" : t_fin - t_inicio,
        "energia_kwh"         : kwh_total,
        "energia_mwh"         : kwh_total * 1_000_000.0,
        "cpu_energia_mwh"     : _kwh("_total_cpu_energy") * 1_000_000.0,
        "gpu_energia_mwh"     : _kwh("_total_gpu_energy") * 1_000_000.0,
        "ram_energia_mwh"     : _kwh("_total_ram_energy") * 1_000_000.0,
    }

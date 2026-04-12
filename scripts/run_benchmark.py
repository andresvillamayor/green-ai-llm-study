"""
Script principal para benchmarks comparativos de inferencia LLM con medición de energía.

Este módulo ejecuta pruebas de generación de texto utilizando diferentes backends
(llama.cpp y MLX) y configuraciones de cuantización. Incluye:
- Medición de consumo energético estimado (modo software, sin requerir sudo)
- Emisiones de CO2 comparativas por país (Paraguay vs España)
- Bucle de repeticiones para análisis estadístico (media ± desviación estándar)
- Tres tareas estandarizadas: texto, matemáticas, traducción

Los resultados se exportan a CSV para análisis estadístico posterior en el contexto
de la tesis sobre GREEN AI y consumo energético.

Autor: Andrés Rubén Villamayor Ruiz Diaz
Fecha: 2026
Repositorio: https://github.com/andresvillamayor/green-ai-llm-study
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="llm_runner")

import sys
import platform
import logging
from datetime import date, datetime
import os
import time
import csv
import statistics

# Configurar logging: archivo y consola
os.makedirs("results", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('results/environment_log.txt', encoding='utf-8', mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# FACTORES DE EMISIÓN POR PAÍS (actualizables desde ESIOS / Electricity Maps)
# =============================================================================
EMISSION_FACTORS = {
    # Valores promedio 2024-2025 (g CO2/kWh)
    "PY": {"name": "Paraguay", "factor": 70, "source": "IEA 2024 (hidroeléctrica)"},
    "ES": {"name": "España", "factor": 245, "source": "ESIOS/REE promedio 2024"},
    "FR": {"name": "Francia", "factor": 65, "source": "IEA 2024 (nuclear)"},
    "DE": {"name": "Alemania", "factor": 400, "source": "IEA 2024 (mix carbón)"},
    "US": {"name": "Estados Unidos", "factor": 415, "source": "IEA 2024"},
    "BR": {"name": "Brasil", "factor": 100, "source": "IEA 2024 (hidroeléctrica)"},
    "EU": {"name": "Unión Europea", "factor": 275, "source": "EU promedio 2024"}
}


def log_environment():
    """
    Registra la configuración técnica del entorno para reproducibilidad académica.
    """
    fecha = date.today()
    hora = datetime.now().time()
    
    logger.info("Fecha de ejecucion: %s", fecha)
    logger.info("Hora de ejecucion: %s", hora)
    logger.info("Python: %s", sys.version)
    logger.info("Plataforma: %s", platform.platform())
    logger.info("Arquitectura: %s", platform.machine())
    
    try:
        import mlx
        logger.info("MLX version: %s", mlx.__version__)
    except (ImportError, AttributeError):
        logger.info("MLX: no disponible")
    
    try:
        import llama_cpp
        logger.info("llama-cpp-python version: %s", llama_cpp.__version__)
    except (ImportError, AttributeError):
        logger.info("llama-cpp-python: no disponible")


# =============================================================================
# PROMPTS ESTANDARIZADOS PARA LAS 3 TAREAS
# =============================================================================
TASKS = {
    "texto": {
        "prompt": "Que es la inteligencia artificial y como impacta en la sostenibilidad energetica?",
        "description": "Generación de texto académico sobre IA y energía"
    },
    "matematicas": {
        "prompt": "Resuelve paso a paso: Si un modelo consume 40W durante inferencia y genera 20 tokens por segundo, cuanta energia en Joules se requiere para generar un documento de 100 tokens? Muestra el calculo completo.",
        "description": "Problema matemático de cálculo energético"
    },
    "traduccion": {
        "prompt": "Traduce al ingles: La cuantizacion de modelos de lenguaje permite ejecutar inferencia eficiente en dispositivos con recursos limitados, reduciendo el consumo energetico sin comprometer significativamente la calidad de salida.",
        "description": "Traducción técnico-científica español -> inglés"
    }
}


def ejecutar_benchmark(runner, prompt, max_tokens=100, temperature=0.1, 
                      project_name="benchmark", country_code="PY", do_cleanup=False):
    """
    Ejecuta una prueba de inferencia con estimación de consumo energético.
    
    Args:
        runner: Objeto con método generate() (LLMRunner o MLXRunner)
        prompt (str): Texto de entrada para la generación
        max_tokens (int): Máximo de tokens a generar
        temperature (float): Temperatura de muestreo (baja para reproducibilidad)
        project_name (str): Nombre para identificar el experimento en los logs
        country_code (str): Código de país para factor de emisión. Por defecto: "PY"
        do_cleanup (bool): Si True, llama a runner.cleanup() al finalizar.
            Por defecto: False (para permitir reutilización del runner)
    
    Returns:
        dict: Resultados con métricas de rendimiento y energía
    """
    from src.energy_tracker import EnergyTracker
    
    start = time.time()
    
    # Crear tracker con nombre único
    tracker_name = "%s_%s" % (project_name, runner.__class__.__name__)
    tracker = EnergyTracker(
        project_name=tracker_name, 
        country=EMISSION_FACTORS.get(country_code, EMISSION_FACTORS["PY"])["name"]
    )
    
    try:
        # Iniciar medición (modo software, sin threads)
        tracker.start()
        
        # Ejecutar inferencia
        result = runner.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Detener medición y obtener stats
        tracker.stop()
        energy_stats = tracker.get_stats()
        
        elapsed = time.time() - start
        
        # Calcular métricas para cada país de comparación
        comparisons = {}
        for code, data in EMISSION_FACTORS.items():
            factor_kg = data["factor"] / 1000  # Convertir g/kWh a kg/kWh
            energy_kwh = energy_stats["energy_kwh"]
            co2_kg = energy_kwh * factor_kg
            comparisons[code] = {
                "co2_g": round(co2_kg * 1000, 4),
                "factor_g_per_kwh": data["factor"]
            }
        
        return {
            "success": True,
            "tokens": result.get("tokens", 0),
            "time_s": round(elapsed, 3),
            "tokens_per_s": round(result.get("tokens", 0) / elapsed, 2) if elapsed > 0 else 0.0,
            "prompt_tokens": result.get("prompt_tokens", 0),
            "energy_kwh": round(energy_stats["energy_kwh"], 6),
            "joules_per_token": round((tracker._power_estimate_watts * elapsed) / result.get("tokens", 1), 3),
            "co2_py_g": comparisons["PY"]["co2_g"],
            "co2_es_g": comparisons["ES"]["co2_g"],
            "comparison": comparisons,
            "country_code": country_code,
            "error": None
        }
        
    except Exception as e:
        logger.error("Error en inferencia: %s", e)
        return {
            "success": False,
            "tokens": 0, "time_s": 0, "tokens_per_s": 0, "prompt_tokens": 0,
            "energy_kwh": 0, "joules_per_token": 0,
            "co2_py_g": 0, "co2_es_g": 0, "comparison": {},
            "country_code": country_code,
            "error": str(e)
        }
    finally:
        # Solo limpiar si se solicita explícitamente (al final de todas las repeticiones)
        if do_cleanup and hasattr(runner, 'cleanup'):
            runner.cleanup()


def guardar_resultados_csv(resultados, ruta_archivo):
    """
    Guarda resultados en formato CSV para análisis estadístico posterior.
    """
    if not resultados:
        logger.warning("No hay resultados para guardar")
        return
    
    try:
        with open(ruta_archivo, 'w', newline='', encoding='utf-8-sig') as archivo:
            writer = csv.DictWriter(archivo, fieldnames=resultados[0].keys())
            writer.writeheader()
            writer.writerows(resultados)
        logger.info("Resultados guardados en: %s", ruta_archivo)
    except Exception as e:
        logger.error("Error al guardar CSV: %s", e)


def calcular_estadisticas(resultados_por_config):
    """
    Calcula media y desviación estándar para cada configuración.
    
    Args:
        resultados_por_config (dict): {config_key: [lista_de_resultados]}
    
    Returns:
        dict: Estadísticas resumidas por configuración
    """
    resumen = {}
    
    for config, resultados in resultados_por_config.items():
        exitos = [r for r in resultados if r["success"]]
        if not exitos:
            continue
        
        # Calcular estadísticas para cada métrica numérica
        metrics = ["tokens_per_s", "joules_per_token", "co2_py_g", "co2_es_g"]
        stats = {}
        
        for m in metrics:
            valores = [r[m] for r in exitos if r[m] > 0]
            if len(valores) >= 2:
                stats[m] = {
                    "mean": round(statistics.mean(valores), 2),
                    "std": round(statistics.stdev(valores), 2),
                    "min": round(min(valores), 2),
                    "max": round(max(valores), 2),
                    "n": len(valores)
                }
            elif len(valores) == 1:
                stats[m] = {
                    "mean": round(valores[0], 2),
                    "std": 0.0,
                    "min": round(valores[0], 2),
                    "max": round(valores[0], 2),
                    "n": 1
                }
        
        resumen[config] = {
            "framework": exitos[0]["framework"],
            "quant": exitos[0]["quant"],
            "task": exitos[0].get("task", "general"),
            "stats": stats,
            "success_rate": len(exitos) / len(resultados)
        }
    
    return resumen


def mostrar_resumen_estadistico(resumen, countries=("PY", "ES")):
    """
    Muestra resumen comparativo con estadísticas en consola.
    """
    if not resumen:
        return
    
    logger.info("")
    logger.info("Resumen estadístico (media ± desviación estándar)")
    logger.info("=" * 95)
    
    # Encabezado dinámico según países seleccionados
    co2_cols = " ".join([f"CO2({c})g" for c in countries])
    formato = "%-18s %-12s %-10s %10s %12s %15s"
    logger.info(formato % ("Config", "Framework", "Tarea", "tok/s", "J/token", co2_cols))
    logger.info("-" * 95)
    
    for config, data in resumen.items():
        stats = data["stats"]
        if not stats:
            continue
        
        # Formatear cada métrica como "media ± std"
        tok_s = "%s ± %s" % (stats["tokens_per_s"]["mean"], stats["tokens_per_s"]["std"]) if "tokens_per_s" in stats else "N/A"
        j_tok = "%s ± %s" % (stats["joules_per_token"]["mean"], stats["joules_per_token"]["std"]) if "joules_per_token" in stats else "N/A"
        
        co2_vals = []
        for c in countries:
            key = "co2_%s_g" % c.lower()
            if key in stats:
                co2_vals.append("%s ± %s" % (stats[key]["mean"], stats[key]["std"]))
            else:
                co2_vals.append("N/A")
        co2_str = " | ".join(co2_vals)
        
        logger.info(
            formato % (
                config,
                data["framework"],
                data["task"],
                tok_s,
                j_tok,
                co2_str
            )
        )
    
    logger.info("=" * 95)


# Importar módulos desde src/
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from llm_runner import LLMRunner
from mlx_runner import MLXRunner


if __name__ == "__main__":
    
    # ========================================================================
    # CONFIGURACIÓN DEL EXPERIMENTO
    # ========================================================================
    log_environment()
    logger.info("Iniciando benchmark GREEN AI con repeticiones y comparación internacional")
    
    # Parámetros fijos
    MAX_TOKENS = 100
    TEMPERATURE = 0.1
    N_CTX = 2048
    REPETICIONES = 5  # Opción A: bucle de repeticiones
    PAISES_COMPARACION = ["PY", "ES"]  # Paraguay vs España
    
    # Modelos a probar
    MODELS = {
        "llama_cpp_q4": {
            "runner": "LLMRunner",
            "path": "models/qwen2.5-7b/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            "quant": "INT4_Q4_K_M",
            "framework": "llama-cpp-python"
        },
        "llama_cpp_q8": {
            "runner": "LLMRunner", 
            "path": "models/qwen2.5-7b/Qwen2.5-7B-Instruct-Q8_0.gguf",
            "quant": "INT8_Q8_0",
            "framework": "llama-cpp-python"
        },
        "mlx_int4": {
            "runner": "MLXRunner",
            "path": "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "quant": "INT4",
            "framework": "MLX"
        }
    }
    
    # ========================================================================
    # EJECUCIÓN: Bucle de repeticiones × tareas × modelos
    # ========================================================================
    resultados_todos = []
    resultados_por_config = {}  # Para cálculo estadístico
    
    for task_key, task_data in TASKS.items():  # Opción D: 3 tareas
        logger.info("")
        logger.info(">>> Tarea: %s - %s", task_key, task_data["description"])
        
        for model_key, model_cfg in MODELS.items():
            config_key = "%s__%s__%s" % (model_key, task_key, "PY")
            resultados_config = []
            
            logger.info("  Probando: %s | %s | %s", model_key, model_cfg["framework"], model_cfg["quant"])
            
            for rep in range(REPETICIONES):  # Opción A: 5 repeticiones
                logger.info("    Repetición %d/%d", rep+1, REPETICIONES)
                
                try:
                    # Crear runner
                    if model_cfg["runner"] == "LLMRunner":
                        if not os.path.exists(model_cfg["path"]):
                            logger.warning("      Modelo no encontrado: %s", model_cfg["path"])
                            break
                        runner = LLMRunner(model_cfg["path"], n_ctx=N_CTX, n_gpu_layers=-1, verbose=False)
                    else:
                        runner = MLXRunner(model_cfg["path"], verbose=False)
                    
                    # Ejecutar benchmark para cada país de comparación
                    for i, country in enumerate(PAISES_COMPARACION):
                        # do_cleanup=True SOLO para el último país de la iteración
                        last_country = (i == len(PAISES_COMPARACION) - 1)
    
                        result = ejecutar_benchmark(
                            runner=runner,
                            prompt=task_data["prompt"],
                            max_tokens=MAX_TOKENS,
                            temperature=TEMPERATURE,
                            project_name="tesis_green_ai",
                            country_code=country,
                            do_cleanup=last_country  #  limpiar al final
                        )
                        
                        # Agregar metadatos
                        result.update({
                            "model_key": model_key,
                            "framework": model_cfg["framework"],
                            "quant": model_cfg["quant"],
                            "task": task_key,
                            "country": country,
                            "repeticion": rep+1,
                            "model_path": model_cfg["path"]
                        })
                        resultados_todos.append(result)
                        resultados_config.append(result)
                    
                    # Pequeña pausa para enfriamiento entre repeticiones
                    time.sleep(2)
                    
                except Exception as e:
                    logger.error("      Error en %s rep %d: %s", model_key, rep+1, e)
            
            # Guardar resultados por configuración para estadísticas
            if resultados_config:
                resultados_por_config[config_key] = resultados_config
            
            # Log inmediato del promedio de esta configuración
            exitos = [r for r in resultados_config if r["success"]]
            if exitos:
                avg_tok_s = statistics.mean([r["tokens_per_s"] for r in exitos])
                logger.info("    OK: %s - tok/s promedio: %.2f", model_key, avg_tok_s)
    
    # ========================================================================
    # ANÁLISIS ESTADÍSTICO Y EXPORTACIÓN
    # ========================================================================
    
    # Calcular estadísticas (media ± std)
    resumen_stats = calcular_estadisticas(resultados_por_config)
    
    # Guardar todos los resultados crudos en CSV
    csv_crudo = "results/benchmark_comparativa_crudo.csv"
    guardar_resultados_csv(resultados_todos, csv_crudo)
    
    # Guardar resumen estadístico en CSV separado
    csv_resumen = "results/benchmark_resumen_estadistico.csv"
    if resumen_stats:
        with open(csv_resumen, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(["config", "framework", "quant", "task", 
                           "tok_s_mean", "tok_s_std", "j_token_mean", "j_token_std",
                           "co2_py_mean", "co2_py_std", "co2_es_mean", "co2_es_std", "success_rate"])
            for config, data in resumen_stats.items():
                s = data["stats"]
                writer.writerow([
                    config, data["framework"], data["quant"], data["task"],
                    s.get("tokens_per_s", {}).get("mean", ""), s.get("tokens_per_s", {}).get("std", ""),
                    s.get("joules_per_token", {}).get("mean", ""), s.get("joules_per_token", {}).get("std", ""),
                    s.get("co2_py_g", {}).get("mean", ""), s.get("co2_py_g", {}).get("std", ""),
                    s.get("co2_es_g", {}).get("mean", ""), s.get("co2_es_g", {}).get("std", ""),
                    data["success_rate"]
                ])
        logger.info("Resumen estadístico guardado en: %s", csv_resumen)
    
    # Mostrar resumen en consola
    mostrar_resumen_estadistico(resumen_stats, countries=PAISES_COMPARACION)
    
    # Comparativa España vs Paraguay
    logger.info("")
    logger.info("Comparativa de emisiones: Paraguay vs España")
    logger.info("-" * 60)
    for config, data in resumen_stats.items():
        s = data["stats"]
        if "co2_py_g" in s and "co2_es_g" in s:
            py_mean = s["co2_py_g"]["mean"]
            es_mean = s["co2_es_g"]["mean"]
            ratio = es_mean / py_mean if py_mean > 0 else 0
            logger.info("%s: PY=%.4fg | ES=%.4fg | ES es %.1fx más contaminante", 
                       config, py_mean, es_mean, ratio)
    
    logger.info("")
    logger.info("Proceso de benchmark finalizado")
    logger.info("Archivos generados:")
    logger.info("  - %s (resultados crudos)", csv_crudo)
    logger.info("  - %s (resumen estadístico)", csv_resumen)
    logger.info("  - results/environment_log.txt (logs técnicos)")
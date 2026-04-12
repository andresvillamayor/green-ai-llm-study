#!/usr/bin/env python3
"""
Benchmark comprehensivo con métricas de hardware, energía y emisiones de CO₂.

Este módulo ejecuta pruebas de inferencia de modelos LLM capturando:
- Métricas de hardware (CPU, memoria, frecuencia)
- Consumo energético estimado para Apple Silicon
- Emisiones de CO₂ para múltiples países (Paraguay, España, etc.)
- Comparativa de modelos: Qwen/Llama con cuantización INT4 vs INT8
- Estadísticas de rendimiento (tiempo, throughput, carga)

Los resultados se exportan en formato CSV y JSON para análisis estadístico
y visualización en la tesis sobre GREEN AI y eficiencia energética.

Autor: Andrés Rubén Villamayor Ruiz Diaz
Fecha: 2026
Repositorio: https://github.com/andresvillamayor/green-ai-llm-study
"""

import sys
import os
import time
import csv
import json
import logging
import platform
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Configurar logging: archivo y consola simultáneamente
os.makedirs("results", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('results/comprehensive_benchmark.log', encoding='utf-8', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# FACTORES DE EMISIÓN DE CO₂ POR PAÍS
# Fuentes: ESIOS/REE (España), Electricity Maps (Paraguay), IEA (otros)
# ============================================================================
EMISSION_FACTORS = {
    "PY": {"name": "Paraguay", "factor_g_kwh": 70, "source": "Electricity Maps"},
    "ES": {"name": "España", "factor_g_kwh": 245, "source": "ESIOS/REE"},
    "FR": {"name": "Francia", "factor_g_kwh": 65, "source": "IEA"},
    "DE": {"name": "Alemania", "factor_g_kwh": 400, "source": "IEA"},
    "US": {"name": "Estados Unidos", "factor_g_kwh": 415, "source": "IEA"},
    "BR": {"name": "Brasil", "factor_g_kwh": 100, "source": "IEA"},
    "AR": {"name": "Argentina", "factor_g_kwh": 350, "source": "IEA"},
}

class HardwareMetricsCollector:
    """
    Colector de métricas de hardware usando powermetrics en macOS.
    
    Requiere permisos de sudo. Si no están disponibles, cae a estimación.
    """
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.metrics_history = []
        self.powermetrics_collector = None
        self.use_powermetrics = False
    
    def start_collection(self):
        """Inicia la recolección de métricas."""
        self.start_time = time.time()
        self.metrics_history = []
        
        # Intentar usar powermetrics con sudo
        try:
            from src.powermetrics_collector import PowerMetricsCollector
            self.powermetrics_collector = PowerMetricsCollector(sampling_interval_ms=100)
            self.powermetrics_collector.start_collection()
            self.use_powermetrics = True
            logger.info("Recolección iniciada con powermetrics (sudo)")
        except Exception as e:
            logger.warning("No se pudo iniciar powermetrics: %s", e)
            logger.info("Usando estimación con psutil")
            self.use_powermetrics = False
        
        logger.info("Iniciando recolección de métricas de hardware...")
    
    def collect_sample(self) -> Dict:
        """Captura una muestra de métricas actuales."""
        sample = {
            "timestamp": time.time() - self.start_time if self.start_time else 0,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            "memory_used_mb": self.process.memory_info().rss / 1024 / 1024,
            "memory_percent": self.process.memory_percent(),
        }
        
        # Si estamos usando powermetrics, obtener métricas actuales
        if self.use_powermetrics and self.powermetrics_collector:
            current = self.powermetrics_collector.get_current_metrics()
            if current:
                sample.update(current)
        
        self.metrics_history.append(sample)
        return sample
    
    def stop_collection(self) -> Dict:
        """Detiene la recolección y calcula promedios."""
        avg_metrics = {}
        
        # Si usamos powermetrics, obtener promedios reales
        if self.use_powermetrics and self.powermetrics_collector:
            try:
                pm_metrics = self.powermetrics_collector.stop_collection()
                avg_metrics.update(pm_metrics)
                logger.info("PowerMetrics: CPU=%.3fW, GPU=%.3fW, Total=%.3fW",
                           pm_metrics.get('cpu_power_avg_w', 0),
                           pm_metrics.get('gpu_power_avg_w', 0),
                           pm_metrics.get('total_power_avg_w', 0))
            except Exception as e:
                logger.warning("Error al obtener métricas de powermetrics: %s", e)
        
        # Calcular promedios de psutil como fallback
        if self.metrics_history:
            if 'cpu_percent_avg' not in avg_metrics:
                avg_metrics["cpu_percent_avg"] = sum(m["cpu_percent"] for m in self.metrics_history) / len(self.metrics_history)
            if 'cpu_freq_avg_ghz' not in avg_metrics:
                avg_metrics["cpu_freq_avg_ghz"] = (sum(m["cpu_freq_mhz"] for m in self.metrics_history) / len(self.metrics_history)) / 1000
            if 'memory_used_avg_mb' not in avg_metrics:
                avg_metrics["memory_used_avg_mb"] = sum(m["memory_used_mb"] for m in self.metrics_history) / len(self.metrics_history)
        
        return avg_metrics



class EnergyEstimator:
    """
    Estimador de consumo energético para Apple Silicon (M4).
    
    Calcula potencia estimada basada en uso de CPU/GPU y energía consumida
    durante la inferencia. Usa valores de TDP típicos del chip M4 en carga
    de trabajo de inferencia LLM.
    
    Atributos de clase:
        POWER_ESTIMATES (dict): Potencias estimadas en Watts para diferentes
            componentes del sistema (CPU idle/active, GPU, ANE, overhead)
    """
    
    # Potencias estimadas para Mac Mini M4 en inferencia LLM
    # Basado en: TDP del M4 + mediciones públicas de carga sostenida
    POWER_ESTIMATES = {
        "cpu_idle_w": 5,
        "cpu_active_w": 15,
        "gpu_active_w": 20,
        "ane_active_w": 5,
        "system_overhead_w": 10,
    }
    
    @staticmethod
    def estimate_power(cpu_percent: float, gpu_active: bool = True) -> float:
        """
        Estima potencia instantánea basada en uso de CPU/GPU.
        
        Args:
            cpu_percent: Porcentaje de uso de CPU (0-100)
            gpu_active: Indica si la GPU está activa (True por defecto)
        
        Returns:
            float: Potencia estimada en Watts
        """
        # Calcular potencia de CPU (idle si uso < 10%, activa proporcional si no)
        cpu_power = EnergyEstimator.POWER_ESTIMATES["cpu_idle_w"]
        if cpu_percent > 10:
            cpu_power = EnergyEstimator.POWER_ESTIMATES["cpu_active_w"] * (cpu_percent / 100)
        
        # Agregar potencia de GPU y ANE si están activas
        gpu_power = EnergyEstimator.POWER_ESTIMATES["gpu_active_w"] if gpu_active else 0
        ane_power = EnergyEstimator.POWER_ESTIMATES["ane_active_w"] if gpu_active else 0
        
        # Sumar overhead del sistema (ventiladores, RAM, etc.)
        total = cpu_power + gpu_power + ane_power + EnergyEstimator.POWER_ESTIMATES["system_overhead_w"]
        return round(total, 2)
    
    @staticmethod
    def calculate_energy(power_w: float, duration_s: float) -> Dict:
        """
        Calcula energía consumida y emisiones de CO₂ para múltiples países.
        
        Args:
            power_w: Potencia promedio en Watts
            duration_s: Duración de la inferencia en segundos
        
        Returns:
            dict: Energía en Wh/kWh y emisiones de CO₂ por país
        """
        # Convertir a Watt-hora y kiloWatt-hora
        energy_wh = (power_w * duration_s) / 3600
        energy_kwh = energy_wh / 1000
        
        # Calcular emisiones para cada país usando su factor de emisión
        emissions = {}
        for country_code, data in EMISSION_FACTORS.items():
            factor_g_kwh = data["factor_g_kwh"]
            co2_g = energy_kwh * factor_g_kwh
            emissions[country_code] = {
                "name": data["name"],
                "factor_g_kwh": factor_g_kwh,
                "co2_g": round(co2_g, 4),
                "source": data["source"]
            }
        
        return {
            "energy_wh": round(energy_wh, 4),
            "energy_kwh": round(energy_kwh, 6),
            "power_avg_w": power_w,
            "duration_s": round(duration_s, 2),
            "emissions": emissions
        }


def run_inference_benchmark(runner_class, model_path: str, prompt: str, 
                           max_tokens: int = 100, temperature: float = 0.1,
                           model_name: str = "Qwen2.5-7B", quantization: str = "INT4") -> Dict:
    """
    Ejecuta benchmark completo de inferencia con métricas de hardware.
    
    Esta función orquesta todo el proceso: carga del modelo, recolección de
    métricas de hardware, ejecución de inferencia, cálculo de energía y
    emisiones. Retorna un diccionario completo con todos los resultados.
    
    Args:
        runner_class: Clase del runner (LLMRunner o MLXRunner)
        model_path (str): Ruta al archivo del modelo o ID de HuggingFace
        prompt (str): Texto de entrada para la generación
        max_tokens (int): Máximo de tokens a generar. Por defecto: 100
        temperature (float): Temperatura de sampling (0.0-1.0). Por defecto: 0.1
        model_name (str): Nombre del modelo para logs. Por defecto: "Qwen2.5-7B"
        quantization (str): Tipo de cuantización. Por defecto: "INT4"
    
    Returns:
        dict: Resultados completos con las siguientes claves:
            - success (bool): Indica si el benchmark fue exitoso
            - timestamp (str): Fecha y hora de ejecución (ISO format)
            - model (dict): Información del modelo (nombre, cuantización, etc.)
            - inference (dict): Métricas de inferencia (tokens, tiempo, throughput)
            - hardware (dict): Métricas de hardware (CPU, memoria, temperatura)
            - energy (dict): Energía consumida y emisiones de CO₂ por país
            - load_time_s (float): Tiempo de carga del modelo en segundos
            - error (str, opcional): Mensaje de error si success=False
    """
    logger.info("Iniciando benchmark: %s (%s)", model_name, quantization)
    
    # Inicializar componentes del benchmark
    metrics_collector = HardwareMetricsCollector()
    energy_estimator = EnergyEstimator()
    
    # Inicializar variables que se usarán en finally (evita warning de Pylance)
    # Esto es importante para que el código sea robusto ante excepciones
    inference_time = 0.0
    tokens_generated = 0
    result = {}
    load_time = 0.0
    runner = None
    
    # Cargar modelo en memoria
    logger.info("Cargando modelo: %s", model_path)
    load_start = time.time()
    
    try:
        if runner_class.__name__ == "LLMRunner":
            runner = runner_class(model_path, n_ctx=2048, n_gpu_layers=-1, verbose=False)
        else:
            runner = runner_class(model_path, verbose=False)
        
        load_time = time.time() - load_start
        logger.info("Modelo cargado en %.2f segundos", load_time)
        
    except Exception as e:
        logger.error("Error al cargar modelo: %s", e)
        return {"success": False, "error": str(e)}
    
    # Iniciar recolección de métricas de hardware
    metrics_collector.start_collection()
    
    # Ejecutar inferencia
    inference_start = time.time()
    
    try:
        result = runner.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        inference_time = time.time() - inference_start
        tokens_generated = result.get("tokens", 0)
        tokens_per_sec = tokens_generated / inference_time if inference_time > 0 else 0
        
        logger.info("Generados %d tokens en %.2f segundos (%.2f tok/s)", 
                   tokens_generated, inference_time, tokens_per_sec)
        
    except Exception as e:
        logger.error("Error en inferencia: %s", e)
        return {"success": False, "error": str(e)}
    finally:
        # Detener recolección de métricas (siempre se ejecuta, incluso con error)
        avg_hardware_metrics = metrics_collector.stop_collection()
        
        # Calcular potencia promedio basada en uso de CPU
        avg_cpu_percent = avg_hardware_metrics.get("cpu_percent_avg", 50)
        avg_power_w = energy_estimator.estimate_power(avg_cpu_percent, gpu_active=True)
        
        # Calcular energía y emisiones (usar inference_time seguro)
        energy_results = energy_estimator.calculate_energy(avg_power_w, inference_time)
    
    # Limpiar recursos del runner si tiene método cleanup
    if runner and hasattr(runner, 'cleanup'):
        runner.cleanup()
    
    # Compilar resultados completos para exportación
    benchmark_results = {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "model": {
            "name": model_name,
            "quantization": quantization,
            "path": model_path,
            "framework": runner_class.__name__
        },
        "inference": {
            "tokens_generated": tokens_generated,
            "inference_time_s": round(inference_time, 3),
            "tokens_per_sec": round(tokens_generated / inference_time, 2) if inference_time > 0 else 0.0,
            "prompt_tokens": result.get("prompt_tokens", 0) if result else 0,
            "temperature": temperature
        },
        "hardware": avg_hardware_metrics,
        "energy": energy_results,
        "load_time_s": round(load_time, 3)
    }
    
    return benchmark_results


def generate_comparison_table(results: List[Dict], output_path: str = "results/comparison_table.csv"):
    """
    Genera tabla comparativa en formato CSV para análisis estadístico.
    
    Agrupa los resultados por configuración de modelo y calcula promedios
    de todas las métricas. La tabla incluye emisiones de CO₂ para todos
    los países configurados en EMISSION_FACTORS.
    
    Args:
        results (list): Lista de diccionarios con resultados de benchmarks
        output_path (str): Ruta para guardar el archivo CSV. Por defecto: "results/comparison_table.csv"
    """
    if not results:
        logger.warning("No hay resultados para generar tabla comparativa")
        return
    
    # Agrupar resultados por configuración (modelo + cuantización)
    configs = {}
    for r in results:
        if not r.get("success"):
            continue
        
        key = "%s_%s" % (r['model']['name'], r['model']['quantization'])
        if key not in configs:
            configs[key] = {
                "model_name": r['model']['name'],
                "quantization": r['model']['quantization'],
                "framework": r['model']['framework'],
                "results": []
            }
        configs[key]["results"].append(r)
    
    # Calcular promedios por configuración
    rows = []
    for key, config in configs.items():
        config_results = config["results"]
        
        # Promedios de métricas de inferencia
        avg_tokens_per_sec = sum(r["inference"]["tokens_per_sec"] for r in config_results) / len(config_results)
        avg_inference_time = sum(r["inference"]["inference_time_s"] for r in config_results) / len(config_results)
        avg_load_time = sum(r["load_time_s"] for r in config_results) / len(config_results)
        
        # Promedios de hardware
        avg_cpu_percent = sum(r["hardware"].get("cpu_percent_avg", 0) for r in config_results) / len(config_results)
        avg_cpu_freq = sum(r["hardware"].get("cpu_freq_avg_ghz", 0) for r in config_results) / len(config_results)
        avg_memory_mb = sum(r["hardware"].get("memory_used_avg_mb", 0) for r in config_results) / len(config_results)
        
        # Promedios de energía
        avg_power_w = sum(r["energy"]["power_avg_w"] for r in config_results) / len(config_results)
        avg_energy_wh = sum(r["energy"]["energy_wh"] for r in config_results) / len(config_results)
        
        # Emisiones por país (promedio)
        emissions_by_country = {}
        for country_code in EMISSION_FACTORS.keys():
            country_co2 = [r["energy"]["emissions"][country_code]["co2_g"] for r in config_results]
            emissions_by_country[country_code] = sum(country_co2) / len(country_co2)
        
        row = {
            "Model": config["model_name"],
            "Quantization": config["quantization"],
            "Framework": config["framework"],
            "CPU_Usage_%": round(avg_cpu_percent, 2),
            "CPU_Freq_GHz": round(avg_cpu_freq, 2),
            "Memory_MB": round(avg_memory_mb, 2),
            "Power_Avg_W": round(avg_power_w, 2),
            "Energy_Wh": round(avg_energy_wh, 4),
            "Inference_Time_s": round(avg_inference_time, 3),
            "Load_Time_s": round(avg_load_time, 3),
            "Tokens_Per_Sec": round(avg_tokens_per_sec, 2),
        }
        
        # Agregar emisiones por país al CSV
        for country_code, co2_avg in emissions_by_country.items():
            row["CO2_%s_g" % country_code] = round(co2_avg, 4)
        
        rows.append(row)
    
    # Guardar CSV con encoding utf-8-sig para compatibilidad con Excel
    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info("Tabla comparativa guardada en: %s", output_path)
        
        # Imprimir resumen en consola
        print("\n" + "="*120)
        print("📊 TABLA COMPARATIVA DE MODELOS - MÉTRICAS COMPLETAS")
        print("="*120)
        
        # Encabezado principal
        print("\n%-15s %-8s %-15s %-6s %-8s %-10s %-8s %-10s %-8s" % (
            "Model", "Quant", "Framework", "CPU%", "Freq(GHz)", "Mem(MB)", "Power(W)", "Energy(Wh)", "Tok/s"))
        print("-"*120)
        
        for row in rows:
            print("%-15s %-8s %-15s %-6s %-8s %-10s %-8s %-10s %-8s" % (
                row['Model'], row['Quantization'], row['Framework'],
                row['CPU_Usage_%'], row['CPU_Freq_GHz'], row['Memory_MB'],
                row['Power_Avg_W'], row['Energy_Wh'], row['Tokens_Per_Sec']))
        
        # Emisiones por país
        print("\n🌍 EMISIONES DE CO₂ POR PAÍS (g)")
        print("-"*120)
        country_headers = " | ".join(["%s: %s" % (code, EMISSION_FACTORS[code]['name']) for code in EMISSION_FACTORS.keys()])
        print("Configuración | %s" % country_headers)
        print("-"*120)
        
        for row in rows:
            config_name = "%s_%s" % (row['Model'], row['Quantization'])
            co2_values = " | ".join(["%.4f" % row.get('CO2_%s_g' % code, 0) for code in EMISSION_FACTORS.keys()])
            print("%-15s | %s" % (config_name, co2_values))
        
        print("="*120)


def main():
    """
    Función principal que ejecuta el benchmark comprehensivo completo.
    
    Orquesta todo el proceso:
    1. Registra información del entorno
    2. Importa los runners (LLMRunner, MLXRunner)
    3. Define configuración de modelos a probar
    4. Ejecuta benchmark para cada configuración
    5. Genera tabla comparativa en CSV
    6. Guarda resultados completos en JSON
    """
    logger.info("="*80)
    logger.info("BENCHMARK COMPREHENSIVO - GREEN AI LLM STUDY")
    logger.info("="*80)
    logger.info("Fecha: %s", datetime.now().isoformat())
    logger.info("Plataforma: %s", platform.platform())
    logger.info("Arquitectura: %s", platform.machine())
    logger.info("")
    
    # Importar runners desde src/
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    from llm_runner import LLMRunner
    from mlx_runner import MLXRunner
    
    # Configuración de modelos a probar
    models_to_test = [
        {
            "name": "Qwen2.5-7B",
            "quantization": "INT4",
            "runner": LLMRunner,
            "path": "models/qwen2.5-7b/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
        },
        {
            "name": "Qwen2.5-7B",
            "quantization": "INT8",
            "runner": LLMRunner,
            "path": "models/qwen2.5-7b/Qwen2.5-7B-Instruct-Q8_0.gguf"
        },
        {
            "name": "Llama-2-7B",
            "quantization": "INT4",
            "runner": LLMRunner,
            "path": "models/llama-2-7b/Q4_K_M.gguf"
        },
        {
            "name": "Llama-2-7B",
            "quantization": "INT8",
            "runner": LLMRunner,
            "path": "models/llama-2-7b/llama-2-7b.Q8_0.gguf"
        },
        {
            "name": "Qwen2.5-7B-MLX",
            "quantization": "INT4",
            "runner": MLXRunner,
            "path": "mlx-community/Qwen2.5-7B-Instruct-4bit"
        }
    ]
    
    # Prompt estandarizado para todas las pruebas
    prompt = "Que es la inteligencia artificial y como impacta en la sostenibilidad energetica?"
    
    # Ejecutar benchmarks para cada configuración
    all_results = []
    
    for model_cfg in models_to_test:
        logger.info("\n" + "="*80)
        logger.info("Probando: %s (%s)", model_cfg['name'], model_cfg['quantization'])
        logger.info("="*80)
        
        # Verificar que el archivo existe (para modelos locales)
        if os.path.exists(model_cfg["path"]) or model_cfg["runner"] == MLXRunner:
            result = run_inference_benchmark(
                runner_class=model_cfg["runner"],
                model_path=model_cfg["path"],
                prompt=prompt,
                max_tokens=100,
                temperature=0.1,
                model_name=model_cfg["name"],
                quantization=model_cfg["quantization"]
            )
            
            if result.get("success"):
                all_results.append(result)
                logger.info("✅ Benchmark completado: %s (%s)", model_cfg['name'], model_cfg['quantization'])
            else:
                logger.error("❌ Error en benchmark: %s", result.get('error', 'Unknown'))
        else:
            logger.warning("⚠️  Modelo no encontrado: %s", model_cfg['path'])
    
    # Generar tabla comparativa en CSV
    logger.info("\nGenerando tabla comparativa...")
    generate_comparison_table(all_results, output_path="results/comparison_table.csv")
    
    # Guardar resultados completos en JSON para análisis posterior
    json_path = "results/comprehensive_benchmark_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("Resultados completos guardados en: %s", json_path)
    
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK COMPREHENSIVO FINALIZADO")
    logger.info("="*80)
    
    return all_results


if __name__ == "__main__":
    main()
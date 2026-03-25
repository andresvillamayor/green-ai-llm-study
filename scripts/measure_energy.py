#!/usr/bin/env python3
"""
Script de medición energética individual
"""

import sys
from pathlib import Path

# Agregar carpeta src/ al path de Python
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Ahora SÍ funcionan los imports (sin src.)
from energy_tracker import EnergyTracker
from llm_runner import LLMRunner
from utils import setup_logging, save_json, append_to_csv

from datetime import datetime
import logging

setup_logging()
logger = logging.getLogger(__name__)


def main():
    print("=" * 70)
    print("🔬 MEDICIÓN ENERGÉTICA - LLM INFERENCE")
    print("=" * 70)
    
    # Configuración
    model_path = "models/llama-2-7b.Q8_0.gguf"
    prompt = "Explica qué es la inteligencia artificial en 50 palabras"
    max_tokens = 100
    
    # Verificar que existe el modelo
    if not Path(model_path).exists():
        logger.error(f"❌ Modelo no encontrado: {model_path}")
        logger.info("💡 Descarga el modelo primero con:")
        logger.info("   bash scripts/download_models.sh")
        return 1
    
    # Inicializar componentes
    logger.info("📦 Inicializando componentes...")
    
    tracker = EnergyTracker(
        project_name="single-measurement",
        output_file="single_measurement_emissions.csv"
    )
    
    runner = LLMRunner(
        model_path=model_path,
        n_ctx=2048,
        n_threads=8,
        n_gpu_layers=-1
    )
    
    # Información del modelo
    model_info = runner.get_model_info()
    logger.info(f"📊 Modelo: {model_info['name']}")
    logger.info(f"📏 Tamaño: {model_info['size_mb']:.2f} MB")
    
    # Iniciar medición
    logger.info(f"💬 Prompt: '{prompt}'")
    logger.info(f"🎯 Max tokens: {max_tokens}")
    
    tracker.start()
    
    try:
        result = runner.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9
        )
        
        logger.info(f"📝 Respuesta generada:")
        print("\n" + "─" * 70)
        print(result['text'])
        print("─" * 70 + "\n")
        
    except Exception as e:
        logger.error(f"❌ Error durante generación: {e}")
        return 1
    finally:
        tracker.stop()
    
    # Recopilar resultados
    energy_stats = tracker.get_stats()
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_info['name'],
        "model_size_mb": model_info['size_mb'],
        "prompt": prompt,
        "max_tokens_requested": max_tokens,
        "tokens_generated": result['tokens'],
        "prompt_tokens": result['prompt_tokens'],
        "total_tokens": result['total_tokens'],
        "generation_time_s": result['time_s'],
        "tokens_per_second": result['tokens_per_s'],
        "emissions_kg_co2": energy_stats['emissions_kg_co2'],
        "emissions_g_co2": energy_stats['emissions_g_co2'],
        "energy_wh": energy_stats['energy_wh'],
        "energy_kwh": energy_stats['energy_kwh'],
        "joules_per_token": (energy_stats['energy_wh'] * 3600) / result['tokens'],
        "g_co2_per_token": energy_stats['emissions_g_co2'] / result['tokens'],
        "country": "PRY",
        "carbon_intensity": 0.07
    }
    
    # Mostrar resumen
    print("\n" + "=" * 70)
    print("📊 RESULTADOS DE LA MEDICIÓN")
    print("=" * 70)
    print(f"\n⚡ GENERACIÓN:")
    print(f"   • Tokens generados:    {results['tokens_generated']}")
    print(f"   • Tiempo:              {results['generation_time_s']:.2f}s")
    print(f"   • Velocidad:           {results['tokens_per_second']:.2f} tokens/s")
    
    print(f"\n🌍 ENERGÍA Y EMISIONES:")
    print(f"   • Emisiones totales:   {results['emissions_g_co2']:.4f} g CO2e")
    print(f"   • Energía consumida:   {results['energy_wh']:.4f} Wh")
    print(f"   • Por token:           {results['g_co2_per_token']:.6f} g CO2e/token")
    print(f"   • Energía/token:       {results['joules_per_token']:.2f} J/token")
    
    print(f"\n🇵🇾 CONTEXTO PARAGUAY:")
    print(f"   • Intensidad carbono:  {results['carbon_intensity']} kg CO2/kWh")
    print(f"   • (95% energía hidroeléctrica)")
    
    print("=" * 70 + "\n")
    
    # Guardar resultados
    json_file = save_json(results, filename="last_measurement.json")
    csv_file = append_to_csv(results)
    
    logger.info(f"💾 Resultados guardados:")
    logger.info(f"   • JSON: {json_file}")
    logger.info(f"   • CSV:  {csv_file}")
    
    logger.info("\n✅ Medición completada exitosamente")
    return 0


if __name__ == "__main__":
    exit(main())

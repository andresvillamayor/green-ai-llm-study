"""
experimento_codecarbon.py

Medicion de consumo energetico en inferencia de modelos LLM
con cuantizacion Q4 y Q8 en CPU y GPU.

Utiliza CodeCarbon para medir consumo de CPU, GPU y RAM.

Diseno experimental:
- 2 modelos: Llama-2-7B, Qwen2.5-7B
- 2 cuantizaciones: Q4, Q8
- 2 dispositivos: CPU, GPU
- 10 prompts cientificos
- 15 repeticiones por configuracion
- Total: 1200 mediciones

Autor: Andres Villamayor
Maestria en Ciencia de Datos
Universidad Comunero
Asuncion, Paraguay
"""

import pandas as pd
from pathlib import Path
from llama_cpp import Llama
from codecarbon import EmissionsTracker
import time
from datetime import datetime
import sys

from src.benchmark_prompts_scientific import PROMPTS_CIENTIFICOS
from src.carbon_factors import CARBON_INTENSITY_PARAGUAY


# Configuracion de rutas
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "measurements"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Modelos a evaluar
MODELOS = {
    'llama-2-7b': {
        'q4': MODELS_DIR / "llama-2-7b" / "Q4_K_M.gguf",
        'q8': MODELS_DIR / "llama-2-7b" / "llama-2-7b.Q8_0.gguf"
    },
    'qwen2.5-7b': {
        'q4': MODELS_DIR / "qwen2.5-7b" / "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        'q8': MODELS_DIR / "qwen2.5-7b" / "Qwen2.5-7B-Instruct-Q8_0.gguf"
    }
}

# Parametros de inferencia
PARAMS_INFERENCIA = {
    'n_ctx': 2048,
    'temperature': 0.7,
    'top_p': 0.9,
    'max_tokens': 512,
    'echo': False,
    'stop': None
}

# Configuracion del experimento
NUM_REPETICIONES = 15
DISPOSITIVOS = ['cpu', 'gpu']

# Mensaje inicial
print("=" * 70)
print("EXPERIMENTO: Medicion de Consumo Energetico en LLMs")
print("=" * 70)
print(f"Proyecto: GREEN-IA")
print(f"Autor: Andres Villamayor")
print(f"Universidad: Comunero")
print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total configuraciones: {len(MODELOS) * 2 * len(DISPOSITIVOS)}")
print(f"Total prompts: {len(PROMPTS_CIENTIFICOS)}")
print(f"Repeticiones: {NUM_REPETICIONES}")
total = len(MODELOS) * 2 * len(DISPOSITIVOS) * len(PROMPTS_CIENTIFICOS) * NUM_REPETICIONES
print(f"Total mediciones: {total}")
print("=" * 70)
print()

def ejecutar_inferencia_con_medicion(model_path, prompt, device, prompt_id, paper_ref, repetition):
    """
    Ejecuta una inferencia y mide consumo energetico.
    
    Parametros:
        model_path: ruta al archivo .gguf
        prompt: texto del prompt
        device: 'cpu' o 'gpu'
        prompt_id: numero de prompt (1-10)
        paper_ref: referencia del paper
        repetition: numero de repeticion (1-15)
    
    Retorna:
        diccionario con resultados o None si hay error
    """
    
    n_gpu_layers = -1 if device == 'gpu' else 0
    
    model_name = model_path.parent.parent.name
    quant = 'q4' if 'Q4' in model_path.name else 'q8'
    
    try:
        llm = Llama(
            model_path=str(model_path),
            n_ctx=PARAMS_INFERENCIA['n_ctx'],
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )
    except Exception as e:
        print(f"[ERROR] Carga de modelo: {e}")
        return None
    
    tracker = EmissionsTracker(
        project_name="green-ia",
        output_dir=".",
        output_file="emissions_temp.csv",
        log_level="error",
        save_to_file=False,
        country_iso_code="PRY"
    )
    
    tracker.start()
    tiempo_inicio = time.time()
    
    try:
        respuesta = llm(
            prompt,
            max_tokens=PARAMS_INFERENCIA['max_tokens'],
            temperature=PARAMS_INFERENCIA['temperature'],
            top_p=PARAMS_INFERENCIA['top_p'],
            echo=PARAMS_INFERENCIA['echo'],
            stop=PARAMS_INFERENCIA['stop']
        )
    except Exception as e:
        print(f"[ERROR] Inferencia: {e}")
        tracker.stop()
        return None
    
    tiempo_fin = time.time()
    tiempo_total = tiempo_fin - tiempo_inicio
    
    emisiones = tracker.stop()
    
    texto_generado = respuesta['choices'][0]['text']
    tokens_generados = respuesta['usage']['completion_tokens']
    tokens_por_segundo = tokens_generados / tiempo_total if tiempo_total > 0 else 0
    
    energy_data = tracker._total_energy
    
    resultado = {
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'quantization': quant,
        'device': device,
        'prompt_id': prompt_id,
        'prompt_text': prompt[:100],
        'paper_reference': paper_ref,
        'repetition': repetition,
        'inference_time_s': tiempo_total,
        'tokens_generated': tokens_generados,
        'tokens_per_second': tokens_por_segundo,
        'cpu_power_w': energy_data.cpu_power,
        'gpu_power_w': energy_data.gpu_power,
        'ram_power_w': energy_data.ram_power,
        'cpu_energy_wh': energy_data.cpu_energy,
        'gpu_energy_wh': energy_data.gpu_energy,
        'ram_energy_wh': energy_data.ram_energy,
        'total_energy_wh': energy_data.energy_consumed,
        'emissions_kg_co2': emisiones,
        'emissions_mg_co2': emisiones * 1000000,
        'carbon_intensity': CARBON_INTENSITY_PARAGUAY,
        'duration_s': energy_data.duration,
        'country_iso_code': 'PRY',
        'cpu_count': 10,
        'cpu_model': 'Apple M4',
        'gpu_count': 1,
        'gpu_model': 'Apple M4',
        'ram_total_gb': 16.0
    }
    
    return resultado

def ejecutar_experimento():
    """
    Ejecuta el experimento completo de 1200 mediciones.
    Itera sobre todas las combinaciones y guarda resultados.
    """
    
    resultados = []
    contador = 0
    errores = 0
    inicio = time.time()
    
    total_esperado = len(MODELOS) * 2 * len(DISPOSITIVOS) * len(PROMPTS_CIENTIFICOS) * NUM_REPETICIONES
    
    for nombre_modelo, rutas_modelo in MODELOS.items():
        
        for quant, ruta_modelo in rutas_modelo.items():
            
            for dispositivo in DISPOSITIVOS:
                
                print("\n" + "-" * 70)
                print(f"Modelo: {nombre_modelo}")
                print(f"Cuantizacion: {quant}")
                print(f"Dispositivo: {dispositivo}")
                print("-" * 70)
                
                for idx_prompt, prompt_data in enumerate(PROMPTS_CIENTIFICOS):
                    
                    texto_prompt = prompt_data['prompt']
                    referencia = prompt_data['referencia']
                    
                    print(f"\nPrompt {idx_prompt + 1}/{len(PROMPTS_CIENTIFICOS)}")
                    print(f"Referencia: {referencia}")
                    
                    for num_rep in range(1, NUM_REPETICIONES + 1):
                        
                        print(f"  Repeticion {num_rep}/{NUM_REPETICIONES}...", end=" ")
                        
                        resultado = ejecutar_inferencia_con_medicion(
                            model_path=ruta_modelo,
                            prompt=texto_prompt,
                            device=dispositivo,
                            prompt_id=idx_prompt + 1,
                            paper_ref=referencia,
                            repetition=num_rep
                        )
                        
                        if resultado is not None:
                            resultados.append(resultado)
                            contador += 1
                            print("[OK]")
                        else:
                            errores += 1
                            print("[ERROR]")
                        
                        if contador % 50 == 0 and contador > 0:
                            df_backup = pd.DataFrame(resultados)
                            archivo_backup = RESULTS_DIR / f"backup_{contador}.csv"
                            df_backup.to_csv(archivo_backup, index=False)
                            print(f"  Backup guardado: {archivo_backup.name}")
                        
                        porcentaje = (contador / total_esperado) * 100
                        transcurrido = time.time() - inicio
                        promedio = transcurrido / contador if contador > 0 else 0
                        restantes = total_esperado - contador
                        tiempo_restante = restantes * promedio
                        
                        print(f"  Progreso: {contador}/{total_esperado} ({porcentaje:.1f}%)")
                        print(f"  Tiempo: {transcurrido/60:.1f} min | Restante: {tiempo_restante/60:.1f} min")
    
    tiempo_total = time.time() - inicio
    
    print("\n" + "=" * 70)
    print("EXPERIMENTO FINALIZADO")
    print("=" * 70)
    print(f"Mediciones exitosas: {contador}")
    print(f"Errores: {errores}")
    print(f"Tiempo total: {tiempo_total/60:.1f} minutos")
    
    if len(resultados) > 0:
        df = pd.DataFrame(resultados)
        
        fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"experimento_codecarbon_{fecha}.csv"
        ruta_archivo = RESULTS_DIR / nombre_archivo
        
        df.to_csv(ruta_archivo, index=False)
        
        print(f"\nArchivo guardado: {ruta_archivo}")
        print(f"Registros: {len(df)}")
        print(f"Columnas: {len(df.columns)}")
        
        return ruta_archivo
    else:
        print("\nNo se generaron resultados")
        return None
    
if __name__ == "__main__":
    
    print("\nPresiona Ctrl+C para cancelar")
    print("Los backups se guardan cada 50 mediciones\n")
    
    try:
        archivo_final = ejecutar_experimento()
        
        if archivo_final:
            print("\nExperimento completado")
            print(f"Resultados en: {archivo_final}")
    
    except KeyboardInterrupt:
        print("\n\nExperimento cancelado")
        print("Revisa archivos backup_*.csv en results/measurements/")
    
    except Exception as error:
        print(f"\n\n[ERROR] Experimento: {error}")
        import traceback
        traceback.print_exc()

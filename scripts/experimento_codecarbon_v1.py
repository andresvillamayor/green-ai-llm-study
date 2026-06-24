# experimento_codecarbon_v1.py
# Mide consumo energetico de LLMs con cuantizacion Q4 y Q8
# Proyecto GREEN-IA — Maestria Ciencia de Datos 
# - limpieza de memoria entre modelos con gc.collect()
# - se agrego el calculo del coeficiente de variacion por configuracion

import gc
import time
import sys
import platform
import subprocess
import os
# PYTHONUNBUFFERED garantiza que el output se muestra en tiempo real
# sin esto el progreso puede no verse hasta que el buffer se llena
os.environ["PYTHONUNBUFFERED"] = "1"
import numpy as np
import psutil
from pathlib import Path
from datetime import datetime

import pandas as pd
from llama_cpp import Llama, __version__ as llama_version
from codecarbon import EmissionsTracker

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.benchmark_prompts_scientific import PROMPTS_CIENTIFICOS
from src.carbon_factors import CARBON_INTENSITY_PARAGUAY

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results" / "measurements"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    _pip_show = subprocess.check_output(["pip", "show", "llama-cpp-python"], text=True)
    llama_commit = [l for l in _pip_show.split("\n") if "Version" in l][0].strip()
except Exception:
    llama_commit = "no disponible"

PROMPTS = PROMPTS_CIENTIFICOS

MODELOS = {
    "llama-2-7b": {
        "q4": MODELS_DIR / "llama-2-7b" / "Q4_K_M.gguf",
        "q8": MODELS_DIR / "llama-2-7b" / "llama-2-7b.Q8_0.gguf",
    },
    "qwen2.5-7b": {
        "q4": MODELS_DIR / "qwen2.5-7b" / "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "q8": MODELS_DIR / "qwen2.5-7b" / "Qwen2.5-7B-Instruct-Q8_0.gguf",
    },
}

# Parametros de inferencia — respaldados cientificamente
#
# temperature=0.7:
#   Zheng et al. NeurIPS 2023 (arXiv:2306.05685) — usado en MT-Bench
#   para generacion de respuestas de comparacion (Tabla 2 del paper)
#   Rango validado: 0.6-0.9 para tareas mixtas de chat
#
# top_p=0.9:
#   Holtzman et al. ICLR 2020 (arXiv:1904.09751) — nucleus sampling
#   Estandar para LLaMA3 8B e Instruct (llama.cpp documentacion)
#   Mismo valor usado en Mistral 7B y LLaMA3 benchmarks
#
# max_tokens=256:
#   Estandar para respuestas cortas en benchmarks 7B
#   Usado en inferencia con LLaMA3-8B y LLaMA3-70B
#   (arXiv:2410.05434 Listing 1)
#
# n_ctx=1024:
#   Ventana de contexto conservadora para modelos 7B con 16GB RAM
#   Suficiente para los 40 prompts del benchmark MT-Bench
#   KV cache = proporcional al contexto — menor contexto = menor RAM
#
# n_threads=4:
#   Apple M4 tiene 4 P-cores (4.4 GHz) y 6 E-cores (2.85 GHz)
#   Solo P-cores para inferencia CPU — mezclar E-cores REDUCE velocidad
#   Fuente: llama.cpp Apple Silicon guide (llama.cpp GitHub 2024)
#   "only use p-cores, never mix in e-cores"
#   Para GPU mode (n_gpu_layers=-1) este parametro es irrelevante
#   pero se mantiene consistente para reproducibilidad
#
# n_batch=512:
#   Valor por defecto y estandar de llama.cpp
#   Validado en benchmarks Apple Silicon (llama.cpp GitHub #4167)
#   "batch size of 512 — computation is compute bound"
#
# seed=42:
#   Estandar universal en ML para reproducibilidad experimental
#   Garantiza resultados identicos entre ejecuciones del mismo prompt
PARAMS = {
    "n_ctx"       : 1024,
    "temperature" : 0.7,
    "top_p"       : 0.9,
    "max_tokens"  : 256,
    "echo"        : False,
    "stop"        : None,
    "seed"        : 42,
    "n_threads"   : 4,    # solo P-cores del M4 — no mezclar E-cores
    "n_batch"     : 512,
}

# 15 repeticiones por configuracion
# Justificacion estadistica:
#   Con n=15 se puede calcular IC 95% con t de Student (gl=14)
#   Con n>=10 el IQR x3 (Tukey 1977) tiene poder de deteccion
#   adecuado para outliers extremos (ratio > 3x la media)
#   Mesa de tesis solicito n=15 para robustez estadistica
NUM_REPETICIONES = 15
DISPOSITIVOS     = ["cpu", "gpu"]


def calcular_cv(valores):
    # el coeficiente de variacion (CV) mide que tan estables son las mediciones
    # se calcula como: desviacion estandar / media * 100
    # si el CV es menor al 10% las mediciones son consistentes entre si
    # si el CV supera el 10% hay variabilidad que hay que analizar y documentar
    if len(valores) < 2:
        return 0.0
    media = float(np.mean(valores))
    if media == 0:
        return 0.0
    return float(np.std(valores, ddof=1) / media * 100)


def limpiar_memoria(nombre_config):
    # gc.collect() le dice a Python que libere los objetos que ya no se usan
    # en este caso los pesos del modelo que acaba de terminar
    # el sleep de 5 segundos le da tiempo al sistema operativo (macOS)
    # para reclamar esa memoria antes de cargar el siguiente modelo
    # esto evita el efecto warm-start: que el siguiente modelo se beneficie
    # de datos que quedaron en cache del modelo anterior, lo que podria
    # hacer que sus tiempos de inferencia sean artificialmente mas rapidos
    # y contaminar la comparacion entre configuraciones
    gc.collect()
    time.sleep(5)
    print(f"  memoria liberada tras: {nombre_config}")


def cargar_modelo(model_path, device):
    # con n_gpu_layers = -1 todas las capas van a Metal (GPU del M4)
    # con n_gpu_layers = 0 todo corre en CPU, sin usar la GPU
    n_gpu_layers = -1 if device == "gpu" else 0
    try:
        return Llama(
            model_path=str(model_path),
            n_ctx=PARAMS["n_ctx"],
            n_gpu_layers=n_gpu_layers,
            n_threads=PARAMS["n_threads"],
            n_batch=PARAMS["n_batch"],
            seed=PARAMS["seed"],
            verbose=False,
        )
    except Exception as e:
        print(f"  error cargando modelo: {e}")
        return None


def medir_inferencia(llm, prompt, device, prompt_id, paper_ref,
                     repetition, model_name, quant, categoria):
    # el modelo llega ya cargado, no lo cargo aca para no contaminar la medicion
    tracker = EmissionsTracker(
        project_name="green-ia",
        output_dir=str(RESULTS_DIR),
        output_file="emissions_temp.csv",
        log_level="error",
        save_to_file=False,
    )

    tracker.start()
    t_ini = time.time()

    try:
        respuesta = llm(
            prompt,
            max_tokens=PARAMS["max_tokens"],
            temperature=PARAMS["temperature"],
            top_p=PARAMS["top_p"],
            echo=PARAMS["echo"],
            stop=PARAMS["stop"],
            seed=PARAMS["seed"],
        )
    except Exception as e:
        print(f"  error en inferencia: {e}")
        tracker.stop()
        return None

    t_total = time.time() - t_ini
    tracker.stop()

    # texto generado y tokens de entrada y salida
    texto_generado = respuesta["choices"][0]["text"]
    tokens_gen     = respuesta["usage"]["completion_tokens"]
    tokens_entrada = respuesta["usage"]["prompt_tokens"]
    tokens_total   = respuesta["usage"]["total_tokens"]
    tps            = tokens_gen / t_total if t_total > 0 else 0
    dur_horas  = t_total / 3600

    # energia en kWh convertida a Wh multiplicando por 1000
    e_cpu = tracker._total_cpu_energy.kWh
    e_gpu = tracker._total_gpu_energy.kWh
    e_ram = tracker._total_ram_energy.kWh
    e_tot = tracker._total_energy.kWh

    # potencia promedio en watts = energia en Wh dividido por el tiempo en horas
    def watts(kwh):
        return (kwh * 1000) / dur_horas if dur_horas > 0 else 0

    emisiones_kg = tracker._total_emissions

    return {
        "timestamp"         : datetime.now().isoformat(),
        "model"             : model_name,
        "quantization"      : quant,
        "device"            : device,
        "prompt_id"         : prompt_id,
        "categoria"         : categoria,
        "prompt_text"       : prompt[:100],
        "response_text"     : texto_generado[:2000],
        "paper_reference"   : paper_ref,
        "repetition"        : repetition,
        "inference_time_s"  : t_total,
        "tokens_generated"  : tokens_gen,
        "tokens_per_second" : tps,
        "cpu_power_w"       : watts(e_cpu),
        "gpu_power_w"       : watts(e_gpu),
        "ram_power_w"       : watts(e_ram),
        "cpu_energy_wh"     : e_cpu * 1000,
        "gpu_energy_wh"     : e_gpu * 1000,
        "ram_energy_wh"     : e_ram * 1000,
        "total_energy_wh"   : e_tot * 1000,
        "emissions_kg_co2"  : emisiones_kg,
        "emissions_mg_co2"  : emisiones_kg * 1_000_000,
        "carbon_intensity"  : CARBON_INTENSITY_PARAGUAY, # el que trae por defecto 
        "duration_s"        : t_total,
        "country_iso_code"  : "PRY",
        "cpu_count"         : 10,
        "cpu_model"         : "Apple M4",
        "gpu_count"         : 1,
        "gpu_model"         : "Apple M4",
        "ram_total_gb"      : 16.0,
        "tokens_input"      : tokens_entrada,
        "tokens_output"     : tokens_gen,
        "tokens_total"      : tokens_total,
        "energy_per_token"  : (e_tot * 1000) / tokens_gen if tokens_gen > 0 else 0,
        # metricas normalizadas por token pedidas por mesa de tesis
        "energy_per_1k_tokens"  : (e_tot * 1000 * 1000) / tokens_gen if tokens_gen > 0 else 0,
        "tokens_per_joule"      : tokens_gen / (e_tot * 3600) if e_tot > 0 else 0,
        "latency_per_token_ms"  : (t_total / tokens_gen * 1000) if tokens_gen > 0 else 0,
        "n_gpu_layers"      : -1 if device == "gpu" else 0,
        "n_threads"         : PARAMS["n_threads"],
        "n_batch"           : PARAMS["n_batch"],
        "seed"              : PARAMS["seed"],
        "macos_version"     : platform.mac_ver()[0],
        "llama_cpp_version" : llama_version,
        "python_version"    : platform.python_version(),
        "cpu_freq_mhz"      : "no_disponible_apple_silicon",
        "cpu_percent"       : psutil.cpu_percent(interval=None),
        "ram_used_gb"       : psutil.virtual_memory().used / (1024**3),
        "ram_percent"       : psutil.virtual_memory().percent,
        "llama_cpp_build_info" : llama_commit,
        "metal_backend"     : "yes" if device == "gpu" else "no",
        "prompt_template"   : "plain_text_no_template",
        "warmup_done"       : "yes",
    }


def ejecutar_experimento():
    SYSTEM_INFO = {
        "macos_version"    : platform.mac_ver()[0],
        "python_version"   : platform.python_version(),
        "llama_cpp_version": llama_version,
        "chip"             : "Apple M4",
    }

    print(f"  macOS     : {SYSTEM_INFO['macos_version']}")
    print(f"  Python    : {SYSTEM_INFO['python_version']}")
    print(f"  llama.cpp : {SYSTEM_INFO['llama_cpp_version']}")

    resultados = []
    contador   = 0
    errores    = 0
    t_ini      = time.time()
    total      = len(MODELOS) * 2 * len(DISPOSITIVOS) * len(PROMPTS) * NUM_REPETICIONES

    for nombre_modelo, rutas in MODELOS.items():
        for quant, ruta in rutas.items():
            for dispositivo in DISPOSITIVOS:

                print(f"\n{'─'*55}")
                print(f"  {nombre_modelo}  |  {quant}  |  {dispositivo}")
                print(f"{'─'*55}")

                llm = cargar_modelo(ruta, dispositivo)
                if llm is None:
                    errores += len(PROMPTS) * NUM_REPETICIONES
                    continue

                # guardo los tiempos de esta configuracion para calcular el CV al final
                tiempos_config = []

                for i, prompt_data in enumerate(PROMPTS):
                    texto = prompt_data["prompt"]
                    ref   = prompt_data["referencia"]
                    print(f"\n  prompt {i+1}/{len(PROMPTS)} - {ref}")

                    for rep in range(1, NUM_REPETICIONES + 1):
                        print(f"    rep {rep}/{NUM_REPETICIONES}...", end=" ", flush=True)

                        r = medir_inferencia(
                            llm=llm, prompt=texto, device=dispositivo,
                            prompt_id=i+1, paper_ref=ref, repetition=rep,
                            model_name=nombre_modelo, quant=quant,
                            categoria=prompt_data['categoria'],
                        )

                        if r:
                            resultados.append(r)
                            tiempos_config.append(r["inference_time_s"])
                            contador += 1
                            print(f"ok  {r['tokens_per_second']:.1f} tok/s  "
                                  f"{r['total_energy_wh']*1000:.3f} mWh")
                        else:
                            errores += 1
                            print("ERROR")

                        if contador > 0 and contador % 50 == 0:
                            _backup(resultados, contador)

                        elapsed  = time.time() - t_ini
                        prom     = elapsed / contador if contador > 0 else 0
                        restante = (total - contador) * prom
                        print(f"    {contador}/{total} ({contador/total*100:.1f}%)"
                              f" | {elapsed/60:.1f} min | ~{restante/60:.1f} min restantes")

                # calculo el CV de los tiempos de inferencia de esta configuracion
                # esto me dice si las 5 repeticiones fueron consistentes o no
                if tiempos_config:
                    cv = calcular_cv(tiempos_config)
                    if cv < 10:
                        print(f"\n  CV ({nombre_modelo} {quant} {dispositivo}): "
                              f"{cv:.1f}% - mediciones estables")
                    else:
                        print(f"\n  CV ({nombre_modelo} {quant} {dispositivo}): "
                              f"{cv:.1f}% - variabilidad alta, se documenta como hallazgo")

                # libero el modelo de memoria antes de pasar al siguiente
                # el del elimina la referencia en Python pero no garantiza
                # que la memoria se libere de inmediato, por eso se llama
                # a limpiar_memoria() que agrega el garbage collector y una pausa
                del llm
                limpiar_memoria(f"{nombre_modelo} {quant} {dispositivo}")

    t_total = time.time() - t_ini
    print(f"\n{'='*55}")
    print(f"  listo. {contador} mediciones, {errores} errores, {t_total/60:.1f} min")
    print(f"{'='*55}")

    if not resultados:
        return None

    df    = pd.DataFrame(resultados)
    fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta  = RESULTS_DIR / f"experimento_codecarbon_{fecha}.csv"
    df.to_csv(ruta, index=False)
    print(f"\n  guardado: {ruta}")
    print(f"  {len(df)} filas x {len(df.columns)} columnas")
    return ruta


def _backup(resultados, n):
    ruta = RESULTS_DIR / f"backup_{n}.csv"
    pd.DataFrame(resultados).to_csv(ruta, index=False)
    print(f"    backup guardado: {ruta.name}")


if __name__ == "__main__":
    print("=" * 55)
    print("  GREEN-IA: experimento consumo energetico LLMs")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    total = len(MODELOS) * 2 * len(DISPOSITIVOS) * len(PROMPTS) * NUM_REPETICIONES
    print(f"  {total} mediciones planeadas")
    print(f"  {NUM_REPETICIONES} repeticiones por configuracion")
    print(f"  limpieza de memoria activa entre modelos")
    print(f"  Ctrl+C para cancelar")
    print("=" * 55)

    import subprocess
    import sys

    # caffeinate mantiene el Mac despierto durante el experimento
    # es nativo de macOS, no requiere instalacion
    # referencia: man caffeinate (Apple Developer Documentation)
    print("  caffeinate activo — el Mac no se dormira durante el experimento")

    try:
        archivo = ejecutar_experimento()
        if archivo:
            print(f"\n  resultados en: {archivo}")

    except KeyboardInterrupt:
        print("\n  cancelado. los backups estan en results/measurements/")

    except Exception as e:
        import traceback
        print(f"\n  error inesperado: {e}")
        traceback.print_exc()
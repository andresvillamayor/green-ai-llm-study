# experimento_codecarbon.py
# Mide consumo energetico de LLMs con cuantizacion Q4 y Q8
# Proyecto GREEN-IA — Maestria Ciencia de Datos — Andres Villamayor

import time
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from llama_cpp import Llama
from codecarbon import EmissionsTracker

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.benchmark_prompts_scientific import PROMPTS_CIENTIFICOS
from src.carbon_factors import CARBON_INTENSITY_PARAGUAY

# rutas
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results" / "measurements"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# para prueba uso 2 prompts, para el experimento completo sacar el [:2]
PROMPTS = PROMPTS_CIENTIFICOS

# los 4 modelos descargados en formato GGUF
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

# mismos parametros para todos los modelos, asi la comparacion es justa
PARAMS = {
    "n_ctx"       : 1024,
    "temperature" : 0.7,
    "top_p"       : 0.9,
    "max_tokens"  : 256,
    "echo"        : False,
    "stop"        : None,
}

NUM_REPETICIONES = 10
DISPOSITIVOS     = ["cpu", "gpu"]


def cargar_modelo(model_path, device):
    # gpu usa Metal del M4, cpu no usa GPU
    n_gpu_layers = -1 if device == "gpu" else 0
    try:
        return Llama(
            model_path=str(model_path),
            n_ctx=PARAMS["n_ctx"],
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
    except Exception as e:
        print(f"  Error cargando modelo: {e}")
        return None


def medir_inferencia(llm, prompt, device, prompt_id, paper_ref, repetition, model_name, quant):
    # el modelo llega ya cargado, no lo cargo aca para no contaminar la medicion
    tracker = EmissionsTracker(
        project_name="green-ia",
        output_dir=str(RESULTS_DIR),
        output_file="emissions_temp.csv",
        log_level="error",
        save_to_file=False
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
        )
    except Exception as e:
        print(f"  Error en inferencia: {e}")
        tracker.stop()
        return None

    t_total = time.time() - t_ini
    tracker.stop()

    tokens_gen = respuesta["usage"]["completion_tokens"]
    tps        = tokens_gen / t_total if t_total > 0 else 0
    dur_horas  = t_total / 3600

    # energia en kWh -> convierto a Wh multiplicando por 1000
    e_cpu = tracker._total_cpu_energy.kWh
    e_gpu = tracker._total_gpu_energy.kWh
    e_ram = tracker._total_ram_energy.kWh
    e_tot = tracker._total_energy.kWh

    # potencia promedio W = Wh / h
    def watts(kwh):
        return (kwh * 1000) / dur_horas if dur_horas > 0 else 0

    emisiones_kg = tracker._total_emissions

    return {
        "timestamp"        : datetime.now().isoformat(),
        "model"            : model_name,
        "quantization"     : quant,
        "device"           : device,
        "prompt_id"        : prompt_id,
        "prompt_text"      : prompt[:100],
        "paper_reference"  : paper_ref,
        "repetition"       : repetition,
        "inference_time_s" : t_total,
        "tokens_generated" : tokens_gen,
        "tokens_per_second": tps,
        "cpu_power_w"      : watts(e_cpu),
        "gpu_power_w"      : watts(e_gpu),
        "ram_power_w"      : watts(e_ram),
        "cpu_energy_wh"    : e_cpu * 1000,
        "gpu_energy_wh"    : e_gpu * 1000,
        "ram_energy_wh"    : e_ram * 1000,
        "total_energy_wh"  : e_tot * 1000,
        "emissions_kg_co2" : emisiones_kg,
        "emissions_mg_co2" : emisiones_kg * 1_000_000,
        "carbon_intensity" : CARBON_INTENSITY_PARAGUAY,
        "duration_s"       : t_total,
        "country_iso_code" : "PRY",
        "cpu_count"        : 10,
        "cpu_model"        : "Apple M4",
        "gpu_count"        : 1,
        "gpu_model"        : "Apple M4",
        "ram_total_gb"     : 16.0,
    }


def ejecutar_experimento():
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

                # cargo el modelo una sola vez para toda esta configuracion
                llm = cargar_modelo(ruta, dispositivo)
                if llm is None:
                    errores += len(PROMPTS) * NUM_REPETICIONES
                    continue

                for i, prompt_data in enumerate(PROMPTS):
                    texto = prompt_data["prompt"]
                    ref   = prompt_data["referencia"]
                    print(f"\n  Prompt {i+1}/{len(PROMPTS)} — {ref}")

                    for rep in range(1, NUM_REPETICIONES + 1):
                        print(f"    rep {rep}/{NUM_REPETICIONES}...", end=" ", flush=True)

                        r = medir_inferencia(
                            llm=llm, prompt=texto, device=dispositivo,
                            prompt_id=i+1, paper_ref=ref, repetition=rep,
                            model_name=nombre_modelo, quant=quant,
                        )

                        if r:
                            resultados.append(r)
                            contador += 1
                            print(f"ok  {r['tokens_per_second']:.1f} tok/s  "
                                  f"{r['total_energy_wh']*1000:.3f} mWh")
                        else:
                            errores += 1
                            print("ERROR")

                        # backup por si se corta
                        if contador > 0 and contador % 50 == 0:
                            _backup(resultados, contador)

                        elapsed  = time.time() - t_ini
                        prom     = elapsed / contador if contador > 0 else 0
                        restante = (total - contador) * prom
                        print(f"    {contador}/{total} ({contador/total*100:.1f}%)"
                              f" | {elapsed/60:.1f} min | ~{restante/60:.1f} min restantes")

                del llm  # libero RAM antes de cargar el siguiente

    t_total = time.time() - t_ini
    print(f"\n{'='*55}")
    print(f"  Listo. {contador} mediciones, {errores} errores, {t_total/60:.1f} min")
    print(f"{'='*55}")

    if not resultados:
        return None

    df    = pd.DataFrame(resultados)
    fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta  = RESULTS_DIR / f"experimento_codecarbon_{fecha}.csv"
    df.to_csv(ruta, index=False)
    print(f"\n  Guardado: {ruta}")
    print(f"  {len(df)} filas x {len(df.columns)} columnas")
    return ruta


def _backup(resultados, n):
    ruta = RESULTS_DIR / f"backup_{n}.csv"
    pd.DataFrame(resultados).to_csv(ruta, index=False)
    print(f"    [backup guardado: {ruta.name}]")


if __name__ == "__main__":
    print("=" * 55)
    print("  GREEN-IA: Experimento Consumo Energetico LLMs")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    total = len(MODELOS) * 2 * len(DISPOSITIVOS) * len(PROMPTS) * NUM_REPETICIONES
    print(f"  {total} mediciones planeadas  |  Ctrl+C para cancelar")
    print("=" * 55)

    try:
        archivo = ejecutar_experimento()
        if archivo:
            print(f"\n  Resultados en: {archivo}")

    except KeyboardInterrupt:
        print("\n  Cancelado. Los backups estan en results/measurements/")

    except Exception as e:
        import traceback
        print(f"\n  Error inesperado: {e}")
        traceback.print_exc()
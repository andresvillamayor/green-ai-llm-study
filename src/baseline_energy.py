"""
baseline_energy.py
Proyecto GREEN-IA — Maestria Ciencia de Datos
Universidad Comunero — Paraguay

Medicion de consumo energetico de referencia (baseline) en reposo
y correccion de mediciones de inferencia.

Justificacion metodologica:
  La energia medida durante la inferencia LLM incluye el consumo de
  fondo del sistema operativo (procesos del SO, pantalla, perifericos,
  etc.). Para aislar el consumo especifico del workload LLM se mide
  el consumo en reposo y se resta de la medicion de inferencia.

  Este metodo se conoce como "idle subtraction" o "baseline correction"
  y es el estandar en medicion de eficiencia energetica de software:

    Bannour et al. (2021). Evaluating the Carbon Footprint of NLP
    Models: A Systematic Survey and Experiments. EMNLP 2021.
    "We subtract the idle consumption of the server from the total
    consumption to isolate the ML workload contribution."

    Anthony et al. (2020). Carbontracker: Tracking and Predicting
    the Carbon Footprint of Training Deep Learning Models.
    ICML 2020 Workshop.
    Duracion minima recomendada de baseline: 30 segundos.

Nota sobre Apple Silicon y CodeCarbon:
  En macOS sin root, CodeCarbon usa estimacion constante basada en TDP
  (tracking_mode='constant'). El baseline captura este TDP de fondo y
  permite corregirlo. En Windows con NVIDIA y NVML disponible,
  CodeCarbon usa medicion directa de hardware.

Nota sobre valores negativos:
  En mediciones cortas o con poco contraste energetico, la energia neta
  puede resultar negativa por ruido de medicion. Se aplica max(0, net)
  para evitar valores fisicamente imposibles, y se registra un flag
  'baseline_clip_applied' para trazabilidad.
"""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from codecarbon import EmissionsTracker

# Duracion estandar del baseline (segundos)
# Anthony et al. (2020): minimo 30s para estabilidad estadistica
BASELINE_DURATION_S: int = 30
BASELINE_WARMUP_S:   int = 5    # pausa previa para que el sistema llegue a reposo

# Ruta del CSV de resultados de calibracion de baseline
BASELINE_RAW_DIR  = Path(__file__).parent.parent / "results" / "raw"
BASELINE_CSV_PATH = BASELINE_RAW_DIR / "baseline_results.csv"

BASELINE_CSV_COLUMNS = [
    "timestamp", "config_label", "rep", "duration_s",
    "baseline_cpu_power_w", "baseline_gpu_power_w",
    "baseline_ram_power_w", "baseline_total_power_w",
    "baseline_cpu_energy_wh", "baseline_gpu_energy_wh",
    "baseline_ram_energy_wh", "baseline_total_energy_wh",
]


def _append_baseline_csv(row: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=BASELINE_CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def medir_baseline(duracion_s: int = BASELINE_DURATION_S) -> dict:
    """
    Mide el consumo energetico del sistema en reposo.

    Espera BASELINE_WARMUP_S segundos antes de medir para que el
    sistema llegue a estado estable tras la ultima actividad.

    Parametros:
        duracion_s: duracion de la ventana de medicion en segundos.

    Retorna:
        dict con potencia (W) y energia (Wh) de baseline por componente,
        listo para pasarse a corregir_energia().
    """
    print(
        f"  midiendo baseline ({duracion_s}s en reposo)...",
        end=" ",
        flush=True,
    )

    time.sleep(BASELINE_WARMUP_S)

    # CodeCarbon 3.2.6 — parametros correctos para Apple M4
    # force_cpu_power=20W: TDP tipico M4 durante inferencia LLM
    #   Fuente: NotebookCheck 2024, Apple Support 2024
    # force_ram_power=3W: estimado LPDDR5X 16GB
    # allow_multiple_runs=True: evita error con caffeinate activo
    # La intensidad de carbono se detecta automaticamente por IP
    # CodeCarbon 3.2.6 no acepta country_iso_code en constructor
    tracker = EmissionsTracker(
        project_name       = "green_ai_llm_quality_energy",
        measure_power_secs = 1,
        save_to_file       = False,
        log_level          = "error",
        allow_multiple_runs = True,
        force_cpu_power    = 20,  # TDP Apple M4 en watts
        force_ram_power    = 3,   # RAM LPDDR5X 16GB estimado
    )

    tracker.start()
    time.sleep(duracion_s)
    tracker.stop()

    dur_h = duracion_s / 3600

    def _wh(kwh: float) -> float:
        return kwh * 1000

    def _w(kwh: float) -> float:
        return (_wh(kwh) / dur_h) if dur_h > 0 else 0.0

    try:
        cpu_kwh = tracker._total_cpu_energy.kWh
        gpu_kwh = tracker._total_gpu_energy.kWh
        ram_kwh = tracker._total_ram_energy.kWh
        tot_kwh = tracker._total_energy.kWh
    except AttributeError:
        # fallback si la version de CodeCarbon no expone estos atributos
        cpu_kwh = gpu_kwh = ram_kwh = 0.0
        tot_kwh = getattr(tracker, "_total_energy", type("", (), {"kWh": 0.0})()).kWh

    baseline = {
        "baseline_duration_s"      : duracion_s,
        "baseline_cpu_power_w"     : _w(cpu_kwh),
        "baseline_gpu_power_w"     : _w(gpu_kwh),
        "baseline_ram_power_w"     : _w(ram_kwh),
        "baseline_total_power_w"   : _w(tot_kwh),
        "baseline_cpu_energy_wh"   : _wh(cpu_kwh),
        "baseline_gpu_energy_wh"   : _wh(gpu_kwh),
        "baseline_ram_energy_wh"   : _wh(ram_kwh),
        "baseline_total_energy_wh" : _wh(tot_kwh),
    }

    print(f"ok  {baseline['baseline_total_power_w']:.3f} W")
    return baseline


def medir_baseline_calibracion(
    duracion_s: int,
    repetitions: int,
    config_label: str = "",
    save_to_csv: bool = True,
    baseline_csv_path: Optional[Path] = None,
) -> dict:
    """
    Mide el baseline idle `repetitions` veces y retorna potencias promediadas.

    Flujo por repeticion:
      1. Llama medir_baseline(duracion_s) — incluye pausa de estabilizacion interna.
      2. Guarda la fila en baseline_results.csv si save_to_csv = True.

    El campo `baseline_total_power_w` del resultado es el promedio de todas las
    repeticiones y es el valor que se usa para calcular
    `baseline_corrected_energy_joules` en cada inferencia.

    Referencia:
      Anthony et al. (2020): minimo 30s por medicion; promedio de varias repeticiones
      reduce el ruido de estimacion de TDP en CodeCarbon.

    Retorna:
        dict con los mismos campos que medir_baseline() pero con potencias y
        energias promediadas, mas el campo `baseline_repetitions`.
    """
    if repetitions < 1:
        repetitions = 1
    if baseline_csv_path is None:
        baseline_csv_path = BASELINE_CSV_PATH

    resultados: list[dict] = []
    for rep in range(1, repetitions + 1):
        if repetitions > 1:
            print(f"  [baseline {rep}/{repetitions}]", end=" ", flush=True)
        b = medir_baseline(duracion_s)
        row = {
            "timestamp"              : datetime.now().isoformat(),
            "config_label"           : config_label,
            "rep"                    : rep,
            "duration_s"             : duracion_s,
            **b,
        }
        if save_to_csv:
            _append_baseline_csv(row, baseline_csv_path)
        resultados.append(b)

    def _avg(key: str) -> float:
        return sum(r[key] for r in resultados) / len(resultados)

    avg = {
        "baseline_duration_s"      : duracion_s,
        "baseline_cpu_power_w"     : _avg("baseline_cpu_power_w"),
        "baseline_gpu_power_w"     : _avg("baseline_gpu_power_w"),
        "baseline_ram_power_w"     : _avg("baseline_ram_power_w"),
        "baseline_total_power_w"   : _avg("baseline_total_power_w"),
        "baseline_cpu_energy_wh"   : _avg("baseline_cpu_energy_wh"),
        "baseline_gpu_energy_wh"   : _avg("baseline_gpu_energy_wh"),
        "baseline_ram_energy_wh"   : _avg("baseline_ram_energy_wh"),
        "baseline_total_energy_wh" : _avg("baseline_total_energy_wh"),
        "baseline_repetitions"     : repetitions,
    }

    if repetitions > 1:
        print(
            f"  baseline promedio ({repetitions} reps):"
            f" {avg['baseline_total_power_w']:.3f} W total"
        )
    return avg


def corregir_energia(medicion: dict, baseline: dict) -> dict:
    """
    Resta el consumo de baseline de las mediciones de inferencia.

    Energia neta de inferencia = energia_medida - potencia_baseline * tiempo
    Esto aísla el consumo especifico del LLM del consumo de fondo del sistema.

    Bannour et al. (2021): "idle subtraction" aplicada a cada medicion
    individual usando la potencia promedio del baseline de la misma
    configuracion (mismo modelo, misma cuantizacion, mismo dispositivo).

    Parametros:
        medicion: dict con campos *_energy_wh e inference_time_s.
        baseline: dict retornado por medir_baseline().

    Retorna:
        dict con campos net_*_energy_wh y baseline_subtracted_wh.
        El flag 'baseline_clip_applied' indica si se clipo a 0.
    """
    inf_time_h = medicion.get("inference_time_s", 0.0) / 3600

    cpu_baseline_wh = baseline["baseline_cpu_power_w"] * inf_time_h
    gpu_baseline_wh = baseline["baseline_gpu_power_w"] * inf_time_h
    ram_baseline_wh = baseline["baseline_ram_power_w"] * inf_time_h
    tot_baseline_wh = baseline["baseline_total_power_w"] * inf_time_h

    raw_cpu = medicion.get("cpu_energy_wh", 0.0) - cpu_baseline_wh
    raw_gpu = medicion.get("gpu_energy_wh", 0.0) - gpu_baseline_wh
    raw_ram = medicion.get("ram_energy_wh", 0.0) - ram_baseline_wh
    raw_tot = medicion.get("total_energy_wh", 0.0) - tot_baseline_wh

    clipped = any(v < 0 for v in [raw_cpu, raw_gpu, raw_ram, raw_tot])

    return {
        "net_cpu_energy_wh"      : max(0.0, raw_cpu),
        "net_gpu_energy_wh"      : max(0.0, raw_gpu),
        "net_ram_energy_wh"      : max(0.0, raw_ram),
        "net_total_energy_wh"    : max(0.0, raw_tot),
        "baseline_subtracted_wh" : tot_baseline_wh,
        "baseline_clip_applied"  : int(clipped),
    }


def calcular_emisiones_netas(
    net_energy_wh: float,
    carbon_intensity_g_kwh: int,
) -> dict:
    """
    Calcula emisiones de CO2 a partir de energia neta de inferencia.

    Unidad: gCO2eq/kWh * kWh = gCO2eq
    Factor Paraguay: 26 gCO2eq/kWh (matriz 100% hidroelectrica).
    Fuente: Electricity Maps (2026).

    Parametros:
        net_energy_wh: energia neta de inferencia en Wh.
        carbon_intensity_g_kwh: factor de emision en gCO2eq/kWh.
    """
    net_kwh = net_energy_wh / 1000.0
    net_g   = net_kwh * carbon_intensity_g_kwh
    return {
        "net_emissions_g_co2"  : net_g,
        "net_emissions_kg_co2" : net_g / 1000.0,
        "net_emissions_mg_co2" : net_g * 1000.0,
    }

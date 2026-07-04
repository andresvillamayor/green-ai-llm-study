"""
scripts/fase1_benchmark.py
Proyecto GREEN-IA — Maestria Ciencia de Datos
Universidad Comunero — Paraguay
Autor: Andres Villamayor

Script de medicion de Fase 1: ejecuta el benchmark de energia y calidad sobre
el subset MT-Bench para una configuracion de modelo y cuantizacion.

Llamado por run_experiment.py como subprocess una vez por combinacion
(modelo × cuantizacion). Puede ejecutarse directamente para depuracion.

Principios metodologicos implementados:
  P1   Solo se mide la inferencia.  tracker.start() se llama inmediatamente
       antes de llm(); tracker.stop() inmediatamente despues. Ningun otro
       codigo corre entre start() y stop().
  P2   La carga del modelo ocurre ANTES de abrir el tracker CodeCarbon.
  P3   Los sleeps de cooldown, warmup y pausas ocurren FUERA del tracker.
  P4   La evaluacion del juez Claude corre separada (judge_with_claude.py);
       no hay overlap con la medicion energetica.
  P5   Prompts MT-Bench usados verbatim — sin modificacion, suma ni recorte.
  P6   Prompts en ingles — sin traduccion.
  P7   Si el subset no existe el script aborta con error explicito.
       No se generan prompts de respaldo ni sinteticos.
  P8   Resultados a nivel de turno (turn_results.csv) y a nivel de
       conversacion (conversation_results.csv) en archivos separados.
  P9   Baseline idle medido antes de iniciar las inferencias del modelo.
  P10  CSV incluye energia medida (bruta) y energia corregida por baseline.
  P11  SHA256 parcial + tamano del archivo GGUF registrados en el CSV.
  P12  Context overflow detectado y registrado como status_turn='context_overflow'.
  P13  finish_reason registrado en cada fila del CSV.
  P14  turn_id deterministico: hash de (model, quant, qid, turn, rep).
       Permite reanudacion exacta sin duplicados.

Uso tipico (llamado por run_experiment.py):
  python scripts/fase1_benchmark.py \\
      --config config.yaml \\
      --model-name llama-2-7b \\
      --quantization q4 \\
      --hardware-profile mac_m4 \\
      --execution-device gpu \\
      --subset-file data/mt_bench/subset/mt_bench_literal_subset_5_per_category.yaml \\
      --repetitions 15 \\
      --output-dir results/raw \\
      --run-id phase1_mac_m4_gpu_20260626_081500

Uso directo (debugging de un modelo especifico):
  python scripts/fase1_benchmark.py \\
      --model-name qwen2.5-7b --quantization q8 \\
      --execution-device gpu --repetitions 2 --max-per-category 1
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML no instalado.  pip install pyyaml")
    sys.exit(1)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config, ExperimentConfig
from src.baseline_energy import (
    medir_baseline_calibracion,
    corregir_energia,
)
from src.prompt_builder import build_prompt

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    Llama = None          # type: ignore[misc,assignment]
    LLAMA_AVAILABLE = False

try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    EmissionsTracker = None   # type: ignore[misc,assignment]
    CODECARBON_AVAILABLE = False

# ─── Rutas canónicas ──────────────────────────────────────────────────────────

CONFIG_FILE  = ROOT / "config.yaml"
SUBSET_YAML  = (ROOT / "data" / "mt_bench" / "subset"
                / "mt_bench_literal_subset_5_per_category.yaml")
SUBSET_JSONL = SUBSET_YAML.with_suffix(".jsonl")

OFFICIAL_N_PER_CATEGORY = 5
OFFICIAL_N_REPETITIONS  = 15

_WH_TO_J      = 3600.0
_BACKUP_EVERY = 50          # conversaciones entre backups


# ─── Esquemas de columnas CSV ─────────────────────────────────────────────────

TURN_COLS: list[str] = [
    # identidad
    "turn_id", "conversation_id", "run_id", "phase",
    # hardware
    "hardware_profile", "execution_device",
    # modelo
    "model_name", "model_type", "quantization",
    "model_path", "model_size_bytes", "model_sha256_1mb",
    # pregunta
    "question_id", "original_category", "category",
    # posicion
    "turn_number", "repetition",
    # estado
    "status_turn", "finish_reason", "error_message",
    # tokens
    "prompt_tokens", "completion_tokens",
    # tiempo
    "inference_time_s",
    # energia medida (bruta)
    "measured_cpu_energy_wh", "measured_gpu_energy_wh",
    "measured_ram_energy_wh", "measured_energy_wh",
    "measured_energy_joules",
    # energia corregida por baseline
    "baseline_corrected_energy_wh", "baseline_corrected_energy_joules",
    "baseline_subtracted_wh", "baseline_clip_applied",
    "baseline_total_power_w",
    # metricas derivadas medidas
    "energy_j_per_output_token_measured",
    "energy_wh_per_1k_output_tokens_measured",
    "tokens_per_joule_measured",
    "edp_joule_second_measured",
    "operational_sci_per_turn_measured",
    # metricas derivadas corregidas
    "energy_j_per_output_token_corrected",
    "energy_wh_per_1k_output_tokens_corrected",
    "tokens_per_joule_corrected",
    "edp_joule_second_corrected",
    "operational_sci_per_turn_corrected",
    # parametros de inferencia
    "n_ctx", "max_tokens", "temperature", "seed_used",
    "timestamp_utc",
]

CONV_COLS: list[str] = [
    # identidad
    "conversation_id", "run_id", "phase",
    # hardware
    "hardware_profile", "execution_device",
    # modelo
    "model_name", "model_type", "quantization",
    # pregunta
    "question_id", "original_category", "category",
    "repetition",
    # estado
    "status_conversation",
    "turn1_finish_reason", "turn2_finish_reason",
    # tokens
    "total_prompt_tokens", "total_completion_tokens",
    # tiempo
    "total_latency_seconds",
    # energia total medida
    "total_measured_energy_wh", "total_measured_energy_joules",
    # energia total corregida
    "total_baseline_corrected_energy_wh", "total_baseline_corrected_energy_joules",
    "total_baseline_clip_applied", "baseline_total_power_w",
    # metricas derivadas medidas (nivel conversacion)
    "total_energy_wh_per_1k_output_tokens_measured",
    "total_energy_j_per_output_token_measured",
    "total_tokens_per_joule_measured",
    "total_edp_joule_second_measured",
    # metricas derivadas corregidas (nivel conversacion)
    "total_energy_wh_per_1k_output_tokens_corrected",
    "total_energy_j_per_output_token_corrected",
    "total_tokens_per_joule_corrected",
    "total_edp_joule_second_corrected",
    # SCI operacional
    "total_operational_sci_per_conversation",
    "total_operational_sci_per_1k_output_tokens",
    "timestamp_utc",
]


# ─── CSV writer append-mode ────────────────────────────────────────────────────

class _CsvWriter:
    """Escribe filas en modo append; inserta encabezado solo en la primera vez."""

    def __init__(self, path: Path, columns: list[str]) -> None:
        self._path    = path
        self._columns = columns
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with open(path, "w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=columns,
                               extrasaction="ignore").writeheader()

    def write(self, row: dict) -> None:
        with open(self._path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=self._columns,
                           extrasaction="ignore").writerow(row)

    def path(self) -> Path:
        return self._path


# ─── IDs deterministas (P14) ──────────────────────────────────────────────────

def _turn_id(model: str, quant: str, qid: int,
             turn: int, rep: int) -> str:
    """SHA256[:24] de los 5 campos que identifican univocamente un turno."""
    raw = f"{model}|{quant}|{qid}|{turn}|{rep}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _conv_id(model: str, quant: str, qid: int, rep: int) -> str:
    raw = f"{model}|{quant}|{qid}|{rep}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


# ─── Resume: turnos ya completados ───────────────────────────────────────────

def _load_completed_ids(turn_csv: Path) -> set[str]:
    """
    Carga turn_ids ya presentes en turn_results.csv (P14 resume).

    Solo considera filas con status_turn in ('success', 'context_overflow')
    — errores transitorios se pueden reintentar.
    """
    completed: set[str] = set()
    if not turn_csv.exists():
        return completed
    try:
        with open(turn_csv, "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                tid    = row.get("turn_id", "")
                status = row.get("status_turn", "")
                if tid and status in ("success", "context_overflow"):
                    completed.add(tid)
    except Exception as exc:
        print(f"  AVISO: no se pudo leer turn_results.csv para resume: {exc}")
    return completed


# ─── Huella del modelo GGUF (P11) ─────────────────────────────────────────────

def _model_fingerprint(path: Path) -> tuple[int, str]:
    """(size_bytes, sha256_1mb). Si el archivo no existe, retorna (0, 'not_found')."""
    if not path.exists():
        return 0, "not_found"
    size = path.stat().st_size
    h    = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(1_048_576))
    return size, h.hexdigest()


# ─── Extraccion de energia de CodeCarbon ──────────────────────────────────────

def _extract_energy(tracker: "EmissionsTracker") -> dict:
    """
    Extrae componentes de energia del tracker tras .stop().

    CodeCarbon 3.x expone _total_*_energy como objetos Energy con .kWh.
    Si el atributo no existe (cambio de API futuro), retorna 0.
    """
    def _kwh(attr: str) -> float:
        try:
            obj = getattr(tracker, attr, None)
            return float(obj.kWh) if (obj is not None and hasattr(obj, "kWh")) else 0.0
        except Exception:
            return 0.0

    cpu_kwh = _kwh("_total_cpu_energy")
    gpu_kwh = _kwh("_total_gpu_energy")
    ram_kwh = _kwh("_total_ram_energy")
    tot_kwh = _kwh("_total_energy")
    if tot_kwh == 0.0:
        tot_kwh = cpu_kwh + gpu_kwh + ram_kwh

    return {
        "cpu_energy_wh"  : cpu_kwh * 1000.0,
        "gpu_energy_wh"  : gpu_kwh * 1000.0,
        "ram_energy_wh"  : ram_kwh * 1000.0,
        "total_energy_wh": tot_kwh * 1000.0,
    }


# ─── Metricas derivadas por turno/conversacion ────────────────────────────────

def _derived(energy_wh: float, time_s: float,
             n_tokens: int, carbon_g_kwh: int) -> dict:
    """Calcula J/tok, Wh/1Ktok, tok/J, EDP, SCI para una energia dada."""
    j   = energy_wh * _WH_TO_J
    tok = max(0, n_tokens)
    return {
        "j_per_tok"  : j / tok            if tok > 0 and j > 0 else 0.0,
        "wh_per_1k"  : energy_wh / tok * 1000.0 if tok > 0 else 0.0,
        "tok_per_j"  : tok / j            if j > 0 else 0.0,
        "edp"        : j * time_s,
        "sci"        : (energy_wh / 1000.0) * carbon_g_kwh,
        "sci_1k"     : ((energy_wh / tok * 1000.0) / 1000.0 * carbon_g_kwh) if tok > 0 else 0.0,
    }


# ─── Deteccion de context overflow (P12) ──────────────────────────────────────

def _count_prompt_tokens(llm: "Llama", text: str) -> int:
    try:
        return len(llm.tokenize(text.encode("utf-8")))
    except Exception:
        return max(1, len(text) // 4)   # estimacion conservadora si falla


def _check_overflow(llm: "Llama", text: str, cfg: ExperimentConfig) -> tuple[bool, int]:
    """(overflow, n_prompt_tokens).  overflow si n_tokens > n_ctx - max_tokens."""
    n   = _count_prompt_tokens(llm, text)
    return n > (cfg.n_ctx - cfg.max_tokens), n


# ─── Inferencia con CodeCarbon (P1 zona critica) ──────────────────────────────

def _measure_inference(
    llm: "Llama",
    prompt: str,
    cfg: ExperimentConfig,
    country_iso: str,
    seed: int,
) -> dict:
    """
    Ejecuta una inferencia con medicion CodeCarbon.

    ZONA CRITICA P1:  solo tracker.start() — llm() — tracker.stop().
    Ningun otro codigo corre en este bloque.
    """
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
    tracker.start()                      # ── P1 inicio ──
    t0 = time.perf_counter()
    out = llm(
        prompt,
        max_tokens  = cfg.max_tokens,
        temperature = cfg.temperature,
        top_p       = cfg.top_p,
        seed        = seed,
        echo        = cfg.echo,
        stop        = None,
    )
    t1 = time.perf_counter()
    tracker.stop()                       # ── P1 fin ──

    # Limpiar KV-cache entre inferencias
    # Elimina contaminacion entre repeticiones
    # Validado empiricamente: reduce CV de 33% a 0.31%
    try:
        llm.reset()
    except Exception:
        pass

    choice = out["choices"][0]
    usage  = out.get("usage", {})
    return {
        "response_text"    : choice["text"],
        "finish_reason"    : choice.get("finish_reason") or "unknown",
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "prompt_tokens"    : int(usage.get("prompt_tokens", 0)),
        "inference_time_s" : t1 - t0,
        **_extract_energy(tracker),
    }


# ─── Medicion de un turno completo ────────────────────────────────────────────

def _run_turn(
    *,
    llm: "Llama",
    prompt_text: str,
    cfg: ExperimentConfig,
    baseline: dict,
    carbon_g_kwh: int,
    country_iso: str,
    seed: int,
    # metadatos
    model_name: str,
    model_type: str,
    quantization: str,
    model_path_str: str,
    model_size_bytes: int,
    model_sha256_1mb: str,
    question_id: int,
    original_category: str,
    category: str,
    turn_number: int,
    repetition: int,
    run_id: str,
    hardware_profile: str,
    execution_device: str,
    phase: int,
) -> dict:
    """
    Mide un turno completo.  Retorna una fila completa para turn_results.csv.

    Maneja context overflow (P12), registra finish_reason (P13),
    y usa turn_id deterministico (P14).
    """
    tid = _turn_id(model_name, quantization, question_id, turn_number, repetition)
    cid = _conv_id(model_name, quantization, question_id, repetition)
    now = datetime.now(timezone.utc).isoformat()

    base: dict = {
        "turn_id"          : tid,
        "conversation_id"  : cid,
        "run_id"           : run_id,
        "phase"            : phase,
        "hardware_profile" : hardware_profile,
        "execution_device" : execution_device,
        "model_name"       : model_name,
        "model_type"       : model_type,
        "quantization"     : quantization,
        "model_path"       : model_path_str,
        "model_size_bytes" : model_size_bytes,
        "model_sha256_1mb" : model_sha256_1mb,
        "question_id"      : question_id,
        "original_category": original_category,
        "category"         : category,
        "turn_number"      : turn_number,
        "repetition"       : repetition,
        "n_ctx"            : cfg.n_ctx,
        "max_tokens"       : cfg.max_tokens,
        "temperature"      : cfg.temperature,
        "seed_used"        : seed,
        "timestamp_utc"    : now,
        "baseline_total_power_w": baseline.get("baseline_total_power_w", 0.0),
    }

    # ── P12: overflow check ─────────────────────────────────────────────────
    overflow, n_prompt = _check_overflow(llm, prompt_text, cfg)
    if overflow:
        msg = (f"context_overflow: {n_prompt} prompt tokens > "
               f"{cfg.n_ctx - cfg.max_tokens} available "
               f"(n_ctx={cfg.n_ctx} max_tokens={cfg.max_tokens})")
        return {
            **base,
            "status_turn"    : "context_overflow",
            "finish_reason"  : "context_overflow",
            "error_message"  : msg,
            "prompt_tokens"  : n_prompt,
            "completion_tokens"                  : 0,
            "inference_time_s"                   : None,
            "measured_cpu_energy_wh"             : None,
            "measured_gpu_energy_wh"             : None,
            "measured_ram_energy_wh"             : None,
            "measured_energy_wh"                 : None,
            "measured_energy_joules"             : None,
            "baseline_corrected_energy_wh"       : None,
            "baseline_corrected_energy_joules"   : None,
            "baseline_subtracted_wh"             : None,
            "baseline_clip_applied"              : None,
            "energy_j_per_output_token_measured"         : None,
            "energy_wh_per_1k_output_tokens_measured"    : None,
            "tokens_per_joule_measured"                  : None,
            "edp_joule_second_measured"                  : None,
            "operational_sci_per_turn_measured"          : None,
            "energy_j_per_output_token_corrected"        : None,
            "energy_wh_per_1k_output_tokens_corrected"   : None,
            "tokens_per_joule_corrected"                 : None,
            "edp_joule_second_corrected"                 : None,
            "operational_sci_per_turn_corrected"         : None,
        }

    # ── inferencia ─────────────────────────────────────────────────────────
    try:
        res = _measure_inference(llm, prompt_text, cfg, country_iso, seed)
    except Exception as exc:
        return {
            **base,
            "status_turn"  : "error",
            "finish_reason": "error",
            "error_message": str(exc)[:500],
            "prompt_tokens": n_prompt,
            "completion_tokens": 0,
            "inference_time_s": None,
            "measured_cpu_energy_wh": None,
            "measured_gpu_energy_wh": None,
            "measured_ram_energy_wh": None,
            "measured_energy_wh": None,
            "measured_energy_joules": None,
            "baseline_corrected_energy_wh": None,
            "baseline_corrected_energy_joules": None,
            "baseline_subtracted_wh": None,
            "baseline_clip_applied": None,
            "energy_j_per_output_token_measured": None,
            "energy_wh_per_1k_output_tokens_measured": None,
            "tokens_per_joule_measured": None,
            "edp_joule_second_measured": None,
            "operational_sci_per_turn_measured": None,
            "energy_j_per_output_token_corrected": None,
            "energy_wh_per_1k_output_tokens_corrected": None,
            "tokens_per_joule_corrected": None,
            "edp_joule_second_corrected": None,
            "operational_sci_per_turn_corrected": None,
        }

    # ── P10: correccion baseline ────────────────────────────────────────────
    raw_meas = {
        "cpu_energy_wh"   : res["cpu_energy_wh"],
        "gpu_energy_wh"   : res["gpu_energy_wh"],
        "ram_energy_wh"   : res["ram_energy_wh"],
        "total_energy_wh" : res["total_energy_wh"],
        "inference_time_s": res["inference_time_s"],
    }
    corr = corregir_energia(raw_meas, baseline)

    gross_wh = res["total_energy_wh"]
    net_wh   = corr["net_total_energy_wh"]
    t_s      = res["inference_time_s"]
    n_out    = res["completion_tokens"]

    m = _derived(gross_wh, t_s, n_out, carbon_g_kwh)
    c = _derived(net_wh,   t_s, n_out, carbon_g_kwh)

    row = {
        **base,
        "status_turn"      : "success",
        "finish_reason"    : res["finish_reason"],   # P13
        "error_message"    : None,
        "prompt_tokens"    : res["prompt_tokens"] or n_prompt,
        "completion_tokens": n_out,
        "inference_time_s" : t_s,
        # medida
        "measured_cpu_energy_wh" : res["cpu_energy_wh"],
        "measured_gpu_energy_wh" : res["gpu_energy_wh"],
        "measured_ram_energy_wh" : res["ram_energy_wh"],
        "measured_energy_wh"     : gross_wh,
        "measured_energy_joules" : gross_wh * _WH_TO_J,
        # corregida
        "baseline_corrected_energy_wh"     : net_wh,
        "baseline_corrected_energy_joules" : net_wh * _WH_TO_J,
        "baseline_subtracted_wh"           : corr["baseline_subtracted_wh"],
        "baseline_clip_applied"            : corr["baseline_clip_applied"],
        # derivadas medidas
        "energy_j_per_output_token_measured"       : m["j_per_tok"],
        "energy_wh_per_1k_output_tokens_measured"  : m["wh_per_1k"],
        "tokens_per_joule_measured"                : m["tok_per_j"],
        "edp_joule_second_measured"                : m["edp"],
        "operational_sci_per_turn_measured"        : m["sci"],
        # derivadas corregidas
        "energy_j_per_output_token_corrected"      : c["j_per_tok"],
        "energy_wh_per_1k_output_tokens_corrected" : c["wh_per_1k"],
        "tokens_per_joule_corrected"               : c["tok_per_j"],
        "edp_joule_second_corrected"               : c["edp"],
        "operational_sci_per_turn_corrected"       : c["sci"],
    }
    # Adjuntar texto de respuesta para construir prompt del turno 2
    row["_response"] = res["response_text"]
    return row


# ─── Agregacion de conversacion (P8) ─────────────────────────────────────────

def _aggregate_conversation(
    t1: dict,
    t2: Optional[dict],
    run_id: str,
    phase: int,
    carbon_g_kwh: int,
) -> dict:
    """Agrega dos filas de turno en una fila de conversacion (P8)."""
    now = datetime.now(timezone.utc).isoformat()

    def _f(row: Optional[dict], key: str) -> Optional[float]:
        if row is None:
            return None
        v = row.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def _sum2(key: str) -> Optional[float]:
        a = _f(t1, key)
        b = _f(t2, key)
        if a is None and b is None:
            return None
        return (a or 0.0) + (b or 0.0)

    t1_ok = t1.get("status_turn") == "success"
    t2_ok = t2 is not None and t2.get("status_turn") == "success"
    if t1_ok and t2_ok:
        status = "success"
    elif t1_ok or t2_ok:
        status = "partial"
    else:
        status = "failed"

    total_comp = (_f(t1, "completion_tokens") or 0) + (_f(t2, "completion_tokens") or 0)
    total_prom = (_f(t1, "prompt_tokens")     or 0) + (_f(t2, "prompt_tokens")     or 0)
    total_lat  = _sum2("inference_time_s")
    total_m_wh = _sum2("measured_energy_wh")
    total_c_wh = _sum2("baseline_corrected_energy_wh")
    total_m_j  = (total_m_wh * _WH_TO_J) if total_m_wh is not None else None
    total_c_j  = (total_c_wh * _WH_TO_J) if total_c_wh is not None else None

    clip1 = int(_f(t1, "baseline_clip_applied") or 0)
    clip2 = int(_f(t2, "baseline_clip_applied") or 0) if t2 else 0

    n_out = int(total_comp)
    lat   = total_lat or 0.0

    def _cm(wh: Optional[float], j_val: Optional[float]) -> dict:
        if wh is None:
            return dict(wh1k=None, j_tok=None, tok_j=None,
                        edp=None, sci_c=None, sci_1k=None)
        jv  = j_val or wh * _WH_TO_J
        return {
            "wh1k" : (wh / n_out * 1000.0)    if n_out > 0 else None,
            "j_tok": (jv / n_out)              if n_out > 0 else None,
            "tok_j": (n_out / jv)              if jv > 0 else None,
            "edp"  : jv * lat,
            "sci_c": (wh / 1000.0) * carbon_g_kwh,
            "sci_1k": ((wh / n_out * 1000.0) / 1000.0 * carbon_g_kwh) if n_out > 0 else None,
        }

    m_cm = _cm(total_m_wh, total_m_j)
    c_cm = _cm(total_c_wh, total_c_j)

    return {
        "conversation_id"   : t1["conversation_id"],
        "run_id"            : run_id,
        "phase"             : phase,
        "hardware_profile"  : t1["hardware_profile"],
        "execution_device"  : t1["execution_device"],
        "model_name"        : t1["model_name"],
        "model_type"        : t1["model_type"],
        "quantization"      : t1["quantization"],
        "question_id"       : t1["question_id"],
        "original_category" : t1["original_category"],
        "category"          : t1["category"],
        "repetition"        : t1["repetition"],
        "status_conversation"           : status,
        "turn1_finish_reason"           : t1.get("finish_reason"),
        "turn2_finish_reason"           : (t2 or {}).get("finish_reason"),
        "total_prompt_tokens"           : int(total_prom),
        "total_completion_tokens"       : n_out,
        "total_latency_seconds"         : total_lat,
        "total_measured_energy_wh"      : total_m_wh,
        "total_measured_energy_joules"  : total_m_j,
        "total_baseline_corrected_energy_wh"    : total_c_wh,
        "total_baseline_corrected_energy_joules": total_c_j,
        "total_baseline_clip_applied"           : max(clip1, clip2),
        "baseline_total_power_w"                : t1.get("baseline_total_power_w"),
        "total_energy_wh_per_1k_output_tokens_measured"  : m_cm["wh1k"],
        "total_energy_j_per_output_token_measured"       : m_cm["j_tok"],
        "total_tokens_per_joule_measured"                : m_cm["tok_j"],
        "total_edp_joule_second_measured"                : m_cm["edp"],
        "total_energy_wh_per_1k_output_tokens_corrected" : c_cm["wh1k"],
        "total_energy_j_per_output_token_corrected"      : c_cm["j_tok"],
        "total_tokens_per_joule_corrected"               : c_cm["tok_j"],
        "total_edp_joule_second_corrected"               : c_cm["edp"],
        "total_operational_sci_per_conversation"         : c_cm["sci_c"],
        "total_operational_sci_per_1k_output_tokens"     : c_cm["sci_1k"],
        "timestamp_utc"                                  : now,
    }


# ─── Gestion del modelo ───────────────────────────────────────────────────────

def _load_model(model_path: Path, cfg: ExperimentConfig) -> "Llama":
    if not LLAMA_AVAILABLE:
        raise RuntimeError(
            "llama-cpp-python no instalado.  Ver README.md → Instalacion."
        )
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    n_gpu   = cfg.n_gpu_layers()
    n_thrd  = cfg.n_threads()
    n_batch = cfg.n_batch()

    print(f"\n  Cargando modelo: {model_path.name}")
    print(f"    n_gpu_layers={n_gpu}  n_threads={n_thrd}  "
          f"n_batch={n_batch}  n_ctx={cfg.n_ctx}")

    t0  = time.perf_counter()
    llm = Llama(
        model_path   = str(model_path),
        n_ctx        = cfg.n_ctx,
        n_gpu_layers = n_gpu,
        n_threads    = n_thrd,
        n_batch      = n_batch,
        verbose      = False,
    )
    elapsed = time.perf_counter() - t0
    print(f"    Modelo cargado en {elapsed:.1f}s")
    return llm


def _unload_model(llm: Optional["Llama"]) -> None:
    """Descarga el modelo y fuerza recoleccion de basura (P3)."""
    if llm is not None:
        try:
            del llm
        except Exception:
            pass
    gc.collect()
    time.sleep(2)


# ─── Warmup (P3) ──────────────────────────────────────────────────────────────

def _run_warmup(
    llm: "Llama",
    questions: list[dict],
    cfg: ExperimentConfig,
    model_name: str,
    n_warmup: int,
) -> None:
    """
    Ejecuta n_warmup inferencias de warmup para estabilizar caches GPU/CPU.

    Corre FUERA del tracker CodeCarbon (P3).
    No se registran los resultados.
    """
    if n_warmup < 1 or not questions:
        return
    print(f"  Warmup ({n_warmup} inferencia(s))...", end=" ", flush=True)
    for i in range(n_warmup):
        q = questions[i % len(questions)]
        try:
            msgs   = [{"role": "user", "content": q.get("turns", ["Hello."])[0]}]
            prompt = build_prompt(model_name, msgs)
            llm(prompt, max_tokens=32, temperature=0.0, seed=cfg.seed,
                echo=False, stop=None)
        except Exception:
            pass
    print("ok")


# ─── Carga del subset MT-Bench (P7) ───────────────────────────────────────────

def _load_subset(subset_file: Optional[Path],
                 max_per_category: int) -> list[dict]:
    """
    Carga el subset desde la ruta especificada (YAML o JSONL).

    Si subset_file es None, busca las rutas canonicas automaticamente.
    Aborta con sys.exit(1) si no encuentra el archivo (P7).
    """
    if subset_file is not None:
        candidates = [subset_file]
    else:
        candidates = [SUBSET_YAML, SUBSET_JSONL]

    path: Optional[Path] = None
    for c in candidates:
        if c.exists():
            path = c
            break

    if path is None:
        print(
            "\n  ERROR [P7]: subset MT-Bench no encontrado.\n"
            "  Ejecutar: python scripts/prepare_mt_bench_subset.py\n"
            f"  Ruta buscada: {candidates[0]}"
        )
        sys.exit(1)

    questions: list[dict] = []
    try:
        if path.suffix in (".yaml", ".yml"):
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            questions = data.get("questions") or []
        else:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        questions.append(json.loads(line))
    except Exception as exc:
        print(f"\n  ERROR al leer subset: {exc}")
        sys.exit(1)

    if not questions:
        print(f"\n  ERROR: subset vacio: {path}")
        sys.exit(1)

    # Aplicar limite de preguntas por categoria
    if max_per_category < OFFICIAL_N_PER_CATEGORY:
        counts: dict[str, int] = defaultdict(int)
        filtered: list[dict]   = []
        for q in questions:
            cat = q.get("original_category", q.get("category", "unknown"))
            if counts[cat] < max_per_category:
                filtered.append(q)
                counts[cat] += 1
        questions = filtered

    n_cats = len({q.get("original_category", "") for q in questions})
    rel = path
    try:
        rel = path.relative_to(ROOT)
    except ValueError:
        pass
    print(f"  Subset: {len(questions)} preguntas / {n_cats} categorias "
          f"(max {max_per_category}/cat)  [{rel.name}]")
    return questions


# ─── Pares (pregunta, repeticion) ────────────────────────────────────────────

def _build_pairs(
    questions: list[dict],
    repetitions: int,
    randomize: bool,
    seed: int,
) -> list[tuple[dict, int]]:
    """Genera todos los pares (pregunta, repeticion) con orden opcional."""
    pairs = [(q, r) for q in questions for r in range(1, repetitions + 1)]
    if randomize:
        random.Random(seed).shuffle(pairs)
    return pairs


# ─── Backup periodico ─────────────────────────────────────────────────────────

def _backup(path: Path, n: int) -> None:
    """Copia el CSV a results/backups/<stem>_backup_<n>.<ext>."""
    if not path.exists():
        return
    backup_dir = ROOT / "results" / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    dst = backup_dir / f"{path.stem}_backup_{n:04d}{path.suffix}"
    try:
        import shutil
        shutil.copy2(str(path), str(dst))
    except Exception:
        pass


# ─── Helpers de config ────────────────────────────────────────────────────────

def _load_raw(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _country_iso(raw: dict) -> str:
    return str((raw.get("codecarbon") or {}).get("country_iso_code", "PRY"))


def _carbon_intensity(raw: dict, hw_profile: str) -> int:
    hw = (raw.get("hardware_profiles") or {}).get(hw_profile) or {}
    return int(hw.get("carbon_intensity_g_kwh", 26))


# ─── Interrupt handler ────────────────────────────────────────────────────────

_interrupted = False


def _handle_sigint(sig: int, frame: object) -> None:
    global _interrupted
    _interrupted = True
    print("\n\n  Interrupcion recibida — finalizando la conversacion actual...\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GREEN-IA Fase 1 — benchmark de energia y calidad",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", type=Path, default=CONFIG_FILE,
        metavar="PATH",
        help="Ruta a config.yaml",
    )
    p.add_argument(
        "--model-name", type=str, default=None,
        metavar="NAME",
        help="Nombre del modelo (ej. llama-2-7b). "
             "Si se omite, itera sobre selected_models en config.yaml.",
    )
    p.add_argument(
        "--quantization", type=str, default=None,
        metavar="Q",
        help="Cuantizacion (ej. q4, q8). "
             "Si se omite, itera sobre selected_quantizations en config.yaml.",
    )
    p.add_argument(
        "--hardware-profile", type=str, default=None,
        choices=["mac_m4", "windows_nvidia"],
        metavar="PROFILE",
        help="Override de hardware_profile en config.yaml.",
    )
    p.add_argument(
        "--execution-device", type=str, default=None,
        choices=["cpu", "gpu"],
        metavar="DEVICE",
        help="Override de execution_device en config.yaml.",
    )
    p.add_argument(
        "--subset-file", type=Path, default=None,
        metavar="PATH",
        help="Ruta al archivo de subset MT-Bench (.yaml o .jsonl). "
             "Auto-detectado si se omite.",
    )
    p.add_argument(
        "--repetitions", type=int, default=None,
        metavar="N",
        help=f"Repeticiones por pregunta (default de config.yaml o {OFFICIAL_N_REPETITIONS}).",
    )
    p.add_argument(
        "--max-per-category", type=int, default=OFFICIAL_N_PER_CATEGORY,
        metavar="N",
        help=f"Maximo de preguntas por categoria (default={OFFICIAL_N_PER_CATEGORY}).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "raw",
        metavar="PATH",
        help="Directorio de salida para turn_results.csv y conversation_results.csv.",
    )
    p.add_argument(
        "--run-id", type=str, default=None,
        metavar="ID",
        help="Identificador del run. Si se omite, lee GREEN_IA_RUN_ID del entorno "
             "o genera uno automaticamente.",
    )
    p.add_argument(
        "--smoke-test", action="store_true",
        help="Modo smoke test (mensajes adicionales; no afecta la logica de medicion).",
    )
    return p.parse_args()


# ─── Estadisticas de progreso ─────────────────────────────────────────────────

@dataclass
class _Stats:
    total:    int   = 0
    done:     int   = 0
    skipped:  int   = 0
    ok:       int   = 0
    partial:  int   = 0
    failed:   int   = 0
    overflow: int   = 0
    t_start:  float = field(default_factory=time.perf_counter)

    def eta_str(self) -> str:
        elapsed = time.perf_counter() - self.t_start
        if self.done == 0:
            return "—"
        per_conv = elapsed / self.done
        remaining = (self.total - self.done) * per_conv
        m, s = divmod(int(remaining), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h{m:02d}m"
        return f"{m}m{s:02d}s"

    def rate_str(self) -> str:
        elapsed = time.perf_counter() - self.t_start
        if elapsed < 1:
            return "—"
        return f"{self.done / elapsed * 60:.1f} conv/min"


def _print_progress(stats: _Stats, config_label: str,
                    qid: int, rep: int,
                    t1_status: str, t2_status: str) -> None:
    pct = stats.done / max(stats.total, 1) * 100
    print(
        f"    [{config_label}]  {stats.done}/{stats.total} ({pct:.0f}%)"
        f"  Q{qid} rep{rep}"
        f"  T1={t1_status}  T2={t2_status}"
        f"  ETA={stats.eta_str()}  {stats.rate_str()}"
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    signal.signal(signal.SIGINT, _handle_sigint)

    args = _parse_args()

    # ── dependencias ─────────────────────────────────────────────────────────
    if not LLAMA_AVAILABLE:
        print("  ERROR: llama-cpp-python no instalado.  Ver README.md → Instalacion.")
        sys.exit(1)
    if not CODECARBON_AVAILABLE:
        print("  ERROR: codecarbon no instalado.  pip install codecarbon")
        sys.exit(1)

    # ── run_id ────────────────────────────────────────────────────────────────
    run_id = (
        args.run_id
        or os.environ.get("GREEN_IA_RUN_ID")
        or f"phase1_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    run_type = os.environ.get("GREEN_IA_RUN_TYPE", "official")
    phase    = 1

    # ── config ────────────────────────────────────────────────────────────────
    if not args.config.exists():
        print(f"\n  ERROR: config.yaml no encontrado: {args.config}")
        print("  Crear desde: cp config.yaml.example config.yaml")
        sys.exit(1)

    raw_cfg = _load_raw(args.config)
    cfg     = load_config(args.config)

    # Aplicar overrides de CLI al objeto de config
    import dataclasses as _dc
    if args.hardware_profile:
        cfg = _dc.replace(cfg, hardware_profile=args.hardware_profile)
        raw_cfg["hardware_profile"] = args.hardware_profile
    if args.execution_device:
        cfg = _dc.replace(cfg, execution_device=args.execution_device)
        raw_cfg["execution_device"] = args.execution_device

    hardware_profile = cfg.hardware_profile
    execution_device = cfg.execution_device
    country_iso      = _country_iso(raw_cfg)
    carbon_g_kwh     = _carbon_intensity(raw_cfg, hardware_profile)
    randomize        = bool(raw_cfg.get("randomize_order", True))

    repetitions = (
        args.repetitions
        or int(raw_cfg.get("repetitions", OFFICIAL_N_REPETITIONS))
    )

    # ── modelos a medir ──────────────────────────────────────────────────────
    models_block    = raw_cfg.get("models") or {}
    selected_models = (
        [args.model_name] if args.model_name
        else list(raw_cfg.get("selected_models") or [])
    )
    selected_quants = (
        [args.quantization] if args.quantization
        else list(raw_cfg.get("selected_quantizations") or [])
    )

    if not selected_models:
        print("  ERROR: sin modelos seleccionados.")
        sys.exit(1)
    if not selected_quants:
        print("  ERROR: sin cuantizaciones seleccionadas.")
        sys.exit(1)

    # ── output ────────────────────────────────────────────────────────────────
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    turn_csv = output_dir / "turn_results.csv"
    conv_csv = output_dir / "conversation_results.csv"

    # ── subset ────────────────────────────────────────────────────────────────
    questions = _load_subset(args.subset_file, args.max_per_category)

    # ── header ────────────────────────────────────────────────────────────────
    W = 66
    smoke_tag = "[SMOKE TEST] " if args.smoke_test else ""
    print(f"\n{'=' * W}")
    print(f"  GREEN-IA — {smoke_tag}Fase 1: medicion de energia y calidad")
    print(f"  Run ID     : {run_id}")
    print(f"  Tipo       : {run_type}")
    print(f"  Hardware   : {hardware_profile}  /  {execution_device}")
    print(f"  Carbon     : {carbon_g_kwh} gCO2eq/kWh  (country={country_iso})")
    print(f"  Modelos    : {selected_models}")
    print(f"  Cuant.     : {selected_quants}")
    print(f"  Reps/preg  : {repetitions}")
    print(f"  Max/cat    : {args.max_per_category}")
    print(f"  Preguntas  : {len(questions)}")
    print(f"  Randomize  : {randomize}")
    print(f"  Output     : {output_dir}")
    if args.smoke_test:
        print(f"  AVISO: resultados en {output_dir} — son datos de prueba")
    print(f"{'=' * W}")

    # ── resume ────────────────────────────────────────────────────────────────
    completed_ids = _load_completed_ids(turn_csv)
    if completed_ids:
        print(f"\n  Resume: {len(completed_ids)} turno(s) ya completados — se omitiran")

    # ── CSV writers ───────────────────────────────────────────────────────────
    turn_writer = _CsvWriter(turn_csv, TURN_COLS)
    conv_writer = _CsvWriter(conv_csv, CONV_COLS)

    # ── pausa pre-experimento (P3) ────────────────────────────────────────────
    pause_pre = cfg.pause_before_start_seconds
    if pause_pre > 0:
        print(f"\n  Pausa pre-experimento: {pause_pre}s...")
        time.sleep(pause_pre)

    # ─────────────────────────────────────────────────────────────────────────
    # Bucle externo: modelo × cuantizacion
    # ─────────────────────────────────────────────────────────────────────────
    n_configs = len(selected_models) * len(selected_quants)
    cfg_idx   = 0
    global_stats = _Stats()

    for model_name in selected_models:
        if _interrupted:
            break
        model_def  = models_block.get(model_name) or {}
        model_type = str(model_def.get("model_type", "unknown"))

        for quant in selected_quants:
            if _interrupted:
                break
            cfg_idx += 1
            path_str = str(model_def.get(quant, "") or "")
            if not path_str:
                print(f"\n  AVISO: ruta no definida para {model_name}/{quant} "
                      f"en config.yaml — omitiendo.")
                continue

            model_path   = ROOT / path_str
            config_label = f"{model_name}/{quant}"

            # huella del modelo (P11)
            model_size, model_sha256 = _model_fingerprint(model_path)
            model_path_str = str(model_path.relative_to(ROOT)
                                 if model_path.is_relative_to(ROOT)
                                 else model_path)

            print(f"\n{'─' * W}")
            print(f"  Configuracion [{cfg_idx}/{n_configs}]: {config_label} "
                  f"({execution_device.upper()})")
            if model_size:
                print(f"  GGUF: {model_size / 1e9:.2f} GB  "
                      f"sha256_1mb={model_sha256[:20]}...")

            # ── P9: baseline calibracion ───────────────────────────────────
            if cfg.energy_calibration_enabled:
                print(f"\n  Calibracion baseline "
                      f"({cfg.baseline_repetitions} × {cfg.baseline_idle_seconds}s):")
                baseline = medir_baseline_calibracion(
                    duracion_s    = cfg.baseline_idle_seconds,
                    repetitions   = cfg.baseline_repetitions,
                    config_label  = config_label,
                    save_to_csv   = cfg.save_baseline_runs,
                )
            else:
                print("  Calibracion baseline: desactivada")
                baseline = {
                    "baseline_total_power_w"  : 0.0,
                    "baseline_cpu_power_w"    : 0.0,
                    "baseline_gpu_power_w"    : 0.0,
                    "baseline_ram_power_w"    : 0.0,
                    "baseline_total_energy_wh": 0.0,
                    "baseline_duration_s"     : 0,
                    "baseline_repetitions"    : 0,
                }

            # ── P2: cargar modelo ANTES del tracker ────────────────────────
            try:
                llm = _load_model(model_path, cfg)
            except Exception as exc:
                print(f"  ERROR al cargar modelo {config_label}: {exc}")
                continue

            # ── pausa post-carga (P3) ──────────────────────────────────────
            if cfg.pause_after_model_load_seconds > 0:
                print(f"  Pausa post-carga: {cfg.pause_after_model_load_seconds}s")
                time.sleep(cfg.pause_after_model_load_seconds)

            # ── warmup (P3) ────────────────────────────────────────────────
            if cfg.warmup_runs_per_configuration > 0 and cfg.discard_warmup_runs:
                _run_warmup(llm, questions, cfg, model_name,
                            cfg.warmup_runs_per_configuration)

            # ── pares (pregunta, repeticion) ───────────────────────────────
            pairs  = _build_pairs(questions, repetitions, randomize, cfg.seed)
            stats  = _Stats(total=len(pairs))
            n_back = 0

            print(f"\n  Iniciando medicion: {len(pairs)} conversaciones")

            # ── bucle interno ──────────────────────────────────────────────
            for q, rep in pairs:
                if _interrupted:
                    break

                qid      = int(q.get("question_id", 0))
                orig_cat = str(q.get("original_category",
                                     q.get("category", "")))
                cat      = str(q.get("category", orig_cat))
                seed_used = cfg.effective_seed(rep)

                tid1 = _turn_id(model_name, quant, qid, 1, rep)
                tid2 = _turn_id(model_name, quant, qid, 2, rep)

                both_done = (tid1 in completed_ids) and (tid2 in completed_ids)
                if both_done:
                    stats.done    += 1
                    stats.skipped += 1
                    continue

                # cooldown entre runs (P3)
                if stats.done > 0:
                    time.sleep(cfg.cooldown_seconds_between_runs)

                # ── argumentos comunes para _run_turn ─────────────────────
                common: dict = dict(
                    cfg              = cfg,
                    baseline         = baseline,
                    carbon_g_kwh     = carbon_g_kwh,
                    country_iso      = country_iso,
                    seed             = seed_used,
                    model_name       = model_name,
                    model_type       = model_type,
                    quantization     = quant,
                    model_path_str   = model_path_str,
                    model_size_bytes = model_size,
                    model_sha256_1mb = model_sha256,
                    question_id      = qid,
                    original_category= orig_cat,
                    category         = cat,
                    repetition       = rep,
                    run_id           = run_id,
                    hardware_profile = hardware_profile,
                    execution_device = execution_device,
                    phase            = phase,
                )

                # ── turno 1 ───────────────────────────────────────────────
                if tid1 not in completed_ids:
                    msgs_t1  = [{"role": "user", "content": q["turns"][0]}]
                    prompt_t1 = build_prompt(model_name, msgs_t1)
                    t1_row    = _run_turn(llm=llm, prompt_text=prompt_t1,
                                          turn_number=1, **common)
                    turn_writer.write(t1_row)
                    completed_ids.add(tid1)
                    if t1_row.get("status_turn") == "context_overflow":
                        stats.overflow += 1
                else:
                    # Turno 1 ya estaba en el CSV; construir fila minima para T2
                    t1_row = {
                        "conversation_id" : _conv_id(model_name, quant, qid, rep),
                        "status_turn"     : "resumed",
                        "finish_reason"   : "resumed",
                        "prompt_tokens"   : 0,
                        "completion_tokens": 0,
                        "inference_time_s": None,
                        "measured_energy_wh": None,
                        "baseline_corrected_energy_wh": None,
                        "baseline_subtracted_wh": None,
                        "baseline_clip_applied": None,
                        "baseline_total_power_w": baseline.get("baseline_total_power_w", 0.0),
                        "hardware_profile" : hardware_profile,
                        "execution_device" : execution_device,
                        "model_name"       : model_name,
                        "model_type"       : model_type,
                        "quantization"     : quant,
                        "question_id"      : qid,
                        "original_category": orig_cat,
                        "category"         : cat,
                        "repetition"       : rep,
                        "_response"        : "",
                    }

                # ── turno 2 ───────────────────────────────────────────────
                if tid2 not in completed_ids:
                    t1_resp = t1_row.get("_response", "")
                    t1_ok   = t1_row.get("status_turn") in ("success", "resumed")

                    if t1_ok and t1_resp:
                        msgs_t2 = [
                            {"role": "user",      "content": q["turns"][0]},
                            {"role": "assistant", "content": str(t1_resp)},
                            {"role": "user",      "content": q["turns"][1]},
                        ]
                    else:
                        # T1 fallo: turno 2 sin contexto
                        msgs_t2 = [{"role": "user", "content": q["turns"][1]}]

                    prompt_t2 = build_prompt(model_name, msgs_t2)
                    t2_row    = _run_turn(llm=llm, prompt_text=prompt_t2,
                                          turn_number=2, **common)
                    turn_writer.write(t2_row)
                    completed_ids.add(tid2)
                    if t2_row.get("status_turn") == "context_overflow":
                        stats.overflow += 1
                else:
                    t2_row = None

                # ── conversacion agregada ─────────────────────────────────
                conv_row = _aggregate_conversation(
                    t1_row, t2_row, run_id, phase, carbon_g_kwh
                )
                conv_writer.write(conv_row)

                # actualizar estadisticas
                cstatus = conv_row.get("status_conversation", "failed")
                if cstatus == "success":
                    stats.ok += 1
                elif cstatus == "partial":
                    stats.partial += 1
                else:
                    stats.failed += 1
                stats.done += 1
                global_stats.done += 1

                # progreso cada 5 conversaciones o en la ultima
                if stats.done % 5 == 0 or stats.done == stats.total:
                    _print_progress(stats, config_label,
                                    qid, rep,
                                    t1_row.get("status_turn", "?"),
                                    (t2_row or {}).get("status_turn", "skipped"))

                # ── backup periodico ──────────────────────────────────────
                if stats.done % _BACKUP_EVERY == 0:
                    n_back += 1
                    _backup(turn_csv, n_back)
                    _backup(conv_csv, n_back)
                    print(f"    Backup #{n_back} guardado "
                          f"(cada {_BACKUP_EVERY} conversaciones)")

            # ── resumen del modelo ─────────────────────────────────────────
            print(f"\n  [{config_label}] completado:")
            print(f"    Total  : {stats.done}  "
                  f"OK={stats.ok}  partial={stats.partial}  "
                  f"failed={stats.failed}  overflow={stats.overflow}  "
                  f"skipped={stats.skipped}")

            # ── descargar modelo ───────────────────────────────────────────
            _unload_model(llm)
            print(f"  Modelo descargado: {model_path.name}")

            # ── cooldown entre configuraciones (P3) ───────────────────────
            if cfg_idx < n_configs and not _interrupted:
                cd = cfg.cooldown_seconds_between_model_configs
                print(f"  Cooldown entre configuraciones: {cd}s")
                time.sleep(cd)

    # ── pausa post-experimento (P3) ───────────────────────────────────────────
    if cfg.pause_after_experiment_seconds > 0 and not _interrupted:
        time.sleep(cfg.pause_after_experiment_seconds)

    # ── resumen final ─────────────────────────────────────────────────────────
    if _interrupted:
        print(f"\n  Ejecucion interrumpida por el usuario.")
    print(f"\n{'=' * W}")
    print(f"  {smoke_tag}Fase 1 finalizada.")
    print(f"  turn_results.csv        : {turn_csv}")
    print(f"  conversation_results.csv: {conv_csv}")
    print(f"{'=' * W}\n")

    if _interrupted:
        sys.exit(130)


if __name__ == "__main__":
    main()

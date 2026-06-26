"""
optimize_winner.py
GREEN-IA — Maestria Ciencia de Datos
Universidad Comunero — Paraguay
Autor: Andres Villamayor

FASE 2: Optimizacion de parametros de inferencia del modelo ganador.

Selecciona el ganador de Fase 1 y busca parametros de inferencia optimos
(n_ctx, max_tokens, n_batch, n_threads, temperature, top_p, repeat_penalty,
top_k, seed) usando grid_search o random_search.

Principios metodologicos heredados de Fase 1:
  P1  Solo se mide la inferencia: tracker CodeCarbon inicia justo antes de
      llamar al modelo y se detiene inmediatamente despues.
  P2  La carga del modelo ocurre ANTES de iniciar el tracker.
  P3  Los sleep de cooldown / warmup ocurren FUERA del tracker.
  P4  Evaluacion del juez (Claude) corre separada, sin overlap con medicion.
  P5  Prompts verbatim del mismo subset MT-Bench de Fase 1.
  P7  Si el subset no existe, el script falla. Sin prompts alternativos.
  P9  Se mide baseline antes de cada configuracion unica de carga de modelo.
  P10 CSV incluye energia medida y energia corregida por baseline.
  P12 Se detecta context overflow y truncamiento.
  P13 Se guarda finish_reason en cada fila.
  P14 Cada fila tiene turn_id deterministico para reanudacion sin duplicados.
  P22 El ganador de Fase 2 debe cumplir restriccion de calidad:
      calidad >= calidad_baseline - 0.25 pts
      calidad >= calidad_baseline * 0.97

Modos de seleccion de baseline (optimization_config.yaml):
  auto_best_pareto                         — configuracion Pareto-eficiente con
                                             mayor calidad entre las del frente
  auto_best_quality_under_energy_constraint — mayor calidad sin exceder P75 energia
  explicit:<config_id>                     — ID explicito de la columna config_id

Busqueda en dos etapas:
  Stage A: repetitions_per_trial (default 5) — descarte rapido
  Stage B: validation_repetitions (default 15) — validacion completa

Salidas en results/optimization/:
  stage_a/turn_results.csv|jsonl
  stage_a/conversation_results.csv|jsonl
  stage_a/judge_results.csv|jsonl
  stage_a/stage_a_summary.csv
  stage_b/turn_results.csv|jsonl
  stage_b/conversation_results.csv|jsonl
  stage_b/judge_results.csv|jsonl
  stage_b/stage_b_summary.csv
  optimization_report.json

Uso:
  python optimize_winner.py
  python optimize_winner.py --dry-run
  python optimize_winner.py --smoke-test
  python optimize_winner.py --no-resume
  python optimize_winner.py --stage a
  python optimize_winner.py --stage b
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import os
import platform
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import scipy.stats
import yaml

try:
    from llama_cpp import Llama, __version__ as llama_version
    LLAMA_AVAILABLE = True
except ImportError:
    Llama = None          # type: ignore[misc,assignment]
    llama_version = "unavailable"
    LLAMA_AVAILABLE = False

try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
    try:
        import codecarbon as _cc_pkg
        _codecarbon_version: str = getattr(_cc_pkg, "__version__", "unknown")
    except Exception:
        _codecarbon_version = "unknown"
except ImportError:
    EmissionsTracker = None          # type: ignore[misc,assignment]
    CODECARBON_AVAILABLE = False
    _codecarbon_version = "unavailable"

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config, ExperimentConfig
from src.hardware_profile import detect_hardware, HardwareProfile
from src.baseline_energy import medir_baseline_calibracion
from src.prompt_builder import build_prompt
from src.reproducibility import stable_row_id

# ─── Paths ────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG     = ROOT / "config.yaml"
DEFAULT_OPT_CONFIG = ROOT / "optimization_config.yaml"
SUMMARY_CSV        = ROOT / "results" / "summary" / "summary_full_ranking.csv"
DEFAULT_OUT_DIR    = ROOT / "results" / "optimization"
SUBSET_JSONL = (
    ROOT / "data" / "mt_bench" / "subset"
    / "mt_bench_literal_subset_5_per_category.jsonl"
)
SUBSET_YAML = (
    ROOT / "data" / "mt_bench" / "subset"
    / "mt_bench_literal_subset_5_per_category.yaml"
)
REFERENCES_PATH = ROOT / "data" / "mt_bench" / "subset" / "references.yaml"
MODELS_DIR = ROOT / "models"

# ─── Experiment constants ─────────────────────────────────────────────────────

EXPERIMENT_TAG         = "fase2_v1"
PHASE                  = "2"
CONTEXT_OVERFLOW_RATIO = 0.90

MODEL_TYPES: dict[str, str] = {
    "llama-2-7b" : "base",
    "qwen2.5-7b" : "instruct",
}

# Column order for optimization_trials.csv / .jsonl — matches the 42-field spec exactly
TRIALS_COLS: list[str] = [
    "trial_id", "phase", "baseline_configuration_id",
    "model_name", "model_type", "quantization",
    "hardware_profile", "execution_device",
    "n_ctx", "max_tokens", "n_batch", "n_threads",
    "temperature", "top_p", "repeat_penalty", "top_k", "seed",
    "mean_quality_score", "std_quality_score", "ci95_quality_score",
    "mean_energy_wh_per_1k_output_tokens_measured",
    "mean_energy_wh_per_1k_output_tokens_corrected",
    "mean_energy_j_per_output_token_measured",
    "mean_energy_j_per_output_token_corrected",
    "mean_tokens_per_joule_measured",
    "mean_tokens_per_joule_corrected",
    "mean_latency_seconds",
    "mean_edp_joule_second_measured",
    "mean_edp_joule_second_corrected",
    "quality_per_joule_measured",
    "quality_per_joule_corrected",
    "quality_drop_absolute",
    "quality_drop_relative_percent",
    "quality_constraint_passed",
    "energy_reduction_percent",
    "latency_reduction_percent",
    "tokens_per_joule_improvement_percent",
    "pareto_efficient",
    "trial_status",
    "error_message",
]

# ─── Judge constants ──────────────────────────────────────────────────────────

JUDGE_MODEL        = "claude-sonnet-4-6"
JUDGE_MAX_TOKENS   = 1024
JUDGE_DELAY_S      = 0.5
JUDGE_MAX_RETRIES  = 3
JUDGE_RETRY_BASE_S = 2.0
JUDGE_TIMEOUT_S    = 45

JUDGE_TEMPLATE = """\
Act as an impartial judge and evaluate the quality of the multi-turn response provided by an AI assistant to the user questions below.

You must evaluate the assistant's performance across both turns of the conversation.

Evaluation criteria:

1. Correctness: the answers are factually and technically correct.
2. Instruction following: the assistant follows the user instructions in both turns.
3. Relevance: the answers remain focused on the user questions.
4. Completeness: the answers cover the necessary aspects without important omissions.
5. Clarity: the answers are easy to understand and well organized.
6. Conciseness: the answers are not unnecessarily long or repetitive.
7. Usefulness: the assistant genuinely helps the user complete the task.
8. Multi-turn coherence: the second answer correctly uses the context from the first turn.

Do not favor longer answers just because they are longer.
Do not penalize short answers if they are correct and complete.
Do not evaluate the model based on its name, quantization, hardware, or device.
Be objective.

User turn 1:
{turn_1}

Assistant response 1:
{response_1}

User turn 2:
{turn_2}

Assistant response 2:
{response_2}

Return only valid JSON with this format:

{{
  "score": 1-10,
  "correctness": 1-10,
  "instruction_following": 1-10,
  "relevance": 1-10,
  "completeness": 1-10,
  "clarity": 1-10,
  "conciseness": 1-10,
  "usefulness": 1-10,
  "multi_turn_coherence": 1-10,
  "main_strengths": "...",
  "main_weaknesses": "...",
  "final_comment": "..."
}}
"""

JUDGE_TEMPLATE_REFERENCE = """\
Act as an impartial judge and evaluate the quality of the multi-turn response provided by an AI assistant to the user questions below.

You must evaluate the assistant's performance across both turns of the conversation.

You have access to reference answers for this question. Use the reference answers only to assess factual and technical correctness. Do not penalize stylistic or formatting differences from the reference.

Evaluation criteria:

1. Correctness: the answers are factually and technically correct. Use the reference answers to verify accuracy.
2. Instruction following: the assistant follows the user instructions in both turns.
3. Relevance: the answers remain focused on the user questions.
4. Completeness: the answers cover the necessary aspects without important omissions.
5. Clarity: the answers are easy to understand and well organized.
6. Conciseness: the answers are not unnecessarily long or repetitive.
7. Usefulness: the assistant genuinely helps the user complete the task.
8. Multi-turn coherence: the second answer correctly uses the context from the first turn.

Do not favor longer answers just because they are longer.
Do not penalize short answers if they are correct and complete.
Do not evaluate the model based on its name, quantization, hardware, or device.
Be objective.

{reference_section}

User turn 1:
{turn_1}

Assistant response 1:
{response_1}

User turn 2:
{turn_2}

Assistant response 2:
{response_2}

Return only valid JSON with this format:

{{
  "score": 1-10,
  "correctness": 1-10,
  "instruction_following": 1-10,
  "relevance": 1-10,
  "completeness": 1-10,
  "clarity": 1-10,
  "conciseness": 1-10,
  "usefulness": 1-10,
  "multi_turn_coherence": 1-10,
  "main_strengths": "...",
  "main_weaknesses": "...",
  "final_comment": "..."
}}
"""

REFERENCE_GUIDED_CATEGORIES = frozenset({"math", "reasoning", "coding", "stem"})

# ─── CSV column schemas (Phase 1 columns + trial_id, combo_id, trial_stage) ──

TURN_COLS_P2 = [
    "experiment_id", "conversation_id", "phase", "turn_id",
    "timestamp_start", "timestamp_end",
    "model_name", "model_type", "quantization", "model_path",
    "hardware_profile", "execution_device",
    "subset_id", "official_question_id", "original_category", "internal_category",
    "repetition",
    "current_turn_question", "original_question_text", "current_turn_response",
    "status", "error_message",
    "latency_seconds", "prompt_tokens", "completion_tokens", "total_tokens",
    "finish_reason", "truncated_output", "context_overflow",
    "measured_energy_kwh", "measured_energy_wh", "measured_energy_joules",
    "baseline_power_watts",
    "baseline_corrected_energy_joules", "baseline_corrected_energy_wh",
    "emissions_kg_co2", "emissions_g_co2", "emissions_mg_co2",
    "energy_j_per_output_token_measured", "energy_wh_per_1k_output_tokens_measured",
    "tokens_per_joule_measured",
    "energy_j_per_output_token_corrected", "energy_wh_per_1k_output_tokens_corrected",
    "tokens_per_joule_corrected",
    "latency_per_output_token_ms", "tokens_per_second",
    "edp_joule_second_measured", "edp_joule_second_corrected",
    "operational_sci_per_response", "operational_sci_per_1k_output_tokens",
    "n_ctx", "max_tokens", "temperature", "top_p", "repeat_penalty", "top_k", "seed",
    "n_threads", "n_batch", "n_gpu_layers",
    "python_version", "platform_system", "platform_release",
    "llama_cpp_version", "codecarbon_version",
    "cpu_detected", "ram_total_gb_detected", "gpu_name_detected",
    "gpu_memory_total_mb",
    "gpu_memory_used_before_mb", "gpu_memory_used_after_mb",
    "gpu_utilization_before_percent", "gpu_utilization_after_percent",
    "gpu_power_before_w", "gpu_power_after_w",
    "trial_id", "combo_id", "trial_stage",
]

CONV_COLS_P2 = [
    "conversation_id", "phase",
    "timestamp_start", "timestamp_end",
    "model_name", "model_type", "quantization", "model_path",
    "hardware_profile", "execution_device",
    "subset_id", "official_question_id", "original_category", "internal_category",
    "repetition",
    "turn_1_text", "turn_2_text", "response_1_text", "response_2_text",
    "status_conversation", "error_message_conversation",
    "total_latency_seconds",
    "total_prompt_tokens", "total_completion_tokens", "total_tokens",
    "total_measured_energy_wh", "total_measured_energy_joules",
    "total_baseline_corrected_energy_wh", "total_baseline_corrected_energy_joules",
    "total_emissions_kg_co2",
    "total_energy_j_per_output_token_measured",
    "total_energy_wh_per_1k_output_tokens_measured",
    "total_tokens_per_joule_measured",
    "total_energy_j_per_output_token_corrected",
    "total_energy_wh_per_1k_output_tokens_corrected",
    "total_tokens_per_joule_corrected",
    "total_edp_joule_second_measured", "total_edp_joule_second_corrected",
    "total_operational_sci_per_conversation",
    "total_operational_sci_per_1k_output_tokens",
    "any_truncated_output", "any_context_overflow",
    "turn_1_status", "turn_2_status",
    "n_ctx", "max_tokens", "temperature", "top_p", "repeat_penalty", "top_k", "seed",
    "n_threads", "n_batch", "n_gpu_layers",
    "trial_id", "combo_id", "trial_stage",
]

JUDGE_COLS = [
    "conversation_id", "phase",
    "model_name", "model_type", "quantization", "hardware_profile", "execution_device",
    "subset_id", "official_question_id", "original_category", "internal_category",
    "repetition",
    "score", "correctness", "instruction_following", "relevance",
    "completeness", "clarity", "conciseness", "usefulness", "multi_turn_coherence",
    "main_strengths", "main_weaknesses", "final_comment",
    "judge_model", "judge_timestamp", "judge_status", "judge_error_message",
    "judge_prompt_mode", "reference_guided",
    "trial_id", "combo_id", "trial_stage",
]

# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class TrialParams:
    trial_id: int
    n_ctx: int
    max_tokens: int
    n_batch: int
    n_threads: int
    temperature: float
    top_p: float
    repeat_penalty: float
    top_k: int
    seed: int

    @property
    def combo_id(self) -> str:
        key = (
            f"{self.n_ctx}_{self.max_tokens}_{self.n_batch}_{self.n_threads}_"
            f"{self.temperature}_{self.top_p}_{self.repeat_penalty}_{self.top_k}_{self.seed}"
        )
        return hashlib.md5(key.encode()).hexdigest()[:8]

    @property
    def load_key(self) -> tuple:
        """Parameters that require model reload."""
        return (self.n_ctx, self.n_batch, self.n_threads)


@dataclass
class OptimizationConfig:
    selected_configuration: str
    search_method: str
    max_trials: int
    repetitions_per_trial: int
    validation_repetitions: int
    # Discrete search space for the 6 optimizable parameters.
    # temperature, top_p, and seed are NOT here — they live in fixed_parameters.
    parameter_space: dict
    # Fixed values injected into every trial (not searched over).
    fixed_temperature: float = 0.0
    fixed_top_p: float = 1.0
    fixed_seed: int = 42
    quality_constraint_max_drop_absolute: float = 0.25
    quality_constraint_max_drop_relative_percent: float = 3.0
    energy_constraint_percentile: float = 75.0
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUT_DIR)
    stage_a_pareto_top_n: int = 5
    # Optional validation split: separate prompt subsets for tuning (Stage A)
    # and validation (Stage B).  When enabled the final winner is selected on
    # validation prompts that were never seen during the Stage A parameter search.
    validation_split_enabled: bool = False
    validation_split_strategy: str = "by_category"   # by_category | random
    validation_tuning_categories: list = field(default_factory=list)
    validation_validation_categories: list = field(default_factory=list)
    validation_tuning_fraction: float = 0.6           # used only for strategy=random


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe_div(
    num: Optional[float], den: Optional[float], scale: float = 1.0
) -> Optional[float]:
    if num is None or den is None or den == 0:
        return None
    return num / den * scale


def _flt(v: object) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _gpu_snapshot() -> dict:
    snap: dict = {"memory_used_mb": None, "utilization_percent": None, "power_w": None}
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        snap["memory_used_mb"] = mem.used / (1024 * 1024)
        snap["utilization_percent"] = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
        snap["power_w"] = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
    except Exception:
        pass
    return snap


def _gpu_total_mb(hw: HardwareProfile) -> Optional[float]:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(h).total / (1024 * 1024)
    except Exception:
        pass
    vram = hw.gpu_vram_gb
    return vram * 1024 if vram else None


# ─── Config loading ───────────────────────────────────────────────────────────

def _parse_space_entry(v: object) -> list:
    """Parse one parameter_space entry into a list of values."""
    if isinstance(v, dict):
        vals = v.get("values") or v.get("options") or []
        return list(vals) if isinstance(vals, (list, tuple)) else [vals]
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def _parse_validation_split(vsplit: dict) -> dict:
    """Parse optional validation_split section from optimization_config.yaml."""
    enabled  = bool(vsplit.get("enabled", False))
    strategy = str(vsplit.get("strategy", "by_category"))
    if strategy not in ("by_category", "random"):
        raise ValueError(
            f"validation_split.strategy debe ser 'by_category' o 'random', "
            f"no '{strategy}'."
        )
    return {
        "validation_split_enabled"              : enabled,
        "validation_split_strategy"             : strategy,
        "validation_tuning_categories"          : list(vsplit.get("tuning_categories") or []),
        "validation_validation_categories"      : list(vsplit.get("validation_categories") or []),
        "validation_tuning_fraction"            : float(vsplit.get("tuning_fraction", 0.6)),
    }


def load_optimization_config(path: Path) -> OptimizationConfig:
    if not path.exists():
        raise FileNotFoundError(f"optimization_config.yaml no encontrado: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        root = yaml.safe_load(fh) or {}

    # Support both top-level keys and wrapped under "optimization:"
    raw = root.get("optimization", root)

    # ── parameter_space: the 6 optimizable parameters ─────────────────────
    # temperature, top_p, seed are NOT expected here — they are in fixed_parameters.
    space: dict = {
        k: _parse_space_entry(v)
        for k, v in raw.get("parameter_space", {}).items()
    }
    required_search = {"n_ctx", "max_tokens", "n_batch", "n_threads",
                       "repeat_penalty", "top_k"}
    missing = required_search - set(space)
    if missing:
        raise ValueError(
            f"parameter_space en {path.name} no contiene: {sorted(missing)}\n"
            "  Parametros requeridos: n_ctx, max_tokens, n_batch, n_threads, "
            "repeat_penalty, top_k"
        )

    # Reject fixed params being accidentally placed in parameter_space
    forbidden_in_space = {"temperature", "top_p", "seed"}
    intruders = forbidden_in_space & set(space)
    if intruders:
        raise ValueError(
            f"{path.name}: {sorted(intruders)} son parametros fijos y no deben "
            "estar en parameter_space. Moverlos a fixed_parameters."
        )

    # ── fixed_parameters: temperature, top_p, seed ────────────────────────
    fixed = raw.get("fixed_parameters", {})
    fixed_temperature = float(fixed.get("temperature", 0.0))
    fixed_top_p       = float(fixed.get("top_p", 1.0))
    fixed_seed        = int(fixed.get("seed", 42))

    if fixed_temperature != 0.0:
        raise ValueError(
            f"{path.name}: temperature debe ser 0.0 para medicion de energia "
            f"determinista. Valor encontrado: {fixed_temperature}"
        )

    constraints = raw.get("quality_constraint", {})
    return OptimizationConfig(
        selected_configuration=str(raw.get("selected_configuration", "auto_best_pareto")),
        search_method=str(raw.get("search_method", "random_search")),
        max_trials=int(raw.get("max_trials", 50)),
        repetitions_per_trial=int(raw.get("repetitions_per_trial", 5)),
        validation_repetitions=int(raw.get("validation_repetitions", 15)),
        parameter_space=space,
        fixed_temperature=fixed_temperature,
        fixed_top_p=fixed_top_p,
        fixed_seed=fixed_seed,
        quality_constraint_max_drop_absolute=float(
            constraints.get("max_allowed_drop_absolute", 0.25)
        ),
        quality_constraint_max_drop_relative_percent=float(
            constraints.get("max_allowed_drop_relative_percent", 3.0)
        ),
        energy_constraint_percentile=float(raw.get("energy_constraint_percentile", 75.0)),
        output_dir=Path(raw.get("output_dir", str(DEFAULT_OUT_DIR))),
        stage_a_pareto_top_n=int(raw.get("stage_a_pareto_top_n", 5)),
        **_parse_validation_split(raw.get("validation_split", {})),
    )


# ─── Subset loading ───────────────────────────────────────────────────────────

def load_subset_questions(max_per_category: Optional[int] = None) -> list[dict]:
    """Load MT-Bench subset. P7: fails if file not found."""
    if SUBSET_JSONL.exists():
        questions: list[dict] = []
        with open(SUBSET_JSONL, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))
    elif SUBSET_YAML.exists():
        with open(SUBSET_YAML, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        questions = data.get("questions", []) if isinstance(data, dict) else list(data or [])
    else:
        raise FileNotFoundError(
            f"Subset MT-Bench no encontrado.\n"
            f"  Esperado: {SUBSET_JSONL}\n"
            "  Ejecutar: python scripts/prepare_mt_bench_subset.py"
        )

    normalized: list[dict] = []
    for q in questions:
        turns = q.get("turns") or []
        if len(turns) < 2:
            continue
        normalized.append({
            "question_id"       : int(q["question_id"]),
            "original_category" : str(q.get("original_category") or ""),
            "category"          : str(
                q.get("internal_category")
                or q.get("category")
                or q.get("original_category")
                or ""
            ),
            "turn1": turns[0],
            "turn2": turns[1],
        })

    if not normalized:
        raise ValueError("El subset MT-Bench no contiene preguntas validas.")

    if max_per_category is not None:
        by_cat: dict = defaultdict(list)
        for q in normalized:
            by_cat[q["category"]].append(q)
        subset: list[dict] = []
        for qs in by_cat.values():
            subset.extend(qs[:max_per_category])
        normalized = subset

    return normalized


def _split_questions(
    questions: list[dict],
    opt_cfg: "OptimizationConfig",
) -> tuple[list[dict], list[dict]]:
    """
    Partition MT-Bench questions into tuning and validation sets.

    Returns (tuning_questions, validation_questions).

    Strategies:
      by_category — tuning_categories → Stage A; validation_categories → Stage B.
                    Questions not listed in either set are assigned to validation.
      random      — deterministic random split controlled by validation_tuning_fraction
                    and random seed 42.

    Both lists are guaranteed non-empty even when the full subset is tiny.
    """
    strategy = opt_cfg.validation_split_strategy

    if strategy == "by_category":
        tuning_cats = {c.lower() for c in opt_cfg.validation_tuning_categories}
        val_cats    = {c.lower() for c in opt_cfg.validation_validation_categories}
        tuning: list[dict] = []
        val:    list[dict] = []
        for q in questions:
            cat = q.get("category", "").lower()
            if cat in tuning_cats:
                tuning.append(q)
            else:
                # questions not explicitly in tuning go to validation
                val.append(q)
        if val_cats:
            # if explicit validation_categories given, only keep those in val
            val = [q for q in val if q.get("category", "").lower() in val_cats] or val
    else:  # random
        rng = random.Random(42)
        shuffled = list(questions)
        rng.shuffle(shuffled)
        n_tuning = max(1, int(len(shuffled) * opt_cfg.validation_tuning_fraction))
        tuning = shuffled[:n_tuning]
        val    = shuffled[n_tuning:]

    # Safety: neither list must be empty
    if not tuning:
        tuning = questions
    if not val:
        val = questions

    return tuning, val


def _load_references() -> Optional[dict]:
    if not REFERENCES_PATH.exists():
        return None
    try:
        with open(REFERENCES_PATH, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            return None
        return {str(k): v for k, v in data.items()}
    except Exception:
        return None


# ─── Baseline selection ───────────────────────────────────────────────────────

def _read_summary(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Phase 1 summary no encontrado: {path}\n"
            "  Ejecutar primero: python analysis_summary.py"
        )
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    if not rows:
        raise ValueError(f"Phase 1 summary vacio: {path}")
    return rows


def select_baseline(summary_path: Path, opt_cfg: OptimizationConfig) -> dict:
    rows = _read_summary(summary_path)
    mode = opt_cfg.selected_configuration

    if mode == "auto_best_pareto":
        return _select_pareto(rows)
    elif mode == "auto_best_quality_under_energy_constraint":
        return _select_quality_energy(rows, opt_cfg.energy_constraint_percentile)
    elif mode.startswith("explicit:"):
        cfg_id = mode.split(":", 1)[1].strip()
        return _select_explicit(rows, cfg_id)
    else:
        raise ValueError(
            f"selected_configuration desconocido: '{mode}'\n"
            "  Valores validos: auto_best_pareto, "
            "auto_best_quality_under_energy_constraint, explicit:<config_id>"
        )


def _select_pareto(rows: list[dict]) -> dict:
    pareto = [r for r in rows if str(r.get("pareto_efficient", "")).lower() == "true"]
    if not pareto:
        pareto = rows
        print("  AVISO: ninguna fila con pareto_efficient=True, usando todas.")
    best = max(pareto, key=lambda r: _flt(r.get("mean_quality_score")) or 0.0)
    print(
        f"  [Baseline] auto_best_pareto → "
        f"{best.get('model_name')}/{best.get('quantization')} "
        f"calidad={best.get('mean_quality_score')}"
    )
    return best


def _select_quality_energy(rows: list[dict], energy_pct: float) -> dict:
    energies = []
    for r in rows:
        v = r.get("mean_total_energy_wh_per_1k_output_tokens_corrected") or \
            r.get("mean_total_energy_wh_per_1k_output_tokens_measured")
        val = _flt(v)
        if val is not None:
            energies.append(val)
    if not energies:
        return _select_pareto(rows)
    threshold = float(np.percentile(energies, energy_pct))
    candidates = []
    for r in rows:
        v = r.get("mean_total_energy_wh_per_1k_output_tokens_corrected") or \
            r.get("mean_total_energy_wh_per_1k_output_tokens_measured")
        val = _flt(v)
        if val is not None and val <= threshold:
            candidates.append(r)
    if not candidates:
        candidates = rows
    best = max(candidates, key=lambda r: _flt(r.get("mean_quality_score")) or 0.0)
    print(
        f"  [Baseline] auto_best_quality_under_energy_constraint → "
        f"{best.get('model_name')}/{best.get('quantization')} "
        f"calidad={best.get('mean_quality_score')} "
        f"(threshold P{int(energy_pct)} = {threshold:.4f} Wh/1k)"
    )
    return best


def _select_explicit(rows: list[dict], config_id: str) -> dict:
    for r in rows:
        if str(r.get("config_id", "")).strip() == config_id:
            print(
                f"  [Baseline] explicit:{config_id} → "
                f"{r.get('model_name')}/{r.get('quantization')}"
            )
            return r
    ids = [r.get("config_id", "") for r in rows[:5]]
    raise ValueError(
        f"config_id '{config_id}' no encontrado en summary.\n"
        f"  Primeros IDs disponibles: {ids}"
    )


# ─── Model path resolution ────────────────────────────────────────────────────

def resolve_model_path(model_name: str, quantization: str) -> Path:
    quant_upper = quantization.upper()
    model_dir = MODELS_DIR / model_name
    candidates: list[Path] = []

    if model_dir.exists():
        for p in model_dir.glob("*.gguf"):
            if quant_upper in p.name.upper():
                candidates.append(p)
        if not candidates:
            candidates = list(model_dir.glob("*.gguf"))

    if not candidates:
        for p in MODELS_DIR.rglob("*.gguf"):
            dir_up = p.parent.name.upper().replace("-", "")
            mod_up = model_name.upper().replace("-", "")
            if mod_up in dir_up and quant_upper in p.name.upper():
                candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"No se encontro archivo GGUF para {model_name}/{quantization}\n"
            f"  Buscado en: {MODELS_DIR}"
        )
    return candidates[0]


# ─── Trial generation ─────────────────────────────────────────────────────────

def generate_trials(opt_cfg: OptimizationConfig) -> list[TrialParams]:
    """
    Generate parameter trials by searching over the 6 optimizable parameters.
    temperature, top_p, and seed are taken from fixed_parameters and injected
    into every trial unchanged.
    """
    space = opt_cfg.parameter_space
    # Ordered list of the searchable parameter keys
    search_keys = ["n_ctx", "max_tokens", "n_batch", "n_threads", "repeat_penalty", "top_k"]
    lists = [space[k] for k in search_keys]
    all_combos = list(itertools.product(*lists))

    if opt_cfg.search_method == "grid_search":
        selected = all_combos[: opt_cfg.max_trials]
    else:
        rng = random.Random(42)
        n = min(opt_cfg.max_trials, len(all_combos))
        selected = rng.sample(all_combos, n)

    return [
        TrialParams(
            trial_id=i + 1,
            **dict(zip(search_keys, combo)),  # type: ignore[arg-type]
            # Fixed — same for every trial
            temperature=opt_cfg.fixed_temperature,
            top_p=opt_cfg.fixed_top_p,
            seed=opt_cfg.fixed_seed,
        )
        for i, combo in enumerate(selected)
    ]


# ─── Model loading (P2) ──────────────────────────────────────────────────────

def cargar_modelo_fase2(
    model_path: Path,
    device: str,
    hw: HardwareProfile,
    trial: TrialParams,
    base_cfg: ExperimentConfig,
) -> Optional["Llama"]:
    """Load model with trial params. P2: outside tracker."""
    if not LLAMA_AVAILABLE:
        return None
    n_gpu = hw.get_n_gpu_layers(device)
    try:
        return Llama(
            model_path=str(model_path),
            n_ctx=trial.n_ctx,
            n_gpu_layers=n_gpu,
            n_threads=trial.n_threads,
            n_batch=trial.n_batch,
            seed=trial.seed,
            verbose=False,
        )
    except Exception as e:
        print(f"  ERROR cargando modelo: {e}")
        return None


def warmup_modelo(llm: "Llama", n_runs: int = 1) -> None:
    """P3: warmup outside tracker."""
    for _ in range(n_runs):
        try:
            llm("Hello.", max_tokens=5, temperature=0.0, echo=False, stream=False)
        except Exception:
            pass


# ─── Turn measurement (P1) ───────────────────────────────────────────────────

def medir_turno_fase2(
    llm: Optional["Llama"],
    prompt: str,
    original_question_text: str,
    turn: int,
    question: dict,
    device: str,
    model_name: str,
    quant: str,
    model_path: Path,
    repetition: int,
    hw: HardwareProfile,
    baseline: dict,
    base_cfg: ExperimentConfig,
    trial: TrialParams,
    trial_stage: str,
    experiment_id: str,
    conversation_id: str,
    subset_idx: int,
    gpu_total_mb_val: Optional[float],
    dry_run: bool = False,
) -> dict:
    """
    Measure one inference turn for Phase 2.
    P1: CodeCarbon tracker only wraps the inference call.
    P10: baseline-corrected energy computed same as Phase 1.
    """
    hw_profile = base_cfg.hardware_profile
    n_gpu      = hw.get_n_gpu_layers(device)

    try:
        rel_path = str(model_path.relative_to(ROOT))
    except ValueError:
        rel_path = str(model_path)

    turn_id = stable_row_id(
        EXPERIMENT_TAG, model_name, quant,
        hw_profile, device,
        question["question_id"], turn, repetition,
        str(trial.trial_id), trial_stage,
    )

    baseline_pw = float(baseline.get("baseline_total_power_w", 0.0))

    def _build_row(
        status: str,
        error_message: str,
        finish_reason: str,
        text_gen: str = "",
        ts_start: Optional[str] = None,
        ts_end: Optional[str] = None,
        latency: Optional[float] = None,
        tok_in: Optional[int] = None,
        tok_out: Optional[int] = None,
        tok_total: Optional[int] = None,
        measured_j: Optional[float] = None,
        measured_wh: Optional[float] = None,
        measured_kwh: Optional[float] = None,
        bc_j: Optional[float] = None,
        bc_wh: Optional[float] = None,
        emis_kg: float = 0.0,
        truncated: Optional[bool] = None,
        ctx_overflow: Optional[bool] = None,
        gpu_before: Optional[dict] = None,
        gpu_after: Optional[dict] = None,
    ) -> dict:
        ts  = ts_start or datetime.now(timezone.utc).isoformat()
        te  = ts_end or ts
        em_g = emis_kg * 1000.0
        em_mg = emis_kg * 1_000_000.0
        net_wh = bc_wh if bc_wh is not None else 0.0
        net_co2_g = net_wh / 1000.0 * hw.carbon_intensity_g_kwh
        sci_1k = _safe_div(net_co2_g, tok_out, scale=1000.0)
        edp_meas = (measured_j * latency) if (measured_j and latency) else None
        edp_corr = (bc_j * latency) if (bc_j is not None and latency) else None
        gb = gpu_before or {}
        ga = gpu_after or {}
        return {
            "experiment_id"                           : experiment_id,
            "conversation_id"                         : conversation_id,
            "phase"                                   : PHASE,
            "turn_id"                                 : turn_id,
            "timestamp_start"                         : ts,
            "timestamp_end"                           : te,
            "model_name"                              : model_name,
            "model_type"                              : MODEL_TYPES.get(model_name, "unknown"),
            "quantization"                            : quant,
            "model_path"                              : rel_path,
            "hardware_profile"                        : hw_profile,
            "execution_device"                        : device,
            "subset_id"                               : subset_idx,
            "official_question_id"                    : question["question_id"],
            "original_category"                       : question.get("original_category", ""),
            "internal_category"                       : question.get("category", ""),
            "repetition"                              : repetition,
            "current_turn_question"                   : prompt,
            "original_question_text"                  : original_question_text,
            "current_turn_response"                   : text_gen,
            "status"                                  : status,
            "error_message"                           : error_message,
            "latency_seconds"                         : latency,
            "prompt_tokens"                           : tok_in,
            "completion_tokens"                       : tok_out,
            "total_tokens"                            : tok_total,
            "finish_reason"                           : finish_reason,
            "truncated_output"                        : truncated,
            "context_overflow"                        : ctx_overflow,
            "measured_energy_kwh"                     : measured_kwh,
            "measured_energy_wh"                      : measured_wh,
            "measured_energy_joules"                  : measured_j,
            "baseline_power_watts"                    : baseline_pw,
            "baseline_corrected_energy_joules"        : bc_j,
            "baseline_corrected_energy_wh"            : bc_wh,
            "emissions_kg_co2"                        : emis_kg,
            "emissions_g_co2"                         : em_g,
            "emissions_mg_co2"                        : em_mg,
            "energy_j_per_output_token_measured"      : _safe_div(measured_j, tok_out),
            "energy_wh_per_1k_output_tokens_measured" : _safe_div(measured_wh, tok_out, 1000.0),
            "tokens_per_joule_measured"               : _safe_div(tok_out, measured_j),
            "energy_j_per_output_token_corrected"     : _safe_div(bc_j, tok_out),
            "energy_wh_per_1k_output_tokens_corrected": (
                _safe_div(bc_wh, tok_out, 1000.0) if bc_wh is not None else None
            ),
            "tokens_per_joule_corrected"              : _safe_div(tok_out, bc_j),
            "latency_per_output_token_ms"             : (
                latency / tok_out * 1000.0 if (latency and tok_out) else None
            ),
            "tokens_per_second"                       : (
                tok_out / latency if (tok_out and latency) else None
            ),
            "edp_joule_second_measured"               : edp_meas,
            "edp_joule_second_corrected"              : edp_corr,
            "operational_sci_per_response"            : net_co2_g,
            "operational_sci_per_1k_output_tokens"    : sci_1k,
            "n_ctx"                                   : trial.n_ctx,
            "max_tokens"                              : trial.max_tokens,
            "temperature"                             : trial.temperature,
            "top_p"                                   : trial.top_p,
            "repeat_penalty"                          : trial.repeat_penalty,
            "top_k"                                   : trial.top_k,
            "seed"                                    : trial.seed,
            "n_threads"                               : trial.n_threads,
            "n_batch"                                 : trial.n_batch,
            "n_gpu_layers"                            : n_gpu,
            "python_version"                          : platform.python_version(),
            "platform_system"                         : platform.system(),
            "platform_release"                        : platform.release(),
            "llama_cpp_version"                       : llama_version,
            "codecarbon_version"                      : _codecarbon_version,
            "cpu_detected"                            : hw.cpu_model,
            "ram_total_gb_detected"                   : hw.ram_total_gb,
            "gpu_name_detected"                       : hw.gpu_model,
            "gpu_memory_total_mb"                     : gpu_total_mb_val,
            "gpu_memory_used_before_mb"               : gb.get("memory_used_mb"),
            "gpu_memory_used_after_mb"                : ga.get("memory_used_mb"),
            "gpu_utilization_before_percent"          : gb.get("utilization_percent"),
            "gpu_utilization_after_percent"           : ga.get("utilization_percent"),
            "gpu_power_before_w"                      : gb.get("power_w"),
            "gpu_power_after_w"                       : ga.get("power_w"),
            "trial_id"                                : trial.trial_id,
            "combo_id"                                : trial.combo_id,
            "trial_stage"                             : trial_stage,
        }

    if dry_run:
        return _build_row(
            status="dry_run", error_message="", finish_reason="dry_run",
            text_gen="[DRY RUN] simulated response",
            latency=0.1, tok_in=10, tok_out=5, tok_total=15,
            measured_j=0.001, measured_wh=2.78e-7, measured_kwh=2.78e-10,
            bc_j=0.0005, bc_wh=1.39e-7,
            truncated=False, ctx_overflow=False,
        )

    if llm is None:
        return _build_row(
            status="error", error_message="llm not loaded", finish_reason="error",
        )

    # P12: pre-check context overflow
    try:
        est_pt = len(llm.tokenize(prompt.encode("utf-8"), add_bos=True))
    except Exception:
        est_pt = max(1, len(prompt) // 4)

    if (
        not base_cfg.allow_prompt_truncation
        and est_pt + trial.max_tokens > trial.n_ctx
        and base_cfg.on_context_overflow == "skip"
    ):
        print(f"\n  [CONTEXT OVERFLOW — skip] ~{est_pt}+{trial.max_tokens}>{trial.n_ctx}")
        return _build_row(
            status="context_overflow",
            error_message=f"~{est_pt}+{trial.max_tokens}>{trial.n_ctx}",
            finish_reason="context_overflow",
            ctx_overflow=True,
        )

    gpu_before = _gpu_snapshot()

    if not CODECARBON_AVAILABLE:
        t_ini = time.perf_counter()
        ts_start = datetime.now(timezone.utc).isoformat()
        try:
            response = llm(
                prompt,
                max_tokens=trial.max_tokens,
                temperature=trial.temperature,
                top_p=trial.top_p,
                top_k=trial.top_k,
                repeat_penalty=trial.repeat_penalty,
                echo=False, stop=None, seed=trial.seed, stream=False,
            )
        except Exception as e:
            t_total = time.perf_counter() - t_ini
            return _build_row(
                status="error", error_message=str(e), finish_reason="error",
                ts_start=ts_start, ts_end=datetime.now(timezone.utc).isoformat(),
                latency=t_total,
            )
        t_total = time.perf_counter() - t_ini
        ts_end = datetime.now(timezone.utc).isoformat()
        gpu_after = _gpu_snapshot()
        ch = (response.get("choices") or [{}])[0]
        usage = response.get("usage") or {}
        tok_out = int(usage.get("completion_tokens", 0))
        return _build_row(
            status="ok", error_message="", finish_reason=ch.get("finish_reason", "unknown"),
            text_gen=ch.get("text", ""), ts_start=ts_start, ts_end=ts_end, latency=t_total,
            tok_in=int(usage.get("prompt_tokens", 0)),
            tok_out=tok_out,
            tok_total=int(usage.get("total_tokens", 0)),
            measured_j=0.0, measured_wh=0.0, measured_kwh=0.0,
            bc_j=None, bc_wh=None,
            truncated=(ch.get("finish_reason") == "length" or tok_out >= trial.max_tokens),
            ctx_overflow=False,
            gpu_before=gpu_before, gpu_after=gpu_after,
        )

    # P1: tracker only around inference call
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
    ts_start = datetime.now(timezone.utc).isoformat()
    tracker.start()
    t_ini = time.perf_counter()

    try:
        response = llm(
            prompt,
            max_tokens=trial.max_tokens,
            temperature=trial.temperature,
            top_p=trial.top_p,
            top_k=trial.top_k,
            repeat_penalty=trial.repeat_penalty,
            echo=False, stop=None, seed=trial.seed, stream=False,
        )
    except Exception as e:
        t_total = time.perf_counter() - t_ini
        tracker.stop()
        return _build_row(
            status="error", error_message=str(e), finish_reason="error",
            ts_start=ts_start, ts_end=datetime.now(timezone.utc).isoformat(),
            latency=t_total,
        )

    t_total = time.perf_counter() - t_ini
    tracker.stop()  # P1: stop immediately after inference
    ts_end = datetime.now(timezone.utc).isoformat()
    gpu_after = _gpu_snapshot()

    ch = (response.get("choices") or [{}])[0]
    usage = response.get("usage") or {}
    tok_in    = int(usage.get("prompt_tokens", 0))
    tok_out   = int(usage.get("completion_tokens", 0))
    tok_total = int(usage.get("total_tokens", 0))
    finish_reason = ch.get("finish_reason", "unknown")
    text_gen = ch.get("text", "")

    ctx_overflow = (tok_in + tok_out) / trial.n_ctx >= CONTEXT_OVERFLOW_RATIO
    truncated    = (finish_reason == "length" or tok_out >= trial.max_tokens)

    try:
        e_kwh = float(tracker._total_energy.kWh)
    except AttributeError:
        e_kwh = 0.0
    m_kwh = e_kwh
    m_wh  = e_kwh * 1000.0
    m_j   = m_wh * 3600.0
    emis_kg = getattr(tracker, "_total_emissions", 0.0) or 0.0

    # P10: baseline-corrected energy
    bc_raw = m_j - baseline_pw * t_total
    bc_j  = bc_raw if bc_raw >= 0 else None
    bc_wh = bc_j / 3600.0 if bc_j is not None else None

    return _build_row(
        status="ok", error_message="", finish_reason=finish_reason,
        text_gen=text_gen, ts_start=ts_start, ts_end=ts_end, latency=t_total,
        tok_in=tok_in, tok_out=tok_out, tok_total=tok_total,
        measured_j=m_j, measured_wh=m_wh, measured_kwh=m_kwh,
        bc_j=bc_j, bc_wh=bc_wh, emis_kg=emis_kg,
        truncated=truncated, ctx_overflow=ctx_overflow,
        gpu_before=gpu_before, gpu_after=gpu_after,
    )


# ─── Conversation row (P8) ────────────────────────────────────────────────────

def _conversation_row_fase2(r1: dict, r2: dict, experiment_id: str) -> dict:
    """P8: aggregate two turn rows into a conversation row."""
    def _add(f: str) -> Optional[float]:
        v1, v2 = r1.get(f), r2.get(f)
        if v1 is None and v2 is None:
            return None
        return float(v1 or 0.0) + float(v2 or 0.0)

    conv_id = stable_row_id(
        EXPERIMENT_TAG,
        r1["model_name"], r1["quantization"],
        r1["hardware_profile"], r1["execution_device"],
        r1["official_question_id"], "conv", r1["repetition"],
        str(r1.get("trial_id", "")), r1.get("trial_stage", ""),
    )

    lat     = _add("latency_seconds") or 0.0
    tok_out = int(_add("completion_tokens") or 0)
    tok_in  = int(_add("prompt_tokens") or 0)
    tok_tot = int(_add("total_tokens") or 0)
    meas_j  = _add("measured_energy_joules") or 0.0
    meas_wh = _add("measured_energy_wh") or 0.0

    bc_j1 = r1.get("baseline_corrected_energy_joules")
    bc_j2 = r2.get("baseline_corrected_energy_joules")
    if bc_j1 is not None or bc_j2 is not None:
        bc_j  = float(bc_j1 or 0.0) + float(bc_j2 or 0.0)
        bc_wh = bc_j / 3600.0
    else:
        bc_j  = None
        bc_wh = None

    emis_kg  = _add("emissions_kg_co2") or 0.0
    sci_conv = _add("operational_sci_per_response") or 0.0
    sci_1k   = _safe_div(sci_conv, tok_out, scale=1000.0)
    edp_meas = meas_j * lat
    edp_corr = bc_j * lat if bc_j is not None else None

    any_overflow = bool(r1.get("context_overflow")) or bool(r2.get("context_overflow"))
    any_trunc    = bool(r1.get("truncated_output"))  or bool(r2.get("truncated_output"))
    status_conv  = "ok" if r1.get("status") == "ok" and r2.get("status") == "ok" else "partial"
    turn_2_q     = r2.get("original_question_text") or r2.get("current_turn_question", "")

    return {
        "conversation_id"                              : conv_id,
        "phase"                                        : PHASE,
        "timestamp_start"                              : r1.get("timestamp_start"),
        "timestamp_end"                                : r2.get("timestamp_end"),
        "model_name"                                   : r1["model_name"],
        "model_type"                                   : r1["model_type"],
        "quantization"                                 : r1["quantization"],
        "model_path"                                   : r1["model_path"],
        "hardware_profile"                             : r1["hardware_profile"],
        "execution_device"                             : r1["execution_device"],
        "subset_id"                                    : r1["subset_id"],
        "official_question_id"                         : r1["official_question_id"],
        "original_category"                            : r1["original_category"],
        "internal_category"                            : r1["internal_category"],
        "repetition"                                   : r1["repetition"],
        "turn_1_text"                                  : r1.get("original_question_text", ""),
        "turn_2_text"                                  : turn_2_q,
        "response_1_text"                              : r1.get("current_turn_response", ""),
        "response_2_text"                              : r2.get("current_turn_response", ""),
        "status_conversation"                          : status_conv,
        "error_message_conversation"                   : "",
        "total_latency_seconds"                        : lat,
        "total_prompt_tokens"                          : tok_in,
        "total_completion_tokens"                      : tok_out,
        "total_tokens"                                 : tok_tot,
        "total_measured_energy_wh"                     : meas_wh,
        "total_measured_energy_joules"                 : meas_j,
        "total_baseline_corrected_energy_wh"           : bc_wh,
        "total_baseline_corrected_energy_joules"       : bc_j,
        "total_emissions_kg_co2"                       : emis_kg,
        "total_energy_j_per_output_token_measured"     : _safe_div(meas_j, tok_out),
        "total_energy_wh_per_1k_output_tokens_measured": _safe_div(meas_wh, tok_out, 1000.0),
        "total_tokens_per_joule_measured"              : _safe_div(tok_out, meas_j),
        "total_energy_j_per_output_token_corrected"    : _safe_div(bc_j, tok_out),
        "total_energy_wh_per_1k_output_tokens_corrected": (
            _safe_div(bc_wh, tok_out, 1000.0) if bc_wh is not None else None
        ),
        "total_tokens_per_joule_corrected"             : _safe_div(tok_out, bc_j),
        "total_edp_joule_second_measured"              : edp_meas,
        "total_edp_joule_second_corrected"             : edp_corr,
        "total_operational_sci_per_conversation"       : sci_conv,
        "total_operational_sci_per_1k_output_tokens"   : sci_1k,
        "any_truncated_output"                         : any_trunc,
        "any_context_overflow"                         : any_overflow,
        "turn_1_status"                                : r1.get("status"),
        "turn_2_status"                                : r2.get("status"),
        "n_ctx"                                        : r1["n_ctx"],
        "max_tokens"                                   : r1["max_tokens"],
        "temperature"                                  : r1["temperature"],
        "top_p"                                        : r1["top_p"],
        "repeat_penalty"                               : r1["repeat_penalty"],
        "top_k"                                        : r1["top_k"],
        "seed"                                         : r1["seed"],
        "n_threads"                                    : r1["n_threads"],
        "n_batch"                                      : r1["n_batch"],
        "n_gpu_layers"                                 : r1.get("n_gpu_layers"),
        "trial_id"                                     : r1.get("trial_id"),
        "combo_id"                                     : r1.get("combo_id", ""),
        "trial_stage"                                  : r1.get("trial_stage", ""),
    }


# ─── CSV + JSONL writer with resume (P14) ────────────────────────────────────

class ResultWriter:
    def __init__(
        self, csv_path: Path, jsonl_path: Path, cols: list[str], no_resume: bool
    ):
        self.csv_path    = csv_path
        self.jsonl_path  = jsonl_path
        self.cols        = cols
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        existing_ids: set[str] = set()
        id_col = cols[0]
        if csv_path.exists() and not no_resume:
            with open(csv_path, "r", encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    if row.get(id_col):
                        existing_ids.add(row[id_col])
            if existing_ids:
                print(f"    Resume: {len(existing_ids)} filas previas en {csv_path.name}")

        self.existing_ids = existing_ids
        mode = "a" if (csv_path.exists() and not no_resume) else "w"
        self._csv_fh   = open(csv_path,   mode, newline="", encoding="utf-8")
        self._jsonl_fh = open(jsonl_path, mode if mode == "a" else "w", encoding="utf-8")
        self._writer   = csv.DictWriter(self._csv_fh, fieldnames=cols, extrasaction="ignore")
        if mode == "w":
            self._writer.writeheader()

    def write(self, row: dict) -> bool:
        id_col = self.cols[0]
        rid = str(row.get(id_col, ""))
        if rid and rid in self.existing_ids:
            return False
        self._writer.writerow({c: row.get(c, "") for c in self.cols})
        self._csv_fh.flush()
        self._jsonl_fh.write(json.dumps(row, default=str) + "\n")
        self._jsonl_fh.flush()
        if rid:
            self.existing_ids.add(rid)
        return True

    def close(self) -> None:
        self._csv_fh.close()
        self._jsonl_fh.close()


# ─── Judge ────────────────────────────────────────────────────────────────────

def _parse_judge_response(text: str) -> tuple[Optional[dict], Optional[str]]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data, None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                return data, None
        except json.JSONDecodeError as exc:
            return None, str(exc)
    return None, "no JSON object found"


def _call_claude_judge(
    user_content: str, api_key: str, dry_run: bool
) -> tuple[Optional[dict], Optional[str]]:
    if dry_run:
        return {
            "score": 7, "correctness": 7, "instruction_following": 7,
            "relevance": 7, "completeness": 7, "clarity": 7,
            "conciseness": 7, "usefulness": 7, "multi_turn_coherence": 7,
            "main_strengths": "[DRY RUN]",
            "main_weaknesses": "None",
            "final_comment": "[DRY RUN]",
        }, None

    last_err = "unknown"
    for attempt in range(1, JUDGE_MAX_RETRIES + 1):
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type"     : "application/json",
                    "x-api-key"        : api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model"     : JUDGE_MODEL,
                    "max_tokens": JUDGE_MAX_TOKENS,
                    "messages"  : [{"role": "user", "content": user_content}],
                },
                timeout=JUDGE_TIMEOUT_S,
            )
            if resp.status_code == 429:
                last_err = "rate_limit_429"
                if attempt < JUDGE_MAX_RETRIES:
                    time.sleep(JUDGE_RETRY_BASE_S * (2 ** (attempt - 1)))
                    continue
                return None, last_err
            if resp.status_code != 200:
                last_err = f"http_{resp.status_code}: {resp.text[:200]}"
                if attempt < JUDGE_MAX_RETRIES:
                    time.sleep(JUDGE_RETRY_BASE_S)
                    continue
                return None, last_err
            raw = resp.json()["content"][0]["text"].strip()
            return _parse_judge_response(raw)
        except requests.exceptions.Timeout:
            last_err = f"timeout_{JUDGE_TIMEOUT_S}s"
            if attempt < JUDGE_MAX_RETRIES:
                time.sleep(JUDGE_RETRY_BASE_S * attempt)
                continue
        except requests.exceptions.RequestException as exc:
            last_err = str(exc)
            if attempt < JUDGE_MAX_RETRIES:
                time.sleep(JUDGE_RETRY_BASE_S)
                continue
        return None, last_err
    return None, last_err


def _build_judge_prompt(conv: dict, references: Optional[dict]) -> tuple[str, bool]:
    q_id = str(conv.get("official_question_id", ""))
    cat  = str(conv.get("original_category", "")).lower()
    ref = None
    if references and cat in REFERENCE_GUIDED_CATEGORIES:
        ref = references.get(q_id)

    turn1 = conv.get("turn_1_text", "")
    resp1 = conv.get("response_1_text", "")
    turn2 = conv.get("turn_2_text", "")
    resp2 = conv.get("response_2_text", "")

    if ref:
        r1 = (ref.get("turn_1") or ref.get("reference_1") or "").strip()
        r2 = (ref.get("turn_2") or ref.get("reference_2") or "").strip()
        lines = ["Reference answers (use ONLY to verify factual and technical correctness):"]
        if r1:
            lines.append(f"\nReference for turn 1:\n{r1}")
        if r2:
            lines.append(f"\nReference for turn 2:\n{r2}")
        return JUDGE_TEMPLATE_REFERENCE.format(
            reference_section="\n".join(lines),
            turn_1=turn1, response_1=resp1,
            turn_2=turn2, response_2=resp2,
        ), True

    return JUDGE_TEMPLATE.format(
        turn_1=turn1, response_1=resp1,
        turn_2=turn2, response_2=resp2,
    ), False


def judge_conversations(
    conv_rows: list[dict],
    api_key: str,
    references: Optional[dict],
    dry_run: bool,
    judge_writer: ResultWriter,
    existing_judge_ids: set,
) -> list[dict]:
    results: list[dict] = []
    n = len(conv_rows)
    for idx, conv in enumerate(conv_rows, 1):
        conv_id = conv.get("conversation_id", "")
        if conv_id in existing_judge_ids:
            continue

        prompt_text, ref_guided = _build_judge_prompt(conv, references)
        print(f"    [{idx}/{n}] juez conv {conv_id[:16]}... ", end="", flush=True)
        parsed, err = _call_claude_judge(prompt_text, api_key, dry_run)

        ts = datetime.now(timezone.utc).isoformat()
        score_fields = [
            "score", "correctness", "instruction_following", "relevance",
            "completeness", "clarity", "conciseness", "usefulness", "multi_turn_coherence",
        ]
        if parsed:
            print(f"score={parsed.get('score', '?')}")
            scores = {f: parsed.get(f) for f in score_fields}
            texts  = {f: parsed.get(f, "") for f in ["main_strengths", "main_weaknesses", "final_comment"]}
            jstatus = "ok"
        else:
            print(f"ERROR: {err}")
            scores  = {f: None for f in score_fields}
            texts   = {"main_strengths": "", "main_weaknesses": "", "final_comment": ""}
            jstatus = "error"

        jrow = {
            "conversation_id"     : conv_id,
            "phase"               : conv.get("phase", PHASE),
            "model_name"          : conv.get("model_name", ""),
            "model_type"          : conv.get("model_type", ""),
            "quantization"        : conv.get("quantization", ""),
            "hardware_profile"    : conv.get("hardware_profile", ""),
            "execution_device"    : conv.get("execution_device", ""),
            "subset_id"           : conv.get("subset_id", ""),
            "official_question_id": conv.get("official_question_id", ""),
            "original_category"   : conv.get("original_category", ""),
            "internal_category"   : conv.get("internal_category", ""),
            "repetition"          : conv.get("repetition", ""),
            **scores,
            **texts,
            "judge_model"         : JUDGE_MODEL,
            "judge_timestamp"     : ts,
            "judge_status"        : jstatus,
            "judge_error_message" : err or "",
            "judge_prompt_mode"   : "multi_turn",
            "reference_guided"    : ref_guided,
            "trial_id"            : conv.get("trial_id", ""),
            "combo_id"            : conv.get("combo_id", ""),
            "trial_stage"         : conv.get("trial_stage", ""),
        }
        judge_writer.write(jrow)
        results.append(jrow)
        existing_judge_ids.add(conv_id)

        if not dry_run:
            time.sleep(JUDGE_DELAY_S)

    return results


# ─── Stage execution ──────────────────────────────────────────────────────────

def run_stage(
    stage_name: str,
    trials: list[TrialParams],
    repetitions: int,
    questions: list[dict],
    hw: HardwareProfile,
    base_cfg: ExperimentConfig,
    opt_cfg: OptimizationConfig,
    model_name: str,
    quant: str,
    model_path: Path,
    device: str,
    out_dir: Path,
    api_key: str,
    references: Optional[dict],
    dry_run: bool,
    no_resume: bool,
    experiment_id: str,
) -> tuple[list[dict], list[dict]]:
    """
    Run one stage (stage_a or stage_b).
    Groups trials by load_key to minimize model reloads (P2).
    Returns (all_conv_rows, all_judge_rows).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    turn_writer  = ResultWriter(
        out_dir / "turn_results.csv",
        out_dir / "turn_results.jsonl",
        TURN_COLS_P2, no_resume,
    )
    conv_writer  = ResultWriter(
        out_dir / "conversation_results.csv",
        out_dir / "conversation_results.jsonl",
        CONV_COLS_P2, no_resume,
    )
    judge_writer = ResultWriter(
        out_dir / "judge_results.csv",
        out_dir / "judge_results.jsonl",
        JUDGE_COLS, no_resume,
    )
    existing_judge_ids: set = set(judge_writer.existing_ids)

    gpu_total_mb_val = _gpu_total_mb(hw)

    load_groups: dict[tuple, list[TrialParams]] = defaultdict(list)
    for t in trials:
        load_groups[t.load_key].append(t)

    all_conv_rows:  list[dict] = []
    all_judge_rows: list[dict] = []
    n_groups = len(load_groups)

    for g_idx, (load_key, group_trials) in enumerate(load_groups.items(), 1):
        n_ctx, n_batch, n_threads = load_key
        print(
            f"\n  [{stage_name.upper()}] Grupo {g_idx}/{n_groups}: "
            f"n_ctx={n_ctx} n_batch={n_batch} n_threads={n_threads} "
            f"({len(group_trials)} trial(s))"
        )

        # P9: baseline per unique load configuration
        print(f"    Midiendo baseline idle...")
        if dry_run:
            baseline = {"baseline_total_power_w": 5.0}
        else:
            try:
                baseline = medir_baseline_calibracion(
                    duracion_s=base_cfg.baseline_idle_seconds,
                    repetitions=base_cfg.baseline_repetitions,
                    hw=hw,
                )
            except Exception as e:
                print(f"    AVISO: baseline fallido ({e}), usando cero.")
                baseline = {"baseline_total_power_w": 0.0}
        print(f"    baseline_power_w={baseline.get('baseline_total_power_w', 0):.2f}")

        # P3: cooldown before model load
        if not dry_run and getattr(base_cfg, "cooldown_seconds", 0) > 0:
            print(f"    Cooldown {base_cfg.cooldown_seconds}s...")
            time.sleep(base_cfg.cooldown_seconds)

        # P2: load model outside tracker
        print(f"    Cargando {model_name}/{quant}...")
        llm = None if dry_run else cargar_modelo_fase2(
            model_path, device, hw, group_trials[0], base_cfg
        )
        if llm is None and not dry_run:
            print("    ERROR: modelo no cargado, saltando grupo.")
            continue

        # P3: warmup outside tracker
        if llm is not None:
            warmup_n = getattr(base_cfg, "warmup_runs", 1)
            print(f"    Warmup ({warmup_n} run(s))...")
            warmup_modelo(llm, warmup_n)

        for trial in group_trials:
            print(
                f"\n    Trial {trial.trial_id} [{trial.combo_id}]: "
                f"max_tokens={trial.max_tokens} temp={trial.temperature} "
                f"rp={trial.repeat_penalty} top_k={trial.top_k} seed={trial.seed}"
            )

            for rep in range(1, repetitions + 1):
                print(f"      Rep {rep}/{repetitions}: ", end="", flush=True)

                if rep > 1 and not dry_run and getattr(base_cfg, "cooldown_seconds", 0) > 0:
                    time.sleep(base_cfg.cooldown_seconds)

                conv_id_placeholder = stable_row_id(
                    EXPERIMENT_TAG, model_name, quant,
                    base_cfg.hardware_profile, device,
                    "multi_q_conv", "conv", rep,
                    str(trial.trial_id), stage_name,
                )

                new_convs: list[dict] = []
                for q_idx, question in enumerate(questions):
                    t1_prompt = build_prompt(model_name, [
                        {"role": "user", "content": question["turn1"]},
                    ])
                    r1 = medir_turno_fase2(
                        llm=llm,
                        prompt=t1_prompt,
                        original_question_text=question["turn1"],
                        turn=1,
                        question=question,
                        device=device,
                        model_name=model_name,
                        quant=quant,
                        model_path=model_path,
                        repetition=rep,
                        hw=hw,
                        baseline=baseline,
                        base_cfg=base_cfg,
                        trial=trial,
                        trial_stage=stage_name,
                        experiment_id=experiment_id,
                        conversation_id=conv_id_placeholder,
                        subset_idx=q_idx,
                        gpu_total_mb_val=gpu_total_mb_val,
                        dry_run=dry_run,
                    )
                    turn_writer.write(r1)

                    t2_prompt = build_prompt(model_name, [
                        {"role": "user",      "content": question["turn1"]},
                        {"role": "assistant", "content": r1.get("current_turn_response", "")},
                        {"role": "user",      "content": question["turn2"]},
                    ])
                    r2 = medir_turno_fase2(
                        llm=llm,
                        prompt=t2_prompt,
                        original_question_text=question["turn2"],
                        turn=2,
                        question=question,
                        device=device,
                        model_name=model_name,
                        quant=quant,
                        model_path=model_path,
                        repetition=rep,
                        hw=hw,
                        baseline=baseline,
                        base_cfg=base_cfg,
                        trial=trial,
                        trial_stage=stage_name,
                        experiment_id=experiment_id,
                        conversation_id=conv_id_placeholder,
                        subset_idx=q_idx,
                        gpu_total_mb_val=gpu_total_mb_val,
                        dry_run=dry_run,
                    )
                    turn_writer.write(r2)

                    conv_row = _conversation_row_fase2(r1, r2, experiment_id)
                    conv_writer.write(conv_row)
                    new_convs.append(conv_row)

                print(f"OK ({len(questions)} preguntas)")
                all_conv_rows.extend(new_convs)

            # Judge conversations for this trial
            trial_convs = [
                c for c in all_conv_rows
                if str(c.get("trial_id", "")) == str(trial.trial_id)
            ]
            print(f"    Evaluando {len(trial_convs)} convs con juez...")
            jrows = judge_conversations(
                trial_convs, api_key, references, dry_run,
                judge_writer, existing_judge_ids,
            )
            all_judge_rows.extend(jrows)

        if llm is not None:
            del llm
            gc.collect()

    turn_writer.close()
    conv_writer.close()
    judge_writer.close()
    return all_conv_rows, all_judge_rows


# ─── Trial metric aggregation ─────────────────────────────────────────────────

def aggregate_trial_metrics(
    conv_rows: list[dict],
    judge_rows: list[dict],
    baseline_quality: Optional[float] = None,
    drop_abs: float = 0.25,
    drop_rel: float = 3.0,
) -> list[dict]:
    """
    Aggregate per-conversation rows into per-trial summary dicts.

    Computes all metrics required by TRIALS_COLS including EDP, tokens/joule,
    quality_per_joule, CI95, and per-trial quality constraint fields.

    quality_scores_raw (list[float]) is stored in each dict for the non-inferiority
    t-test; callers must strip it via _strip_for_csv() before writing to CSV.
    """
    judge_by_conv: dict[str, dict] = {
        j["conversation_id"]: j
        for j in judge_rows
        if j.get("judge_status") == "ok"
    }

    by_trial: dict[str, dict] = {}
    for conv in conv_rows:
        tid = str(conv.get("trial_id", ""))
        if not tid:
            continue
        if tid not in by_trial:
            by_trial[tid] = {
                "trial_id"        : tid,
                "combo_id"        : conv.get("combo_id"),
                "trial_stage"     : conv.get("trial_stage"),
                "model_name"      : conv.get("model_name"),
                "quantization"    : conv.get("quantization"),
                "hardware_profile": conv.get("hardware_profile"),
                "execution_device": conv.get("execution_device"),
                "n_ctx"           : conv.get("n_ctx"),
                "max_tokens"      : conv.get("max_tokens"),
                "n_batch"         : conv.get("n_batch"),
                "n_threads"       : conv.get("n_threads"),
                "temperature"     : conv.get("temperature"),
                "top_p"           : conv.get("top_p"),
                "repeat_penalty"  : conv.get("repeat_penalty"),
                "top_k"           : conv.get("top_k"),
                "seed"            : conv.get("seed"),
                "_quality"            : [],
                "_energy_wh_1k_meas"  : [],
                "_energy_wh_1k_corr"  : [],
                "_energy_j_tok_meas"  : [],
                "_energy_j_tok_corr"  : [],
                "_tok_per_j_meas"     : [],
                "_tok_per_j_corr"     : [],
                "_energy_j_total_meas": [],
                "_energy_j_total_corr": [],
                "_edp_meas"           : [],
                "_edp_corr"           : [],
                "_latency"            : [],
                "_statuses"           : [],
                "_errors"             : [],
            }
        entry = by_trial[tid]

        conv_id = conv.get("conversation_id", "")
        j = judge_by_conv.get(conv_id)
        if j:
            q = _flt(j.get("score"))
            if q is not None:
                entry["_quality"].append(q)

        def _a(key: str, bucket: str) -> None:
            v = _flt(conv.get(key))
            if v is not None:
                entry[bucket].append(v)

        _a("total_energy_wh_per_1k_output_tokens_measured",  "_energy_wh_1k_meas")
        _a("total_energy_wh_per_1k_output_tokens_corrected", "_energy_wh_1k_corr")
        _a("total_energy_j_per_output_token_measured",        "_energy_j_tok_meas")
        _a("total_energy_j_per_output_token_corrected",       "_energy_j_tok_corr")
        _a("total_tokens_per_joule_measured",                 "_tok_per_j_meas")
        _a("total_tokens_per_joule_corrected",                "_tok_per_j_corr")
        _a("total_measured_energy_joules",                    "_energy_j_total_meas")
        _a("total_baseline_corrected_energy_joules",          "_energy_j_total_corr")
        _a("total_edp_joule_second_measured",                 "_edp_meas")
        _a("total_edp_joule_second_corrected",                "_edp_corr")
        _a("total_latency_seconds",                           "_latency")

        st = conv.get("status_conversation", "")
        if st:
            entry["_statuses"].append(str(st))
        err = conv.get("error_message_conversation", "")
        if err:
            entry["_errors"].append(str(err))

    def _mean(lst: list) -> Optional[float]:
        return float(np.mean(lst)) if lst else None

    def _div(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None or b == 0:
            return None
        return a / b

    def _pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
        if new is None or old is None or old == 0:
            return None
        return (new - old) / old * 100.0

    summaries: list[dict] = []
    for entry in by_trial.values():
        qs               = entry.pop("_quality")
        em1k             = entry.pop("_energy_wh_1k_meas")
        ec1k             = entry.pop("_energy_wh_1k_corr")
        ej_tok_m         = entry.pop("_energy_j_tok_meas")
        ej_tok_c         = entry.pop("_energy_j_tok_corr")
        tpj_m            = entry.pop("_tok_per_j_meas")
        tpj_c            = entry.pop("_tok_per_j_corr")
        ej_total_m       = entry.pop("_energy_j_total_meas")
        ej_total_c       = entry.pop("_energy_j_total_corr")
        edp_m            = entry.pop("_edp_meas")
        edp_c            = entry.pop("_edp_corr")
        lt               = entry.pop("_latency")
        statuses         = entry.pop("_statuses")
        errors           = entry.pop("_errors")

        n    = len(qs)
        mean_q = float(np.mean(qs)) if qs else None
        std_q  = float(np.std(qs, ddof=1)) if n > 1 else None
        ci95_q = (1.96 * std_q / (n ** 0.5)) if std_q is not None and n > 1 else None

        mean_ej_m = _mean(ej_total_m)
        mean_ej_c = _mean(ej_total_c)

        # trial_status: "ok" if all conversations succeeded and judge results present
        n_ok = statuses.count("ok")
        if qs and n_ok == len(statuses):
            t_status = "ok"
        elif not qs:
            t_status = "no_judge"
        elif n_ok == 0:
            t_status = "failed"
        else:
            t_status = "partial"
        error_msg = errors[0] if errors else ""

        # Quality constraint — populated when baseline_quality is known
        if baseline_quality is not None and mean_q is not None:
            drop_a = baseline_quality - mean_q
            drop_r = (drop_a / baseline_quality * 100.0) if baseline_quality else None
            passed = (
                mean_q >= (baseline_quality - drop_abs)
                and mean_q >= baseline_quality * (1.0 - drop_rel / 100.0)
            )
        else:
            drop_a = None
            drop_r = None
            passed = None

        summaries.append({
            **entry,
            "n_reps"                                     : n,
            "mean_quality_score"                         : mean_q,
            "std_quality_score"                          : std_q,
            "ci95_quality_score"                         : ci95_q,
            "quality_scores_raw"                         : qs,   # stripped before CSV write
            "mean_energy_wh_per_1k_output_tokens_measured" : _mean(em1k),
            "mean_energy_wh_per_1k_output_tokens_corrected": _mean(ec1k),
            "mean_energy_j_per_output_token_measured"    : _mean(ej_tok_m),
            "mean_energy_j_per_output_token_corrected"   : _mean(ej_tok_c),
            "mean_tokens_per_joule_measured"             : _mean(tpj_m),
            "mean_tokens_per_joule_corrected"            : _mean(tpj_c),
            "mean_latency_seconds"                       : _mean(lt),
            "mean_edp_joule_second_measured"             : _mean(edp_m),
            "mean_edp_joule_second_corrected"            : _mean(edp_c),
            "quality_per_joule_measured"                 : _div(mean_q, mean_ej_m),
            "quality_per_joule_corrected"                : _div(mean_q, mean_ej_c),
            "quality_drop_absolute"                      : drop_a,
            "quality_drop_relative_percent"              : drop_r,
            "quality_constraint_passed"                  : passed,
            "trial_status"                               : t_status,
            "error_message"                              : error_msg,
        })

    summaries.sort(key=lambda r: r.get("mean_quality_score") or 0, reverse=True)
    return summaries


def _strip_for_csv(metrics: list[dict]) -> list[dict]:
    """Return copies of metric dicts with quality_scores_raw removed (not CSV-safe)."""
    return [{k: v for k, v in m.items() if k != "quality_scores_raw"} for m in metrics]


# ─── Pareto and candidate selection ──────────────────────────────────────────

def _pareto_front_2d(items: list[dict]) -> list[dict]:
    """Pareto front: maximize quality, minimize energy."""
    if not items:
        return []

    def _q(m: dict) -> float:
        return _flt(m.get("mean_quality_score")) or 0.0

    def _e(m: dict) -> float:
        v = _flt(m.get("mean_energy_wh_per_1k_output_tokens_corrected")) or \
            _flt(m.get("mean_energy_wh_per_1k_output_tokens_measured"))
        return v if v is not None else float("inf")

    front: list[dict] = []
    for i, a in enumerate(items):
        qa, ea = _q(a), _e(a)
        dominated = any(
            _q(b) >= qa and _e(b) <= ea and (_q(b) > qa or _e(b) < ea)
            for j, b in enumerate(items) if j != i
        )
        if not dominated:
            front.append(a)
    return front


def select_stage_b_candidates(
    stage_a_metrics: list[dict],
    baseline_quality: float,
    opt_cfg: OptimizationConfig,
) -> list[TrialParams]:
    drop_abs = opt_cfg.quality_constraint_max_drop_absolute
    drop_rel = opt_cfg.quality_constraint_max_drop_relative_percent
    min_q = max(
        baseline_quality - drop_abs,
        baseline_quality * (1.0 - drop_rel / 100.0),
    )

    eligible = [
        m for m in stage_a_metrics
        if (_flt(m.get("mean_quality_score")) or 0.0) >= min_q
    ]
    if not eligible:
        print(
            f"  AVISO: ningun trial Stage A cumple P22 (min_q={min_q:.2f}). "
            "Tomando top-N por calidad."
        )
        eligible = [m for m in stage_a_metrics if m.get("mean_quality_score") is not None]

    front = _pareto_front_2d(eligible) or eligible
    top_n = opt_cfg.stage_a_pareto_top_n
    if len(front) > top_n:
        front.sort(key=lambda r: _flt(r.get("mean_quality_score")) or 0, reverse=True)
        front = front[:top_n]

    print(
        f"  Stage B: {len(front)} candidato(s) seleccionado(s) "
        f"de {len(stage_a_metrics)} trials Stage A."
    )

    param_keys = [
        "n_ctx", "max_tokens", "n_batch", "n_threads",
        "temperature", "top_p", "repeat_penalty", "top_k", "seed",
    ]
    candidates: list[TrialParams] = []
    for i, m in enumerate(front):
        try:
            candidates.append(TrialParams(
                trial_id=int(m.get("trial_id") or i + 1000),
                n_ctx=int(m["n_ctx"]),
                max_tokens=int(m["max_tokens"]),
                n_batch=int(m["n_batch"]),
                n_threads=int(m["n_threads"]),
                temperature=float(m["temperature"]),
                top_p=float(m["top_p"]),
                repeat_penalty=float(m["repeat_penalty"]),
                top_k=int(m["top_k"]),
                seed=int(m["seed"]),
            ))
        except (KeyError, TypeError, ValueError) as e:
            print(f"  AVISO: candidato {i} ignorado: {e}")
    return candidates


# ─── P22 winner selection ─────────────────────────────────────────────────────

def _non_inferiority_test(
    scores: list[float],
    baseline_quality: float,
    drop_abs: float,
) -> dict:
    """
    One-sample, one-sided t-test for non-inferiority.

    H0 (inferior): μ_winner ≤ baseline_quality - drop_abs
    H1 (non-inferior): μ_winner > baseline_quality - drop_abs

    Requires scipy.stats. Returns a dict with test details and a plain-language
    conclusion suitable for inclusion in optimization_report.json.
    """
    n = len(scores)
    if n < 2:
        return {
            "method"               : "one_sample_t_test_one_sided",
            "non_inferiority_bound": baseline_quality - drop_abs,
            "n"                    : n,
            "conclusion"           : "inconclusive_insufficient_data",
        }
    result = scipy.stats.ttest_1samp(
        scores,
        popmean=baseline_quality - drop_abs,
        alternative="greater",
    )
    t_stat = float(result.statistic)
    p_val  = float(result.pvalue)
    if p_val < 0.05:
        conclusion = "non_inferior"
    elif p_val < 0.10:
        conclusion = "inconclusive_marginal"
    else:
        conclusion = "inconclusive"
    return {
        "method"               : "one_sample_t_test_one_sided",
        "non_inferiority_bound": baseline_quality - drop_abs,
        "null_hypothesis"      : f"mean_quality <= {baseline_quality - drop_abs:.4f}",
        "alternative"          : f"mean_quality > {baseline_quality - drop_abs:.4f}",
        "n"                    : n,
        "mean_scores"          : float(np.mean(scores)),
        "std_scores"           : float(np.std(scores, ddof=1)),
        "t_statistic"          : t_stat,
        "p_value_one_sided"    : p_val,
        "alpha"                : 0.05,
        "conclusion"           : conclusion,
    }


def select_winner_p22(
    stage_b_metrics: list[dict],
    baseline: dict,
    opt_cfg: OptimizationConfig,
) -> Optional[dict]:
    """
    P22: winner must satisfy the quality constraint vs. Phase 1 baseline.

    A candidate is valid only if quality_constraint_passed == True, meaning:
      mean_quality_score >= baseline_quality - max_allowed_drop_absolute
      AND
      relative_quality_drop_percent <= max_allowed_drop_relative_percent

    Among valid candidates, selects the one with minimum energy (corrected
    preferred; falls back to measured). Attaches a non-inferiority t-test
    result to the returned dict.
    """
    bl_quality = _flt(baseline.get("mean_quality_score")) or 0.0
    drop_abs   = opt_cfg.quality_constraint_max_drop_absolute
    drop_rel   = opt_cfg.quality_constraint_max_drop_relative_percent
    min_q      = max(bl_quality - drop_abs, bl_quality * (1.0 - drop_rel / 100.0))

    # Log constraint check for every Stage B candidate
    for m in stage_b_metrics:
        mq  = _flt(m.get("mean_quality_score"))
        qcp = m.get("quality_constraint_passed")
        tid = m.get("trial_id")
        if qcp is None and mq is not None:
            # pre-computed field absent (e.g. loaded from CSV without baseline) — recompute
            qcp = mq >= min_q
            m["quality_constraint_passed"] = qcp
        status = "PASS" if qcp else "FAIL"
        print(
            f"  trial {tid}: quality={mq:.2f}  "
            f"drop={bl_quality - (mq or 0):.2f}  constraint={status}"
        )

    qualified = [m for m in stage_b_metrics if m.get("quality_constraint_passed") is True]
    if not qualified:
        print(
            f"  AVISO P22: ningún candidato Stage B cumple "
            f"calidad >= {min_q:.2f} (baseline={bl_quality:.2f}). Sin ganador."
        )
        return None

    def _e(m: dict) -> float:
        v = _flt(m.get("mean_energy_wh_per_1k_output_tokens_corrected")) or \
            _flt(m.get("mean_energy_wh_per_1k_output_tokens_measured"))
        return v if v is not None else float("inf")

    winner = min(qualified, key=_e)

    # Attach non-inferiority test result using raw quality scores if available
    scores = winner.get("quality_scores_raw") or []
    winner["non_inferiority_test"] = _non_inferiority_test(
        scores, bl_quality, drop_abs
    )
    return winner


# ─── Report ───────────────────────────────────────────────────────────────────

def generate_report(
    winner: Optional[dict],
    baseline: dict,
    stage_a_metrics: list[dict],
    stage_b_metrics: list[dict],
    out_dir: Path,
    experiment_id: str,
    opt_cfg: Optional[OptimizationConfig] = None,
) -> Path:
    def _pct(new: Optional[float], old: Optional[float]) -> Optional[float]:
        if new is None or old is None or old == 0:
            return None
        return (new - old) / old * 100.0

    bl_q   = _flt(baseline.get("mean_quality_score"))
    bl_em  = _flt(
        baseline.get("mean_total_energy_wh_per_1k_output_tokens_corrected")
        or baseline.get("mean_total_energy_wh_per_1k_output_tokens_measured")
    )
    bl_lat = _flt(baseline.get("mean_total_latency_seconds"))
    bl_tpj = _flt(
        baseline.get("mean_total_tokens_per_joule_corrected")
        or baseline.get("mean_total_tokens_per_joule_measured")
    )
    bl_edp = _flt(
        baseline.get("mean_total_edp_joule_second_corrected")
        or baseline.get("mean_total_edp_joule_second_measured")
    )
    bl_qpj = _flt(
        baseline.get("mean_total_quality_per_joule_corrected")
        or baseline.get("mean_total_quality_per_joule_measured")
        or baseline.get("quality_per_joule_corrected")
        or baseline.get("quality_per_joule_measured")
    )

    report: dict = {
        "experiment_id"  : experiment_id,
        "phase"          : PHASE,
        "timestamp_utc"  : datetime.now(timezone.utc).isoformat(),
        "baseline"       : {
            "model_name"           : baseline.get("model_name"),
            "quantization"         : baseline.get("quantization"),
            "hardware_profile"     : baseline.get("hardware_profile"),
            "execution_device"     : baseline.get("execution_device"),
            "mean_quality_score"   : bl_q,
            "mean_energy_wh_per_1k": bl_em,
            "mean_latency_s"       : bl_lat,
            "mean_tokens_per_joule": bl_tpj,
            "mean_edp_joule_second": bl_edp,
            "quality_per_joule"    : bl_qpj,
            "pareto_efficient"     : baseline.get("pareto_efficient"),
        },
        "quality_constraint": {
            "max_allowed_drop_absolute"        : opt_cfg.quality_constraint_max_drop_absolute if opt_cfg else None,
            "max_allowed_drop_relative_percent": opt_cfg.quality_constraint_max_drop_relative_percent if opt_cfg else None,
            "min_quality_threshold"            : (
                max(
                    (bl_q or 0) - opt_cfg.quality_constraint_max_drop_absolute,
                    (bl_q or 0) * (1.0 - opt_cfg.quality_constraint_max_drop_relative_percent / 100.0),
                ) if (bl_q is not None and opt_cfg is not None) else None
            ),
        },
        "validation_split": {
            "enabled" : opt_cfg.validation_split_enabled if opt_cfg else False,
            "strategy": opt_cfg.validation_split_strategy if opt_cfg else None,
        },
        "stage_a_summary": {
            "n_trials": len(stage_a_metrics),
            "top3"    : [
                {k: v for k, v in m.items() if k != "quality_scores_raw"}
                for m in stage_a_metrics[:3]
            ],
        },
        "stage_b_summary": {
            "n_candidates"        : len(stage_b_metrics),
            "n_constraint_passed" : sum(1 for m in stage_b_metrics if m.get("quality_constraint_passed") is True),
            "results"             : [
                {k: v for k, v in m.items() if k != "quality_scores_raw"}
                for m in stage_b_metrics
            ],
        },
        "winner"         : None,
        "improvement"    : None,
    }

    if winner:
        w_q   = _flt(winner.get("mean_quality_score"))
        w_em  = _flt(
            winner.get("mean_energy_wh_per_1k_output_tokens_corrected")
            or winner.get("mean_energy_wh_per_1k_output_tokens_measured")
        )
        w_lat  = _flt(winner.get("mean_latency_seconds"))
        w_tpj  = _flt(
            winner.get("mean_tokens_per_joule_corrected")
            or winner.get("mean_tokens_per_joule_measured")
        )
        w_edp  = _flt(
            winner.get("mean_edp_joule_second_corrected")
            or winner.get("mean_edp_joule_second_measured")
        )
        w_qpj  = _flt(
            winner.get("quality_per_joule_corrected")
            or winner.get("quality_per_joule_measured")
        )
        report["winner"] = {
            "trial_id"                    : winner.get("trial_id"),
            "combo_id"                    : winner.get("combo_id"),
            "n_ctx"                       : winner.get("n_ctx"),
            "max_tokens"                  : winner.get("max_tokens"),
            "n_batch"                     : winner.get("n_batch"),
            "n_threads"                   : winner.get("n_threads"),
            "temperature"                 : winner.get("temperature"),
            "top_p"                       : winner.get("top_p"),
            "repeat_penalty"              : winner.get("repeat_penalty"),
            "top_k"                       : winner.get("top_k"),
            "seed"                        : winner.get("seed"),
            "mean_quality_score"          : w_q,
            # quality_drop_absolute = baseline_quality_score - optimized_quality_score
            "quality_drop_absolute"       : winner.get("quality_drop_absolute"),
            # quality_drop_relative_percent = 100 * quality_drop_absolute / baseline_quality_score
            "quality_drop_relative_percent": winner.get("quality_drop_relative_percent"),
            "quality_constraint_passed"   : winner.get("quality_constraint_passed"),
            "mean_energy_wh_per_1k"       : w_em,
            "mean_latency_s"              : w_lat,
            "mean_tokens_per_joule"       : w_tpj,
            "mean_edp_joule_second"       : w_edp,
            "quality_per_joule"           : w_qpj,
            "pareto_efficient"            : winner.get("pareto_efficient"),
            "non_inferiority_test"        : winner.get("non_inferiority_test"),
        }
        report["improvement"] = {
            # quality_drop_absolute = baseline_quality_score - optimized_quality_score
            "quality_drop_absolute"               : winner.get("quality_drop_absolute"),
            # quality_drop_relative_percent = 100 * quality_drop_absolute / baseline_quality_score
            "quality_drop_relative_percent"       : winner.get("quality_drop_relative_percent"),
            # energy_reduction_percent = 100 * (baseline_energy - optimized_energy) / baseline_energy
            "energy_reduction_percent"            : winner.get("energy_reduction_percent"),
            # latency_reduction_percent = 100 * (baseline_latency - optimized_latency) / baseline_latency
            "latency_reduction_percent"           : winner.get("latency_reduction_percent"),
            # tokens_per_joule_improvement_percent = 100 * (opt_tpj - bl_tpj) / bl_tpj
            "tokens_per_joule_improvement_percent": winner.get("tokens_per_joule_improvement_percent"),
            # quality_per_joule_improvement_percent = 100 * (opt_qpj - bl_qpj) / bl_qpj
            "quality_per_joule_improvement_percent": winner.get("quality_per_joule_improvement_percent"),
            # edp_reduction_percent = 100 * (baseline_edp - optimized_edp) / baseline_edp
            "edp_reduction_percent"               : winner.get("edp_reduction_percent"),
            "quality_constraint_passed"           : True,
        }
    else:
        report["improvement"] = {"quality_constraint_passed": False}

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "optimization_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"  Reporte guardado: {report_path}")
    return report_path


# ─── Baseline annotation and Pareto flag ─────────────────────────────────────

def _annotate_with_baseline(
    metrics: list[dict],
    baseline: dict,
    baseline_config_id: str,
    phase: str = PHASE,
) -> list[dict]:
    """
    Inject baseline-comparison fields and pareto_efficient flag into each metric dict.

    Adds: phase, baseline_configuration_id, model_type,
          energy_reduction_percent, latency_reduction_percent,
          tokens_per_joule_improvement_percent, pareto_efficient.

    Mutates dicts in-place and returns the same list.
    """
    bl_e  = _flt(
        baseline.get("mean_total_energy_wh_per_1k_output_tokens_corrected")
        or baseline.get("mean_total_energy_wh_per_1k_output_tokens_measured")
    )
    bl_lat = _flt(baseline.get("mean_total_latency_seconds"))
    bl_tpj = _flt(
        baseline.get("mean_total_tokens_per_joule_corrected")
        or baseline.get("mean_total_tokens_per_joule_measured")
    )
    bl_edp = _flt(
        baseline.get("mean_total_edp_joule_second_corrected")
        or baseline.get("mean_total_edp_joule_second_measured")
    )
    bl_qpj = _flt(
        baseline.get("mean_total_quality_per_joule_corrected")
        or baseline.get("mean_total_quality_per_joule_measured")
        or baseline.get("quality_per_joule_corrected")
        or baseline.get("quality_per_joule_measured")
    )

    def _pct_reduction(new: Optional[float], old: Optional[float]) -> Optional[float]:
        if new is None or old is None or old == 0:
            return None
        return (old - new) / old * 100.0   # positive = improvement (reduction)

    def _pct_increase(new: Optional[float], old: Optional[float]) -> Optional[float]:
        if new is None or old is None or old == 0:
            return None
        return (new - old) / old * 100.0   # positive = improvement (increase)

    # Mark Pareto-efficient trials (maximize quality, minimize energy)
    def _q(m: dict) -> float:
        return _flt(m.get("mean_quality_score")) or 0.0

    def _e(m: dict) -> float:
        v = (
            _flt(m.get("mean_energy_wh_per_1k_output_tokens_corrected"))
            or _flt(m.get("mean_energy_wh_per_1k_output_tokens_measured"))
        )
        return v if v is not None else float("inf")

    pareto_ids: set[str] = set()
    for i, a in enumerate(metrics):
        qa, ea = _q(a), _e(a)
        dominated = any(
            _q(b) >= qa and _e(b) <= ea and (_q(b) > qa or _e(b) < ea)
            for j, b in enumerate(metrics) if j != i
        )
        if not dominated:
            pareto_ids.add(str(a.get("trial_id", "")))

    for m in metrics:
        m_e = (
            _flt(m.get("mean_energy_wh_per_1k_output_tokens_corrected"))
            or _flt(m.get("mean_energy_wh_per_1k_output_tokens_measured"))
        )
        m_lat = _flt(m.get("mean_latency_seconds"))
        m_tpj = (
            _flt(m.get("mean_tokens_per_joule_corrected"))
            or _flt(m.get("mean_tokens_per_joule_measured"))
        )
        m_edp = (
            _flt(m.get("mean_edp_joule_second_corrected"))
            or _flt(m.get("mean_edp_joule_second_measured"))
        )
        m_qpj = (
            _flt(m.get("quality_per_joule_corrected"))
            or _flt(m.get("quality_per_joule_measured"))
        )
        m["phase"]                                 = phase
        m["baseline_configuration_id"]            = baseline_config_id
        m["model_type"]                            = MODEL_TYPES.get(
            str(m.get("model_name", "")), "unknown"
        )
        # Formulas (positive = improvement):
        # energy_reduction_percent  = 100 * (bl_e  - m_e  ) / bl_e
        # latency_reduction_percent = 100 * (bl_lat - m_lat) / bl_lat
        # tokens_per_joule_improvement_percent = 100 * (m_tpj - bl_tpj) / bl_tpj
        # edp_reduction_percent     = 100 * (bl_edp - m_edp) / bl_edp
        # quality_per_joule_improvement_percent = 100 * (m_qpj - bl_qpj) / bl_qpj
        m["energy_reduction_percent"]                = _pct_reduction(m_e,   bl_e)
        m["latency_reduction_percent"]               = _pct_reduction(m_lat, bl_lat)
        m["tokens_per_joule_improvement_percent"]    = _pct_increase(m_tpj,  bl_tpj)
        m["edp_reduction_percent"]                   = _pct_reduction(m_edp, bl_edp)
        m["quality_per_joule_improvement_percent"]   = _pct_increase(m_qpj,  bl_qpj)
        m["pareto_efficient"]                        = str(m.get("trial_id", "")) in pareto_ids

    return metrics


# ─── Output writers ───────────────────────────────────────────────────────────

def _row_for_output(m: dict) -> dict:
    """Project a metric dict onto TRIALS_COLS, filling missing keys with empty string."""
    return {col: m.get(col, "") for col in TRIALS_COLS}


def write_optimization_trials(
    metrics: list[dict],
    out_dir: Path,
) -> tuple[Path, Path]:
    """Write optimization_trials.csv and optimization_trials.jsonl."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = out_dir / "optimization_trials.csv"
    jsonl_path = out_dir / "optimization_trials.jsonl"

    rows = [_row_for_output(m) for m in metrics]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=TRIALS_COLS)
        w.writeheader()
        w.writerows(rows)

    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")

    print(f"  optimization_trials     : {csv_path}")
    print(f"  optimization_trials     : {jsonl_path}")
    return csv_path, jsonl_path


def write_optimization_summary(
    metrics: list[dict],
    out_dir: Path,
) -> Path:
    """
    Write optimization_summary.csv — quality-constraint-passing trials only,
    sorted by energy_reduction_percent DESC then quality_per_joule_corrected DESC.
    Uses TRIALS_COLS column order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "optimization_summary.csv"

    passing = [m for m in metrics if m.get("quality_constraint_passed") is True]
    if not passing:
        passing = list(metrics)   # fallback: include all if none pass

    passing.sort(
        key=lambda r: (
            r.get("energy_reduction_percent") or 0,
            r.get("quality_per_joule_corrected") or 0,
        ),
        reverse=True,
    )

    rows = [_row_for_output(m) for m in passing]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=TRIALS_COLS)
        w.writeheader()
        w.writerows(rows)

    print(f"  optimization_summary    : {path}")
    return path


def write_pareto_front(
    metrics: list[dict],
    out_dir: Path,
) -> Path:
    """Write optimization_pareto_front.csv — only pareto_efficient=True trials."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "optimization_pareto_front.csv"

    front = [m for m in metrics if m.get("pareto_efficient") is True]
    front.sort(key=lambda r: r.get("mean_quality_score") or 0, reverse=True)

    rows = [_row_for_output(m) for m in front]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=TRIALS_COLS)
        w.writeheader()
        w.writerows(rows)

    print(f"  optimization_pareto_front: {path}")
    return path


def write_best_configuration(
    winner: dict,
    out_dir: Path,
    experiment_id: str,
) -> tuple[Path, Path]:
    """Write best_optimized_configuration.yaml and .json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = out_dir / "best_optimized_configuration.yaml"
    json_path = out_dir / "best_optimized_configuration.json"

    cfg: dict = {
        "experiment_id"   : experiment_id,
        "phase"           : winner.get("phase", PHASE),
        "baseline_id"     : winner.get("baseline_configuration_id", ""),
        "model": {
            "model_name"  : winner.get("model_name"),
            "model_type"  : winner.get("model_type"),
            "quantization": winner.get("quantization"),
        },
        "inference": {
            "n_ctx"        : winner.get("n_ctx"),
            "max_tokens"   : winner.get("max_tokens"),
            "n_batch"      : winner.get("n_batch"),
            "n_threads"    : winner.get("n_threads"),
            "temperature"  : winner.get("temperature"),
            "top_p"        : winner.get("top_p"),
            "repeat_penalty": winner.get("repeat_penalty"),
            "top_k"        : winner.get("top_k"),
            "seed"         : winner.get("seed"),
        },
        "performance": {
            "mean_quality_score"                     : winner.get("mean_quality_score"),
            "std_quality_score"                      : winner.get("std_quality_score"),
            "ci95_quality_score"                     : winner.get("ci95_quality_score"),
            "mean_energy_wh_per_1k_output_tokens_corrected":
                winner.get("mean_energy_wh_per_1k_output_tokens_corrected"),
            "mean_energy_wh_per_1k_output_tokens_measured":
                winner.get("mean_energy_wh_per_1k_output_tokens_measured"),
            "quality_per_joule_corrected"            : winner.get("quality_per_joule_corrected"),
            "mean_latency_seconds"                   : winner.get("mean_latency_seconds"),
            "mean_edp_joule_second_corrected"        : winner.get("mean_edp_joule_second_corrected"),
        },
        "quality_constraint": {
            "quality_drop_absolute"           : winner.get("quality_drop_absolute"),
            "quality_drop_relative_percent"   : winner.get("quality_drop_relative_percent"),
            "quality_constraint_passed"       : winner.get("quality_constraint_passed"),
        },
        "improvement_vs_baseline": {
            "energy_reduction_percent"              : winner.get("energy_reduction_percent"),
            "latency_reduction_percent"             : winner.get("latency_reduction_percent"),
            "tokens_per_joule_improvement_percent"  : winner.get("tokens_per_joule_improvement_percent"),
        },
        "pareto_efficient": winner.get("pareto_efficient"),
        "non_inferiority_test": winner.get("non_inferiority_test"),
    }

    with open(yaml_path, "w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, default=str)

    print(f"  best_config (yaml)      : {yaml_path}")
    print(f"  best_config (json)      : {json_path}")
    return yaml_path, json_path


def write_optimization_report_md(
    winner: Optional[dict],
    baseline: dict,
    stage_a_metrics: list[dict],
    stage_b_metrics: list[dict],
    opt_cfg: OptimizationConfig,
    out_dir: Path,
    experiment_id: str,
    n_tuning_prompts: Optional[int] = None,
    n_validation_prompts: Optional[int] = None,
) -> Path:
    """Write optimization_report.md — human-readable Markdown summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "optimization_report.md"

    def _fmt(v: object, decimals: int = 4) -> str:
        if v is None or v == "":
            return "N/A"
        try:
            return f"{float(v):.{decimals}f}"
        except (TypeError, ValueError):
            return str(v)

    def _bool(v: object) -> str:
        if v is True:
            return "Yes"
        if v is False:
            return "No"
        return "N/A"

    bl_q   = _flt(baseline.get("mean_quality_score"))
    bl_e   = _flt(
        baseline.get("mean_total_energy_wh_per_1k_output_tokens_corrected")
        or baseline.get("mean_total_energy_wh_per_1k_output_tokens_measured")
    )
    bl_lat = _flt(baseline.get("mean_total_latency_seconds"))
    bl_tpj = _flt(
        baseline.get("mean_total_tokens_per_joule_corrected")
        or baseline.get("mean_total_tokens_per_joule_measured")
    )
    bl_edp = _flt(
        baseline.get("mean_total_edp_joule_second_corrected")
        or baseline.get("mean_total_edp_joule_second_measured")
    )
    bl_qpj = _flt(
        baseline.get("mean_total_quality_per_joule_corrected")
        or baseline.get("mean_total_quality_per_joule_measured")
        or baseline.get("quality_per_joule_corrected")
        or baseline.get("quality_per_joule_measured")
    )
    bl_pareto = str(baseline.get("pareto_efficient", "N/A"))

    min_q = max(
        (bl_q or 0) - opt_cfg.quality_constraint_max_drop_absolute,
        (bl_q or 0) * (1.0 - opt_cfg.quality_constraint_max_drop_relative_percent / 100.0),
    ) if bl_q is not None else None

    n_pass_b = sum(1 for m in stage_b_metrics if m.get("quality_constraint_passed") is True)

    lines: list[str] = [
        "# GREEN-IA Phase 2 Optimization Report",
        "",
        f"**Experiment ID:** `{experiment_id}`  ",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "---",
        "",
        "## Baseline Configuration",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| Config ID | `{baseline.get('config_id', 'N/A')}` |",
        f"| Model | {baseline.get('model_name', 'N/A')} / {baseline.get('quantization', 'N/A')} |",
        f"| Hardware | {baseline.get('hardware_profile', 'N/A')} / {baseline.get('execution_device', 'N/A')} |",
        f"| Mean quality score | {_fmt(bl_q, 3)} |",
        f"| Mean energy (Wh/1k tokens, corrected) | {_fmt(bl_e, 6)} |",
        f"| Mean latency (s) | {_fmt(bl_lat, 3)} |",
        f"| Mean tokens per Joule (corrected) | {_fmt(bl_tpj, 4)} |",
        f"| Mean quality per Joule (corrected) | {_fmt(bl_qpj, 6)} |",
        f"| Mean EDP (J·s, corrected) | {_fmt(bl_edp, 6)} |",
        f"| Pareto status (Phase 1) | {bl_pareto} |",
        "",
        "---",
        "",
        "## Quality Constraint (P22)",
        "",
        f"A configuration is valid only if **both** conditions hold:",
        "",
        f"- `mean_quality_score >= {_fmt(bl_q, 3)} - {opt_cfg.quality_constraint_max_drop_absolute}`"
        f" → must be ≥ **{_fmt(min_q, 3)}**",
        f"- `quality_drop_relative_percent <= {opt_cfg.quality_constraint_max_drop_relative_percent}%`",
        "",
        "Configurations that fail either condition have `quality_constraint_passed = false`"
        " and are excluded from winner selection.",
        "",
        "---",
        "",
        "## Search Strategy",
        "",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| Method | {opt_cfg.search_method} |",
        f"| Max trials | {opt_cfg.max_trials} |",
        f"| Stage A reps | {opt_cfg.repetitions_per_trial} |",
        f"| Stage B reps | {opt_cfg.validation_repetitions} |",
        f"| Stage A → B candidates | {opt_cfg.stage_a_pareto_top_n} |",
        f"| Fixed: temperature | {opt_cfg.fixed_temperature} |",
        f"| Fixed: top_p | {opt_cfg.fixed_top_p} |",
        f"| Fixed: seed | {opt_cfg.fixed_seed} |",
        f"| Validation split | "
        f"{'enabled (' + opt_cfg.validation_split_strategy + ')' if opt_cfg.validation_split_enabled else 'disabled'} |",
        "",
        "**Search space:**",
        "",
    ]
    for param, vals in opt_cfg.parameter_space.items():
        lines.append(f"- `{param}`: {vals}")
    lines += [
        "",
        "---",
        "",
        "## Stage A Results",
        "",
        f"Trials evaluated: **{len(stage_a_metrics)}**"
        + (f"  — prompts used: {n_tuning_prompts} (tuning split)"
           if opt_cfg.validation_split_enabled and n_tuning_prompts is not None else ""),
        "",
    ]
    if stage_a_metrics:
        lines += [
            "| trial_id | quality | energy (Wh/1k, corr) | constraint |",
            "|---|---|---|---|",
        ]
        for m in stage_a_metrics[:10]:
            lines.append(
                f"| {m.get('trial_id','')} "
                f"| {_fmt(m.get('mean_quality_score'), 3)} "
                f"| {_fmt(m.get('mean_energy_wh_per_1k_output_tokens_corrected'), 6)} "
                f"| {_bool(m.get('quality_constraint_passed'))} |"
            )
        if len(stage_a_metrics) > 10:
            lines.append(f"| ... | _{len(stage_a_metrics) - 10} more_ | | |")
    lines += [
        "",
        "---",
        "",
        "## Stage B Results",
        "",
        f"Candidates validated: **{len(stage_b_metrics)}**"
        + (f"  — prompts used: {n_validation_prompts} (validation split)"
           if opt_cfg.validation_split_enabled and n_validation_prompts is not None else ""),
        "  ",
        f"Constraint passed: **{n_pass_b} / {len(stage_b_metrics)}**",
        "",
    ]
    if stage_b_metrics:
        lines += [
            "| trial_id | quality | ±CI95 | energy (Wh/1k, corr) | energy Δ% | constraint | Pareto |",
            "|---|---|---|---|---|---|---|",
        ]
        for m in stage_b_metrics:
            e_pct = m.get("energy_reduction_percent")
            e_pct_str = f"{e_pct:+.1f}%" if e_pct is not None else "N/A"
            lines.append(
                f"| {m.get('trial_id','')} "
                f"| {_fmt(m.get('mean_quality_score'), 3)} "
                f"| ±{_fmt(m.get('ci95_quality_score'), 3)} "
                f"| {_fmt(m.get('mean_energy_wh_per_1k_output_tokens_corrected'), 6)} "
                f"| {e_pct_str} "
                f"| {_bool(m.get('quality_constraint_passed'))} "
                f"| {_bool(m.get('pareto_efficient'))} |"
            )
    lines += ["", "---", ""]

    if winner:
        ni  = winner.get("non_inferiority_test") or {}
        lines += [
            "## Best Optimized Configuration",
            "",
            f"**Trial ID:** `{winner.get('trial_id')}`  ",
            f"**Combo ID:** `{winner.get('combo_id')}`",
            "",
            "### Inference Parameters",
            "",
            f"| Parameter | Value |",
            f"|---|---|",
            f"| n_ctx | {winner.get('n_ctx')} |",
            f"| max_tokens | {winner.get('max_tokens')} |",
            f"| n_batch | {winner.get('n_batch')} |",
            f"| n_threads | {winner.get('n_threads')} |",
            f"| temperature | {winner.get('temperature')} |",
            f"| top_p | {winner.get('top_p')} |",
            f"| repeat_penalty | {winner.get('repeat_penalty')} |",
            f"| top_k | {winner.get('top_k')} |",
            f"| seed | {winner.get('seed')} |",
            "",
            "### Performance vs Baseline",
            "",
            "Formulas (positive Δ = improvement):",
            "- `energy_reduction_percent = 100 × (baseline_energy − optimized_energy) / baseline_energy`",
            "- `latency_reduction_percent = 100 × (baseline_latency − optimized_latency) / baseline_latency`",
            "- `tokens_per_joule_improvement_percent = 100 × (opt_tpj − bl_tpj) / bl_tpj`",
            "- `quality_drop_absolute = baseline_quality_score − optimized_quality_score`",
            "- `quality_drop_relative_percent = 100 × quality_drop_absolute / baseline_quality_score`",
            "",
            f"| Metric | Baseline (Phase 1) | Optimized (Phase 2) | Δ |",
            f"|---|---|---|---|",
            f"| Mean quality score | {_fmt(bl_q, 3)} | {_fmt(winner.get('mean_quality_score'), 3)}"
            f" | −{_fmt(winner.get('quality_drop_absolute'), 3)} pts |",
            f"| Quality drop (relative) | — | — | {_fmt(winner.get('quality_drop_relative_percent'), 2)}% ↓ |",
            f"| Energy (Wh/1k tokens, corr.) | {_fmt(bl_e, 6)}"
            f" | {_fmt(winner.get('mean_energy_wh_per_1k_output_tokens_corrected'), 6)}"
            f" | {_fmt(winner.get('energy_reduction_percent'), 2)}% ↓ |",
            f"| Latency (s) | {_fmt(bl_lat, 3)} | {_fmt(winner.get('mean_latency_seconds'), 3)}"
            f" | {_fmt(winner.get('latency_reduction_percent'), 2)}% ↓ |",
            f"| Tokens per Joule (corr.) | {_fmt(bl_tpj, 4)}"
            f" | {_fmt(winner.get('mean_tokens_per_joule_corrected') or winner.get('mean_tokens_per_joule_measured'), 4)}"
            f" | {_fmt(winner.get('tokens_per_joule_improvement_percent'), 2)}% ↑ |",
            f"| Quality per Joule (corr.) | {_fmt(bl_qpj, 6)}"
            f" | {_fmt(winner.get('quality_per_joule_corrected') or winner.get('quality_per_joule_measured'), 6)}"
            f" | {_fmt(winner.get('quality_per_joule_improvement_percent'), 2)}% ↑ |",
            f"| EDP (J·s, corr.) | {_fmt(bl_edp, 6)}"
            f" | {_fmt(winner.get('mean_edp_joule_second_corrected') or winner.get('mean_edp_joule_second_measured'), 6)}"
            f" | {_fmt(winner.get('edp_reduction_percent'), 2)}% ↓ |",
            f"| Quality constraint passed | — | {_bool(winner.get('quality_constraint_passed'))} | — |",
            f"| Pareto status (Phase 1 baseline) | {bl_pareto} | — | — |",
            f"| Pareto status (Phase 2 optimized) | — | {_bool(winner.get('pareto_efficient'))} | — |",
            "",
            "### Statistical Non-Inferiority Test",
            "",
            f"Method: {ni.get('method', 'N/A')}  ",
            f"H₀: mean quality ≤ {_fmt(ni.get('non_inferiority_bound'), 4)}  ",
            f"H₁: mean quality > {_fmt(ni.get('non_inferiority_bound'), 4)}  ",
            f"n = {ni.get('n', 'N/A')}, t = {_fmt(ni.get('t_statistic'), 4)}, "
            f"p (one-sided) = {_fmt(ni.get('p_value_one_sided'), 4)}  ",
            f"**Conclusion: {ni.get('conclusion', 'N/A')}**",
            "",
        ]
    else:
        lines += [
            "## Best Optimized Configuration",
            "",
            "> **No configuration satisfied the quality constraint (P22).**  ",
            "> The Phase 1 baseline remains the recommended configuration.",
            "",
        ]

    lines += [
        "---",
        "",
        "_Report generated by GREEN-IA optimize_winner.py_",
    ]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"  optimization_report.md  : {path}")
    return path


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GREEN-IA Fase 2: Optimizacion de parametros de inferencia."
    )
    p.add_argument("--config",     default=str(DEFAULT_CONFIG),
                   help="Ruta a config.yaml (default: config.yaml)")
    p.add_argument("--opt-config", default=str(DEFAULT_OPT_CONFIG),
                   help="Ruta a optimization_config.yaml")
    p.add_argument("--summary",    default=str(SUMMARY_CSV),
                   help="CSV de resumen de Fase 1")
    p.add_argument("--dry-run",    action="store_true",
                   help="Sin modelo ni API; datos simulados")
    p.add_argument("--smoke-test", action="store_true",
                   help="1 trial, 1 rep, 1 pregunta por categoria")
    p.add_argument("--no-resume",  action="store_true",
                   help="Ignorar datos previos y recalcular")
    p.add_argument("--stage",      choices=["a", "b", "both"], default="both",
                   help="Etapa a ejecutar (default: both)")
    p.add_argument("--out-dir",    default=None,
                   help="Directorio de salida")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 68)
    print("GREEN-IA FASE 2 — Optimizacion de parametros de inferencia")
    print("=" * 68)

    # ── Load configs ───────────────────────────────────────────────────────────
    print("\n[1/7] Cargando configuracion...")
    base_cfg = load_config(Path(args.config))
    print(f"  hardware_profile : {base_cfg.hardware_profile}")
    print(f"  execution_device : {base_cfg.execution_device}")

    opt_cfg = load_optimization_config(Path(args.opt_config))
    print(f"  search_method    : {opt_cfg.search_method}")
    print(f"  max_trials       : {opt_cfg.max_trials}")
    print(f"  stage_a_reps     : {opt_cfg.repetitions_per_trial}")
    print(f"  stage_b_reps     : {opt_cfg.validation_repetitions}")
    print(f"  fixed params     : temperature={opt_cfg.fixed_temperature} "
          f"top_p={opt_cfg.fixed_top_p} seed={opt_cfg.fixed_seed}")
    print(f"  search params    : {list(opt_cfg.parameter_space.keys())}")

    out_dir = Path(args.out_dir) if args.out_dir else opt_cfg.output_dir

    if args.smoke_test:
        opt_cfg.max_trials = 1
        opt_cfg.repetitions_per_trial = 1
        opt_cfg.validation_repetitions = 1
        print("  [SMOKE TEST] max_trials=1, reps=1")

    # ── Hardware ───────────────────────────────────────────────────────────────
    print("\n[2/7] Detectando hardware...")
    hw     = detect_hardware()
    device = base_cfg.execution_device
    print(f"  CPU: {hw.cpu_model}")
    print(f"  RAM: {hw.ram_total_gb} GB")
    print(f"  GPU: {hw.gpu_model or 'N/A'}")

    # ── MT-Bench subset ────────────────────────────────────────────────────────
    print("\n[3/7] Cargando subset MT-Bench...")
    questions  = load_subset_questions(max_per_category=1 if args.smoke_test else None)
    references = _load_references()
    print(f"  {len(questions)} pregunta(s) | "
          f"referencias={'si' if references else 'no'}")

    # ── Validation split ───────────────────────────────────────────────────────
    if opt_cfg.validation_split_enabled:
        tuning_qs, validation_qs = _split_questions(questions, opt_cfg)
        print(
            f"  Validation split ({opt_cfg.validation_split_strategy}): "
            f"{len(tuning_qs)} tuning (Stage A) / "
            f"{len(validation_qs)} validation (Stage B)"
        )
    else:
        tuning_qs    = questions
        validation_qs = questions

    # ── Baseline ───────────────────────────────────────────────────────────────
    print("\n[4/7] Seleccionando baseline de Fase 1...")
    baseline         = select_baseline(Path(args.summary), opt_cfg)
    baseline_quality = _flt(baseline.get("mean_quality_score")) or 0.0
    model_name       = str(baseline.get("model_name", ""))
    quantization     = str(baseline.get("quantization", ""))
    hw_profile       = str(baseline.get("hardware_profile") or base_cfg.hardware_profile)
    print(f"  {model_name}/{quantization}  calidad={baseline_quality:.2f}")

    try:
        model_path = resolve_model_path(model_name, quantization)
        print(f"  Modelo: {model_path}")
    except FileNotFoundError as e:
        if args.dry_run or args.smoke_test:
            model_path = MODELS_DIR / model_name / f"{quantization}.gguf"
            print(f"  [dry-run] modelo ficticio: {model_path}")
        else:
            raise

    # ── API key ────────────────────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key and not args.dry_run:
        print(
            "\n  ERROR: ANTHROPIC_API_KEY no configurada.\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...\n"
            "  O usar --dry-run para simular."
        )
        sys.exit(1)

    # ── Trials ─────────────────────────────────────────────────────────────────
    print("\n[5/7] Generando trials...")
    all_trials = generate_trials(opt_cfg)
    print(f"  {len(all_trials)} trial(s) ({opt_cfg.search_method})")

    experiment_id = stable_row_id(
        EXPERIMENT_TAG, model_name, quantization,
        hw_profile, device,
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
    )
    print(f"  experiment_id: {experiment_id}")

    # ── Stage A ────────────────────────────────────────────────────────────────
    stage_a_metrics: list[dict] = []
    stage_a_out = out_dir / "stage_a"

    if args.stage in ("a", "both"):
        split_note_a = (
            f" [tuning split: {len(tuning_qs)}/{len(questions)} prompts]"
            if opt_cfg.validation_split_enabled else ""
        )
        print(
            f"\n[6a/7] Stage A — {len(all_trials)} trials × "
            f"{opt_cfg.repetitions_per_trial} reps × "
            f"{len(tuning_qs)} preguntas{split_note_a}"
        )
        conv_a, judge_a = run_stage(
            stage_name="stage_a",
            trials=all_trials,
            repetitions=opt_cfg.repetitions_per_trial,
            questions=tuning_qs,
            hw=hw,
            base_cfg=base_cfg,
            opt_cfg=opt_cfg,
            model_name=model_name,
            quant=quantization,
            model_path=model_path,
            device=device,
            out_dir=stage_a_out,
            api_key=api_key,
            references=references,
            dry_run=args.dry_run,
            no_resume=args.no_resume,
            experiment_id=experiment_id,
        )
        stage_a_metrics = aggregate_trial_metrics(
            conv_a, judge_a,
            baseline_quality=baseline_quality,
            drop_abs=opt_cfg.quality_constraint_max_drop_absolute,
            drop_rel=opt_cfg.quality_constraint_max_drop_relative_percent,
        )
        sa_path = stage_a_out / "stage_a_summary.csv"
        if stage_a_metrics:
            csv_rows = _strip_for_csv(stage_a_metrics)
            with open(sa_path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
                w.writeheader()
                w.writerows(csv_rows)
            print(f"\n  Stage A summary: {sa_path}")
            top = stage_a_metrics[0]
            print(
                f"  Top trial: id={top.get('trial_id')} "
                f"calidad={top.get('mean_quality_score')}"
            )

    elif args.stage == "b":
        sa_path = stage_a_out / "stage_a_summary.csv"
        if not sa_path.exists():
            print(
                f"\n  ERROR: --stage b pero falta Stage A: {sa_path}\n"
                "  Ejecutar primero con --stage a o --stage both."
            )
            sys.exit(1)
        with open(sa_path, "r", encoding="utf-8", newline="") as fh:
            raw_rows = list(csv.DictReader(fh))
        for m in raw_rows:
            for k in ["mean_quality_score", "mean_energy_wh_per_1k_output_tokens_measured",
                      "mean_energy_wh_per_1k_output_tokens_corrected", "mean_latency_seconds",
                      "n_ctx", "max_tokens", "n_batch", "n_threads",
                      "temperature", "top_p", "repeat_penalty", "top_k", "seed"]:
                if k in m and m[k] != "":
                    try:
                        m[k] = float(m[k]) if "." in str(m[k]) else int(m[k])
                    except (ValueError, TypeError):
                        pass
        stage_a_metrics = raw_rows
        print(f"\n[6a/7] Stage A cargado desde {sa_path} ({len(stage_a_metrics)} trials)")

    # ── Stage B ────────────────────────────────────────────────────────────────
    stage_b_metrics: list[dict] = []
    stage_b_out = out_dir / "stage_b"

    if args.stage in ("b", "both"):
        candidates = select_stage_b_candidates(stage_a_metrics, baseline_quality, opt_cfg)
        if candidates:
            split_note_b = (
                f" [validation split: {len(validation_qs)}/{len(questions)} prompts]"
                if opt_cfg.validation_split_enabled else ""
            )
            print(
                f"\n[6b/7] Stage B — {len(candidates)} candidato(s) × "
                f"{opt_cfg.validation_repetitions} reps × "
                f"{len(validation_qs)} preguntas{split_note_b}"
            )
            conv_b, judge_b = run_stage(
                stage_name="stage_b",
                trials=candidates,
                repetitions=opt_cfg.validation_repetitions,
                questions=validation_qs,
                hw=hw,
                base_cfg=base_cfg,
                opt_cfg=opt_cfg,
                model_name=model_name,
                quant=quantization,
                model_path=model_path,
                device=device,
                out_dir=stage_b_out,
                api_key=api_key,
                references=references,
                dry_run=args.dry_run,
                no_resume=args.no_resume,
                experiment_id=experiment_id,
            )
            stage_b_metrics = aggregate_trial_metrics(
                conv_b, judge_b,
                baseline_quality=baseline_quality,
                drop_abs=opt_cfg.quality_constraint_max_drop_absolute,
                drop_rel=opt_cfg.quality_constraint_max_drop_relative_percent,
            )
            sb_path = stage_b_out / "stage_b_summary.csv"
            if stage_b_metrics:
                csv_rows_b = _strip_for_csv(stage_b_metrics)
                with open(sb_path, "w", newline="", encoding="utf-8") as fh:
                    w2 = csv.DictWriter(fh, fieldnames=list(csv_rows_b[0].keys()))
                    w2.writeheader()
                    w2.writerows(csv_rows_b)
                print(f"\n  Stage B summary: {sb_path}")
        else:
            print("\n[6b/7] Sin candidatos Stage B.")

    # ── Annotate Stage B with baseline comparison and Pareto flags ────────────
    baseline_config_id: str = str(baseline.get("config_id", ""))
    if stage_b_metrics:
        _annotate_with_baseline(
            stage_b_metrics,
            baseline,
            baseline_config_id=baseline_config_id,
            phase=PHASE,
        )

    # ── P22 winner ─────────────────────────────────────────────────────────────
    print("\n[7/7] Seleccionando ganador (P22)...")
    winner: Optional[dict] = None
    if stage_b_metrics:
        winner = select_winner_p22(stage_b_metrics, baseline, opt_cfg)
        if winner:
            w_em = _flt(
                winner.get("mean_energy_wh_per_1k_output_tokens_corrected")
                or winner.get("mean_energy_wh_per_1k_output_tokens_measured")
            )
            e_red = winner.get("energy_reduction_percent")
            print(
                f"  GANADOR: trial_id={winner.get('trial_id')} "
                f"calidad={winner.get('mean_quality_score'):.2f} "
                f"energia={w_em:.4f} Wh/1k "
                f"reduccion={e_red:.1f}%" if e_red is not None else
                f"  GANADOR: trial_id={winner.get('trial_id')} "
                f"calidad={winner.get('mean_quality_score'):.2f} "
                f"energia={w_em:.4f} Wh/1k"
            )
        else:
            print("  Sin ganador P22.")
    else:
        print("  Stage B no ejecutado.")

    # ── Output files ───────────────────────────────────────────────────────────
    print("\n[Outputs] Guardando archivos de optimizacion...")
    if stage_b_metrics:
        write_optimization_trials(stage_b_metrics, out_dir)
        write_optimization_summary(stage_b_metrics, out_dir)
        write_pareto_front(stage_b_metrics, out_dir)
    if winner:
        write_best_configuration(winner, out_dir, experiment_id)

    # ── JSON report ────────────────────────────────────────────────────────────
    report_path = generate_report(
        winner=winner,
        baseline=baseline,
        stage_a_metrics=stage_a_metrics,
        stage_b_metrics=stage_b_metrics,
        out_dir=out_dir,
        experiment_id=experiment_id,
        opt_cfg=opt_cfg,
    )

    # ── Markdown report ────────────────────────────────────────────────────────
    write_optimization_report_md(
        winner=winner,
        baseline=baseline,
        stage_a_metrics=stage_a_metrics,
        stage_b_metrics=stage_b_metrics,
        opt_cfg=opt_cfg,
        out_dir=out_dir,
        experiment_id=experiment_id,
        n_tuning_prompts=len(tuning_qs),
        n_validation_prompts=len(validation_qs),
    )

    print("\n" + "=" * 68)
    print("Fase 2 completada.")
    print(f"  Salidas : {out_dir}")
    print(f"  Reporte : {report_path}")
    if winner:
        bl_em = _flt(
            baseline.get("mean_total_energy_wh_per_1k_output_tokens_corrected")
            or baseline.get("mean_total_energy_wh_per_1k_output_tokens_measured")
        )
        w_em = _flt(
            winner.get("mean_energy_wh_per_1k_output_tokens_corrected")
            or winner.get("mean_energy_wh_per_1k_output_tokens_measured")
        )
        if bl_em and w_em and bl_em > 0:
            pct = (w_em - bl_em) / bl_em * 100.0
            print(f"  Energia : {'↓' if pct < 0 else '↑'}{abs(pct):.1f}% vs baseline")
    print("=" * 68)


if __name__ == "__main__":
    main()

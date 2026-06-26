"""
metadata_writer.py
GREEN-IA — Maestria Ciencia de Datos
Universidad Comunero — Paraguay
Autor: Andres Villamayor

Escritura centralizada de los 6 archivos de metadatos de reproducibilidad (P11).

Archivos de salida:
  results/metadata/experiment_manifest.json
  results/metadata/environment_report.json
  results/metadata/model_hashes.csv
  results/metadata/dataset_manifest.csv
  results/metadata/config_snapshot.yaml
  results/metadata/requirements_freeze.txt

Cada build_* construye el dict o lista de datos desde los parametros del
experimento.  Cada write_* serializa y persiste en disco; es idempotente.

Uso tipico (run_experiment.py):
    from src.metadata_writer import (
        build_experiment_manifest, write_experiment_manifest,
        build_environment_report, write_environment_report,
        build_model_hashes, write_model_hashes,
        build_dataset_manifest, write_dataset_manifest,
        write_config_snapshot, write_requirements_freeze,
    )

Principio P11: reproducibilidad — guardar metadatos completos:
  hashes de archivos, versiones de paquetes, configuracion, manifiesto
  del dataset y reporte del entorno.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

ROOT = Path(__file__).parent.parent
METADATA_DIR = ROOT / "results" / "metadata"

# ─── Canonical column schemas ─────────────────────────────────────────────────

MODEL_HASHES_COLS: list[str] = [
    "model_name",
    "quantization",
    "path",
    "file_size_bytes",
    "sha256",
    "hash_type",
    "verified_at",
]

DATASET_MANIFEST_COLS: list[str] = [
    "official_question_file",
    "sha256",
    "subset_file",
    "subset_sha256",
    "official_question_id",
    "original_category",
    "internal_category",
    "subset_id",
]


# ─── File-integrity helpers ───────────────────────────────────────────────────

def sha256_full(path: Path) -> str:
    """Full SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_partial(path: Path, head_bytes: int = 1_048_576) -> str:
    """SHA256 of the first head_bytes of path (GGUF header fingerprint).

    The first 1 MB of a GGUF file contains the architecture, quantization
    level, and model metadata. Combined with the total file size this
    uniquely identifies the model version at a fraction of the cost of
    a full hash (~60-90 s for 4-8 GB files). Ref: P11.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(head_bytes))
    return h.hexdigest()


def _sha256_file(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists():
        return None
    return sha256_full(path)


def _pip_freeze() -> str:
    try:
        return subprocess.check_output(
            ["pip", "freeze"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return "pip_freeze_unavailable"


def _get_key_versions() -> dict:
    """Versions of packages critical to the experiment (for readability)."""
    result: dict[str, str] = {}
    pairs = [
        ("llama_cpp_python", "llama_cpp"),
        ("codecarbon",       "codecarbon"),
        ("pandas",           "pandas"),
        ("numpy",            "numpy"),
        ("psutil",           "psutil"),
        ("requests",         "requests"),
        ("scipy",            "scipy"),
        ("matplotlib",       "matplotlib"),
    ]
    for key, import_name in pairs:
        if key in result:
            continue
        try:
            mod = __import__(import_name)
            result[key] = getattr(mod, "__version__", "unknown")
        except ImportError:
            result[key] = "not_installed"
    return result


def _os_name() -> str:
    s = platform.system()
    return {"Darwin": "macOS", "Windows": "Windows"}.get(s, s or "unknown")


def _find_subset(root: Optional[Path] = None) -> Optional[Path]:
    r = root or ROOT
    for name in (
        "mt_bench_literal_subset_5_per_category.yaml",
        "mt_bench_literal_subset_5_per_category.jsonl",
    ):
        p = r / "data" / "mt_bench" / "subset" / name
        if p.exists():
            return p
    return None


def _read_optimization_settings() -> Optional[dict]:
    """
    Summarize Phase 2 optimization settings from optimization_config.yaml.

    Returns None when the file is absent or optimization is disabled,
    so the manifest field is null rather than an empty object.
    """
    optim_path = ROOT / "optimization_config.yaml"
    if not optim_path.exists() or yaml is None:
        return None
    try:
        with open(optim_path, "r", encoding="utf-8") as fh:
            raw: dict = yaml.safe_load(fh) or {}
        opt = raw.get("optimization") or {}
        if not opt.get("enabled"):
            return None
        qc = opt.get("quality_constraint") or {}
        vs = opt.get("validation_split") or {}
        return {
            "enabled"                      : True,
            "phase"                        : opt.get("phase"),
            "selected_configuration"       : opt.get("selected_configuration"),
            "objective"                    : opt.get("objective"),
            "search_method"                : opt.get("search_method"),
            "max_trials"                   : opt.get("max_trials"),
            "repetitions_per_trial"        : opt.get("repetitions_per_trial"),
            "validation_repetitions"       : opt.get("validation_repetitions"),
            "stage_a_pareto_top_n"         : opt.get("stage_a_pareto_top_n"),
            "quality_constraint": {
                "max_drop_absolute"         : qc.get("max_allowed_drop_absolute"),
                "max_drop_relative_percent" : qc.get("max_allowed_drop_relative_percent"),
                "require_non_inferiority"   : qc.get("require_non_inferiority"),
            },
            "parameter_space"              : sorted((opt.get("parameter_space") or {}).keys()),
            "fixed_parameters"             : opt.get("fixed_parameters"),
            "validation_split_enabled"     : vs.get("enabled", False),
            "validation_split_strategy"    : vs.get("strategy"),
            "output_dir"                   : opt.get("output_dir"),
        }
    except Exception:
        return None


# ─── 1. experiment_manifest.json ──────────────────────────────────────────────

def build_experiment_manifest(
    *,
    run_id: str,
    run_ts: str,
    run_type: str,
    cfg,                       # ExperimentConfig
    raw_cfg: dict,
    questions: list[dict],
    eff_n_per_category: int,
    eff_n_repetitions: int,
    output_dir: Optional[Path] = None,
    phase: int = 1,
    status: str = "initialized",
) -> dict:
    """
    Build the complete experiment manifest dict.

    Required fields (all present):
        experiment_name, timestamp_utc, phase, hardware_profile,
        execution_device, selected_models, selected_quantizations,
        repetitions, prompt_dataset (with subset_strategy),
        official_question_ids, primary_energy_metric,
        judge_model, generation_mode, optimization_settings.
    """
    pd_cfg  = raw_cfg.get("prompt_dataset") or {}
    lp      = raw_cfg.get("llama_cpp_params") or {}
    gm      = raw_cfg.get("generation_mode") or {}
    jd      = raw_cfg.get("judge") or {}
    an      = raw_cfg.get("analysis") or {}
    rt      = raw_cfg.get("runtime_control") or {}
    ec      = raw_cfg.get("energy_calibration") or {}

    selected_models = raw_cfg.get("selected_models") or []
    selected_quants = raw_cfg.get("selected_quantizations") or []
    n_configs       = len(selected_models) * len(selected_quants)
    n_cats          = 8  # MT-Bench always has 8 categories

    categories = sorted({
        str(q.get("category") or q.get("original_category") or "unknown")
        for q in questions
    })

    official_question_ids = sorted(
        int(q["question_id"])
        for q in questions
        if q.get("question_id") is not None
    )

    subset_path = _find_subset()
    subset_sha  = _sha256_file(subset_path)
    cfg_sha     = _sha256_file(ROOT / "config.yaml")

    return {
        # ─ identity ─────────────────────────────────────────────────────────
        "experiment_name"        : raw_cfg.get("experiment_name", "green_ai_llm_quality_energy"),
        "run_id"                 : run_id,
        "run_type"               : run_type,
        "phase"                  : phase,
        "status"                 : status,
        "timestamp_utc"          : run_ts,

        # ─ hardware ─────────────────────────────────────────────────────────
        "hardware_profile"       : cfg.hardware_profile,
        "execution_device"       : cfg.execution_device,
        "cli_overrides"          : raw_cfg.get("_cli_overrides", {}),

        # ─ models ────────────────────────────────────────────────────────────
        "selected_models"        : selected_models,
        "selected_quantizations" : selected_quants,
        "n_configurations"       : n_configs,

        # ─ repetitions ───────────────────────────────────────────────────────
        "repetitions"            : eff_n_repetitions,
        "n_turns_per_question"   : 2,
        "n_measurements_planned" : n_configs * len(official_question_ids) * 2 * eff_n_repetitions,

        # ─ prompt dataset ────────────────────────────────────────────────────
        "prompt_dataset": {
            "name"                    : pd_cfg.get("name", "mt_bench_literal_subset"),
            "source"                  : pd_cfg.get("source", "official_fastchat_mt_bench"),
            "language"                : pd_cfg.get("language", "english"),
            "n_questions_per_category": eff_n_per_category,
            "n_categories"            : n_cats,
            "n_questions_total"       : eff_n_per_category * n_cats,
            "subset_file"             : str(subset_path.relative_to(ROOT)) if subset_path else None,
            "subset_sha256"           : subset_sha,
            "subset_strategy"         : pd_cfg.get("selection_strategy", "first_n_per_category"),
        },

        # ─ official question IDs (P7, P11) ───────────────────────────────────
        "official_question_ids"  : official_question_ids,
        "categories"             : categories,

        # ─ energy metric ─────────────────────────────────────────────────────
        "primary_energy_metric"  : an.get("primary_energy_metric", "baseline_corrected_energy"),

        # ─ judge ─────────────────────────────────────────────────────────────
        "judge_model"            : jd.get("claude_model", "configurable"),
        "judge_provider"         : jd.get("provider", "anthropic"),
        "judge_mode"             : jd.get("mode", "claude_multidimensional"),

        # ─ generation ────────────────────────────────────────────────────────
        "generation_mode"        : str(gm.get("name", "deterministic_energy")),
        "temperature"            : float(lp.get("temperature", 0.0)),
        "top_p"                  : float(lp.get("top_p", 1.0)),
        "seed"                   : int(lp.get("seed", 42)),
        "vary_seed_by_repetition": bool(gm.get("vary_seed_by_repetition", False)),
        "n_ctx"                  : int(lp.get("n_ctx", 4096)),
        "max_tokens"             : int(lp.get("max_tokens", 1024)),

        # ─ runtime ───────────────────────────────────────────────────────────
        "pause_before_start_s"   : int(rt.get("pause_before_start_seconds", 30)),
        "warmup_runs"            : int(rt.get("warmup_runs_per_configuration", 1)),
        "baseline_idle_s"        : int(ec.get("baseline_idle_seconds", 60)),
        "baseline_repetitions"   : int(ec.get("baseline_repetitions", 3)),

        # ─ output ────────────────────────────────────────────────────────────
        "results_dir"            : str((output_dir or ROOT / "results" / "raw").relative_to(ROOT)),

        # ─ optimization settings (Phase 2) ───────────────────────────────────
        "optimization_settings"  : _read_optimization_settings(),

        # ─ reproducibility ───────────────────────────────────────────────────
        "config_sha256"          : cfg_sha,
        "python_version"         : platform.python_version(),
        "platform"               : platform.platform(),
    }


def write_experiment_manifest(
    manifest: dict,
    out_dir: Optional[Path] = None,
) -> Path:
    d = out_dir or METADATA_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / "experiment_manifest.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False, default=str)
    return path


# ─── 2. environment_report.json ───────────────────────────────────────────────

def build_environment_report(
    hw_profile=None,           # HardwareProfile instance (src.hardware_profile)
    raw_cfg: Optional[dict] = None,
    run_ts: Optional[str] = None,
    model_fingerprints: Optional[dict] = None,
) -> dict:
    """
    Build the full environment report.

    Top-level fields for immediate readability:
        os, cpu_detected, ram_detected, gpu_detected,
        codecarbon_version, llama_cpp_python_version, package_versions.

    The complete hardware_profile dict is also preserved for
    backward compatibility and fine-grained inspection.
    """
    hw: dict = hw_profile.to_dict() if hw_profile is not None else {}
    kv = _get_key_versions()

    return {
        # ─ timestamp ─────────────────────────────────────────────────────────
        "timestamp_utc"            : run_ts or datetime.now(timezone.utc).isoformat(),

        # ─ OS (structured + raw string) ──────────────────────────────────────
        "os": {
            "name"            : _os_name(),
            "version"         : hw.get("os_version") or platform.version(),
            "platform_string" : platform.platform(),
        },
        "platform"                 : platform.platform(),
        "python_version"           : platform.python_version(),
        "python_implementation"    : platform.python_implementation(),
        "architecture"             : platform.machine(),
        "processor"                : platform.processor() or hw.get("cpu_vendor", ""),

        # ─ packages (top-level convenience + full dict) ───────────────────────
        "codecarbon_version"       : kv.get("codecarbon", "unknown"),
        "llama_cpp_python_version" : kv.get("llama_cpp_python", "unknown"),
        "package_versions"         : kv,

        # ─ hardware (promoted top-level fields) ──────────────────────────────
        "cpu_detected": {
            "model"          : hw.get("cpu_model") or platform.processor() or "unknown",
            "cores_physical" : hw.get("cpu_cores_physical"),
            "cores_logical"  : hw.get("cpu_cores_logical"),
            "vendor"         : hw.get("cpu_vendor"),
        },
        "ram_detected": {
            "total_gb"       : hw.get("ram_total_gb"),
            "type"           : hw.get("ram_type", "unknown"),
        },
        "gpu_detected": {
            "available"      : hw.get("gpu_available", False),
            "model"          : hw.get("gpu_model"),
            "backend"        : hw.get("gpu_backend"),
            "vram_gb"        : hw.get("gpu_vram_gb"),
        },

        # ─ full hardware profile ─────────────────────────────────────────────
        "hardware_profile"         : hw,

        # ─ experiment config ─────────────────────────────────────────────────
        "experiment_config": {
            k: v for k, v in (raw_cfg or {}).items()
            if not k.startswith("_")
        } if raw_cfg else None,

        # ─ model fingerprints ────────────────────────────────────────────────
        "model_fingerprints"       : model_fingerprints,
    }


def write_environment_report(
    report: dict,
    out_dir: Optional[Path] = None,
) -> Path:
    d = out_dir or METADATA_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / "environment_report.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
    return path


# ─── 3. model_hashes.csv ──────────────────────────────────────────────────────

def build_model_hashes(
    raw_cfg: dict,
    root: Optional[Path] = None,
    full_hash: bool = False,
) -> list[dict]:
    """
    Compute model file hashes for all selected model / quantization pairs.

    Columns: model_name, quantization, path, file_size_bytes, sha256,
             hash_type, verified_at.

    Default: SHA256 of the first 1 MB (GGUF header) — sufficient to
    identify the model version without reading multi-GB files (P11).
    Use full_hash=True for a complete SHA256 (~60-90 s per 4-8 GB file).
    """
    r               = root or ROOT
    now_iso         = datetime.now(timezone.utc).isoformat()
    selected_models = raw_cfg.get("selected_models") or []
    selected_quants = raw_cfg.get("selected_quantizations") or []
    models_block    = raw_cfg.get("models") or {}
    rows: list[dict] = []

    for model in selected_models:
        model_def = models_block.get(model) or {}
        for q in selected_quants:
            path_str = model_def.get(q)
            if not path_str:
                rows.append({
                    "model_name"     : model,
                    "quantization"   : q,
                    "path"           : "",
                    "file_size_bytes": "",
                    "sha256"         : "",
                    "hash_type"      : "missing_path",
                    "verified_at"    : now_iso,
                })
                continue

            path = r / str(path_str)
            rel  = path.relative_to(r) if path.is_relative_to(r) else path

            if not path.exists():
                rows.append({
                    "model_name"     : model,
                    "quantization"   : q,
                    "path"           : str(rel),
                    "file_size_bytes": "",
                    "sha256"         : "",
                    "hash_type"      : "file_not_found",
                    "verified_at"    : now_iso,
                })
                continue

            size      = path.stat().st_size
            digest    = sha256_full(path) if full_hash else sha256_partial(path)
            hash_type = "sha256_full" if full_hash else "sha256_partial_1mb"

            rows.append({
                "model_name"     : model,
                "quantization"   : q,
                "path"           : str(rel),
                "file_size_bytes": size,
                "sha256"         : digest,
                "hash_type"      : hash_type,
                "verified_at"    : now_iso,
            })

    return rows


def write_model_hashes(
    rows: list[dict],
    out_dir: Optional[Path] = None,
) -> Path:
    d = out_dir or METADATA_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / "model_hashes.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MODEL_HASHES_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


# ─── 4. dataset_manifest.csv ──────────────────────────────────────────────────

def build_dataset_manifest(
    questions: list[dict],
    official_file: Optional[Path] = None,
    subset_file: Optional[Path] = None,
    root: Optional[Path] = None,
) -> list[dict]:
    """
    Build per-question dataset manifest rows.

    One row per question in the experiment subset.  File-level integrity
    fields (official_question_file, sha256, subset_file, subset_sha256)
    repeat on every row so that the CSV is self-contained: any row can
    be used to reconstruct the full provenance chain.

    Columns: official_question_file, sha256, subset_file, subset_sha256,
             official_question_id, original_category, internal_category,
             subset_id.
    """
    r         = root or ROOT
    off_path  = official_file or (r / "data" / "mt_bench" / "official" / "question.jsonl")
    sub_path  = subset_file or _find_subset(r)

    off_sha   = _sha256_file(off_path) or "file_not_found"
    sub_sha   = _sha256_file(sub_path) or "file_not_found"

    def _rel(p: Optional[Path]) -> str:
        if p is None:
            return ""
        try:
            return str(p.relative_to(r))
        except ValueError:
            return str(p)

    off_rel = _rel(off_path)
    sub_rel = _rel(sub_path)

    rows: list[dict] = []
    for i, q in enumerate(questions, start=1):
        qid      = q.get("question_id") or q.get("official_question_id")
        orig_cat = str(q.get("original_category") or q.get("category") or "")
        int_cat  = str(q.get("category") or q.get("internal_category") or orig_cat)
        sid      = q.get("subset_id") or f"S{i:03d}"
        rows.append({
            "official_question_file": off_rel,
            "sha256"                : off_sha,
            "subset_file"           : sub_rel,
            "subset_sha256"         : sub_sha,
            "official_question_id"  : qid,
            "original_category"     : orig_cat,
            "internal_category"     : int_cat,
            "subset_id"             : sid,
        })
    return rows


def write_dataset_manifest(
    rows: list[dict],
    out_dir: Optional[Path] = None,
) -> Path:
    d = out_dir or METADATA_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / "dataset_manifest.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DATASET_MANIFEST_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


# ─── 5. config_snapshot.yaml ──────────────────────────────────────────────────

def write_config_snapshot(
    raw_cfg: dict,
    run_ts: str,
    out_dir: Optional[Path] = None,
) -> Path:
    """Write effective config (with CLI overrides applied) as YAML."""
    if yaml is None:
        raise ImportError("PyYAML not installed: pip install pyyaml")
    snapshot = {k: v for k, v in raw_cfg.items() if not k.startswith("_")}
    snapshot["_snapshot_generated_at"] = run_ts
    d = out_dir or METADATA_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / "config_snapshot.yaml"
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(
            snapshot, fh,
            default_flow_style=False, allow_unicode=True,
            sort_keys=False, width=10000,
        )
    return path


# ─── 6. requirements_freeze.txt ───────────────────────────────────────────────

def write_requirements_freeze(out_dir: Optional[Path] = None) -> Path:
    """Write pip freeze output to requirements_freeze.txt."""
    freeze = _pip_freeze()
    d = out_dir or METADATA_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / "requirements_freeze.txt"
    path.write_text(freeze, encoding="utf-8")
    return path


# ─── Convenience: write all 6 in one call ────────────────────────────────────

def write_all_metadata(
    *,
    manifest: dict,
    env_report: dict,
    model_rows: list[dict],
    dataset_rows: list[dict],
    raw_cfg: dict,
    run_ts: str,
    out_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """
    Write all 6 metadata files and return a mapping of filename -> Path.

    Typical call from run_experiment.py after building each data structure:
        paths = write_all_metadata(
            manifest=manifest, env_report=env_report,
            model_rows=model_rows, dataset_rows=dataset_rows,
            raw_cfg=raw_cfg, run_ts=run_ts,
        )
    """
    d = out_dir or METADATA_DIR
    return {
        "experiment_manifest.json" : write_experiment_manifest(manifest, d),
        "environment_report.json"  : write_environment_report(env_report, d),
        "model_hashes.csv"         : write_model_hashes(model_rows, d),
        "dataset_manifest.csv"     : write_dataset_manifest(dataset_rows, d),
        "config_snapshot.yaml"     : write_config_snapshot(raw_cfg, run_ts, d),
        "requirements_freeze.txt"  : write_requirements_freeze(d),
    }

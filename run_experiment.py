#!/usr/bin/env python3
"""
run_experiment.py
Proyecto GREEN-IA — Maestria Ciencia de Datos
Universidad Comunero — Paraguay
Autor: Andres Villamayor

Punto de entrada principal para la Fase 1 del experimento.

Setup (pasos 1–8):
  1. Lee config.yaml  (+ overrides de CLI)
  2. Ejecuta validaciones previas al experimento
  3. Crea carpetas de salida
  4. Guarda config_snapshot.yaml en results/metadata/
  5. Genera environment_report.json  (plataforma, paquetes, hardware, hashes)
  6. Genera experiment_manifest.json (ID, parametros, estadisticas del subset)
  7. Carga el subset oficial desde:
       data/mt_bench/subset/mt_bench_literal_subset_5_per_category.yaml
       o el equivalente .jsonl como fallback
  8. Soporta overrides de hardware via CLI

Tras el setup, delega la medicion a scripts/fase1_benchmark.py.

Smoke test vs oficial
─────────────────────
Un run es smoke test cuando --limit-conversations < 5 o --repetitions < 15.
Los resultados de smoke test van a results/smoke_test/ y NUNCA se mezclan
con los resultados oficiales de results/raw/.
Usar --limit-conversations y --repetitions solo para smoke tests; omitirlos
(o igualar a los valores oficiales) para una corrida oficial completa.

Uso:
  # Runs oficiales
  python run_experiment.py --hardware-profile mac_m4          --execution-device gpu
  python run_experiment.py --hardware-profile mac_m4          --execution-device cpu
  python run_experiment.py --hardware-profile windows_nvidia  --execution-device gpu
  python run_experiment.py --hardware-profile windows_nvidia  --execution-device cpu

  # Smoke tests (resultados en results/smoke_test/)
  python run_experiment.py --hardware-profile mac_m4         --execution-device gpu --limit-conversations 2 --repetitions 1
  python run_experiment.py --hardware-profile windows_nvidia --execution-device gpu --limit-conversations 2 --repetitions 1

  # Solo setup (sin correr la medicion)
  python run_experiment.py --setup-only
  python run_experiment.py --dry-run
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML no instalado. Ejecutar: pip install pyyaml")
    sys.exit(1)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config, ExperimentConfig, HARDWARE_SPECS
from src.reproducibility import model_file_info, config_fingerprint
from src.hardware_profile import detect_hardware
import src.metadata_writer as metadata_writer

CONFIG_FILE       = ROOT / "config.yaml"
OPTIM_CONFIG      = ROOT / "optimization_config.yaml"
SUBSET_YAML       = ROOT / "data" / "mt_bench" / "subset" / "mt_bench_literal_subset_5_per_category.yaml"
SUBSET_JSONL      = ROOT / "data" / "mt_bench" / "subset" / "mt_bench_literal_subset_5_per_category.jsonl"
METADATA_DIR      = ROOT / "results" / "metadata"
FASE1_SCRIPT      = ROOT / "scripts" / "fase1_benchmark.py"

# Valores de referencia del experimento oficial completo
OFFICIAL_N_PER_CATEGORY = 5   # 5 preguntas por categoria × 8 categorias = 40 total
OFFICIAL_N_REPETITIONS  = 15  # 15 repeticiones por pregunta

OUTPUT_DIRS: list[Path] = [
    ROOT / "results" / "raw",
    ROOT / "results" / "smoke_test",
    ROOT / "results" / "judge",
    ROOT / "results" / "summary",
    ROOT / "results" / "optimization",
    ROOT / "results" / "plots",
    ROOT / "results" / "logs",
    ROOT / "results" / "codecarbon",
    ROOT / "results" / "backups",
    ROOT / "results" / "metadata",
]


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GREEN-IA Fase 1 — driver del experimento de energia"
    )
    parser.add_argument(
        "--hardware-profile",
        choices=["mac_m4", "windows_nvidia"],
        metavar="PROFILE",
        help="Override de hardware_profile en config.yaml (mac_m4 | windows_nvidia)",
    )
    parser.add_argument(
        "--execution-device",
        choices=["cpu", "gpu"],
        metavar="DEVICE",
        help="Override de execution_device en config.yaml (cpu | gpu)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra la configuracion efectiva y el plan sin ejecutar el experimento",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Ejecuta los pasos de setup (1-8) sin iniciar la medicion",
    )
    parser.add_argument(
        "--limit-conversations",
        type=int,
        default=None,
        metavar="N",
        help=(
            f"Preguntas por categoria a medir (max {OFFICIAL_N_PER_CATEGORY}). "
            f"Cualquier valor < {OFFICIAL_N_PER_CATEGORY} activa modo smoke test: "
            "los resultados van a results/smoke_test/ y no mezclan con los oficiales."
        ),
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=None,
        metavar="N",
        help=(
            f"Repeticiones por pregunta (default={OFFICIAL_N_REPETITIONS}). "
            f"Cualquier valor < {OFFICIAL_N_REPETITIONS} activa modo smoke test."
        ),
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Omite las validaciones de pre-vuelo (no recomendado)",
    )
    return parser.parse_args()


# ─── Paso 1: config.yaml + overrides ─────────────────────────────────────────

def load_config_with_overrides(
    args: argparse.Namespace,
) -> tuple[dict, ExperimentConfig]:
    """
    Carga config.yaml y aplica overrides de CLI.

    Retorna (raw_dict, ExperimentConfig) con los valores efectivos.
    El raw_dict se usa para config_snapshot.yaml; ExperimentConfig para el codigo.
    """
    if not CONFIG_FILE.exists():
        print(f"\n  ERROR: config.yaml no encontrado.")
        print(f"  Crear con: cp config.yaml.example config.yaml")
        sys.exit(1)

    with open(CONFIG_FILE, "r", encoding="utf-8") as fh:
        raw_cfg: dict = yaml.safe_load(fh) or {}

    cfg = load_config(CONFIG_FILE)

    overrides: dict[str, str] = {}
    if args.hardware_profile:
        raw_cfg["hardware_profile"] = args.hardware_profile
        cfg = dataclasses.replace(cfg, hardware_profile=args.hardware_profile)
        overrides["hardware_profile"] = args.hardware_profile
    if args.execution_device:
        raw_cfg["execution_device"] = args.execution_device
        cfg = dataclasses.replace(cfg, execution_device=args.execution_device)
        overrides["execution_device"] = args.execution_device

    if overrides:
        raw_cfg.setdefault("_cli_overrides", {}).update(overrides)

    return raw_cfg, cfg


# ─── Paso 2: validaciones ─────────────────────────────────────────────────────

def run_validations(raw_cfg: dict, cfg: ExperimentConfig) -> list[str]:
    """
    Validaciones criticas antes de iniciar.

    Retorna lista de mensajes de error (vacia = todo correcto).
    Los avisos (WARN) se imprimen pero no bloquean.
    """
    errors: list[str] = []
    W = 58

    print(f"\n  {'─' * W}")
    print(f"  Validaciones pre-experimento")
    print(f"  {'─' * W}")

    profile = cfg.hardware_profile
    device  = cfg.execution_device
    print(f"  {'OK':<6} hardware_profile = {profile}  /  execution_device = {device}")

    # Modelos
    models_block    = raw_cfg.get("models") or {}
    selected_models = raw_cfg.get("selected_models") or []
    selected_quants = raw_cfg.get("selected_quantizations") or []

    for model in selected_models:
        model_def = models_block.get(model) or {}
        for q in selected_quants:
            path_str = model_def.get(q)
            if not path_str:
                print(f"  {'WARN':<6} ruta no definida en config.yaml: models.{model}.{q}")
                continue
            path = ROOT / str(path_str)
            if path.exists():
                size_gb = path.stat().st_size / 1e9
                print(f"  {'OK':<6} modelo encontrado: {model} {q.upper()}  ({size_gb:.2f} GB)")
            else:
                print(f"  {'WARN':<6} modelo no encontrado: {path_str}")
                print(f"         Descargar antes de ejecutar el experimento")

    # Subset MT-Bench
    subset_path = _find_subset()
    if subset_path is None:
        errors.append(
            "Subset MT-Bench no encontrado. Ejecutar: python scripts/prepare_mt_bench_subset.py"
        )
        print(f"  {'FAIL':<6} subset no encontrado: {SUBSET_YAML.relative_to(ROOT)}")
    else:
        print(f"  {'OK':<6} subset: {subset_path.relative_to(ROOT)}")

    # CodeCarbon importa
    try:
        from codecarbon import EmissionsTracker  # noqa: F401
        print(f"  {'OK':<6} codecarbon importa")
    except ImportError:
        errors.append("codecarbon no instalado: pip install codecarbon")
        print(f"  {'FAIL':<6} codecarbon no instalado")

    # llama-cpp-python importa
    try:
        from llama_cpp import Llama  # noqa: F401
        print(f"  {'OK':<6} llama-cpp-python importa")
    except ImportError:
        errors.append("llama-cpp-python no instalado. Ver README.md → Instalacion")
        print(f"  {'FAIL':<6} llama-cpp-python no instalado")

    return errors


# ─── Paso 3: carpetas de salida ────────────────────────────────────────────────

def create_output_folders() -> None:
    for d in OUTPUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    print(f"\n  Carpetas de salida listas ({len(OUTPUT_DIRS)} directorios)")


# ─── Paso 4: config_snapshot.yaml ─────────────────────────────────────────────

def save_config_snapshot(raw_cfg: dict, run_ts: str) -> Path:
    path = metadata_writer.write_config_snapshot(raw_cfg, run_ts, METADATA_DIR)
    print(f"  config_snapshot.yaml  →  {path.relative_to(ROOT)}")
    return path


# ─── Paso 5: environment_report.json ─────────────────────────────────────────

def generate_environment_report(
    cfg: ExperimentConfig,
    raw_cfg: dict,
    run_ts: str,
) -> dict:
    """
    Genera el reporte completo del entorno de ejecucion (P11).

    Incluye OS estructurado, cpu_detected, ram_detected, gpu_detected,
    codecarbon_version, llama_cpp_python_version, package_versions al
    nivel superior para lectura inmediata.
    """
    codecarbon_country = (raw_cfg.get("codecarbon") or {}).get("country_iso_code", "PRY")
    carbon_int_val     = 26  # Paraguay default
    hw_cfg = raw_cfg.get("hardware_profiles", {}).get(cfg.hardware_profile, {})
    if isinstance(hw_cfg, dict):
        carbon_int_val = hw_cfg.get("carbon_intensity_g_per_kwh", carbon_int_val)

    hw = detect_hardware(
        experiment_config=cfg,
        country_iso=codecarbon_country,
        carbon_intensity=carbon_int_val,
    )

    # Fingerprints de los modelos seleccionados
    model_fps: dict[str, dict] = {}
    for model in (raw_cfg.get("selected_models") or []):
        model_def = (raw_cfg.get("models") or {}).get(model) or {}
        for q in (raw_cfg.get("selected_quantizations") or []):
            path_str = model_def.get(q)
            if path_str:
                model_fps[f"{model}_{q}"] = model_file_info(ROOT / str(path_str))

    report = metadata_writer.build_environment_report(
        hw_profile=hw,
        raw_cfg={k: v for k, v in raw_cfg.items() if not k.startswith("_")},
        run_ts=run_ts,
        model_fingerprints=model_fps,
    )
    report["config_fingerprint"] = config_fingerprint(
        {k: v for k, v in raw_cfg.items() if not k.startswith("_")}
    )

    path = metadata_writer.write_environment_report(report, METADATA_DIR)
    print(f"  environment_report.json  →  {path.relative_to(ROOT)}")
    return report


# ─── Paso 6: experiment_manifest.json ────────────────────────────────────────

def generate_experiment_manifest(
    cfg: ExperimentConfig,
    raw_cfg: dict,
    questions: list[dict],
    run_ts: str,
    run_id: str,
    run_type: str = "official",
    eff_n_per_category: int = OFFICIAL_N_PER_CATEGORY,
    eff_n_repetitions: int = OFFICIAL_N_REPETITIONS,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Genera el manifiesto del experimento con todos los parametros requeridos.

    Campos obligatorios presentes:
        experiment_name, timestamp_utc, phase, hardware_profile,
        execution_device, selected_models, selected_quantizations,
        repetitions, prompt_dataset (con subset_strategy),
        official_question_ids, primary_energy_metric,
        judge_model, generation_mode, optimization_settings.
    """
    manifest = metadata_writer.build_experiment_manifest(
        run_id=run_id,
        run_ts=run_ts,
        run_type=run_type,
        cfg=cfg,
        raw_cfg=raw_cfg,
        questions=questions,
        eff_n_per_category=eff_n_per_category,
        eff_n_repetitions=eff_n_repetitions,
        output_dir=output_dir,
        phase=1,
        status="initialized",
    )
    path = metadata_writer.write_experiment_manifest(manifest, METADATA_DIR)
    print(f"  experiment_manifest.json →  {path.relative_to(ROOT)}")
    return manifest


# ─── Paso 7: subset MT-Bench ──────────────────────────────────────────────────

def _find_subset() -> Optional[Path]:
    """Devuelve la ruta del subset: YAML primero, JSONL como fallback."""
    if SUBSET_YAML.exists():
        return SUBSET_YAML
    if SUBSET_JSONL.exists():
        return SUBSET_JSONL
    return None


def load_subset() -> list[dict]:
    """
    Carga el subset oficial MT-Bench.

    Busca primero el archivo YAML generado por prepare_mt_bench_subset.py;
    si no existe, intenta el JSONL equivalente.

    Falla con sys.exit(1) si ninguno de los dos existe — el experimento
    no puede continuar sin los prompts verificados.
    """
    path = _find_subset()

    if path is None:
        print(f"\n  ERROR: subset MT-Bench no encontrado.")
        print(f"  Ejecutar primero:")
        print(f"    python scripts/prepare_mt_bench_subset.py")
        sys.exit(1)

    questions: list[dict] = []

    if path.suffix == ".yaml":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        questions = data.get("questions") or []
    else:
        import json as _json
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    questions.append(_json.loads(line))

    if not questions:
        print(f"\n  ERROR: el archivo de subset esta vacio: {path.relative_to(ROOT)}")
        sys.exit(1)

    n_cats = len({str(q.get("category", q.get("original_category", "")))
                  for q in questions})
    print(
        f"  Subset cargado: {len(questions)} preguntas / "
        f"{n_cats} categorias  ({path.suffix})  "
        f"[{path.name}]"
    )
    return questions


# ─── caffeinate (macOS) ───────────────────────────────────────────────────────

def _start_caffeinate(cfg: ExperimentConfig) -> "subprocess.Popen | None":
    if not cfg.should_caffeinate():
        return None
    try:
        proc = subprocess.Popen(
            ["caffeinate", "-dimsu"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"  caffeinate activo (PID {proc.pid})")
        return proc
    except FileNotFoundError:
        return None


def _stop_caffeinate(proc: "subprocess.Popen | None") -> None:
    if proc is not None:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            pass


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _sha256_file(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_smoke_test(args: argparse.Namespace) -> bool:
    if args.limit_conversations is not None and args.limit_conversations < OFFICIAL_N_PER_CATEGORY:
        return True
    if args.repetitions is not None and args.repetitions < OFFICIAL_N_REPETITIONS:
        return True
    return False


def _make_run_id(cfg: ExperimentConfig, smoke: bool = False) -> str:
    ts     = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = "smoke" if smoke else "phase1"
    return f"{prefix}_{cfg.hardware_profile}_{cfg.execution_device}_{ts}"


def _smoke_output_dir() -> Path:
    return ROOT / "results" / "smoke_test"


def _official_output_dir() -> Path:
    return ROOT / "results" / "raw"


def _print_header(run_id: str, cfg: ExperimentConfig, smoke: bool = False) -> None:
    W = 62
    print(f"\n{'=' * W}")
    label = "SMOKE TEST" if smoke else "Fase 1: Benchmark de Energia"
    print(f"  GREEN-IA — {label}")
    if smoke:
        print(f"  AVISO: resultados en results/smoke_test/ — NO son datos oficiales")
    print(f"  Run ID : {run_id}")
    print(f"  Perfil : {cfg.hardware_profile}  /  {cfg.execution_device}")
    print(f"{'=' * W}")


def _print_summary(
    cfg: ExperimentConfig,
    questions: list[dict],
    run_id: str,
    smoke: bool = False,
    eff_n_per_cat: int = OFFICIAL_N_PER_CATEGORY,
    eff_reps: int = OFFICIAL_N_REPETITIONS,
    output_dir: Optional[Path] = None,
) -> None:
    n_q  = len(questions)
    W    = 62
    dest = (output_dir or _official_output_dir()).relative_to(ROOT)
    print(f"\n  {'─' * (W - 2)}")
    print(f"  Setup completado {'[SMOKE TEST]' if smoke else '[OFICIAL]'}")
    print(f"  Run ID       : {run_id}")
    print(f"  Hardware     : {cfg.hardware_profile}  /  {cfg.execution_device}")
    print(f"  Preguntas    : {eff_n_per_cat}/categoria  ({eff_n_per_cat * 8} total efectivo)")
    print(f"  Repeticiones : {eff_reps}")
    print(f"  Resultados   : {dest}")
    print(f"  Metadata     : results/metadata/")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Paso 1+8: config + overrides CLI ──────────────────────────────────────
    raw_cfg, cfg = load_config_with_overrides(args)

    smoke          = _is_smoke_test(args)
    run_type       = "smoke_test" if smoke else "official"
    output_dir     = _smoke_output_dir() if smoke else _official_output_dir()
    eff_n_per_cat  = args.limit_conversations if args.limit_conversations is not None else OFFICIAL_N_PER_CATEGORY
    eff_reps       = args.repetitions         if args.repetitions         is not None else OFFICIAL_N_REPETITIONS

    run_id  = _make_run_id(cfg, smoke=smoke)
    run_ts  = datetime.now(timezone.utc).isoformat()

    _print_header(run_id, cfg, smoke=smoke)

    caff = _start_caffeinate(cfg)

    try:
        # ── Paso 2: validaciones ──────────────────────────────────────────────
        if not args.skip_checks:
            errors = run_validations(raw_cfg, cfg)
            if errors:
                print(f"\n  {'─' * 58}")
                print(f"  {len(errors)} error(s) critico(s):")
                for e in errors:
                    print(f"    • {e}")
                print(f"\n  Corregir los errores antes de continuar.")
                print(f"  O usar --skip-checks para omitir validaciones.")
                sys.exit(1)
        else:
            print(f"\n  AVISO: validaciones omitidas (--skip-checks)")

        # ── Paso 3: carpetas ──────────────────────────────────────────────────
        create_output_folders()
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  Generando archivos de metadata:")

        # ── Paso 4: config_snapshot.yaml ──────────────────────────────────────
        save_config_snapshot(raw_cfg, run_ts)

        # ── Paso 5: environment_report.json ───────────────────────────────────
        env_report = generate_environment_report(cfg, raw_cfg, run_ts)

        # ── requirements_freeze.txt ───────────────────────────────────────────
        rf_path = metadata_writer.write_requirements_freeze(METADATA_DIR)
        print(f"  requirements_freeze.txt  →  {rf_path.relative_to(ROOT)}")

        # ── Paso 7: cargar subset (antes del manifest para incluir su sha) ───
        questions = load_subset()

        # ── Paso 6: experiment_manifest.json ──────────────────────────────────
        manifest = generate_experiment_manifest(
            cfg, raw_cfg, questions, run_ts, run_id,
            run_type=run_type,
            eff_n_per_category=eff_n_per_cat,
            eff_n_repetitions=eff_reps,
            output_dir=output_dir,
        )

        # ── model_hashes.csv ──────────────────────────────────────────────────
        model_rows = metadata_writer.build_model_hashes(raw_cfg, ROOT)
        mh_path    = metadata_writer.write_model_hashes(model_rows, METADATA_DIR)
        print(f"  model_hashes.csv         →  {mh_path.relative_to(ROOT)}")

        # ── dataset_manifest.csv ──────────────────────────────────────────────
        ds_rows = metadata_writer.build_dataset_manifest(questions)
        dm_path = metadata_writer.write_dataset_manifest(ds_rows, METADATA_DIR)
        print(f"  dataset_manifest.csv     →  {dm_path.relative_to(ROOT)}")

        _print_summary(
            cfg, questions, run_id,
            smoke=smoke,
            eff_n_per_cat=eff_n_per_cat,
            eff_reps=eff_reps,
            output_dir=output_dir,
        )

        if args.dry_run:
            print(f"\n  Modo dry-run: experimento NO iniciado.")
            print(f"{'=' * 62}\n")
            return

        if args.setup_only:
            print(f"\n  Modo setup-only: medicion NO iniciada (--setup-only).")
            print(f"{'=' * 62}\n")
            return

        # ── Fase 1: medicion ──────────────────────────────────────────────────
        if not FASE1_SCRIPT.exists():
            print(f"\n  ERROR: script de medicion no encontrado:")
            print(f"    {FASE1_SCRIPT.relative_to(ROOT)}")
            sys.exit(1)

        W = 62
        print(f"\n{'=' * W}")
        tag = "[SMOKE TEST] " if smoke else ""
        print(f"  {tag}Iniciando medicion — scripts/fase1_benchmark.py")
        print(f"{'=' * W}\n")

        # Construir args para fase1_benchmark.py
        fase1_args: list[str] = []
        if eff_n_per_cat != OFFICIAL_N_PER_CATEGORY:
            fase1_args += ["--max-per-category", str(eff_n_per_cat)]
        if eff_reps != OFFICIAL_N_REPETITIONS:
            fase1_args += ["--repetitions", str(eff_reps)]
        fase1_args += ["--output-dir", str(output_dir)]
        if smoke:
            fase1_args.append("--smoke-test")

        env = {
            **os.environ,
            "GREEN_IA_RUN_ID"  : run_id,
            "GREEN_IA_RUN_TYPE": run_type,
        }

        result = subprocess.run(
            [sys.executable, str(FASE1_SCRIPT)] + fase1_args,
            cwd=str(ROOT),
            env=env,
        )

        status = "completed" if result.returncode == 0 else "failed"
        manifest["status"]       = status
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        path = METADATA_DIR / "experiment_manifest.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False, default=str)

        if result.returncode != 0:
            sys.exit(result.returncode)

        print(f"\n{'=' * W}")
        print(f"  {tag}Fase 1 completada.")
        print(f"  Resultados en: {output_dir.relative_to(ROOT)}/")
        print(f"{'=' * W}\n")

    finally:
        _stop_caffeinate(caff)


if __name__ == "__main__":
    main()

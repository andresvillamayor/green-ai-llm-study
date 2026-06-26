"""
hash_artifacts.py
Proyecto GREEN-IA — Maestria Ciencia de Datos
Universidad Comunero — Paraguay
Autor: Andres Villamayor

Genera huellas digitales SHA256 de los artefactos del experimento y las
persiste en los manifiestos de reproducibilidad.

Artefactos cubiertos:
  - Archivos GGUF (selected_models x selected_quantizations de config.yaml)
      -> SHA256 parcial: primer 1 MB + tamanio total
      -> SHA256 completo si se usa --full-hash
  - data/mt_bench/official/question.jsonl
      -> SHA256 completo
  - data/mt_bench/subset/*.{jsonl,yaml,csv}
      -> SHA256 completo de cada archivo generado
  - config.yaml
      -> SHA256 completo
  - optimization_config.yaml
      -> SHA256 completo

Salida:
  results/metadata/model_hashes.csv
  results/metadata/dataset_manifest.csv

Justificacion del SHA256 parcial para modelos (P11):
  Los archivos .gguf pesan 4-8 GB. Un SHA256 completo tardaria ~60-90 s.
  El SHA256 del primer 1 MB (header GGUF con arquitectura, cuantizacion y
  metadatos) mas el tamanio total identifica inequivocamente la version del
  modelo sin el costo de leer el archivo completo.

Uso:
  python scripts/hash_artifacts.py              # mostrar todo, no guardar
  python scripts/hash_artifacts.py --save       # mostrar + guardar CSVs
  python scripts/hash_artifacts.py --models     # solo modelos GGUF
  python scripts/hash_artifacts.py --dataset    # solo question.jsonl y subset
  python scripts/hash_artifacts.py --config     # solo archivos YAML de config
  python scripts/hash_artifacts.py --full-hash --save  # SHA256 completo + guardar
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML no instalado. Ejecutar: pip install pyyaml")
    sys.exit(1)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import src.metadata_writer as metadata_writer

CONFIG_FILE       = ROOT / "config.yaml"
OPTIM_CONFIG_FILE = ROOT / "optimization_config.yaml"
OFFICIAL_QFILE    = ROOT / "data" / "mt_bench" / "official" / "question.jsonl"
SUBSET_DIR        = ROOT / "data" / "mt_bench" / "subset"
METADATA_DIR      = ROOT / "results" / "metadata"
MODEL_HASHES_CSV  = METADATA_DIR / "model_hashes.csv"
DATASET_CSV       = METADATA_DIR / "dataset_manifest.csv"

# Canonical column schemas come from metadata_writer
MODEL_HASHES_COLS = metadata_writer.MODEL_HASHES_COLS
DATASET_COLS      = metadata_writer.DATASET_MANIFEST_COLS


# ─── Hashing ──────────────────────────────────────────────────────────────────

def sha256_full(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_partial(path: Path, head_bytes: int = 1_048_576) -> str:
    """SHA256 del primer head_bytes del archivo."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(head_bytes))
    return h.hexdigest()


# ─── Lectura de config ────────────────────────────────────────────────────────

def load_config() -> Optional[dict]:
    if not CONFIG_FILE.exists():
        print(f"  AVISO  config.yaml no encontrado")
        return None
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        if not isinstance(cfg, dict):
            print("  AVISO  config.yaml no es un mapping YAML valido")
            return None
        return cfg
    except yaml.YAMLError as exc:
        print(f"  AVISO  config.yaml no es YAML valido: {exc}")
        return None


def model_paths_from_config(cfg: dict) -> list[tuple[str, str, Optional[Path]]]:
    """Devuelve lista de (model_name, quantization, path) segun config.yaml."""
    selected_models = cfg.get("selected_models") or []
    selected_quants = cfg.get("selected_quantizations") or []
    models_block    = cfg.get("models") or {}
    entries: list[tuple[str, str, Optional[Path]]] = []
    for model in selected_models:
        model_def = models_block.get(model) or {}
        for q in selected_quants:
            path_str = model_def.get(q)
            entries.append((model, q, ROOT / str(path_str) if path_str else None))
    return entries


# ─── Modelos GGUF ─────────────────────────────────────────────────────────────

def hash_models(cfg: Optional[dict], full_hash: bool = False) -> list[dict]:
    """
    Compute SHA256 fingerprints for all selected model GGUF files.

    Returns rows with columns matching MODEL_HASHES_COLS:
        model_name, quantization, path, file_size_bytes, sha256,
        hash_type, verified_at.
    """
    if cfg is None:
        print("  (sin config.yaml — no se pueden determinar los modelos)")
        return []

    entries = model_paths_from_config(cfg)
    if not entries:
        print("  (selected_models o selected_quantizations vacio en config.yaml)")
        return []

    hash_label = "SHA256 completo" if full_hash else "SHA256 parcial 1 MB"
    print(f"\n  Modelos GGUF ({hash_label}):")

    rows = metadata_writer.build_model_hashes(cfg, ROOT, full_hash=full_hash)

    for row in rows:
        digest = str(row.get("sha256", ""))
        size_b = row.get("file_size_bytes", "")
        size_s = f"{int(size_b)/1e9:.2f} GB" if size_b != "" else "—"
        ht     = row.get("hash_type", "")
        if "not_found" in ht or "missing" in ht:
            print(f"    FALTA  {row['model_name']} {row['quantization'].upper():<3}  {row['path']}")
        else:
            print(f"    OK     {row['model_name']} {row['quantization'].upper():<3}  "
                  f"{size_s}  {digest[:20]}...")

    return rows


# ─── Dataset y subsets ────────────────────────────────────────────────────────

def _count_jsonl(path: Path) -> tuple[int, Optional[int], Optional[int]]:
    """Devuelve (n_questions, first_question_id, last_question_id)."""
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return 0, None, None
    return len(rows), rows[0].get("question_id"), rows[-1].get("question_id")


def _count_csv_rows(path: Path) -> int:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return max(0, sum(1 for _ in csv.reader(fh)) - 1)


def hash_dataset() -> list[dict]:
    """
    Build per-question dataset manifest rows.

    Reads the official question.jsonl and the prepared subset file,
    then returns one row per question with columns matching
    DATASET_MANIFEST_COLS:
        official_question_file, sha256, subset_file, subset_sha256,
        official_question_id, original_category, internal_category,
        subset_id.
    """
    print("\n  Dataset MT-Bench (manifiesto por pregunta):")

    # Verify official file
    if not OFFICIAL_QFILE.exists():
        print(f"    FALTA  {OFFICIAL_QFILE.relative_to(ROOT)}")
        print(f"           git clone https://github.com/lm-sys/FastChat.git")
        print(f"           cp FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl \\")
        print(f"              data/mt_bench/official/question.jsonl")
    else:
        try:
            n_q, fid, lid = _count_jsonl(OFFICIAL_QFILE)
            print(f"    OK     question.jsonl  {n_q} preguntas  IDs {fid}-{lid}")
        except Exception:
            pass

    # Load questions from subset file (JSONL or YAML)
    questions = _load_subset_questions()
    if not questions:
        print("    (sin subset — ejecutar scripts/prepare_mt_bench_subset.py)")
        return []

    # Find subset file path for the SHA256
    subset_path = _find_subset_path()
    rows = metadata_writer.build_dataset_manifest(
        questions=questions,
        official_file=OFFICIAL_QFILE,
        subset_file=subset_path,
        root=ROOT,
    )
    print(f"    {len(rows)} filas generadas (una por pregunta del subset)")
    return rows


def _find_subset_path() -> Optional[Path]:
    for name in (
        "mt_bench_literal_subset_5_per_category.yaml",
        "mt_bench_literal_subset_5_per_category.jsonl",
    ):
        p = SUBSET_DIR / name
        if p.exists():
            return p
    return None


def _load_subset_questions() -> list[dict]:
    """Load questions from the prepared subset file (YAML or JSONL)."""
    path = _find_subset_path()
    if path is None:
        return []
    try:
        if path.suffix == ".yaml":
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            qs = data.get("questions") or []
        else:
            qs = []
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        qs.append(json.loads(line))
        return qs
    except Exception as exc:
        print(f"    AVISO: no se pudo leer el subset: {exc}")
        return []


# ─── Archivos de configuracion ────────────────────────────────────────────────

def hash_configs() -> list[dict]:
    """Print hashes of config files (not written to dataset_manifest.csv)."""
    print("\n  Configuracion:")
    rows: list[dict] = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for cfg_path in (CONFIG_FILE, OPTIM_CONFIG_FILE):
        if not cfg_path.exists():
            print(f"    FALTA  {cfg_path.relative_to(ROOT)}")
        else:
            digest = sha256_full(cfg_path)
            size   = cfg_path.stat().st_size
            print(f"    OK     {cfg_path.name}  {size} bytes  {digest[:20]}...")
            rows.append({"file": str(cfg_path.relative_to(ROOT)), "sha256": digest,
                         "size_bytes": size, "verified_at": now_iso})
    return rows


# ─── Persistencia en CSV ──────────────────────────────────────────────────────

def save_model_hashes_csv(rows: list[dict]) -> None:
    path = metadata_writer.write_model_hashes(rows, METADATA_DIR)
    print(f"\n  Guardado: {path.relative_to(ROOT)}  ({len(rows)} fila(s))")


def save_dataset_csv(rows: list[dict]) -> None:
    path = metadata_writer.write_dataset_manifest(rows, METADATA_DIR)
    print(f"  Guardado: {path.relative_to(ROOT)}  ({len(rows)} fila(s))")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GREEN-IA: genera huellas SHA256 de artefactos del experimento"
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="Procesar solo archivos GGUF",
    )
    parser.add_argument(
        "--dataset",
        action="store_true",
        help="Procesar solo question.jsonl y archivos subset",
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Procesar solo config.yaml y optimization_config.yaml",
    )
    parser.add_argument(
        "--full-hash",
        action="store_true",
        help="SHA256 completo para modelos GGUF (lento: ~60-90 s por archivo de 4-8 GB)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Guardar resultados en results/metadata/model_hashes.csv y dataset_manifest.csv",
    )
    args = parser.parse_args()

    run_all     = not (args.models or args.dataset or args.config)
    run_models  = run_all or args.models
    run_dataset = run_all or args.dataset
    run_config  = run_all or args.config

    W = 62
    print(f"\n{'=' * W}")
    print(f"  GREEN-IA — Verificacion de artefactos")
    if args.full_hash:
        print(f"  Modo: SHA256 completo para modelos GGUF")
    print(f"{'=' * W}")

    cfg = load_config()

    model_rows:   list[dict] = []
    dataset_rows: list[dict] = []

    if run_models:
        model_rows = hash_models(cfg, full_hash=args.full_hash)

    if run_dataset:
        dataset_rows.extend(hash_dataset())

    if run_config:
        dataset_rows.extend(hash_configs())

    # ── resumen ───────────────────────────────────────────────────────────────
    total_ok = (
        sum(1 for r in model_rows  if r.get("sha256"))
        + sum(1 for r in dataset_rows if r.get("sha256"))
    )
    total_missing = (
        sum(1 for r in model_rows  if not r.get("sha256"))
        + sum(1 for r in dataset_rows if not r.get("sha256"))
    )

    print(f"\n  {'─' * (W - 2)}")
    print(f"  {total_ok} artefacto(s) verificado(s)", end="")
    if total_missing:
        print(f"  /  {total_missing} faltante(s)")
    else:
        print()

    # ── guardar ───────────────────────────────────────────────────────────────
    if args.save:
        if model_rows:
            save_model_hashes_csv(model_rows)
        if dataset_rows:
            save_dataset_csv(dataset_rows)
    else:
        print(f"  (usar --save para escribir los archivos CSV)")

    print(f"{'=' * W}\n")

    if total_missing > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

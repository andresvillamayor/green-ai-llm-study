"""
reproducibility.py
Proyecto GREEN-IA — Maestria Ciencia de Datos
Universidad Comunero — Paraguay

Utilidades de reproducibilidad cientifica para el experimento.

Principio metodologico P11:
  "Guardar metadatos completos de reproducibilidad: hashes de archivos,
  versiones de paquetes, configuracion, manifiesto del dataset y
  reporte del entorno."

Implementa:
  - Huella digital de archivos de modelo (SHA256 parcial: primer 1 MB + tamaño)
    Justificacion del SHA256 parcial:
      Los archivos .gguf pesan 4-8 GB. Un SHA256 completo tardaria ~60-90 s
      por archivo y se ejecutaria antes de cada configuracion. El SHA256 de
      los primeros 1 MB del archivo mas el tamaño total es suficiente para
      identificar unequivocamente la version del archivo (el header GGUF
      contiene arquitectura, cuantizacion y metadatos).
      Si se requiere el SHA256 completo, usar `--full-hash` en el script.

  - Freeze de dependencias (pip freeze) para replicar el entorno exacto.

  - Hash de la configuracion del experimento para detectar cambios entre
    corridas y validar que Fase 2 usa los mismos archivos que Fase 1.

  - IDs estables por fila para permitir la reanudacion del experimento
    sin duplicar mediciones (P14).

  - Reporte de entorno completo guardado como JSON junto al CSV.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ─── Model file fingerprint ───────────────────────────────────────────────────

def model_file_info(path: Path) -> dict:
    """
    Retorna tamano y SHA256 parcial de un archivo de modelo como dict estructurado.

    Campos separados permiten consultas programaticas sin parsear la cadena
    de model_fingerprint(). SHA256 es del primer 1 MB (mismo criterio que
    model_fingerprint: suficiente para identificar el header GGUF + tamanio total).

    Retorna:
        dict con path (solo nombre), size_bytes, y sha256_1mb (24 hex chars).
    """
    if not path.exists():
        return {"path": path.name, "size_bytes": 0, "sha256_1mb": "file_not_found"}
    size = path.stat().st_size
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(1024 * 1024))
    h.update(str(size).encode())
    return {
        "path"      : path.name,
        "size_bytes": size,
        "sha256_1mb": h.hexdigest()[:24],
    }


def model_fingerprint(path: Path, full_hash: bool = False) -> str:
    """
    Calcula la huella digital del archivo de modelo.

    Modo rapido (default): SHA256 de los primeros 1 MB + tamanio total.
    Modo completo (full_hash=True): SHA256 completo del archivo.

    Retorna una cadena descriptiva con el algoritmo, hash y tamaño.
    """
    if not path.exists():
        return "file_not_found"

    size = path.stat().st_size
    h = hashlib.sha256()

    if full_hash:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return f"sha256_full_{h.hexdigest()}_bytes_{size}"
    else:
        with open(path, "rb") as f:
            h.update(f.read(1024 * 1024))  # primeros 1 MB (header GGUF)
        h.update(str(size).encode())
        return f"sha256_1mb+size_{h.hexdigest()[:24]}_bytes_{size}"


# ─── Configuration fingerprint ────────────────────────────────────────────────

def config_fingerprint(config: dict) -> str:
    """
    SHA256 de la configuracion del experimento serializada.

    Permite detectar si entre corridas cambio algun parametro.
    El hash es de 16 hex chars (suficiente para identificar cambios).
    """
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


# ─── Package versions ─────────────────────────────────────────────────────────

def get_pip_freeze() -> str:
    """Retorna la salida de `pip freeze` como cadena."""
    try:
        return subprocess.check_output(
            ["pip", "freeze"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return "pip_freeze_unavailable"


def get_key_versions() -> dict:
    """
    Retorna las versiones de los paquetes clave del experimento.
    Complementa pip_freeze con acceso directo para legibilidad.
    """
    versions: dict[str, str] = {}
    for pkg in [
        "llama_cpp",
        "codecarbon",
        "pandas",
        "numpy",
        "psutil",
        "requests",
        "scipy",
    ]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not_installed"
    return versions


# ─── Stable row ID ────────────────────────────────────────────────────────────

def stable_row_id(*components) -> str:
    """
    ID deterministico para una fila de medicion.

    Basado en SHA256 de los componentes que identifican unequivocamente
    una medicion: experiment_tag | model | quant | device | question_id
                                | turn | repetition.

    Uso en resumabilidad (P14): al inicio del experimento se cargan
    los IDs ya medidos del CSV parcial. Las filas cuyo ID ya existe
    se saltan sin re-medirse.

    Retorna 20 caracteres hex (80 bits de entropia — suficiente para
    un dataset de decenas de miles de filas sin colisiones esperadas).
    """
    key = "|".join(str(c) for c in components)
    return hashlib.sha256(key.encode()).hexdigest()[:20]


# ─── Environment report ───────────────────────────────────────────────────────

def build_environment_report(
    hw_profile=None,
    experiment_config: Optional[dict] = None,
    dataset_manifest: Optional[list] = None,
    model_fingerprints: Optional[dict] = None,
) -> dict:
    """
    Construye el reporte completo del entorno de ejecucion.

    Guardado como JSON junto al CSV de resultados para permitir
    la reproduccion exacta del experimento (P11).

    Contenido:
      - timestamp_utc: momento de inicio del experimento
      - platform: OS, version, arquitectura
      - python_version: version exacta de Python
      - key_package_versions: versiones de paquetes criticos
      - pip_freeze: salida completa de pip freeze
      - hardware_profile: perfil de hardware detectado
      - experiment_config: parametros del experimento
      - config_fingerprint: hash de la config para detectar cambios
      - dataset_manifest: checksums de los prompts usados
      - model_fingerprints: huellas digitales de los archivos GGUF
    """
    config_fp = (
        config_fingerprint(experiment_config)
        if experiment_config
        else "no_config"
    )

    report = {
        "timestamp_utc"          : datetime.now(timezone.utc).isoformat(),
        "platform"               : platform.platform(),
        "python_version"         : platform.python_version(),
        "python_implementation"  : platform.python_implementation(),
        "architecture"           : platform.machine(),
        "processor"              : platform.processor(),
        "key_package_versions"   : get_key_versions(),
        "pip_freeze"             : get_pip_freeze(),
        "hardware_profile"       : hw_profile.to_dict() if hw_profile else None,
        "experiment_config"      : experiment_config,
        "config_fingerprint"     : config_fp,
        "dataset_manifest"       : dataset_manifest,
        "model_fingerprints"     : model_fingerprints,
    }

    return report


def save_environment_report(report: dict, output_dir: Path, stem: str) -> Path:
    """
    Guarda el reporte de entorno como JSON junto al CSV de resultados.

    Parametros:
        report: dict retornado por build_environment_report().
        output_dir: directorio donde guardar el archivo.
        stem: prefijo del nombre del archivo (mismo que el CSV).

    Retorna la ruta del archivo guardado.
    """
    ruta = output_dir / f"{stem}_environment.json"
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    return ruta


# ─── Resumability helpers ─────────────────────────────────────────────────────

def load_completed_ids(results_dir: Path, experiment_tag: str) -> set[str]:
    """
    Carga los row_ids ya medidos de CSVs existentes con el mismo experiment_tag.

    Permite reanudar el experimento sin re-medir lo que ya esta completo (P14).
    Solo considera CSVs con el mismo experiment_tag para evitar mezclar
    resultados de experimentos distintos.
    """
    try:
        import pandas as pd
    except ImportError:
        return set()

    completed: set[str] = set()
    for csv_path in sorted(results_dir.glob("*.csv")):
        try:
            df = pd.read_csv(
                csv_path,
                usecols=["row_id", "experiment_tag"],
                dtype=str,
            )
            mask = df["experiment_tag"] == experiment_tag
            completed.update(df.loc[mask, "row_id"].dropna().tolist())
        except Exception:
            continue

    if completed:
        print(f"  [resume] {len(completed)} mediciones ya completadas cargadas")

    return completed

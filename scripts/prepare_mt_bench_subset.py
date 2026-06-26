"""
prepare_mt_bench_subset.py
Proyecto GREEN-IA — Maestria Ciencia de Datos
Universidad Comunero — Paraguay
Autor: Andres Villamayor

Prepara el subconjunto literal de MT-Bench para el experimento.

Lee config.yaml, valida el archivo oficial question.jsonl,
selecciona exactamente N preguntas por categoria segun la
estrategia configurada, y guarda tres artefactos:

  data/mt_bench/subset/mt_bench_literal_subset_5_per_category.jsonl
  data/mt_bench/subset/mt_bench_literal_subset_5_per_category.yaml
  data/mt_bench/subset/mt_bench_subset_manifest.csv

Principios metodologicos:
  P5: Texto de cada turno preservado verbatim — sin modificaciones.
  P6: Texto en ingles — sin traduccion.
  P7: Si question.jsonl no existe, el script aborta. No se inventan
      ni se usan prompts de respaldo.
  P8: Estructura de 2 turnos preservada por pregunta.

Uso:
  python scripts/prepare_mt_bench_subset.py
  python scripts/prepare_mt_bench_subset.py --dry-run
  python scripts/prepare_mt_bench_subset.py --stats
  python scripts/prepare_mt_bench_subset.py --force
  python scripts/prepare_mt_bench_subset.py --config otra_config.yaml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    print("  ERROR: PyYAML no instalado. Ejecutar: pip install pyyaml")
    sys.exit(1)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CONFIG_FILE   = ROOT / "config.yaml"
OFFICIAL_FILE = ROOT / "data" / "mt_bench" / "official" / "question.jsonl"
SUBSET_DIR    = ROOT / "data" / "mt_bench" / "subset"

OFFICIAL_CATEGORIES = frozenset({
    "writing", "roleplay", "extraction", "reasoning",
    "math", "coding", "stem", "humanities",
})

REQUIRED_FIELDS = ("question_id", "category", "turns")

MANIFEST_PREVIEW_LEN = 120

_OBTAIN_MSG = (
    "\n"
    "  Obtener el archivo desde el repositorio oficial de FastChat:\n\n"
    "    git clone https://github.com/lm-sys/FastChat.git\n"
    "    mkdir -p data/mt_bench/official\n"
    "    mkdir -p data/mt_bench/subset\n"
    "    cp FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl \\\n"
    "       data/mt_bench/official/question.jsonl\n\n"
    "  El experimento no puede continuar sin este archivo.\n"
    "  No se usan prompts de respaldo, sinteticos ni parafraseados.\n"
    "  Solo los prompts verbatim del dataset oficial garantizan\n"
    "  comparabilidad con la literatura (Zheng et al. NeurIPS 2023)."
)


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    if not path.exists():
        print(f"\n  ERROR: {path.relative_to(ROOT)} no encontrado.")
        print("  Crear desde la plantilla: cp config.yaml.example config.yaml")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg


def extract_prompt_dataset(cfg: dict) -> dict:
    pd_cfg = cfg.get("prompt_dataset") or {}
    if not pd_cfg:
        print("  AVISO: bloque 'prompt_dataset' no encontrado en config.yaml — usando valores por defecto")
    return pd_cfg


# ─── SHA256 ───────────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── Carga ────────────────────────────────────────────────────────────────────

def load_official(path: Path) -> list[dict]:
    """
    Carga question.jsonl. Aborta si el archivo no existe (P7).
    Parsea cada linea no vacia como JSON. No modifica el contenido.
    """
    if not path.exists():
        print(f"\n  ERROR [P7]: {path.relative_to(ROOT)} no encontrado.")
        print(_OBTAIN_MSG)
        sys.exit(1)

    rows: list[dict] = []
    parse_errors: list[str] = []

    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, 1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                parse_errors.append(f"    linea {lineno}: {exc}")

    if parse_errors:
        print(f"\n  ERROR: {len(parse_errors)} error(s) JSON en {path.name}:")
        for msg in parse_errors:
            print(msg)
        sys.exit(1)

    return rows


# ─── Validacion ───────────────────────────────────────────────────────────────

def validate_official(questions: list[dict]) -> None:
    """
    Valida la estructura de cada fila.

    Verifica por fila:
      - Campos requeridos: question_id, category, turns
      - turns es una lista con exactamente 2 elementos
      - Cada turno es una cadena no vacia
      - category es una de las 8 categorias oficiales

    Verifica globalmente:
      - Las 8 categorias oficiales estan presentes

    Aborta con sys.exit(1) si se encuentra cualquier error.
    """
    errors: list[str] = []

    for i, q in enumerate(questions, 1):
        qid = q.get("question_id", f"<fila {i}>")

        for field in REQUIRED_FIELDS:
            if field not in q:
                errors.append(f"    question_id={qid}: campo requerido '{field}' ausente")

        if "turns" in q:
            turns = q["turns"]
            if not isinstance(turns, list):
                errors.append(
                    f"    question_id={qid}: 'turns' debe ser lista, "
                    f"obtenido {type(turns).__name__}"
                )
            elif len(turns) != 2:
                errors.append(
                    f"    question_id={qid}: 'turns' debe tener exactamente 2 elementos, "
                    f"obtenido {len(turns)}"
                )
            else:
                for t_idx, turn in enumerate(turns, 1):
                    if not isinstance(turn, str):
                        errors.append(
                            f"    question_id={qid}: turno {t_idx} debe ser str, "
                            f"obtenido {type(turn).__name__}"
                        )
                    elif not turn.strip():
                        errors.append(
                            f"    question_id={qid}: turno {t_idx} esta vacio o es solo espacios"
                        )

        if "category" in q:
            cat = str(q["category"]).lower()
            if cat not in OFFICIAL_CATEGORIES:
                errors.append(
                    f"    question_id={qid}: categoria desconocida '{q['category']}' — "
                    f"validas: {sorted(OFFICIAL_CATEGORIES)}"
                )

    if errors:
        print(f"\n  ERROR: {len(errors)} error(s) de validacion en question.jsonl:")
        for msg in errors[:20]:
            print(msg)
        if len(errors) > 20:
            print(f"    ... ({len(errors) - 20} errores adicionales)")
        sys.exit(1)

    present = {str(q["category"]).lower() for q in questions}
    missing = OFFICIAL_CATEGORIES - present
    if missing:
        print(f"\n  ERROR: categorias faltantes en el archivo oficial: {sorted(missing)}")
        print(_OBTAIN_MSG)
        sys.exit(1)


# ─── Seleccion ────────────────────────────────────────────────────────────────

def select_subset(
    questions: list[dict],
    n_per_category: int,
    selected_categories: list[str],
    strategy: str,
    random_seed: int,
    category_mapping: dict[str, str],
) -> list[dict]:
    """
    Selecciona exactamente n_per_category preguntas de cada categoria.

    Estrategias:
      first_n_per_category  — primeras n en orden del archivo (determinista)
      random_n_per_category — muestra aleatoria reproducible usando random_seed

    El texto de cada turno se preserva verbatim (P5, P6).
    El question_id oficial se preserva sin modificacion.
    """
    valid_strategies = {"first_n_per_category", "random_n_per_category"}
    if strategy not in valid_strategies:
        print(f"\n  ERROR: selection_strategy '{strategy}' desconocida.")
        print(f"  Valores validos: {sorted(valid_strategies)}")
        sys.exit(1)

    by_category: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        cat = str(q["category"]).lower()
        by_category[cat].append(q)

    subset: list[dict] = []
    subset_id = 1

    for raw_cat in selected_categories:
        cat = raw_cat.lower()

        if cat not in OFFICIAL_CATEGORIES:
            print(f"\n  ERROR: categoria seleccionada '{raw_cat}' no es oficial.")
            print(f"  Categorias validas: {sorted(OFFICIAL_CATEGORIES)}")
            sys.exit(1)

        pool = by_category.get(cat, [])

        if len(pool) < n_per_category:
            print(
                f"\n  ERROR: categoria '{cat}' tiene {len(pool)} preguntas "
                f"en el archivo oficial, se necesitan exactamente {n_per_category}."
            )
            sys.exit(1)

        if strategy == "first_n_per_category":
            chosen = pool[:n_per_category]
        else:
            rng = random.Random(random_seed)
            chosen = rng.sample(pool, n_per_category)
            chosen = sorted(chosen, key=lambda q: q["question_id"])

        internal_cat = category_mapping.get(cat, cat)

        for q in chosen:
            # Preserve turns verbatim — P5, P6
            turn1: str = q["turns"][0]
            turn2: str = q["turns"][1]

            subset.append({
                "subset_id"          : subset_id,
                "question_id"        : int(q["question_id"]),
                "category"           : str(q["category"]),
                "internal_category"  : internal_cat,
                "turns"              : [turn1, turn2],
                "reference"          : q.get("reference", None),
                "source_file"        : str(OFFICIAL_FILE.relative_to(ROOT)),
                "selection_strategy" : strategy,
            })
            subset_id += 1

    return subset


# ─── Salidas ──────────────────────────────────────────────────────────────────

def save_jsonl(subset: list[dict], path: Path) -> None:
    """
    Guarda el subconjunto como JSONL.
    Serializa con ensure_ascii=False — el texto de los turnos no se modifica.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in subset:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  guardado: {path.relative_to(ROOT)}  ({len(subset)} filas)")


def save_yaml_output(
    subset: list[dict],
    path: Path,
    source_sha256: str,
) -> None:
    """
    Guarda el subconjunto como YAML.
    allow_unicode=True: preserva caracteres Unicode sin escaping.
    El contenido textual de los turnos no se altera durante la serializacion.
    """
    doc = {
        "generated_at"   : datetime.now(timezone.utc).isoformat(),
        "source_file"    : str(OFFICIAL_FILE.relative_to(ROOT)),
        "source_sha256"  : source_sha256,
        "reference"      : "Zheng et al. (2023). NeurIPS 2023. arXiv:2306.05685v4",
        "n_questions"    : len(subset),
        "notes"          : [
            "P5: turns text is verbatim — not modified.",
            "P6: turns text is in English — not translated.",
            "P7: only official MT-Bench prompts are used.",
            "P8: 2-turn structure preserved per question.",
        ],
        "questions"      : subset,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(
            doc,
            fh,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
            width=10000,
        )
    print(f"  guardado: {path.relative_to(ROOT)}")


def save_manifest(
    subset: list[dict],
    path: Path,
    source_sha256: str,
) -> None:
    """
    Guarda el manifiesto CSV con los campos requeridos.

    Campos:
      subset_id, official_question_id, original_category, internal_category,
      turn_1_preview, turn_2_preview, source_file, selection_strategy, source_sha256

    turn_1_preview y turn_2_preview son los primeros MANIFEST_PREVIEW_LEN
    caracteres de cada turno. El texto no es modificado — el preview es
    una subcadena truncada del texto verbatim.
    """
    FIELDNAMES = [
        "subset_id",
        "official_question_id",
        "original_category",
        "internal_category",
        "turn_1_preview",
        "turn_2_preview",
        "source_file",
        "selection_strategy",
        "source_sha256",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in subset:
            t1, t2 = row["turns"]
            writer.writerow({
                "subset_id"            : row["subset_id"],
                "official_question_id" : row["question_id"],
                "original_category"    : row["category"],
                "internal_category"    : row["internal_category"],
                "turn_1_preview"       : t1[:MANIFEST_PREVIEW_LEN],
                "turn_2_preview"       : t2[:MANIFEST_PREVIEW_LEN],
                "source_file"          : row["source_file"],
                "selection_strategy"   : row["selection_strategy"],
                "source_sha256"        : source_sha256,
            })
    print(f"  guardado: {path.relative_to(ROOT)}  ({len(subset)} filas)")


# ─── Estadisticas ─────────────────────────────────────────────────────────────

def print_stats(questions: list[dict], label: str = "dataset") -> None:
    cat_counts = Counter(str(q.get("category", "?")).lower() for q in questions)
    qids = [q.get("question_id") for q in questions if q.get("question_id")]
    print(f"\n  {label}: {len(questions)} preguntas")
    if qids:
        print(f"  IDs: {min(qids)}-{max(qids)}")
    print(f"  Distribucion por categoria:")
    for cat in sorted(cat_counts):
        print(f"    {cat:<22}  {cat_counts[cat]:>3}")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepara el subconjunto literal de MT-Bench para GREEN-IA. "
            "Lee config.yaml, valida question.jsonl y guarda JSONL + YAML + CSV."
        )
    )
    parser.add_argument(
        "--config",
        default=str(CONFIG_FILE),
        metavar="PATH",
        help=f"Ruta a config.yaml (default: {CONFIG_FILE.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Cargar y validar sin escribir archivos de salida",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Mostrar distribucion por categoria del archivo oficial y del subconjunto",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobreescribir archivos de subconjunto si ya existen",
    )
    args = parser.parse_args()

    print(f"\n{'='*62}")
    print(f"  GREEN-IA — Preparacion del subconjunto MT-Bench")
    print(f"{'='*62}\n")

    # ── 1. Leer config.yaml ───────────────────────────────────────────────────
    cfg = load_config(Path(args.config))
    pd_cfg = extract_prompt_dataset(cfg)

    n_per_category       = int(pd_cfg.get("questions_per_category", 5))
    strategy             = str(pd_cfg.get("selection_strategy", "first_n_per_category"))
    random_seed          = int(pd_cfg.get("random_seed", 42))
    selected_categories  = list(pd_cfg.get("selected_categories") or sorted(OFFICIAL_CATEGORIES))
    category_mapping     = dict(pd_cfg.get("category_mapping") or {})

    out_jsonl = ROOT / str(pd_cfg.get(
        "subset_output_jsonl",
        "data/mt_bench/subset/mt_bench_literal_subset_5_per_category.jsonl",
    ))
    out_yaml  = ROOT / str(pd_cfg.get(
        "subset_output_yaml",
        "data/mt_bench/subset/mt_bench_literal_subset_5_per_category.yaml",
    ))
    out_csv   = ROOT / str(pd_cfg.get(
        "subset_manifest_csv",
        "data/mt_bench/subset/mt_bench_subset_manifest.csv",
    ))

    print(f"  config            : {Path(args.config).relative_to(ROOT)}")
    print(f"  selection_strategy: {strategy}")
    print(f"  questions/category: {n_per_category}")
    print(f"  categories        : {selected_categories}")
    print(f"  random_seed       : {random_seed}")

    # ── 2. Verificar y cargar question.jsonl ──────────────────────────────────
    print(f"\n  Verificando: {OFFICIAL_FILE.relative_to(ROOT)}")
    questions = load_official(OFFICIAL_FILE)
    print(f"  cargadas: {len(questions)} preguntas")

    # ── 3. Validar JSONL ──────────────────────────────────────────────────────
    validate_official(questions)

    if args.stats:
        print_stats(questions, label="dataset oficial completo")

    # ── 4. Seleccionar subconjunto ────────────────────────────────────────────
    expected_total = n_per_category * len(selected_categories)
    print(f"\n  Seleccionando {n_per_category} × {len(selected_categories)} categorias "
          f"= {expected_total} preguntas ...")

    subset = select_subset(
        questions,
        n_per_category=n_per_category,
        selected_categories=selected_categories,
        strategy=strategy,
        random_seed=random_seed,
        category_mapping=category_mapping,
    )

    assert len(subset) == expected_total, (
        f"ERROR interno: se seleccionaron {len(subset)} preguntas, "
        f"se esperaban {expected_total}"
    )

    if args.stats:
        print_stats(subset, label="subconjunto seleccionado")

    # ── 5. Verificar que no se modifica el texto (doble check) ────────────────
    for item in subset:
        orig = next(q for q in questions if q["question_id"] == item["question_id"])
        assert item["turns"][0] == orig["turns"][0], (
            f"ERROR [P5]: turno 1 modificado en question_id={item['question_id']}"
        )
        assert item["turns"][1] == orig["turns"][1], (
            f"ERROR [P5]: turno 2 modificado en question_id={item['question_id']}"
        )

    if args.dry_run:
        print(f"\n  Modo dry-run — archivos no escritos.")
        print(f"  Salidas planeadas:")
        print(f"    {out_jsonl.relative_to(ROOT)}")
        print(f"    {out_yaml.relative_to(ROOT)}")
        print(f"    {out_csv.relative_to(ROOT)}")
        print(f"\n{'='*62}\n")
        return

    # ── 6. Verificar sobreescritura ───────────────────────────────────────────
    existing = [p for p in (out_jsonl, out_yaml, out_csv) if p.exists()]
    if existing and not args.force:
        print(f"\n  Los siguientes archivos ya existen:")
        for p in existing:
            print(f"    {p.relative_to(ROOT)}")
        print(f"\n  Usar --force para sobreescribir.")
        sys.exit(1)

    # ── 7. Calcular SHA256 del archivo fuente ─────────────────────────────────
    source_sha256 = sha256_file(OFFICIAL_FILE)
    print(f"\n  SHA256 fuente: {source_sha256}")

    # ── 8. Guardar salidas ────────────────────────────────────────────────────
    print()
    save_jsonl(subset, out_jsonl)
    save_yaml_output(subset, out_yaml, source_sha256)
    save_manifest(subset, out_csv, source_sha256)

    print(f"\n  Subconjunto listo: {len(subset)} preguntas.")
    print(f"  SHA256 fuente   : {source_sha256}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()

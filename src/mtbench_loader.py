"""
mtbench_loader.py
Proyecto GREEN-IA — Maestria Ciencia de Datos
Universidad Comunero — Paraguay

Cargador del dataset oficial MT-Bench (Zheng et al. NeurIPS 2023).

Principios metodologicos que implementa este modulo:
  P5: No modificar los prompts oficiales de MT-Bench.
  P6: No traducir los prompts oficiales de MT-Bench.
  P7: No inventar prompts si el dataset oficial esta ausente.
  P8: Separar resultados a nivel de turno de resultados a nivel de
      conversacion (este modulo provee la estructura de 2 turnos).

Fuente del dataset:
  Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and
  Chatbot Arena. NeurIPS 2023. arXiv:2306.05685v4.
  Dataset: https://github.com/lm-sys/FastChat/tree/main/
           fastchat/llm_judge/data/mt_bench/question.jsonl
  Licencia: Apache 2.0

Dataset oficial:
  80 preguntas, IDs 81–160
  8 categorias: writing, roleplay, reasoning, math,
                coding, extraction, stem, humanities
  2 turnos por pregunta (P8: turn-level vs conversation-level)

IMPORTANTE: este modulo carga los prompts verbatim.
  No aplica ningun preprocesamiento, traduccion, modificacion o
  expansion a los textos. Si el archivo de dataset no existe,
  lanza FileNotFoundError. No usa prompts de respaldo ni inventa
  preguntas alternativas (P7).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Ruta canonica del dataset oficial — no modificar
DATASET_FILE = Path(__file__).parent.parent / "data" / "mt_bench" / "official" / "question.jsonl"

# Numero esperado de preguntas en el dataset oficial
MT_BENCH_N_QUESTIONS = 80

# Primer question_id esperado — invariante del dataset oficial
MT_BENCH_FIRST_QID = 81

# Categorias oficiales MT-Bench (Seccion 2.2, Zheng et al. 2023)
MT_BENCH_CATEGORIES = [
    "writing",
    "roleplay",
    "reasoning",
    "math",
    "coding",
    "extraction",
    "stem",
    "humanities",
]

# Mapeo opcional de nombres internos de categoria.
# original_category se preserva siempre en el campo homónimo del CSV.
# El campo category usa el nombre mapeado para agrupaciones internas.
CATEGORY_MAP: dict[str, str] = {
    "humanities": "humanities_social_sciences",
}


def load_questions(
    dataset_path: Path = DATASET_FILE,
    categories: Optional[list[str]] = None,
    max_per_category: Optional[int] = None,
) -> list[dict]:
    """
    Carga las preguntas oficiales de MT-Bench desde el archivo JSONL.

    Las preguntas se cargan verbatim (P5, P6): sin modificacion, sin
    traduccion, sin normalizacion de texto.

    Si el archivo no existe, lanza FileNotFoundError con instrucciones
    para descargarlo (P7: no se usan preguntas de respaldo).

    Parametros:
        dataset_path: ruta al archivo question.jsonl oficial.
        categories: si se especifica, solo retorna estas categorias.
        max_per_category: si se especifica, limita a N preguntas por categoria.

    Retorna:
        list de dicts con:
          question_id       : int   (81–160)
          original_category : str   categoria verbatim del JSONL oficial (ej. "humanities")
          category          : str   categoria mapeada (ej. "humanities_social_sciences")
          turn1             : str   texto del turno 1 (verbatim)
          turn2             : str   texto del turno 2 (verbatim)
          reference         : str | None  respuesta de referencia si existe

    Raises:
        FileNotFoundError: si el archivo no existe.
        ValueError: si el archivo no es el dataset oficial esperado.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"\n"
            f"  ERROR [P7]: Dataset MT-Bench oficial no encontrado.\n"
            f"  Ruta esperada: {dataset_path}\n\n"
            f"  Obtenerlo desde el repositorio oficial de FastChat:\n\n"
            f"    git clone https://github.com/lm-sys/FastChat.git\n"
            f"    mkdir -p data/mt_bench/official\n"
            f"    cp FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl \\\n"
            f"       data/mt_bench/official/question.jsonl\n\n"
            f"  El experimento no puede continuar sin este archivo.\n"
            f"  No se usan prompts de respaldo, sinteticos ni parafraseados.\n"
            f"  Solo los prompts verbatim del dataset oficial garantizan\n"
            f"  comparabilidad con la literatura (Zheng et al. NeurIPS 2023)."
        )

    target_cats: Optional[set[str]] = (
        {c.lower() for c in categories} if categories else None
    )

    questions: list[dict] = []
    with open(dataset_path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue

            try:
                q = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"  ERROR: linea {i+1} del dataset no es JSON valido: {e}"
                )

            # Validacion de integridad en la primera linea
            if i == 0:
                if int(q.get("question_id", -1)) != MT_BENCH_FIRST_QID:
                    raise ValueError(
                        f"  ERROR: el archivo no parece ser el dataset oficial.\n"
                        f"  Se esperaba question_id={MT_BENCH_FIRST_QID} en la primera linea.\n"
                        f"  Se encontro: question_id={q.get('question_id')}.\n"
                        f"  Obtener el dataset oficial con:\n"
                        f"    git clone https://github.com/lm-sys/FastChat.git\n"
                        f"    cp FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl \\\n"
                        f"       data/mt_bench/official/question.jsonl"
                    )

            original_cat = q.get("category", "").lower()
            if target_cats and original_cat not in target_cats:
                continue

            turns = q.get("turns", [])
            if len(turns) < 2:
                continue

            # CATEGORY_MAP: mapeo opcional de nombres internos.
            # original_category preserva siempre el valor verbatim del JSONL.
            mapped_cat = CATEGORY_MAP.get(original_cat, original_cat)

            questions.append(
                {
                    "question_id"      : int(q["question_id"]),
                    "original_category": original_cat,   # verbatim del JSONL oficial
                    "category"         : mapped_cat,     # nombre interno (puede ser mapeado)
                    "turn1"            : turns[0],       # verbatim — P5, P6
                    "turn2"            : turns[1],       # verbatim — P5, P6
                    "reference"        : q.get("reference", None),
                }
            )

    if not questions:
        raise ValueError(
            f"  ERROR: no se cargaron preguntas desde {dataset_path}.\n"
            f"  Verifica que el archivo sea el dataset oficial MT-Bench."
        )

    # Limitar por categoria si se especifico
    if max_per_category is not None:
        count: dict[str, int] = defaultdict(int)
        filtered: list[dict] = []
        for q in questions:
            if count[q["category"]] < max_per_category:
                filtered.append(q)
                count[q["category"]] += 1
        questions = filtered

    return questions


def format_turn2_prompt(turn1_question: str, turn1_response: str, turn2_question: str) -> str:
    """
    Construye el prompt para el segundo turno de MT-Bench.

    Formato: incluye el historial completo de la conversacion para que
    el modelo pueda dar una respuesta coherente con el turno 1.

    Este formato se aplica de forma consistente en todos los modelos
    para garantizar comparabilidad (no se usan templates de chat
    especificos por modelo, ya que los archivos GGUF se usan via
    la interfaz de completado sin template de sistema).

    Principio P5/P6: el texto de turn2_question es verbatim del dataset oficial.
    """
    return (
        f"[Turn 1 Question]\n{turn1_question}\n\n"
        f"[Turn 1 Answer]\n{turn1_response}\n\n"
        f"[Turn 2 Question]\n{turn2_question}"
    )


def compute_manifest(questions: list[dict]) -> list[dict]:
    """
    Calcula el manifiesto del dataset para trazabilidad de reproducibilidad.

    El manifiesto incluye hashes SHA256 de cada turno para verificar que
    los prompts usados son exactamente los del dataset oficial, sin que
    sea necesario almacenar el texto completo en cada fila del CSV.

    Referencia P11: reproducibility metadata — dataset manifest.
    """
    return [
        {
            "question_id"      : q["question_id"],
            "original_category": q["original_category"],
            "category"         : q["category"],
            "turn1_sha256"     : hashlib.sha256(q["turn1"].encode("utf-8")).hexdigest(),
            "turn2_sha256"     : hashlib.sha256(q["turn2"].encode("utf-8")).hexdigest(),
            "turn1_length"     : len(q["turn1"]),
            "turn2_length"     : len(q["turn2"]),
        }
        for q in questions
    ]


def questions_per_category(questions: list[dict]) -> dict[str, int]:
    """Retorna el numero de preguntas por categoria."""
    count: dict[str, int] = defaultdict(int)
    for q in questions:
        count[q["category"]] += 1
    return dict(count)


def dataset_file_sha256(dataset_path: Path = DATASET_FILE) -> dict:
    """
    SHA256 completo y tamano del archivo dataset oficial (question.jsonl).

    Registrado en el reporte de entorno para verificar que el archivo
    no cambio entre sesiones de medicion (estabilidad del dataset).

    Retorna:
        dict con sha256, size_bytes y filename del archivo.
        Si el archivo no existe, retorna sha256 = "file_not_found".
    """
    if not dataset_path.exists():
        return {
            "sha256"    : "file_not_found",
            "size_bytes": 0,
            "filename"  : dataset_path.name,
        }
    h = hashlib.sha256()
    size = dataset_path.stat().st_size
    with open(dataset_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return {
        "sha256"    : h.hexdigest(),
        "size_bytes": size,
        "filename"  : dataset_path.name,
    }


def selected_question_ids(questions: list[dict]) -> list[int]:
    """
    Lista ordenada de question_ids incluidos en el subconjunto del experimento.

    Registrado en el reporte de entorno para identificar exactamente
    que preguntas se usaron, independientemente del orden de ejecucion.
    """
    return sorted(q["question_id"] for q in questions)

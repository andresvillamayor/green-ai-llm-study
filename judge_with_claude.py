"""
judge_with_claude.py
GREEN-IA — Maestria Ciencia de Datos
Universidad Comunero — Paraguay
Autor: Andres Villamayor

Evaluacion LLM-as-a-Judge de conversaciones completas MT-Bench de dos turnos.
Lee conversation_results.csv (generado por fase1_benchmark.py / fase2_optimizacion.py).
Guarda resultados en results/judge/judge_results.{csv,jsonl}.

Principios metodologicos:
  P17  El puntaje se denomina "Claude-based MT-Bench-style quality score".
       NO es el score oficial MT-Bench (GPT-4-based, Zheng et al. NeurIPS 2023).
  P4   Este script corre separado del benchmark. No se mide energia local aqui.

Metodo: Single Answer Grading — LLM-as-a-Judge (Zheng et al. NeurIPS 2023),
        extendido a evaluacion de conversacion completa de dos turnos.

Juez: claude-sonnet-4-6 via Anthropic API (requests, sin SDK).
      El paper usa GPT-4. Claude evalua Llama y Qwen, sin sesgo de
      auto-preferencia (Panickssery et al. 2024, arXiv:2404.13076).

El juez recibe los CUATRO textos de la conversacion:
  - Pregunta turno 1  (turn_1_text)
  - Respuesta turno 1 (response_1_text)
  - Pregunta turno 2  (turn_2_text)
  - Respuesta turno 2 (response_2_text)
y devuelve un JSON con 9 puntajes (1-10) y 3 campos cualitativos.

Uso:
  export ANTHROPIC_API_KEY=sk-ant-...

  python judge_with_claude.py                       # modo normal
  python judge_with_claude.py --phase 1             # solo fase 1
  python judge_with_claude.py --phase 2             # solo fase 2
  python judge_with_claude.py --dry-run             # sin llamadas API
  python judge_with_claude.py --max-rows 20         # limite de filas (pruebas)
  python judge_with_claude.py --input alt.csv       # archivo alternativo
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def _rel(p: Path) -> Path:
    """Display path relative to ROOT when possible, absolute otherwise."""
    try:
        return p.relative_to(ROOT)
    except ValueError:
        return p

# ─── Constantes ───────────────────────────────────────────────────────────────

JUDGE_MODEL       = "claude-sonnet-4-6"
JUDGE_SCORE_LABEL = "Claude-based MT-Bench-style quality score"
JUDGE_PROMPT_MODE = "multi_turn"

MAX_TOKENS        = 1024
DELAY_S           = 0.5
MAX_RETRIES       = 3
RETRY_WAIT_BASE_S = 2.0    # backoff: base * 2^(attempt-1)
API_TIMEOUT_S     = 45

DEFAULT_INPUT_CSV  = ROOT / "results" / "raw" / "conversation_results.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "judge"
OUTPUT_CSV         = "judge_results.csv"
OUTPUT_JSONL       = "judge_results.jsonl"

# Esquema de salida — orden exacto requerido
JUDGE_COLS = [
    "conversation_id",
    "phase",
    "model_name",
    "model_type",
    "quantization",
    "hardware_profile",
    "execution_device",
    "subset_id",
    "official_question_id",
    "original_category",
    "internal_category",
    "repetition",
    "score",
    "correctness",
    "instruction_following",
    "relevance",
    "completeness",
    "clarity",
    "conciseness",
    "usefulness",
    "multi_turn_coherence",
    "main_strengths",
    "main_weaknesses",
    "final_comment",
    "judge_model",
    "judge_timestamp",
    "judge_status",
    "judge_error_message",
    "judge_prompt_mode",
    "reference_guided",
]

_SCORE_FIELDS = [
    "score",
    "correctness",
    "instruction_following",
    "relevance",
    "completeness",
    "clarity",
    "conciseness",
    "usefulness",
    "multi_turn_coherence",
]
_TEXT_FIELDS = ["main_strengths", "main_weaknesses", "final_comment"]

# ─── Prompt del juez ─────────────────────────────────────────────────────────
#
# Template unico: las instrucciones, la conversacion y el formato de salida
# van juntos en el mismo mensaje de usuario. No se usa system prompt separado.
# Placeholders: {turn_1}, {response_1}, {turn_2}, {response_2}

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

# ─── Reference-guided judging ─────────────────────────────────────────────────
#
# Categorias objetivas con respuestas de referencia: math, reasoning, coding, stem.
# Si references.yaml no existe o no tiene entrada para la pregunta: juez general.
# NUNCA se inventan referencias — solo se usan las del archivo.

REFERENCE_GUIDED_CATEGORIES = frozenset({"math", "reasoning", "coding", "stem"})
REFERENCES_PATH = ROOT / "data" / "mt_bench" / "subset" / "references.yaml"

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


def load_references(path: Path) -> Optional[dict]:
    """
    Load references.yaml. Returns None if the file does not exist.
    Keys are normalized to strings (question_id). Does not invent references.
    Expected YAML structure:
      <question_id>:
        turn_1: "reference answer for turn 1"  # optional
        turn_2: "reference answer for turn 2"  # optional
    """
    if not path.exists():
        return None
    try:
        import yaml  # PyYAML — already in requirements.txt
    except ImportError:
        print("  AVISO: PyYAML no instalado — references.yaml no sera cargado.")
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            print(f"  AVISO: {path.name} no es un mapping YAML valido — ignorado.")
            return None
        return {str(k): v for k, v in data.items()}
    except Exception as exc:
        print(f"  AVISO: no se pudo cargar {path.name}: {exc}")
        return None


def _build_reference_section(ref: dict) -> str:
    """Format the reference block to inject into JUDGE_TEMPLATE_REFERENCE."""
    lines = ["Reference answers (use ONLY to verify factual and technical correctness):"]
    r1 = (ref.get("turn_1") or ref.get("reference_1") or "").strip()
    r2 = (ref.get("turn_2") or ref.get("reference_2") or "").strip()
    if r1:
        lines.append(f"\nReference for turn 1:\n{r1}")
    if r2:
        lines.append(f"\nReference for turn 2:\n{r2}")
    return "\n".join(lines)


# ─── Llamada a la API ─────────────────────────────────────────────────────────

def _call_claude(
    user_content: str,
    api_key: str,
    model: str,
    dry_run: bool,
) -> tuple[Optional[dict], Optional[str]]:
    """
    Returns (parsed_json, error_message). Exactly one of them is None.
    Retries on network errors and rate limits with exponential backoff.
    """
    if dry_run:
        return {
            "score": 7,
            "correctness": 7,
            "instruction_following": 7,
            "relevance": 7,
            "completeness": 7,
            "clarity": 7,
            "conciseness": 7,
            "usefulness": 7,
            "multi_turn_coherence": 7,
            "main_strengths": "[DRY RUN] simulated evaluation — no API call made",
            "main_weaknesses": "None",
            "final_comment": "[DRY RUN]",
        }, None

    last_error: str = "unknown"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type"     : "application/json",
                    "x-api-key"        : api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model"     : model,
                    "max_tokens": MAX_TOKENS,
                    "messages"  : [{"role": "user", "content": user_content}],
                },
                timeout=API_TIMEOUT_S,
            )

            if resp.status_code == 429:
                wait = RETRY_WAIT_BASE_S * (2 ** (attempt - 1))
                last_error = "rate_limit_429"
                if attempt < MAX_RETRIES:
                    time.sleep(wait)
                    continue
                return None, last_error

            if resp.status_code != 200:
                last_error = f"http_{resp.status_code}: {resp.text[:200]}"
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_WAIT_BASE_S)
                    continue
                return None, last_error

            raw_text = resp.json()["content"][0]["text"].strip()
            parsed, parse_err = _parse_json_response(raw_text)
            if parsed is not None:
                return parsed, None
            # JSON parse failure is not retryable
            return None, f"json_parse: {parse_err}"

        except requests.exceptions.Timeout:
            last_error = f"timeout_{API_TIMEOUT_S}s"
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT_BASE_S * attempt)
                continue
            return None, last_error

        except requests.exceptions.RequestException as exc:
            last_error = f"request: {exc}"
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT_BASE_S)
                continue
            return None, last_error

    return None, last_error


def _parse_json_response(text: str) -> tuple[Optional[dict], Optional[str]]:
    """
    Parse and validate Claude's JSON response.
    Strips optional markdown code fences (```json ... ```).
    """
    cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    data: Optional[dict] = None
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: extract first {...} block
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError as exc:
                return None, str(exc)
        else:
            return None, "no JSON object found"

    missing = [f for f in _SCORE_FIELDS + _TEXT_FIELDS if f not in data]
    if missing:
        return None, f"missing fields: {missing}"

    for field in _SCORE_FIELDS:
        try:
            val = int(data[field])
            if not (1 <= val <= 10):
                return None, f"{field}={val} out of [1,10]"
            data[field] = val
        except (TypeError, ValueError):
            return None, f"{field} not int: {data[field]!r}"

    for field in _TEXT_FIELDS:
        data[field] = str(data.get(field, "")).strip()

    return data, None


# ─── Construccion del prompt ──────────────────────────────────────────────────

def _build_user_content(
    row: dict,
    ref: Optional[dict] = None,
) -> Optional[str]:
    """
    Returns the formatted judge prompt, or None if any text field is missing.
    If ref is provided (from references.yaml), uses JUDGE_TEMPLATE_REFERENCE.
    """
    t1q = str(row.get("turn_1_text")     or "").strip()
    r1  = str(row.get("response_1_text") or "").strip()
    t2q = str(row.get("turn_2_text")     or "").strip()
    r2  = str(row.get("response_2_text") or "").strip()

    if not (t1q and r1 and t2q and r2):
        return None

    if ref:
        return JUDGE_TEMPLATE_REFERENCE.format(
            reference_section=_build_reference_section(ref),
            turn_1=t1q,
            response_1=r1,
            turn_2=t2q,
            response_2=r2,
        )

    return JUDGE_TEMPLATE.format(
        turn_1=t1q,
        response_1=r1,
        turn_2=t2q,
        response_2=r2,
    )


# ─── Writer (append-mode, header solo al crear) ───────────────────────────────

class JudgeWriter:
    def __init__(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path   = output_dir / OUTPUT_CSV
        self._jsonl_path = output_dir / OUTPUT_JSONL

        csv_new = not self._csv_path.exists()

        self._csv_fh   = open(self._csv_path,   "a", newline="", encoding="utf-8")
        self._jsonl_fh = open(self._jsonl_path, "a",             encoding="utf-8")

        self._csv_writer = csv.DictWriter(
            self._csv_fh,
            fieldnames=JUDGE_COLS,
            extrasaction="ignore",
        )
        if csv_new:
            self._csv_writer.writeheader()

    def write(self, row: dict) -> None:
        self._csv_writer.writerow(row)
        self._csv_fh.flush()
        self._jsonl_fh.write(
            json.dumps({k: row.get(k) for k in JUDGE_COLS}, ensure_ascii=False) + "\n"
        )
        self._jsonl_fh.flush()

    def close(self) -> None:
        self._csv_fh.close()
        self._jsonl_fh.close()

    @property
    def csv_path(self) -> Path:
        return self._csv_path

    @property
    def jsonl_path(self) -> Path:
        return self._jsonl_path


# ─── Resume: IDs ya evaluados ─────────────────────────────────────────────────

def _load_completed_ids(judge_csv: Path) -> set[str]:
    """
    Returns conversation_ids already in judge_results.csv with
    judge_status in {success, skipped}. Errors are NOT excluded so
    they can be retried on the next run.
    """
    if not judge_csv.exists():
        return set()
    completed: set[str] = set()
    try:
        with open(judge_csv, "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                status = row.get("judge_status", "")
                cid    = row.get("conversation_id", "")
                if cid and status in ("success", "skipped"):
                    completed.add(cid)
    except Exception as exc:
        print(f"  AVISO: no se pudo leer {judge_csv.name}: {exc}")
    return completed


# ─── Evaluabilidad ────────────────────────────────────────────────────────────

def _is_evaluable(row: dict) -> tuple[bool, str]:
    """Returns (evaluable, reason_if_not)."""
    status = str(row.get("status_conversation") or "").lower()
    if status != "success":
        return False, f"status_conversation={status!r}"

    t1s = str(row.get("turn_1_status") or "").lower()
    t2s = str(row.get("turn_2_status") or "").lower()
    if t1s != "success" or t2s != "success":
        return False, f"turn_statuses=({t1s!r},{t2s!r})"

    if _build_user_content(row) is None:
        return False, "missing_text_fields"

    return True, ""


# ─── Loop principal ───────────────────────────────────────────────────────────

def run(
    input_csv:   Path,
    output_dir:  Path,
    phase:       Optional[int],
    judge_model: str,
    max_rows:    Optional[int],
    dry_run:     bool,
    delay_s:     float,
) -> None:

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key and not dry_run:
        print("ERROR: ANTHROPIC_API_KEY no configurada en el entorno.")
        print("       export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    if not input_csv.exists():
        print(f"ERROR: {input_csv} no encontrado.")
        sys.exit(1)

    with open(input_csv, "r", encoding="utf-8", newline="") as fh:
        all_rows = list(csv.DictReader(fh))

    if phase is not None:
        all_rows = [r for r in all_rows if str(r.get("phase", "")).strip() == str(phase)]

    # Load references.yaml — only used for math/reasoning/coding/stem categories.
    # If the file does not exist, all evaluations use the general judge (reference_guided=False).
    references = load_references(REFERENCES_PATH)

    judge_csv = output_dir / OUTPUT_CSV
    completed = _load_completed_ids(judge_csv)
    pending   = [r for r in all_rows if r.get("conversation_id") not in completed]

    if max_rows is not None:
        pending = pending[:max_rows]

    W = 64
    print(f"\n{'=' * W}")
    print(f"  GREEN-IA — Evaluacion LLM-as-a-Judge")
    print(f"{'=' * W}")
    print(f"  Input           : {_rel(input_csv)}")
    print(f"  Output          : {_rel(output_dir)}/")
    print(f"  Juez            : {judge_model}")
    print(f"  Score label     : {JUDGE_SCORE_LABEL}")
    print(f"  Fase filtro     : {phase if phase is not None else 'todas'}")
    print(f"  Total en CSV    : {len(all_rows)}")
    print(f"  Ya evaluados    : {len(completed)}")
    print(f"  A evaluar ahora : {len(pending)}")
    if dry_run:
        print(f"  MODO DRY-RUN    : sin llamadas a la API")
    print(f"{'─' * W}")

    if not pending:
        print("  Nada nuevo que evaluar. Saliendo.\n")
        return

    writer    = JudgeWriter(output_dir)
    n_success = n_error = n_skipped = 0

    for idx, row in enumerate(pending):
        conv_id = row.get("conversation_id", f"row_{idx}")
        label   = (
            f"{row.get('model_name','?')} {row.get('quantization','?')} "
            f"{row.get('hardware_profile','?')}/{row.get('execution_device','?')} "
            f"{row.get('original_category','?')} "
            f"q{row.get('official_question_id','?')} "
            f"rep{row.get('repetition','?')}"
        )
        print(f"  [{idx+1:04d}/{len(pending)}] {label[:52]:<52}", end=" ", flush=True)

        ts_now = datetime.now(timezone.utc).isoformat()

        # Reference lookup — only for objective categories, never invented
        category = str(row.get("original_category") or "").lower()
        ref: Optional[dict] = None
        if references and category in REFERENCE_GUIDED_CATEGORIES:
            qid = str(row.get("official_question_id") or "")
            ref = references.get(qid)

        out: dict = {k: row.get(k) for k in JUDGE_COLS}
        out["judge_model"]       = judge_model
        out["judge_timestamp"]   = ts_now
        out["judge_prompt_mode"] = JUDGE_PROMPT_MODE
        out["reference_guided"]  = ref is not None

        evaluable, reason = _is_evaluable(row)
        if not evaluable:
            for f in _SCORE_FIELDS + _TEXT_FIELDS:
                out[f] = None
            out["judge_status"]        = "skipped"
            out["judge_error_message"] = reason
            writer.write(out)
            n_skipped += 1
            print(f"SKIPPED ({reason})")
            continue

        user_content = _build_user_content(row, ref=ref)
        if user_content is None:
            for f in _SCORE_FIELDS + _TEXT_FIELDS:
                out[f] = None
            out["judge_status"]        = "skipped"
            out["judge_error_message"] = "missing_text_fields"
            writer.write(out)
            n_skipped += 1
            print("SKIPPED (missing_text_fields)")
            continue

        parsed, error_msg = _call_claude(
            user_content=user_content,
            api_key=api_key,
            model=judge_model,
            dry_run=dry_run,
        )

        if parsed is not None:
            for f in _SCORE_FIELDS + _TEXT_FIELDS:
                out[f] = parsed[f]
            out["judge_status"]        = "success"
            out["judge_error_message"] = None
            writer.write(out)
            n_success += 1
            print(f"score={parsed['score']}/10")
        else:
            for f in _SCORE_FIELDS + _TEXT_FIELDS:
                out[f] = None
            out["judge_status"]        = "error"
            out["judge_error_message"] = error_msg
            writer.write(out)
            n_error += 1
            print(f"ERROR ({error_msg})")

        if not dry_run:
            time.sleep(delay_s)

    writer.close()

    print(f"\n{'─' * W}")
    print(f"  Exitosos  : {n_success}")
    print(f"  Errores   : {n_error}  (se pueden reintentar con otra ejecucion)")
    print(f"  Saltados  : {n_skipped}")
    print(f"\n  Resultados guardados en:")
    print(f"    {_rel(writer.csv_path)}")
    print(f"    {_rel(writer.jsonl_path)}")

    if n_success > 0:
        _print_score_summary(writer.csv_path)

    print(f"{'=' * W}\n")


# ─── Resumen de puntajes ──────────────────────────────────────────────────────

def _print_score_summary(judge_csv: Path) -> None:
    rows: list[dict] = []
    try:
        with open(judge_csv, "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("judge_status") == "success" and row.get("score"):
                    rows.append(row)
    except Exception:
        return

    if not rows:
        return

    groups: dict[tuple, list[float]] = {}
    for r in rows:
        key = (
            r.get("model_name", "?"),
            r.get("quantization", "?"),
            r.get("hardware_profile", "?"),
            r.get("execution_device", "?"),
        )
        try:
            groups.setdefault(key, []).append(float(r["score"]))
        except (ValueError, TypeError):
            pass

    print(f"\n  {JUDGE_SCORE_LABEL}")
    print(f"  {'─' * 60}")
    for (mn, q, hw, dev), scores in sorted(groups.items()):
        n    = len(scores)
        mean = sum(scores) / n
        std  = (sum((s - mean) ** 2 for s in scores) / n) ** 0.5
        print(f"  {mn} {q} {hw}/{dev:<6}  n={n:4d}  score={mean:.2f}±{std:.2f}")


# ─── Entrypoint ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"GREEN-IA: Evaluacion LLM-as-a-Judge — {JUDGE_SCORE_LABEL}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="CSV de conversaciones (conversation_results.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directorio de salida para judge_results.{csv,jsonl}",
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=None,
        help="Filtrar por fase (1 o 2); sin este argumento procesa todas",
    )
    parser.add_argument(
        "--model",
        default=JUDGE_MODEL,
        help="Modelo Claude a usar como juez",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        metavar="N",
        help="Limitar a N conversaciones (para pruebas)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DELAY_S,
        metavar="SECONDS",
        help="Pausa en segundos entre llamadas a la API",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Sin llamadas a la API — puntajes simulados (7/10)",
    )

    args = parser.parse_args()

    try:
        run(
            input_csv   = args.input,
            output_dir  = args.output_dir,
            phase       = args.phase,
            judge_model = args.model,
            max_rows    = args.max_rows,
            dry_run     = args.dry_run,
            delay_s     = args.delay,
        )
    except KeyboardInterrupt:
        print("\n\n  Interrumpido por el usuario.")
        sys.exit(0)
    except Exception as exc:
        import traceback
        print(f"\n  ERROR inesperado: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

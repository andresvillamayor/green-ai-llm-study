"""
scripts/juez_calidad.py
Proyecto GREEN-IA — Universidad Comunero, Paraguay

Panel de tres jueces LLM (Anthropic, OpenAI, Google) que comparan por pares
las respuestas Q4 vs Q8 de cada modelo, con el protocolo pareado de MT-Bench.

Fuentes:
  - Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot
    Arena. NeurIPS 2023. arXiv:2306.05685. Sec. 3.4: se evalua cada par en los
    dos ordenes (A/B y B/A) y, si los veredictos no coinciden, se declara
    empate (criterio conservador ante el sesgo de posicion).
  - Codigo FastChat, llm_judge/common.py: NEED_REF_CATS = math, reasoning,
    coding (esas categorias usan el prompt con referencia "pair-math-v1"; las
    otras cinco usan "pair-v2"). Los prompts se leen tal cual de
    data/mt_bench/official/judge_prompts.jsonl.
  - Verga, P., et al. (2024). Replacing Judges with Juries: Evaluating LLM
    Generations with a Panel of Diverse Models. arXiv:2404.18796 (panel de
    jueces de distintas familias; veredicto por mayoria).

Entrada:
  results/comparacion_temperaturas/{modelo}_{esquema}/{modelo}_{q4|q8}/{categoria}/resultados_completos.csv
  (filas es_warmup == "False" y repeticion == "1", semilla 42; columna
  respuesta_texto), data/mt_bench/official/question.jsonl (solo turns[0]),
  judge_prompts.jsonl y reference_answer_gpt-4.jsonl.

Salida (results/judge/{esquema}/):
  juicios.csv               una fila por llamada (se escribe en modo append)
  juicios_consolidados.csv  una fila por par, columnas por juez y veredicto
                            del panel (se regenera al terminar)

Veredicto consolidado por juez: si los dos ordenes coinciden, ese; si no,
"empate". Veredicto del panel: mayoria (>= 2 de 3 jueces); sin mayoria,
"empate". Si algun juez no tiene filas, el panel queda vacio (incompleto).
Se usa la ULTIMA marca [[A]]/[[B]]/[[C]] del texto del juez.

Limitaciones:
  - Las referencias de math, reasoning y coding fueron generadas por GPT-4
    (reference_answer_gpt-4.jsonl), no verificadas por humanos.
  - Solo se juzga el turno 1 de cada pregunta de MT-Bench.
  - Algunos modelos pueden ignorar temperature=0 (p. ej. la documentacion de
    Gemini 4 / 3.6+ dice que ignora temperature); el juicio podria no ser
    determinista.
  - Los tres jueces son de distinto nivel y familia; llama-2-7b-chat y
    qwen2.5-7b no son directamente comparables entre si.
  - Los costos son estimaciones solo si se cargan los precios (ver abajo).

Seguridad: nunca se lee ni imprime ninguna clave (solo os.environ.get para
comprobar que exista) y nunca se imprime el mensaje de una excepcion.
No correr con sudo.
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ─── Jueces (nombres y parametros vigentes al 2026-11-07) ────────────────────
JUECES = ["anthropic", "openai", "google"]

MODELO_ANTHROPIC = "claude-opus-4-5"      # 2026-11-07
# Fijada porque la version 2026-10-21 de la API elimina temperature; verificado
# a mano el 2026-11-07 con anthropic 2.6.0 y claude-opus-4-5.
ANTHROPIC_VERSION_API = "2023-06-01"
MODELO_OPENAI = "gpt-6.1-terra"           # 2026-11-07
ESFUERZO_OPENAI = "none"                  # 2026-11-07
MODELO_GOOGLE = "gemini-3.6-flash"        # 2026-11-09: se descarta gemini-3.8-flash porque una llamada minima tardo 7 min 39 s; gemini-3.6-flash respondio en 4 s
TEMPERATURA = 0
MAX_TOKENS_JUEZ = 2048
# Google: max_output_tokens incluye los tokens de razonamiento (docs /docs/thinking),
# y gemini-3.6-flash admite hasta 65.536 de salida (docs /docs/models/gemini-3.6-flash).
# Con 2048 el razonamiento se comia el limite y la respuesta quedaba sin veredicto.
MAX_TOKENS_GOOGLE = 2048
# 2026-11-09: sin razonamiento oculto, en igualdad de condiciones con los otros jueces (Anthropic sin pensamiento extendido, OpenAI esfuerzo none); el razonamiento queda en la explicación visible que pide el prompt oficial de MT-Bench
THINKING_LEVEL_GOOGLE = "minimal"

MODELO_DE = {"anthropic": MODELO_ANTHROPIC, "openai": MODELO_OPENAI,
             "google": MODELO_GOOGLE}
VARIABLE_DE = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
               "google": "GEMINI_API_KEY"}

# ─── Precios por millon de tokens (USD) — NO inventar ───────────────────────
# Precios de lista consultados el 2026-11-09 en las paginas oficiales de cada proveedor (precio de entrada normal, sin cache). Google: nivel de pago vigente hasta 2026-12-31; desde 2027-01-01 sube a 1.50 de entrada y 7.50 de salida. En el nivel gratuito el gasto real es 0.
PRECIO_ENTRADA_POR_MILLON = {"anthropic": 5.0, "openai": 1.0, "google": 0.75}
PRECIO_SALIDA_POR_MILLON = {"anthropic": 25.0, "openai": 5.0, "google": 3.75}

# ─── Datos ───────────────────────────────────────────────────────────────────
MODELOS = ["llama-2-7b-chat", "qwen2.5-7b"]
ESQUEMAS = ["determinista", "fabricante"]
CATEGORIAS = ["writing", "roleplay", "reasoning", "math", "coding",
              "extraction", "stem", "humanities"]
CATEGORIAS_CON_REF = {"math", "reasoning", "coding"}  # NEED_REF_CATS (FastChat)
LIMITE_TRUNCADA = 1024  # max_tokens de config.yaml

DIR_OFICIAL = ROOT / "data" / "mt_bench" / "official"
RUTA_PREGUNTAS = DIR_OFICIAL / "question.jsonl"
RUTA_PROMPTS_JUEZ = DIR_OFICIAL / "judge_prompts.jsonl"
RUTA_REFERENCIAS = DIR_OFICIAL / "reference_answer_gpt-4.jsonl"
DIR_RESPUESTAS = ROOT / "results" / "comparacion_temperaturas"

COLUMNAS = ["esquema", "modelo", "categoria", "prompt_id", "juez", "modelo_juez",
            "orden", "temperatura_pedida", "esfuerzo_razonamiento", "prompt_usado",
            "veredicto_crudo", "veredicto_mapeado", "tokens_entrada",
            "tokens_salida", "costo_usd", "es_truncada_q4", "es_truncada_q8",
            "texto_juicio", "marca_de_tiempo", "tokens_razonamiento"]

REINTENTOS_SIN_VEREDICTO = 2
REINTENTOS_429 = 5
ERRORES_SEGUIDOS_MAX = 3
PATRON_VEREDICTO = re.compile(r"\[\[([ABC])\]\]")
MARCADORES = re.compile(r"\{(question|answer_a|answer_b|ref_answer_1)\}")


def _datos_error(e: Exception) -> str:
    """Solo cuatro datos del error; nunca str(e) ni el mensaje."""
    return (f"tipo={type(e).__name__}, status_code={getattr(e, 'status_code', None)}, "
            f"code={getattr(e, 'code', None)}, param={getattr(e, 'param', None)}")


# ─── Carga de datos ──────────────────────────────────────────────────────────
def leer_jsonl(ruta: Path):
    with open(ruta, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def cargar_oficiales():
    for ruta in (RUTA_PREGUNTAS, RUTA_PROMPTS_JUEZ, RUTA_REFERENCIAS):
        if not ruta.exists():
            sys.exit(f"ERROR: falta {ruta.relative_to(ROOT)}")
    preguntas = {q["question_id"]: q["turns"][0] for q in leer_jsonl(RUTA_PREGUNTAS)}
    prompts = {p["name"]: p for p in leer_jsonl(RUTA_PROMPTS_JUEZ)}
    referencias = {r["question_id"]: r["choices"][0]["turns"][0]
                   for r in leer_jsonl(RUTA_REFERENCIAS)}
    for nombre in ("pair-v2", "pair-math-v1"):
        if nombre not in prompts:
            sys.exit(f"ERROR: judge_prompts.jsonl no contiene '{nombre}'")
    return preguntas, prompts, referencias


def filas_a_juzgar(esquema: str, modelo: str, nivel: str, categoria: str):
    """{prompt_id: fila} con es_warmup == 'False' y repeticion == '1'."""
    ruta = (DIR_RESPUESTAS / f"{modelo}_{esquema}" / f"{modelo}_{nivel}"
            / categoria / "resultados_completos.csv")
    if not ruta.exists():
        return None
    out = {}
    with open(ruta, newline="", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            if fila["es_warmup"].strip() == "False" and fila["repeticion"].strip() == "1":
                out.setdefault(fila["prompt_id"].strip(), fila)
    return out


def construir_pares(esquema, modelos, categorias, preguntas, referencias):
    pares = []
    for modelo in modelos:
        for cat in categorias:
            q4 = filas_a_juzgar(esquema, modelo, "q4", cat)
            q8 = filas_a_juzgar(esquema, modelo, "q8", cat)
            if q4 is None or q8 is None:
                print(f"AVISO: faltan CSV de {modelo}/{cat} ({esquema}); se omite.")
                continue
            ids = sorted(set(q4) & set(q8), key=int)
            if len(ids) != 5 or len(q4) != 5 or len(q8) != 5:
                print(f"AVISO: {modelo}/{cat}: q4={len(q4)}, q8={len(q8)}, "
                      f"en comun={len(ids)} (se esperaban 5).")
            for pid in ids:
                qid = int(pid)
                if qid not in preguntas:
                    sys.exit(f"ERROR: question_id {qid} no esta en question.jsonl")
                if cat in CATEGORIAS_CON_REF and qid not in referencias:
                    sys.exit(f"ERROR: falta la referencia de question_id {qid} "
                             f"({cat}) en reference_answer_gpt-4.jsonl")
                pares.append({
                    "esquema": esquema, "modelo": modelo, "categoria": cat,
                    "prompt_id": pid, "pregunta": preguntas[qid],
                    "ref": referencias.get(qid) if cat in CATEGORIAS_CON_REF else None,
                    "resp_q4": q4[pid]["respuesta_texto"].strip(),
                    "resp_q8": q8[pid]["respuesta_texto"].strip(),
                    "trunc_q4": int(q4[pid]["completion_tokens"]) >= LIMITE_TRUNCADA,
                    "trunc_q8": int(q8[pid]["completion_tokens"]) >= LIMITE_TRUNCADA,
                    "prompt_usado": "pair-math-v1" if cat in CATEGORIAS_CON_REF else "pair-v2",
                })
    return pares


def armar_prompt(par, orden: int, prompts):
    """(system, user) con los textos del archivo oficial, sin tocar su redaccion."""
    p = prompts[par["prompt_usado"]]
    a, b = (par["resp_q4"], par["resp_q8"]) if orden == 1 else (par["resp_q8"], par["resp_q4"])
    valores = {"question": par["pregunta"], "answer_a": a, "answer_b": b,
               "ref_answer_1": par["ref"] or ""}
    # Una sola pasada: los marcadores dentro de las respuestas no se re-reemplazan.
    user = MARCADORES.sub(lambda m: valores[m.group(1)], p["prompt_template"])
    return p["system_prompt"], user


def mapear(crudo: str, orden: int) -> str:
    if crudo == "C":
        return "empate"
    if crudo not in ("A", "B"):
        return "error"
    if orden == 1:
        return "q4" if crudo == "A" else "q8"
    return "q8" if crudo == "A" else "q4"


# ─── Llamadas a las APIs (sintaxis copiada de scripts/verificar_claves.py) ───
def crear_cliente(juez: str):
    """Devuelve una funcion llamar(system, user) -> (texto, tok_in, tok_out)."""
    if juez == "anthropic":
        import anthropic
        cliente = anthropic.Anthropic()  # lee ANTHROPIC_API_KEY del entorno

        def llamar(system, user):
            r = cliente.messages.create(
                model=MODELO_ANTHROPIC, max_tokens=MAX_TOKENS_JUEZ,
                system=system, messages=[{"role": "user", "content": user}],
                extra_body={"temperature": TEMPERATURA},
                extra_headers={"anthropic-version": ANTHROPIC_VERSION_API})
            texto = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            return texto, r.usage.input_tokens, r.usage.output_tokens
        return llamar
    if juez == "openai":
        from openai import OpenAI
        cliente = OpenAI()  # lee OPENAI_API_KEY del entorno

        def llamar(system, user):
            r = cliente.chat.completions.create(
                model=MODELO_OPENAI,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                reasoning_effort=ESFUERZO_OPENAI, temperature=TEMPERATURA,
                max_completion_tokens=MAX_TOKENS_JUEZ)
            return (r.choices[0].message.content or ""), r.usage.prompt_tokens, r.usage.completion_tokens
        return llamar
    if juez == "google":
        from google import genai
        cliente = genai.Client()  # lee GEMINI_API_KEY del entorno

        def llamar(system, user):
            # La documentacion consultada no describe un parametro de system
            # instruction en la Interactions API: el system va al inicio del input.
            r = cliente.interactions.create(
                model=MODELO_GOOGLE, input=f"{system}\n\n{user}",
                generation_config={"thinking_level": THINKING_LEVEL_GOOGLE,
                                   "max_output_tokens": MAX_TOKENS_GOOGLE})
            uso = getattr(r, "usage", None)
            # Datos extra (razonamiento y estado) para juzgar_una, sin cambiar la
            # tupla de retorno que usan los otros jueces.
            llamar.extra = {"razonamiento": getattr(uso, "total_thought_tokens", None),
                            "estado": getattr(r, "status", None)}
            return (r.output_text or ""), getattr(uso, "total_input_tokens", None), \
                getattr(uso, "total_output_tokens", None)
        return llamar
    raise ValueError(juez)


def llamar_con_backoff(llamar, system, user):
    """Backoff exponencial ante 429. Relanza cualquier otro error."""
    espera = 2
    for intento in range(REINTENTOS_429 + 1):
        try:
            return llamar(system, user)
        except Exception as e:
            if getattr(e, "status_code", None) == 429 and intento < REINTENTOS_429:
                print(f"    limite de uso ({_datos_error(e)}); espero {espera}s")
                time.sleep(espera)
                espera *= 2
                continue
            raise


def costo(juez, tok_in, tok_out):
    pe, ps = PRECIO_ENTRADA_POR_MILLON[juez], PRECIO_SALIDA_POR_MILLON[juez]
    if pe is None or ps is None or tok_in is None or tok_out is None:
        return ""
    return round(tok_in / 1e6 * pe + tok_out / 1e6 * ps, 6)


# ─── CSV: append y reanudacion ───────────────────────────────────────────────
def leer_filas(ruta: Path):
    if not ruta.exists():
        return []
    with open(ruta, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clave(fila):
    return (fila["juez"], fila["modelo_juez"], fila["modelo"], fila["categoria"],
            fila["prompt_id"], str(fila["orden"]))


def agregar_fila(ruta: Path, fila: dict):
    nuevo = not ruta.exists()
    with open(ruta, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        if nuevo:
            w.writeheader()
        w.writerow(fila)
        f.flush()


def juzgar_una(juez, llamar, par, orden, prompts):
    """Una llamada (con reintentos si no hay veredicto). Devuelve la fila."""
    system, user = armar_prompt(par, orden, prompts)
    tok_in = tok_out = tok_razon = 0
    texto, crudo, error = "", "error", False
    for _ in range(1 + REINTENTOS_SIN_VEREDICTO):
        try:
            texto, ti, to = llamar_con_backoff(llamar, system, user)
        except Exception as e:
            texto, crudo, error = _datos_error(e), "error", True
            print(f"    fallo la llamada: {texto}")
            break
        tok_in += ti or 0
        tok_out += to or 0
        extra = getattr(llamar, "extra", None) if juez == "google" else None
        if extra:
            tok_razon += extra["razonamiento"] or 0
        marcas = PATRON_VEREDICTO.findall(texto)
        if marcas:
            crudo = marcas[-1]
            break
        if extra and extra["estado"] == "incomplete":
            # Sin veredicto y cortada (la doc da status=incomplete, p. ej. por max_tokens;
            # no hay un campo aparte con el motivo): queda como error con el estado.
            texto = f"[status=incomplete: respuesta cortada, p. ej. por limite de tokens] {texto}"
    es_google = juez == "google"
    return {
        "esquema": par["esquema"], "modelo": par["modelo"], "categoria": par["categoria"],
        "prompt_id": par["prompt_id"], "juez": juez, "modelo_juez": MODELO_DE[juez],
        "orden": orden,
        "temperatura_pedida": "no aplica" if es_google else TEMPERATURA,
        "esfuerzo_razonamiento": (ESFUERZO_OPENAI if juez == "openai"
                                  else THINKING_LEVEL_GOOGLE if es_google else ""),
        "prompt_usado": par["prompt_usado"], "veredicto_crudo": crudo,
        "veredicto_mapeado": mapear(crudo, orden), "tokens_entrada": tok_in,
        "tokens_salida": tok_out,
        # Google: la tabla de precios dice que la salida incluye el razonamiento.
        "costo_usd": costo(juez, tok_in, tok_out + tok_razon if es_google else tok_out),
        "es_truncada_q4": par["trunc_q4"], "es_truncada_q8": par["trunc_q8"],
        "texto_juicio": texto, "marca_de_tiempo": datetime.now().isoformat(timespec="seconds"),
        "tokens_razonamiento": tok_razon if es_google else "",
    }, error


# ─── Consolidacion ───────────────────────────────────────────────────────────
def consolidar_juez(v1, v2):
    """(veredicto, consistente) a partir de los veredictos mapeados de los 2 ordenes."""
    if v1 is None or v2 is None or "error" in (v1, v2):
        return "error", False
    return (v1, True) if v1 == v2 else ("empate", False)


def veredicto_panel(por_juez):
    if any(v is None for v in por_juez.values()):
        return ""  # panel incompleto: algun juez sin filas
    validos = [v for v in por_juez.values() if v != "error"]
    if validos:
        valor, n = Counter(validos).most_common(1)[0]
        if n >= 2:
            return valor
    return "empate"


def escribir_consolidado(esquema: str, dir_salida: Path):
    filas = leer_filas(dir_salida / "juicios.csv")
    ultima = {}
    for f in filas:  # la ultima fila de cada clave gana (reintentos posteriores)
        if f["modelo_juez"] != MODELO_DE.get(f["juez"]):
            continue  # solo filas del modelo configurado actualmente para ese juez
        ultima[clave(f)] = f
    pares = {}
    for f in ultima.values():
        pares.setdefault((f["modelo"], f["categoria"], f["prompt_id"]), []).append(f)
    cols = ["esquema", "modelo", "categoria", "prompt_id", "prompt_usado",
            "es_truncada_q4", "es_truncada_q8"]
    for j in JUECES:
        cols += [f"{j}_orden1", f"{j}_orden2", f"{j}_consolidado",
                 f"consistente_en_ambos_ordenes_{j}"]
    cols.append("veredicto_panel")
    with open(dir_salida / "juicios_consolidados.csv", "w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=cols)
        w.writeheader()
        for (modelo, cat, pid), fs in sorted(pares.items(), key=lambda x: (x[0][0], CATEGORIAS.index(x[0][1]), int(x[0][2]))):
            fila = {"esquema": esquema, "modelo": modelo, "categoria": cat, "prompt_id": pid,
                    "prompt_usado": fs[0]["prompt_usado"],
                    "es_truncada_q4": fs[0]["es_truncada_q4"],
                    "es_truncada_q8": fs[0]["es_truncada_q8"]}
            por_juez = {}
            for j in JUECES:
                o = {int(f["orden"]): f["veredicto_mapeado"] for f in fs if f["juez"] == j}
                if not o:
                    por_juez[j] = None
                    fila.update({f"{j}_orden1": "", f"{j}_orden2": "",
                                 f"{j}_consolidado": "", f"consistente_en_ambos_ordenes_{j}": ""})
                    continue
                cons, ok = consolidar_juez(o.get(1), o.get(2))
                por_juez[j] = cons
                fila.update({f"{j}_orden1": o.get(1, ""), f"{j}_orden2": o.get(2, ""),
                             f"{j}_consolidado": cons, f"consistente_en_ambos_ordenes_{j}": ok})
            fila["veredicto_panel"] = veredicto_panel(por_juez)
            w.writerow(fila)


# ─── Principal ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Panel de tres jueces LLM (Q4 vs Q8)")
    ap.add_argument("--esquema", required=True, choices=ESQUEMAS)
    ap.add_argument("--juez", default="todos", choices=JUECES + ["todos"])
    ap.add_argument("--modelo", choices=MODELOS, help="filtrar por modelo")
    ap.add_argument("--categoria", choices=CATEGORIAS, help="filtrar por categoria")
    ap.add_argument("--limite", type=int, help="procesar solo N pares (piloto)")
    ap.add_argument("--dry-run", action="store_true",
                    help="arma los prompts y cuenta llamadas; no llama a ninguna API")
    args = ap.parse_args()

    preguntas, prompts, referencias = cargar_oficiales()
    pares = construir_pares(args.esquema, [args.modelo] if args.modelo else MODELOS,
                            [args.categoria] if args.categoria else CATEGORIAS,
                            preguntas, referencias)
    if args.limite is not None:
        pares = pares[:args.limite]
    jueces = JUECES if args.juez == "todos" else [args.juez]

    if args.dry_run:
        print(f"[dry-run] esquema={args.esquema}  pares={len(pares)}")
        for j in jueces:
            print(f"  juez {j} ({MODELO_DE[j]}): {len(pares)} pares, {2 * len(pares)} llamadas")
        for etiqueta, con_ref in (("CON referencia (pair-math-v1)", True),
                                  ("SIN referencia (pair-v2)", False)):
            par = next((p for p in pares if (p["ref"] is not None) == con_ref), None)
            print(f"\n{'=' * 70}\nPrimer prompt {etiqueta}")
            if par is None:
                print("  (no hay pares de este tipo en la seleccion)")
                continue
            system, user = armar_prompt(par, 1, prompts)
            print(f"par: {par['modelo']} / {par['categoria']} / prompt_id {par['prompt_id']} / orden 1")
            print(f"--- SYSTEM ---\n{system}\n--- USER ---\n{user}")
        return

    dir_salida = ROOT / "results" / "judge" / args.esquema
    dir_salida.mkdir(parents=True, exist_ok=True)
    ruta = dir_salida / "juicios.csv"
    for juez in jueces:
        if not os.environ.get(VARIABLE_DE[juez], "").strip():
            print(f"{VARIABLE_DE[juez]}: ausente — se salta el juez {juez}.")
            continue
        try:
            llamar = crear_cliente(juez)
        except ImportError:
            print(f"Falta el paquete del juez {juez}; se salta (no se instala nada).")
            continue
        hechas = {clave(f) for f in leer_filas(ruta) if f["veredicto_crudo"] != "error"}
        seguidos = 0
        print(f"\n=== Juez {juez} ({MODELO_DE[juez]}) ===")
        for i, par in enumerate(pares, 1):
            for orden in (1, 2):
                k = (juez, MODELO_DE[juez], par["modelo"], par["categoria"],
                     par["prompt_id"], str(orden))
                if k in hechas:
                    continue
                fila, fallo_api = juzgar_una(juez, llamar, par, orden, prompts)
                agregar_fila(ruta, fila)
                seguidos = seguidos + 1 if fallo_api else 0
                print(f"  [{i}/{len(pares)}] {par['modelo']}/{par['categoria']}/"
                      f"{par['prompt_id']} orden {orden}: {fila['veredicto_mapeado']}")
                if seguidos >= ERRORES_SEGUIDOS_MAX:
                    print(f"  {ERRORES_SEGUIDOS_MAX} fallos seguidos de API: se detiene el juez {juez}.")
                    break
            if seguidos >= ERRORES_SEGUIDOS_MAX:
                break
    escribir_consolidado(args.esquema, dir_salida)
    print(f"\nListo: {dir_salida.relative_to(ROOT)}/juicios.csv y juicios_consolidados.csv")


if __name__ == "__main__":
    main()

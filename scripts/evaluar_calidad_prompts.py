# evaluar_calidad_prompts.py
# Juez LLM para evaluacion de calidad de respuestas
# Proyecto GREEN-IA — Andres Villamayor
# Universidad Comunero — Paraguay
#
# Metodo: Single Answer Grading — LLM-as-a-Judge
# Referencia: Zheng et al. (2023). Judging LLM-as-a-Judge
#   with MT-Bench and Chatbot Arena. NeurIPS 2023.
#   arXiv:2306.05685v4. Figura 6 del paper.
#
# Juez: Claude Sonnet 4.6 via API Anthropic
# Equivalencia: Claude Sonnet 4.6 ~ GPT-4 en capacidad evaluativa
# Acuerdo juez LLM vs humanos: >80% (Zheng et al. 2023, Tabla 5)
#
# Sesgos documentados y mitigaciones (Seccion 3.3 del paper):
#
#   Sesgo de posicion (position bias):
#     No aplica en single answer grading — se evalua una
#     respuesta a la vez, sin comparacion de orden.
#
#   Sesgo de verbosidad (verbosity bias):
#     Mitigacion: instruccion explicita en rubrica
#     "Do not allow the length of the response to influence
#     your evaluation" — copiado de Figura 6 del paper.
#
#   Sesgo de auto-preferencia (self-enhancement bias):
#     No aplica — Claude evalua Llama y Qwen, no sus propias
#     respuestas. (Panickssery et al. 2024. arXiv:2404.13076)
#
# Prompt del juez: basado en Figura 6 de Zheng et al. 2023
# Escala: 1-10 con formato [[rating]] como en el paper
#
# Muestra: repeticion 1 de cada configuracion
#   40 prompts x 8 configs = 320 evaluaciones
#   Justificacion: representativa del comportamiento del modelo

import os
import json
import time
import random
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results" / "measurements"

# Prompt del juez — Figura 6, Zheng et al. NeurIPS 2023
# Traducido al contexto del experimento GREEN-IA
# Se mantiene la instruccion original del paper sobre verbosidad
SYSTEM_PROMPT = """Please act as an impartial judge and evaluate \
the quality of the response provided by an AI assistant to the \
user question displayed below. Your evaluation should consider \
factors such as the helpfulness, relevance, accuracy, depth, \
creativity, and level of detail of the response. Begin your \
evaluation by providing a short explanation. Be as objective \
as possible. Do not allow the length of the response to \
influence your evaluation. After providing your explanation, \
please rate the response on a scale of 1 to 10 by strictly \
following this format: "[[rating]]", for example: "Rating: [[5]]"."""

USER_TEMPLATE = """[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""


def cargar_csv():
    """Carga el CSV mas reciente del experimento v1."""
    archivos = sorted(RESULTS_DIR.glob("experimento_codecarbon_*.csv"))
    if not archivos:
        raise FileNotFoundError(
            "No hay CSV del experimento. "
            "Corre experimento_codecarbon_v1.py primero."
        )
    ruta = archivos[-1]
    print(f"  CSV: {ruta.name}")
    df = pd.read_csv(ruta)

    if "response_text" not in df.columns:
        raise ValueError(
            "El CSV no tiene columna response_text. "
            "Verifica que experimento_codecarbon_v1.py "
            "incluye ese campo."
        )

    vacias = df["response_text"].isna().sum()
    print(f"  Filas totales   : {len(df)}")
    print(f"  Con respuesta   : {len(df) - vacias}")
    print(f"  Sin respuesta   : {vacias}")
    return df


def llamar_juez(prompt, respuesta, intento=1):
    """
    Llama a Claude Sonnet 4.6 como juez.
    Implementa Single Answer Grading (Figura 6, Zheng et al. 2023).
    Retorna el puntaje 1-10 y la justificacion.
    """
    mensaje_usuario = USER_TEMPLATE.format(
        question=prompt,
        answer=respuesta
    )

    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type"      : "application/json",
                "x-api-key"         : os.environ.get("ANTHROPIC_API_KEY", ""),
                "anthropic-version" : "2023-06-01"
            },
            json={
                "model"      : "claude-sonnet-4-6",
                "max_tokens" : 500,
                "system"     : SYSTEM_PROMPT,
                "messages"   : [
                    {"role": "user", "content": mensaje_usuario}
                ]
            },
            timeout=30
        )

        if r.status_code != 200:
            print(f"    Error API {r.status_code}")
            return None

        texto = r.json()["content"][0]["text"].strip()

        # extraer puntaje del formato [[rating]] — Figura 6 del paper
        puntaje = None
        if "[[" in texto and "]]" in texto:
            try:
                inicio = texto.rfind("[[") + 2
                fin    = texto.rfind("]]")
                puntaje = int(texto[inicio:fin].strip())
            except ValueError:
                puntaje = None

        return {
            "puntaje"       : puntaje,
            "justificacion" : texto,
        }

    except Exception as e:
        if intento < 3:
            time.sleep(2)
            return llamar_juez(prompt, respuesta, intento + 1)
        print(f"    Error: {e}")
        return None


def evaluar():
    """
    Evalua la calidad de respuestas usando LLM-as-a-Judge.
    Usa repeticion 1 de cada configuracion para cada prompt.
    Total: 40 prompts x 8 configs = 320 evaluaciones.
    """
    df = cargar_csv()

    # usar solo repeticion 1
    df_rep1 = df[df["repetition"] == 1].copy()
    df_rep1 = df_rep1[
        df_rep1["response_text"].notna() &
        (df_rep1["response_text"] != "")
    ]

    print(f"\n  Pares a evaluar : {len(df_rep1)}")
    print(f"  Juez            : claude-sonnet-4-6")
    print(f"  Metodo          : Single Answer Grading")
    print(f"  Referencia      : Zheng et al. NeurIPS 2023 Fig.6")
    print(f"  {'─'*50}")

    resultados = []
    errores    = 0
    total      = len(df_rep1)

    for idx, (_, fila) in enumerate(df_rep1.iterrows()):
        contador = idx + 1
        config = (f"{fila['model']} {fila['quantization']} "
                  f"{fila['device']} prompt_{fila['prompt_id']} "
                  f"cat_{fila.get('categoria','?')}")

        print(f"  [{contador}/{total}] {config}...",
              end=" ", flush=True)

        evaluacion = llamar_juez(
            prompt   = fila["prompt_text"],
            respuesta= str(fila["response_text"])
        )

        if evaluacion and evaluacion["puntaje"] is not None:
            resultados.append({
                "timestamp_eval"   : datetime.now().isoformat(),
                "model"            : fila["model"],
                "quantization"     : fila["quantization"],
                "device"           : fila["device"],
                "categoria"        : fila.get("categoria", ""),
                "prompt_id"        : fila["prompt_id"],
                "prompt_text"      : fila["prompt_text"],
                "response_text"    : str(fila["response_text"])[:500],
                "tokens_generated" : fila["tokens_generated"],
                "total_energy_wh"  : fila["total_energy_wh"],
                "energy_per_token" : fila.get("energy_per_token", None),
                "tokens_per_second": fila["tokens_per_second"],
                "puntaje_calidad"  : evaluacion["puntaje"],
                "justificacion"    : evaluacion["justificacion"][:500],
                "juez_modelo"      : "claude-sonnet-4-6",
                "metodo_juez"      : "single_answer_grading",
                "referencia_juez"  : "Zheng et al. NeurIPS 2023 Fig.6",
                "params_experimento": "temp=0.7 top_p=0.9 max=256",
            })
            print(f"puntaje={evaluacion['puntaje']}/10")
        else:
            errores += 1
            print("ERROR")

        time.sleep(0.5)

    print(f"\n  {'─'*50}")
    print(f"  Evaluaciones OK : {len(resultados)}")
    print(f"  Errores         : {errores}")

    if not resultados:
        print("  Sin resultados.")
        return None

    df_eval = pd.DataFrame(resultados)

    # resumen por configuracion y categoria
    print(f"\n  RESUMEN POR MODELO Y CONFIGURACION")
    print(f"  {'─'*60}")
    resumen = (df_eval
        .groupby(["model", "quantization", "device"])
        .agg(
            n                 = ("puntaje_calidad", "count"),
            puntaje_medio     = ("puntaje_calidad", "mean"),
            puntaje_std       = ("puntaje_calidad", "std"),
            energia_media_mwh = ("total_energy_wh",
                                 lambda x: x.mean() * 1000),
            tps_medio         = ("tokens_per_second", "mean"),
        )
        .reset_index()
    )

    for _, r in resumen.iterrows():
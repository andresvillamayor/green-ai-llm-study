# evaluar_calidad_prompts.py
# Juez LLM: compara accuracy de respuestas Experimento 1 vs Experimento 2
# Proyecto GREEN-IA — Maestria Ciencia de Datos — Andres Villamayor
#
# Metodo: LLM-as-a-Judge
# Referencia principal:
#   Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench
#   and Chatbot Arena. NeurIPS 2023. arXiv:2306.05685
#   Resultado: jueces LLM alcanzan >80% de acuerdo con humanos
#
# Juez utilizado: Claude Sonnet 4.6 via API Anthropic
# Justificacion: Claude Sonnet 4.6 es equivalente a GPT-4 en capacidad
# evaluativa. El paper de Zheng et al. valida jueces de esta categoria.
# No hay sesgo de auto-preferencia: Claude evalua Llama y Qwen,
# modelos distintos al juez.
#
# Sesgos conocidos y mitigaciones aplicadas:
#
#   Sesgo de posicion (position bias):
#     Definicion: el juez tiende a preferir la primera respuesta presentada
#     Fuente: Zheng et al. 2023
#     Mitigacion: orden de presentacion randomizado con random.choice([True,False])
#     Implementacion: parametro invertir=True/False en llamar_juez()
#
#   Sesgo de verbosidad (verbosity bias):
#     Definicion: el juez tiende a preferir respuestas mas largas
#     Fuente: Zheng et al. 2023
#     Mitigacion: rubrica explicita con criterio de concision y
#     instruccion explicita "No favorecer la respuesta mas larga"
#
#   Sesgo de auto-preferencia (self-preference bias):
#     Definicion: un modelo prefiere sus propias respuestas
#     Fuente: Panickssery et al. 2024. arXiv:2404.13076
#     Mitigacion: NO APLICA — Claude evalua Llama y Qwen, no a si mismo
#
# Criterio de evaluacion: ACCURACY
#     Correctitud factual, precision logica, completitud, concision
#     Puntaje: 1-10 por respuesta + ganador + justificacion
#
# Comparacion: Experimento 1 vs Experimento 2
#     Exp1: temperature=0.7, top_p=0.9,  max_tokens=256
#     Exp2: temperature=0.1, top_p=0.95, max_tokens=512
#     Ref parametros: Caravaca et al. ACL 2025. arXiv:2602.05712
#
# Muestra evaluada: repeticion 1 de cada configuracion
#     Justificacion: representativa del comportamiento del modelo
#     y reduce costo de API manteniendo cobertura de los 15 prompts
#     en las 8 configuraciones (120 pares evaluados)

import pandas as pd
import numpy as np
import requests
import json
import random
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results" / "measurements"

# Rubrica del juez basada en MT-Bench (Zheng et al. NeurIPS 2023)
# Criterio principal: ACCURACY
# Se incluye instruccion explicita contra sesgo de verbosidad
# (Zheng et al. 2023: verbosity bias es el sesgo mas comun en LLM-as-a-Judge)
RUBRICA_JUEZ = """
Sos un evaluador experto de respuestas de modelos de lenguaje.
Tu tarea es evaluar la ACCURACY (precision y correctitud) de dos
respuestas generadas por el mismo modelo para el mismo prompt,
pero con distintos parametros de inferencia.

Prompt original:
{prompt}

Respuesta A (parametros: temperature=0.7, max_tokens=256):
{respuesta_a}

Respuesta B (parametros: temperature=0.1, max_tokens=512):
{respuesta_b}

Evalua la ACCURACY de cada respuesta segun estos 4 criterios:
1. Correctitud factual: la respuesta es correcta y sin errores
2. Precision logica: el razonamiento es valido y coherente
3. Completitud: responde completamente lo que se pregunto
4. Concision: sin informacion incorrecta o confusa

Reglas estrictas para evitar sesgos conocidos:
- NO favorecer la respuesta mas larga por ser mas larga
- NO favorecer la respuesta que aparece primero
- Evaluar SOLO la correctitud y precision del contenido
- Una respuesta corta pero correcta vale mas que una larga con errores

Puntaje 1-10 donde:
  1-3  = respuesta incorrecta o incoherente
  4-6  = respuesta parcialmente correcta
  7-8  = respuesta correcta y completa
  9-10 = respuesta excelente en todos los criterios

Responde SOLO con este JSON sin texto adicional:
{{
  "puntaje_a": <1-10>,
  "puntaje_b": <1-10>,
  "ganador": "A" o "B" o "empate",
  "justificacion_a": "<una oracion sobre la accuracy de A>",
  "justificacion_b": "<una oracion sobre la accuracy de B>",
  "razon_ganador": "<una oracion explicando por que gana>"
}}
"""


def cargar_csvs():
    """Carga el CSV del Exp1 y del Exp2."""
    todos = sorted(RESULTS_DIR.glob("experimento_codecarbon_*.csv"))
    v2s   = sorted(RESULTS_DIR.glob("experimento_params_v2_*.csv"))

    if not todos:
        raise FileNotFoundError("No hay CSV del Experimento 1")
    if not v2s:
        raise FileNotFoundError("No hay CSV del Experimento 2")

    csv_v1 = todos[-1]
    csv_v2 = v2s[-1]

    print(f"  Exp1: {csv_v1.name}")
    print(f"  Exp2: {csv_v2.name}")

    df1 = pd.read_csv(csv_v1)
    df2 = pd.read_csv(csv_v2)

    # verificar que tienen response_text
    if "response_text" not in df1.columns:
        raise ValueError("El CSV del Exp1 no tiene response_text. "
                         "Corre experimento_codecarbon_v1.py primero.")
    if "response_text" not in df2.columns:
        raise ValueError("El CSV del Exp2 no tiene response_text. "
                         "Corre experimento_codecarbon_v2.py primero.")

    return df1, df2


def llamar_juez(prompt, respuesta_a, respuesta_b, invertir=False, intento=1):
    """
    Llama a Claude Sonnet 4.6 como juez.
    invertir=True randomiza el orden para mitigar sesgo de posicion
    (Zheng et al. 2023: position bias es un sesgo conocido del metodo)
    """
    if invertir:
        texto_a = respuesta_b
        texto_b = respuesta_a
    else:
        texto_a = respuesta_a
        texto_b = respuesta_b

    mensaje = RUBRICA_JUEZ.format(
        prompt     = prompt,
        respuesta_a= texto_a,
        respuesta_b= texto_b,
    )

    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type" : "application/json",
                "x-api-key"    : __import__("os").environ.get("ANTHROPIC_API_KEY", ""),
                "anthropic-version": "2023-06-01"
            },
            json={
                "model"     : "claude-sonnet-4-6",
                "max_tokens": 400,
                "messages"  : [{"role": "user", "content": mensaje}]
            },
            timeout=30
        )

        if r.status_code != 200:
            print(f"    Error API {r.status_code}: {r.text[:100]}")
            return None

        texto = r.json()["content"][0]["text"].strip()
        texto = texto.replace("```json", "").replace("```", "").strip()
        resultado = json.loads(texto)

        # si se invirtio el orden, invertir los puntajes de vuelta
        if invertir:
            resultado["puntaje_a"], resultado["puntaje_b"] = \
                resultado["puntaje_b"], resultado["puntaje_a"]
            resultado["justificacion_a"], resultado["justificacion_b"] = \
                resultado["justificacion_b"], resultado["justificacion_a"]
            if resultado["ganador"] == "A":
                resultado["ganador"] = "B"
            elif resultado["ganador"] == "B":
                resultado["ganador"] = "A"

        return resultado

    except Exception as e:
        if intento < 3:
            time.sleep(2)
            return llamar_juez(prompt, respuesta_a, respuesta_b,
                               invertir, intento + 1)
        print(f"    Error juez: {e}")
        return None


def evaluar_pares():
    """
    Para cada combinacion (modelo, cuant, device, prompt_id),
    toma la primera repeticion de cada experimento y las compara.
    """
    df1, df2 = cargar_csvs()

    # usar solo repeticion 1 para la comparacion
    # (reducir costo de API y tiempo de evaluacion)
    df1_rep1 = df1[df1["repetition"] == 1].copy()
    df2_rep1 = df2[df2["repetition"] == 1].copy()

    # merge por modelo, cuantizacion, dispositivo y prompt_id
    merge_cols = ["model", "quantization", "device", "prompt_id"]
    df_merged = df1_rep1.merge(
        df2_rep1,
        on=merge_cols,
        suffixes=("_v1", "_v2")
    )

    print(f"\n  {len(df_merged)} pares para evaluar")
    print(f"  Juez: Claude Sonnet 4.6 (Zheng et al. NeurIPS 2023)")
    print(f"  Criterio: accuracy — correctitud y precision")
    print(f"  {'─'*50}")

    resultados = []
    errores    = 0
    total      = len(df_merged)

    for idx, fila in df_merged.iterrows():
        contador = idx + 1
        config   = f"{fila['model']} {fila['quantization']} {fila['device']}"
        print(f"  [{contador}/{total}] {config} prompt {fila['prompt_id']}...",
              end=" ", flush=True)

        # randomizar orden para mitigar sesgo de posicion
        invertir = random.choice([True, False])

        evaluacion = llamar_juez(
            prompt      = fila["prompt_text_v1"],
            respuesta_a = str(fila["response_text_v1"]),
            respuesta_b = str(fila["response_text_v2"]),
            invertir    = invertir
        )

        if evaluacion:
            resultados.append({
                "timestamp_eval"       : datetime.now().isoformat(),
                "model"                : fila["model"],
                "quantization"         : fila["quantization"],
                "device"               : fila["device"],
                "prompt_id"            : fila["prompt_id"],
                "prompt_text"          : fila["prompt_text_v1"],
                # respuestas
                "response_v1"          : str(fila["response_text_v1"])[:500],
                "response_v2"          : str(fila["response_text_v2"])[:500],
                # metricas energeticas Exp1
                "energia_v1_mwh"       : fila["total_energy_wh_v1"] * 1000,
                "tokens_gen_v1"        : fila["tokens_generated_v1"],
                "tps_v1"               : fila["tokens_per_second_v1"],
                # metricas energeticas Exp2
                "energia_v2_mwh"       : fila["total_energy_wh_v2"] * 1000,
                "tokens_gen_v2"        : fila["tokens_generated_v2"],
                "tps_v2"               : fila["tokens_per_second_v2"],
                # resultado del juez
                "puntaje_v1"           : evaluacion["puntaje_a"],
                "puntaje_v2"           : evaluacion["puntaje_b"],
                "ganador"              : evaluacion["ganador"],
                "justificacion_v1"     : evaluacion["justificacion_a"],
                "justificacion_v2"     : evaluacion["justificacion_b"],
                "razon_ganador"        : evaluacion["razon_ganador"],
                "orden_invertido"      : invertir,
                # parametros de cada experimento
                "params_v1"            : "temp=0.7 top_p=0.9 max=256",
                "params_v2"            : "temp=0.1 top_p=0.95 max=512",
                "juez_modelo"          : "claude-sonnet-4-6",
                "metodo"               : "LLM-as-a-Judge Zheng et al. NeurIPS 2023",
            })
            print(f"v1={evaluacion['puntaje_a']}/10  "
                  f"v2={evaluacion['puntaje_b']}/10  "
                  f"gana={evaluacion['ganador']}")
        else:
            errores += 1
            print("ERROR")

        time.sleep(0.5)

    print(f"\n  {'─'*50}")
    print(f"  Evaluaciones OK: {len(resultados)}  |  Errores: {errores}")

    if not resultados:
        return None

    df_eval = pd.DataFrame(resultados)

    # resumen por configuracion
    print(f"\n  RESUMEN POR CONFIGURACION")
    print(f"  {'─'*60}")
    resumen = (df_eval
        .groupby(["model", "quantization", "device"])
        .agg(
            n              = ("puntaje_v1", "count"),
            puntaje_v1_med = ("puntaje_v1", "mean"),
            puntaje_v2_med = ("puntaje_v2", "mean"),
            energia_v1_med = ("energia_v1_mwh", "mean"),
            energia_v2_med = ("energia_v2_mwh", "mean"),
            gana_v1        = ("ganador", lambda x: (x == "A").sum()),
            gana_v2        = ("ganador", lambda x: (x == "B").sum()),
            empates        = ("ganador", lambda x: (x == "empate").sum()),
        )
        .reset_index()
    )

    for _, r in resumen.iterrows():
        print(f"  {r['model']} {r['quantization']} {r['device']}")
        print(f"    accuracy  v1={r['puntaje_v1_med']:.1f}/10  "
              f"v2={r['puntaje_v2_med']:.1f}/10")
        print(f"    energia   v1={r['energia_v1_med']:.3f}mWh  "
              f"v2={r['energia_v2_med']:.3f}mWh")
        print(f"    victorias v1={r['gana_v1']}  "
              f"v2={r['gana_v2']}  empates={r['empates']}")
        print()

    # guardar resultados
    fecha    = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta     = RESULTS_DIR / f"evaluacion_calidad_prompts_{fecha}.csv"
    ruta_res = RESULTS_DIR / f"resumen_calidad_prompts_{fecha}.csv"

    df_eval.to_csv(ruta, index=False)
    resumen.to_csv(ruta_res, index=False)

    print(f"  Guardado: {ruta.name}")
    print(f"  Guardado: {ruta_res.name}")

    return ruta


if __name__ == "__main__":
    print("=" * 60)
    print("  GREEN-IA: Evaluacion de Calidad — LLM-as-a-Judge")
    print("  Juez: Claude Sonnet 4.6 via API Anthropic")
    print("  Metodo: Zheng et al. NeurIPS 2023 arXiv:2306.05685")
    print("  Criterio: accuracy — correctitud y precision")
    print("  Compara: Exp1 (temp=0.7) vs Exp2 (temp=0.1)")
    print("=" * 60)

    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n  ERROR: ANTHROPIC_API_KEY no configurada")
        print("  Correr: export ANTHROPIC_API_KEY='tu-key'")
        exit(1)

    try:
        archivo = evaluar_pares()
        if archivo:
            print(f"\n  Resultados en: {archivo}")

    except KeyboardInterrupt:
        print("\n  Cancelado.")

    except Exception as e:
        import traceback
        print(f"\n  Error: {e}")
        traceback.print_exc()

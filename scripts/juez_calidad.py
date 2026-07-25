"""
scripts/juez_calidad.py
Proyecto GREEN-IA — Universidad Comunero, Paraguay

Evalua la calidad de respuestas de Llama-2-7B (Q4 vs Q8)
usando el paradigma LLM-as-a-Judge con metodo de comparacion
pareada (pairwise comparison).

Respaldo cientifico:
Zheng, L., Chiang, W-L., Sheng, Y., et al. (2023).
Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.
NeurIPS 2023. arXiv:2306.05685

El paper demuestra que la comparacion pareada (mostrar dos
respuestas juntas al juez y pedir cual es mejor) produce
evaluaciones mas consistentes que puntuar cada respuesta
por separado sin punto de referencia. Este es el metodo
adoptado en este script.

Entrada:
  results/analisis_categoria_prueba/{categoria}_prompts_respuestas_Q4.csv
  results/analisis_categoria_prueba/{categoria}_prompts_respuestas_Q8.csv

Salida:
  results/analisis_categoria_prueba/{categoria}_evaluacion_calidad.csv

Juez: Claude via API Anthropic, temperature=0 (determinista,
para reproducibilidad del juicio — mismo criterio de
determinismo aplicado a la generacion de las respuestas
evaluadas).
"""

import os
import csv
import json
import time
from pathlib import Path
import anthropic

ROOT = Path(__file__).parent.parent
CATEGORIA = "math"

RUTA_Q4 = ROOT / "results" / "analisis_categoria_prueba" / f"{CATEGORIA}_prompts_respuestas_Q4.csv"
RUTA_Q8 = ROOT / "results" / "analisis_categoria_prueba" / f"{CATEGORIA}_prompts_respuestas_Q8.csv"
RUTA_SALIDA = ROOT / "results" / "analisis_categoria_prueba" / f"{CATEGORIA}_evaluacion_calidad.csv"

# Criterios de evaluacion por tipo de categoria.
# Aplicables a categorias con resultado verificable
# (math, coding, stem): se prioriza correctitud.
CRITERIOS_MATEMATICO = [
    "correctitud",           # el resultado numerico/logico es correcto
    "claridad_razonamiento", # el proceso para llegar al resultado es claro
    "completitud",           # responde completamente lo que pide el prompt
]

MODELO_JUEZ = "claude-opus-4-5"  # usar el modelo mas capaz disponible para el juicio

PLANTILLA_JUEZ = """Sos un evaluador experto e imparcial. Tu tarea es comparar dos respuestas (A y B) generadas por el mismo modelo de lenguaje en dos configuraciones distintas de cuantizacion, para el mismo prompt matematico.

No sabes cual respuesta corresponde a que configuracion. Evalua unicamente en base al contenido.

PROMPT ORIGINAL:
{prompt}

RESPUESTA A:
{respuesta_a}

RESPUESTA B:
{respuesta_b}

Evalua ambas respuestas en estos 3 criterios, cada uno de 0 a 100:
1. correctitud: el resultado matematico es correcto
2. claridad_razonamiento: el proceso para llegar al resultado es claro y esta explicado
3. completitud: responde completamente lo que pide el prompt, sin repeticiones innecesarias que resten claridad

Nota importante: si una respuesta repite el mismo contenido varias veces sin agregar informacion nueva, esto debe penalizar completitud y claridad_razonamiento, aunque el resultado numerico sea correcto.

Respondé en formato JSON exacto, sin texto adicional antes o despues:
{{
  "correctitud_a": <0-100>,
  "claridad_razonamiento_a": <0-100>,
  "completitud_a": <0-100>,
  "correctitud_b": <0-100>,
  "claridad_razonamiento_b": <0-100>,
  "completitud_b": <0-100>,
  "ganador": "A" o "B" o "empate",
  "justificacion": "<explicacion breve en 1-2 oraciones>"
}}"""


def cargar_csv(ruta):
    with open(ruta, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluar_par(cliente, prompt_texto, respuesta_q4, respuesta_q8):
    """Llama al juez para comparar una respuesta Q4 vs Q8 del mismo prompt."""
    mensaje = PLANTILLA_JUEZ.format(
        prompt=prompt_texto,
        respuesta_a=respuesta_q4,
        respuesta_b=respuesta_q8,
    )
    respuesta = cliente.messages.create(
        model=MODELO_JUEZ,
        max_tokens=500,
        temperature=0,  # determinista, para juicio reproducible
        messages=[{"role": "user", "content": mensaje}],
    )
    texto = respuesta.content[0].text.strip()
    # limpiar posibles marcadores de codigo markdown
    texto = texto.replace("```json", "").replace("```", "").strip()
    return json.loads(texto)


def main():
    print(f"\n{'='*60}")
    print(f"  GREEN-IA — Juez de calidad (LLM-as-a-Judge)")
    print(f"  Metodo: comparacion pareada (Zheng et al. NeurIPS 2023)")
    print(f"  Categoria: {CATEGORIA}")
    print(f"{'='*60}\n")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: variable ANTHROPIC_API_KEY no configurada.")
        return

    if not RUTA_Q4.exists() or not RUTA_Q8.exists():
        print(f"ERROR: faltan archivos de entrada.")
        print(f"  Q4: {RUTA_Q4} — existe: {RUTA_Q4.exists()}")
        print(f"  Q8: {RUTA_Q8} — existe: {RUTA_Q8.exists()}")
        return

    filas_q4 = cargar_csv(RUTA_Q4)
    filas_q8 = cargar_csv(RUTA_Q8)

    # emparejar por prompt_id
    mapa_q8 = {f["prompt_id"]: f for f in filas_q8}

    cliente = anthropic.Anthropic()  # usa ANTHROPIC_API_KEY del entorno
    resultados = []

    for fila_q4 in filas_q4:
        pid = fila_q4["prompt_id"]
        if pid not in mapa_q8:
            print(f"  AVISO: prompt_id {pid} no encontrado en Q8, se omite")
            continue
        fila_q8 = mapa_q8[pid]

        print(f"  Evaluando prompt_id {pid}...", end=" ", flush=True)
        try:
            evaluacion = evaluar_par(
                cliente,
                fila_q4["prompt_enviado"],
                fila_q4["respuesta_texto"],
                fila_q8["respuesta_texto"],
            )
            resultados.append({
                "categoria": CATEGORIA,
                "prompt_id": pid,
                "correctitud_q4": evaluacion["correctitud_a"],
                "claridad_razonamiento_q4": evaluacion["claridad_razonamiento_a"],
                "completitud_q4": evaluacion["completitud_a"],
                "correctitud_q8": evaluacion["correctitud_b"],
                "claridad_razonamiento_q8": evaluacion["claridad_razonamiento_b"],
                "completitud_q8": evaluacion["completitud_b"],
                "ganador": evaluacion["ganador"],
                "justificacion": evaluacion["justificacion"],
            })
            print(f"listo — ganador: {evaluacion['ganador']}")
        except Exception as e:
            print(f"ERROR: {e}")

        time.sleep(1)  # pausa entre llamadas a la API

    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with open(RUTA_SALIDA, "w", newline="", encoding="utf-8") as f:
        columnas = list(resultados[0].keys()) if resultados else []
        writer = csv.DictWriter(f, fieldnames=columnas)
        writer.writeheader()
        writer.writerows(resultados)

    print(f"\n  Guardado: {RUTA_SALIDA}")
    print(f"  Total evaluaciones: {len(resultados)}")

    # resumen
    ganadores = [r["ganador"] for r in resultados]
    print(f"\n  Resumen de victorias:")
    print(f"    Q4 (A):    {ganadores.count('A')}")
    print(f"    Q8 (B):    {ganadores.count('B')}")
    print(f"    Empates:   {ganadores.count('empate')}")


if __name__ == "__main__":
    main()

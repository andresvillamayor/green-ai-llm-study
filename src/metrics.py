"""
metrics.py — Proyecto GREEN-IA

Calculo de las metricas energeticas primarias del experimento.
Todas las metricas derivan de marcos reconocidos de Green AI, eficiencia
de ML, benchmarking de potencia y contabilidad de carbono de software.

Referencias:
  EDP   : Brooks, D. et al. (2000). Wattch: A Framework for Architectural-Level
           Power Analysis and Optimizations. ISCA 2000.
  SCI   : Green Software Foundation (2023). Software Carbon Intensity (SCI)
           Specification v1. https://sci.greensoftware.foundation/
  J/tok : Canziani, A. et al. (2017). An Analysis of Deep Neural Network
           Models for Practical Applications. arXiv:1605.07678.
           (accuracy-per-Joule como metrica de eficiencia de modelos)
  W/tok : Bannour, N. et al. (2021). Evaluating the Carbon Footprint of
           NLP Methods. EMNLP 2021. (metodologia de correccion de baseline)

IMPORTANTE: NO se usan indices compuestos ponderados como metricas primarias.
Si se agrega algun indice compuesto, se etiqueta explicitamente como
[EXPLORATORIO] o analisis de sensibilidad.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

_WH_TO_J: float = 3600.0  # 1 Wh = 3600 J


def compute_primary_metrics(
    net_energy_wh: float,
    gross_energy_wh: float,
    inference_time_s: float,
    tokens_output: int,
    net_emissions_g_co2: float,
) -> dict:
    """
    Computa las metricas energeticas primarias para un turno o conversacion.

    Metricas primarias (documentadas con referencia de literatura):

      net_energy_j               [J]         Joules netos de inferencia
      gross_energy_j             [J]         Joules brutos (sin correccion de baseline)
      net_j_per_output_token     [J/tok]     Joules por token de salida (Canziani 2017)
      net_j_per_response         [J]         Joules por respuesta/turno completo
      net_wh_per_1k_output_tokens[Wh/1Ktok]  Wh por 1000 tokens de salida (Bannour 2021)
      net_tokens_per_joule       [tok/J]     Tokens por Joule — eficiencia energetica
      sci_op_g_co2_per_1k_output_tokens
                                 [gCO2/1Ktok] SCI operacional (GSF 2023), R = 1K tokens
      sci_op_g_co2_per_response  [gCO2]      SCI operacional por respuesta, R = 1 respuesta
      edp_j_s                    [J·s]       Energy-Delay Product (Brooks et al. 2000)

    NOTA SCI: solo se reporta el componente operacional E × I.
    El carbono embebido (M) del hardware se omite por falta de datos de
    analisis de ciclo de vida (LCA) especificos para Apple M4 y NVIDIA RTX 4060.
    SCI = (E [kWh] × I [gCO2/kWh]) / R  →  net_emissions_g_co2 / R

    NOTA: quality_per_joule NO se computa aqui; se calcula post-hoc en
    juez_calidad_v2.py una vez disponible el score del juez.
    """
    net_j   = net_energy_wh   * _WH_TO_J
    gross_j = gross_energy_wh * _WH_TO_J

    tok = max(int(tokens_output), 0)

    net_j_per_tok  = net_j  / tok  if tok  > 0   else 0.0
    net_wh_per_1k  = (net_energy_wh / tok * 1_000.0) if tok > 0 else 0.0
    net_tok_per_j  = tok   / net_j if net_j > 0.0 else 0.0

    # SCI operacional (Green Software Foundation v1, 2023)
    sci_per_1k   = (net_emissions_g_co2 / tok * 1_000.0) if tok > 0 else 0.0
    sci_per_resp = net_emissions_g_co2

    # Energy-Delay Product: captura el trade-off energia-velocidad
    edp = net_j * inference_time_s

    return {
        "net_energy_j"                      : net_j,
        "gross_energy_j"                    : gross_j,
        "net_j_per_output_token"            : net_j_per_tok,
        "net_j_per_response"                : net_j,
        "net_wh_per_1k_output_tokens"       : net_wh_per_1k,
        "net_tokens_per_joule"              : net_tok_per_j,
        "sci_op_g_co2_per_1k_output_tokens" : sci_per_1k,
        "sci_op_g_co2_per_response"         : sci_per_resp,
        "edp_j_s"                           : edp,
    }


def compute_quality_metrics(
    quality_score: float,
    net_energy_wh: float,
) -> dict:
    """
    Computa quality_per_joule post-hoc, tras recibir el score del juez.

    quality_per_joule [puntos/J]:
      Adaptacion operacional de accuracy-per-Joule (Canziani et al. 2017).
      quality_per_joule = quality_score / net_energy_J

    NO es un indice compuesto ponderado: es una metrica de eficiencia de
    calidad directa, analoga a accuracy/energy en la literatura de Green AI.
    Ver tambien: Patterson et al. (2021). Carbon Dioxide Equivalent (CO2eq)
    Emissions of Language Models. arXiv:2104.10350.
    """
    net_j = net_energy_wh * _WH_TO_J
    qpj   = quality_score / net_j if net_j > 0.0 else float("nan")
    return {"quality_per_joule": qpj}


def pareto_optimal_3d(
    df: "pd.DataFrame",
    quality_col: str,
    energy_col: str,
    latency_col: str,
    group_col: str,
) -> "pd.Series":
    """
    Identifica configuraciones Pareto-optimas en espacio 3D (P16):
      quality  ↑ (mayor = mejor)
      energy   ↓ (menor = mejor)
      latency  ↓ (menor = mejor)

    Un punto A domina a B si:
      quality_A >= quality_B  AND  energy_A <= energy_B  AND
      latency_A <= latency_B  con al menos una desigualdad estricta.

    Retorna pd.Series booleana alineada con df.index.
    """
    import pandas as pd  # importacion local para no requerir pandas en tiempo de modulo

    resumen = (
        df.dropna(subset=[quality_col, energy_col, latency_col])
        .groupby(group_col)
        .agg(
            q=(quality_col,  "mean"),
            e=(energy_col,   "mean"),
            t=(latency_col,  "mean"),
        )
        .reset_index()
    )

    pareto_ids: set = set()
    rows = resumen.to_dict("records")
    for ra in rows:
        dominated = False
        for rb in rows:
            if rb[group_col] == ra[group_col]:
                continue
            if (
                rb["q"] >= ra["q"]
                and rb["e"] <= ra["e"]
                and rb["t"] <= ra["t"]
                and (rb["q"] > ra["q"] or rb["e"] < ra["e"] or rb["t"] < ra["t"])
            ):
                dominated = True
                break
        if not dominated:
            pareto_ids.add(ra[group_col])

    return df[group_col].isin(pareto_ids)

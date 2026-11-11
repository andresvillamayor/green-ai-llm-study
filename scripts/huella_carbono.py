#!/usr/bin/env python3
"""Huella de carbono por configuración (esquema, modelo, cuantización).

Entrada (solo lectura):
  results/comparacion_temperaturas/{modelo}_{esquema}/{modelo}_{q4|q8}/{categoria}/resultados_completos.csv

Filtro: es_warmup == "False" y TODAS las repeticiones (1 a 15). No se filtra por
`repeticion`. Se excluyen únicamente las filas con energía vacía (NaN), que se
cuentan y listan por configuración.

Energía: `measured_energy_mwh` = energía TOTAL (CPU + GPU + RAM) de UNA respuesta,
en mWh, estimada por CodeCarbon (TDP, no medición física). Se verifica que coincida
con cpu+gpu+ram y con measured_energy_kwh.

gCO2eq = kWh x INTENSIDAD_GCO2_KWH

Comparación Q4 vs Q8: se promedian primero las repeticiones de cada prompt; ahorro
por prompt = 1 - Q4/Q8; mediana y media del ahorro; Wilcoxon de rangos con signo
(scipy.stats.wilcoxon) sobre los pares prompt a prompt.

Salida: results/huella_carbono/huella_por_configuracion.csv
        results/huella_carbono/comparacion_q4_vs_q8.csv
"""
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

# Electricity Maps, Paraguay, octubre 2026 (estimación)
INTENSIDAD_GCO2_KWH = 25.0

RAIZ = Path(__file__).resolve().parent.parent
DIR_ENTRADA = RAIZ / "results" / "comparacion_temperaturas"
DIR_SALIDA = RAIZ / "results" / "huella_carbono"
COL_ENERGIA = "measured_energy_mwh"
MWH_POR_KWH = 1_000_000.0  # 1 kWh = 1e6 mWh
CLAVE_PROMPT = ["esquema", "modelo", "categoria", "prompt_id"]


def cargar() -> pd.DataFrame:
    """Filas sin warm-up, todas las repeticiones, con energía no vacía."""
    marcos = []
    for ruta in sorted(DIR_ENTRADA.glob("*/*/*/resultados_completos.csv")):
        modelo, esquema = ruta.parents[2].name.rsplit("_", 1)
        cuant = ruta.parents[1].name.rsplit("_", 1)[1]
        df = pd.read_csv(ruta)
        df = df[df["es_warmup"].astype(str).str.strip() == "False"].copy()
        df["esquema"], df["modelo"], df["cuant"] = esquema, modelo, cuant
        marcos.append(df)
    df = pd.concat(marcos, ignore_index=True)

    vacia = df[COL_ENERGIA].isna()
    print(f"Filas sin warm-up (repeticiones 1 a 15): {len(df)}. "
          f"Con {COL_ENERGIA} vacío (NaN): {int(vacia.sum())}; se excluyen solo esas.")
    for (e, m, c), g in df.groupby(["esquema", "modelo", "cuant"]):
        v = g[g[COL_ENERGIA].isna()]
        det = ", ".join(f"{r.categoria}/{r.prompt_id}/rep{r.repeticion}" for r in v.itertuples())
        print(f"  {m} {c} ({e}): {len(v)} excluida(s) de {len(g)}"
              + (f" [{det}]" if len(v) else ""))
    print()
    df = df[~vacia].copy()

    cols = ["cpu_energy_mwh", "gpu_energy_mwh", "ram_energy_mwh"]
    suma = df[cols].sum(axis=1)
    dif = (suma - df[COL_ENERGIA]).abs()
    dif_kwh = (df["measured_energy_kwh"] * MWH_POR_KWH - df[COL_ENERGIA]).abs()
    rel = dif / df[COL_ENERGIA]
    n_dif = int((dif > 1e-6).sum())
    print(f"Coherencia: filas con NaN en cpu/gpu/ram: {int(df[cols].isna().any(axis=1).sum())}; "
          f"máx |total - (cpu+gpu+ram)| = {dif.max():.2e} mWh "
          f"(relativo {rel.max():.2e}; {n_dif} fila(s) con diferencia > 1e-6 mWh, "
          "ruido de redondeo de CodeCarbon); "
          f"máx |kWh*1e6 - mWh| = {dif_kwh.max():.2e} mWh.\n")
    assert rel.max() < 1e-5, "total != cpu+gpu+ram"
    assert dif_kwh.max() < 1e-6, "measured_energy_kwh != measured_energy_mwh"
    return df


def tabla_config(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["esquema", "modelo", "cuant"])[COL_ENERGIA]
    t = g.agg(n_respuestas="count", energia_media_mwh="mean",
              energia_desv_mwh="std", energia_total_mwh="sum").reset_index()
    for c in ("media", "desv", "total"):
        t[f"energia_{c}_kwh"] = t[f"energia_{c}_mwh"] / MWH_POR_KWH
    t["gco2_media"] = t["energia_media_kwh"] * INTENSIDAD_GCO2_KWH
    t["gco2_total"] = t["energia_total_kwh"] * INTENSIDAD_GCO2_KWH
    return t[["esquema", "modelo", "cuant", "n_respuestas",
              "energia_media_mwh", "energia_desv_mwh", "energia_total_mwh",
              "energia_media_kwh", "energia_desv_kwh", "energia_total_kwh",
              "gco2_media", "gco2_total"]]


def comparacion(df: pd.DataFrame) -> pd.DataFrame:
    # Promedio por prompt (sobre sus repeticiones válidas), luego pares Q4/Q8
    pp = (df.groupby(CLAVE_PROMPT + ["cuant"])[COL_ENERGIA].mean()
            .unstack("cuant").reset_index())
    filas = []
    for (esq, mod), g in pp.groupby(["esquema", "modelo"]):
        if not {"q4", "q8"} <= set(g.columns):
            continue
        par = g.dropna(subset=["q4", "q8"])
        ahorro = 1 - par["q4"] / par["q8"]
        if len(par) >= 1 and (par["q4"] - par["q8"]).abs().sum() > 0:
            p = wilcoxon(par["q4"], par["q8"]).pvalue
        else:
            p = float("nan")
        filas.append({
            "esquema": esq, "modelo": mod,
            "prompts_q4": int(g["q4"].notna().sum()), "prompts_q8": int(g["q8"].notna().sum()),
            "pares_usados": len(par),
            "energia_media_q4_mwh": par["q4"].mean(), "energia_media_q8_mwh": par["q8"].mean(),
            "gco2_media_q4": par["q4"].mean() / MWH_POR_KWH * INTENSIDAD_GCO2_KWH,
            "gco2_media_q8": par["q8"].mean() / MWH_POR_KWH * INTENSIDAD_GCO2_KWH,
            "ahorro_mediana_pct": ahorro.median() * 100,
            "ahorro_media_pct": ahorro.mean() * 100,
            "wilcoxon_p": p,
        })
    return pd.DataFrame(filas)


def main():
    df = cargar()
    t = tabla_config(df)
    c = comparacion(df)
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    t.to_csv(DIR_SALIDA / "huella_por_configuracion.csv", index=False)
    c.to_csv(DIR_SALIDA / "comparacion_q4_vs_q8.csv", index=False)

    pd.set_option("display.width", 250, "display.float_format", "{:.6g}".format)
    print(f"Intensidad: {INTENSIDAD_GCO2_KWH} gCO2eq/kWh (Electricity Maps, Paraguay, "
          "oct 2026, estimación). Energía: estimación CodeCarbon (TDP).\n")
    print("== Huella por configuración (todas las repeticiones, sin warm-up) ==")
    print(t.rename(columns={
        "cuant": "cuantización", "n_respuestas": "respuestas",
        "energia_media_mwh": "E media (mWh)", "energia_desv_mwh": "desv (mWh)",
        "energia_total_mwh": "E total (mWh)", "energia_media_kwh": "E media (kWh)",
        "energia_desv_kwh": "desv (kWh)", "energia_total_kwh": "E total (kWh)",
        "gco2_media": "gCO2eq media", "gco2_total": "gCO2eq total"}).to_string(index=False))
    print("\n== Comparación Q4 vs Q8 (por prompt; ahorro = 1 - Q4/Q8; negativo = Q4 consume más) ==")
    print(c.rename(columns={
        "energia_media_q4_mwh": "E Q4 (mWh)", "energia_media_q8_mwh": "E Q8 (mWh)",
        "gco2_media_q4": "gCO2 Q4", "gco2_media_q8": "gCO2 Q8",
        "ahorro_mediana_pct": "ahorro mediana %", "ahorro_media_pct": "ahorro media %",
        "wilcoxon_p": "p Wilcoxon"}).to_string(index=False))
    sin_par = t.groupby(["esquema", "modelo"]).cuant.nunique()
    for (e, m), n in sin_par[sin_par < 2].items():
        print(f"\nNota: {m} ({e}) solo tiene una cuantización; sin comparación.")
    print(f"\nGuardado en {DIR_SALIDA.relative_to(RAIZ)}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Energía por respuesta y por token, por esquema, modelo y cuantización.

Entrada (solo lectura): results/comparacion_temperaturas/*/*/*/resultados_completos.csv
Filtro: es_warmup == "False" y TODAS las repeticiones (1 a 15); se excluyen solo las
filas con energía vacía (ver huella_carbono.cargar). Se descartan además filas con
completion_tokens vacío o 0 (se avisa).

Columnas: completion_tokens (tokens generados), measured_energy_mwh (energía total
CPU+GPU+RAM de una respuesta, estimación CodeCarbon/TDP, no medición física).
Energía por token = suma de energía / suma de tokens.
Ahorro de Q4 vs Q8 = (1 - Q4/Q8) * 100 sobre los promedios de la configuración
(positivo = Q4 consume menos).

Salida: results/huella_carbono/energia_por_token.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from huella_carbono import COL_ENERGIA, DIR_SALIDA, RAIZ, cargar  # noqa: E402

COL_TOKENS = "completion_tokens"


def tabla(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["esquema", "modelo", "cuant"])
    t = g.agg(n_respuestas=(COL_ENERGIA, "count"),
              tokens_medios=(COL_TOKENS, "mean"),
              energia_media_mwh=(COL_ENERGIA, "mean"),
              energia_total_mwh=(COL_ENERGIA, "sum"),
              tokens_totales=(COL_TOKENS, "sum")).reset_index()
    t["energia_por_token_mwh"] = t["energia_total_mwh"] / t["tokens_totales"]
    return t.drop(columns=["energia_total_mwh", "tokens_totales"])


def ahorro(t: pd.DataFrame) -> pd.DataFrame:
    q4 = t[t.cuant == "q4"].set_index(["esquema", "modelo"])
    q8 = t[t.cuant == "q8"].set_index(["esquema", "modelo"])
    idx = q4.index.intersection(q8.index)
    a = pd.DataFrame(index=idx)
    for col, nom in [("tokens_medios", "ahorro_tokens_pct"),
                     ("energia_media_mwh", "ahorro_energia_respuesta_pct"),
                     ("energia_por_token_mwh", "ahorro_energia_token_pct")]:
        a[nom] = (1 - q4.loc[idx, col] / q8.loc[idx, col]) * 100
    return a.reset_index()


def main():
    df = cargar()
    n0 = len(df)
    df = df.dropna(subset=[COL_TOKENS])
    df = df[df[COL_TOKENS] > 0]
    if len(df) < n0:
        print(f"Aviso: {n0 - len(df)} fila(s) descartada(s) por tokens vacíos o 0.\n")
    t = tabla(df)
    a = ahorro(t)
    out = t.merge(a, on=["esquema", "modelo"], how="left")
    cols_a = a.columns.difference(["esquema", "modelo"])
    out.loc[out.cuant != "q4", cols_a] = pd.NA
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    out.to_csv(DIR_SALIDA / "energia_por_token.csv", index=False)

    pd.set_option("display.width", 200, "display.float_format", "{:.4f}".format)
    print("Energía: estimación CodeCarbon (TDP). Todas las repeticiones, sin warm-up.\n")
    print("== Por configuración ==")
    print(t.rename(columns={"cuant": "cuantización", "n_respuestas": "respuestas",
          "tokens_medios": "tokens medios", "energia_media_mwh": "E/respuesta (mWh)",
          "energia_por_token_mwh": "E/token (mWh)"}).to_string(index=False))
    print("\n== Ahorro de Q4 frente a Q8 (%; negativo = Q4 consume más) ==")
    print(a.rename(columns={"ahorro_tokens_pct": "tokens %",
          "ahorro_energia_respuesta_pct": "E/respuesta %",
          "ahorro_energia_token_pct": "E/token %"}).to_string(index=False))
    sp = t.groupby(["esquema", "modelo"]).cuant.nunique()
    for (e, m), n in sp[sp < 2].items():
        print(f"\nNota: {m} ({e}) solo tiene una cuantización; sin ahorro.")
    print(f"\nGuardado en {(DIR_SALIDA / 'energia_por_token.csv').relative_to(RAIZ)}")


if __name__ == "__main__":
    main()

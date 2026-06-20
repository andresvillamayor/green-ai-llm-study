# analisis_desviaciones.py
# Analiza las desviaciones estandar y outliers del experimento
# Proyecto GREEN-IA — Andres Villamayor
#
# Este programa no elimina datos ni modifica el CSV.
# Solo lee los resultados y muestra lo que hay, sin agregar interpretaciones
# que no esten respaldadas por los propios datos del experimento.

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results" / "measurements"


def cargar_csv():
    archivos = sorted(RESULTS_DIR.glob("experimento_codecarbon_*.csv"))
    if not archivos:
        raise FileNotFoundError(f"no hay CSVs en {RESULTS_DIR}")
    ruta = archivos[-1]
    print(f"archivo: {ruta.name}")
    return pd.read_csv(ruta)


def calcular_cv(valores):
    # coeficiente de variacion: que tan dispersas estan las mediciones
    # CV = desviacion estandar / media * 100
    # un CV bajo significa que las repeticiones fueron consistentes entre si
    if len(valores) < 2:
        return 0.0
    media = np.mean(valores)
    if media == 0:
        return 0.0
    return np.std(valores, ddof=1) / media * 100


def detectar_outliers_iqr(valores, factor=3.0):
    # el metodo IQR calcula el rango intercuartil (diferencia entre Q3 y Q1)
    # una observacion es outlier si esta mas alla de factor*IQR del cuartil
    # con factor=3.0 solo se marcan valores verdaderamente extremos
    # un factor de 1.5 es el estandar, 3.0 es mas conservador
    q1 = np.percentile(valores, 25)
    q3 = np.percentile(valores, 75)
    iqr = q3 - q1
    limite_inf = q1 - factor * iqr
    limite_sup = q3 + factor * iqr
    return limite_inf, limite_sup


def analizar(df):
    df["quantization"] = df["quantization"].str.upper()
    df["device"]       = df["device"].str.upper()

    print("\n" + "=" * 60)
    print("  resumen de desviaciones por configuracion")
    print("=" * 60)

    total_outliers = 0

    for (modelo, quant, device), grp in df.groupby(["model", "quantization", "device"]):
        tiempos = grp["inference_time_s"].values
        media   = np.mean(tiempos)
        std     = np.std(tiempos, ddof=1)
        cv      = calcular_cv(tiempos)
        lim_inf, lim_sup = detectar_outliers_iqr(tiempos)

        outliers = grp[
            (grp["inference_time_s"] < lim_inf) |
            (grp["inference_time_s"] > lim_sup)
        ]

        print(f"\n  {modelo} | {quant} | {device}")
        print(f"  repeticiones : {len(grp)}")
        print(f"  media        : {media:.2f} s")
        print(f"  std          : {std:.2f} s")
        print(f"  CV           : {cv:.1f}%")
        print(f"  minimo       : {tiempos.min():.2f} s")
        print(f"  maximo       : {tiempos.max():.2f} s")

        if cv < 10:
            print(f"  estabilidad  : las {len(grp)} repeticiones son consistentes entre si")
        else:
            print(f"  estabilidad  : hay variabilidad alta, ver outliers abajo")

        if outliers.empty:
            print(f"  outliers     : ninguno con IQR x3")
        else:
            print(f"  outliers     : {len(outliers)} encontrados")
            for _, row in outliers.iterrows():
                diferencia = row["inference_time_s"] - media
                veces      = row["inference_time_s"] / media
                print(f"    repeticion {int(row['repetition'])}: "
                      f"{row['inference_time_s']:.2f} s  "
                      f"({veces:.1f}x la media, "
                      f"{diferencia:+.2f} s respecto al promedio)")
            total_outliers += len(outliers)

    print("\n" + "=" * 60)
    print(f"  total outliers encontrados: {total_outliers} de {len(df)} mediciones")
    print(f"  porcentaje: {total_outliers / len(df) * 100:.1f}%")
    print("=" * 60)


def mostrar_patron_outliers(df):
    # muestra si los outliers tienen algun patron en comun
    # por ejemplo si siempre son la primera repeticion, o siempre en cpu, etc.
    # esto ayuda a entender si hay una causa sistematica o si son aleatorios
    df["quantization"] = df["quantization"].str.upper()
    df["device"]       = df["device"].str.upper()

    print("\n" + "=" * 60)
    print("  patron de los outliers")
    print("=" * 60)

    registros = []

    for (modelo, quant, device), grp in df.groupby(["model", "quantization", "device"]):
        tiempos = grp["inference_time_s"].values
        lim_inf, lim_sup = detectar_outliers_iqr(tiempos)
        outliers = grp[
            (grp["inference_time_s"] < lim_inf) |
            (grp["inference_time_s"] > lim_sup)
        ]
        for _, row in outliers.iterrows():
            registros.append({
                "modelo"     : modelo,
                "quant"      : quant,
                "device"     : device,
                "repeticion" : int(row["repetition"]),
                "prompt_id"  : int(row["prompt_id"]),
                "tiempo_s"   : round(row["inference_time_s"], 2),
            })

    if not registros:
        print("  no hay outliers con el criterio IQR x3")
        return

    out_df = pd.DataFrame(registros)

    # repeticiones mas frecuentes entre los outliers
    print("\n  repeticiones donde aparecen los outliers:")
    rep_conteo = out_df["repeticion"].value_counts().sort_index()
    for rep, n in rep_conteo.items():
        print(f"    repeticion {rep}: {n} veces")

    # prompts mas frecuentes entre los outliers
    print("\n  prompts donde aparecen los outliers:")
    prompt_conteo = out_df["prompt_id"].value_counts().sort_index()
    for pid, n in prompt_conteo.items():
        print(f"    prompt {pid}: {n} veces")

    # dispositivos
    print("\n  dispositivos donde aparecen los outliers:")
    dev_conteo = out_df["device"].value_counts()
    for dev, n in dev_conteo.items():
        print(f"    {dev}: {n} veces")

    print("\n  detalle completo:")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    print("=" * 60)
    print("  GREEN-IA: analisis de desviaciones y outliers")
    print("=" * 60)

    df = cargar_csv()

    analizar(df)
    mostrar_patron_outliers(df)

    print("\n  listo.")
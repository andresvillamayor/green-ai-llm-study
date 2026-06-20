# diagnosticar_datos.py
# SOLO DIAGNOSTICO — este script no modifica ni elimina datos.
# Su función es verificar completitud y detectar mediciones anómalas
# antes de proceder con el análisis gráfico.

import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results" / "measurements"

archivos = sorted(RESULTS_DIR.glob("experimento_codecarbon_*.csv"))
if not archivos:
    print(f"No hay CSVs en {RESULTS_DIR}")
    exit()

ruta = archivos[-1]
df   = pd.read_csv(ruta)

for col in ["gpu_power_w", "gpu_energy_wh", "emissions_mg_co2", "total_energy_wh"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("=" * 60)
print(f"  Archivo : {ruta.name}")
print(f"  Filas   : {len(df)}")
print(f"  Columnas: {len(df.columns)}")
print("=" * 60)

print(f"\nModelos      : {sorted(df['model'].unique())}")
print(f"Cuantizacion : {sorted(df['quantization'].unique())}")
print(f"Dispositivos : {sorted(df['device'].unique())}")

# repeticiones por configuracion
print("\nRepeticiones por configuracion:")
conteo = df.groupby(["model", "quantization", "device"]).size().reset_index(name="n")
for _, r in conteo.iterrows():
    estado = "OK" if r["n"] >= 10 else f"INCOMPLETO ({r['n']})"
    print(f"  {r['model']:<15} {r['quantization']:<4} {r['device']:<4}  {r['n']:>3} reps  [{estado}]")

# columnas con valores nulos
nulos = df.isnull().sum()
nulos = nulos[nulos > 0]
if nulos.empty:
    print("\nNo hay valores nulos.")
else:
    print("\nColumnas con nulos:")
    for col, n in nulos.items():
        print(f"  {col}: {n} nulos")

# outliers via IQR x3 (criterio Tukey 1977, factor conservador)
# consistente con el criterio usado en graficar_resultados.py
print("\n" + "=" * 60)
print("  Outliers por configuracion — criterio IQR x3 (Tukey, 1977)")
print("=" * 60)

total_outliers = 0
df["outlier"]  = False

for (mod, q, dev), grp in df.groupby(["model", "quantization", "device"]):
    tiempos = grp["inference_time_s"].to_numpy(dtype=float)
    q1      = np.percentile(tiempos, 25)
    q3      = np.percentile(tiempos, 75)
    iqr     = q3 - q1
    lim_inf = q1 - 3 * iqr
    lim_sup = q3 + 3 * iqr
    media   = float(np.mean(tiempos))

    print(f"\n  {mod}  {q}  {dev}")
    print(f"    Q1         = {q1:.3f} s")
    print(f"    Q3         = {q3:.3f} s")
    print(f"    IQR        = {iqr:.3f} s")
    print(f"    umbral inf = Q1 - 3*IQR = {lim_inf:.3f} s")
    print(f"    umbral sup = Q3 + 3*IQR = {lim_sup:.3f} s")

    outs = grp[(grp["inference_time_s"] < lim_inf) | (grp["inference_time_s"] > lim_sup)]
    df.loc[outs.index, "outlier"] = True
    if outs.empty:
        print(f"    outliers   : ninguno")
    else:
        print(f"    outliers   : {len(outs)}")
        for _, row in outs.iterrows():
            veces    = row["inference_time_s"] / media
            e_tot    = row["total_energy_wh"]  * 1000
            e_cpu    = row["cpu_energy_wh"]    * 1000
            e_gpu    = row["gpu_energy_wh"]    * 1000
            e_ram    = row["ram_energy_wh"]    * 1000
            print(f"      rep {int(row['repetition'])}  prompt {int(row['prompt_id'])}")
            print(f"        duracion : {row['inference_time_s']:.2f} s  "
                  f"({veces:.1f}x la media de {media:.1f} s)")
            print(f"        energia  : {e_tot:.4f} mWh total  "
                  f"(CPU {e_cpu:.4f}  GPU {e_gpu:.4f}  RAM {e_ram:.4f})")
            total_outliers += 1

print(f"\n{'=' * 60}")
print(f"  Total outliers identificados : {total_outliers} de {len(df)} "
      f"({total_outliers / len(df) * 100:.1f}%)")
print(f"  Estos valores NO se eliminan — se documentan y se muestran")
print(f"  en las graficas con marcador rojo.")
print(f"{'=' * 60}")

# tabla comparativa: estadisticas con y sin outliers por configuracion
df_clean = df[~df["outlier"]]

W = 96
print("\n" + "=" * W)
print("  TABLA COMPARATIVA — impacto de outliers en estadisticas clave")
print("=" * W)
print(f"  {'Configuracion':<26} {'n':>3} {'ok':>4}  "
      f"{'tiempo medio (s)':^30}  {'energia media (mWh)':^28}  {'tok/s':^20}")
print(f"  {'':26} {'':3} {'':4}  "
      f"{'todos':>8} {'limpio':>9} {'Δ%':>9}  "
      f"{'todos':>8} {'limpio':>9} {'Δ%':>7}  "
      f"{'todos':>8} {'limpio':>9}")
print("  " + "-" * (W - 2))

for (mod, q, dev), grp_f in df.groupby(["model", "quantization", "device"]):
    grp_c = df_clean[
        (df_clean["model"]        == mod) &
        (df_clean["quantization"] == q)   &
        (df_clean["device"]       == dev)
    ]

    n_f = len(grp_f)
    n_c = len(grp_c)

    t_f  = grp_f["inference_time_s"].mean()
    t_c  = grp_c["inference_time_s"].mean() if n_c > 0 else t_f
    t_d  = (t_f - t_c) / t_c * 100 if t_c != 0 else 0.0

    e_f  = grp_f["total_energy_wh"].mean() * 1000
    e_c  = grp_c["total_energy_wh"].mean() * 1000 if n_c > 0 else e_f
    e_d  = (e_f - e_c) / e_c * 100 if e_c != 0 else 0.0

    tps_f = grp_f["tokens_per_second"].mean()
    tps_c = grp_c["tokens_per_second"].mean() if n_c > 0 else tps_f

    config = f"{mod} {q} {dev}"
    print(f"  {config:<26} {n_f:>3} {n_c:>4}  "
          f"{t_f:>8.2f} {t_c:>9.2f} {t_d:>+8.1f}%  "
          f"{e_f:>8.4f} {e_c:>9.4f} {e_d:>+6.1f}%  "
          f"{tps_f:>8.1f} {tps_c:>9.1f}")

print("  " + "-" * (W - 2))
print(f"  Δ% = (todos - limpio) / limpio * 100  "
      f"— valores cercanos a 0% indican que los outliers no afectan las conclusiones.")
print("=" * W)

print("\nListo. Si todo esta OK, corre graficar_resultados.py")

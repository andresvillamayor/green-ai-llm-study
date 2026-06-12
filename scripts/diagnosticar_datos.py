# diagnosticar_datos.py
# Chequeo rapido del CSV antes de graficar
# Corre esto primero para verificar que los datos esten completos
# Proyecto GREEN-IA — Andres Villamayor

import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results" / "measurements"

# busca el CSV mas reciente
archivos = sorted(RESULTS_DIR.glob("experimento_codecarbon_*.csv"))
if not archivos:
    print(f"No hay CSVs en {RESULTS_DIR}")
    exit()

ruta = archivos[-1]
df   = pd.read_csv(ruta)

# convierte columnas que pueden tener vacios
for col in ["gpu_power_w", "gpu_energy_wh", "emissions_mg_co2", "total_energy_wh"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("=" * 55)
print(f"  Archivo : {ruta.name}")
print(f"  Filas   : {len(df)}")
print(f"  Columnas: {len(df.columns)}")
print("=" * 55)

# modelos, cuantizaciones y dispositivos presentes
print(f"\nModelos      : {sorted(df['model'].unique())}")
print(f"Cuantizacion : {sorted(df['quantization'].unique())}")
print(f"Dispositivos : {sorted(df['device'].unique())}")

# repeticiones por configuracion — aca se ve si falta algo
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
    print(f"\nColumnas con nulos:")
    for col, n in nulos.items():
        print(f"  {col}: {n} nulos")

# outliers rapidos por tiempo de inferencia
print("\nOutliers de tiempo (> media + 3*std por config):")
encontro = False
for (mod, q, dev), grp in df.groupby(["model", "quantization", "device"]):
    media = grp["inference_time_s"].mean()
    std   = grp["inference_time_s"].std()
    outs  = grp[grp["inference_time_s"] > media + 3 * std]
    for _, row in outs.iterrows():
        print(f"  {mod} {q} {dev}  rep {row['repetition']}  {row['inference_time_s']:.1f}s  (media={media:.1f}s)")
        encontro = True
if not encontro:
    print("  Ninguno.")

print("\nListo. Si todo esta OK, corre graficar_resultados.py")
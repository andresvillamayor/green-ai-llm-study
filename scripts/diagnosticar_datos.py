"""
diagnosticar_datos.py
Verifica los valores exactos en el CSV
"""

import pandas as pd

df = pd.read_csv("results/measurements_detailed/energy_breakdown_20260509_105050.csv")

print("=" * 70)
print("DIAGNÓSTICO DE DATOS")
print("=" * 70)

# Ver valores únicos de cuantización
print("\n1. Valores únicos de 'quantization':")
print(df['quantization'].unique())
print(f"   Total valores únicos: {df['quantization'].nunique()}")

# Ver valores únicos de device
print("\n2. Valores únicos de 'device':")
print(df['device'].unique())

# Ver valores únicos de model
print("\n3. Valores únicos de 'model':")
print(df['model'].unique())

# Contar por configuración
print("\n4. Configuraciones completas:")
config_counts = df.groupby(['model', 'quantization', 'device']).size()
print(config_counts)

# Estadísticas agregadas
print("\n5. Estadísticas por configuración:")
stats = df.groupby(['model', 'quantization', 'device']).agg({
    'total_energy_wh': ['mean', 'count']
}).reset_index()
print(stats)
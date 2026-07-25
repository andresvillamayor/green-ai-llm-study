"""Regenera los 3 PNG de analisis_categoria para todos los CSVs existentes.

Recorre results/analisis_categoria/**/{modelo}_{cuant}/{categoria}/resultados_completos.csv,
reconstruye los diccionarios de datos y llama a generar_plots() del script original.
No carga ningun modelo ni requiere sudo.
"""

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analisis_categoria import generar_plots, calcular_estadisticas


def reconstruir_datos(csv_path: Path) -> tuple[dict, dict, dict, dict, dict]:
    """Lee el CSV y agrupa mediciones por prompt_num."""
    df = pd.read_csv(csv_path)
    datos_mwh: dict = defaultdict(list)
    datos_cpu: dict = defaultdict(list)
    datos_gpu: dict = defaultdict(list)
    datos_ram: dict = defaultdict(list)
    etiquetas: dict = {}
    for num in sorted(df["prompt_num"].unique()):
        fila = df[df["prompt_num"] == num]
        datos_mwh[num] = fila["measured_energy_mwh"].tolist()
        datos_cpu[num] = fila["cpu_energy_mwh"].tolist()
        datos_gpu[num] = fila["gpu_energy_mwh"].tolist()
        datos_ram[num] = fila["ram_energy_mwh"].tolist()
        etiquetas[num] = f"Prompt {num}"
    return dict(datos_mwh), dict(datos_cpu), dict(datos_gpu), dict(datos_ram), etiquetas


def main() -> None:
    base = ROOT / "results" / "analisis_categoria"
    csvs = sorted(base.rglob("resultados_completos.csv"))

    if not csvs:
        print(f"No se encontraron CSVs bajo {base.relative_to(ROOT)}")
        sys.exit(0)

    print(f"CSVs encontrados: {len(csvs)}\n")
    procesadas = 0

    for csv_path in csvs:
        directorio = csv_path.parent
        # La categoria es el nombre de la carpeta que contiene el CSV
        categoria = directorio.name
        ruta_relativa = csv_path.relative_to(ROOT)
        print(f"[{procesadas + 1}/{len(csvs)}] Procesando: {ruta_relativa}")

        try:
            datos_mwh, datos_cpu, datos_gpu, datos_ram, etiquetas = reconstruir_datos(csv_path)
            generar_plots(directorio, categoria, datos_mwh, etiquetas, datos_cpu, datos_gpu, datos_ram)
            procesadas += 1
            print(f"  OK\n")
        except Exception as exc:
            print(f"  ERROR: {exc}\n")

    print(f"Resumen: {procesadas}/{len(csvs)} carpetas procesadas correctamente.")


if __name__ == "__main__":
    main()

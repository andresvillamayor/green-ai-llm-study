#!/usr/bin/env python3
"""
Generación de gráficos comparativos para tesis GREEN AI.

Este script lee los resultados del benchmark y genera:
1. Barras comparativas de tok/s por configuración (con error bars)
2. Barras de J/token (eficiencia energética)
3. Comparativa de CO2: Paraguay vs España

Autor: Andrés Rubén Villamayor Ruiz Diaz
Fecha: 2026
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurar matplotlib para estilo académico
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.figsize'] = (10, 6)


def load_results(csv_path):
    """Carga resultados desde CSV y calcula estadísticas por configuración."""
    if not os.path.exists(csv_path):
        logger.error("Archivo no encontrado: %s", csv_path)
        return None
    
    df = pd.read_csv(csv_path)
    
    # Filtrar solo resultados exitosos
    df = df[df["success"] == True].copy()
    
    if df.empty:
        logger.warning("No hay datos válidos para graficar")
        return None
    
    # Agrupar por configuración y calcular estadísticas
    group_cols = ["model_key", "framework", "quant", "task"]
    stats = df.groupby(group_cols).agg({
        "tokens_per_s": ["mean", "std", "count"],
        "joules_per_token": ["mean", "std"],
        "co2_py_g": ["mean", "std"],
        "co2_es_g": ["mean", "std"]
    }).round(3)
    
    # Aplanar columnas multi-nivel
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    stats = stats.reset_index()
    
    logger.info("Datos cargados: %d configuraciones", len(stats))
    return stats


def plot_tok_per_s(stats, output_path="results/plot_tok_per_s.png"):
    """Gráfico de barras: tokens por segundo con error bars."""
    plt.figure(figsize=(12, 7))
    
    # Preparar datos
    labels = [f"{row['model_key']}\\n({row['task']})" for _, row in stats.iterrows()]
    values = stats["tokens_per_s_mean"].values
    errors = stats["tokens_per_s_std"].values
    
    # Colores por framework
    colors = ['#2E86AB' if 'llama' in f else '#A23B72' if 'MLX' in f else '#F18F01' 
              for f in stats["framework"]]
    
    # Gráfico de barras con error bars
    bars = plt.barh(labels, values, xerr=errors, color=colors, 
                   edgecolor='black', capsize=5, alpha=0.9)
    
    plt.xlabel('Tokens por segundo (tok/s)')
    plt.title('Velocidad de inferencia por configuración\nMac Mini M4 16GB - Media ± Desviación Estándar')
    plt.gca().invert_yaxis()  # Mejor legibilidad
    plt.tight_layout()
    
    # Leyenda de frameworks
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E86AB', edgecolor='black', label='llama-cpp-python'),
        Patch(facecolor='#A23B72', edgecolor='black', label='MLX')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info("Gráfico guardado: %s", output_path)
    plt.close()


def plot_energy_efficiency(stats, output_path="results/plot_joules_per_token.png"):
    """Gráfico de eficiencia energética: Joules por token (menor = mejor)."""
    plt.figure(figsize=(12, 7))
    
    labels = [f"{row['model_key']}\\n({row['task']})" for _, row in stats.iterrows()]
    values = stats["joules_per_token_mean"].values
    errors = stats["joules_per_token_std"].values
    
    # Invertir colores: menor consumo = verde, mayor = rojo
    colors = plt.cm.RdYlGn(1 - np.array(values) / max(values))
    
    bars = plt.barh(labels, values, xerr=errors, color=colors, 
                   edgecolor='black', capsize=5, alpha=0.9)
    
    plt.xlabel('Energía por token (J/token) - Menor es mejor')
    plt.title('Eficiencia energética por configuración\nEstimación software: 40W Apple M4')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info("Gráfico guardado: %s", output_path)
    plt.close()


def plot_co2_comparison(stats, output_path="results/plot_co2_comparison.png"):
    """Gráfico comparativo de CO2: Paraguay vs España."""
    plt.figure(figsize=(12, 7))
    
    labels = [f"{row['model_key']}\\n({row['task']})" for _, row in stats.iterrows()]
    x = np.arange(len(labels))
    width = 0.35
    
    py_values = stats["co2_py_g_mean"].values
    es_values = stats["co2_es_g_mean"].values
    py_errors = stats["co2_py_g_std"].values
    es_errors = stats["co2_es_g_std"].values
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bars1 = ax.barh(x - width/2, py_values, width, label='Paraguay (70 g/kWh)', 
                   color='#2ECC71', edgecolor='black', capsize=5, alpha=0.9)
    bars2 = ax.barh(x + width/2, es_values, width, label='España (245 g/kWh)', 
                   color='#E74C3C', edgecolor='black', capsize=5, alpha=0.9)
    
    ax.set_xlabel('Emisiones de CO₂ por inferencia (gramos)')
    ax.set_title('Impacto ambiental por país: Paraguay vs España\nFactor de emisión × Energía estimada')
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info("Gráfico guardado: %s", output_path)
    plt.close()


def generate_all_plots(csv_input="results/benchmark_comparativa_crudo.csv", 
                      output_dir="results/plots"):
    """Genera todos los gráficos y guarda en carpeta especificada."""
    os.makedirs(output_dir, exist_ok=True)
    
    stats = load_results(csv_input)
    if stats is None:
        return False
    
    plot_tok_per_s(stats, os.path.join(output_dir, "tok_per_s.png"))
    plot_energy_efficiency(stats, os.path.join(output_dir, "joules_per_token.png"))
    plot_co2_comparison(stats, os.path.join(output_dir, "co2_comparison.png"))
    
    logger.info("✅ Todos los gráficos generados en: %s", output_dir)
    return True


if __name__ == "__main__":
    generate_all_plots()
#!/usr/bin/env python3
"""
Generación de gráficos comparativos para tesis GREEN AI.

Lee los resultados del benchmark y genera visualizaciones profesionales:
- Velocidad de inferencia (tok/s) por modelo y cuantización
- Eficiencia energética (J/token)
- Emisiones de CO2 por país
- Comparativa INT4 vs INT8

Autor: Andrés Rubén Villamayor Ruiz Diaz
Fecha: 2026
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from pathlib import Path

# Configurar estilo profesional para gráficos
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

def load_results(csv_path: str = "results/comparison_table.csv") -> pd.DataFrame:
    """
    Carga resultados desde CSV con manejo robusto de errores.
    
    Args:
        csv_path: Ruta al archivo CSV
    
    Returns:
        DataFrame con los resultados
    
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el archivo está vacío o no tiene columnas válidas
    """
    # Verificar que el archivo existe
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "Archivo no encontrado: %s\n"
            "Ejecutá primero: python scripts/comprehensive_benchmark.py" % csv_path
        )
    
    # Verificar que el archivo no está vacío
    if os.path.getsize(csv_path) == 0:
        raise ValueError(
            "Archivo vacío: %s\n"
            "Ejecutá primero: python scripts/comprehensive_benchmark.py" % csv_path
        )
    
    # Intentar cargar el CSV
    try:
        df = pd.read_csv(csv_path)
        
        # Verificar que tiene columnas
        if df.empty or len(df.columns) == 0:
            raise ValueError("El archivo CSV no tiene columnas válidas")
        
        print("Datos cargados: %d configuraciones" % len(df))
        return df
        
    except pd.errors.EmptyDataError:
        raise ValueError(
            "El archivo CSV está vacío o no tiene datos válidos: %s\n"
            "Ejecutá primero: python scripts/comprehensive_benchmark.py" % csv_path
        )
    except pd.errors.ParserError as e:
        raise ValueError(
            "Error al parsear el archivo CSV: %s\n"
            "Verificá que el archivo tenga formato CSV válido" % e
        )

def plot_inference_speed(df: pd.DataFrame, output_path: str = "results/plots/inference_speed.png"):
    """
    Gráfico de barras: velocidad de inferencia (tok/s).
    """
    plt.figure(figsize=(12, 7))
    
    # Preparar datos
    labels = ["%s\n(%s)" % (row['Model'], row['Quantization']) for _, row in df.iterrows()]
    values = df['Tokens_Per_Sec'].values
    
    # Colores por tipo de cuantización
    colors = ['#2E86AB' if 'INT4' in str(q) else '#E94F37' for q in df['Quantization']]
    
    # Gráfico de barras horizontales
    bars = plt.barh(labels, values, color=colors, edgecolor='black', alpha=0.9)
    
    # Agregar valores en las barras
    for bar, value in zip(bars, values):
        plt.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                '%.2f' % value, va='center', fontsize=10)
    
    plt.xlabel('Tokens por segundo (tok/s)', fontsize=12)
    plt.title('Velocidad de Inferencia por Modelo y Cuantización\nMac Mini M4 16GB', 
             fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Leyenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E86AB', label='INT4 (4-bit)'),
        Patch(facecolor='#E94F37', label='INT8 (8-bit)')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("Gráfico guardado: %s" % output_path)
    plt.close()


def plot_energy_efficiency(df: pd.DataFrame, output_path: str = "results/plots/energy_efficiency.png"):
    """
    Gráfico de eficiencia energética (J/token aproximado).
    """
    plt.figure(figsize=(12, 7))
    
    labels = ["%s\n(%s)" % (row['Model'], row['Quantization']) for _, row in df.iterrows()]
    # Calcular J/token aproximado: Energy_Wh * 3600000 / 100 tokens
    values = df['Energy_Wh'].values * 3600000 / 100
    
    # Colores: verde para eficiente, rojo para menos eficiente
    colors = plt.cm.RdYlGn(1 - np.array(values) / max(values))
    
    bars = plt.barh(labels, values, color=colors, edgecolor='black', alpha=0.9)
    
    # Agregar valores
    for bar, value in zip(bars, values):
        plt.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                '%.2f' % value, va='center', fontsize=10)
    
    plt.xlabel('Energía por token (J/token) - Menor es mejor', fontsize=12)
    plt.title('Eficiencia Energética por Configuración\nEstimación software: 40W Apple M4',
             fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("Gráfico guardado: %s" % output_path)
    plt.close()


def plot_co2_comparison(df: pd.DataFrame, output_path: str = "results/plots/co2_comparison.png"):
    """
    Gráfico comparativo de emisiones de CO2 por país.
    """
    plt.figure(figsize=(14, 8))
    
    # Seleccionar países para comparar
    countries = ['PY', 'ES', 'DE', 'US']
    country_names = {'PY': 'Paraguay', 'ES': 'España', 'DE': 'Alemania', 'US': 'EE.UU.'}
    
    labels = ["%s\n(%s)" % (row['Model'], row['Quantization']) for _, row in df.iterrows()]
    x = np.arange(len(labels))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, country in enumerate(countries):
        col_name = 'CO2_%s_g' % country
        if col_name in df.columns:
            values = df[col_name].values * 1000  # Convertir a mg para mejor visualización
            ax.barh(x + i*width - width/2, values, width, 
                   label='%s' % country_names[country], alpha=0.9)
    
    ax.set_xlabel('Emisiones de CO2 (mg por inferencia)', fontsize=12)
    ax.set_title('Emisiones de CO2 por País y Configuración\n100 tokens generados',
                fontsize=14, fontweight='bold')
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("Gráfico guardado: %s" % output_path)
    plt.close()


def plot_int4_vs_int8_comparison(df: pd.DataFrame, output_path: str = "results/plots/int4_vs_int8.png"):
    """
    Gráfico comparativo directo: INT4 vs INT8.
    
    Maneja casos donde no hay pares exactos de modelos entre cuantizaciones.
    """
    # Separar por cuantización
    df_int4 = df[df['Quantization'] == 'INT4'].copy()
    df_int8 = df[df['Quantization'] == 'INT8'].copy()
    
    # Encontrar modelos que tienen ambas cuantizaciones
    models_int4 = set(df_int4['Model'].values)
    models_int8 = set(df_int8['Model'].values)
    common_models = sorted(models_int4 & models_int8)
    
    if not common_models:
        print("Advertencia: No hay modelos comunes entre INT4 e INT8 para comparar")
        return
    
    # Filtrar solo modelos comunes
    df_int4_common = df_int4[df_int4['Model'].isin(common_models)].sort_values('Model')
    df_int8_common = df_int8[df_int8['Model'].isin(common_models)].sort_values('Model')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Gráfico 1: Velocidad
    ax1 = axes[0]
    speed_int4 = df_int4_common['Tokens_Per_Sec'].values
    speed_int8 = df_int8_common['Tokens_Per_Sec'].values
    
    x = np.arange(len(common_models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, speed_int4, width, label='INT4', color='#2E86AB')
    bars2 = ax1.bar(x + width/2, speed_int8, width, label='INT8', color='#E94F37')
    
    ax1.set_ylabel('Tokens por segundo', fontsize=11)
    ax1.set_title('Velocidad de Inferencia', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(common_models, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Agregar valores
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    '%.1f' % height, ha='center', va='bottom', fontsize=9)
    
    # Gráfico 2: Energía
    ax2 = axes[1]
    energy_int4 = df_int4_common['Energy_Wh'].values * 1000  # mWh
    energy_int8 = df_int8_common['Energy_Wh'].values * 1000
    
    bars3 = ax2.bar(x - width/2, energy_int4, width, label='INT4', color='#2E86AB')
    bars4 = ax2.bar(x + width/2, energy_int8, width, label='INT8', color='#E94F37')
    
    ax2.set_ylabel('Energía por inferencia (mWh)', fontsize=11)
    ax2.set_title('Consumo Energético', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(common_models, rotation=15, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Agregar valores
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    '%.2f' % height, ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Comparativa INT4 vs INT8 - Green AI LLM Study', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("Gráfico guardado: %s" % output_path)
    plt.close()


def generate_all_plots():
    """Genera todos los gráficos y los guarda en results/plots/."""
    # Crear directorio
    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar datos
    df = load_results("results/comparison_table.csv")
    
    # Generar gráficos
    plot_inference_speed(df, plots_dir / "inference_speed.png")
    plot_energy_efficiency(df, plots_dir / "energy_efficiency.png")
    plot_co2_comparison(df, plots_dir / "co2_comparison.png")
    plot_int4_vs_int8_comparison(df, plots_dir / "int4_vs_int8.png")
    
    print("\nTodos los gráficos generados en: results/plots/")
    print("\nArchivos creados:")
    for f in plots_dir.glob("*.png"):
        print("   - %s" % f.name)


if __name__ == "__main__":
    generate_all_plots()
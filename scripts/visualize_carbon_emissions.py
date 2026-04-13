#!/usr/bin/env python3
"""
Carbon Footprint Visualization for AI Models

This script reads carbon footprint comparison results and generates
multiple visualization charts for the Master's thesis.

Author: Andres Ruben Villamayor Ruiz Diaz
Date: 2026
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime


# PLOT STYLE CONFIGURATION
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


def load_comparison_data(csv_path: str) -> list:
    """Load data from carbon footprint comparison CSV."""
    if not Path(csv_path).exists():
        print(f"Error: File not found {csv_path}")
        return []
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_energy_by_model(data: list, output_path: str):
    """Bar chart: Energy consumption by model."""
    models = [d['Modelo'] for d in data]
    energy = [float(d['Energia_Wh']) for d in data]
    quant = [d['Cuantizacion'] for d in data]
    
    labels = [f"{m}\n({q})" for m, q in zip(models, quant)]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, energy, color='#3498db', edgecolor='black', linewidth=1.2)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('AI Model', fontsize=11)
    plt.ylabel('Energy Consumption (Wh)', fontsize=11)
    plt.title('Energy Consumption by AI Model', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_carbon_intensity_by_country(data: list, output_path: str):
    """Horizontal bar chart: Carbon intensity by country."""
    if not data:
        return
    
    first_row = data[0]
    countries = []
    factors = []
    
    country_names = {
        'PY': 'Paraguay', 'AR': 'Argentina', 'BR-S': 'Brazil',
        'CL': 'Chile', 'UY': 'Uruguay', 'BO': 'Bolivia', 'CO': 'Colombia',
        'MX': 'Mexico', 'US': 'USA', 'CA': 'Canada',
        'ES': 'Spain', 'FR': 'France', 'DE': 'Germany',
        'IT': 'Italy', 'PT': 'Portugal', 'GB': 'UK',
        'NL': 'Netherlands', 'CN': 'China', 'IN': 'India',
        'JP': 'Japan', 'AU': 'Australia'
    }
    
    for key, value in first_row.items():
        if key.startswith('Factor_') and key.endswith('_g_kwh'):
            code = key.split('_')[1]
            name = country_names.get(code, code)
            countries.append(name)
            factors.append(float(value))
    
    order = sorted(range(len(factors)), key=lambda i: factors[i])
    countries = [countries[i] for i in order]
    factors = [factors[i] for i in order]
    
    colors = []
    for f in factors:
        if f < 100:
            colors.append('#27ae60')
        elif f < 300:
            colors.append('#f39c12')
        else:
            colors.append('#e74c3c')
    
    plt.figure(figsize=(12, 9))
    bars = plt.barh(countries, factors, color=colors, edgecolor='black', linewidth=0.8)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 8, bar.get_y() + bar.get_height()/2,
                f'{width:.0f}', ha='left', va='center', fontsize=8)
    
    plt.xlabel('Carbon Intensity (g CO2/kWh)', fontsize=11)
    plt.title('Carbon Intensity by Country', fontsize=13, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    from matplotlib.patches import Patch
    legend = [
        Patch(color='#27ae60', label='Very Clean (<100)'),
        Patch(color='#f39c12', label='Moderate (100-300)'),
        Patch(color='#e74c3c', label='High Impact (>300)')
    ]
    plt.legend(handles=legend, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_co2_comparison_grouped(data: list, output_path: str):
    """Grouped bar chart: CO2 emissions by model for key countries."""
    key_countries = [
        ('PY', 'Paraguay'),
        ('ES', 'Spain'),
        ('FR', 'France'),
        ('DE', 'Germany'),
        ('US', 'USA'),
        ('CN', 'China')
    ]
    
    models = [d['Modelo'] for d in data]
    x = np.arange(len(models))
    bar_width = 0.13
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#27ae60', '#e74c3c', '#3498db', '#e67e22', '#9b59b6', '#1abc9c']
    
    for i, (code, name) in enumerate(key_countries):
        co2_values = []
        for d in data:
            key = f"CO2_{code}_g"
            co2_values.append(float(d.get(key, 0)) * 1000)
        
        ax.bar(
            x + i * bar_width,
            co2_values,
            width=bar_width,
            label=name,
            color=colors[i],
            edgecolor='black',
            linewidth=0.8
        )
    
    ax.set_xlabel('AI Model', fontsize=11)
    ax.set_ylabel('CO2 Emissions (mg)', fontsize=11)
    ax.set_title('CO2 Emissions by Model and Country', fontsize=13, fontweight='bold')
    ax.set_xticks(x + bar_width * 2.5)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(title='Country', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_py_vs_es_comparison(data: list, output_path: str):
    """Comparison chart: Paraguay vs Spain for each model."""
    models = [d['Modelo'] for d in data]
    
    co2_py = [float(d.get('CO2_PY_g', 0)) * 1000 for d in data]
    co2_es = [float(d.get('CO2_ES_g', 0)) * 1000 for d in data]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(
        x - width/2,
        co2_py,
        width=width,
        label='Paraguay',
        color='#27ae60',
        edgecolor='black',
        linewidth=1.2
    )
    
    bars2 = ax.bar(
        x + width/2,
        co2_es,
        width=width,
        label='Spain',
        color='#e74c3c',
        edgecolor='black',
        linewidth=1.2
    )
    
    ax.set_xlabel('AI Model', fontsize=11)
    ax.set_ylabel('CO2 Emissions (mg)', fontsize=11)
    ax.set_title('CO2 Emissions Comparison: Paraguay vs Spain', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    for i, (py, es) in enumerate(zip(co2_py, co2_es)):
        if es > 0:
            savings = ((es - py) / es) * 100
            ax.text(i, max(py, es) + 0.3, f'-{savings:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2c3e50')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_energy_pie_chart(data: list, output_path: str):
    """Pie chart: Total energy distribution by model."""
    models = [d['Modelo'] for d in data]
    energy = [float(d['Energia_Wh']) for d in data]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    plt.figure(figsize=(9, 9))
    wedges, texts, autotexts = plt.pie(
        energy,
        labels=models,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 9}
    )
    
    plt.title('Energy Distribution by Model', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_real_time_carbon_intensity(data: list, output_path: str):
    """Line chart: Real-time carbon intensity for all countries."""
    if not data:
        return
    
    first_row = data[0]
    countries = []
    factors = []
    
    country_names = {
        'PY': 'Paraguay', 'AR': 'Argentina', 'BR-S': 'Brazil',
        'CL': 'Chile', 'UY': 'Uruguay', 'BO': 'Bolivia', 'CO': 'Colombia',
        'MX': 'Mexico', 'US': 'USA', 'CA': 'Canada',
        'ES': 'Spain', 'FR': 'France', 'DE': 'Germany',
        'IT': 'Italy', 'PT': 'Portugal', 'GB': 'UK',
        'NL': 'Netherlands', 'CN': 'China', 'IN': 'India',
        'JP': 'Japan', 'AU': 'Australia'
    }
    
    for key, value in first_row.items():
        if key.startswith('Factor_') and key.endswith('_g_kwh'):
            code = key.split('_')[1]
            name = country_names.get(code, code)
            countries.append(name)
            factors.append(float(value))
    
    order = sorted(range(len(factors)), key=lambda i: factors[i])
    countries = [countries[i] for i in order]
    factors = [factors[i] for i in order]
    
    plt.figure(figsize=(14, 6))
    
    plt.plot(range(len(countries)), factors, marker='o', linewidth=2, 
             markersize=6, color='#3498db', markerfacecolor='#e74c3c')
    
    plt.fill_between(range(len(countries)), factors, alpha=0.3, color='#3498db')
    
    for i, (country, factor) in enumerate(zip(countries, factors)):
        plt.annotate(f'{factor:.0f}', (i, factor), textcoords="offset points",
                    xytext=(0,5), ha='center', fontsize=7)
    
    plt.xlabel('Country', fontsize=11)
    plt.ylabel('Carbon Intensity (g CO2/kWh)', fontsize=11)
    plt.title('Real-Time Carbon Intensity Across Countries', fontsize=13, fontweight='bold')
    plt.xticks(range(len(countries)), countries, rotation=90, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Main function."""
    print("\n" + "="*80)
    print("GENERATING CARBON FOOTPRINT VISUALIZATIONS")
    print("="*80)
    
    data = load_comparison_data('results/comparacion_huella_carbono.csv')
    if not data:
        print("No data to visualize. Run compare_carbon_footprint.py first.")
        return
    
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating visualizations...")
    
    print("\n1. Energy by model...")
    plot_energy_by_model(data, 'results/figures/01_energy_by_model.png')
    
    print("2. Carbon intensity by country...")
    plot_carbon_intensity_by_country(data, 'results/figures/02_carbon_intensity_by_country.png')
    
    print("3. CO2 grouped comparison...")
    plot_co2_comparison_grouped(data, 'results/figures/03_co2_grouped_comparison.png')
    
    print("4. Paraguay vs Spain comparison...")
    plot_py_vs_es_comparison(data, 'results/figures/04_paraguay_vs_spain.png')
    
    print("5. Energy distribution (pie chart)...")
    plot_energy_pie_chart(data, 'results/figures/05_energy_pie_chart.png')
    
    print("6. Real-time carbon intensity line chart...")
    plot_real_time_carbon_intensity(data, 'results/figures/06_realtime_carbon_intensity.png')
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
    print("="*80)
    print(f"\nFigures saved in: results/figures/")
    print(f"Total charts: 6")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nReady to include in your thesis!")


if __name__ == '__main__':
    main()
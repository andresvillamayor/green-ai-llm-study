"""GREEN-IA — Funciones de graficación y estadísticas extraídas de analisis_categoria."""

import math
from pathlib import Path

ROOT = Path(__file__).parent.parent

COLORES_BARRAS = ["#F5C518", "#7EC86A", "#4A90D9", "#E34948", "#EB6834"]
COLORES_CATEGORIAS = {"writing": "#F5C518", "reasoning": "#4A90D9", "coding": "#7EC86A"}


def calcular_estadisticas(valores: list) -> dict:
    """Calcula media, desviacion estandar, coeficiente de variacion e IC95."""
    valores = [v for v in valores if v == v]  # filtra NaN (v==v es False solo para NaN)
    n = len(valores)
    if n == 0:
        return {"media": 0.0, "std": 0.0, "cv_pct": 0.0, "ic95": 0.0, "n": 0}
    media = sum(valores) / n
    std = math.sqrt(sum((v - media) ** 2 for v in valores) / max(n - 1, 1))
    cv = (std / media * 100) if media > 0 else 0.0
    ic95 = 1.96 * std / math.sqrt(n)
    return {"media": media, "std": std, "cv_pct": cv, "ic95": ic95, "n": n}


def generar_plots(directorio: Path, categoria: str, datos_mwh: dict, etiquetas: dict,
                  datos_cpu: dict, datos_gpu: dict, datos_ram: dict) -> None:
    """Genera boxplot, barras con std y desglose CPU/GPU/RAM, guardados como PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  AVISO: matplotlib no instalado — plots omitidos. pip install matplotlib")
        return
    numeros_prompt = sorted(datos_mwh.keys())
    etiquetas_eje = [etiquetas[p] for p in numeros_prompt]
    # Plot 1: boxplot de repeticiones por prompt
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig1.patch.set_facecolor("white")
    ax1.set_facecolor("white")
    bp = ax1.boxplot(
        [datos_mwh[p] for p in numeros_prompt],
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"color": "#555555", "linewidth": 1.0},
        capprops={"color": "#555555", "linewidth": 1.0},
        flierprops={"marker": "o", "markersize": 4, "markerfacecolor": "#aaaaaa", "linestyle": "none"},
    )
    for caja, color in zip(bp["boxes"], COLORES_BARRAS):
        caja.set_facecolor(color)
        caja.set_alpha(0.85)
        caja.set_linewidth(0)
    ax1.set_xticks(range(1, len(numeros_prompt) + 1))
    ax1.set_xticklabels(etiquetas_eje, fontsize=10)
    ax1.set_ylabel("Energía (mWh)", fontsize=10)
    ax1.set_title(f"Categoría {categoria.capitalize()} — distribución por repetición", fontsize=12, fontweight="semibold", pad=10)
    ax1.grid(axis="y", linewidth=0.5, color="#e0e0e0", zorder=0)
    ax1.grid(axis="x", visible=False)
    ax1.set_axisbelow(True)
    for borde in ("top", "right"):
        ax1.spines[borde].set_visible(False)
    for borde in ("left", "bottom"):
        ax1.spines[borde].set_color("#cccccc")
        ax1.spines[borde].set_linewidth(0.8)
    ruta_plot1 = directorio / "boxplot_repeticiones.png"
    fig1.savefig(ruta_plot1, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig1)
    try:
        print(f"  Plot 1: {ruta_plot1.relative_to(ROOT)}")
    except ValueError:
        print(f"  Plot 1: {ruta_plot1}")
    # Plot 2: barras con desviacion estandar
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig2.patch.set_facecolor("white")
    ax2.set_facecolor("white")
    posiciones = list(range(len(numeros_prompt)))
    medias = [calcular_estadisticas(datos_mwh[p])["media"] for p in numeros_prompt]
    stds   = [calcular_estadisticas(datos_mwh[p])["std"]   for p in numeros_prompt]
    ax2.bar(posiciones, medias, width=0.55, color=COLORES_BARRAS, linewidth=0, zorder=3)
    ax2.errorbar(posiciones, medias, yerr=stds, fmt="none", color="black",
                 linewidth=1.5, capsize=5, capthick=1.5, zorder=4)
    # Valor encima de cada barra: "media ± std mWh"
    max_y = max(m + s for m, s in zip(medias, stds)) if medias else 1.0
    desplazamiento = max_y * 0.02
    for xi, (media, std) in enumerate(zip(medias, stds)):
        ax2.text(xi, media + std + desplazamiento, f"{media:.4f} ± {std:.4f} mWh",
                 ha="center", va="bottom", fontsize=8.5, color="#1a1a1a", zorder=5)
    ax2.set_xticks(posiciones)
    ax2.set_xticklabels(etiquetas_eje, fontsize=10)
    ax2.set_ylabel("Energía (mWh)", fontsize=10)
    ax2.set_ylim(0, max_y * 1.28)
    ax2.set_title(f"Categoría {categoria.capitalize()}", fontsize=13, fontweight="semibold", pad=10)
    ax2.grid(axis="y", linewidth=0.5, color="#e0e0e0", zorder=0)
    ax2.grid(axis="x", visible=False)
    ax2.set_axisbelow(True)
    for borde in ("top", "right"):
        ax2.spines[borde].set_visible(False)
    for borde in ("left", "bottom"):
        ax2.spines[borde].set_color("#cccccc")
        ax2.spines[borde].set_linewidth(0.8)
    ruta_plot2 = directorio / "barras_con_std.png"
    fig2.savefig(ruta_plot2, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    try:
        print(f"  Plot 2: {ruta_plot2.relative_to(ROOT)}")
    except ValueError:
        print(f"  Plot 2: {ruta_plot2}")
    # Plot 3: desglose energetico CPU / GPU / RAM por prompt
    COLOR_CPU = "#1B3A6B"  # azul oscuro
    COLOR_GPU = "#5BA85A"  # verde
    COLOR_RAM = "#9E9E9E"  # gris
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    fig3.patch.set_facecolor("white")
    ax3.set_facecolor("white")
    posiciones3 = list(range(len(numeros_prompt)))
    medias_cpu = [calcular_estadisticas(datos_cpu.get(p, [0]))["media"] for p in numeros_prompt]
    medias_gpu = [calcular_estadisticas(datos_gpu.get(p, [0]))["media"] for p in numeros_prompt]
    medias_ram = [calcular_estadisticas(datos_ram.get(p, [0]))["media"] for p in numeros_prompt]
    ax3.bar(posiciones3, medias_cpu, width=0.55, color=COLOR_CPU, label="CPU", linewidth=0, zorder=3)
    ax3.bar(posiciones3, medias_gpu, width=0.55, color=COLOR_GPU, label="GPU", linewidth=0,
            bottom=medias_cpu, zorder=3)
    ax3.bar(posiciones3, medias_ram, width=0.55, color=COLOR_RAM, label="RAM", linewidth=0,
            bottom=[c + g for c, g in zip(medias_cpu, medias_gpu)], zorder=3)
    for xi, (cpu, gpu, ram) in enumerate(zip(medias_cpu, medias_gpu, medias_ram)):
        umbral_barra = (cpu + gpu + ram) * 0.05
        if cpu > umbral_barra:
            ax3.text(xi, cpu / 2, f"{cpu:.4f}", ha="center", va="center",
                     fontsize=7.5, color="white", fontweight="bold", zorder=5)
        if gpu > umbral_barra:
            ax3.text(xi, cpu + gpu / 2, f"{gpu:.4f}", ha="center", va="center",
                     fontsize=7.5, color="white", fontweight="bold", zorder=5)
        if ram > umbral_barra:
            ax3.text(xi, cpu + gpu + ram / 2, f"{ram:.4f}", ha="center", va="center",
                     fontsize=7.5, color="#1a1a1a", fontweight="bold", zorder=5)
    ax3.set_xticks(posiciones3)
    ax3.set_xticklabels(etiquetas_eje, fontsize=10)
    ax3.set_ylabel("Energía (mWh)", fontsize=10)
    ax3.set_title(f"Desglose energético por componente — Categoría {categoria.capitalize()}",
                  fontsize=12, fontweight="semibold", pad=10)
    ax3.legend(loc="upper right", frameon=False, fontsize=10)
    ax3.grid(axis="y", linewidth=0.5, color="#e0e0e0", zorder=0)
    ax3.grid(axis="x", visible=False)
    ax3.set_axisbelow(True)
    for borde in ("top", "right"):
        ax3.spines[borde].set_visible(False)
    for borde in ("left", "bottom"):
        ax3.spines[borde].set_color("#cccccc")
        ax3.spines[borde].set_linewidth(0.8)
    ruta_plot3 = directorio / "desglose_cpu_gpu_ram.png"
    fig3.savefig(ruta_plot3, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig3)
    try:
        print(f"  Plot 3: {ruta_plot3.relative_to(ROOT)}")
    except ValueError:
        print(f"  Plot 3: {ruta_plot3}")


def generar_plot_comparacion(datos_resumen: dict) -> None:
    """Genera barras verticales comparando energia media de las 3 categorias."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  AVISO: matplotlib no instalado — plot comparativo omitido.")
        return
    categorias = list(datos_resumen.keys())
    medias = [datos_resumen[c]["media_global"] for c in categorias]
    stds = [datos_resumen[c]["std_global"] for c in categorias]
    colores = [COLORES_CATEGORIAS.get(c, "#999999") for c in categorias]
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    posiciones = list(range(len(categorias)))
    ax.bar(posiciones, medias, width=0.55, color=colores, linewidth=0, zorder=3)
    ax.errorbar(posiciones, medias, yerr=stds, fmt="none", color="black",
                linewidth=1.5, capsize=5, capthick=1.5, zorder=4)
    max_y = max(m + s for m, s in zip(medias, stds)) if medias else 1.0
    desplazamiento = max_y * 0.02
    for xi, (media, std) in enumerate(zip(medias, stds)):
        ax.text(xi, media + std + desplazamiento, f"{media:.4f} mWh",
                ha="center", va="bottom", fontsize=9, color="#1a1a1a", zorder=5)
    ax.set_xticks(posiciones)
    ax.set_xticklabels([c.capitalize() for c in categorias], fontsize=11)
    ax.set_ylabel("Energía media (mWh)", fontsize=10)
    ax.set_ylim(0, max_y * 1.28)
    ax.set_title("Comparacion de categorias — Energia media", fontsize=13, fontweight="semibold", pad=10)
    ax.grid(axis="y", linewidth=0.5, color="#e0e0e0", zorder=0)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    for borde in ("top", "right"):
        ax.spines[borde].set_visible(False)
    for borde in ("left", "bottom"):
        ax.spines[borde].set_color("#cccccc")
        ax.spines[borde].set_linewidth(0.8)
    ruta_plot = ROOT / "results" / "analisis_categoria" / "comparacion_categorias.png"
    fig.savefig(ruta_plot, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Plot comparacion: {ruta_plot.relative_to(ROOT)}")

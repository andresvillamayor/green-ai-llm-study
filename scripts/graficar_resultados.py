# graficar_resultados.py
# Genera las graficas para la tesis a partir del CSV del experimento
# Proyecto GREEN-IA — Andres Villamayor

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# rutas relativas al script
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results" / "measurements"
PLOTS_DIR    = PROJECT_ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi"       : 150,
    "font.size"        : 10,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "grid.linestyle"   : "--",
})

# colores por configuracion
COLORES = {
    "Llama-2-7B Q4 CPU": "#42A5F5",
    "Llama-2-7B Q4 GPU": "#1565C0",
    "Llama-2-7B Q8 CPU": "#FFCA28",
    "Llama-2-7B Q8 GPU": "#F57F17",
    "Qwen2.5-7B Q4 CPU": "#66BB6A",
    "Qwen2.5-7B Q4 GPU": "#1B5E20",
    "Qwen2.5-7B Q8 CPU": "#EF5350",
    "Qwen2.5-7B Q8 GPU": "#B71C1C",
}


# --------------------------------------------------------------------------
# Carga de datos
# --------------------------------------------------------------------------

def cargar_csv():
    # busca el CSV mas reciente automaticamente
    archivos = sorted(RESULTS_DIR.glob("experimento_codecarbon_*.csv"))
    if not archivos:
        raise FileNotFoundError(f"No hay CSVs en {RESULTS_DIR}")
    ruta = archivos[-1]
    print(f"  Leyendo: {ruta.name}")
    return pd.read_csv(ruta)


def limpiar_datos(df):
    # convierte columnas numericas (hay celdas vacias en algunos experimentos)
    for col in ["gpu_power_w", "gpu_energy_wh", "emissions_mg_co2", "total_energy_wh"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["quantization"] = df["quantization"].str.upper()
    df["device"]       = df["device"].str.upper()
    df["model_short"]  = df["model"].map({
        "llama-2-7b" : "Llama-2-7B",
        "qwen2.5-7b" : "Qwen2.5-7B",
    })
    df["config"] = df["model_short"] + " " + df["quantization"] + " " + df["device"]

    # saco outliers con IQR x3 por configuracion
    df["outlier"] = False
    for (mod, q, dev), grp in df.groupby(["model", "quantization", "device"]):
        q1, q3 = grp["inference_time_s"].quantile([0.25, 0.75])
        iqr    = q3 - q1
        mask   = (grp["inference_time_s"] < q1 - 3*iqr) | (grp["inference_time_s"] > q3 + 3*iqr)
        df.loc[mask.index, "outlier"] = mask.values

    n_out = df["outlier"].sum()
    if n_out:
        print(f"  {n_out} outliers excluidos del analisis")

    return df[~df["outlier"]].copy()


def estadisticas(df):
    # promedios y desvio por configuracion
    return (
        df.groupby(["model_short", "quantization", "device", "config"])
        .agg(
            n             = ("inference_time_s",  "count"),
            tiempo_media  = ("inference_time_s",  "mean"),
            tiempo_std    = ("inference_time_s",  "std"),
            tps_media     = ("tokens_per_second", "mean"),
            tps_std       = ("tokens_per_second", "std"),
            energia_media = ("total_energy_wh",   "mean"),
            energia_std   = ("total_energy_wh",   "std"),
            co2_media     = ("emissions_mg_co2",  "mean"),
            co2_std       = ("emissions_mg_co2",  "std"),
            e_cpu_media   = ("cpu_energy_wh",     "mean"),
            e_gpu_media   = ("gpu_energy_wh",     "mean"),
            e_ram_media   = ("ram_energy_wh",     "mean"),
        )
        .reset_index()
    )


# --------------------------------------------------------------------------
# Graficas
# --------------------------------------------------------------------------

def fig1_energia_apilada(stats):
    """Consumo energetico desglosado por componente (CPU, GPU, RAM)"""
    fig, ax = plt.subplots(figsize=(13, 5))

    x      = np.arange(len(stats))
    ancho  = 0.6
    labels = [c.replace(" ", "\n") for c in stats["config"]]
    colores = [COLORES.get(c, "#999") for c in stats["config"]]

    # barras apiladas: RAM abajo, CPU en el medio, GPU arriba
    e_ram = stats["e_ram_media"].fillna(0) * 1000  # a mWh
    e_cpu = stats["e_cpu_media"].fillna(0) * 1000
    e_gpu = stats["e_gpu_media"].fillna(0) * 1000

    ax.bar(x, e_ram, ancho, label="RAM",           color="#FF8C42", edgecolor="white")
    ax.bar(x, e_cpu, ancho, bottom=e_ram,          label="CPU",     color="#DC3545", edgecolor="white")
    ax.bar(x, e_gpu, ancho, bottom=e_ram + e_cpu,  label="GPU",     color="#4A90E2", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Energia consumida (mWh)")
    ax.set_title("Consumo Energetico por Componente\nLLM Inference — GREEN-IA", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    ruta = PLOTS_DIR / "fig1_energia_desglosada.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {ruta.name}")


def fig2_velocidad(stats):
    """Tokens por segundo por configuracion"""
    fig, ax = plt.subplots(figsize=(13, 5))

    x      = np.arange(len(stats))
    colores = [COLORES.get(c, "#999") for c in stats["config"]]
    labels  = [c.replace(" ", "\n") for c in stats["config"]]

    ax.bar(x, stats["tps_media"], yerr=stats["tps_std"].fillna(0),
           color=colores, edgecolor="white", capsize=4,
           error_kw={"linewidth": 1.5, "ecolor": "#333"})

    # valor encima de cada barra
    for i, (v, e) in enumerate(zip(stats["tps_media"], stats["tps_std"].fillna(0))):
        ax.text(i, v + e + 0.3, f"{v:.1f}", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Tokens por segundo (tok/s)")
    ax.set_title("Velocidad de Inferencia por Configuracion\n(media ± desvio estandar)", fontweight="bold")

    plt.tight_layout()
    ruta = PLOTS_DIR / "fig2_velocidad_inferencia.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {ruta.name}")


def fig3_co2(stats):
    """Emisiones CO2 por configuracion"""
    fig, ax = plt.subplots(figsize=(13, 5))

    x       = np.arange(len(stats))
    colores = [COLORES.get(c, "#999") for c in stats["config"]]
    labels  = [c.replace(" ", "\n") for c in stats["config"]]
    co2_ug  = stats["co2_media"] * 1000  # mg a µg
    co2_std = stats["co2_std"].fillna(0) * 1000

    ax.bar(x, co2_ug, yerr=co2_std, color=colores, edgecolor="white",
           capsize=4, error_kw={"linewidth": 1.5, "ecolor": "#333"})

    for i, (v, e) in enumerate(zip(co2_ug, co2_std)):
        ax.text(i, v + e + 0.01, f"{v:.2f}", ha="center", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Emisiones CO₂ (µg CO₂eq)")
    ax.set_title("Emisiones de CO₂ por Configuracion\n(Paraguay: 26 gCO₂/kWh)", fontweight="bold")

    plt.tight_layout()
    ruta = PLOTS_DIR / "fig3_emisiones_co2.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {ruta.name}")


def fig4_boxplot_tiempos(df):
    """Distribucion de tiempos de inferencia por modelo"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    orden = [
        "Q4 CPU", "Q4 GPU", "Q8 CPU", "Q8 GPU"
    ]

    for ax, modelo in zip(axes, ["Llama-2-7B", "Qwen2.5-7B"]):
        sub  = df[df["model_short"] == modelo]
        cfgs = sub["config"].unique()

        # solo las configs que existen en los datos
        cfgs_orden = [f"{modelo} {s}" for s in orden if f"{modelo} {s}" in cfgs]
        datos      = [sub[sub["config"] == c]["inference_time_s"].values for c in cfgs_orden]
        colores    = [COLORES.get(c, "#999") for c in cfgs_orden]

        bp = ax.boxplot(datos, patch_artist=True,
                        medianprops={"color": "black", "linewidth": 2},
                        whiskerprops={"linewidth": 1.5},
                        capprops={"linewidth": 1.5},
                        flierprops={"marker": "o", "markersize": 4, "alpha": 0.5})

        for patch, color in zip(bp["boxes"], colores):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        etiquetas = [c.replace(f"{modelo} ", "").replace(" ", "\n") for c in cfgs_orden]
        ax.set_xticklabels(etiquetas, fontsize=9)
        ax.set_ylabel("Tiempo de inferencia (s)")
        ax.set_title(modelo, fontsize=11, fontweight="bold")

    fig.suptitle("Distribucion de Tiempos de Inferencia (sin outliers)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    ruta = PLOTS_DIR / "fig4_boxplot_tiempos.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {ruta.name}")


def fig5_q4_vs_q8(stats):
    """Comparacion directa Q4 vs Q8 en velocidad y energia"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    metricas = [
        ("tps_media",     "tps_std",     "Velocidad (tok/s)"),
        ("energia_media", "energia_std", "Energia (mWh)"),
    ]

    C4L = "#42A5F5"; C8L = "#1565C0"
    C4Q = "#66BB6A"; C8Q = "#2E7D32"

    for ax, (met, std_col, titulo) in zip(axes, metricas):
        cur_x, ticks, tick_lbl = 0, [], []

        for modelo in ["Llama-2-7B", "Qwen2.5-7B"]:
            for dev in ["CPU", "GPU"]:
                r4 = stats[(stats["model_short"] == modelo) &
                            (stats["quantization"] == "Q4") &
                            (stats["device"] == dev)]
                r8 = stats[(stats["model_short"] == modelo) &
                            (stats["quantization"] == "Q8") &
                            (stats["device"] == dev)]
                if r4.empty or r8.empty:
                    continue

                v4, s4 = r4.iloc[0][met], r4.iloc[0][std_col]
                v8, s8 = r8.iloc[0][met], r8.iloc[0][std_col]

                # para energia paso a mWh
                if met == "energia_media":
                    v4, s4, v8, s8 = v4*1000, s4*1000, v8*1000, s8*1000

                c4 = C4L if modelo == "Llama-2-7B" else C4Q
                c8 = C8L if modelo == "Llama-2-7B" else C8Q

                ax.bar(cur_x,        v4, 0.4, yerr=s4, color=c4,
                       edgecolor="white", capsize=3, error_kw={"lw": 1.2})
                ax.bar(cur_x + 0.45, v8, 0.4, yerr=s8, color=c8,
                       edgecolor="white", capsize=3, error_kw={"lw": 1.2})

                ticks.append(cur_x + 0.225)
                tick_lbl.append(f"{modelo[:4]}\n{dev}")
                cur_x += 1.3

        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_lbl, fontsize=9)
        ax.set_title(titulo, fontweight="bold")

    leyenda = [
        mpatches.Patch(color=C4L, label="Llama-2 Q4"),
        mpatches.Patch(color=C8L, label="Llama-2 Q8"),
        mpatches.Patch(color=C4Q, label="Qwen2.5 Q4"),
        mpatches.Patch(color=C8Q, label="Qwen2.5 Q8"),
    ]
    fig.legend(handles=leyenda, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Comparacion Q4 vs Q8 por Modelo y Dispositivo",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    ruta = PLOTS_DIR / "fig5_q4_vs_q8.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {ruta.name}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("  GREEN-IA: Generando graficas")
    print("=" * 50)

    df    = cargar_csv()
    df    = limpiar_datos(df)
    stats = estadisticas(df)

    print(f"\n  {len(df)} mediciones limpias | {len(stats)} configuraciones\n")

    fig1_energia_apilada(stats)
    fig2_velocidad(stats)
    fig3_co2(stats)
    fig4_boxplot_tiempos(df)
    fig5_q4_vs_q8(stats)

    print(f"\n  Listo. Graficas en: {PLOTS_DIR}")
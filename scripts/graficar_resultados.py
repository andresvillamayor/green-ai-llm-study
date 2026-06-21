# graficar_resultados.py
# Genera las graficas para la tesis a partir del CSV del experimento
# Proyecto GREEN-IA — Andres Villamayor
#
# Los outliers se identifican con IQR x3 y se muestran en los graficos.
# No se excluye ninguna medicion. Los outliers extremos tienen una
# explicacion documentada en results/measurements/powermetrics_log.txt

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from scipy.stats import t as t_dist

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


def cargar_csv():
    archivos = sorted(RESULTS_DIR.glob("experimento_codecarbon_*.csv"))
    if not archivos:
        raise FileNotFoundError(f"no hay CSVs en {RESULTS_DIR}")
    ruta = archivos[-1]
    print(f"  leyendo: {ruta.name}")
    return pd.read_csv(ruta)


def identificar_outliers(df):
    # identifica los outliers con IQR x3 pero no los elimina
    # IQR x3 es un criterio conservador: solo marca valores verdaderamente
    # extremos, muy por encima o por debajo del rango normal
    # referencia: Tukey, J.W. (1977). Exploratory Data Analysis.
    print("\n" + "-" * 55)
    print("  identificacion de outliers (criterio IQR x3)")
    print("  criterio: Tukey (1977), factor 3 para valores extremos")
    print("-" * 55)

    df["outlier"] = False
    cpu_count = 0
    gpu_count = 0

    for (mod, q, dev), grp in df.groupby(["model", "quantization", "device"]):
        tiempos = grp["inference_time_s"].values
        q1 = np.percentile(tiempos, 25)
        q3 = np.percentile(tiempos, 75)
        iqr = q3 - q1
        lim_inf = q1 - 3 * iqr
        lim_sup = q3 + 3 * iqr

        mask = (
            (grp["inference_time_s"] < lim_inf) |
            (grp["inference_time_s"] > lim_sup)
        )
        df.loc[mask[mask].index, "outlier"] = True

        outliers = grp[mask]
        for _, row in outliers.iterrows():
            media = np.mean(tiempos)
            veces = row["inference_time_s"] / media
            print(f"\n  {mod} {q} {dev}")
            print(f"    rep {int(row['repetition'])} prompt {int(row['prompt_id'])}: "
                  f"{row['inference_time_s']:.2f} s  ({veces:.1f}x la media de {media:.1f} s)")

            # los tiempos extremos en CPU estan documentados en powermetrics_log.txt
            # el log muestra que en esas corridas los P-cores del M4 estaban al 0%
            # y el sistema corria en E-cores a 1080 MHz con 18 mW de consumo total
            # eso multiplica el tiempo de inferencia hasta 25 veces respecto al normal
            if row["inference_time_s"] > 100 and dev.upper() == "CPU":
                print(f"    causa documentada en powermetrics_log.txt:")
                print(f"    P-cores al 0% de actividad, CPU a 1080 MHz, consumo 18 mW")
                print(f"    el sistema operativo ejecuto la inferencia solo en E-cores")

            if dev.upper() == "CPU":
                cpu_count += 1
            else:
                gpu_count += 1

    total = df["outlier"].sum()
    print(f"\n  total identificados : {total} de {len(df)} mediciones "
          f"({total / len(df) * 100:.1f}%)")
    print(f"  en CPU              : {cpu_count}")
    print(f"  en GPU              : {gpu_count}")
    print(f"\n  estos valores se muestran en los graficos, no se eliminan")
    print(f"  la alta desviacion estandar en Llama CPU refleja la")
    print(f"  variabilidad real del sistema operativo bajo presion de memoria")
    print("-" * 55)

    return df


def preparar_datos(df):
    for col in ["gpu_power_w", "gpu_energy_wh", "emissions_mg_co2", "total_energy_wh"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["quantization"] = df["quantization"].str.upper()
    df["device"]       = df["device"].str.upper()
    df["model_short"]  = df["model"].map({
        "llama-2-7b" : "Llama-2-7B",
        "qwen2.5-7b" : "Qwen2.5-7B",
    })
    df["config"] = df["model_short"] + " " + df["quantization"] + " " + df["device"]

    # métricas normalizadas por token (con fallback para CSVs anteriores)
    if "energy_per_token" not in df.columns:
        tok = df["tokens_generated"].replace(0, np.nan)
        df["energy_per_token"] = df["total_energy_wh"] / tok
    tok_col = "tokens_output" if "tokens_output" in df.columns else "tokens_generated"
    df["co2_per_1k_tokens"] = df["emissions_mg_co2"] / df[tok_col].replace(0, np.nan) * 1000

    # identifica outliers pero usa todos los datos para graficar
    df = identificar_outliers(df)

    print(f"\n  {len(df)} mediciones totales | {df['outlier'].sum()} identificados como outliers")
    print(f"  todos los datos se incluyen en las graficas")
    return df


def estadisticas(df):
    # calcula estadisticas con todos los datos incluyendo outliers
    # esto refleja la variabilidad real del experimento
    agg = (
        df.groupby(["model_short", "quantization", "device", "config"])
        .agg(
            n              = ("inference_time_s",  "count"),
            tiempo_media   = ("inference_time_s",  "mean"),
            tiempo_std     = ("inference_time_s",  "std"),
            tiempo_median  = ("inference_time_s",  "median"),
            tiempo_iqr     = ("inference_time_s",  lambda x: x.quantile(0.75) - x.quantile(0.25)),
            tps_media      = ("tokens_per_second", "mean"),
            tps_std        = ("tokens_per_second", "std"),
            tps_median     = ("tokens_per_second", "median"),
            tps_iqr        = ("tokens_per_second", lambda x: x.quantile(0.75) - x.quantile(0.25)),
            energia_media  = ("total_energy_wh",   "mean"),
            energia_std    = ("total_energy_wh",   "std"),
            energia_median = ("total_energy_wh",   "median"),
            energia_iqr    = ("total_energy_wh",   lambda x: x.quantile(0.75) - x.quantile(0.25)),
            co2_media      = ("emissions_mg_co2",  "mean"),
            co2_std        = ("emissions_mg_co2",  "std"),
            co2_median     = ("emissions_mg_co2",  "median"),
            co2_iqr        = ("emissions_mg_co2",  lambda x: x.quantile(0.75) - x.quantile(0.25)),
            e_cpu_media    = ("cpu_energy_wh",     "mean"),
            e_gpu_media    = ("gpu_energy_wh",     "mean"),
            e_ram_media    = ("ram_energy_wh",     "mean"),
            ept_media      = ("energy_per_token",  "mean"),
            ept_std        = ("energy_per_token",  "std"),
            co2_1k_media   = ("co2_per_1k_tokens", "mean"),
            co2_1k_std     = ("co2_per_1k_tokens", "std"),
        )
        .reset_index()
    )

    # coeficiente de variacion (CV) en porcentaje: std / media * 100
    for base, med_col, std_col in [
        ("tiempo",   "tiempo_media",  "tiempo_std"),
        ("tps",      "tps_media",     "tps_std"),
        ("energia",  "energia_media", "energia_std"),
        ("co2",      "co2_media",     "co2_std"),
    ]:
        agg[f"{base}_cv"] = np.where(
            agg[med_col] != 0,
            agg[std_col].fillna(0) / agg[med_col] * 100,
            0.0,
        )

    # intervalo de confianza al 95% via distribucion t de Student
    # se usa t en lugar de z porque n=5 es una muestra pequeña
    def _ci95(row, std_col):
        n = row["n"]
        return t_dist.ppf(0.975, df=n - 1) * row[std_col] / np.sqrt(n) if n > 1 else 0.0

    for base, std_col in [
        ("tiempo",  "tiempo_std"),
        ("tps",     "tps_std"),
        ("energia", "energia_std"),
        ("co2",     "co2_std"),
    ]:
        agg[f"{base}_ci95"] = agg.apply(lambda r, c=std_col: _ci95(r, c), axis=1)

    return agg


def fig1_energia_apilada(stats, stats_clean=None):
    fig, ax = plt.subplots(figsize=(14, 5))

    x      = np.arange(len(stats))
    labels = [c.replace(" ", "\n") for c in stats["config"]]
    ancho  = 0.3 if stats_clean is not None else 0.6
    xf     = x - (ancho / 2 + 0.02) if stats_clean is not None else x

    e_ram = stats["e_ram_media"].fillna(0) * 1000
    e_cpu = stats["e_cpu_media"].fillna(0) * 1000
    e_gpu = stats["e_gpu_media"].fillna(0) * 1000
    e_std = stats["energia_std"].fillna(0) * 1000

    ax.bar(xf, e_ram, ancho, label="RAM", color="#FF8C42", edgecolor="white")
    ax.bar(xf, e_cpu, ancho, bottom=e_ram, label="CPU", color="#DC3545", edgecolor="white")
    # yerr en el segmento superior ubica las barras de error en la cima del stack total
    ax.bar(xf, e_gpu, ancho, bottom=e_ram + e_cpu, label="GPU", color="#4A90E2",
           edgecolor="white", yerr=e_std, capsize=4,
           error_kw={"linewidth": 1.5, "ecolor": "#333"})

    if stats_clean is not None:
        xk    = x + (ancho / 2 + 0.02)
        r_ram = stats_clean["e_ram_media"].fillna(0) * 1000
        r_cpu = stats_clean["e_cpu_media"].fillna(0) * 1000
        r_gpu = stats_clean["e_gpu_media"].fillna(0) * 1000
        r_std = stats_clean["energia_std"].fillna(0) * 1000

        ax.bar(xk, r_ram, ancho, color="#FF8C42", edgecolor="white", hatch="///", alpha=0.7)
        ax.bar(xk, r_cpu, ancho, bottom=r_ram, color="#DC3545", edgecolor="white", hatch="///", alpha=0.7)
        ax.bar(xk, r_gpu, ancho, bottom=r_ram + r_cpu, color="#4A90E2", edgecolor="white",
               hatch="///", alpha=0.7, yerr=r_std, capsize=4,
               error_kw={"linewidth": 1.5, "ecolor": "#555"})

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Energia consumida (mWh)")

    if stats_clean is not None:
        handles, _ = ax.get_legend_handles_labels()
        handles += [
            mpatches.Patch(facecolor="gray", alpha=0.9, label="Con outliers"),
            mpatches.Patch(facecolor="gray", hatch="///", alpha=0.7, label="Sin outliers"),
        ]
        ax.legend(handles=handles, loc="upper left", fontsize=9)
        ax.set_title("Consumo Energetico por Componente\n"
                     "con y sin outliers extremos (IQR x3)", fontweight="bold")
    else:
        ax.legend(loc="upper left", fontsize=9)
        ax.set_title("Consumo Energetico por Componente\nLLM Inference — GREEN-IA", fontweight="bold")

    ax.annotate(
        "* Energía GPU estimada por CodeCarbon. "
        "Apple Silicon no expone API directa de consumo GPU.",
        xy=(0, -0.12), xycoords='axes fraction',
        fontsize=7, color='gray', style='italic'
    )
    plt.tight_layout()
    ruta = PLOTS_DIR / "fig1_energia_desglosada.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  guardado: {ruta.name}")


def fig2_velocidad(stats, stats_clean=None):
    fig, ax = plt.subplots(figsize=(13, 5))

    x       = np.arange(len(stats))
    colores = [COLORES.get(c, "#999") for c in stats["config"]]
    labels  = [c.replace(" ", "\n") for c in stats["config"]]
    ancho   = 0.38 if stats_clean is not None else 0.6
    xf      = x - ancho / 2 if stats_clean is not None else x

    ax.bar(xf, stats["tps_media"], ancho,
           yerr=stats["tps_std"].fillna(0), color=colores, edgecolor="white",
           capsize=4, error_kw={"linewidth": 1.5, "ecolor": "#333"},
           label="Con outliers" if stats_clean is not None else None)

    if stats_clean is not None:
        ax.bar(x + ancho / 2, stats_clean["tps_media"], ancho,
               yerr=stats_clean["tps_std"].fillna(0), color=colores,
               edgecolor="white", capsize=4, hatch="///", alpha=0.7,
               error_kw={"linewidth": 1.5, "ecolor": "#555"},
               label="Sin outliers")
        ax.legend(fontsize=9)
    else:
        for i, (v, e) in enumerate(zip(stats["tps_media"], stats["tps_std"].fillna(0))):
            ax.text(i, v + e + 0.3, f"{v:.1f}", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Tokens por segundo (tok/s)")
    titulo = ("Velocidad de Inferencia por Configuracion\n"
              "(media +/- desvio estandar, con y sin outliers)"
              if stats_clean is not None
              else "Velocidad de Inferencia por Configuracion\n"
                   "(media +/- desvio estandar, datos completos)")
    ax.set_title(titulo, fontweight="bold")

    plt.tight_layout()
    ruta = PLOTS_DIR / "fig2_velocidad_inferencia.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  guardado: {ruta.name}")


def fig3_co2(stats, stats_clean=None):
    fig, ax = plt.subplots(figsize=(13, 5))

    x       = np.arange(len(stats))
    colores = [COLORES.get(c, "#999") for c in stats["config"]]
    labels  = [c.replace(" ", "\n") for c in stats["config"]]
    ancho   = 0.38 if stats_clean is not None else 0.6
    xf      = x - ancho / 2 if stats_clean is not None else x
    co2_ug  = stats["co2_media"] * 1000
    co2_std = stats["co2_std"].fillna(0) * 1000

    ax.bar(xf, co2_ug, ancho, yerr=co2_std, color=colores, edgecolor="white",
           capsize=4, error_kw={"linewidth": 1.5, "ecolor": "#333"},
           label="Con outliers" if stats_clean is not None else None)

    if stats_clean is not None:
        co2_ug_c  = stats_clean["co2_media"] * 1000
        co2_std_c = stats_clean["co2_std"].fillna(0) * 1000
        ax.bar(x + ancho / 2, co2_ug_c, ancho, yerr=co2_std_c, color=colores,
               edgecolor="white", capsize=4, hatch="///", alpha=0.7,
               error_kw={"linewidth": 1.5, "ecolor": "#555"},
               label="Sin outliers")
        ax.legend(fontsize=9)
    else:
        for i, (v, e) in enumerate(zip(co2_ug, co2_std)):
            ax.text(i, v + e + 0.01, f"{v:.2f}", ha="center", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Emisiones CO2 (µg CO2eq)")
    titulo = ("Emisiones de CO2 por Configuracion\n"
              "(Paraguay: 26 gCO2/kWh, con y sin outliers)"
              if stats_clean is not None
              else "Emisiones de CO2 por Configuracion\n"
                   "(Paraguay: 26 gCO2/kWh, datos completos)")
    ax.set_title(titulo, fontweight="bold")

    ax.annotate(
        "* Energía GPU estimada por CodeCarbon. "
        "Apple Silicon no expone API directa de consumo GPU.",
        xy=(0, -0.12), xycoords='axes fraction',
        fontsize=7, color='gray', style='italic'
    )
    plt.tight_layout()
    ruta = PLOTS_DIR / "fig3_emisiones_co2.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  guardado: {ruta.name}")


def fig4_boxplot_tiempos(df):
    # muestra todos los datos incluyendo outliers
    # los outliers extremos aparecen como puntos rojos
    # con anotacion que referencia el powermetrics_log.txt
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    orden = ["Q4 CPU", "Q4 GPU", "Q8 CPU", "Q8 GPU"]

    for ax, modelo in zip(axes, ["Llama-2-7B", "Qwen2.5-7B"]):
        sub  = df[df["model_short"] == modelo]
        cfgs = sub["config"].unique()

        cfgs_orden = [f"{modelo} {s}" for s in orden if f"{modelo} {s}" in cfgs]
        datos      = [sub[sub["config"] == c]["inference_time_s"].values for c in cfgs_orden]
        colores    = [COLORES.get(c, "#999") for c in cfgs_orden]

        # outliers en rojo para que sean visibles
        bp = ax.boxplot(datos, patch_artist=True,
                        medianprops={"color": "black", "linewidth": 2},
                        whiskerprops={"linewidth": 1.5},
                        capprops={"linewidth": 1.5},
                        flierprops={
                            "marker"          : "o",
                            "markersize"      : 6,
                            "alpha"           : 0.8,
                            "markerfacecolor" : "red",
                            "markeredgecolor" : "red",
                        })

        for patch, color in zip(bp["boxes"], colores):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        # anota los outliers extremos con su valor real y referencia al powermetrics
        for i, (cfg, data) in enumerate(zip(cfgs_orden, datos), start=1):
            extremos = [v for v in data if v > 100]
            for val in extremos:
                ax.annotate(
                    f"{val:.0f} s\n(ver powermetrics_log.txt)",
                    xy=(i, val),
                    xytext=(i + 0.3, val * 0.85),
                    fontsize=7,
                    color="red",
                    arrowprops={"arrowstyle": "->", "color": "red", "lw": 1},
                )

        etiquetas = [c.replace(f"{modelo} ", "").replace(" ", "\n") for c in cfgs_orden]
        ax.set_xticklabels(etiquetas, fontsize=9)
        ax.set_ylabel("Tiempo de inferencia (s)")
        ax.set_title(modelo, fontsize=11, fontweight="bold")

    # nota explicativa de los puntos rojos
    fig.text(0.5, 0.01,
             "Puntos rojos: outliers identificados con IQR x3 (Tukey, 1977). "
             "Valores extremos en CPU documentados en powermetrics_log.txt",
             ha="center", fontsize=8, style="italic", color="#555")

    fig.suptitle("Distribucion de Tiempos de Inferencia\n(datos completos, outliers marcados en rojo)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    ruta = PLOTS_DIR / "fig4_boxplot_tiempos.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  guardado: {ruta.name}")


def fig5_q4_vs_q8(stats, stats_clean=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metricas = [
        ("tps_media",     "tps_std",     "Velocidad (tok/s)"),
        ("energia_media", "energia_std", "Energia (mWh)"),
    ]

    C4L = "#42A5F5"; C8L = "#1565C0"
    C4Q = "#66BB6A"; C8Q = "#2E7D32"

    # con outliers: barras llenas; sin outliers: barras rayadas mas angostas
    ancho_f = 0.2 if stats_clean is not None else 0.4
    ancho_k = 0.18

    for ax, (met, std_col, titulo) in zip(axes, metricas):
        cur_x, ticks, tick_lbl = 0, [], []

        for modelo in ["Llama-2-7B", "Qwen2.5-7B"]:
            for dev in ["CPU", "GPU"]:
                def _get(s):
                    r4 = s[(s["model_short"] == modelo) & (s["quantization"] == "Q4") & (s["device"] == dev)]
                    r8 = s[(s["model_short"] == modelo) & (s["quantization"] == "Q8") & (s["device"] == dev)]
                    return r4, r8

                r4, r8 = _get(stats)
                if r4.empty or r8.empty:
                    continue

                v4, s4 = r4.iloc[0][met], r4.iloc[0][std_col]
                v8, s8 = r8.iloc[0][met], r8.iloc[0][std_col]
                if met == "energia_media":
                    v4, s4, v8, s8 = v4*1000, s4*1000, v8*1000, s8*1000

                c4 = C4L if modelo == "Llama-2-7B" else C4Q
                c8 = C8L if modelo == "Llama-2-7B" else C8Q

                if stats_clean is not None:
                    r4c, r8c = _get(stats_clean)
                    v4c = r4c.iloc[0][met] if not r4c.empty else v4
                    s4c = r4c.iloc[0][std_col] if not r4c.empty else 0
                    v8c = r8c.iloc[0][met] if not r8c.empty else v8
                    s8c = r8c.iloc[0][std_col] if not r8c.empty else 0
                    if met == "energia_media":
                        v4c, s4c, v8c, s8c = v4c*1000, s4c*1000, v8c*1000, s8c*1000

                    ax.bar(cur_x,                    v4,  ancho_f, yerr=s4,  color=c4, edgecolor="white", capsize=3, error_kw={"lw": 1.2})
                    ax.bar(cur_x + ancho_f + 0.02,   v4c, ancho_k, yerr=s4c, color=c4, edgecolor="white", capsize=3, hatch="///", alpha=0.7, error_kw={"lw": 1.0})
                    ax.bar(cur_x + ancho_f*2 + 0.1,  v8,  ancho_f, yerr=s8,  color=c8, edgecolor="white", capsize=3, error_kw={"lw": 1.2})
                    ax.bar(cur_x + ancho_f*3 + 0.12, v8c, ancho_k, yerr=s8c, color=c8, edgecolor="white", capsize=3, hatch="///", alpha=0.7, error_kw={"lw": 1.0})
                    ticks.append(cur_x + ancho_f + 0.11)
                    cur_x += 1.3
                else:
                    ax.bar(cur_x,        v4, ancho_f, yerr=s4, color=c4, edgecolor="white", capsize=3, error_kw={"lw": 1.2})
                    ax.bar(cur_x + 0.45, v8, ancho_f, yerr=s8, color=c8, edgecolor="white", capsize=3, error_kw={"lw": 1.2})
                    ticks.append(cur_x + 0.225)
                    cur_x += 1.3

                tick_lbl.append(f"{modelo[:4]}\n{dev}")

        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_lbl, fontsize=9)
        ax.set_title(titulo, fontweight="bold")

    leyenda = [
        mpatches.Patch(color=C4L, label="Llama-2 Q4"),
        mpatches.Patch(color=C8L, label="Llama-2 Q8"),
        mpatches.Patch(color=C4Q, label="Qwen2.5 Q4"),
        mpatches.Patch(color=C8Q, label="Qwen2.5 Q8"),
    ]
    if stats_clean is not None:
        leyenda += [
            mpatches.Patch(facecolor="gray", alpha=0.9, label="Con outliers"),
            mpatches.Patch(facecolor="gray", hatch="///", alpha=0.7, label="Sin outliers"),
        ]
    fig.legend(handles=leyenda, loc="lower center", ncol=len(leyenda),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    titulo_suptitle = ("Comparacion Q4 vs Q8 por Modelo y Dispositivo\n"
                       "con y sin outliers extremos (IQR x3)"
                       if stats_clean is not None
                       else "Comparacion Q4 vs Q8 por Modelo y Dispositivo\n"
                            "(datos completos incluyendo outliers)")
    fig.suptitle(titulo_suptitle, fontsize=12, fontweight="bold")
    axes[1].annotate(
        "* Energía GPU estimada por CodeCarbon. "
        "Apple Silicon no expone API directa de consumo GPU.",
        xy=(0, -0.12), xycoords='axes fraction',
        fontsize=7, color='gray', style='italic'
    )
    plt.tight_layout(rect=(0, 0.07, 1, 1))
    ruta = PLOTS_DIR / "fig5_q4_vs_q8.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  guardado: {ruta.name}")


def fig6_metricas_normalizadas(stats, stats_clean=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x       = np.arange(len(stats))
    colores = [COLORES.get(c, "#999") for c in stats["config"]]
    labels  = [c.replace(" ", "\n") for c in stats["config"]]
    ancho   = 0.38 if stats_clean is not None else 0.6
    xf      = x - ancho / 2 if stats_clean is not None else x

    for ax, (med_col, std_col, escala, ylabel, titulo_sub) in zip(axes, [
        ("ept_media",    "ept_std",    1e6, "Energia por token (µWh/token)",         "Eficiencia Energetica\n(energia por token generado)"),
        ("co2_1k_media", "co2_1k_std", 1.0, "CO2eq por 1.000 tokens (mg CO2eq)",     "Huella de Carbono Normalizada\n(CO2eq por cada 1.000 tokens generados)"),
    ]):
        v    = stats[med_col].fillna(0) * escala
        v_sd = stats[std_col].fillna(0) * escala

        ax.bar(xf, v, ancho, yerr=v_sd, color=colores, edgecolor="white",
               capsize=4, error_kw={"linewidth": 1.5, "ecolor": "#333"},
               label="Con outliers" if stats_clean is not None else None)

        if stats_clean is not None:
            vc    = stats_clean[med_col].fillna(0) * escala
            vc_sd = stats_clean[std_col].fillna(0) * escala
            ax.bar(x + ancho / 2, vc, ancho, yerr=vc_sd, color=colores,
                   edgecolor="white", capsize=4, hatch="///", alpha=0.7,
                   error_kw={"linewidth": 1.5, "ecolor": "#555"},
                   label="Sin outliers")
            ax.legend(fontsize=9)
        else:
            for i, (val, err) in enumerate(zip(v, v_sd)):
                ax.text(i, val + err + 0.001, f"{val:.2f}", ha="center", fontsize=7.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylabel(ylabel)
        ax.set_title(titulo_sub, fontweight="bold")

    subtitulo = ("con y sin outliers extremos (IQR x3)"
                 if stats_clean is not None
                 else "Paraguay: 26 gCO2/kWh, datos completos")
    fig.suptitle(f"Metricas Normalizadas por Token — GREEN-IA\n({subtitulo})",
                 fontsize=12, fontweight="bold")
    axes[0].annotate(
        "* Energía GPU estimada por CodeCarbon. "
        "Apple Silicon no expone API directa de consumo GPU.",
        xy=(0, -0.12), xycoords='axes fraction',
        fontsize=7, color='gray', style='italic'
    )
    plt.tight_layout()
    ruta = PLOTS_DIR / "fig6_metricas_normalizadas.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  guardado: {ruta.name}")


if __name__ == "__main__":
    print("=" * 55)
    print("  GREEN-IA: generando graficas")
    print("=" * 55)

    df          = cargar_csv()
    df          = preparar_datos(df)
    stats       = estadisticas(df)
    stats_clean = estadisticas(df[~df["outlier"]].copy())

    n_out = df["outlier"].sum()
    print(f"\n  stats_clean: {len(df) - n_out} mediciones ({n_out} outliers excluidos)")

    print("\n" + "=" * 55)
    print("  generando graficas...")
    print("=" * 55)

    fig1_energia_apilada(stats, stats_clean)
    fig2_velocidad(stats, stats_clean)
    fig3_co2(stats, stats_clean)
    fig4_boxplot_tiempos(df)
    fig5_q4_vs_q8(stats, stats_clean)
    fig6_metricas_normalizadas(stats, stats_clean)

    print(f"\n  listo. graficas en: {PLOTS_DIR}")
"""GREEN-IA — Analisis energetico de categorias MT-Bench. 5 prompts x 15 reps, CodeCarbon, CSV y plots."""

import csv
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config
from src.prompt_builder import build_prompt

try:
    from llama_cpp import Llama
except ImportError:
    print("ERROR: llama-cpp-python no instalado. Ver README.md → Instalacion.")
    sys.exit(1)

try:
    from codecarbon import EmissionsTracker
except ImportError:
    print("ERROR: codecarbon no instalado. pip install codecarbon")
    sys.exit(1)

CATEGORIAS = ["math"]
MODEL_NAME = "llama-2-7b"        # nombre del modelo segun config.yaml
CUANTIZACION = "q8"              # clave de cuantizacion usada en config.yaml ("q4" o "q8")
REPETICIONES = 1                 # veces que se repite cada prompt
WARMUP_REPETICIONES = 3          # inferencias descartadas antes de medir oficialmente
                                 # hallazgo empirico: CV baja de 5-6% a <1% desde la 3ra rep
SUBSET_YAML = ROOT / "data" / "mt_bench" / "subset" / "mt_bench_literal_subset_5_per_category.yaml"  # subset de prompts
COLORES_BARRAS = ["#F5C518", "#7EC86A", "#4A90D9", "#E34948", "#EB6834"]  # uno por prompt (5)
COLORES_CATEGORIAS = {"writing": "#F5C518", "reasoning": "#4A90D9", "coding": "#7EC86A"}  # uno por categoria
COLUMNAS_CSV = [
    "categoria", "prompt_id", "prompt_num", "repeticion",
    "prompt_texto_corto", "measured_energy_kwh", "measured_energy_mwh",
    "cpu_energy_mwh", "gpu_energy_mwh", "ram_energy_mwh",
    "inference_time_s", "completion_tokens",
]  # columnas del CSV de salida
COLUMNAS_CSV_PRUEBA = [
    "modelo", "categoria", "prompt_id", "repeticion",
    "prompt_enviado", "respuesta_texto",
]
MAX_CHARS_PROMPT = 80            # caracteres del texto del prompt a guardar en el CSV
#len(prompts)

def cargar_prompts(ruta_yaml: Path, categoria: str) -> list:
    """Carga los prompts de una categoria desde el archivo YAML del subset."""
    if not ruta_yaml.exists():
        print(f"\nERROR: subset no encontrado: {ruta_yaml}")
        print("Ejecutar: python scripts/prepare_mt_bench_subset.py")
        sys.exit(1)
    with open(ruta_yaml, "r", encoding="utf-8") as f:
        datos = yaml.safe_load(f) or {}
    preguntas = datos.get("questions") or []
    filtradas = [q for q in preguntas
                 if q.get("category") == categoria or q.get("internal_category") == categoria]
    if not filtradas:
        disponibles = sorted({q.get("category", "?") for q in preguntas})
        print(f"\nERROR: categoria '{categoria}' no encontrada en el subset.")
        print(f"  Categorias disponibles: {disponibles}")
        sys.exit(1)
    return filtradas


def medir_energia(llm, prompt: str, cfg) -> dict:
    """Ejecuta una inferencia y mide energia con CodeCarbon. Devuelve metricas."""
    tracker = EmissionsTracker(
        project_name="green_ai_analisis_categoria",
        measure_power_secs=1,
        save_to_file=False,
        log_level="error",
        allow_multiple_runs=True,
        # force_cpu_power y force_ram_power removidos:
        # con sudo, CodeCarbon usa powermetrics real (verificado:
        # CPU Power y GPU Power medidos por separado, no estimados)
        # Esto permite medir CPU, GPU y RAM de forma independiente
    )
    tracker.start()
    t_inicio = time.perf_counter()
    resultado = llm(
        prompt,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        seed=cfg.seed,
        echo=cfg.echo,
        stop=None,
    )
    t_fin = time.perf_counter()
    tracker.stop()
    # Limpiar KV-cache para que cada medicion sea independiente
    try:
        llm.reset()
    except Exception:
        pass
    def _kwh(nombre) -> float:
        obj = getattr(tracker, nombre, None)
        if obj is not None and hasattr(obj, "kWh"):
            try:
                return float(obj.kWh)
            except Exception:
                pass
        return 0.0
    kwh_total = _kwh("_total_energy")
    if kwh_total == 0:
        kwh_total = _kwh("_total_cpu_energy") + _kwh("_total_gpu_energy") + _kwh("_total_ram_energy")
    uso = resultado.get("usage", {})
    return {
        "tokens_generados"    : int(uso.get("completion_tokens", 0)),
        "tiempo_inferencia_s" : t_fin - t_inicio,
        "energia_kwh"         : kwh_total,
        "energia_mwh"         : kwh_total * 1_000_000.0,
        "cpu_energia_mwh"     : _kwh("_total_cpu_energy") * 1_000_000.0,
        "gpu_energia_mwh"     : _kwh("_total_gpu_energy") * 1_000_000.0,
        "ram_energia_mwh"     : _kwh("_total_ram_energy") * 1_000_000.0,
        "prompt_enviado"      : prompt,
        "respuesta_texto"     : resultado["choices"][0]["text"],
    }


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
    print(f"  Plot 1: {ruta_plot1.relative_to(ROOT)}")
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
    print(f"  Plot 2: {ruta_plot2.relative_to(ROOT)}")
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
    totales3 = [c + g + r for c, g, r in zip(medias_cpu, medias_gpu, medias_ram)]
    umbral3 = max(totales3) * 0.05 if totales3 else 0
    for xi, (cpu, gpu, ram) in enumerate(zip(medias_cpu, medias_gpu, medias_ram)):
        if cpu > umbral3:
            ax3.text(xi, cpu / 2, f"{cpu:.4f}", ha="center", va="center",
                     fontsize=7.5, color="white", fontweight="bold", zorder=5)
        if gpu > umbral3:
            ax3.text(xi, cpu + gpu / 2, f"{gpu:.4f}", ha="center", va="center",
                     fontsize=7.5, color="white", fontweight="bold", zorder=5)
        if ram > umbral3:
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
    print(f"  Plot 3: {ruta_plot3.relative_to(ROOT)}")


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


def main() -> None:
    """Ejecuta el experimento completo: carga modelo UNA vez, itera categorias, guarda CSV y genera plots."""
    ANCHO = 68
    cfg = load_config(ROOT / "config.yaml")
    # Leer ruta del modelo Q4 desde config.yaml
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        config_raw = yaml.safe_load(f) or {}
    ruta_modelo_str = config_raw.get("models", {}).get(MODEL_NAME, {}).get(
        CUANTIZACION, f"models/{MODEL_NAME}/{MODEL_NAME}.Q4_K_M.gguf")
    ruta_modelo = ROOT / ruta_modelo_str
    # caffeinate evita que macOS entre en reposo durante el experimento
    proceso_cafe = None
    if sys.platform == "darwin":
        try:
            proceso_cafe = subprocess.Popen(["caffeinate", "-dimsu"])
            print(f"  caffeinate activo (PID {proceso_cafe.pid})")
        except Exception:
            print("  AVISO: no se pudo activar caffeinate")
    # Verificar modelo antes de cargar
    if not ruta_modelo.exists():
        print(f"\nERROR: modelo no encontrado: {ruta_modelo}")
        if proceso_cafe is not None:
            proceso_cafe.terminate()
        sys.exit(1)
    print(f"\n  Cargando {ruta_modelo.name}...")
    t_carga = time.perf_counter()
    llm = Llama(
        model_path=str(ruta_modelo), n_ctx=cfg.n_ctx,
        n_gpu_layers=cfg.n_gpu_layers(), n_threads=cfg.n_threads(),
        n_batch=cfg.n_batch(), verbose=False,
    )
    print(f"  Modelo cargado en {time.perf_counter() - t_carga:.1f}s")

    resumen_categorias = {}  # categoria -> {"media_global", "std_global", "cv_global"}

    for CATEGORIA in CATEGORIAS:
        output_dir = ROOT / "results" / "analisis_categoria" / CATEGORIA
        output_csv = output_dir / "resultados_completos.csv"

        print(f"\n{'=' * ANCHO}")
        print(f"  GREEN-IA — Analisis de categoria: {CATEGORIA.upper()}")
        print(f"  Modelo   : {MODEL_NAME} Q4  (n_gpu_layers={cfg.n_gpu_layers()}, {cfg.expected_backend()})")
        print(f"  Params   : n_ctx={cfg.n_ctx}  max_tokens={cfg.max_tokens}  temperature={cfg.temperature}  seed={cfg.seed}")
        print(f"  Reps     : {REPETICIONES} por prompt")
        print(f"  Salida   : {output_csv.relative_to(ROOT)}")
        print(f"{'=' * ANCHO}")
        # Cargar prompts de la categoria y tomar los primeros 5
        preguntas = cargar_prompts(SUBSET_YAML, CATEGORIA)
        preguntas = preguntas[:5]
        print(f"\n  {len(preguntas)} prompts en categoria '{CATEGORIA}'")
        etiquetas_prompt = {i + 1: f"Prompt {i + 1}" for i in range(len(preguntas))}
        # Preparar directorio de salida y abrir CSV
        output_dir.mkdir(parents=True, exist_ok=True)
        archivo_csv = open(output_csv, "w", newline="", encoding="utf-8")
        escritor = csv.DictWriter(archivo_csv, fieldnames=COLUMNAS_CSV)
        escritor.writeheader()
        # Este CSV es la entrada para scripts/juez_calidad.py.
        # El juez toma prompt_enviado y respuesta_texto de cada fila,
        # los envia a la API de Claude (temperature=0, para juicio
        # reproducible), y devuelve un accuracy_total mas criterios
        # desglosados especificos de la categoria. El resultado del
        # juez se guarda en un CSV SEPARADO ({categoria}_evaluacion_calidad.csv)
        # para no mezclar datos experimentales (energia, texto generado)
        # con evaluaciones de un tercero (juicio del modelo juez).
        output_dir_prueba = ROOT / "results" / "analisis_categoria_prueba"
        output_csv_prueba = output_dir_prueba / f"{CATEGORIA}_prompts_respuestas.csv"
        output_dir_prueba.mkdir(parents=True, exist_ok=True)
        archivo_csv_prueba = open(output_csv_prueba, "w", newline="", encoding="utf-8")
        escritor_prueba = csv.DictWriter(archivo_csv_prueba, fieldnames=COLUMNAS_CSV_PRUEBA)
        escritor_prueba.writeheader()
        total, n_hecho = len(preguntas) * REPETICIONES, 0
        energia_por_prompt = defaultdict(list)
        cpu_por_prompt = defaultdict(list)
        gpu_por_prompt = defaultdict(list)
        ram_por_prompt = defaultdict(list)
        print(f"\n  {len(preguntas)} prompts x {REPETICIONES} reps = {total} inferencias\n")
        try:
            # Warmup: descartar las primeras inferencias para evitar
            # contaminacion por throttling termico (CV alto en primeras reps)
            print(f"\n  Warmup ({WARMUP_REPETICIONES} inferencias descartadas)...")
            primer_prompt_texto = (preguntas[0].get("turns") or [""])[0]
            prompt_warmup = build_prompt(MODEL_NAME, [{"role": "user", "content": primer_prompt_texto}])
            for w in range(WARMUP_REPETICIONES):
                medir_energia(llm, prompt_warmup, cfg)
                print(f"    warmup {w+1}/{WARMUP_REPETICIONES} ok")
            n_reintentos = 0
            for num_prompt, pregunta in enumerate(preguntas, start=1):
                id_pregunta = int(pregunta.get("question_id", 0))
                turnos = pregunta.get("turns") or []
                texto_turno1 = turnos[0] if turnos else ""
                prompt = build_prompt(MODEL_NAME, [{"role": "user", "content": texto_turno1}])
                for rep in range(1, REPETICIONES + 1):
                    medicion = medir_energia(llm, prompt, cfg)
                    mwh = medicion["energia_mwh"]
                    if mwh != mwh:  # NaN check - bug conocido CodeCarbon/powermetrics (mlco2/codecarbon#985)
                        n_reintentos += 1
                        print(f"    AVISO: medicion NaN (bug CodeCarbon #985), reintentando...")
                        medicion = medir_energia(llm, prompt, cfg)
                        mwh = medicion["energia_mwh"]
                    n_hecho += 1
                    energia_por_prompt[num_prompt].append(mwh)
                    cpu_por_prompt[num_prompt].append(medicion["cpu_energia_mwh"])
                    gpu_por_prompt[num_prompt].append(medicion["gpu_energia_mwh"])
                    ram_por_prompt[num_prompt].append(medicion["ram_energia_mwh"])
                    escritor.writerow({
                        "categoria": CATEGORIA, "prompt_id": id_pregunta,
                        "prompt_num": num_prompt, "repeticion": rep,
                        "prompt_texto_corto": texto_turno1[:MAX_CHARS_PROMPT].replace("\n", " "),
                        "measured_energy_kwh": medicion["energia_kwh"],
                        "measured_energy_mwh": mwh,
                        "cpu_energy_mwh": medicion["cpu_energia_mwh"],
                        "gpu_energy_mwh": medicion["gpu_energia_mwh"],
                        "ram_energy_mwh": medicion["ram_energia_mwh"],
                        "inference_time_s": round(medicion["tiempo_inferencia_s"], 4),
                        "completion_tokens": medicion["tokens_generados"],
                    })
                    archivo_csv.flush()
                    escritor_prueba.writerow({
                        "modelo"         : f"{MODEL_NAME}-{CUANTIZACION}",
                        "categoria"      : CATEGORIA,
                        "prompt_id"      : id_pregunta,
                        "repeticion"     : rep,
                        "prompt_enviado" : medicion["prompt_enviado"],
                        "respuesta_texto": medicion["respuesta_texto"],
                    })
                    archivo_csv_prueba.flush()
                    print(f"  [{n_hecho:3d}/{total}]  Prompt {num_prompt}  rep {rep:2d}/{REPETICIONES}"
                          f"  {medicion['tiempo_inferencia_s']:6.1f}s  {medicion['tokens_generados']:4d} tok  {mwh:.4f} mWh")
        finally:
            archivo_csv.close()
            archivo_csv_prueba.close()
        # Resumen estadistico por prompt
        print(f"\n{'─' * ANCHO}")
        print(f"  {'Prompt':<10}  {'Media (mWh)':>12}  {'Std':>10}  {'CV%':>7}  {'IC95':>10}")
        print(f"  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*7}  {'─'*10}")
        for num in sorted(energia_por_prompt.keys()):
            stats = calcular_estadisticas(energia_por_prompt[num])
            print(f"  {etiquetas_prompt[num]:<10}  {stats['media']:12.4f}  {stats['std']:10.4f}  {stats['cv_pct']:7.2f}  {stats['ic95']:10.4f}")
        print(f"\n  CSV guardado : {output_csv}")
        print(f"  Total filas  : {n_hecho}")
        print(f"  Reintentos por NaN: {n_reintentos} de {total} ({n_reintentos/total*100:.1f}%)")
        print(f"\n  Generando plots...")
        generar_plots(output_dir, CATEGORIA, dict(energia_por_prompt), etiquetas_prompt,
                      dict(cpu_por_prompt), dict(gpu_por_prompt), dict(ram_por_prompt))
        # Acumular estadisticas globales de la categoria
        todos_mwh = [v for vals in energia_por_prompt.values() for v in vals]
        stats_global = calcular_estadisticas(todos_mwh)
        resumen_categorias[CATEGORIA] = {
            "media_global": stats_global["media"],
            "std_global"  : stats_global["std"],
            "cv_global"   : stats_global["cv_pct"],
        }

    # Resumen comparativo de todas las categorias
    SEP = "═" * 44
    print(f"\n{SEP}")
    print(f"  RESUMEN COMPARATIVO — {len(CATEGORIAS)} categorias")
    print(f"{SEP}")
    for cat, stats in resumen_categorias.items():
        print(f"  {cat + ':':<12}  media={stats['media_global']:.3f} mWh   CV_global={stats['cv_global']:.1f}%")
    cat_max = max(resumen_categorias, key=lambda c: resumen_categorias[c]["media_global"])
    cat_min = min(resumen_categorias, key=lambda c: resumen_categorias[c]["media_global"])
    print(f"\n  Categoria mas consumidora: {cat_max}")
    print(f"  Categoria mas eficiente:   {cat_min}")
    print(f"{SEP}")
    # Plot comparativo entre categorias omitido: prueba de concepto corre una sola categoria.
    # Cerrar caffeinate al terminar
    if proceso_cafe is not None:
        proceso_cafe.terminate()
        print("  caffeinate detenido")
    print(f"{'=' * ANCHO}\n")


if __name__ == "__main__":
    main()

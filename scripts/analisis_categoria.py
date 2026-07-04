"""
scripts/analisis_categoria.py
Proyecto GREEN-IA — Maestria Ciencia de Datos
Universidad Comunero — Paraguay

Analisis energetico de una categoria fija de MT-Bench.
Carga Llama-2-7B Q8 en GPU Metal, corre 15 repeticiones por prompt
(5 prompts x 15 reps = 75 mediciones), mide energia con CodeCarbon,
calcula estadisticas y genera 2 plots.

Salida:
  results/analisis_categoria/resultados_completos.csv
  results/analisis_categoria/boxplot_repeticiones.png
  results/analisis_categoria/barras_con_std.png

Uso:
  python scripts/analisis_categoria.py
"""

from __future__ import annotations

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

from src.config_loader import load_config, ExperimentConfig
from src.prompt_builder import build_prompt

try:
    from llama_cpp import Llama
except ImportError:
    print("ERROR: llama-cpp-python no instalado.  Ver README.md → Instalacion.")
    sys.exit(1)

try:
    from codecarbon import EmissionsTracker
except ImportError:
    print("ERROR: codecarbon no instalado.  pip install codecarbon")
    sys.exit(1)

# ─── Configuracion ────────────────────────────────────────────────────────────

CATEGORIA    = "reasoning"     # categoria a analizar (debe existir en el subset)
MODEL_NAME   = "llama-2-7b"   # clave en config.yaml
REPETITIONS  = 15             # repeticiones por prompt

SUBSET_YAML  = (ROOT / "data" / "mt_bench" / "subset"
                / "mt_bench_literal_subset_5_per_category.yaml")
OUTPUT_DIR   = ROOT / "results" / "analisis_categoria"
OUTPUT_CSV   = OUTPUT_DIR / "resultados_completos.csv"

# Colores de las 5 barras (orden: amarillo, verde claro, azul, rojo, naranja)
BAR_COLORS = ["#F5C518", "#7EC86A", "#4A90D9", "#E34948", "#EB6834"]

CSV_COLUMNS = [
    "categoria",
    "prompt_id",
    "prompt_num",      # 1-5 dentro de la categoria
    "repeticion",
    "prompt_texto_corto",
    "measured_energy_kwh",
    "measured_energy_mwh",
    "inference_time_s",
    "completion_tokens",
]

PROMPT_SHORT_LEN = 80


# ─── Carga del subset ─────────────────────────────────────────────────────────

def _load_categoria(path: Path, categoria: str) -> list[dict]:
    if not path.exists():
        print(f"\nERROR: subset no encontrado: {path}")
        print("Ejecutar: python scripts/prepare_mt_bench_subset.py")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    questions = data.get("questions") or []
    cat_qs = [
        q for q in questions
        if q.get("category") == categoria
        or q.get("internal_category") == categoria
    ]
    if not cat_qs:
        available = sorted({q.get("category", "?") for q in questions})
        print(f"\nERROR: categoria '{categoria}' no encontrada en el subset.")
        print(f"  Categorias disponibles: {available}")
        sys.exit(1)
    return cat_qs


# ─── Extraccion de energia de CodeCarbon ──────────────────────────────────────

def _extract_kwh(tracker: "EmissionsTracker") -> float:
    def _attr_kwh(name: str) -> float:
        obj = getattr(tracker, name, None)
        if obj is not None and hasattr(obj, "kWh"):
            try:
                return float(obj.kWh)
            except Exception:
                pass
        return 0.0

    total = _attr_kwh("_total_energy")
    if total > 0:
        return total
    return (_attr_kwh("_total_cpu_energy")
            + _attr_kwh("_total_gpu_energy")
            + _attr_kwh("_total_ram_energy"))


# ─── Inferencia con medicion CodeCarbon ───────────────────────────────────────

def _measure(llm: "Llama", prompt: str, cfg: "ExperimentConfig") -> dict:
    tracker = EmissionsTracker(
        project_name        = "green_ai_analisis_categoria",
        measure_power_secs  = 1,
        save_to_file        = False,
        log_level           = "error",
        allow_multiple_runs = True,
        force_cpu_power     = 20,
        force_ram_power     = 3,
    )
    tracker.start()
    t0  = time.perf_counter()
    out = llm(
        prompt,
        max_tokens  = cfg.max_tokens,
        temperature = cfg.temperature,
        top_p       = cfg.top_p,
        seed        = cfg.seed,
        echo        = cfg.echo,
        stop        = None,
    )
    t1 = time.perf_counter()
    tracker.stop()

    # Limpiar KV-cache entre inferencias para evitar contaminacion
    # entre repeticiones — garantiza que cada medicion es independiente
    try:
        llm.reset()
    except Exception:
        pass

    kwh   = _extract_kwh(tracker)
    usage = out.get("usage", {})
    return {
        "response_text"       : out["choices"][0]["text"],
        "completion_tokens"   : int(usage.get("completion_tokens", 0)),
        "inference_time_s"    : t1 - t0,
        "measured_energy_kwh" : kwh,
        "measured_energy_mwh" : kwh * 1_000_000.0,
    }


# ─── Estadisticas por prompt ──────────────────────────────────────────────────

def _stats(values: list[float]) -> dict:
    n    = len(values)
    mean = sum(values) / n
    std  = math.sqrt(sum((v - mean) ** 2 for v in values) / max(n - 1, 1))
    cv   = (std / mean * 100) if mean > 0 else 0.0
    ic95 = 1.96 * std / math.sqrt(n)
    return {"mean": mean, "std": std, "cv_pct": cv, "ic95": ic95, "n": n}


# ─── Generacion de plots ──────────────────────────────────────────────────────

def _generate_plots(
    output_dir: Path,
    categoria: str,
    prompt_mwh: dict[int, list[float]],   # prompt_num (1-5) → lista de mWh
    prompt_labels: dict[int, str],        # prompt_num → etiqueta "Prompt N"
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("\n  AVISO: matplotlib no instalado — plots omitidos.")
        print("  pip install matplotlib")
        return

    prompt_nums = sorted(prompt_mwh.keys())

    # ═══════════════════════════════════════════════════════════════════════════
    # PLOT 1 — boxplot_repeticiones.png
    # ═══════════════════════════════════════════════════════════════════════════

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig1.patch.set_facecolor("white")
    ax1.set_facecolor("white")

    data_box = [prompt_mwh[p] for p in prompt_nums]
    bp = ax1.boxplot(
        data_box,
        patch_artist = True,
        medianprops  = {"color": "black", "linewidth": 1.5},
        whiskerprops = {"color": "#555555", "linewidth": 1.0},
        capprops     = {"color": "#555555", "linewidth": 1.0},
        flierprops   = {"marker": "o", "markersize": 4,
                        "markerfacecolor": "#aaaaaa", "linestyle": "none"},
    )
    for patch, color in zip(bp["boxes"], BAR_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_linewidth(0)

    ax1.set_xticks(range(1, len(prompt_nums) + 1))
    ax1.set_xticklabels(
        [prompt_labels[p] for p in prompt_nums],
        fontsize=10,
    )
    ax1.set_ylabel("Energía (mWh)", fontsize=10)
    ax1.set_title(
        f"Categoría {categoria.capitalize()} — distribución por repetición",
        fontsize=12, fontweight="semibold", pad=10,
    )
    ax1.grid(axis="y", linewidth=0.5, color="#e0e0e0", zorder=0)
    ax1.grid(axis="x", visible=False)
    ax1.set_axisbelow(True)
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax1.spines[sp].set_color("#cccccc")
        ax1.spines[sp].set_linewidth(0.8)

    plot1_path = output_dir / "boxplot_repeticiones.png"
    fig1.savefig(plot1_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig1)
    print(f"  Plot 1: {plot1_path.relative_to(ROOT)}")

    # ═══════════════════════════════════════════════════════════════════════════
    # PLOT 2 — barras_con_std.png
    # ═══════════════════════════════════════════════════════════════════════════

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig2.patch.set_facecolor("white")
    ax2.set_facecolor("white")

    x_pos  = list(range(len(prompt_nums)))
    means  = [_stats(prompt_mwh[p])["mean"] for p in prompt_nums]
    stds   = [_stats(prompt_mwh[p])["std"]  for p in prompt_nums]

    ax2.bar(
        x_pos,
        means,
        width    = 0.55,
        color    = BAR_COLORS,
        linewidth= 0,
        zorder   = 3,
    )

    # Barrita negra = desviacion estandar
    ax2.errorbar(
        x_pos,
        means,
        yerr      = stds,
        fmt       = "none",
        color     = "black",
        linewidth = 1.5,
        capsize   = 5,
        capthick  = 1.5,
        zorder    = 4,
    )

    # Valor encima de cada barra: "media ± std mWh"
    y_top = max(m + s for m, s in zip(means, stds)) if means else 1.0
    offset = y_top * 0.02
    for xi, (mean, std) in enumerate(zip(means, stds)):
        ax2.text(
            xi,
            mean + std + offset,
            f"{mean:.4f} ± {std:.4f} mWh",
            ha        = "center",
            va        = "bottom",
            fontsize  = 8.5,
            color     = "#1a1a1a",
            zorder    = 5,
        )

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(
        [prompt_labels[p] for p in prompt_nums],
        fontsize = 10,
    )
    ax2.set_ylabel("Energía (mWh)", fontsize=10)
    ax2.set_ylim(0, y_top * 1.28)
    ax2.set_title(
        f"Categoría {categoria.capitalize()}",
        fontsize  = 13,
        fontweight= "semibold",
        pad       = 10,
    )

    ax2.grid(axis="y", linewidth=0.5, color="#e0e0e0", zorder=0)
    ax2.grid(axis="x", visible=False)
    ax2.set_axisbelow(True)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax2.spines[sp].set_color("#cccccc")
        ax2.spines[sp].set_linewidth(0.8)

    # Sin leyenda de colores (colores son decorativos)

    plot2_path = output_dir / "barras_con_std.png"
    fig2.savefig(plot2_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"  Plot 2: {plot2_path.relative_to(ROOT)}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    W = 68

    cfg = load_config(ROOT / "config.yaml")

    with open(ROOT / "config.yaml", encoding="utf-8") as fh:
        _raw = yaml.safe_load(fh) or {}
    _model_path_str = (
        _raw.get("models", {})
            .get(MODEL_NAME, {})
            .get("q8", f"models/{MODEL_NAME}/{MODEL_NAME}.Q8_0.gguf")
    )
    model_path = ROOT / _model_path_str

    # Prevenir que macOS entre en reposo durante el experimento
    # caffeinate -dimsu: previene sleep de disco, idle, sistema y pantalla
    _caffeinate = None
    if sys.platform == "darwin":
        try:
            _caffeinate = subprocess.Popen(["caffeinate", "-dimsu"])
            print(f"  caffeinate activo (PID {_caffeinate.pid})")
        except Exception:
            print("  AVISO: no se pudo activar caffeinate")

    print(f"\n{'=' * W}")
    print(f"  GREEN-IA — Analisis de categoria: {CATEGORIA.upper()}")
    print(f"  Modelo   : {MODEL_NAME} Q8  "
          f"(n_gpu_layers={cfg.n_gpu_layers()}, {cfg.expected_backend()})")
    print(f"  Params   : n_ctx={cfg.n_ctx}  max_tokens={cfg.max_tokens}  "
          f"temperature={cfg.temperature}  seed={cfg.seed}")
    print(f"  Reps     : {REPETITIONS} por prompt")
    print(f"  Salida   : {OUTPUT_CSV.relative_to(ROOT)}")
    print(f"{'=' * W}")

    # ── cargar prompts de la categoria ────────────────────────────────────────
    questions = _load_categoria(SUBSET_YAML, CATEGORIA)
    questions = questions[:5]   # exactamente 5 prompts
    print(f"\n  {len(questions)} prompts en categoria '{CATEGORIA}'")

    # Mapeo prompt_num (1-5) → question
    prompt_labels = {i + 1: f"Prompt {i + 1}" for i in range(len(questions))}

    # ── cargar modelo ─────────────────────────────────────────────────────────
    if not model_path.exists():
        print(f"\nERROR: modelo no encontrado: {model_path}")
        sys.exit(1)

    print(f"\n  Cargando {model_path.name}...")
    t_load = time.perf_counter()
    llm = Llama(
        model_path   = str(model_path),
        n_ctx        = cfg.n_ctx,
        n_gpu_layers = cfg.n_gpu_layers(),
        n_threads    = cfg.n_threads(),
        n_batch      = cfg.n_batch(),
        verbose      = False,
    )
    print(f"  Modelo cargado en {time.perf_counter() - t_load:.1f}s")

    # ── preparar directorio y CSV ──────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_fh = open(OUTPUT_CSV, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_fh, fieldnames=CSV_COLUMNS)
    writer.writeheader()

    # ── loop de medicion ──────────────────────────────────────────────────────
    n_total    = len(questions) * REPETITIONS
    n_done     = 0
    prompt_mwh: dict[int, list[float]] = defaultdict(list)

    print(f"\n  {len(questions)} prompts x {REPETITIONS} reps = {n_total} inferencias\n")

    try:
        for prompt_num, q in enumerate(questions, start=1):
            qid       = int(q.get("question_id", 0))
            turns     = q.get("turns") or []
            t1_text   = turns[0] if turns else ""

            msgs   = [{"role": "user", "content": t1_text}]
            prompt = build_prompt(MODEL_NAME, msgs)

            for rep in range(1, REPETITIONS + 1):
                res    = _measure(llm, prompt, cfg)
                mwh    = res["measured_energy_mwh"]
                n_done += 1

                prompt_mwh[prompt_num].append(mwh)

                writer.writerow({
                    "categoria"           : CATEGORIA,
                    "prompt_id"           : qid,
                    "prompt_num"          : prompt_num,
                    "repeticion"          : rep,
                    "prompt_texto_corto"  : t1_text[:PROMPT_SHORT_LEN].replace("\n", " "),
                    "measured_energy_kwh" : res["measured_energy_kwh"],
                    "measured_energy_mwh" : mwh,
                    "inference_time_s"    : round(res["inference_time_s"], 4),
                    "completion_tokens"   : res["completion_tokens"],
                })
                csv_fh.flush()

                print(
                    f"  [{n_done:3d}/{n_total}]  "
                    f"Prompt {prompt_num}  rep {rep:2d}/{REPETITIONS}"
                    f"  {res['inference_time_s']:6.1f}s"
                    f"  {res['completion_tokens']:4d} tok"
                    f"  {mwh:.4f} mWh"
                )

    finally:
        csv_fh.close()

    # ── resumen estadistico ────────────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  {'Prompt':<10}  {'Media (mWh)':>12}  {'Std':>10}  {'CV%':>7}  {'IC95':>10}")
    print(f"  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*7}  {'─'*10}")
    for pn in sorted(prompt_mwh.keys()):
        s = _stats(prompt_mwh[pn])
        print(
            f"  {prompt_labels[pn]:<10}  "
            f"{s['mean']:12.4f}  {s['std']:10.4f}  "
            f"{s['cv_pct']:7.2f}  {s['ic95']:10.4f}"
        )

    print(f"\n  CSV guardado : {OUTPUT_CSV}")
    print(f"  Total filas  : {n_done}")

    # ── plots ─────────────────────────────────────────────────────────────────
    print(f"\n  Generando plots...")
    _generate_plots(OUTPUT_DIR, CATEGORIA, dict(prompt_mwh), prompt_labels)

    # Cerrar caffeinate al terminar
    if _caffeinate is not None:
        _caffeinate.terminate()
        print("  caffeinate detenido")

    print(f"{'=' * W}\n")


if __name__ == "__main__":
    main()

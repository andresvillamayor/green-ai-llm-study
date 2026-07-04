"""
scripts/prueba_consumo_por_categoria.py
Proyecto GREEN-IA — Maestria Ciencia de Datos
Universidad Comunero — Paraguay
Autor: Andres Villamayor

Prueba de consumo energetico por categoria MT-Bench.

Carga Llama-2-7B Q8 en GPU Metal y ejecuta 1 inferencia por turno de
cada pregunta del subset oficial (40 preguntas x 2 turnos = 80 mediciones).
Los parametros de inferencia son identicos al experimento oficial (config.yaml).

Salida:
  results/prueba_categoria/consumo_por_prompt.csv

Uso:
  python scripts/prueba_consumo_por_categoria.py
"""

from __future__ import annotations

import csv
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

# ─── Constantes ───────────────────────────────────────────────────────────────

MODEL_NAME  = "llama-2-7b"   # clave en config.yaml + selector de template de prompt
SUBSET_YAML = (ROOT / "data" / "mt_bench" / "subset"
               / "mt_bench_literal_subset_5_per_category.yaml")
OUTPUT_DIR  = ROOT / "results" / "prueba_categoria"
OUTPUT_CSV  = OUTPUT_DIR / "consumo_por_prompt.csv"

# Parametros de inferencia — leidos de config.yaml via load_config()
# (n_ctx, max_tokens, temperature, top_p, seed, echo, n_gpu_layers, n_threads, n_batch)
# NO se usa src/baseline_energy.py — esta prueba no necesita baseline.

PROMPT_SHORT_LEN = 80

# Paleta validada (dataviz skill) — 8 slots, orden fijo CVD-safe
# Worst adjacent ΔE = 24.2 en modo claro (target ≥ 12)
# Slots aqua/yellow/magenta < 3:1 contraste → relief por etiquetas directas (ver plots)
CATEGORY_ORDER = [
    "writing", "roleplay", "extraction", "reasoning",
    "math", "coding", "stem", "humanities",
]
CAT_COLORS = {
    "writing"    : "#2a78d6",  # slot 1 — blue
    "roleplay"   : "#1baf7a",  # slot 2 — aqua
    "extraction" : "#eda100",  # slot 3 — yellow
    "reasoning"  : "#008300",  # slot 4 — green
    "math"       : "#4a3aa7",  # slot 5 — violet
    "coding"     : "#e34948",  # slot 6 — red
    "stem"       : "#e87ba4",  # slot 7 — magenta
    "humanities" : "#eb6834",  # slot 8 — orange
}

# Tokens de color del sistema (chart chrome)
_SURFACE    = "#fcfcfb"
_INK        = "#0b0b0b"
_INK2       = "#52514e"
_GRID       = "#e1e0d9"
_AXIS_LINE  = "#c3c2b7"

CSV_COLUMNS = [
    "categoria",
    "prompt_id",
    "prompt_texto_corto",
    "turno",
    "measured_energy_kwh",
    "measured_energy_wh",
    "inference_time_s",
    "completion_tokens",
]


# ─── Carga del subset ─────────────────────────────────────────────────────────

def _load_subset(path: Path) -> list[dict]:
    if not path.exists():
        print(f"\nERROR: subset no encontrado: {path}")
        print("Ejecutar: python scripts/prepare_mt_bench_subset.py")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    questions = data.get("questions") or []
    if not questions:
        print(f"\nERROR: subset vacio: {path}")
        sys.exit(1)
    return questions


# ─── Extraccion de energia de CodeCarbon ──────────────────────────────────────

def _extract_kwh(tracker: "EmissionsTracker") -> float:
    """Retorna la energia total medida en kWh; hace fallback a suma de componentes."""
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


# ─── Inferencia con medicion CodeCarbon (zona critica P1) ────────────────────

def _measure(llm: "Llama", prompt: str, cfg: "ExperimentConfig") -> dict:
    """
    Ejecuta una inferencia con medicion CodeCarbon.
    Solo tracker.start() — llm() — tracker.stop() en la zona critica.
    Params de inferencia leidos de cfg (ExperimentConfig desde config.yaml).
    """
    tracker = EmissionsTracker(
        project_name        = "green_ai_prueba_categoria",
        measure_power_secs  = 1,
        save_to_file        = False,
        log_level           = "error",
        allow_multiple_runs = True,
        force_cpu_power     = 20,  # TDP Apple M4 (NotebookCheck 2024)
        force_ram_power     = 3,   # LPDDR5X 16 GB estimado
    )
    tracker.start()                       # ── inicio zona critica ──
    t0 = time.perf_counter()
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
    tracker.stop()                        # ── fin zona critica ──

    kwh   = _extract_kwh(tracker)
    usage = out.get("usage", {})
    return {
        "response_text"      : out["choices"][0]["text"],
        "completion_tokens"  : int(usage.get("completion_tokens", 0)),
        "inference_time_s"   : t1 - t0,
        "measured_energy_kwh": kwh,
        "measured_energy_wh" : kwh * 1000.0,
    }


# ─── Generacion de plots ──────────────────────────────────────────────────────

def _generate_plots(output_dir: Path, csv_path: Path) -> None:
    """
    Genera Plot 1 (consumo por prompt, barras horizontales) y
    Plot 2 (consumo por categoria, barras verticales con IC 95%).

    Lee el CSV producido en el mismo run; no necesita los resultados en memoria.
    Paleta CVD-safe validada segun dataviz skill; etiquetas directas en todas
    las barras satisfacen la regla de relief para slots < 3:1 contraste.
    """
    import math as _math
    from collections import defaultdict as _defaultdict

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.ticker as ticker
    except ImportError:
        print("\n  AVISO: matplotlib no instalado — plots omitidos.")
        print("  pip install matplotlib")
        return

    # ── leer CSV ──────────────────────────────────────────────────────────────
    # prompt_wh[(cat, qid)] = Wh total (T1 + T2)
    prompt_wh: dict[tuple[str, int], float] = {}
    with open(csv_path, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cat = row["categoria"]
            qid = int(row["prompt_id"])
            wh  = float(row["measured_energy_wh"] or 0)
            key = (cat, qid)
            prompt_wh[key] = prompt_wh.get(key, 0.0) + wh

    if not prompt_wh:
        print("\n  AVISO: CSV vacio — plots omitidos.")
        return

    ordered_cats = [c for c in CATEGORY_ORDER if any(k[0] == c for k in prompt_wh)]

    # Barras para Plot 1: (cat, qid, mWh_total) ordenadas por cat → qid
    bars: list[tuple[str, int, float]] = []
    for cat in ordered_cats:
        items = sorted(
            [(qid, wh * 1000) for (c, qid), wh in prompt_wh.items() if c == cat],
            key=lambda x: x[0],
        )
        bars.extend((cat, qid, mwh) for qid, mwh in items)

    # Estadisticas para Plot 2
    cat_mwh: dict[str, list[float]] = _defaultdict(list)
    for (cat, _), wh in prompt_wh.items():
        cat_mwh[cat].append(wh * 1000)

    cat_mean: dict[str, float] = {}
    cat_ci95: dict[str, float] = {}
    for cat in ordered_cats:
        vals = cat_mwh[cat]
        n    = len(vals)
        mean = sum(vals) / n
        std  = _math.sqrt(sum((v - mean) ** 2 for v in vals) / max(n - 1, 1))
        cat_mean[cat] = mean
        cat_ci95[cat] = 1.96 * std / _math.sqrt(n)

    def _setup_ax(ax: "plt.Axes") -> None:
        """Estilo comun: recesivo, spines minimos, grid hairline."""
        ax.set_facecolor(_SURFACE)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(_AXIS_LINE)
            ax.spines[sp].set_linewidth(0.8)
        ax.tick_params(colors=_INK2, labelsize=8.5)

    # ═══════════════════════════════════════════════════════════════════════════
    # PLOT 1 — consumo_por_prompt.png  (barras horizontales, 40 barras)
    # ═══════════════════════════════════════════════════════════════════════════

    # Posiciones Y con pequeno gap entre categorias
    y_pos: list[float] = []
    cat_gaps: list[float] = []   # posicion Y de la linea separadora
    y, prev_cat = 0.0, None
    for cat, _, _ in bars:
        if prev_cat is not None and cat != prev_cat:
            y += 0.6
            cat_gaps.append(y - 0.3)  # a mitad del gap
        y_pos.append(y)
        y += 1.0
        prev_cat = cat

    max_mwh   = max(mwh for _, _, mwh in bars) if bars else 1.0
    fig_h     = max(12, round(len(bars) * 0.37 + 3))
    fig1, ax1 = plt.subplots(figsize=(12, fig_h))
    fig1.patch.set_facecolor(_SURFACE)
    _setup_ax(ax1)

    for yp, (cat, qid, mwh) in zip(y_pos, bars):
        color = CAT_COLORS.get(cat, "#898781")
        ax1.barh(yp, mwh, height=0.65, color=color, linewidth=0, zorder=3)
        # Etiqueta directa — satisface relief rule para slots < 3:1 contraste
        ax1.text(
            mwh + max_mwh * 0.012,
            yp,
            f"{mwh:.3f}",
            va="center",
            ha="left",
            fontsize=7.5,
            color=_INK,
            zorder=4,
        )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(
        [f"Q{qid}  {cat}" for cat, qid, _ in bars],
        fontsize=7.5,
        color=_INK,
    )
    ax1.invert_yaxis()
    ax1.set_xlabel("Energía (mWh) — suma turno 1 + turno 2", fontsize=9, color=_INK2)
    ax1.set_xlim(0, max_mwh * 1.22)
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax1.grid(axis="x", linewidth=0.5, color=_GRID, zorder=0)
    ax1.grid(axis="y", visible=False)
    ax1.spines["left"].set_visible(False)
    ax1.tick_params(axis="y", length=0)

    # Lineas separadoras entre categorias
    for yg in cat_gaps:
        ax1.axhline(yg, color=_AXIS_LINE, linewidth=0.6, zorder=2)

    # Leyenda (≥ 2 series → obligatoria segun dataviz skill)
    legend_handles = [
        mpatches.Patch(facecolor=CAT_COLORS[c], label=c, linewidth=0)
        for c in ordered_cats
    ]
    ax1.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=8,
        frameon=True,
        framealpha=0.92,
        edgecolor=_GRID,
        facecolor=_SURFACE,
    )

    ax1.set_title(
        "Hardware: Apple Mac Mini M4 | MT-Bench oficial",
        fontsize=9,
        color=_INK2,
        pad=6,
        loc="left",
    )
    fig1.suptitle(
        "Consumo energético por prompt — Llama-2-7B Q8 GPU",
        fontsize=12,
        fontweight="semibold",
        color=_INK,
        x=0.125,
        ha="left",
    )

    plot1_path = output_dir / "consumo_por_prompt.png"
    fig1.savefig(plot1_path, dpi=150, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig1)
    print(f"  Plot 1: {plot1_path.relative_to(ROOT)}")

    # ═══════════════════════════════════════════════════════════════════════════
    # PLOT 2 — consumo_por_categoria.png  (barras verticales, IC 95%)
    # ═══════════════════════════════════════════════════════════════════════════

    fig2, ax2 = plt.subplots(figsize=(10, 6.5))
    fig2.patch.set_facecolor(_SURFACE)
    fig2.subplots_adjust(bottom=0.18)
    _setup_ax(ax2)

    x_pos      = list(range(len(ordered_cats)))
    means_vals = [cat_mean[c] for c in ordered_cats]
    ci95_vals  = [cat_ci95[c]  for c in ordered_cats]
    colors_p2  = [CAT_COLORS.get(c, "#898781") for c in ordered_cats]

    ax2.bar(x_pos, means_vals, width=0.58, color=colors_p2, linewidth=0, zorder=3)

    # Barras de error — IC 95% en negro (segun especificacion)
    ax2.errorbar(
        x_pos,
        means_vals,
        yerr=ci95_vals,
        fmt="none",
        color="black",
        linewidth=1.5,
        capsize=5,
        capthick=1.5,
        zorder=4,
    )

    # Valores encima de cada barra (satisface relief rule + caps)
    y_ceil = max(m + c for m, c in zip(means_vals, ci95_vals)) if means_vals else 1.0
    for xi, (mean, ci) in enumerate(zip(means_vals, ci95_vals)):
        ax2.text(
            xi,
            mean + ci + y_ceil * 0.015,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=_INK,
            zorder=5,
        )

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(ordered_cats, rotation=22, ha="right", fontsize=9, color=_INK)
    ax2.set_ylabel("Energía media (mWh)", fontsize=9, color=_INK2)
    ax2.set_ylim(0, y_ceil * 1.20)
    ax2.grid(axis="y", linewidth=0.5, color=_GRID, zorder=0)
    ax2.grid(axis="x", visible=False)

    # Leyenda (≥ 2 series → obligatoria)
    legend_handles2 = [
        mpatches.Patch(facecolor=CAT_COLORS[c], label=c, linewidth=0)
        for c in ordered_cats
    ]
    ax2.legend(
        handles=legend_handles2,
        loc="upper right",
        fontsize=8,
        frameon=True,
        framealpha=0.92,
        edgecolor=_GRID,
        facecolor=_SURFACE,
        ncol=2,
    )

    ax2.set_title(
        "Promedio de 5 prompts por categoría | Hardware: Apple Mac Mini M4",
        fontsize=9,
        color=_INK2,
        pad=6,
        loc="left",
    )
    fig2.suptitle(
        "Consumo energético por categoría — Llama-2-7B Q8 GPU",
        fontsize=12,
        fontweight="semibold",
        color=_INK,
        x=0.125,
        ha="left",
    )

    # Nota al pie
    fig2.text(
        0.5,
        0.01,
        "Medición: CodeCarbon 3.2.6 | temperature=0.0, seed=42, n_ctx=4096",
        ha="center",
        fontsize=7.5,
        color=_INK2,
    )

    plot2_path = output_dir / "consumo_por_categoria.png"
    fig2.savefig(plot2_path, dpi=150, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig2)
    print(f"  Plot 2: {plot2_path.relative_to(ROOT)}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    W = 66

    # ── leer configuracion oficial (src/config_loader.py) ────────────────────
    cfg = load_config(ROOT / "config.yaml")

    # Ruta del modelo Q8 desde el bloque models en config.yaml
    import yaml as _yaml
    with open(ROOT / "config.yaml", encoding="utf-8") as _fh:
        _raw = _yaml.safe_load(_fh) or {}
    _model_path_str = (
        _raw.get("models", {})
            .get(MODEL_NAME, {})
            .get("q8", f"models/{MODEL_NAME}/{MODEL_NAME}.Q8_0.gguf")
    )
    model_path = ROOT / _model_path_str

    print(f"\n{'=' * W}")
    print("  GREEN-IA — Prueba de consumo energetico por categoria MT-Bench")
    print(f"  Modelo  : {MODEL_NAME} Q8  "
          f"(n_gpu_layers={cfg.n_gpu_layers()}, {cfg.expected_backend()})")
    print(f"  Params  : n_ctx={cfg.n_ctx}  max_tokens={cfg.max_tokens}  "
          f"temperature={cfg.temperature}  seed={cfg.seed}")
    print(f"  Threads : n_threads={cfg.n_threads()}  n_batch={cfg.n_batch()}")
    print(f"  Subset  : {SUBSET_YAML.name}")
    print(f"  Salida  : {OUTPUT_CSV.relative_to(ROOT)}")
    print(f"{'=' * W}")

    # ── cargar subset ─────────────────────────────────────────────────────────
    questions = _load_subset(SUBSET_YAML)
    cats = sorted({q.get("original_category", q.get("category", "?"))
                   for q in questions})
    print(f"\n  {len(questions)} preguntas / {len(cats)} categorias: {cats}")

    # ── cargar modelo ─────────────────────────────────────────────────────────
    if not model_path.exists():
        print(f"\nERROR: modelo no encontrado: {model_path}")
        sys.exit(1)

    print(f"\n  Cargando modelo {model_path.name}...")
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

    # ── preparar CSV ──────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_fh = open(OUTPUT_CSV, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_fh, fieldnames=CSV_COLUMNS)
    writer.writeheader()

    # ── loop de medicion ──────────────────────────────────────────────────────
    n_total = len(questions) * 2
    n_done  = 0

    cat_wh      : dict[str, float] = defaultdict(float)
    cat_tok     : dict[str, int]   = defaultdict(int)
    cat_n_prompts: dict[str, int]  = defaultdict(int)  # prompts completados por cat

    print(f"\n  {len(questions)} preguntas x 2 turnos = {n_total} inferencias\n")

    try:
        for q in questions:
            qid      = int(q.get("question_id", 0))
            cat      = str(q.get("original_category", q.get("category", "unknown")))
            turns    = q.get("turns") or []

            if len(turns) < 2:
                print(f"  AVISO: pregunta {qid} tiene <2 turnos — omitiendo")
                continue

            t1_text = turns[0]
            t2_text = turns[1]

            # ── turno 1 ───────────────────────────────────────────────────────
            msgs_t1   = [{"role": "user", "content": t1_text}]
            prompt_t1 = build_prompt(MODEL_NAME, msgs_t1)
            res_t1    = _measure(llm, prompt_t1, cfg)
            n_done   += 1

            row_t1: dict = {
                "categoria"          : cat,
                "prompt_id"          : qid,
                "prompt_texto_corto" : t1_text[:PROMPT_SHORT_LEN].replace("\n", " "),
                "turno"              : 1,
                "measured_energy_kwh": res_t1["measured_energy_kwh"],
                "measured_energy_wh" : res_t1["measured_energy_wh"],
                "inference_time_s"   : round(res_t1["inference_time_s"], 4),
                "completion_tokens"  : res_t1["completion_tokens"],
            }
            writer.writerow(row_t1)
            csv_fh.flush()

            cat_wh[cat]  += res_t1["measured_energy_wh"]
            cat_tok[cat] += res_t1["completion_tokens"]

            print(
                f"  [{n_done:3d}/{n_total}]  {cat:<35}  Q{qid} T1"
                f"  {res_t1['inference_time_s']:6.1f}s"
                f"  {res_t1['completion_tokens']:4d} tok"
                f"  {res_t1['measured_energy_wh']:.6f} Wh"
            )

            # ── turno 2 ───────────────────────────────────────────────────────
            msgs_t2 = [
                {"role": "user",      "content": t1_text},
                {"role": "assistant", "content": res_t1["response_text"]},
                {"role": "user",      "content": t2_text},
            ]
            prompt_t2 = build_prompt(MODEL_NAME, msgs_t2)
            res_t2    = _measure(llm, prompt_t2, cfg)
            n_done   += 1

            row_t2: dict = {
                "categoria"          : cat,
                "prompt_id"          : qid,
                "prompt_texto_corto" : t2_text[:PROMPT_SHORT_LEN].replace("\n", " "),
                "turno"              : 2,
                "measured_energy_kwh": res_t2["measured_energy_kwh"],
                "measured_energy_wh" : res_t2["measured_energy_wh"],
                "inference_time_s"   : round(res_t2["inference_time_s"], 4),
                "completion_tokens"  : res_t2["completion_tokens"],
            }
            writer.writerow(row_t2)
            csv_fh.flush()

            cat_wh[cat]       += res_t2["measured_energy_wh"]
            cat_tok[cat]      += res_t2["completion_tokens"]
            cat_n_prompts[cat] += 1

            print(
                f"  [{n_done:3d}/{n_total}]  {cat:<35}  Q{qid} T2"
                f"  {res_t2['inference_time_s']:6.1f}s"
                f"  {res_t2['completion_tokens']:4d} tok"
                f"  {res_t2['measured_energy_wh']:.6f} Wh"
            )

    finally:
        csv_fh.close()

    # ── resumen por categoria ─────────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print("  Resumen por categoria:")
    print(f"  {'categoria':<35}  {'Wh total':>12}  {'tokens':>8}")
    print(f"  {'─'*35}  {'─'*12}  {'─'*8}")
    for cat in sorted(cat_wh):
        print(f"  {cat:<35}  {cat_wh[cat]:12.6f}  {cat_tok[cat]:8d}")
    total_wh  = sum(cat_wh.values())
    total_tok = sum(cat_tok.values())
    print(f"  {'TOTAL':<35}  {total_wh:12.6f}  {total_tok:8d}")

    print(f"\n  CSV guardado: {OUTPUT_CSV}")
    print(f"  Total filas : {n_done}")

    # ── plots ─────────────────────────────────────────────────────────────────
    print(f"\n  Generando plots...")
    _generate_plots(OUTPUT_DIR, OUTPUT_CSV)

    # ── resumen terminal (punto 6) ─────────────────────────────────────────────
    # cat_wh acumula Wh (T1+T2) de TODOS los prompts de cada categoria
    # media por prompt = cat_wh[cat] / cat_n_prompts[cat] * 1000  (mWh)
    if cat_n_prompts:
        cat_mean_mwh = {
            c: cat_wh[c] / cat_n_prompts[c] * 1000
            for c in cat_n_prompts
        }
        cat_max = max(cat_mean_mwh, key=lambda c: cat_mean_mwh[c])
        cat_min = min(cat_mean_mwh, key=lambda c: cat_mean_mwh[c])
        total_mwh = total_wh * 1000

        print(f"\n{'─' * W}")
        print(f"  Categoría más consumidora: {cat_max:<25} "
              f"({cat_mean_mwh[cat_max]:.4f} mWh promedio)")
        print(f"  Categoría más eficiente:   {cat_min:<25} "
              f"({cat_mean_mwh[cat_min]:.4f} mWh promedio)")
        print(f"  Total energía experimento: {total_mwh:.4f} mWh")

    print(f"{'=' * W}\n")


if __name__ == "__main__":
    main()

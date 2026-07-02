"""
analysis_plots.py
GREEN-IA — Maestria Ciencia de Datos
Universidad Comunero — Paraguay
Autor: Andres Villamayor

Genera graficas matplotlib del experimento de energia y calidad.

Lee:
  results/raw/conversation_results.csv
  results/judge/judge_results.csv

Guarda en:
  results/plots/energy/      — energia por conv, Wh/1K tokens, J/token, tokens/J
  results/plots/quality/     — score y 8 subdimensiones
  results/plots/efficiency/  — quality/J, quality/Wh, EDP, SCI
  results/plots/latency/     — latencia por categoria y config
  results/plots/pareto/      — scatter Pareto calidad vs energia
  results/plots/boxplots/    — distribuciones por config
  results/plots/heatmaps/    — categoria x configuracion
  results/plots/ranking/     — rankings horizontales

Filtros:
  python analysis_plots.py --hardware-profile mac_m4
  python analysis_plots.py --hardware-profile windows_nvidia
  python analysis_plots.py --execution-device gpu
  python analysis_plots.py --category math
  python analysis_plots.py --model-name qwen2.5-7b
  python analysis_plots.py --phase 1
  python analysis_plots.py --optimization     # comparacion fase 1 vs fase 2
"""
from __future__ import annotations

import argparse
import copy
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ─── Paths ─────────────────────────────────────────────────────────────────────

DEFAULT_CONV_CSV  = ROOT / "results" / "raw"   / "conversation_results.csv"
DEFAULT_JUDGE_CSV = ROOT / "results" / "judge" / "judge_results.csv"
DEFAULT_PLOT_DIR  = ROOT / "results" / "plots"

_JUDGE_COLS = [
    "score", "correctness", "instruction_following", "relevance",
    "completeness", "clarity", "conciseness", "usefulness",
    "multi_turn_coherence", "judge_status",
]

_QUALITY_DIMS = [
    "score", "correctness", "instruction_following", "relevance",
    "completeness", "clarity", "conciseness", "usefulness", "multi_turn_coherence",
]

# ─── Style ─────────────────────────────────────────────────────────────────────

_PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
_QUANT_MARKERS = {"q4": "o", "q8": "s"}
_QUANT_HATCHES = {"q4": "",   "q8": "///"}
_EXPERIMENT_SUBTITLE = (
    "Hardware: Apple Mac Mini M4 16GB | Backend: Metal GPU | MT-Bench 40 prompts"
)
_FOOTER_NOTE = (
    "Medición: CodeCarbon 3.2.6 | Parámetros: temperature=0.0, seed=42, n_ctx=4096"
)
_MSG_INSUFFICIENT = "Datos insuficientes — pendiente experimento completo"
_NEGLIGIBLE_THRESHOLD = 0.001


_CONFIG_LABELS: dict[str, str] = {
    "llama-2-7b/q4":  "Llama-2-7B Q4 (4-bit)",
    "llama-2-7b/q8":  "Llama-2-7B Q8 (8-bit)",
    "qwen2.5-7b/q4":  "Qwen2.5-7B Q4 (4-bit)",
    "qwen2.5-7b/q8":  "Qwen2.5-7B Q8 (8-bit)",
}


def _fmt_config(raw: Any) -> str:
    return _CONFIG_LABELS.get(str(raw), str(raw))


def _apply_subtitle(fig: plt.Figure, ax: plt.Axes) -> None:
    """Promote the ax title to fig suptitle; set experiment context as subtitle."""
    fig.suptitle(ax.get_title(), fontsize=12, y=1.02)
    ax.set_title(_EXPERIMENT_SUBTITLE, fontsize=8, color="#666", style="italic")


def _setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi":         300,
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "grid.linestyle":     "--",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "font.size":          10,
        "axes.titlesize":     11,
        "axes.labelsize":     10,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    8,
        "legend.framealpha":  0.85,
    })


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _ci95(s: pd.Series) -> float:
    vals = s.dropna()
    n = len(vals)
    return float(1.96 * vals.std(ddof=1) / np.sqrt(n)) if n >= 2 else 0.0


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.text(0.5, -0.02, _FOOTER_NOTE, ha="center", va="top",
             fontsize=7, color="#888", transform=fig.transFigure)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    try:
        print(f"    {path.relative_to(ROOT)}")
    except ValueError:
        print(f"    {path}")


def _cmap(keys: list) -> dict:
    return {str(k): _PALETTE[i % len(_PALETTE)]
            for i, k in enumerate(sorted(set(str(k) for k in keys)))}


def _agg(
    df: pd.DataFrame,
    group_cols: list[str],
    val_col: str,
    ok_mask: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Grouped mean + CI95 for val_col, optionally filtered by ok_mask."""
    d = df if ok_mask is None else df[ok_mask]
    gc = [c for c in group_cols if c in d.columns]
    if not gc or val_col not in d.columns:
        return pd.DataFrame()
    d = d[d[val_col].notna()].copy()
    if d.empty:
        return pd.DataFrame()
    return (
        d.groupby(gc, dropna=False)
        .agg(mean=(val_col, "mean"), ci=(val_col, _ci95))
        .reset_index()
    )


def _grouped_bar(
    ax: plt.Axes,
    agg_df: pd.DataFrame,
    x_col: str,
    group_col: str,
    title: str,
    ylabel: str,
    cmap: Optional[dict] = None,
    hmap: Optional[dict] = None,
    val_fmt: str = ".2f",
) -> None:
    """Grouped bar chart with CI95 error bars, drawn on ax."""
    x_vals = sorted(agg_df[x_col].dropna().unique().tolist(), key=str)
    g_vals = sorted(agg_df[group_col].dropna().unique().tolist(), key=str)
    nx, ng = len(x_vals), len(g_vals)
    if not nx or not ng:
        ax.set_title(title + " (no data)")
        return
    _vals = agg_df["mean"].dropna().to_numpy()
    if len(_vals) > 0 and float(np.nanmax(np.abs(_vals))) < _NEGLIGIBLE_THRESHOLD:
        ax.set_title(title)
        ax.text(0.5, 0.5, _MSG_INSUFFICIENT, ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="#888", style="italic")
        return
    w  = 0.75 / ng
    xs = np.arange(nx)
    bar_info = []
    for gi, gv in enumerate(g_vals):
        sub   = agg_df[agg_df[group_col] == gv]
        xm    = dict(zip(sub[x_col], sub["mean"]))
        xe    = dict(zip(sub[x_col], sub["ci"]))
        means = [float(xm.get(xv, np.nan)) for xv in x_vals]
        errs  = [float(xe.get(xv, 0.0))   for xv in x_vals]
        clr   = (cmap or {}).get(str(gv), _PALETTE[gi % len(_PALETTE)])
        htch  = (hmap or {}).get(str(gv), "")
        rects = ax.bar(xs + gi * w, means, w, label=_fmt_config(gv),
                       color=clr, hatch=htch, alpha=0.85,
                       yerr=errs, capsize=3, error_kw={"elinewidth": 0.8, "ecolor": "#555"})
        bar_info.append((rects, means, errs))
    ax.set_xticks(xs + w * (ng - 1) / 2)
    ax.set_xticklabels([_fmt_config(v) for v in x_vals], rotation=35, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.12)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", ncol=max(1, ng // 4))
    offset = 0.01 * ax.get_ylim()[1]
    for rects, means, errs in bar_info:
        for rect, m, e in zip(rects, means, errs):
            if np.isnan(m):
                continue
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                max(m, 0) + e + offset,
                format(m, val_fmt),
                ha="center", va="bottom", fontsize=6.5, color="#333",
            )


def _simple_bar(
    ax: plt.Axes,
    agg_df: pd.DataFrame,
    x_col: str,
    title: str,
    ylabel: str,
    cmap: Optional[dict] = None,
    sort_col: str = "mean",
    ascending: bool = False,
    val_fmt: str = ".2f",
) -> None:
    """Single-group horizontal or vertical bar chart."""
    d = agg_df.sort_values(sort_col, ascending=ascending, na_position="last")
    _vals = d["mean"].dropna().to_numpy()
    if len(_vals) > 0 and float(np.nanmax(np.abs(_vals))) < _NEGLIGIBLE_THRESHOLD:
        ax.set_title(title)
        ax.text(0.5, 0.5, _MSG_INSUFFICIENT, ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="#888", style="italic")
        return
    colors = [(cmap or {}).get(str(c), _PALETTE[i % len(_PALETTE)])
              for i, c in enumerate(d[x_col])]
    rects = ax.bar(range(len(d)), d["mean"], color=colors, alpha=0.85,
                   yerr=d["ci"], capsize=3, error_kw={"elinewidth": 0.8, "ecolor": "#555"})
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels([_fmt_config(v) for v in d[x_col]], rotation=35, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.12)
    ax.set_axisbelow(True)
    offset = 0.01 * ax.get_ylim()[1]
    for rect, m, e in zip(rects, d["mean"].values, d["ci"].values):
        if pd.isna(m):
            continue
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            max(float(m), 0) + float(e) + offset,
            format(float(m), val_fmt),
            ha="center", va="bottom", fontsize=7, color="#333",
        )


def _horizontal_bar(
    ax: plt.Axes,
    agg_df: pd.DataFrame,
    label_col: str,
    title: str,
    xlabel: str,
    best_high: bool = True,
    cmap: Optional[dict] = None,
    val_fmt: str = ".2f",
) -> None:
    """Horizontal ranked bar chart."""
    d = agg_df.sort_values("mean", ascending=not best_high, na_position="last")
    _vals = d["mean"].dropna().to_numpy()
    if len(_vals) > 0 and float(np.nanmax(np.abs(_vals))) < _NEGLIGIBLE_THRESHOLD:
        ax.set_title(title)
        ax.text(0.5, 0.5, _MSG_INSUFFICIENT, ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="#888", style="italic")
        return
    ys = np.arange(len(d))
    colors = [(cmap or {}).get(str(c), _PALETTE[i % len(_PALETTE)])
              for i, c in enumerate(d[label_col])]
    rects = ax.barh(ys, d["mean"], color=colors, alpha=0.85,
                    xerr=d["ci"], capsize=3, error_kw={"elinewidth": 0.8, "ecolor": "#555"})
    ax.set_yticks(ys)
    ax.set_yticklabels([_fmt_config(v) for v in d[label_col]], fontsize=9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_xlim(left=0)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.15)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    offset = 0.01 * ax.get_xlim()[1]
    for rect, m, e in zip(rects, d["mean"].values, d["ci"].values):
        if pd.isna(m):
            continue
        ax.text(
            max(float(m), 0) + float(e) + offset,
            rect.get_y() + rect.get_height() / 2,
            format(float(m), val_fmt),
            ha="left", va="center", fontsize=7, color="#333",
        )


# ─── Data loading ───────────────────────────────────────────────────────────────

def load_data(conv_csv: Path, judge_csv: Path, args: argparse.Namespace) -> pd.DataFrame:
    if not conv_csv.exists():
        print(f"  ERROR: {conv_csv} not found.")
        sys.exit(1)

    df_conv = pd.read_csv(conv_csv, low_memory=False)
    print(f"  conversation_results.csv : {len(df_conv):,} rows")

    if judge_csv.exists():
        df_j    = pd.read_csv(judge_csv, low_memory=False)
        present = [c for c in _JUDGE_COLS if c in df_j.columns]
        df_slim = df_j[["conversation_id"] + present].copy()
        print(f"  judge_results.csv        : {len(df_j):,} rows")
    else:
        print(f"  WARNING: judge_results.csv not found — quality metrics will be NaN")
        df_slim = pd.DataFrame({"conversation_id": pd.Series(dtype=str)})

    df = df_conv.merge(df_slim, on="conversation_id", how="left")

    if getattr(args, "phase", None):
        df = df[df["phase"].astype(str).str.strip() == str(args.phase)]
    if getattr(args, "hardware_profile", None):
        df = df[df.get("hardware_profile", pd.Series("", index=df.index)).astype(str)
                == args.hardware_profile]
    if getattr(args, "execution_device", None):
        df = df[df.get("execution_device", pd.Series("", index=df.index)).astype(str)
                == args.execution_device]
    if getattr(args, "category", None):
        df = df[df.get("original_category", pd.Series("", index=df.index)).astype(str)
                == args.category]
    if getattr(args, "model_name", None):
        df = df[df.get("model_name", pd.Series("", index=df.index)).astype(str)
                == args.model_name]

    _num = [
        "score", "correctness", "instruction_following", "relevance",
        "completeness", "clarity", "conciseness", "usefulness", "multi_turn_coherence",
        "total_measured_energy_wh", "total_baseline_corrected_energy_wh",
        "total_measured_energy_joules", "total_baseline_corrected_energy_joules",
        "total_energy_wh_per_1k_output_tokens_measured",
        "total_energy_wh_per_1k_output_tokens_corrected",
        "total_energy_j_per_output_token_measured",
        "total_energy_j_per_output_token_corrected",
        "total_tokens_per_joule_measured", "total_tokens_per_joule_corrected",
        "total_latency_seconds",
        "total_edp_joule_second_measured", "total_edp_joule_second_corrected",
        "total_operational_sci_per_conversation",
        "total_operational_sci_per_1k_output_tokens",
    ]
    for c in _num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "model_name" in df.columns and "quantization" in df.columns:
        df["config"] = df["model_name"].astype(str) + "/" + df["quantization"].astype(str)

    # Per-conversation quality-per-energy derived columns
    j_ok = df.get("judge_status",        pd.Series("", index=df.index)) == "success"
    c_ok = df.get("status_conversation", pd.Series("", index=df.index)) == "success"
    both = j_ok & c_ok

    for sfx, j_col, wh_col, e1k_col in (
        ("measured",
         "total_measured_energy_joules",
         "total_measured_energy_wh",
         "total_energy_wh_per_1k_output_tokens_measured"),
        ("corrected",
         "total_baseline_corrected_energy_joules",
         "total_baseline_corrected_energy_wh",
         "total_energy_wh_per_1k_output_tokens_corrected"),
    ):
        if "score" in df.columns:
            for col, denom in (
                (f"quality_per_joule_{sfx}",      j_col),
                (f"quality_per_wh_{sfx}",         wh_col),
                (f"quality_per_1k_token_wh_{sfx}", e1k_col),
            ):
                df[col] = np.nan
                if denom in df.columns:
                    mask = both & df["score"].notna() & df[denom].notna() & (df[denom] > 0)
                    df.loc[mask, col] = df.loc[mask, "score"] / df.loc[mask, denom]

    print(f"  After filters            : {len(df):,} rows  "
          f"(judge_ok={int(j_ok.sum())})")
    return df


# ─── 1. Energy plots ───────────────────────────────────────────────────────────

def plot_energy(df: pd.DataFrame, out: Path, sfx: str) -> None:
    cfg = "config"
    cat = "original_category"
    cm  = _cmap(df[cfg].dropna().unique() if cfg in df.columns else [])
    c_ok = df.get("status_conversation", pd.Series("", index=df.index)) == "success"

    specs = [
        (f"total_{'baseline_corrected' if sfx == 'corrected' else 'measured'}_energy_wh",
         "Energy per Conversation (Wh)", "energy_wh"),
        (f"total_energy_wh_per_1k_output_tokens_{sfx}",
         "Wh / 1K Output Tokens", "wh_per_1k_tokens"),
        (f"total_energy_j_per_output_token_{sfx}",
         "J / Output Token", "j_per_token"),
        (f"total_tokens_per_joule_{sfx}",
         "Tokens / Joule", "tokens_per_joule"),
    ]

    for col, ylabel, stem in specs:
        if col not in df.columns:
            continue

        agg_cat = _agg(df, [cat, cfg], col, c_ok)
        if not agg_cat.empty:
            fig, ax = plt.subplots(
                figsize=(12, 7))
            _grouped_bar(ax, agg_cat, cat, cfg,
                         f"{ylabel} by Category", ylabel, cmap=cm)
            _apply_subtitle(fig, ax)
            _save(fig, out / f"{stem}_by_category.png")

        agg_cfg = _agg(df, [cfg], col, c_ok)
        if not agg_cfg.empty:
            fig, ax = plt.subplots(figsize=(12, 7))
            _simple_bar(ax, agg_cfg, cfg, f"{ylabel} by Configuration", ylabel, cmap=cm)
            _apply_subtitle(fig, ax)
            _save(fig, out / f"{stem}_by_config.png")


# ─── 2. Quality plots ──────────────────────────────────────────────────────────

def plot_quality(df: pd.DataFrame, out: Path) -> None:
    cfg  = "config"
    cat  = "original_category"
    cm   = _cmap(df[cfg].dropna().unique() if cfg in df.columns else [])
    j_ok = df.get("judge_status", pd.Series("", index=df.index)) == "success"

    for dim in _QUALITY_DIMS:
        if dim not in df.columns:
            continue
        label = dim.replace("_", " ").title()

        agg_cat = _agg(df, [cat, cfg], dim, j_ok)
        if not agg_cat.empty:
            fig, ax = plt.subplots(
                figsize=(12, 7))
            _grouped_bar(ax, agg_cat, cat, cfg,
                         f"{label} by Category", f"Mean {label}", cmap=cm)
            if dim == "score":
                ax.set_ylim(0, 10.5)
            _apply_subtitle(fig, ax)
            _save(fig, out / f"quality_{dim}_by_category.png")

    # quality score by config (single bar)
    if "score" in df.columns:
        agg = _agg(df, [cfg], "score", j_ok)
        if not agg.empty:
            fig, ax = plt.subplots(figsize=(12, 7))
            _simple_bar(ax, agg, cfg, "Quality Score by Configuration",
                        "Mean Score (1–10)", cmap=cm)
            ax.set_ylim(0, 10.5)
            _apply_subtitle(fig, ax)
            _save(fig, out / "quality_score_by_config.png")

    # Subdimension horizontal bars per config
    sub_dims = [d for d in _QUALITY_DIMS[1:] if d in df.columns]
    configs  = sorted(df[cfg].dropna().unique().tolist(), key=str) if cfg in df.columns else []
    if sub_dims and configs:
        fig, axes = plt.subplots(1, len(configs),
                                 figsize=(12, 7), sharey=True)
        if len(configs) == 1:
            axes = [axes]
        sub_j = df[j_ok] if j_ok.any() else df
        for ax, c in zip(axes, configs):
            sub = sub_j[sub_j[cfg] == c] if cfg in sub_j.columns else sub_j
            means = [float(sub[d].mean()) if d in sub.columns else np.nan for d in sub_dims]
            cis   = [_ci95(sub[d])         if d in sub.columns else 0.0   for d in sub_dims]
            ys = np.arange(len(sub_dims))
            ax.barh(ys, means, xerr=cis, capsize=3,
                    color=cm.get(str(c), _PALETTE[0]), alpha=0.85,
                    error_kw={"elinewidth": 0.8, "ecolor": "#555"})
            ax.set_yticks(ys)
            ax.set_yticklabels([d.replace("_", " ") for d in sub_dims], fontsize=8)
            ax.set_xlim(0, 10.5)
            ax.set_title(_fmt_config(c), fontsize=9)
            ax.set_axisbelow(True)
        fig.suptitle("Quality Subdimensions by Configuration", fontsize=12, y=1.05)
        fig.text(0.5, 1.01, _EXPERIMENT_SUBTITLE, ha="center", fontsize=8,
                 color="#666", style="italic", transform=fig.transFigure)
        plt.tight_layout()
        _save(fig, out / "quality_subdims_overview.png")


# ─── 3. Efficiency plots ───────────────────────────────────────────────────────

def plot_efficiency(df: pd.DataFrame, out: Path, sfx: str) -> None:
    cfg  = "config"
    cat  = "original_category"
    cm   = _cmap(df[cfg].dropna().unique() if cfg in df.columns else [])

    specs = [
        (f"quality_per_joule_{sfx}",         "Quality / Joule",           "quality_per_joule"),
        (f"quality_per_wh_{sfx}",            "Quality / Wh",              "quality_per_wh"),
        (f"quality_per_1k_token_wh_{sfx}",   "Quality / (Wh per 1K tok)", "quality_per_1k_token_wh"),
        (f"total_edp_joule_second_{sfx}",     "EDP (J·s)",                 "edp"),
        ("total_operational_sci_per_1k_output_tokens",
                                              "SCI / 1K Output Tokens",   "sci_1k_tokens"),
    ]

    for col, ylabel, stem in specs:
        if col not in df.columns:
            continue

        agg_cfg = _agg(df, [cfg], col)
        if not agg_cfg.empty:
            fig, ax = plt.subplots(figsize=(12, 7))
            _simple_bar(ax, agg_cfg, cfg, f"{ylabel} by Configuration", ylabel, cmap=cm)
            _apply_subtitle(fig, ax)
            _save(fig, out / f"{stem}_by_config.png")

        agg_cat = _agg(df, [cat, cfg], col)
        if not agg_cat.empty:
            fig, ax = plt.subplots(
                figsize=(12, 7))
            _grouped_bar(ax, agg_cat, cat, cfg,
                         f"{ylabel} by Category", ylabel, cmap=cm)
            _apply_subtitle(fig, ax)
            _save(fig, out / f"{stem}_by_category.png")


# ─── 4. Latency plots ──────────────────────────────────────────────────────────

def plot_latency(df: pd.DataFrame, out: Path) -> None:
    col  = "total_latency_seconds"
    cfg  = "config"
    cat  = "original_category"
    if col not in df.columns:
        return
    cm   = _cmap(df[cfg].dropna().unique() if cfg in df.columns else [])
    c_ok = df.get("status_conversation", pd.Series("", index=df.index)) == "success"

    agg_cat = _agg(df, [cat, cfg], col, c_ok)
    if not agg_cat.empty:
        fig, ax = plt.subplots(
            figsize=(12, 7))
        _grouped_bar(ax, agg_cat, cat, cfg,
                     "Latency by Category", "Mean Latency (s)", cmap=cm, val_fmt=".1f")
        _apply_subtitle(fig, ax)
        _save(fig, out / "latency_by_category.png")

    agg_cfg = _agg(df, [cfg], col, c_ok)
    if not agg_cfg.empty:
        fig, ax = plt.subplots(figsize=(12, 7))
        _simple_bar(ax, agg_cfg, cfg, "Latency by Configuration", "Mean Latency (s)", cmap=cm, val_fmt=".1f")
        _apply_subtitle(fig, ax)
        _save(fig, out / "latency_by_config.png")


# ─── 5. Pareto scatter ─────────────────────────────────────────────────────────

def plot_pareto_scatter(df: pd.DataFrame, out: Path, sfx: str) -> None:
    e_col = f"total_energy_wh_per_1k_output_tokens_{sfx}"
    q_col = "score"
    if e_col not in df.columns or q_col not in df.columns:
        return

    j_ok = df.get("judge_status",        pd.Series("", index=df.index)) == "success"
    c_ok = df.get("status_conversation", pd.Series("", index=df.index)) == "success"
    sub  = df[j_ok & c_ok & df[e_col].notna() & df[q_col].notna()].copy()
    if sub.empty:
        return

    models = sorted(sub["model_name"].dropna().unique()) if "model_name" in sub.columns else []
    model_cm = {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(models)}

    gc = ["model_name", "quantization"]
    gc = [c for c in gc if c in sub.columns]
    agg = sub.groupby(gc, dropna=False).agg(
        mq=(q_col, "mean"),
        me=(e_col, "mean"),
        cq=(q_col, _ci95),
        ce=(e_col, _ci95),
    ).reset_index()

    # 2-objective Pareto (quality↑, energy↓)
    if len(agg) >= 2:
        obj = np.column_stack([agg["mq"].to_numpy(), -agg["me"].to_numpy()])
        pareto = np.ones(len(agg), dtype=bool)
        for i in range(len(agg)):
            for j in range(len(agg)):
                if i != j and (obj[j] >= obj[i]).all() and (obj[j] > obj[i]).any():
                    pareto[i] = False
                    break
        agg["pareto"] = pareto
    else:
        agg["pareto"] = True

    fig, ax = plt.subplots(figsize=(12, 7))

    for _, row in agg.iterrows():
        m   = str(row.get("model_name", ""))
        q   = str(row.get("quantization", ""))
        clr = model_cm.get(m, _PALETTE[0])
        mrk = _QUANT_MARKERS.get(q, "^")
        ax.errorbar(row["me"], row["mq"],
                    xerr=row["ce"], yerr=row["cq"],
                    fmt="none", ecolor=clr, alpha=0.4, elinewidth=1, capsize=3)
        ax.scatter(row["me"], row["mq"],
                   c=clr, marker=mrk, s=100, zorder=5, alpha=0.9,
                   edgecolors="gold" if row["pareto"] else "white",
                   linewidths=2.5 if row["pareto"] else 0.5)
        ax.annotate(_fmt_config(f"{m}/{q}"),
                    xy=(row["me"], row["mq"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=7, alpha=0.75)

    pareto_pts = agg[agg["pareto"]].sort_values("me")
    if len(pareto_pts) >= 2:
        ax.plot(pareto_pts["me"], pareto_pts["mq"],
                "k--", lw=1, alpha=0.45, label="Pareto frontier")

    # Legend: models (color) + quantizations (marker)
    for m, clr in model_cm.items():
        ax.scatter([], [], color=clr, s=50, marker="s", label=m)
    for q, mkr in _QUANT_MARKERS.items():
        ax.scatter([], [], color="gray", s=50, marker=mkr, label=f"quant={q}")
    gold = mpatches.Patch(edgecolor="gold", facecolor="none", lw=2, label="Pareto-efficient")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [gold], loc="lower right", fontsize=7, ncol=2)

    ax.set_xlabel(f"Mean Wh / 1K Output Tokens ({sfx})", fontsize=10)
    ax.set_ylabel("Mean Quality Score (1–10)", fontsize=10)
    ax.set_title("Quality vs. Energy — Pareto Scatter")
    _apply_subtitle(fig, ax)
    _save(fig, out / "pareto_scatter.png")


# ─── 6. Boxplots ───────────────────────────────────────────────────────────────

def plot_boxplots(df: pd.DataFrame, out: Path, sfx: str) -> None:
    cfg  = "config"
    if cfg not in df.columns:
        return
    configs = sorted(df[cfg].dropna().unique().tolist(), key=str)
    cm      = _cmap(configs)
    j_ok    = df.get("judge_status",        pd.Series("", index=df.index)) == "success"
    c_ok    = df.get("status_conversation", pd.Series("", index=df.index)) == "success"

    e_col   = (f"total_baseline_corrected_energy_wh"
               if sfx == "corrected" else "total_measured_energy_wh")
    edp_col = f"total_edp_joule_second_{sfx}"
    qpj_col = f"quality_per_joule_{sfx}"

    specs = [
        (e_col,                "Energy Distribution by Configuration",
         "Energy (Wh)", "boxplot_energy.png", c_ok),
        ("total_latency_seconds", "Latency Distribution by Configuration",
         "Latency (s)", "boxplot_latency.png", c_ok),
        ("score",              "Quality Score Distribution by Configuration",
         "Score (1–10)", "boxplot_quality.png", j_ok),
        (qpj_col,              "Quality per Joule Distribution by Configuration",
         "Quality / Joule", "boxplot_quality_per_joule.png", None),
        (edp_col,              "EDP Distribution by Configuration",
         "EDP (J·s)", "boxplot_edp.png", c_ok),
    ]

    for col, title, ylabel, fname, ok_mask in specs:
        if col not in df.columns:
            continue
        d = df if ok_mask is None else df[ok_mask]
        d = d[d[cfg].notna() & d[col].notna()]
        groups = [d.loc[d[cfg] == c, col].dropna().to_numpy() for c in configs]
        if not any(len(g) > 0 for g in groups):
            continue

        fig, ax = plt.subplots(figsize=(12, 8))
        bp = ax.boxplot(
            groups, patch_artist=True, notch=False,
            medianprops={"color": "black", "linewidth": 1.5},
            whiskerprops={"linewidth": 1}, capprops={"linewidth": 1},
            flierprops={"marker": ".", "markersize": 4, "alpha": 0.4},
        )
        for patch, c in zip(bp["boxes"], configs):
            patch.set_facecolor(cm.get(str(c), _PALETTE[0]))
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(configs) + 1))
        ax.set_xticklabels([_fmt_config(c) for c in configs], rotation=30, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_axisbelow(True)
        _apply_subtitle(fig, ax)
        _save(fig, out / fname)


# ─── 7. Heatmaps ───────────────────────────────────────────────────────────────

def plot_heatmaps(df: pd.DataFrame, out: Path, sfx: str) -> None:
    cat = "original_category"
    cfg = "config"
    if cat not in df.columns or cfg not in df.columns:
        return

    j_ok = df.get("judge_status", pd.Series("", index=df.index)) == "success"
    e_col = (f"total_baseline_corrected_energy_wh"
             if sfx == "corrected" else "total_measured_energy_wh")

    def _hmap(val_col: str, title: str, fname: str,
              cmap_name: str = "YlOrRd", ok_mask=None) -> None:
        if val_col not in df.columns:
            return
        d = df if ok_mask is None else df[ok_mask]
        d = d.copy()
        d[val_col] = pd.to_numeric(d[val_col], errors="coerce")
        pivot = d.groupby([cat, cfg], dropna=False)[val_col].mean().unstack(cfg)
        if pivot.empty or pivot.shape[1] == 0:
            return

        nr, nc = pivot.shape
        fig, ax = plt.subplots(figsize=(14, 8))
        mat = pivot.to_numpy(dtype=float)
        im  = ax.imshow(mat, aspect="auto", cmap=cmap_name)
        plt.colorbar(im, ax=ax, shrink=0.8)

        ax.set_xticks(range(nc))
        ax.set_xticklabels([_fmt_config(c) for c in pivot.columns], rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(nr))
        ax.set_yticklabels(pivot.index.tolist(), fontsize=9)

        vmax = float(np.nanmax(mat)) if not np.all(np.isnan(mat)) else 1.0
        for i in range(nr):
            for j in range(nc):
                v = mat[i, j]
                if not np.isnan(v):
                    txt_color = "white" if v > vmax * 0.65 else "black"
                    ax.text(j, i, f"{v:.2g}", ha="center", va="center",
                            fontsize=7, color=txt_color)
        ax.set_title(title)
        _apply_subtitle(fig, ax)
        plt.tight_layout()
        _save(fig, out / fname)

    _hmap("score", "Mean Quality Score: Category × Configuration",
          "heatmap_quality_category_config.png", "YlGn", ok_mask=j_ok)
    _hmap(e_col, "Mean Energy (Wh): Category × Configuration",
          "heatmap_energy_category_config.png", "YlOrRd")
    _hmap(f"total_energy_wh_per_1k_output_tokens_{sfx}",
          "Mean Wh / 1K Tokens: Category × Configuration",
          "heatmap_energy_1k_tokens_category_config.png", "YlOrRd")


# ─── 8. Rankings ───────────────────────────────────────────────────────────────

def plot_rankings(df: pd.DataFrame, out: Path, sfx: str) -> None:
    cfg  = "config"
    if cfg not in df.columns:
        return
    cm   = _cmap(df[cfg].dropna().unique())
    j_ok = df.get("judge_status",        pd.Series("", index=df.index)) == "success"
    c_ok = df.get("status_conversation", pd.Series("", index=df.index)) == "success"
    e_col = (f"total_baseline_corrected_energy_wh"
             if sfx == "corrected" else "total_measured_energy_wh")

    rankings = [
        ("score",                                    "quality",          "Quality Score",
         "Mean Score (1–10)", True,  j_ok),
        (e_col,                                      "energy",           "Energy (lower = better)",
         "Mean Energy (Wh)", False, c_ok),
        (f"total_tokens_per_joule_{sfx}",            "tokens_per_joule", "Tokens / Joule",
         "Tokens / Joule", True,  c_ok),
        (f"quality_per_joule_{sfx}",                 "quality_per_joule","Quality / Joule",
         "Quality / Joule", True,  None),
        (f"total_edp_joule_second_{sfx}",            "edp",              "EDP (lower = better)",
         "EDP (J·s)", False, c_ok),
        ("total_operational_sci_per_1k_output_tokens","sci",             "SCI / 1K Tokens (lower = better)",
         "SCI / 1K Tokens", False, c_ok),
    ]

    for col, stem, title, xlabel, best_high, ok_mask in rankings:
        if col not in df.columns:
            continue
        agg = _agg(df, [cfg], col, ok_mask)
        if agg.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 8))
        _horizontal_bar(ax, agg, cfg, f"Ranking: {title}", xlabel,
                        best_high=best_high, cmap=cm)
        _apply_subtitle(fig, ax)
        _save(fig, out / f"ranking_{stem}.png")


# ─── Optimization comparison ───────────────────────────────────────────────────

def plot_optimization(conv_csv: Path, judge_csv: Path, out: Path,
                      sfx: str, base_args: argparse.Namespace) -> None:
    """Phase 1 vs Phase 2 comparison plots."""
    phases: dict[int, pd.DataFrame] = {}
    for ph in (1, 2):
        a = copy.copy(base_args)
        a.phase = ph
        df_ph = load_data(conv_csv, judge_csv, a)
        if not df_ph.empty:
            df_ph["phase_label"] = f"Phase {ph}"
            phases[ph] = df_ph

    if len(phases) < 2:
        print("  --optimization: less than 2 phases found, skipping comparison plots.")
        return

    df_all = pd.concat(phases.values(), ignore_index=True)
    ph_col  = "phase_label"
    cfg_col = "config"
    ph_cm   = {"Phase 1": _PALETTE[0], "Phase 2": _PALETTE[1]}
    j_ok    = df_all.get("judge_status",        pd.Series("", index=df_all.index)) == "success"
    c_ok    = df_all.get("status_conversation", pd.Series("", index=df_all.index)) == "success"
    e_col   = (f"total_baseline_corrected_energy_wh"
               if sfx == "corrected" else "total_measured_energy_wh")

    specs = [
        ("score",                    "Quality Score: Phase 1 vs Phase 2",
         "Mean Score (1–10)", j_ok,  ".2f"),
        (e_col,                      "Energy: Phase 1 vs Phase 2",
         "Mean Energy (Wh)", c_ok,   ".2f"),
        (f"quality_per_joule_{sfx}", "Quality / Joule: Phase 1 vs Phase 2",
         "Quality / Joule", None,    ".2f"),
        ("total_latency_seconds",    "Latency: Phase 1 vs Phase 2",
         "Mean Latency (s)", c_ok,   ".1f"),
    ]

    for col, title, ylabel, ok_mask, val_fmt in specs:
        if col not in df_all.columns:
            continue
        agg = _agg(df_all, [cfg_col, ph_col], col, ok_mask)
        if agg.empty:
            continue
        fig, ax = plt.subplots(figsize=(12, 7))
        _grouped_bar(ax, agg, cfg_col, ph_col, title, ylabel, cmap=ph_cm, val_fmt=val_fmt)
        _apply_subtitle(fig, ax)
        fname = "opt_" + col.replace(" ", "_") + "_phase_comparison.png"
        _save(fig, out / fname)


# ─── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GREEN-IA: Generate matplotlib plots from experiment results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--conv-csv",         type=Path, default=DEFAULT_CONV_CSV)
    parser.add_argument("--judge-csv",        type=Path, default=DEFAULT_JUDGE_CSV)
    parser.add_argument("--output-dir",       type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--phase",            type=int,  choices=[1, 2], default=None)
    parser.add_argument("--primary-energy",   choices=["measured", "corrected"],
                        default="corrected",
                        help="Energy metric used in Pareto, rankings and efficiency plots")
    parser.add_argument("--hardware-profile", type=str, default=None,
                        metavar="PROFILE",
                        help="Filter: mac_m4 | windows_nvidia | ...")
    parser.add_argument("--execution-device", type=str, default=None,
                        metavar="DEVICE",
                        help="Filter: cpu | gpu")
    parser.add_argument("--category",         type=str, default=None,
                        help="Filter by MT-Bench category (e.g. math)")
    parser.add_argument("--model-name",       type=str, default=None,
                        help="Filter by model name (e.g. qwen2.5-7b)")
    parser.add_argument("--optimization",     action="store_true",
                        help="Generate Phase 1 vs Phase 2 comparison plots")
    args = parser.parse_args()

    _setup_style()

    O   = args.output_dir
    sfx = args.primary_energy
    W   = 62

    print(f"\n{'=' * W}")
    print(f"  GREEN-IA — analysis_plots.py")
    print(f"{'=' * W}")
    try:
        print(f"  Conv CSV   : {args.conv_csv.relative_to(ROOT)}")
        print(f"  Judge CSV  : {args.judge_csv.relative_to(ROOT)}")
        print(f"  Output     : {O.relative_to(ROOT)}/")
    except ValueError:
        pass
    print(f"  Phase      : {args.phase or 'all'}")
    print(f"  Energy     : {sfx}")
    for flag, val in (
        ("Hardware  ", args.hardware_profile),
        ("Device    ", args.execution_device),
        ("Category  ", args.category),
        ("Model     ", args.model_name),
    ):
        if val:
            print(f"  {flag}: {val}")
    print(f"{'─' * W}")

    df = load_data(args.conv_csv, args.judge_csv, args)
    if df.empty:
        print("  No data after filters — no plots generated.")
        return

    print(f"\n  [energy]")
    plot_energy(df, O / "energy", sfx)

    print(f"\n  [quality]")
    plot_quality(df, O / "quality")

    print(f"\n  [efficiency]")
    plot_efficiency(df, O / "efficiency", sfx)

    print(f"\n  [latency]")
    plot_latency(df, O / "latency")

    print(f"\n  [pareto]")
    plot_pareto_scatter(df, O / "pareto", sfx)

    print(f"\n  [boxplots]")
    plot_boxplots(df, O / "boxplots", sfx)

    print(f"\n  [heatmaps]")
    plot_heatmaps(df, O / "heatmaps", sfx)

    print(f"\n  [ranking]")
    plot_rankings(df, O / "ranking", sfx)

    if args.optimization:
        print(f"\n  [optimization comparison]")
        plot_optimization(args.conv_csv, args.judge_csv, O / "efficiency", sfx, args)

    print(f"\n{'=' * W}")
    print(f"  Plots saved to {O}/")
    print(f"{'=' * W}\n")


if __name__ == "__main__":
    main()

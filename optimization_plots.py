"""
optimization_plots.py
GREEN-IA — Maestria Ciencia de Datos
Universidad Comunero — Paraguay
Autor: Andres Villamayor

Genera graficas de la Fase 2: optimizacion de parametros de inferencia.

Lee:
  results/optimization/optimization_trials.csv
  results/optimization/optimization_report.json

Guarda en:
  results/plots/optimization/optimization_quality_vs_energy.png
  results/plots/optimization/optimization_pareto_front.png
  results/plots/optimization/optimization_energy_reduction.png
  results/plots/optimization/optimization_quality_constraint.png
  results/plots/optimization/optimization_latency_vs_energy.png
  results/plots/optimization/optimization_tokens_per_joule.png

Uso:
  python optimization_plots.py
  python optimization_plots.py --trials-csv path/to/trials.csv
  python optimization_plots.py --report-json path/to/report.json
  python optimization_plots.py --out-dir results/plots/optimization
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional

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

# ─── Paths ────────────────────────────────────────────────────────────────────

DEFAULT_TRIALS_CSV  = ROOT / "results" / "optimization" / "optimization_trials.csv"
DEFAULT_REPORT_JSON = ROOT / "results" / "optimization" / "optimization_report.json"
DEFAULT_PLOT_DIR    = ROOT / "results" / "plots" / "optimization"

# ─── Column names (from TRIALS_COLS in optimize_winner.py) ────────────────────

E_CORR  = "mean_energy_wh_per_1k_output_tokens_corrected"
E_MEAS  = "mean_energy_wh_per_1k_output_tokens_measured"
Q_COL   = "mean_quality_score"
Q_CI    = "ci95_quality_score"
LAT     = "mean_latency_seconds"
TPJ_C   = "mean_tokens_per_joule_corrected"
TPJ_M   = "mean_tokens_per_joule_measured"
EDP_C   = "mean_edp_joule_second_corrected"
EDP_M   = "mean_edp_joule_second_measured"
QPJ_C   = "quality_per_joule_corrected"
PARETO  = "pareto_efficient"
CONSTR  = "quality_constraint_passed"
E_RED   = "energy_reduction_percent"
LAT_RED = "latency_reduction_percent"
TPJ_IMP = "tokens_per_joule_improvement_percent"

_NUM_COLS = [
    Q_COL, Q_CI, "std_quality_score",
    E_CORR, E_MEAS, LAT, TPJ_C, TPJ_M,
    EDP_C, EDP_M, QPJ_C,
    E_RED, LAT_RED, TPJ_IMP,
    "edp_reduction_percent", "quality_per_joule_improvement_percent",
    "quality_drop_absolute", "quality_drop_relative_percent",
    "n_ctx", "max_tokens", "n_batch", "n_threads",
    "temperature", "top_p", "repeat_penalty", "top_k", "seed",
]

# ─── Color palette ────────────────────────────────────────────────────────────

_C_TRIAL  = "#4a9edd"   # steel-blue: Stage B trial (passes constraint)
_C_FAIL   = "#aaaaaa"   # gray: fails quality constraint
_C_BASE   = "#d62728"   # red: Phase 1 baseline
_C_WINNER = "#2ca02c"   # green: Phase 2 winner
_C_CONSTR = "#d62728"   # red: quality floor line
_GOLD     = "#ffd700"   # gold: Pareto-efficient edge


# ─── Style ────────────────────────────────────────────────────────────────────

def _setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi":        150,
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.size":         10,
        "axes.titlesize":    12,
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   8.5,
        "legend.framealpha": 0.88,
    })


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    try:
        print(f"  {path.relative_to(ROOT)}")
    except ValueError:
        print(f"  {path}")


# ─── Data loading ─────────────────────────────────────────────────────────────

def _to_bool(v: object) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if str(v).strip().lower() in ("true", "1", "yes"):
        return True
    if str(v).strip().lower() in ("false", "0", "no"):
        return False
    return None


def load_trials(path: Path) -> pd.DataFrame:
    """Load optimization_trials.csv; coerce numerics and booleans."""
    if not path.exists():
        print(f"  WARN: {path.name} not found — optimization plots will be empty.")
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str)
    for c in _NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in (PARETO, CONSTR):
        if c in df.columns:
            df[c] = df[c].apply(_to_bool)
    return df


def load_report(path: Path) -> dict:
    """Load optimization_report.json; return {} if absent."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ─── Column resolution helpers ────────────────────────────────────────────────

def _ecol(df: pd.DataFrame) -> Optional[str]:
    """Best available energy column (corrected preferred)."""
    for c in (E_CORR, E_MEAS):
        if c in df.columns and df[c].notna().any():
            return c
    return None


def _tpjcol(df: pd.DataFrame) -> Optional[str]:
    for c in (TPJ_C, TPJ_M):
        if c in df.columns and df[c].notna().any():
            return c
    return None


def _flt(v: object) -> Optional[float]:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


# ─── Shared drawing primitives ────────────────────────────────────────────────

def _draw_constraint_zone(ax: plt.Axes, min_q: float) -> None:
    """Draw quality floor dashed line and shade the forbidden zone below."""
    ax.axhspan(0, min_q, alpha=0.07, color=_C_CONSTR, zorder=1)
    ax.axhline(
        min_q, color=_C_CONSTR, lw=1.6, ls="--", zorder=3,
        label=f"Quality floor = {min_q:.3f}",
    )


def _draw_baseline_pt(
    ax: plt.Axes, bx: float, by: float, label: str = "Baseline (Phase 1)"
) -> None:
    ax.scatter(
        [bx], [by], marker="*", s=400, c=_C_BASE, zorder=10,
        edgecolors="white", linewidths=0.7, label=label,
    )


def _draw_winner_pt(
    ax: plt.Axes, wx: float, wy: float, label: str = "Winner (Phase 2)"
) -> None:
    ax.scatter(
        [wx], [wy], marker="*", s=400, c=_C_WINNER, zorder=10,
        edgecolors="white", linewidths=0.7, label=label,
    )


def _draw_pareto_line(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray) -> None:
    """Connect Pareto-efficient points (sorted by X) with a dashed line."""
    if len(xs) < 2:
        return
    order = np.argsort(xs)
    ax.plot(
        xs[order], ys[order], "k--", lw=1.2, alpha=0.45,
        zorder=4, label="Pareto frontier",
    )


def _annotate_pt(
    ax: plt.Axes, x: float, y: float, text: str, color: str,
    dx: float = 10, dy: float = 10,
) -> None:
    ax.annotate(
        text, xy=(x, y),
        xytext=(dx, dy), textcoords="offset points",
        fontsize=8, color=color, fontweight="bold",
        arrowprops=dict(
            arrowstyle="->", color=color, lw=0.9,
            shrinkA=5, shrinkB=5,
        ),
        bbox=dict(
            boxstyle="round,pad=0.3", fc="white",
            alpha=0.82, ec=color, lw=0.7,
        ),
        zorder=12,
    )


def _improvement_arrow(
    ax: plt.Axes, bx: float, by: float, wx: float, wy: float
) -> None:
    """Curved arrow from baseline to winner showing optimization direction."""
    ax.annotate(
        "",
        xy=(wx, wy), xytext=(bx, by),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#555555",
            lw=1.5,
            connectionstyle="arc3,rad=0.25",
            mutation_scale=14,
        ),
        zorder=8,
    )


# ─── Plot 1: Quality vs Energy (most important) ───────────────────────────────

def plot_quality_vs_energy(
    df: pd.DataFrame, report: dict, out: Path
) -> None:
    """
    Scatter: quality score (Y) vs energy Wh/1k output tokens (X).

    Encoding:
    - Gray downward triangles  = constraint-failing trials
    - Steel-blue circles       = constraint-passing trials
    - Gold-bordered circles    = Pareto-efficient, constraint-passing
    - Red star                 = Phase 1 baseline
    - Green star               = Phase 2 winner
    - Red dashed line + shading= quality constraint floor
    - Black dashed line        = Pareto frontier
    - Curved gray arrow        = optimization improvement direction
    """
    ecol = _ecol(df)
    bl   = report.get("baseline", {})
    qc   = report.get("quality_constraint", {})
    wn   = report.get("winner")

    sub = pd.DataFrame()
    if ecol and not df.empty:
        sub = df[df[ecol].notna() & df[Q_COL].notna()].copy()

    fig, ax = plt.subplots(figsize=(10, 7))

    min_q: Optional[float] = _flt(qc.get("min_quality_threshold"))
    if min_q is not None:
        _draw_constraint_zone(ax, min_q)

    if not sub.empty:
        pass_mask   = sub[CONSTR] == True    # noqa: E712
        fail_mask   = sub[CONSTR] == False   # noqa: E712
        unk_mask    = sub[CONSTR].isna()
        pareto_pass = pass_mask & (sub[PARETO] == True)   # noqa: E712
        plain_pass  = (pass_mask | unk_mask) & (sub[PARETO] != True)

        def _yerr(mask: pd.Series) -> Optional[np.ndarray]:
            if Q_CI in sub.columns:
                vals = sub.loc[mask, Q_CI].fillna(0).values
                return vals if vals.any() else None
            return None

        # 1. Failing trials (gray, muted, behind)
        if fail_mask.any():
            ye = _yerr(fail_mask)
            if ye is not None:
                ax.errorbar(
                    sub.loc[fail_mask, ecol], sub.loc[fail_mask, Q_COL],
                    yerr=ye, fmt="none", ecolor=_C_FAIL,
                    elinewidth=0.7, alpha=0.45, capsize=2,
                )
            ax.scatter(
                sub.loc[fail_mask, ecol], sub.loc[fail_mask, Q_COL],
                marker="v", s=55, c=_C_FAIL, alpha=0.42, zorder=3,
                label=f"Fails quality constraint  (n={fail_mask.sum()})",
            )

        # 2. Passing plain (steel-blue circles)
        if plain_pass.any():
            ye = _yerr(plain_pass)
            if ye is not None:
                ax.errorbar(
                    sub.loc[plain_pass, ecol], sub.loc[plain_pass, Q_COL],
                    yerr=ye, fmt="none", ecolor=_C_TRIAL,
                    elinewidth=0.8, alpha=0.5, capsize=2,
                )
            ax.scatter(
                sub.loc[plain_pass, ecol], sub.loc[plain_pass, Q_COL],
                marker="o", s=80, c=_C_TRIAL, alpha=0.78, zorder=4,
                edgecolors="white", linewidths=0.5,
                label=f"Passes constraint  (n={plain_pass.sum()})",
            )

        # 3. Pareto-efficient + passing (gold border, larger)
        if pareto_pass.any():
            ye = _yerr(pareto_pass)
            if ye is not None:
                ax.errorbar(
                    sub.loc[pareto_pass, ecol], sub.loc[pareto_pass, Q_COL],
                    yerr=ye, fmt="none", ecolor=_C_TRIAL,
                    elinewidth=0.8, alpha=0.5, capsize=2,
                )
            ax.scatter(
                sub.loc[pareto_pass, ecol], sub.loc[pareto_pass, Q_COL],
                marker="o", s=150, c=_C_TRIAL, alpha=0.92, zorder=5,
                edgecolors=_GOLD, linewidths=2.2,
                label=f"Pareto-efficient, passes constraint  (n={pareto_pass.sum()})",
            )

        # Pareto frontier connecting line (passing only)
        pareto_pts = sub[pareto_pass]
        if len(pareto_pts) >= 2:
            _draw_pareto_line(ax, pareto_pts[ecol].values, pareto_pts[Q_COL].values)

    # Baseline (Phase 1)
    bl_x = _flt(bl.get("mean_energy_wh_per_1k"))
    bl_y = _flt(bl.get("mean_quality_score"))
    if bl_x is not None and bl_y is not None:
        _draw_baseline_pt(ax, bl_x, bl_y)
        bl_pareto_str = str(bl.get("pareto_efficient", ""))
        bl_suffix = " [Pareto ★]" if bl_pareto_str.lower() == "true" else ""
        _annotate_pt(
            ax, bl_x, bl_y,
            f"Phase 1 baseline{bl_suffix}\nQ={bl_y:.3f}",
            _C_BASE, dx=10, dy=14,
        )

    # Winner (Phase 2)
    if wn:
        w_x = _flt(wn.get("mean_energy_wh_per_1k"))
        w_y = _flt(wn.get("mean_quality_score"))
        if w_x is not None and w_y is not None:
            _draw_winner_pt(ax, w_x, w_y)
            imp   = report.get("improvement", {})
            e_r   = _flt(imp.get("energy_reduction_percent"))
            q_dr  = _flt(imp.get("quality_drop_absolute"))
            parts = [f"Phase 2 winner\nQ={w_y:.3f}"]
            if e_r is not None:
                parts.append(f"Energy −{e_r:.1f}%")
            if q_dr is not None:
                parts.append(f"Quality −{q_dr:.3f} pts")
            _annotate_pt(
                ax, w_x, w_y, "\n".join(parts),
                _C_WINNER, dx=-90, dy=14,
            )
            # Curved improvement arrow from baseline to winner
            if bl_x is not None and bl_y is not None:
                _improvement_arrow(ax, bl_x, bl_y, w_x, w_y)

    ecol_lbl = "corrected" if ecol == E_CORR else "measured"
    ax.set_xlabel(f"Energy  (Wh / 1K output tokens, {ecol_lbl})", fontsize=11)
    ax.set_ylabel("Mean Quality Score  (1–10)", fontsize=11)
    ax.set_title(
        "Quality vs Energy — Parameter Optimization (Phase 2)\n"
        "Baseline highlighted · Quality constraint shown"
        " · Pareto-efficient configurations highlighted",
        fontsize=11, pad=12,
    )
    handles, lbls = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, lbls, loc="lower right", fontsize=8, ncol=1)
    _save(fig, out / "optimization_quality_vs_energy.png")


# ─── Plot 2: Pareto Front ─────────────────────────────────────────────────────

def plot_pareto_front(
    df: pd.DataFrame, report: dict, out: Path
) -> None:
    """
    Close-up Pareto front visualization with per-point parameter labels.

    All non-Pareto trials shown muted in the background.
    Pareto-efficient trials highlighted with gold borders and labeled.
    """
    ecol = _ecol(df)
    if ecol is None or df.empty:
        return
    sub = df[df[ecol].notna() & df[Q_COL].notna()].copy()
    if sub.empty:
        return

    bl = report.get("baseline", {})
    qc = report.get("quality_constraint", {})
    wn = report.get("winner")

    fig, ax = plt.subplots(figsize=(10, 7))

    min_q = _flt(qc.get("min_quality_threshold"))
    if min_q is not None:
        _draw_constraint_zone(ax, min_q)

    # All non-Pareto trials (muted background)
    non_pareto = sub[sub[PARETO] != True]   # noqa: E712
    if not non_pareto.empty:
        ax.scatter(
            non_pareto[ecol], non_pareto[Q_COL],
            marker="o", s=55, c="#cccccc", alpha=0.55, zorder=3,
            edgecolors="white", linewidths=0.4,
            label=f"Non-Pareto trials  (n={len(non_pareto)})",
        )

    # Pareto-efficient trials (prominent, gold-bordered)
    pareto_pts = sub[sub[PARETO] == True].copy()   # noqa: E712
    if not pareto_pts.empty:
        ax.scatter(
            pareto_pts[ecol], pareto_pts[Q_COL],
            marker="o", s=160, c=_C_TRIAL, alpha=0.93, zorder=6,
            edgecolors=_GOLD, linewidths=2.5,
            label=f"Pareto-efficient  (n={len(pareto_pts)})",
        )

        # Label each Pareto point with key inference parameters
        for _, row in pareto_pts.sort_values(ecol).iterrows():
            parts: list[str] = []
            for p in ("n_ctx", "n_batch", "n_threads", "max_tokens"):
                v = _flt(row.get(p))
                if v is not None:
                    parts.append(f"{p}={int(v)}")
            label_text = "\n".join(parts[:3])
            ax.annotate(
                label_text,
                xy=(row[ecol], row[Q_COL]),
                xytext=(6, 6), textcoords="offset points",
                fontsize=7, color="#333333",
                bbox=dict(
                    boxstyle="round,pad=0.2", fc="white",
                    alpha=0.65, ec="#cccccc", lw=0.5,
                ),
                zorder=9,
            )

        # Pareto frontier line
        if len(pareto_pts) >= 2:
            _draw_pareto_line(ax, pareto_pts[ecol].values, pareto_pts[Q_COL].values)

    # Baseline and winner
    bl_x = _flt(bl.get("mean_energy_wh_per_1k"))
    bl_y = _flt(bl.get("mean_quality_score"))
    if bl_x is not None and bl_y is not None:
        _draw_baseline_pt(ax, bl_x, bl_y)
        _annotate_pt(ax, bl_x, bl_y, "Baseline (Ph.1)", _C_BASE, dx=10, dy=12)

    if wn:
        w_x = _flt(wn.get("mean_energy_wh_per_1k"))
        w_y = _flt(wn.get("mean_quality_score"))
        if w_x is not None and w_y is not None:
            _draw_winner_pt(ax, w_x, w_y)
            _annotate_pt(ax, w_x, w_y, "Winner (Ph.2)", _C_WINNER, dx=-75, dy=12)

    ecol_lbl = "corrected" if ecol == E_CORR else "measured"
    ax.set_xlabel(f"Energy  (Wh / 1K output tokens, {ecol_lbl})", fontsize=11)
    ax.set_ylabel("Mean Quality Score  (1–10)", fontsize=11)
    ax.set_title(
        "Pareto Front — Quality vs Energy (Phase 2 Optimization)\n"
        "Gold border = Pareto-efficient · Labels = inference parameters",
        fontsize=11, pad=12,
    )
    ax.legend(loc="lower right", fontsize=8.5)
    _save(fig, out / "optimization_pareto_front.png")


# ─── Plot 3: Energy Reduction ─────────────────────────────────────────────────

def plot_energy_reduction(
    df: pd.DataFrame, report: dict, out: Path
) -> None:
    """
    Horizontal bar chart: energy_reduction_percent per trial (sorted descending).

    Positive = energy saved vs Phase 1 baseline.
    Negative = energy increased.
    Constraint-failing bars are hatched.
    Phase 2 winner is outlined with a dark border.
    """
    if df.empty or E_RED not in df.columns:
        return
    sub = df[df[E_RED].notna()].copy()
    if sub.empty:
        return

    wn    = report.get("winner") or {}
    w_tid = str(wn.get("trial_id", "")) if wn else ""
    sub   = sub.sort_values(E_RED, ascending=False).reset_index(drop=True)
    n     = len(sub)

    fig_h  = max(5, n * 0.38 + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    ys     = np.arange(n)
    labels = [f"T{r.get('trial_id', i)}" for i, (_, r) in enumerate(sub.iterrows())]
    vals   = sub[E_RED].values
    colors = [_C_WINNER if v > 0 else _C_BASE for v in vals]
    pass_v = sub[CONSTR].values if CONSTR in sub.columns else [None] * n

    bars = ax.barh(ys, vals, color=colors, alpha=0.80, edgecolor="none", height=0.7)

    # Hatch constraint-failing bars
    for bar, pv in zip(bars, pass_v):
        if pv is False:
            bar.set_hatch("///")
            bar.set_edgecolor("#777777")
            bar.set_linewidth(0.6)
            bar.set_alpha(0.55)

    # Outline winner and add label
    for bar, (_, row) in zip(bars, sub.iterrows()):
        tid = str(row.get("trial_id", ""))
        if tid == w_tid:
            bar.set_edgecolor("#333333")
            bar.set_linewidth(2.0)
            x_end = bar.get_width()
            ax.text(
                x_end + 0.15,
                bar.get_y() + bar.get_height() / 2,
                " ★ winner",
                va="center", ha="left", fontsize=8, color="#333333",
            )

    ax.axvline(0, color="#555555", lw=1.2, zorder=5)
    ax.invert_yaxis()
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(
        "Energy Reduction  (%, positive = saved vs Phase 1 baseline)", fontsize=10
    )
    ax.set_title(
        "Energy Reduction per Trial — Phase 2 Optimization\n"
        "Green = energy saved · Red = energy increased"
        " · Hatching = fails quality constraint",
        fontsize=11, pad=10,
    )
    legend_handles = [
        mpatches.Patch(color=_C_WINNER, alpha=0.80, label="Energy saved (reduction > 0)"),
        mpatches.Patch(color=_C_BASE,   alpha=0.80, label="Energy increased (reduction < 0)"),
        mpatches.Patch(
            facecolor="#aaaaaa", hatch="///", edgecolor="#777777",
            alpha=0.55, label="Fails quality constraint",
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
    _save(fig, out / "optimization_energy_reduction.png")


# ─── Plot 4: Quality Constraint ───────────────────────────────────────────────

def plot_quality_constraint(
    df: pd.DataFrame, report: dict, out: Path
) -> None:
    """
    Scatter: quality score per trial (sorted descending), colored by pass/fail.

    Green circles  = passes quality constraint.
    Gray triangles = fails quality constraint.
    CI95 error bars shown where available.
    Red dashed line = quality floor.
    Phase 1 baseline shown as dotted reference.
    Phase 2 winner annotated with a star label.
    """
    if df.empty or Q_COL not in df.columns:
        return
    sub = df[df[Q_COL].notna()].copy()
    if sub.empty:
        return

    qc    = report.get("quality_constraint", {})
    wn    = report.get("winner") or {}
    w_tid = str(wn.get("trial_id", "")) if wn else ""
    min_q = _flt(qc.get("min_quality_threshold"))

    sub = sub.sort_values(Q_COL, ascending=False).reset_index(drop=True)
    n   = len(sub)
    xs  = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(9, n * 0.5 + 2), 6))

    for i, (_, row) in enumerate(sub.iterrows()):
        pv     = row.get(CONSTR)
        q      = float(row[Q_COL])
        ci     = _flt(row.get(Q_CI)) or 0.0
        is_win = str(row.get("trial_id", "")) == w_tid
        color  = _C_WINNER if pv is True else _C_FAIL
        marker = "o"       if pv is True else "v"
        size   = 130       if is_win else 70
        edge   = "#333333" if is_win else "white"
        lw     = 2.2       if is_win else 0.5

        if ci > 0:
            ax.errorbar(
                i, q, yerr=ci,
                fmt="none", ecolor=color, elinewidth=0.9,
                capsize=3, alpha=0.6, zorder=3,
            )
        ax.scatter(
            i, q, marker=marker, s=size, c=color,
            edgecolors=edge, linewidths=lw,
            alpha=0.90, zorder=4,
        )
        if is_win:
            ax.annotate(
                f"★ Winner\nQ={q:.3f}",
                xy=(i, q), xytext=(8, 8), textcoords="offset points",
                fontsize=8, color=_C_WINNER, fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.25", fc="white",
                    alpha=0.82, ec=_C_WINNER, lw=0.7,
                ),
                zorder=10,
            )

    if min_q is not None:
        ax.axhline(min_q, color=_C_CONSTR, lw=1.6, ls="--", zorder=5,
                   label=f"Quality floor = {min_q:.3f}")
        ax.axhspan(0, min_q, alpha=0.07, color=_C_CONSTR, zorder=1)

    bl_q = _flt(report.get("baseline", {}).get("mean_quality_score"))
    if bl_q is not None:
        ax.axhline(
            bl_q, color=_C_BASE, lw=1.2, ls=":", alpha=0.7, zorder=4,
            label=f"Baseline quality = {bl_q:.3f}",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"T{r.get('trial_id', i)}" for i, (_, r) in enumerate(sub.iterrows())],
        rotation=60, ha="right", fontsize=8,
    )
    ax.set_xlabel("Trial  (sorted by quality, descending)", fontsize=10)
    ax.set_ylabel("Mean Quality Score  (1–10)", fontsize=10)
    ax.set_title(
        "Quality Constraint Gate — Phase 2 Optimization\n"
        "Green = passes P22 quality constraint · Gray = fails",
        fontsize=11, pad=10,
    )
    legend_handles = [
        mpatches.Patch(color=_C_WINNER, alpha=0.85, label="Passes quality constraint"),
        mpatches.Patch(color=_C_FAIL,   alpha=0.55, label="Fails quality constraint"),
    ]
    handles, _labels = ax.get_legend_handles_labels()
    ax.legend(handles=legend_handles + handles, fontsize=8.5, loc="upper right")
    _save(fig, out / "optimization_quality_constraint.png")


# ─── Plot 5: Latency vs Energy ────────────────────────────────────────────────

def plot_latency_vs_energy(
    df: pd.DataFrame, report: dict, out: Path
) -> None:
    """
    Scatter: mean latency (X) vs energy Wh/1k tokens (Y) for all Stage B trials.

    Shows the latency-energy tradeoff across the inference parameter search space.
    Pareto-efficient configurations are highlighted with gold borders.
    """
    ecol = _ecol(df)
    if ecol is None or df.empty:
        return
    sub = df[df[ecol].notna() & df[LAT].notna()].copy()
    if sub.empty:
        return

    bl = report.get("baseline", {})
    wn = report.get("winner")

    fig, ax = plt.subplots(figsize=(9, 6))

    pass_mask   = sub[CONSTR] == True    # noqa: E712
    fail_mask   = sub[CONSTR] == False   # noqa: E712
    unk_mask    = sub[CONSTR].isna()
    pareto_mask = (sub[PARETO] == True) if PARETO in sub.columns else pd.Series(False, index=sub.index)  # noqa: E712

    plain = (pass_mask | unk_mask) & ~pareto_mask
    if plain.any():
        ax.scatter(
            sub.loc[plain, LAT], sub.loc[plain, ecol],
            marker="o", s=70, c=_C_TRIAL, alpha=0.72, zorder=4,
            edgecolors="white", linewidths=0.5,
            label=f"Passes constraint  (n={plain.sum()})",
        )

    if fail_mask.any():
        ax.scatter(
            sub.loc[fail_mask, LAT], sub.loc[fail_mask, ecol],
            marker="v", s=55, c=_C_FAIL, alpha=0.42, zorder=3,
            label=f"Fails constraint  (n={fail_mask.sum()})",
        )

    p_pass = pareto_mask & (pass_mask | unk_mask)
    if p_pass.any():
        ax.scatter(
            sub.loc[p_pass, LAT], sub.loc[p_pass, ecol],
            marker="o", s=145, c=_C_TRIAL, alpha=0.92, zorder=5,
            edgecolors=_GOLD, linewidths=2.2,
            label=f"Pareto-efficient  (n={p_pass.sum()})",
        )

    # Baseline
    bl_lat = _flt(bl.get("mean_latency_s"))
    bl_e   = _flt(bl.get("mean_energy_wh_per_1k"))
    if bl_lat is not None and bl_e is not None:
        _draw_baseline_pt(ax, bl_lat, bl_e)
        _annotate_pt(ax, bl_lat, bl_e, "Baseline (Ph.1)", _C_BASE, dx=8, dy=12)

    # Winner
    if wn:
        w_lat = _flt(wn.get("mean_latency_s"))
        w_e   = _flt(wn.get("mean_energy_wh_per_1k"))
        if w_lat is not None and w_e is not None:
            _draw_winner_pt(ax, w_lat, w_e)
            _annotate_pt(ax, w_lat, w_e, "Winner (Ph.2)", _C_WINNER, dx=-75, dy=12)

    ax.annotate(
        "← Ideal direction\n(lower latency & lower energy)",
        xy=(0.02, 0.05), xycoords="axes fraction",
        fontsize=8, color="#555555", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.72, ec="#cccccc"),
    )

    ecol_lbl = "corrected" if ecol == E_CORR else "measured"
    ax.set_xlabel("Mean Latency  (seconds)", fontsize=11)
    ax.set_ylabel(f"Energy  (Wh / 1K output tokens, {ecol_lbl})", fontsize=11)
    ax.set_title(
        "Latency vs Energy — Phase 2 Optimization\n"
        "Ideal direction: lower-left · Gold border = Pareto-efficient",
        fontsize=11, pad=10,
    )
    ax.legend(loc="upper right", fontsize=8.5)
    _save(fig, out / "optimization_latency_vs_energy.png")


# ─── Plot 6: Tokens per Joule ─────────────────────────────────────────────────

def plot_tokens_per_joule(
    df: pd.DataFrame, report: dict, out: Path
) -> None:
    """
    Horizontal bar chart: tokens per Joule per trial (sorted descending).

    Phase 1 baseline is shown as a vertical reference line.
    Phase 2 winner bar is highlighted in green with improvement % annotated.
    Higher values = more energy-efficient inference.
    """
    tpjcol = _tpjcol(df)
    if tpjcol is None or df.empty:
        return
    sub = df[df[tpjcol].notna()].copy()
    if sub.empty:
        return

    bl    = report.get("baseline", {})
    wn    = report.get("winner") or {}
    w_tid = str(wn.get("trial_id", "")) if wn else ""
    bl_tpj = _flt(bl.get("mean_tokens_per_joule"))

    sub    = sub.sort_values(tpjcol, ascending=False).reset_index(drop=True)
    n      = len(sub)
    fig_h  = max(5, n * 0.38 + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    ys     = np.arange(n)
    labels = [f"T{r.get('trial_id', i)}" for i, (_, r) in enumerate(sub.iterrows())]
    vals   = sub[tpjcol].values
    pass_v = sub[CONSTR].values if CONSTR in sub.columns else [None] * n

    colors = []
    for _, row in sub.iterrows():
        tid = str(row.get("trial_id", ""))
        if tid == w_tid:
            colors.append(_C_WINNER)
        elif row.get(CONSTR) is False:
            colors.append(_C_FAIL)
        else:
            colors.append(_C_TRIAL)

    bars = ax.barh(ys, vals, color=colors, alpha=0.80, edgecolor="none", height=0.7)

    for bar, pv in zip(bars, pass_v):
        if pv is False:
            bar.set_hatch("///")
            bar.set_edgecolor("#777777")
            bar.set_linewidth(0.6)
            bar.set_alpha(0.55)

    imp_tpj = _flt(report.get("improvement", {}).get("tokens_per_joule_improvement_percent"))
    for bar, (_, row) in zip(bars, sub.iterrows()):
        tid = str(row.get("trial_id", ""))
        if tid == w_tid:
            bar.set_edgecolor("#333333")
            bar.set_linewidth(2.0)
            suffix = " ★ winner"
            if imp_tpj is not None:
                suffix += f"  (+{imp_tpj:.1f}%)"
            ax.text(
                bar.get_width() + max(vals) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                suffix,
                va="center", ha="left", fontsize=8, color="#333333",
            )

    if bl_tpj is not None:
        ax.axvline(
            bl_tpj, color=_C_BASE, lw=1.6, ls="--", zorder=5,
            label=f"Baseline tokens/J = {bl_tpj:.2f}",
        )

    ax.invert_yaxis()
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8)
    tpj_lbl = "corrected" if tpjcol == TPJ_C else "measured"
    ax.set_xlabel(f"Tokens per Joule  ({tpj_lbl})", fontsize=10)
    ax.set_title(
        "Tokens per Joule — Phase 2 Optimization\n"
        "Higher = more energy-efficient · Dashed line = Phase 1 baseline",
        fontsize=11, pad=10,
    )
    legend_handles = [
        mpatches.Patch(color=_C_WINNER, alpha=0.85, label="Phase 2 winner"),
        mpatches.Patch(color=_C_TRIAL,  alpha=0.80, label="Passes quality constraint"),
        mpatches.Patch(
            facecolor=_C_FAIL, hatch="///", edgecolor="#777777",
            alpha=0.55, label="Fails quality constraint",
        ),
    ]
    handles, _labels = ax.get_legend_handles_labels()
    ax.legend(handles=legend_handles + handles, fontsize=8.5, loc="lower right")
    _save(fig, out / "optimization_tokens_per_joule.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GREEN-IA Fase 2: Genera graficas de optimizacion de parametros.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--trials-csv", type=Path, default=DEFAULT_TRIALS_CSV,
        help="optimization_trials.csv producido por optimize_winner.py",
    )
    p.add_argument(
        "--report-json", type=Path, default=DEFAULT_REPORT_JSON,
        help="optimization_report.json producido por optimize_winner.py",
    )
    p.add_argument(
        "--out-dir", type=Path, default=DEFAULT_PLOT_DIR,
        help="Directorio de salida para las graficas",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _setup_style()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("GREEN-IA — Optimization Plots (Phase 2)")
    print("=" * 62)

    print("\n[1/2] Cargando datos...")
    df     = load_trials(args.trials_csv)
    report = load_report(args.report_json)

    if df.empty:
        print("  Sin datos de trials. Graficas generadas con datos vacios.")
    else:
        n_trials = len(df)
        n_pass   = int((df[CONSTR] == True).sum()) if CONSTR in df.columns else "?"   # noqa: E712
        n_pareto = int((df[PARETO] == True).sum()) if PARETO in df.columns else "?"   # noqa: E712
        print(
            f"  {n_trials} trials · "
            f"{n_pass} pasan constraint · "
            f"{n_pareto} Pareto-eficientes"
        )

    bl_q  = _flt(report.get("baseline", {}).get("mean_quality_score"))
    min_q = _flt(report.get("quality_constraint", {}).get("min_quality_threshold"))
    wn    = report.get("winner")

    if bl_q is not None:
        print(f"  Baseline quality   : {bl_q:.3f}")
    if min_q is not None:
        print(f"  Quality floor (P22): {min_q:.3f}")
    if wn:
        w_q = _flt(wn.get("mean_quality_score"))
        print(f"  Winner trial T{wn.get('trial_id')}  Q={w_q:.3f}" if w_q else f"  Winner: trial T{wn.get('trial_id')}")

    print(f"\n[2/2] Generando graficas en {out}...")

    plots = [
        ("Quality vs Energy   ", plot_quality_vs_energy,   "optimization_quality_vs_energy.png"),
        ("Pareto Front        ", plot_pareto_front,         "optimization_pareto_front.png"),
        ("Energy Reduction    ", plot_energy_reduction,     "optimization_energy_reduction.png"),
        ("Quality Constraint  ", plot_quality_constraint,   "optimization_quality_constraint.png"),
        ("Latency vs Energy   ", plot_latency_vs_energy,    "optimization_latency_vs_energy.png"),
        ("Tokens per Joule    ", plot_tokens_per_joule,     "optimization_tokens_per_joule.png"),
    ]

    generated = 0
    for name, fn, fname in plots:
        try:
            fn(df, report, out)
            if (out / fname).exists():
                generated += 1
            else:
                print(f"  INFO: '{name.strip()}' sin datos suficientes — omitida.")
        except Exception as exc:
            print(f"  WARN: '{name.strip()}' falló: {exc}")

    print(f"\n{'=' * 62}")
    print(f"  {generated}/{len(plots)} graficas generadas.")
    print(f"  Salida: {out}")
    print("=" * 62)


if __name__ == "__main__":
    main()

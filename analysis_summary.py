"""
analysis_summary.py
GREEN-IA — Maestria Ciencia de Datos
Universidad Comunero — Paraguay
Autor: Andres Villamayor

Resumen estadistico completo del experimento de energia y calidad.

Lee:
  results/raw/conversation_results.csv   (energia, latencia, tokens — por conv)
  results/judge/judge_results.csv        (puntajes de calidad — por conv)

Une por: conversation_id

Genera 14 archivos CSV en results/summary/:
  summary_by_model.csv
  summary_by_model_quantization.csv
  summary_by_model_quantization_device.csv
  summary_by_hardware_profile.csv
  summary_by_execution_device.csv
  summary_by_category.csv
  summary_by_model_category.csv
  summary_full_ranking.csv
  best_efficiency_by_category.csv
  statistical_tests.csv
  paper_table_energy_quality_by_configuration.csv
  paper_table_by_category.csv
  paper_table_best_by_category.csv
  paper_table_pareto_efficient.csv

Metricas de eficiencia calidad-energia:
  quality_per_joule_measured / corrected
  quality_per_wh_measured / corrected
  quality_per_1k_token_wh_measured / corrected

Estas son adaptaciones operacionales de la metrica Accuracy-per-Joule
(Canziani et al. 2017, arXiv:1605.07678) aplicada a evaluacion abierta de LLMs.
En lugar de accuracy de clasificacion, se usa un puntaje de calidad proxy
(Claude-based MT-Bench-style quality score, escala 1-10). La interpretacion es
"cuanta calidad se obtiene por unidad de energia consumida" y es directamente
comparable entre configuraciones del mismo experimento. No comparar entre
plataformas de hardware distintas ya que el metodo de medicion (CodeCarbon
TDP-based) produce valores absolutos no equivalentes entre ellas.

Uso:
  python analysis_summary.py                         # todas las fases
  python analysis_summary.py --phase 1               # solo fase 1
  python analysis_summary.py --phase 2               # solo fase 2
  python analysis_summary.py --primary-energy corrected
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

try:
    from scipy import stats as _sp_stats
    _SCIPY = True
except ImportError:
    _sp_stats = None
    _SCIPY = False

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─── Rutas ────────────────────────────────────────────────────────────────────

DEFAULT_CONV_CSV  = ROOT / "results" / "raw"   / "conversation_results.csv"
DEFAULT_JUDGE_CSV = ROOT / "results" / "judge" / "judge_results.csv"
DEFAULT_OUT_DIR   = ROOT / "results" / "summary"

# ─── Columnas judge-only (no duplicadas en el CSV de conversaciones) ──────────

_JUDGE_ONLY_COLS = [
    "score", "correctness", "instruction_following", "relevance",
    "completeness", "clarity", "conciseness", "usefulness",
    "multi_turn_coherence", "main_strengths", "main_weaknesses",
    "final_comment", "judge_model", "judge_timestamp", "judge_status",
    "judge_error_message", "judge_prompt_mode", "reference_guided",
]

_ENERGY_COLS = [
    "total_measured_energy_wh",
    "total_measured_energy_joules",
    "total_baseline_corrected_energy_wh",
    "total_baseline_corrected_energy_joules",
    "total_energy_j_per_output_token_measured",
    "total_energy_wh_per_1k_output_tokens_measured",
    "total_tokens_per_joule_measured",
    "total_energy_j_per_output_token_corrected",
    "total_energy_wh_per_1k_output_tokens_corrected",
    "total_tokens_per_joule_corrected",
    "total_latency_seconds",
    "total_edp_joule_second_measured",
    "total_edp_joule_second_corrected",
    "total_operational_sci_per_conversation",
    "total_operational_sci_per_1k_output_tokens",
]

_QUALITY_SUBDIMS = [
    "correctness", "instruction_following", "relevance", "completeness",
    "clarity", "conciseness", "usefulness", "multi_turn_coherence",
]


# ─── Estadisticas auxiliares ──────────────────────────────────────────────────

def _ci95(s: pd.Series) -> float:
    """95 % CI half-width. Uses t-distribution when scipy is available."""
    vals = s.dropna()
    n = len(vals)
    if n < 2:
        return np.nan
    se = float(vals.std(ddof=1)) / np.sqrt(n)
    t_val = _sp_stats.t.ppf(0.975, df=n - 1) if _SCIPY else 1.96
    return float(t_val * se)


def _iqr(s: pd.Series) -> float:
    vals = s.dropna()
    if len(vals) < 4:
        return np.nan
    return float(vals.quantile(0.75) - vals.quantile(0.25))


def _safe_div(num: Any, denom: Any) -> float:
    try:
        n, d = float(num), float(denom)
        if d == 0.0 or np.isnan(d) or np.isnan(n):
            return np.nan
        r = n / d
        return np.nan if np.isinf(r) else r
    except (TypeError, ValueError):
        return np.nan


# ─── Agregacion principal ─────────────────────────────────────────────────────

def aggregate_group(grp: pd.DataFrame) -> dict[str, Any]:
    """
    All aggregated metrics for a subgroup of conversations.

    Quality metrics use rows where judge_status == 'ok'.
    Energy metrics  use rows where status_conversation == 'ok'.
    Efficiency metrics require both to be 'ok'.

    quality_per_joule_* / quality_per_wh_* are operational adaptations of
    Accuracy-per-Joule (Canziani et al. 2017, arXiv:1605.07678) for open-ended
    LLM quality evaluation. The quality score (1-10 proxy) replaces classification
    accuracy; energy is CodeCarbon TDP-based estimation.
    """
    conv_ok  = grp.get("status_conversation", pd.Series(dtype=str)).isin({"ok"})
    judge_ok = grp.get("judge_status",        pd.Series(dtype=str)).isin({"ok"})
    both_ok  = conv_ok & judge_ok

    n_total   = len(grp)
    n_success = int(both_ok.sum())

    r: dict[str, Any] = {}
    r["n_conversations"] = n_total
    r["n_success"]       = n_success
    r["n_errors"]        = n_total - n_success
    r["success_rate"]    = round(n_success / n_total, 4) if n_total > 0 else np.nan

    # ── quality ───────────────────────────────────────────────────────────────
    qdf = grp[judge_ok]
    if not qdf.empty and "score" in qdf.columns:
        s = pd.to_numeric(qdf["score"], errors="coerce")
        r["mean_quality_score"]   = float(s.mean())
        r["std_quality_score"]    = float(s.std(ddof=1))
        r["ci95_quality_score"]   = _ci95(s)
        r["cv_quality_score"]     = _safe_div(s.std(ddof=1) * 100.0, s.mean())
        r["median_quality_score"] = float(s.median())
        r["iqr_quality_score"]    = _iqr(s)
    else:
        for k in ("mean_quality_score", "std_quality_score", "ci95_quality_score",
                  "cv_quality_score", "median_quality_score", "iqr_quality_score"):
            r[k] = np.nan

    for dim in _QUALITY_SUBDIMS:
        r[f"mean_{dim}"] = (
            float(pd.to_numeric(qdf[dim], errors="coerce").mean())
            if dim in qdf.columns
            else np.nan
        )

    # ── energy + latency ──────────────────────────────────────────────────────
    edf = grp[conv_ok]

    def _em(col: str) -> float:
        return float(pd.to_numeric(edf[col], errors="coerce").mean()) if col in edf.columns else np.nan

    def _es(col: str) -> float:
        return float(pd.to_numeric(edf[col], errors="coerce").std(ddof=1)) if col in edf.columns else np.nan

    def _ec(col: str) -> float:
        return _ci95(pd.to_numeric(edf[col], errors="coerce")) if col in edf.columns else np.nan

    r["mean_total_measured_energy_wh"]                       = _em("total_measured_energy_wh")
    r["std_total_measured_energy_wh"]                        = _es("total_measured_energy_wh")
    r["ci95_total_measured_energy_wh"]                       = _ec("total_measured_energy_wh")
    r["mean_total_measured_energy_joules"]                   = _em("total_measured_energy_joules")

    r["mean_total_baseline_corrected_energy_wh"]             = _em("total_baseline_corrected_energy_wh")
    r["std_total_baseline_corrected_energy_wh"]              = _es("total_baseline_corrected_energy_wh")
    r["ci95_total_baseline_corrected_energy_wh"]             = _ec("total_baseline_corrected_energy_wh")
    r["mean_total_baseline_corrected_energy_joules"]         = _em("total_baseline_corrected_energy_joules")

    r["mean_total_energy_j_per_output_token_measured"]       = _em("total_energy_j_per_output_token_measured")
    r["mean_total_energy_wh_per_1k_output_tokens_measured"]  = _em("total_energy_wh_per_1k_output_tokens_measured")
    r["mean_total_tokens_per_joule_measured"]                = _em("total_tokens_per_joule_measured")
    r["mean_total_energy_j_per_output_token_corrected"]      = _em("total_energy_j_per_output_token_corrected")
    r["mean_total_energy_wh_per_1k_output_tokens_corrected"] = _em("total_energy_wh_per_1k_output_tokens_corrected")
    r["mean_total_tokens_per_joule_corrected"]               = _em("total_tokens_per_joule_corrected")

    r["mean_total_latency_seconds"]                          = _em("total_latency_seconds")
    r["std_total_latency_seconds"]                           = _es("total_latency_seconds")
    r["ci95_total_latency_seconds"]                          = _ec("total_latency_seconds")

    r["mean_total_edp_joule_second_measured"]                = _em("total_edp_joule_second_measured")
    r["mean_total_edp_joule_second_corrected"]               = _em("total_edp_joule_second_corrected")
    r["mean_total_operational_sci_per_conversation"]         = _em("total_operational_sci_per_conversation")
    r["mean_total_operational_sci_per_1k_output_tokens"]     = _em("total_operational_sci_per_1k_output_tokens")

    # ── quality-per-energy efficiency ─────────────────────────────────────────
    # Operational adaptation of Accuracy-per-Joule (Canziani et al. 2017,
    # arXiv:1605.07678) for open-ended LLM evaluation.
    #
    # Original metric (classification): accuracy / energy_joules
    # This adaptation replaces classification accuracy with the Claude-based
    # MT-Bench-style quality score (1-10 proxy), measured at the conversation
    # level (T1 + T2 aggregated). Energy is CodeCarbon TDP-based estimation.
    #
    # Formulas:
    #   quality_per_joule_measured  = mean_quality_score / mean_total_measured_energy_joules
    #   quality_per_joule_corrected = mean_quality_score / mean_total_baseline_corrected_energy_joules
    #   quality_per_wh_measured     = mean_quality_score / mean_total_measured_energy_wh
    #   quality_per_wh_corrected    = mean_quality_score / mean_total_baseline_corrected_energy_wh
    #   quality_per_1k_token_wh_measured  = mean_quality_score / mean_total_energy_wh_per_1k_output_tokens_measured
    #   quality_per_1k_token_wh_corrected = mean_quality_score / mean_total_energy_wh_per_1k_output_tokens_corrected
    #
    # Interpretation: higher = more quality per unit of energy.
    # Use quality_per_joule_corrected for the primary comparison; it removes
    # system idle power (P10) and better isolates inference energy.
    # Do NOT compare across hardware platforms: CodeCarbon TDP-based values are
    # not equivalent between Apple Silicon (ANE estimation) and NVIDIA GPU.
    q = r.get("mean_quality_score", np.nan)

    r["quality_per_joule_measured"]        = _safe_div(q, r["mean_total_measured_energy_joules"])
    r["quality_per_joule_corrected"]       = _safe_div(q, r["mean_total_baseline_corrected_energy_joules"])
    r["quality_per_wh_measured"]           = _safe_div(q, r["mean_total_measured_energy_wh"])
    r["quality_per_wh_corrected"]          = _safe_div(q, r["mean_total_baseline_corrected_energy_wh"])
    r["quality_per_1k_token_wh_measured"]  = _safe_div(q, r["mean_total_energy_wh_per_1k_output_tokens_measured"])
    r["quality_per_1k_token_wh_corrected"] = _safe_div(q, r["mean_total_energy_wh_per_1k_output_tokens_corrected"])

    return r


# ─── Agrupacion y guardado ────────────────────────────────────────────────────

def _groupby_save(
    df:          pd.DataFrame,
    group_keys:  list[str],
    output_path: Path,
    sort_by:     str = "mean_quality_score",
) -> int:
    valid_keys = [k for k in group_keys if k in df.columns]
    if not valid_keys:
        return 0

    records: list[dict] = []
    for key_vals, grp in df.groupby(valid_keys, dropna=False):
        row: dict[str, Any] = {}
        if isinstance(key_vals, tuple):
            for k, v in zip(valid_keys, key_vals):
                row[k] = v
        else:
            row[valid_keys[0]] = key_vals
        row.update(aggregate_group(grp))
        records.append(row)

    if not records:
        return 0

    result = pd.DataFrame(records)
    if sort_by in result.columns:
        result = result.sort_values(sort_by, ascending=False, na_position="last")

    result.to_csv(output_path, index=False, float_format="%.6g")
    return len(result)


# ─── Tests estadisticos ───────────────────────────────────────────────────────

def _cohen_d(a: pd.Series, b: pd.Series) -> float:
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2.0)
    return _safe_div(a.mean() - b.mean(), pooled)


def _holm_correction(p_values: list[float]) -> list[float]:
    """Holm step-down correction. Returns adjusted p-values in the same order."""
    n = len(p_values)
    if n == 0:
        return []
    valid = [(i, float(p)) for i, p in enumerate(p_values) if not np.isnan(float(p))]
    result: list[float] = [np.nan] * n
    if not valid:
        return result
    m = len(valid)
    running_max = 0.0
    for rank, (orig_i, p) in enumerate(sorted(valid, key=lambda x: x[1])):
        running_max = max(running_max, p * (m - rank))
        result[orig_i] = min(running_max, 1.0)
    return result


def _bootstrap_ci_mean(values: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    """Bootstrap 95% percentile CI for the mean."""
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(42)
    boot = np.fromiter(
        (rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)),
        dtype=float, count=n_boot,
    )
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def compute_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Statistical tests comparing quantization configurations.

    Tests (all optional — graceful degradation when scipy is unavailable):
      1. Mann-Whitney U (or Welch's t fallback): Q4 vs Q8 independent samples
      2. Wilcoxon signed-rank: Q4 vs Q8 paired by (official_question_id, repetition)
      3. Friedman: across 3+ quantization levels repeated per question
      4. Bootstrap 95% CI: quality_per_joule (measured + corrected)

    Holm step-down correction applied across all p-values in the final output.
    Extra columns: holm_corrected_p_value, significant_at_05, significant_at_05_holm.
    """
    metrics = [
        ("score",                              "quality_score"),
        ("total_measured_energy_wh",           "measured_energy_wh"),
        ("total_baseline_corrected_energy_wh", "corrected_energy_wh"),
        ("total_latency_seconds",              "latency_seconds"),
    ]
    gkeys = [k for k in ("model_name", "hardware_profile", "execution_device") if k in df.columns]
    records: list[dict] = []

    if not _SCIPY:
        warnings.warn(
            "scipy not available — Mann-Whitney U, Wilcoxon, and Friedman tests skipped. "
            "Bootstrap CI is still computed. Install scipy: pip install scipy",
            UserWarning, stacklevel=2,
        )

    match_cols = [c for c in ("official_question_id", "repetition") if c in df.columns]

    for key_vals, grp in df.groupby(gkeys, dropna=False):
        ginfo: dict[str, Any] = dict(
            zip(gkeys, key_vals if isinstance(key_vals, tuple) else (key_vals,))
        )

        if "quantization" not in grp.columns:
            continue

        df_q4 = grp[grp["quantization"] == "q4"]
        df_q8 = grp[grp["quantization"] == "q8"]

        # ── 1. Mann-Whitney U — Q4 vs Q8, independent samples ───────────────────
        if len(df_q4) >= 3 and len(df_q8) >= 3:
            for src_col, label in metrics:
                if src_col not in grp.columns:
                    continue
                v4 = pd.to_numeric(df_q4[src_col], errors="coerce").dropna()
                v8 = pd.to_numeric(df_q8[src_col], errors="coerce").dropna()
                if len(v4) < 3 or len(v8) < 3:
                    continue

                row: dict[str, Any] = {
                    **ginfo,
                    "metric": label, "n_q4": len(v4), "n_q8": len(v8),
                    "mean_q4": float(v4.mean()), "mean_q8": float(v8.mean()),
                    "median_q4": float(v4.median()), "median_q8": float(v8.median()),
                    "n_pairs": np.nan,
                    "bootstrap_ci_lower": np.nan, "bootstrap_ci_upper": np.nan,
                }
                if _SCIPY:
                    try:
                        stat, pval = _sp_stats.mannwhitneyu(v4, v8, alternative="two-sided")
                        row["test_type"] = "mann_whitney_u"
                    except Exception:
                        stat, pval = _sp_stats.ttest_ind(v4, v8, equal_var=False)
                        row["test_type"] = "welch_t_fallback"
                    row["statistic"] = float(stat)
                    row["p_value"]   = float(pval)
                else:
                    row["test_type"] = "no_scipy"
                    row["statistic"] = np.nan
                    row["p_value"]   = np.nan
                row["effect_size_cohen_d"] = _cohen_d(v4, v8)
                records.append(row)

        # ── 2. Wilcoxon signed-rank — Q4 vs Q8 paired by question ───────────────
        if match_cols and len(df_q4) >= 3 and len(df_q8) >= 3 and _SCIPY:
            for src_col, label in metrics:
                if src_col not in grp.columns:
                    continue
                try:
                    paired = df_q4[match_cols + [src_col]].merge(
                        df_q8[match_cols + [src_col]],
                        on=match_cols, suffixes=("_q4", "_q8"),
                    )
                except Exception:
                    continue
                v4p = pd.to_numeric(paired[f"{src_col}_q4"], errors="coerce")
                v8p = pd.to_numeric(paired[f"{src_col}_q8"], errors="coerce")
                mask = v4p.notna() & v8p.notna()
                v4p_arr = v4p[mask].to_numpy()
                v8p_arr = v8p[mask].to_numpy()
                if len(v4p_arr) < 5:
                    continue
                diffs = v4p_arr - v8p_arr
                if np.all(diffs == 0):
                    continue

                row = {
                    **ginfo,
                    "metric": label, "n_q4": len(df_q4), "n_q8": len(df_q8),
                    "mean_q4": float(v4p_arr.mean()), "mean_q8": float(v8p_arr.mean()),
                    "median_q4": float(np.median(v4p_arr)),
                    "median_q8": float(np.median(v8p_arr)),
                    "n_pairs": len(v4p_arr),
                    "bootstrap_ci_lower": np.nan, "bootstrap_ci_upper": np.nan,
                }
                try:
                    stat, pval = _sp_stats.wilcoxon(diffs, alternative="two-sided")
                    row["test_type"] = "wilcoxon_signed_rank"
                    row["statistic"] = float(stat)
                    row["p_value"]   = float(pval)
                except Exception:
                    row["test_type"] = "wilcoxon_error"
                    row["statistic"] = np.nan
                    row["p_value"]   = np.nan
                row["effect_size_cohen_d"] = _cohen_d(pd.Series(v4p_arr), pd.Series(v8p_arr))
                records.append(row)

        # ── 3. Friedman — 3+ quantization levels repeated per question ───────────
        if match_cols and _SCIPY:
            quant_levels = sorted(grp["quantization"].dropna().unique().tolist())
            if len(quant_levels) >= 3:
                for src_col, label in metrics:
                    if src_col not in grp.columns:
                        continue
                    try:
                        base = (
                            grp[grp["quantization"] == quant_levels[0]][match_cols + [src_col]]
                            .rename(columns={src_col: "_v0"})
                        )
                        for qi, ql in enumerate(quant_levels[1:], 1):
                            part = (
                                grp[grp["quantization"] == ql][match_cols + [src_col]]
                                .rename(columns={src_col: f"_v{qi}"})
                            )
                            base = base.merge(part, on=match_cols)
                        v_cols = [f"_v{i}" for i in range(len(quant_levels))]
                        mat = base[v_cols].apply(pd.to_numeric, errors="coerce").dropna()
                        if len(mat) < 5:
                            continue
                        stat, pval = _sp_stats.friedmanchisquare(
                            *[mat[c].to_numpy() for c in v_cols]
                        )
                        records.append({
                            **ginfo,
                            "metric": label, "test_type": "friedman",
                            "statistic": float(stat), "p_value": float(pval),
                            "n_q4": np.nan, "n_q8": np.nan,
                            "mean_q4": np.nan, "mean_q8": np.nan,
                            "median_q4": np.nan, "median_q8": np.nan,
                            "n_pairs": len(mat), "n_quant_groups": len(quant_levels),
                            "effect_size_cohen_d": np.nan,
                            "bootstrap_ci_lower": np.nan, "bootstrap_ci_upper": np.nan,
                        })
                    except Exception:
                        pass

        # ── 4. Bootstrap CI — quality_per_joule (measured + corrected) ───────────
        judge_ok = grp.get("judge_status",        pd.Series(dtype=str)).isin({"ok"})
        conv_ok  = grp.get("status_conversation", pd.Series(dtype=str)).isin({"ok"})
        both_ok  = judge_ok & conv_ok

        for e_col, suffix in (
            ("total_measured_energy_joules",           "measured"),
            ("total_baseline_corrected_energy_joules", "corrected"),
        ):
            if "score" not in grp.columns or e_col not in grp.columns:
                continue
            sub  = grp[both_ok]
            qs   = pd.to_numeric(sub["score"], errors="coerce")
            es   = pd.to_numeric(sub[e_col],   errors="coerce")
            mask = qs.notna() & es.notna() & (es > 0)
            qpj  = (qs[mask] / es[mask]).to_numpy()
            if len(qpj) < 3:
                continue
            ci_lo, ci_hi = _bootstrap_ci_mean(qpj)
            records.append({
                **ginfo,
                "metric": f"quality_per_joule_{suffix}",
                "test_type": "bootstrap_ci",
                "statistic": float(qpj.mean()),
                "p_value":   np.nan,
                "n_q4": np.nan, "n_q8": np.nan,
                "mean_q4": np.nan, "mean_q8": np.nan,
                "median_q4": np.nan, "median_q8": np.nan,
                "n_pairs": len(qpj),
                "effect_size_cohen_d": np.nan,
                "bootstrap_ci_lower": ci_lo, "bootstrap_ci_upper": ci_hi,
            })

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)

    # Holm correction across all rows that have a p_value
    p_raw = [float(p) if not pd.isna(p) else np.nan for p in result.get("p_value", [])]
    result["holm_corrected_p_value"] = _holm_correction(p_raw)
    result["significant_at_05"]      = result["p_value"].fillna(1.0) < 0.05
    result["significant_at_05_holm"] = result["holm_corrected_p_value"].fillna(1.0) < 0.05

    return result


# ─── Pareto efficiency ────────────────────────────────────────────────────────

def _pareto_analysis(
    objectives: np.ndarray,
    directions: list[int],
) -> tuple[np.ndarray, list[list[int]], np.ndarray]:
    """
    Multi-objective Pareto analysis.

    Args:
        objectives: (n, k) float array — k objectives for n configurations.
        directions: list of k ints — +1 maximize, -1 minimize.

    Returns:
        dominated: bool array (n,) — True if dominated by at least one other config.
        dominated_by: list of lists — dominated_by[i] = local indices of configs dominating i.
        ranks: int array (n,) — Pareto rank (1 = Pareto front, 2 = second front, …).
    """
    n = len(objectives)
    obj_norm = objectives * np.array(directions, dtype=float)  # normalize: larger = better

    dominated_by: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i: at least as good everywhere, strictly better somewhere
            if np.all(obj_norm[j] >= obj_norm[i]) and np.any(obj_norm[j] > obj_norm[i]):
                dominated_by[i].append(j)

    dominated = np.array([len(d) > 0 for d in dominated_by])

    # Iterative front-peeling to assign Pareto ranks
    ranks = np.zeros(n, dtype=int)
    remaining = set(range(n))
    rank = 1
    while remaining:
        front = [i for i in remaining
                 if not any(j in remaining for j in dominated_by[i])]
        if not front:
            for i in remaining:
                ranks[i] = rank
            break
        for i in front:
            ranks[i] = rank
        remaining -= set(front)
        rank += 1

    return dominated, dominated_by, ranks


def build_paper_pareto(df: pd.DataFrame, primary_energy_col: str) -> pd.DataFrame:
    """
    Configuration-level Pareto analysis over three objectives:
      maximize  mean_quality_score
      minimize  mean_total_energy_wh_per_1k_output_tokens  (corrected or measured)
      minimize  mean_total_latency_seconds

    The per-1k-token energy column used for Pareto is derived from primary_energy_col:
      *corrected*  →  total_energy_wh_per_1k_output_tokens_corrected
      *measured*   →  total_energy_wh_per_1k_output_tokens_measured

    Adds columns: pareto_efficient, dominated_by, pareto_rank.
    dominated_by is a semicolon-separated string of config labels (model/quant/hw/device).
    pareto_rank = 1 is the Pareto front; higher ranks are iteratively peeled fronts.
    """
    if "corrected" in primary_energy_col:
        pareto_e_src   = "total_energy_wh_per_1k_output_tokens_corrected"
        pareto_e_label = "energy_wh_per_1k_tokens_corrected"
    else:
        pareto_e_src   = "total_energy_wh_per_1k_output_tokens_measured"
        pareto_e_label = "energy_wh_per_1k_tokens_measured"

    config_keys = [k for k in ("model_name", "quantization", "hardware_profile",
                               "execution_device") if k in df.columns]
    records: list[dict] = []

    for key_vals, grp in df.groupby(config_keys, dropna=False):
        row: dict[str, Any] = {}
        kv = key_vals if isinstance(key_vals, tuple) else (key_vals,)
        for k, v in zip(config_keys, kv):
            row[k] = v

        judge_ok = grp.get("judge_status",        pd.Series(dtype=str)).isin({"ok"})
        conv_ok  = grp.get("status_conversation", pd.Series(dtype=str)).isin({"ok"})

        q_vals    = pd.to_numeric(grp.loc[judge_ok, "score"], errors="coerce")
        e_primary = (
            pd.to_numeric(grp.loc[conv_ok, primary_energy_col], errors="coerce")
            if primary_energy_col in grp.columns else pd.Series(dtype=float)
        )
        e1k_vals  = (
            pd.to_numeric(grp.loc[conv_ok, pareto_e_src], errors="coerce")
            if pareto_e_src in grp.columns else pd.Series(dtype=float)
        )
        lat_vals  = (
            pd.to_numeric(grp.loc[conv_ok, "total_latency_seconds"], errors="coerce")
            if "total_latency_seconds" in grp.columns else pd.Series(dtype=float)
        )
        bc_j_vals = (
            pd.to_numeric(grp.loc[conv_ok, "total_baseline_corrected_energy_joules"], errors="coerce")
            if "total_baseline_corrected_energy_joules" in grp.columns else pd.Series(dtype=float)
        )

        row["n_conversations"]               = len(grp)
        row["n_quality_evaluated"]           = int(judge_ok.sum())
        row["mean_quality_score"]            = float(q_vals.mean())      if not q_vals.empty    else np.nan
        row["std_quality_score"]             = float(q_vals.std(ddof=1)) if len(q_vals) > 1     else np.nan
        row["mean_energy_primary"]           = float(e_primary.mean())   if not e_primary.empty else np.nan
        row["energy_col_used"]               = primary_energy_col
        row[f"mean_{pareto_e_label}"]        = float(e1k_vals.mean())    if not e1k_vals.empty  else np.nan
        row["pareto_energy_col"]             = pareto_e_label
        row["mean_total_latency_seconds"]    = float(lat_vals.mean())    if not lat_vals.empty  else np.nan
        row["quality_per_joule_corrected"]   = _safe_div(
            row["mean_quality_score"],
            float(bc_j_vals.mean()) if not bc_j_vals.empty else np.nan,
        )
        records.append(row)

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)

    e1k_col = f"mean_{pareto_e_label}"
    pareto_valid = (
        result["mean_quality_score"].notna()
        & result[e1k_col].notna()
        & result["mean_total_latency_seconds"].notna()
    )

    # Build short config labels for the dominated_by column
    def _label(i: int) -> str:
        return "/".join(str(result.at[i, k]) for k in config_keys if k in result.columns)

    labels = [_label(i) for i in result.index]

    if pareto_valid.sum() >= 2:
        valid_idx = result.index[pareto_valid].tolist()
        obj_mat = np.column_stack([
            result.loc[pareto_valid, "mean_quality_score"].to_numpy(dtype=float),
            result.loc[pareto_valid, e1k_col].to_numpy(dtype=float),
            result.loc[pareto_valid, "mean_total_latency_seconds"].to_numpy(dtype=float),
        ])
        dominated, dominated_by_lists, ranks_arr = _pareto_analysis(
            obj_mat, directions=[+1, -1, -1]
        )

        result["pareto_efficient"] = False
        result["dominated_by"]     = ""
        result["pareto_rank"]      = np.nan

        for local_i, global_i in enumerate(valid_idx):
            result.at[global_i, "pareto_efficient"] = not bool(dominated[local_i])
            result.at[global_i, "pareto_rank"]      = int(ranks_arr[local_i])
            dominators = [labels[valid_idx[j]] for j in dominated_by_lists[local_i]]
            result.at[global_i, "dominated_by"]     = "; ".join(dominators)
    else:
        result["pareto_efficient"] = np.nan
        result["dominated_by"]     = np.nan
        result["pareto_rank"]      = np.nan

    return result.sort_values("mean_quality_score", ascending=False, na_position="last")


def build_best_efficiency_by_category(df_summary: pd.DataFrame) -> pd.DataFrame:
    """
    For each category: best configuration by quality_per_joule_corrected,
    falling back to quality_per_joule_measured if corrected is all-NaN.
    """
    if df_summary.empty:
        return pd.DataFrame()

    eff_col = "quality_per_joule_corrected"
    if eff_col not in df_summary.columns or df_summary[eff_col].isna().all():
        eff_col = "quality_per_joule_measured"
    if eff_col not in df_summary.columns:
        return pd.DataFrame()

    cat_col = next((c for c in ("original_category",) if c in df_summary.columns), None)
    if cat_col is None:
        return pd.DataFrame()

    records: list[dict] = []
    for cat, grp in df_summary.groupby(cat_col):
        valid = grp[pd.to_numeric(grp[eff_col], errors="coerce").notna()]
        if valid.empty:
            continue
        best = valid.loc[pd.to_numeric(valid[eff_col], errors="coerce").idxmax()].to_dict()
        best["efficiency_col_used"] = eff_col
        records.append(best)

    return pd.DataFrame(records)


# ─── Carga y union ────────────────────────────────────────────────────────────

def load_and_join(
    conv_csv:  Path,
    judge_csv: Path,
    phase:     Optional[int],
) -> pd.DataFrame:
    if not conv_csv.exists():
        print(f"ERROR: {conv_csv} no encontrado.")
        sys.exit(1)

    df_conv = pd.read_csv(conv_csv, low_memory=False)
    print(f"  conversation_results.csv : {len(df_conv):,} filas")

    if not judge_csv.exists():
        print(f"  AVISO: {judge_csv} no encontrado — metricas de calidad seran NaN.")
        df_judge_slim = pd.DataFrame({"conversation_id": pd.Series(dtype=str)})
    else:
        df_judge = pd.read_csv(judge_csv, low_memory=False)
        print(f"  judge_results.csv        : {len(df_judge):,} filas")
        present = [c for c in _JUDGE_ONLY_COLS if c in df_judge.columns]
        df_judge_slim = df_judge[["conversation_id"] + present].copy()

    df = df_conv.merge(df_judge_slim, on="conversation_id", how="left")

    if phase is not None:
        df = df[df["phase"].astype(str).str.strip() == str(phase)]
        print(f"  Filtro fase={phase}          : {len(df):,} filas")

    for col in _ENERGY_COLS + ["score"] + _QUALITY_SUBDIMS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  Total en analisis        : {len(df):,} filas")
    n_judged = df.get("judge_status", pd.Series(dtype=str)).isin({"ok"}).sum()
    print(f"  Con judge_status=success : {n_judged:,}")
    return df


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "GREEN-IA: Resumen estadistico de energia y calidad. "
            "Genera 14 tablas CSV en results/summary/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--conv-csv",       type=Path, default=DEFAULT_CONV_CSV)
    parser.add_argument("--judge-csv",      type=Path, default=DEFAULT_JUDGE_CSV)
    parser.add_argument("--output-dir",     type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--phase",          type=int,  choices=[1, 2], default=None,
                        help="Filtrar por fase (sin arg = todas)")
    parser.add_argument("--primary-energy", choices=["measured", "corrected"],
                        default="corrected",
                        help="Metrica de energia para tablas de Pareto")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def _relpath(p: Path) -> str:
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)

    W = 64
    print(f"\n{'=' * W}")
    print(f"  GREEN-IA — Resumen estadistico")
    print(f"{'=' * W}")
    print(f"  Conv CSV    : {_relpath(args.conv_csv)}")
    print(f"  Judge CSV   : {_relpath(args.judge_csv)}")
    print(f"  Output      : {_relpath(args.output_dir)}/")
    print(f"  Fase        : {args.phase if args.phase else 'todas'}")
    sci_note = f"si ({_sp_stats.__name__})" if _SCIPY else "no (CI95 usa z=1.96)"
    print(f"  scipy       : {sci_note}")
    print(f"{'─' * W}")

    df = load_and_join(args.conv_csv, args.judge_csv, args.phase)
    print(f"{'─' * W}")

    primary_energy_col = (
        "total_baseline_corrected_energy_wh"
        if args.primary_energy == "corrected"
        else "total_measured_energy_wh"
    )

    O = args.output_dir
    out: list[tuple[str, int]] = []

    # 1
    out.append(("summary_by_model.csv",
        _groupby_save(df, ["model_name"], O / "summary_by_model.csv")))

    # 2
    out.append(("summary_by_model_quantization.csv",
        _groupby_save(df, ["model_name", "quantization"],
                      O / "summary_by_model_quantization.csv")))

    # 3
    out.append(("summary_by_model_quantization_device.csv",
        _groupby_save(df, ["model_name", "quantization", "execution_device"],
                      O / "summary_by_model_quantization_device.csv")))

    # 4
    out.append(("summary_by_hardware_profile.csv",
        _groupby_save(df, ["hardware_profile"],
                      O / "summary_by_hardware_profile.csv")))

    # 5
    out.append(("summary_by_execution_device.csv",
        _groupby_save(df, ["execution_device"],
                      O / "summary_by_execution_device.csv")))

    # 6
    out.append(("summary_by_category.csv",
        _groupby_save(df, ["original_category"],
                      O / "summary_by_category.csv")))

    # 7
    out.append(("summary_by_model_category.csv",
        _groupby_save(df, ["model_name", "original_category"],
                      O / "summary_by_model_category.csv")))

    # 8  full ranking — sorted by quality_per_joule_corrected
    _full_ranking_path = O / "summary_full_ranking.csv"
    _full_ranking_keys = ["model_name", "quantization", "hardware_profile", "execution_device"]
    n_fr = _groupby_save(df, _full_ranking_keys, _full_ranking_path,
                         sort_by="quality_per_joule_corrected")
    # Add config_id — used by optimize_winner.py --selected-configuration explicit:<config_id>
    if _full_ranking_path.exists() and n_fr > 0:
        _df_fr = pd.read_csv(_full_ranking_path)
        _id_parts = [c for c in _full_ranking_keys if c in _df_fr.columns]
        _df_fr.insert(0, "config_id",
                      _df_fr[_id_parts].apply(
                          lambda r: "/".join(str(r[k]) for k in _id_parts), axis=1
                      ))
        _df_fr.to_csv(_full_ranking_path, index=False, float_format="%.6g")
    out.append(("summary_full_ranking.csv", n_fr))

    # 9  best efficiency per category (from summary_by_model_category)
    cat_model_path = O / "summary_by_model_category.csv"
    if cat_model_path.exists():
        df_best_eff = build_best_efficiency_by_category(pd.read_csv(cat_model_path))
        if not df_best_eff.empty:
            df_best_eff.to_csv(O / "best_efficiency_by_category.csv",
                               index=False, float_format="%.6g")
        out.append(("best_efficiency_by_category.csv", len(df_best_eff)))
    else:
        out.append(("best_efficiency_by_category.csv", 0))

    # 10 statistical tests (Q4 vs Q8)
    df_tests = compute_statistical_tests(df)
    if not df_tests.empty:
        df_tests.to_csv(O / "statistical_tests.csv", index=False, float_format="%.6g")
    out.append(("statistical_tests.csv", len(df_tests)))

    # 11 paper table — energy + quality by configuration
    out.append(("paper_table_energy_quality_by_configuration.csv",
        _groupby_save(
            df,
            ["model_name", "quantization", "hardware_profile", "execution_device"],
            O / "paper_table_energy_quality_by_configuration.csv",
            sort_by="mean_quality_score",
        )))

    # 12 paper table — by category × model × quantization
    out.append(("paper_table_by_category.csv",
        _groupby_save(
            df,
            ["original_category", "model_name", "quantization"],
            O / "paper_table_by_category.csv",
            sort_by="mean_quality_score",
        )))

    # 13 paper table — best per category (from paper_table_by_category)
    cat_path = O / "paper_table_by_category.csv"
    if cat_path.exists():
        df_best = build_best_efficiency_by_category(pd.read_csv(cat_path))
        if not df_best.empty:
            df_best.to_csv(O / "paper_table_best_by_category.csv",
                           index=False, float_format="%.6g")
        out.append(("paper_table_best_by_category.csv", len(df_best)))
    else:
        out.append(("paper_table_best_by_category.csv", 0))

    # 14 paper table — Pareto efficient configurations
    df_pareto = build_paper_pareto(df, primary_energy_col)
    if not df_pareto.empty:
        df_pareto.to_csv(O / "paper_table_pareto_efficient.csv",
                         index=False, float_format="%.6g")
    out.append(("paper_table_pareto_efficient.csv", len(df_pareto)))

    # ── summary ────────────────────────────────────────────────────────────────
    print(f"\n  Archivos generados en {_relpath(O)}/")
    print(f"  {'─' * 58}")
    for fname, nrows in out:
        status = f"{nrows:4d} filas" if nrows > 0 else "  (vacio)"
        print(f"  {fname:<52}  {status}")

    print(f"\n  quality_per_joule_* = adaptacion operacional de Accuracy-per-Joule")
    print(f"  (Canziani et al. 2017). Ver docstring del modulo para detalles.")
    print(f"  Metrica de energia primaria: {primary_energy_col}")
    print(f"{'=' * W}\n")


if __name__ == "__main__":
    main()

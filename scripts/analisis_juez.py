"""Estadísticas sobre los juicios consolidados (solo lectura, sin llamadas a API).

Entradas : results/judge/{determinista,fabricante}/juicios_consolidados.csv
Salidas  : results/analisis_juez/*.csv
"""
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

try:
    from sklearn.metrics import cohen_kappa_score
    HAY_SKLEARN = True
except ImportError:
    HAY_SKLEARN = False

try:
    from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa
    HAY_STATSMODELS = True
except ImportError:
    HAY_STATSMODELS = False

RAIZ = Path(__file__).resolve().parent.parent
ENTRADA = RAIZ / "results" / "judge"
SALIDA = RAIZ / "results" / "analisis_juez"
ESQUEMAS = ["determinista", "fabricante"]
JUECES = ["anthropic", "openai", "google"]
VEREDICTOS = ["q4", "q8", "empate", "error"]


def kappa_cohen(a, b):
    """Kappa de Cohen; usa sklearn si existe, si no una implementación equivalente."""
    if HAY_SKLEARN:
        return float(cohen_kappa_score(a, b))
    a, b = pd.Series(list(a)), pd.Series(list(b))
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in set(a) | set(b))
    return float("nan") if pe == 1 else float((po - pe) / (1 - pe))


def conteos(df, col, claves):
    """Cuenta veredictos agrupando por `claves`."""
    tabla = (df.groupby(claves)[col].value_counts().unstack(fill_value=0)
             .reindex(columns=VEREDICTOS, fill_value=0))
    tabla.columns = ["gana_q4", "gana_q8", "empate", "error"]
    tabla["n_pares"] = tabla.sum(axis=1)
    return tabla.reset_index()


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_csv(ENTRADA / e / "juicios_consolidados.csv") for e in ESQUEMAS],
                   ignore_index=True)
    for c in ["veredicto_panel"] + [f"{j}_consolidado" for j in JUECES]:
        df[c] = df[c].fillna("error").str.strip().str.lower()
        desconocidos = set(df[c]) - set(VEREDICTOS)
        if desconocidos:
            print(f"AVISO: valores no esperados en {c}: {desconocidos}")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    # 1. Panel
    t1a = conteos(df, "veredicto_panel", ["esquema", "modelo"])
    t1b = conteos(df, "veredicto_panel", ["esquema", "modelo", "categoria"])
    t1a.to_csv(SALIDA / "1a_panel_por_esquema_modelo.csv", index=False)
    t1b.to_csv(SALIDA / "1b_panel_por_categoria.csv", index=False)
    print("\n=== 1. Resultado del panel por esquema y modelo ===")
    print(t1a.to_string(index=False))
    print("\n=== 1b. Resultado del panel por categoría ===")
    print(t1b.to_string(index=False))

    # 2. Por juez
    filas = []
    for j in JUECES:
        t = conteos(df, f"{j}_consolidado", ["esquema", "modelo"])
        t.insert(0, "juez", j)
        filas.append(t)
    t2 = pd.concat(filas, ignore_index=True).rename(columns={"error": "pares_con_error"})
    t2.to_csv(SALIDA / "2_por_juez.csv", index=False)
    print("\n=== 2. Resultado por juez (veredicto consolidado) ===")
    print(t2.to_string(index=False))

    # 3. Acuerdo entre jueces (global y por esquema)
    filas = []
    for ambito, sub in [("todos", df)] + [(e, df[df.esquema == e]) for e in ESQUEMAS]:
        for a, b in combinations(JUECES, 2):
            ca, cb = f"{a}_consolidado", f"{b}_consolidado"
            ok = sub[(sub[ca] != "error") & (sub[cb] != "error")]
            n = len(ok)
            filas.append({
                "ambito": ambito, "juez_a": a, "juez_b": b, "n_usado": n,
                "n_excluidos_por_error": len(sub) - n,
                "acuerdo_pct": round(100 * (ok[ca] == ok[cb]).mean(), 2) if n else None,
                "kappa_cohen": round(kappa_cohen(ok[ca], ok[cb]), 4) if n else None,
            })
    t3 = pd.DataFrame(filas)
    t3.to_csv(SALIDA / "3a_acuerdo_pares_jueces.csv", index=False)
    print("\n=== 3. Acuerdo entre pares de jueces (excluye pares con error) ===")
    if not HAY_SKLEARN:
        print("AVISO: sklearn no instalado; kappa de Cohen calculado con implementación propia "
              "(misma fórmula). Para usar sklearn: pip install scikit-learn")
    print(t3.to_string(index=False))

    if HAY_STATSMODELS:
        filas = []
        for ambito, sub in [("todos", df)] + [(e, df[df.esquema == e]) for e in ESQUEMAS]:
            cols = [f"{j}_consolidado" for j in JUECES]
            ok = sub[(sub[cols] != "error").all(axis=1)]
            tabla, _ = aggregate_raters(ok[cols].to_numpy())
            filas.append({"ambito": ambito, "n_usado": len(ok),
                          "kappa_fleiss": round(float(fleiss_kappa(tabla, method="fleiss")), 4)})
        t3b = pd.DataFrame(filas)
        t3b.to_csv(SALIDA / "3b_kappa_fleiss.csv", index=False)
        print("\n=== 3b. Kappa de Fleiss (3 jueces) ===")
        print(t3b.to_string(index=False))
    else:
        print("\nAVISO: statsmodels no instalado; se omite kappa de Fleiss. "
              "Instalar con: pip install statsmodels")

    # 4. Prueba de signos exacta sobre el panel
    filas = []
    for (e, m), sub in df.groupby(["esquema", "modelo"]):
        q4 = int((sub.veredicto_panel == "q4").sum())
        q8 = int((sub.veredicto_panel == "q8").sum())
        n = q4 + q8
        p = binomtest(q4, n, 0.5, alternative="two-sided").pvalue if n else None
        filas.append({"esquema": e, "modelo": m, "n_con_ganador": n, "victorias_q4": q4,
                      "victorias_q8": q8, "p_valor": round(p, 4) if p is not None else None,
                      "empates_excluidos": int((sub.veredicto_panel == "empate").sum()),
                      "errores_excluidos": int((sub.veredicto_panel == "error").sum())})
    t4 = pd.DataFrame(filas)
    t4.to_csv(SALIDA / "4_prueba_signos_panel.csv", index=False)
    print("\n=== 4. Prueba de signos exacta (binomial bilateral, p=0.5) sobre el panel ===")
    print(t4.to_string(index=False))
    print(f"\nCSV guardados en {SALIDA}")


if __name__ == "__main__":
    main()

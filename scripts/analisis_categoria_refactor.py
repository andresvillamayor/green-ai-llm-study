"""GREEN-IA — Analisis energetico de categorias MT-Bench. 5 prompts x 15 reps, CodeCarbon, CSV y plots."""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))  # permite importar graficos_refactor desde scripts/

from src.config_loader import load_config
from src.prompt_builder import build_prompt
from graficos_refactor import calcular_estadisticas, generar_plot_comparacion, generar_plots
from carga_modelo_refactor import resolver_ruta_modelo, cargar_modelo
from sistema_refactor import activar_caffeinate, detener_caffeinate
from medicion_energia_refactor import medir_energia


CATEGORIAS = ["humanities"]
MODEL_NAME = "qwen2.5-7b"        # nombre del modelo segun config.yaml
CUANTIZACION = "q8"             # "q4" o "q8" — cual corrida se ejecuta ahora
REPETICIONES = 15                # veces que se repite cada prompt
WARMUP_REPETICIONES = 3          # inferencias descartadas antes de medir oficialmente
                                 # hallazgo empirico: CV baja de 5-6% a <1% desde la 3ra rep
SUBSET_YAML = ROOT / "data" / "mt_bench" / "subset" / "mt_bench_literal_subset_5_per_category.yaml"  # subset de prompts
COLUMNAS_CSV = [
    "modelo", "categoria", "prompt_id", "prompt_num", "repeticion",
    "prompt_texto_corto", "measured_energy_kwh", "measured_energy_mwh",
    "cpu_energy_mwh", "gpu_energy_mwh", "ram_energy_mwh",
    "inference_time_s", "completion_tokens",
]  # columnas del CSV de salida
MAX_CHARS_PROMPT = 80            # caracteres del texto del prompt a guardar en el CSV


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




def main() -> None:
    """Ejecuta el experimento completo: carga modelo UNA vez, itera categorias, guarda CSV y genera plots."""
    ANCHO = 68
    cfg = load_config(ROOT / "config.yaml")
    ruta_modelo = resolver_ruta_modelo(ROOT, MODEL_NAME, CUANTIZACION)
    proceso_cafe = activar_caffeinate()
    llm = cargar_modelo(ruta_modelo, cfg)

    resumen_categorias = {}  # categoria -> {"media_global", "std_global", "cv_global"}

    for CATEGORIA in CATEGORIAS:
        output_dir = ROOT / "results" / "analisis_categoria" / f"{MODEL_NAME}_{CUANTIZACION}" / CATEGORIA
        output_csv = output_dir / "resultados_completos.csv"

        print(f"\n{'=' * ANCHO}")
        print(f"  GREEN-IA — Analisis de categoria: {CATEGORIA.upper()}")
        print(f"  Modelo   : {MODEL_NAME} {CUANTIZACION.upper()}  (n_gpu_layers={cfg.n_gpu_layers()}, {cfg.expected_backend()})")
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
                        "modelo": f"{MODEL_NAME}-{CUANTIZACION}", "categoria": CATEGORIA, "prompt_id": id_pregunta,
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
                    print(f"  [{n_hecho:3d}/{total}]  Prompt {num_prompt}  rep {rep:2d}/{REPETICIONES}"
                          f"  {medicion['tiempo_inferencia_s']:6.1f}s  {medicion['tokens_generados']:4d} tok  {mwh:.4f} mWh")
        finally:
            archivo_csv.close()
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
    # Plot comparativo entre categorias
    print(f"\n  Generando plot comparativo...")
    generar_plot_comparacion(resumen_categorias)
    detener_caffeinate(proceso_cafe)
    print(f"{'=' * ANCHO}\n")


if __name__ == "__main__":
    main()

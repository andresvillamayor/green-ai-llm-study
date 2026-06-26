# GREEN-IA: Consumo Energético en Inferencia de LLMs

**Tesis de Maestría — Ciencia de Datos**
**Universidad Comunero — Paraguay**
**Autor: Andres Villamayor**

Estudio experimental del impacto energético de cuantización (Q4 vs Q8) en la inferencia
de Large Language Models, con evaluación simultánea de calidad de respuesta.
Ejecutado en Apple M4 Mac Mini y Windows RTX 4060; medición energética con CodeCarbon;
evaluación de calidad con Claude como juez (LLM-as-a-Judge).

---

## Tabla de Contenidos

1. [Descripción del experimento](#1-descripción-del-experimento)
2. [Comparaciones disponibles — Fase 1](#2-comparaciones-disponibles--fase-1)
3. [Comparaciones disponibles — Fase 2](#3-comparaciones-disponibles--fase-2)
4. [Estructura del proyecto](#4-estructura-del-proyecto)
5. [Requisitos previos](#5-requisitos-previos)
6. [Instalación en macOS Apple Silicon (Metal)](#6-instalación-en-macos-apple-silicon-metal)
7. [Instalación en Windows con NVIDIA (CUDA)](#7-instalación-en-windows-con-nvidia-cuda)
8. [Configuración](#8-configuración)
9. [Dataset MT-Bench — Obtención](#9-dataset-mt-bench--obtención)
10. [Dataset MT-Bench — Preparación del subset](#10-dataset-mt-bench--preparación-del-subset)
11. [Verificación de integridad (SHA256)](#11-verificación-de-integridad-sha256)
12. [Validación del entorno](#12-validación-del-entorno)
13. [Fase 1 — Ejecución del benchmark](#13-fase-1--ejecución-del-benchmark)
14. [Evaluación de calidad con Claude](#14-evaluación-de-calidad-con-claude)
15. [Fase 1 — Análisis estadístico](#15-fase-1--análisis-estadístico)
16. [Fase 1 — Gráficas](#16-fase-1--gráficas)
17. [Fase 2 — Optimización del ganador](#17-fase-2--optimización-del-ganador)
18. [Fase 2 — Análisis y gráficas](#18-fase-2--análisis-y-gráficas)
19. [Diseño experimental — Fase 1](#19-diseño-experimental--fase-1)
20. [Principios metodológicos](#20-principios-metodológicos)
21. [Métricas de energía: medida vs. corregida por baseline](#21-métricas-de-energía-medida-vs-corregida-por-baseline)
22. [Métricas de eficiencia: Quality/Joule, EDP, SCI, Pareto](#22-métricas-de-eficiencia-qualityjoule-edp-sci-pareto)
23. [Tests estadísticos y método CI](#23-tests-estadísticos-y-método-ci)
24. [Modelos y tipos de modelo](#24-modelos-y-tipos-de-modelo)
25. [Limitaciones metodológicas](#25-limitaciones-metodológicas)
26. [Advertencias sobre CodeCarbon](#26-advertencias-sobre-codecarbon)
27. [Modo de generación y propósito de las repeticiones](#27-modo-de-generación-y-propósito-de-las-repeticiones)
28. [Reproducibilidad y entorno de referencia](#28-reproducibilidad-y-entorno-de-referencia)
29. [Archivos de salida esperados](#29-archivos-de-salida-esperados)
30. [Checklist de reporte](#30-checklist-de-reporte)
31. [Referencias](#31-referencias)

---

## 1. Descripción del experimento

Este proyecto mide el consumo energético de la inferencia local de LLMs bajo distintas
configuraciones de cuantización (Q4_K_M vs Q8_0), hardware (Apple M4 / Windows RTX 4060)
y dispositivo de ejecución (CPU vs GPU), usando el benchmark oficial MT-Bench como
conjunto de prompts y Claude como juez de calidad.

**Objetivo principal:** determinar si la cuantización Q4 reduce el consumo energético
respecto a Q8 sin degradar la calidad de respuesta por encima de un umbral tolerable.

**Fase 1** — Benchmark comparativo: se ejecutan todas las configuraciones (modelo ×
cuantización × dispositivo × hardware) sobre el subset de MT-Bench (40 preguntas,
8 categorías, 2 turnos, 15 repeticiones) y se evalúa calidad con Claude.

**Fase 2** — Optimización: se selecciona la configuración ganadora de Fase 1 y se buscan
parámetros de inferencia (n_ctx, n_batch, n_threads, max_tokens) que reduzcan el consumo
energético sin degradar la calidad más allá del límite P22.

---

## 2. Comparaciones disponibles — Fase 1

Los scripts de análisis (`analysis_summary.py`, `analysis_plots.py`) producen todas
las comparaciones siguientes:

| Dimensión | Métrica(s) comparadas |
|-----------|----------------------|
| **Q4 vs Q8** | energía, calidad, latencia, tokens/J, Quality/J, EDP, SCI |
| **llama-2-7b vs qwen2.5-7b** | todas las métricas; notar sesgo base/instruct (§24) |
| **CPU vs GPU** | energía, latencia, Wh/1K tokens, EDP |
| **Mac M4 vs Windows RTX 4060** | energía, latencia, Wh/1K tokens, EDP |
| **Por categoría MT-Bench** | calidad por categoría (8 categorías); energía por categoría |
| **Por prompt** | calidad por pregunta; energía por pregunta |
| **Pareto frontier** | Quality↑ vs Energy↓ vs Latency↓ (frente de Pareto 2-obj y 3-obj) |
| **Calidad multi-turno** | score promedio sobre conversaciones de 2 turnos |
| **Energía por conversación** | Wh totales por conversación multi-turno |
| **Energía corregida por baseline** | baseline_corrected_energy_wh |
| **Joules por token de salida** | total_energy_j_per_output_token_corrected |
| **Wh por 1K tokens de salida** | total_energy_wh_per_1k_output_tokens_corrected |
| **Tokens por Joule** | total_tokens_per_joule_corrected |
| **Quality-per-Joule** | score / baseline_corrected_energy_joules |
| **EDP** | energy_joules × latency_seconds |
| **Operational SCI** | Software Carbon Intensity por conversación y por 1K tokens |
| **Latencia** | total_latency_seconds por conversación |

---

## 3. Comparaciones disponibles — Fase 2

Los scripts de optimización (`optimize_winner.py`, `optimization_plots.py`) producen:

| Dimensión | Descripción |
|-----------|-------------|
| **Parámetros optimizados** | n_ctx, n_batch, n_threads, max_tokens del ganador |
| **Reducción de energía** | % de reducción de Wh/1K tokens vs baseline Fase 1 |
| **Reducción de latencia** | % de reducción de latencia vs baseline Fase 1 |
| **Mejora tokens/Joule** | % de mejora en tokens por Joule |
| **Mejora Quality-per-Joule** | % de mejora en calidad/energía |
| **Reducción de EDP** | % de reducción del producto energía × latencia |
| **Baseline vs optimizado** | comparación directa ganador Fase 1 vs ganador Fase 2 |
| **Pareto tras optimización** | frente de Pareto actualizado con configuraciones Stage B |
| **Validación restricción P22** | mean_quality ≥ baseline − 0.25 AND ≥ baseline × 0.97 |

---

## 4. Estructura del proyecto

```
llm-quantization-study/
├── config.yaml                        # configuración principal del experimento
├── config.yaml.example                # plantilla de configuración
├── optimization_config.yaml           # configuración de búsqueda Fase 2
├── requirements.txt                   # dependencias Python
├── run_experiment.py                  # punto de entrada Fase 1
├── judge_with_claude.py               # evaluador de calidad LLM-as-a-Judge
├── analysis_summary.py                # tablas estadísticas Fase 1
├── analysis_plots.py                  # gráficas Fase 1
├── optimize_winner.py                 # optimización Fase 2
├── optimization_plots.py              # gráficas Fase 2
│
├── scripts/
│   ├── validate_config.py             # validación del entorno y config
│   ├── prepare_mt_bench_subset.py     # preparación del subset MT-Bench
│   ├── hash_artifacts.py              # huellas SHA256 de modelos y dataset
│   ├── setup_entorno.py               # reporte del entorno instalado
│   ├── fase1_benchmark.py             # lógica interna de medición Fase 1
│   ├── fase2_optimizacion.py          # lógica interna de medición Fase 2
│   └── juez_calidad_v2.py             # cliente Claude para evaluación
│
├── src/
│   ├── config_loader.py               # carga y validación de config.yaml
│   ├── hardware_profile.py            # detección de hardware
│   ├── metrics.py                     # cálculo de métricas energéticas
│   ├── metadata_writer.py             # escritura de archivos de reproducibilidad
│   ├── mtbench_loader.py              # carga del subset MT-Bench
│   ├── baseline_energy.py             # calibración del baseline idle
│   └── reproducibility.py            # snapshots de configuración y entorno
│
├── models/
│   ├── llama-2-7b/
│   │   ├── Q4_K_M.gguf               # modelo llama-2-7b Q4 (4.1 GB)
│   │   └── llama-2-7b.Q8_0.gguf     # modelo llama-2-7b Q8 (7.2 GB)
│   └── qwen2.5-7b/
│       ├── Qwen2.5-7B-Instruct-Q4_K_M.gguf   # (4.7 GB)
│       └── Qwen2.5-7B-Instruct-Q8_0.gguf     # (8.1 GB)
│
├── data/
│   └── mt_bench/
│       ├── official/
│       │   ├── question.jsonl         # 80 preguntas oficiales (IDs 81–160)
│       │   └── judge_prompts.jsonl    # prompts del juez oficial MT-Bench
│       └── subset/
│           ├── mt_bench_literal_subset_5_per_category.jsonl
│           ├── mt_bench_literal_subset_5_per_category.yaml
│           └── mt_bench_subset_manifest.csv
│
└── results/
    ├── metadata/                      # artefactos de reproducibilidad
    ├── raw/                           # CSV de medición Fase 1
    ├── judge/                         # CSV de evaluación de calidad
    ├── summary/                       # tablas de análisis estadístico
    ├── plots/                         # gráficas Fase 1
    └── optimization/                  # salidas Fase 2
```

---

## 5. Requisitos previos

- Python 3.11 (recomendado) o 3.10. Python 3.13 puede tener incompatibilidades con
  extensiones C de llama-cpp-python.
- **macOS Apple Silicon:** Xcode Command Line Tools (`xcode-select --install`).
- **Windows NVIDIA:** Driver NVIDIA reciente, CUDA Toolkit compatible, Visual Studio
  Build Tools (C++ workload).
- Git (para clonar el repositorio y obtener el dataset MT-Bench).
- Archivos GGUF de los modelos (ver §24). Los archivos GGUF **no están incluidos**
  en el repositorio; deben descargarse por separado desde Hugging Face.
- Clave de API Anthropic (`ANTHROPIC_API_KEY`) para la evaluación de calidad.

---

## 6. Instalación en macOS Apple Silicon (Metal)

```bash
# 1. Clonar el repositorio
git clone <url-del-repositorio>
cd llm-quantization-study

# 2. Crear entorno virtual
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# 3. Instalar dependencias base
pip install -r requirements.txt

# 4. Instalar llama-cpp-python con soporte Metal (GPU Apple Silicon)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python \
    --force-reinstall --no-cache-dir

# 5. Guardar el entorno instalado (lock file de reproducibilidad)
pip freeze > requirements-lock.txt
pip freeze > results/metadata/requirements_freeze.txt

# 6. Generar reporte del entorno
python scripts/setup_entorno.py
```

> **Verificar Metal:** `python scripts/validate_config.py --check-backend`
> Debe aparecer `[PASS] Metal backend` y `n_gpu_layers = -1` en modo GPU.

**Condiciones de medición recomendadas (Mac Mini):**

```bash
# Deshabilitar reposo automático durante el experimento
sudo pmset -a sleep 0 disksleep 0

# Liberar memoria caché antes de cada sesión
sudo purge

# Restaurar al terminar
sudo pmset -a sleep 10 disksleep 10
```

Los scripts `fase1_benchmark.py` y `fase2_optimizacion.py` activan `caffeinate`
automáticamente cuando `use_caffeinate_on_macos: true` en `config.yaml`.

---

## 7. Instalación en Windows con NVIDIA (CUDA)

```powershell
# 1. Clonar el repositorio
git clone <url-del-repositorio>
cd llm-quantization-study

# 2. Crear entorno virtual
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel

# 3. Instalar dependencias base
pip install -r requirements.txt

# 4. Instalar llama-cpp-python con soporte CUDA
$env:CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python --force-reinstall --no-cache-dir

# 5. Guardar el entorno instalado
pip freeze > requirements-lock.txt
pip freeze > results/metadata/requirements_freeze.txt

# 6. Generar reporte del entorno
python scripts/setup_entorno.py
```

> **Verificar CUDA:** `python scripts/validate_config.py --check-backend`
> Debe aparecer `[PASS] CUDA backend` y `n_gpu_layers = -1` en modo GPU.

**Condiciones de medición recomendadas (Windows NVIDIA):**

```powershell
# Activar plan de energía Alto Rendimiento
powercfg /setactive SCHEME_MIN

# Deshabilitar reposo automático durante el experimento
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0

# Restaurar al terminar
powercfg /change standby-timeout-ac 30
powercfg /change monitor-timeout-ac 10
```

Mantener el driver NVIDIA en la misma versión durante todas las sesiones comparadas.
Actualizar el driver entre sesiones introduce una variable de confusión en las métricas.

---

## 8. Configuración

```bash
cp config.yaml.example config.yaml
```

Campos obligatorios a editar antes de ejecutar:

| Campo | Valores permitidos | Descripción |
|-------|--------------------|-------------|
| `hardware_profile` | `mac_m4`, `windows_nvidia` | Perfil de hardware activo |
| `execution_device` | `cpu`, `gpu` | Dispositivo de ejecución de inferencia |

El resto de parámetros tiene valores por defecto validados. Editar solo si se comprende
el impacto metodológico (ver §20 para restricciones).

Para Fase 2, también editar `optimization_config.yaml`:

| Campo | Descripción |
|-------|-------------|
| `winner_selection_mode` | `auto_best_pareto`, `auto_best_quality_under_energy_constraint`, o `explicit:<config_id>` |
| `search_method` | `grid_search` o `random_search` |
| `parameter_space` | rangos de n_ctx, n_batch, n_threads, max_tokens |

---

## 9. Dataset MT-Bench — Obtención

El experimento usa exclusivamente los prompts oficiales de MT-Bench (Zheng et al.,
NeurIPS 2023). Los prompts se usan **verbatim** — sin modificaciones, sin traducción,
sin paráfrasis, sin prompts sintéticos generados.

> **El experimento no puede iniciarse si `data/mt_bench/official/question.jsonl`
> no existe. No hay dataset de respaldo. Ver principios P5–P8 en §20.**

```bash
# Obtener el dataset oficial desde el repositorio FastChat
git clone https://github.com/lm-sys/FastChat.git

# Copiar los archivos al directorio del proyecto
mkdir -p data/mt_bench/official data/mt_bench/subset

cp FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl \
   data/mt_bench/official/question.jsonl

# Opcional — prompts del juez oficial
cp FastChat/fastchat/llm_judge/data/judge_prompts.jsonl \
   data/mt_bench/official/judge_prompts.jsonl

# El clon de FastChat puede eliminarse una vez copiados los archivos
rm -rf FastChat
```

Verificar la integridad del archivo oficial:

```bash
python scripts/prepare_mt_bench_subset.py --stats
```

Salida esperada: **80 preguntas, IDs 81–160, 8 categorías, 2 turnos por pregunta**.
Si el conteo difiere, el archivo oficial está incompleto o corrupto.

---

## 10. Dataset MT-Bench — Preparación del subset

```bash
# Preparar subset: 5 preguntas por categoría = 40 preguntas totales
python scripts/prepare_mt_bench_subset.py --save

# Previsualizar sin guardar
python scripts/prepare_mt_bench_subset.py --dry-run --stats
```

El script genera tres archivos en `data/mt_bench/subset/`:

| Archivo | Descripción |
|---------|-------------|
| `mt_bench_literal_subset_5_per_category.jsonl` | subset en formato JSONL |
| `mt_bench_literal_subset_5_per_category.yaml` | subset en formato YAML |
| `mt_bench_subset_manifest.csv` | manifiesto con IDs seleccionados por categoría |

**Estrategia de selección:** `first_n_per_category` con `seed = 42`. Las primeras 5
preguntas de cada categoría en el orden del dataset oficial. Determinista: dada la
misma versión de `question.jsonl`, siempre produce el mismo subset.

**Categorías y mapeo interno:**

| Categoría oficial (MT-Bench) | Campo `original_category` | Campo `category` |
|------------------------------|--------------------------|------------------|
| writing | writing | writing |
| roleplay | roleplay | roleplay |
| extraction | extraction | extraction |
| reasoning | reasoning | reasoning |
| math | math | math |
| coding | coding | coding |
| stem | stem | stem |
| humanities | humanities | humanities_social_sciences |

El campo `original_category` preserva el valor literal del dataset oficial y es
el campo usado en todos los análisis y gráficas.

---

## 11. Verificación de integridad (SHA256)

```bash
# Verificar todos los artefactos y guardar manifiestos
python scripts/hash_artifacts.py --save

# Solo modelos GGUF
python scripts/hash_artifacts.py --models

# Solo dataset
python scripts/hash_artifacts.py --dataset

# Solo archivos de configuración
python scripts/hash_artifacts.py --config

# SHA256 completo para modelos (lento: ~60-90 s por archivo de 4-8 GB)
python scripts/hash_artifacts.py --full-hash --save
```

Por defecto se usa **SHA256 parcial** (primer 1 MB del archivo GGUF + tamaño total)
como huella de identificación de modelo. El header GGUF de 1 MB contiene la arquitectura,
la cuantización y los metadatos del modelo, y es suficiente para identificar
inequívocamente la versión sin leer el archivo completo.

Los manifiestos se guardan en:
- `results/metadata/model_hashes.csv` — huella de cada modelo GGUF
- `results/metadata/dataset_manifest.csv` — hash del dataset por pregunta

Ambos manifiestos se incluyen en `results/metadata/environment_report.json` para
asegurar la trazabilidad completa del experimento.

---

## 12. Validación del entorno

```bash
# Validación completa
python scripts/validate_config.py --all

# Validar solo backend GPU (Metal / CUDA)
python scripts/validate_config.py --check-backend

# Validar instalación de paquetes Python
python scripts/validate_config.py --check-installation

# Validar optimization_config.yaml (Fase 2)
python scripts/validate_config.py --check-optimization
```

El script emite `[PASS]`, `[WARN]`, `[FAIL]` o `[SKIP]` para cada verificación
y sale con código 0 (éxito) o 1 (al menos un FAIL). No ejecutar el experimento
si hay FAILs sin resolver.

Verificaciones incluidas:

| Sección | Qué verifica |
|---------|-------------|
| Config load | config.yaml parseable y campos obligatorios presentes |
| Hardware | hardware_profile declarado y consistente con la máquina |
| Models | rutas GGUF existen y archivos accesibles |
| Inference params | n_ctx, temperature, seed, generation_mode válidos |
| MT-Bench | question.jsonl presente; subset preparado; IDs válidos |
| Output dirs | results/ y subdirectorios creables |
| Backend GPU | Metal (macOS) o CUDA (Windows) correctamente configurado |
| Installation | llama-cpp-python, codecarbon, pandas, scipy importables |
| Optimization | optimization_config.yaml (si existe); forbidden_changes respetados |

---

## 13. Fase 1 — Ejecución del benchmark

```bash
# Dry run — validar sin ejecutar inferencias
python run_experiment.py --dry-run

# Setup only — preparar directorios y manifiestos sin inferencias
python run_experiment.py --setup-only

# Ejecución completa en GPU (perfil declarado en config.yaml)
python run_experiment.py

# Smoke test — 3 conversaciones, resultados en results/smoke_test/
python run_experiment.py --limit-conversations 3

# Con hardware profile y dispositivo explícitos
python run_experiment.py --hardware-profile mac_m4 --execution-device gpu
python run_experiment.py --hardware-profile mac_m4 --execution-device cpu
python run_experiment.py --hardware-profile windows_nvidia --execution-device gpu
python run_experiment.py --hardware-profile windows_nvidia --execution-device cpu

# Limitar repeticiones (útil para pruebas)
python run_experiment.py --repetitions 3

# Saltar validaciones iniciales (no recomendado)
python run_experiment.py --skip-checks
```

> **Smoke test:** si `--limit-conversations < 5` OR `--repetitions < 15`, los
> resultados se guardan en `results/smoke_test/` y no en `results/raw/`. Esto evita
> contaminar los resultados del experimento real con corridas de prueba.

**Escala del experimento** por sesión (1 hardware profile, 1 execution_device):

```
2 modelos × 2 cuantizaciones × 40 preguntas × 2 turnos × 15 repeticiones
= 2 400 conversaciones = 4 800 llamadas de inferencia local
```

El orden de ejecución es aleatorio por `seed = 42` cuando `randomize_order: true`
(valor por defecto). El orden exacto queda registrado en los timestamps del CSV.

---

## 14. Evaluación de calidad con Claude

```bash
# Requiere ANTHROPIC_API_KEY en el entorno
export ANTHROPIC_API_KEY=sk-ant-...

# Evaluar resultados de Fase 1
python judge_with_claude.py \
    --input results/raw/conversation_results.csv \
    --output-dir results/judge/ \
    --phase 1

# Evaluar con modelo Claude específico
python judge_with_claude.py \
    --input results/raw/conversation_results.csv \
    --output-dir results/judge/ \
    --model claude-sonnet-4-6

# Dry run — mostrar primeras N conversaciones sin llamar a la API
python judge_with_claude.py \
    --input results/raw/conversation_results.csv \
    --dry-run --max-rows 5

# Reanudación automática — saltar conversaciones ya evaluadas
python judge_with_claude.py \
    --input results/raw/conversation_results.csv \
    --output-dir results/judge/
```

La evaluación es **reanudable**: si el script se interrumpe, vuelve a ejecutarse
desde donde quedó sin duplicar evaluaciones.

**Dimensiones de evaluación (escala 1–10 cada una):**

| Campo CSV | Dimensión evaluada |
|-----------|-------------------|
| `score` | Puntuación global del juez (análoga a MT-Bench score) |
| `correctness` | Corrección factual |
| `instruction_following` | Seguimiento de instrucciones |
| `relevance` | Relevancia de la respuesta |
| `completeness` | Completitud |
| `clarity` | Claridad de expresión |
| `conciseness` | Concisión |
| `usefulness` | Utilidad práctica |
| `multi_turn_coherence` | Coherencia entre turno 1 y turno 2 |

**Evaluación con referencias:** para las categorías `math`, `reasoning`, `coding` y
`stem`, si existe `data/mt_bench/official/references.yaml`, el juez recibe la respuesta
de referencia además de la conversación evaluada.

> **Nota sobre sesgo de auto-evaluación:** Claude evalúa las respuestas de Llama y
> Qwen. No evalúa sus propias respuestas. Este diseño evita el sesgo de auto-preferencia
> documentado en LLM-as-a-Judge (Zheng et al., 2023).

---

## 15. Fase 1 — Análisis estadístico

```bash
# Generar todas las tablas de análisis (14 CSVs)
python analysis_summary.py

# Con rutas explícitas
python analysis_summary.py \
    --conv-csv results/raw/conversation_results.csv \
    --judge-csv results/judge/judge_results.csv \
    --output-dir results/summary/

# Filtrar por hardware profile
python analysis_summary.py --hardware-profile mac_m4
python analysis_summary.py --hardware-profile windows_nvidia

# Filtrar por dispositivo
python analysis_summary.py --execution-device gpu
python analysis_summary.py --execution-device cpu

# Filtrar por fase
python analysis_summary.py --phase 1
```

**Tablas generadas en `results/summary/`:**

| Archivo | Contenido |
|---------|-----------|
| `summary_by_model.csv` | Métricas promedio por modelo |
| `summary_by_model_quantization.csv` | Métricas promedio por modelo+cuantización |
| `summary_by_model_quantization_device.csv` | Idem + dispositivo |
| `summary_by_hardware_profile.csv` | Métricas por perfil de hardware |
| `summary_by_execution_device.csv` | Métricas por dispositivo (CPU vs GPU) |
| `summary_by_category.csv` | Métricas promedio por categoría MT-Bench |
| `summary_by_model_category.csv` | Métricas por modelo × categoría |
| `summary_full_ranking.csv` | Ranking completo con `config_id` para Fase 2 |
| `summary_best_efficiency_by_category.csv` | Configuración más eficiente por categoría |
| `summary_statistical_tests.csv` | Resultados de tests estadísticos |
| `paper_table_energy.csv` | Tabla para publicación: energía |
| `paper_table_quality.csv` | Tabla para publicación: calidad |
| `paper_table_efficiency.csv` | Tabla para publicación: eficiencia |
| `paper_table_pareto.csv` | Tabla para publicación: frente de Pareto |

---

## 16. Fase 1 — Gráficas

```bash
# Generar todas las gráficas (resultados en results/plots/)
python analysis_plots.py

# Con métrica de energía primaria explícita
python analysis_plots.py --primary-energy corrected

# Filtros disponibles
python analysis_plots.py --hardware-profile mac_m4
python analysis_plots.py --execution-device gpu
python analysis_plots.py --category math
python analysis_plots.py --model-name qwen2.5-7b
python analysis_plots.py --phase 1

# Comparación Fase 1 vs Fase 2 (requiere resultados de ambas fases)
python analysis_plots.py --optimization
```

**Familias de gráficas generadas:**

| Directorio | Gráficas |
|------------|---------|
| `results/plots/energy/` | Energía por conversación, Wh/1K tokens, J/token, tokens/J — por categoría y por config |
| `results/plots/quality/` | Score y 8 subdimensiones por categoría; score por config; overview de subdimensiones |
| `results/plots/efficiency/` | Quality/J, Quality/Wh, EDP, SCI — por categoría y por config |
| `results/plots/latency/` | Latencia por categoría y por config |
| `results/plots/pareto/` | Scatter Pareto (Quality vs Energy); frontera de Pareto en línea discontinua; puntos Pareto-eficientes con borde dorado |
| `results/plots/boxplots/` | Distribuciones de energía, latencia, score, Quality/J, EDP — por config |
| `results/plots/heatmaps/` | Score/energía/Wh-per-1K × categoría × config |
| `results/plots/ranking/` | Rankings horizontales: calidad, energía, tokens/J, Quality/J, EDP, SCI |

---

## 17. Fase 2 — Optimización del ganador

```bash
# Dry run — mostrar qué se ejecutaría sin lanzar inferencias
python optimize_winner.py --dry-run

# Ejecución completa
python optimize_winner.py

# Smoke test — 2 repeticiones Stage A, sin Stage B
python optimize_winner.py --smoke-test

# Ejecutar solo Stage A (screening rápido)
python optimize_winner.py --stage a

# Ejecutar solo Stage B (validación; requiere Stage A completado)
python optimize_winner.py --stage b

# Sin reanudación — reiniciar desde cero
python optimize_winner.py --no-resume
```

**Protocolo de dos etapas:**

| Etapa | Repeticiones | Propósito |
|-------|-------------|-----------|
| **Stage A** | 5 por trial (default) | Screening rápido; descarte de configuraciones sub-óptimas |
| **Stage B** | 15 por trial (default) | Validación completa de los candidatos supervivientes de Stage A |

**Restricción de calidad P22** (obligatoria para seleccionar ganador):

```
mean_quality ≥ baseline_quality − 0.25  AND
mean_quality ≥ baseline_quality × 0.97
```

El ganador de Fase 2 debe satisfacer ambas condiciones. Si ninguna configuración
las cumple, el script lo reporta explícitamente y **no declara ganador**.

> **Restricciones de optimización (forbidden_changes):** la búsqueda de parámetros
> está restringida por `forbidden_changes` en `optimization_config.yaml`. Por defecto
> se prohíbe modificar el modelo, la cuantización, los prompts y `temperature`.
> Solo se optimizan parámetros que afectan la eficiencia de inferencia: `n_ctx`,
> `n_batch`, `n_threads`, `max_tokens`.

**Modos de selección del baseline para Fase 2:**

| Modo | Descripción |
|------|-------------|
| `auto_best_pareto` | Configuración Pareto-eficiente con mayor calidad en el frente |
| `auto_best_quality_under_energy_constraint` | Mayor calidad sin exceder P75 de energía |
| `explicit:<config_id>` | config_id explícito de `summary_full_ranking.csv` |

---

## 18. Fase 2 — Análisis y gráficas

```bash
# Generar gráficas de Fase 2
python optimization_plots.py

# Con ruta explícita al reporte JSON
python optimization_plots.py \
    --report results/optimization/optimization_report.json \
    --output-dir results/optimization/plots/
```

**Gráficas generadas:**

| Archivo | Contenido |
|---------|-----------|
| `optimization_quality_vs_energy.png` | Calidad vs energía por trial (Stage A + B) |
| `optimization_pareto_front.png` | Frente de Pareto Stage B (calidad vs energía) |
| `optimization_energy_reduction.png` | % reducción energética por trial vs baseline |
| `optimization_quality_constraint.png` | Calidad por trial con línea de constraint P22 |
| `optimization_latency_vs_energy.png` | Latencia vs energía por trial |
| `optimization_tokens_per_joule.png` | Tokens/Joule por trial |

**Salidas en `results/optimization/`:**

```
stage_a/
    turn_results.csv|jsonl
    conversation_results.csv|jsonl
    judge_results.csv|jsonl
    stage_a_summary.csv
stage_b/
    turn_results.csv|jsonl
    conversation_results.csv|jsonl
    judge_results.csv|jsonl
    stage_b_summary.csv
optimization_report.json
optimization_report.md
```

---

## 19. Diseño experimental — Fase 1

### Escala por sesión de medición

| Parámetro | Valor |
|-----------|-------|
| Categorías MT-Bench | 8 (todas las oficiales) |
| Preguntas por categoría | 5 (subset determinista) |
| **Total prompts** | **40** |
| Turnos por pregunta | 2 |
| Repeticiones por pregunta | 15 |
| **Conversaciones por configuración** | **40 × 15 = 600** |
| **Llamadas de inferencia por configuración** | **600 × 2 = 1 200** |

### Escenarios de ejecución

**Escenario A — 1 dispositivo, 1 hardware profile** (p. ej. Mac M4 GPU):

```
2 modelos × 2 cuantizaciones × 1 dispositivo
× 40 preguntas × 2 turnos × 15 repeticiones
= 2 400 conversaciones = 4 800 llamadas de inferencia
```

**Escenario B — CPU + GPU, 1 hardware profile** (Mac M4 CPU y GPU):

```
2 modelos × 2 cuantizaciones × 2 dispositivos
× 40 preguntas × 2 turnos × 15 repeticiones
= 4 800 conversaciones = 9 600 llamadas de inferencia
```

**Escenario C — CPU + GPU, 2 hardware profiles** (Mac M4 + Windows RTX 4060):

```
2 modelos × 2 cuantizaciones × 2 dispositivos × 2 hardware profiles
× 40 preguntas × 2 turnos × 15 repeticiones
= 9 600 conversaciones = 19 200 llamadas de inferencia
```

Cada sesión ejecuta **un solo dispositivo y hardware profile**. Para comparar CPU vs GPU
o dos máquinas distintas, se ejecutan sesiones separadas y se combinan los CSV para el
análisis usando `--hardware-profile` y `--execution-device` como filtros.

### Parámetros de inferencia fijos

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `n_ctx` | 4096 | Mínimo seguro para MT-Bench multi-turno (T2 concatena T1 prompt + T1 respuesta + T2 pregunta) |
| `max_tokens` | 1024 | Límite de tokens de salida por turno |
| `temperature` | 0.0 | Decodificación greedy; determinismo completo |
| `top_p` | 1.0 | Sin filtrado nucleus |
| `seed` | 42 | Semilla fija; no varía entre repeticiones |
| `n_threads` | 8 (mac_m4), 12 (windows_nvidia) | Según perfil de hardware |
| `n_batch` | 512 | Tamaño de batch de procesamiento |
| `n_gpu_layers` | -1 (GPU), 0 (CPU) | Todas las capas en GPU; ninguna en CPU |
| `echo` | false | Solo se miden tokens de completion, no de prompt |
| `on_context_overflow` | skip | Conservativo: no truncar silenciosamente |

---

## 20. Principios metodológicos

| Código | Principio | Descripción |
|--------|-----------|-------------|
| **P1** | Solo se mide la inferencia | `tracker.start()` justo antes de la llamada al modelo; `tracker.stop()` inmediatamente después. Sin operaciones intermedias. |
| **P2** | Carga del modelo fuera del tracker | El modelo se carga en memoria antes de iniciar la medición. La carga no se incluye en la energía reportada. |
| **P3** | Cooldown y warmup fuera del tracker | Los sleeps de cooldown y las corridas de warmup ocurren fuera del intervalo de medición. |
| **P4** | Juez separado de medición | La evaluación Claude corre después de todas las inferencias; no hay overlap con la medición energética. |
| **P5** | Prompts verbatim | Ningún prompt de MT-Bench es modificado, resumido, reescrito ni traducido. |
| **P6** | Prompts en inglés | MT-Bench es en inglés. No se realizan traducciones. |
| **P7** | Abort sin dataset oficial | Si `question.jsonl` no existe, el experimento falla con error explícito. No hay prompts alternativos. |
| **P8** | Estructura de 2 turnos preservada | Cada conversación tiene exactamente turno 1 y turno 2. El contexto de T1 se incluye en T2. |
| **P9** | Baseline antes de cada configuración | La potencia idle se mide antes de cargar cada combinación modelo+cuantización+dispositivo. |
| **P10** | CSV con energía medida y corregida | Cada fila incluye `measured_energy` y `baseline_corrected_energy`. Nunca se sobrescribe la medida bruta. |
| **P11** | SHA256 parcial para modelos | SHA256 del primer 1 MB del GGUF + tamaño total como huella de identificación. Suficiente para identificar la versión. |
| **P12** | Context overflow detectado y registrado | Cuando el prompt supera `n_ctx − max_tokens`, la fila se guarda con `status = context_overflow` y energía en null. |
| **P13** | `finish_reason` en cada fila | Se registra la razón de fin de generación: `stop`, `length`, `context_overflow`. |
| **P14** | `turn_id` determinista | ID de turno calculado como hash del prompt; permite reanudación sin duplicados. |
| **P22** | Restricción de calidad en Fase 2 | El ganador de Fase 2 debe cumplir: `mean_quality ≥ baseline − 0.25` AND `mean_quality ≥ baseline × 0.97`. |

---

## 21. Métricas de energía: medida vs. corregida por baseline

### Dos métricas paralelas

Todos los resultados incluyen **dos métricas de energía**. Ambas siempre se reportan.

| Tipo | Campo CSV | Descripción |
|------|-----------|-------------|
| **Medida** | `total_measured_energy_wh` / `_joules` | Energía total registrada por CodeCarbon durante la inferencia. Incluye consumo de fondo del sistema (SO, periféricos, procesos). |
| **Corregida** | `total_baseline_corrected_energy_wh` / `_joules` | Energía neta atribuible al workload LLM. Se descuenta la potencia idle medida en reposo antes de la configuración. Puede ser `null` si la corrección resulta negativa (ruido de medición). |

### Fórmula de corrección baseline

```
baseline_corrected_energy_joules =
    measured_energy_joules − baseline_power_watts × latency_seconds
```

Si el valor resultante es negativo (la potencia idle supera la potencia de inferencia),
se almacena `null` en el CSV y se emite una advertencia. La energía bruta siempre
se preserva.

### Calibración del baseline idle

| Parámetro (`config.yaml`) | Valor por defecto | Descripción |
|--------------------------|-------------------|-------------|
| `baseline_idle_seconds` | 60 | Duración de cada medición idle |
| `baseline_repetitions` | 3 | Repeticiones; se usa el promedio de las 3 |
| `save_baseline_runs` | true | Cada corrida se guarda en `results/raw/baseline_results.csv` |

Tiempo de calibración por configuración: `3 × (20 s pausa + 60 s medición) ≈ 4 min`.

### Métrica primaria en análisis

La métrica primaria en gráficas y tablas es `baseline_corrected_energy` (configurable
en `config.yaml` con `analysis.primary_energy_metric`). Esta elección sigue la práctica
de "idle subtraction" estándar en la literatura de eficiencia energética de software
(Bannour et al., EMNLP 2021; Anthony et al., ICML Workshop 2020).

Para comparar sesiones distintas (p. ej. Mac M4 vs Windows), **ambas métricas deben
reportarse** para verificar que la potencia idle sea estable entre sesiones.

---

## 22. Métricas de eficiencia: Quality/Joule, EDP, SCI, Pareto

### Quality-per-Joule

Adaptación operacional de Accuracy-per-Joule (Canziani et al., 2017, arXiv:1605.07678):

```
quality_per_joule = mean_quality_score / mean_baseline_corrected_energy_joules
```

Interpretación: puntuación de calidad obtenida por cada Joule de energía consumida.
Mayor es mejor. Permite comparar configuraciones que difieren en calidad Y en energía
en una sola dimensión.

| Campo CSV | Descripción |
|-----------|-------------|
| `quality_per_joule_corrected` | Quality / J (energía corregida) |
| `quality_per_joule_measured` | Quality / J (energía medida) |
| `quality_per_wh_corrected` | Quality / Wh (energía corregida) |
| `quality_per_1k_token_wh_corrected` | Quality / (Wh por 1K tokens) |

### EDP — Energy-Delay Product

```
EDP = energy_joules × latency_seconds
```

Penaliza simultáneamente el alto consumo y la alta latencia. Menor es mejor.
Útil cuando tanto la energía como el tiempo de respuesta son restricciones operativas.

| Campo CSV | Descripción |
|-----------|-------------|
| `total_edp_joule_second_corrected` | EDP con energía corregida por baseline |
| `total_edp_joule_second_measured` | EDP con energía medida |

### SCI — Software Carbon Intensity

```
SCI = (E × I + M) / R
```

Donde `E` = energía operacional (kWh), `I` = intensidad de carbono de la red eléctrica
(gCO₂eq/kWh), `M` = carbono embodied (establecido en 0 en este experimento por falta
de datos de fabricación), `R` = unidad funcional (conversación o 1K tokens de salida).

El factor de intensidad de carbono usado es el de Paraguay: **26 gCO₂eq/kWh**
(EMBER, 2024 — energía predominantemente hidroeléctrica). Este valor se declara
en `config.yaml` (`codecarbon.country_iso_code: "ESP"` puede ajustarse al país del
hardware donde corre el experimento).

| Campo CSV | Descripción |
|-----------|-------------|
| `total_operational_sci_per_conversation` | gCO₂eq por conversación |
| `total_operational_sci_per_1k_output_tokens` | gCO₂eq por 1K tokens de salida |

### Análisis de Pareto

El frente de Pareto se calcula sobre 3 objetivos:

```
Maximizar: quality_score
Minimizar: baseline_corrected_energy_wh
Minimizar: total_latency_seconds
```

Una configuración es Pareto-eficiente si no existe otra que la domine en los 3
objetivos simultáneamente. En las gráficas de Fase 1, los puntos Pareto-eficientes
se marcan con **borde dorado**. La frontera 2-objetivo (calidad vs energía) se traza
como línea discontinua negra.

Para Fase 2, el frente de Pareto se recalcula sobre los trials de Stage B y se
compara con el baseline de Fase 1.

---

## 23. Tests estadísticos y método CI

### Tests utilizados

| Test | Propósito | Paquete |
|------|-----------|---------|
| **Mann-Whitney U** | Comparación de dos configuraciones independientes (p. ej. Q4 vs Q8) | `scipy.stats.mannwhitneyu` |
| **Wilcoxon signed-rank** | Comparación de pares de mediciones del mismo prompt | `scipy.stats.wilcoxon` |
| **Friedman** | Comparación de 3+ configuraciones sobre los mismos prompts | `scipy.stats.friedmanchisquare` |
| **Bootstrap CI (percentil)** | Intervalo de confianza 95% para medias y diferencias | implementado en `analysis_summary.py` |
| **Holm-Bonferroni** | Corrección de comparaciones múltiples | `scipy.stats` |
| **t-test de no-inferioridad** | Validación P22 en Fase 2 | `scipy.stats.ttest_1samp` (una cola) |

### Método CI Bootstrap

```
Iteraciones: 10 000
α: 0.05
IC: percentil 2.5–97.5 del bootstrap
```

Los CI se reportan en todas las tablas de `analysis_summary.py` y en las barras de
error de todas las gráficas de `analysis_plots.py`.

### Corrección de comparaciones múltiples

Con `k` configuraciones comparadas, se aplica la corrección de Holm-Bonferroni sobre
los p-valores de los tests pareados para controlar el FWER (Family-Wise Error Rate).
El umbral efectivo por comparación varía según el ranking del p-valor.

---

## 24. Modelos y tipos de modelo

| Modelo | Cuantización | Archivo GGUF | `model_type` | Fuente |
|--------|-------------|-------------|--------------|--------|
| llama-2-7b | Q4_K_M | `models/llama-2-7b/Q4_K_M.gguf` | `base` | TheBloke/Llama-2-7B-GGUF |
| llama-2-7b | Q8_0 | `models/llama-2-7b/llama-2-7b.Q8_0.gguf` | `base` | TheBloke/Llama-2-7B-GGUF |
| qwen2.5-7b | Q4_K_M | `models/qwen2.5-7b/Qwen2.5-7B-Instruct-Q4_K_M.gguf` | `instruct` | Qwen/Qwen2.5-7B-Instruct-GGUF |
| qwen2.5-7b | Q8_0 | `models/qwen2.5-7b/Qwen2.5-7B-Instruct-Q8_0.gguf` | `instruct` | Qwen/Qwen2.5-7B-Instruct-GGUF |

El campo `model_type` se registra en cada fila del CSV de mediciones. El valor
`chat_or_base_to_be_confirmed` indica que el tipo no pudo determinarse automáticamente
y debe especificarse manualmente antes del análisis.

### Descarga de modelos

Los archivos GGUF no están incluidos en el repositorio. Descargar desde Hugging Face:

```bash
# Usando huggingface-hub
pip install huggingface-hub
huggingface-cli download TheBloke/Llama-2-7B-GGUF \
    llama-2-7b.Q4_K_M.gguf --local-dir models/llama-2-7b/
huggingface-cli download TheBloke/Llama-2-7B-GGUF \
    llama-2-7b.Q8_0.gguf --local-dir models/llama-2-7b/
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
    Qwen2.5-7B-Instruct-Q4_K_M.gguf --local-dir models/qwen2.5-7b/
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
    Qwen2.5-7B-Instruct-Q8_0.gguf --local-dir models/qwen2.5-7b/
```

Verificar SHA256 después de la descarga:

```bash
python scripts/hash_artifacts.py --models --save
```

---

## 25. Limitaciones metodológicas

### Comparación base vs. instruct en MT-Bench

> **ADVERTENCIA:** `llama-2-7b` es un modelo **base** (sin instruction tuning).
> `qwen2.5-7b` es un modelo **instruct**. MT-Bench fue diseñado para evaluar
> asistentes conversacionales instruction-tuned. Esta asimetría favorece
> estructuralmente a qwen2.5-7b en todas las métricas de calidad.

- Los puntajes de calidad de `qwen2.5-7b` son directamente comparables con otros
  modelos instruct evaluados en MT-Bench.
- Los puntajes de `llama-2-7b` reflejan calidad de completación de texto plano,
  no de asistencia conversacional. **No son comparables directamente** con los
  puntajes instruct bajo las mismas condiciones.
- Esta limitación **no afecta** las comparaciones de energía y latencia sin calidad:
  ambos modelos ejecutan inferencia con el mismo proceso de medición.
- Para una comparación de calidad sin sesgo, reemplazar `llama-2-7b` por
  `llama-2-7b-chat` (disponible como `TheBloke/Llama-2-7B-Chat-GGUF`).

### Chat templates y sesgo energético

Usar un chat template incorrecto para un modelo instruct o chat introduce sesgo
energético: el modelo puede producir secuencias más largas para "recuperarse" de
una entrada malformada, aumentando el consumo medido de forma no atribuible a la
cuantización.

| Modelo | Template esperado | Tokens especiales |
|--------|------------------|-------------------|
| qwen2.5-7b (Instruct) | ChatML | `<\|im_start\|>`, `<\|im_end\|>` |
| llama-2-7b-chat (Chat) | Llama-2 Chat | `[INST]`, `[/INST]`, `<<SYS>>` |
| llama-2-7b (base) | Sin template — texto plano | — |

### Política de context overflow

Con `n_ctx = 4096` y la política `on_context_overflow: skip`, las conversaciones
cuyo prompt supere el contexto disponible se marcan con `status = context_overflow`
y se excluyen del análisis energético. Reportar el porcentaje de overflows es
obligatorio para la reproducibilidad del experimento.

---

## 26. Advertencias sobre CodeCarbon

> **CodeCarbon es un estimador de software, no un medidor físico.**

Los valores de energía reportados por CodeCarbon se calculan a partir del TDP (Thermal
Design Power) del procesador, no de una medición eléctrica física directa. Las
implicaciones son:

1. **Subestimación o sobreestimación posible:** el TDP es el consumo máximo de diseño,
   no el consumo real en un workload específico. La carga de trabajo de inferencia LLM
   puede operar bien por debajo o cerca del TDP según la cuantización y el modelo.

2. **GPU Apple M4 (Metal):** CodeCarbon no mide energía de la GPU Apple Silicon
   directamente. Usa un estimador basado en el uso de CPU + estimados del fabricante.
   Para Apple M4, la potencia GPU se aproxima como fracción del TDP total del chip.

3. **GPU NVIDIA (CUDA):** CodeCarbon puede acceder a energía GPU via `pynvml` si está
   instalado. Si `pynvml` no está disponible, recae en estimación por TDP.

4. **Reproducibilidad relativa:** aunque los valores absolutos no son precisos a nivel
   de vatios, las **diferencias relativas** entre configuraciones ejecutadas en el mismo
   hardware y con el mismo entorno son reproducibles y comparables.

5. **Validación con medidor externo:** para valores físicos absolutos, usar un wattímetro
   externo (p. ej. Yokogawa WT310, Monsoon Power Monitor). Este experimento **no usa**
   medidor externo; todos los valores de energía deben reportarse como estimaciones
   basadas en TDP con CodeCarbon v3.x.

**Reportar siempre:** versión de CodeCarbon, sistema operativo, hardware profile,
y si `pynvml` estaba disponible durante la medición.

---

## 27. Modo de generación y propósito de las repeticiones

### Modo determinístico (por defecto)

```yaml
generation_mode:
  name: deterministic_energy
  vary_seed_by_repetition: false
```

Con `temperature = 0.0` y semilla fija (`seed = 42`), el modelo usa decodificación
**greedy**: dado el mismo prompt, produce exactamente la misma secuencia de tokens
en cada repetición. Las 15 repeticiones **miden estabilidad energética y de latencia**,
no diversidad semántica.

| Las 15 repeticiones miden | Lo que NO miden |
|---------------------------|-----------------|
| Variabilidad del consumo energético entre corridas | Diversidad de respuestas del modelo |
| Estabilidad de latencia (CV% del tiempo de inferencia) | Calidad promedio sobre distintas generaciones |
| Reproducibilidad del benchmark energético | Distribución estocástica de outputs |

La estabilidad se reporta como **Coeficiente de Variación (CV%)** de la latencia por
pregunta. Un CV < 10% indica condiciones de medición estables.

### Modo estocástico (opcional, no recomendado para comparación energética)

```yaml
generation_mode:
  name: stochastic_quality_variability
  temperature: 0.7
  vary_seed_by_repetition: true
```

> **Este modo NO debe usarse para la comparación energética principal.** Cambiar
> `temperature` altera la distribución de tokens generados, modifica la longitud
> de salida y, por tanto, el consumo energético de forma no controlada. Las diferencias
> de energía observadas entre configuraciones en este modo pueden reflejar la longitud
> variable de los outputs, no la eficiencia de la cuantización.

---

## 28. Reproducibilidad y entorno de referencia

### Entorno exacto del experimento de referencia

| Campo | Valor |
|-------|-------|
| Hardware | Apple Mac Mini (Late 2024) |
| Chip | Apple M4 (10-core CPU + 10-core GPU) |
| RAM | 16 GB LPDDR5 unificada |
| macOS | 15.6.1 |
| Python | 3.13.5 (CPython) |
| llama-cpp-python | 0.3.21 |
| CodeCarbon | 3.2.6 |
| Backend GPU | Metal |
| `n_gpu_layers` (GPU) | -1 (todas las capas) |

### Pasos para reproducir el entorno exacto

```bash
# 1. Instalar Python 3.13.5 (o la versión del lock file)
# 2. Crear venv y activar
python3.13 -m venv .venv && source .venv/bin/activate

# 3. Instalar desde el lock file (versiones exactas)
pip install -r requirements-lock.txt

# 4. Instalar llama-cpp-python con Metal (macOS)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python==0.3.21 \
    --force-reinstall --no-cache-dir

# 5. Verificar que el entorno coincide
python scripts/setup_entorno.py
diff <(pip freeze) requirements-lock.txt
```

### Archivos de reproducibilidad generados

| Archivo | Contenido |
|---------|-----------|
| `results/metadata/environment_report.json` | Hardware, OS, Python, versiones de paquetes, huellas SHA256 de modelos |
| `results/metadata/config_snapshot.yaml` | Copia exacta de `config.yaml` al momento del experimento |
| `results/metadata/model_hashes.csv` | SHA256 parcial + tamaño de cada modelo GGUF |
| `results/metadata/dataset_manifest.csv` | SHA256 del dataset por pregunta |
| `results/metadata/requirements_freeze.txt` | Salida de `pip freeze` en el momento del experimento |
| `results/metadata/experiment_manifest.json` | ID del experimento, timestamp, subset seleccionado |

---

## 29. Archivos de salida esperados

### Fase 1 — Medición

```
results/raw/
    conversation_results.csv     # 1 fila por conversación (2 turnos agregados)
    turn_results.csv             # 1 fila por turno de inferencia
    baseline_results.csv         # mediciones de baseline idle
results/metadata/
    environment_report.json
    config_snapshot.yaml
    model_hashes.csv
    dataset_manifest.csv
    requirements_freeze.txt
    experiment_manifest.json
```

### Fase 1 — Calidad (después de judge_with_claude.py)

```
results/judge/
    judge_results.csv            # 1 fila por conversación, 9 dimensiones + status
    judge_results.jsonl          # mismo contenido en JSONL (append mode)
```

### Fase 1 — Análisis (después de analysis_summary.py y analysis_plots.py)

```
results/summary/
    summary_by_model.csv
    summary_by_model_quantization.csv
    summary_by_model_quantization_device.csv
    summary_by_hardware_profile.csv
    summary_by_execution_device.csv
    summary_by_category.csv
    summary_by_model_category.csv
    summary_full_ranking.csv        # incluye config_id para selección Fase 2
    summary_best_efficiency_by_category.csv
    summary_statistical_tests.csv
    paper_table_energy.csv
    paper_table_quality.csv
    paper_table_efficiency.csv
    paper_table_pareto.csv
results/plots/
    energy/    quality/    efficiency/    latency/
    pareto/    boxplots/   heatmaps/     ranking/
```

### Fase 2 — Optimización (después de optimize_winner.py y optimization_plots.py)

```
results/optimization/
    stage_a/
        turn_results.csv|jsonl
        conversation_results.csv|jsonl
        judge_results.csv|jsonl
        stage_a_summary.csv
    stage_b/
        turn_results.csv|jsonl
        conversation_results.csv|jsonl
        judge_results.csv|jsonl
        stage_b_summary.csv
    optimization_report.json
    optimization_report.md
    plots/
        optimization_quality_vs_energy.png
        optimization_pareto_front.png
        optimization_energy_reduction.png
        optimization_quality_constraint.png
        optimization_latency_vs_energy.png
        optimization_tokens_per_joule.png
```

---

## 30. Checklist de reporte

Esta lista debe completarse antes de publicar o presentar los resultados del experimento.
Cada ítem debe estar documentado en `results/metadata/environment_report.json` o en el
texto del reporte.

### Entorno y hardware

- [ ] **Hardware profile** declarado: `mac_m4` / `windows_nvidia` / otro
- [ ] **Modelo de hardware exacto** (p. ej. Apple Mac Mini Late 2024, Dell XPS con RTX 4060)
- [ ] **OS y versión** (p. ej. macOS 15.6.1, Windows 11 22H2)
- [ ] **Versión de Python** (mayor.menor.micro, p. ej. 3.13.5)
- [ ] **Archivo de dependencias lock** archivado (`requirements-lock.txt` o `requirements_freeze.txt`)
- [ ] **Versión de CodeCarbon** reportada (p. ej. 3.2.6)
- [ ] **Versión de llama-cpp-python** reportada (p. ej. 0.3.21)
- [ ] **Backend de aceleración** declarado: Metal / CUDA / CPU-only
- [ ] **`pynvml` disponible** (relevante para precisión de energía GPU en NVIDIA): sí / no

### Modelos

- [ ] **Nombre del modelo** para cada configuración (p. ej. `llama-2-7b`, `qwen2.5-7b`)
- [ ] **Tipo de modelo** (`base`, `instruct`, `chat`) declarado por configuración
- [ ] **Cuantización** declarada (`Q4_K_M`, `Q8_0`) por configuración
- [ ] **Ruta al archivo GGUF** por configuración
- [ ] **SHA256 del modelo GGUF** reportado (parcial 1 MB + tamaño total, o completo)
- [ ] **Fuente del modelo** (repositorio Hugging Face, commit o tag)

### Dataset

- [ ] **SHA256 de `question.jsonl`** reportado (SHA256 completo del archivo oficial)
- [ ] **IDs de preguntas seleccionadas** por categoría (p. ej. "writing: 81–85")
- [ ] **Número de categorías** usadas (debe ser 8 para usar el nombre MT-Bench)
- [ ] **Preguntas por categoría** (5 en este experimento)
- [ ] **Turnos por pregunta** (2, estructura original MT-Bench)
- [ ] **Estrategia de selección** del subset (`first_n_per_category`, seed=42)

### Configuración de inferencia

- [ ] **Repeticiones por prompt** (15 en este experimento)
- [ ] **Parámetros de generación** reportados: `temperature`, `top_p`, `seed`
- [ ] **Longitud de contexto** declarada: `n_ctx = 4096`
- [ ] **Máximo de tokens de salida** declarado: `max_tokens = 1024`
- [ ] **`n_threads`** y **`n_batch`** por perfil de hardware
- [ ] **`n_gpu_layers`** por modo de ejecución (−1 para GPU, 0 para CPU)

### Condiciones de medición

- [ ] **Valor de cooldown** entre corridas declarado (segundos)
- [ ] **Configuración de warmup** declarada: número de corridas, si se descartan
- [ ] **Configuración de baseline idle** declarada: `baseline_idle_seconds`, `baseline_repetitions`
- [ ] **Corrección de baseline aplicada**: sí / no (debe ser sí para `baseline_corrected_energy`)
- [ ] **Porcentaje de context overflows** reportado por configuración
- [ ] **Validación con medidor externo**: sí / no (si no: declarar explícitamente que CodeCarbon es un estimador TDP-based)

### Evaluación de calidad

- [ ] **Modelo juez** declarado (p. ej. `claude-sonnet-4-6`)
- [ ] **Modo de prompt del juez** declarado (`claude_multidimensional`)
- [ ] **Juicio con referencias** declarado: sí / no; categorías con referencia (math, reasoning, coding, stem)
- [ ] **Número de conversaciones evaluadas** vs total generadas

### Análisis estadístico

- [ ] **Tests estadísticos usados** declarados (Mann-Whitney U, Wilcoxon, Friedman, Bootstrap CI, Holm)
- [ ] **Método CI** declarado: Bootstrap percentil, 10 000 iteraciones, α = 0.05
- [ ] **Corrección de comparaciones múltiples**: Holm-Bonferroni aplicada
- [ ] **Criterios de Pareto** declarados: Quality↑, Energy↓, Latency↓

### Fase 2 (si aplica)

- [ ] **Criterio de selección del ganador Fase 1** declarado (`auto_best_pareto` / `auto_best_quality_under_energy_constraint` / `explicit:<config_id>`)
- [ ] **Método de búsqueda** declarado: `grid_search` / `random_search`
- [ ] **Espacio de búsqueda** declarado (rangos de cada parámetro)
- [ ] **Restricción de calidad P22** reportada: `mean_quality ≥ baseline − 0.25` AND `≥ baseline × 0.97`
- [ ] **Resultado del test de no-inferioridad** (p-valor del t-test unilateral)
- [ ] **Parámetros optimizados finales** reportados: `n_ctx`, `n_batch`, `n_threads`, `max_tokens`
- [ ] **% de reducción de energía** (baseline Fase 1 vs ganador Fase 2)
- [ ] **% de reducción de latencia** (baseline Fase 1 vs ganador Fase 2)
- [ ] **Calidad del ganador Fase 2** comparada con baseline Fase 1

---

## 31. Referencias

- Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). **Judging LLM-as-a-judge with
  MT-Bench and Chatbot Arena.** NeurIPS 2023.
  https://arxiv.org/abs/2306.05685

- Bannour, N., Ghannay, S., Névéol, A., Ligozat, A.-L. (2021). **Evaluating the
  carbon footprint of NLP methods: a survey and analysis of existing tools.**
  EMNLP 2021.

- Anthony, L. F. W., Kanding, B., Selvan, R. (2020). **Carbontracker: Tracking and
  Predicting the Carbon Footprint of Training Deep Learning Models.**
  ICML Workshop on Challenges in Deploying and Monitoring Machine Learning Systems.
  https://arxiv.org/abs/2007.03051

- Canziani, A., Paszke, A., Culurciello, E. (2017). **An Analysis of Deep Neural
  Network Models for Practical Applications.** arXiv:1605.07678.

- Lottick, K., Susai, S., Friedler, S. A., Wilson, J. P. (2019). **Energy Usage
  Reports: Environmental awareness as part of algorithmic accountability.**
  NeurIPS Workshop on Tackling Climate Change with Machine Learning.

- Strubell, E., Ganesh, A., McCallum, A. (2019). **Energy and Policy Considerations
  for Deep Learning in NLP.** ACL 2019. https://arxiv.org/abs/1906.02629

- EMBER (2024). **Global Electricity Review 2024.** Carbon intensity by country.
  https://ember-climate.org/

- llama.cpp (2024). **GGUF format specification.**
  https://github.com/ggerganov/llama.cpp

- CodeCarbon (2024). **CodeCarbon: Tracking Carbon Emissions from Compute.**
  https://github.com/mlco2/codecarbon

---

*Proyecto GREEN-IA — Maestría Ciencia de Datos — Universidad Comunero — Paraguay*
*Autor: Andres Villamayor | andres.villamayor@gmail.com*

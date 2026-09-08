# GREEN-IA: Consumo Energético en Inferencia de LLMs Cuantizados

**Tesis de Maestría en Ciencia de Datos — Universidad Comunero, Paraguay**
**Autor: Andres Villamayor**

Estudio experimental del consumo energético de la inferencia local de modelos de
lenguaje (LLMs) bajo dos niveles de cuantización (Q4 vs Q8), ejecutado en un Mac
Mini Apple Silicon (M4, 16 GB). La energía se mide con CodeCarbon y `powermetrics`;
la calidad de las respuestas se evalúa con Claude como juez (LLM-as-a-Judge) sobre
un subset de MT-Bench.

---

## Tabla de contenidos

1. [Modelos](#1-modelos)
2. [Benchmark: subset de MT-Bench](#2-benchmark-subset-de-mt-bench)
3. [Protocolo de medición](#3-protocolo-de-medición)
4. [Estado actual del experimento](#4-estado-actual-del-experimento)
5. [Estructura del proyecto](#5-estructura-del-proyecto)
6. [Instalación (macOS Apple Silicon)](#6-instalación-macos-apple-silicon)
7. [Uso](#7-uso)
8. [Limitaciones conocidas](#8-limitaciones-conocidas)
9. [Referencias](#9-referencias)

---

## 1. Modelos

| Modelo | Tipo | Cuantizaciones | Fuente (Hugging Face) |
|--------|------|-----------------|------------------------|
| Llama-2-7B | base (Meta) | Q4_K_M, Q8_0 | `TheBloke/Llama-2-7B-GGUF` |
| Qwen2.5-7B-Instruct | instruct (Alibaba) | Q4_K_M, Q8_0 | `bartowski/Qwen2.5-7B-Instruct-GGUF` |

> **Nota:** `models/qwen2.5-7b/*.gguf` son symlinks al caché local de Hugging
> Face; `readlink -f` resuelve a
> `~/.cache/huggingface/hub/models--bartowski--Qwen2.5-7B-Instruct-GGUF/...`,
> lo que confirma la fuente real como **bartowski**, no `Qwen/Qwen2.5-7B-Instruct-GGUF`
> (bartowski es un publicador de cuantizaciones GGUF de terceros, distinto del
> repositorio oficial de Alibaba/Qwen).

Llama-2-7B es un modelo **base**, sin ajuste de instrucciones; Qwen2.5-7B es
**instruct**. Por diseño, sus puntajes de calidad en MT-Bench no son directamente
comparables entre sí (ver [§8, L4](#8-limitaciones-conocidas)).

Los archivos GGUF no están incluidos en el repositorio (`models/` está en
`.gitignore`) y deben obtenerse por separado:

| Archivo | Tamaño |
|---------|-------:|
| `models/llama-2-7b/Q4_K_M.gguf` | 4.08 GB |
| `models/llama-2-7b/llama-2-7b.Q8_0.gguf` | 7.16 GB |
| `models/qwen2.5-7b/Qwen2.5-7B-Instruct-Q4_K_M.gguf` | 4.68 GB |
| `models/qwen2.5-7b/Qwen2.5-7B-Instruct-Q8_0.gguf` | 8.10 GB |

Las plantillas de prompt correctas para cada modelo están implementadas en
[`src/prompt_builder.py`](src/prompt_builder.py): ChatML para Qwen
(`<|im_start|>`/`<|im_end|>`) y texto plano (fallback) para Llama-2-7B base.

---

## 2. Benchmark: subset de MT-Bench

- **Fuente:** dataset oficial de MT-Bench del repositorio
  [FastChat](https://github.com/lm-sys/FastChat) (Apache 2.0), archivo
  `question.jsonl` (80 preguntas, IDs 81–160), citado en Zheng et al. (2023).
- **Subset usado:** 40 preguntas — 5 por categoría × 8 categorías (`coding`,
  `extraction`, `humanities`, `math`, `reasoning`, `roleplay`, `stem`, `writing`),
  seleccionadas de forma determinística (`first_n_per_category`, seed=42).
- **Un turno por pregunta:** aunque MT-Bench oficial es multi-turno, el código de
  medición actual (`scripts/analisis_categoria.py`) usa únicamente `turns[0]` de
  cada pregunta — el segundo turno no se ejecuta en el pipeline vigente.
- Los prompts se usan verbatim, en inglés, sin traducción ni parafraseo.

---

## 3. Protocolo de medición

| Parámetro | Valor |
|-----------|-------|
| Repeticiones de warmup (descartadas) | 3 |
| Repeticiones oficiales por prompt | 15 |
| `temperature` | 0.0 |
| `seed` | 42 |
| `max_tokens` | 1024 |
| `n_ctx` | 4096 |
| Dispositivo | GPU (Metal, Apple M4) |
| Medición de energía | CodeCarbon 3.2.8 (envuelve solo la llamada de inferencia) + `powermetrics` (requiere `sudo`) para el desglose CPU/GPU/RAM |
| Carga del modelo, warmup y cooldown | fuera del tracker de CodeCarbon |

Con `temperature=0` y `seed` fijo, las 15 repeticiones de un mismo prompt son
determinísticas: no miden diversidad de salida, sino estabilidad de
energía/latencia entre repeticiones idénticas.

Cada celda (modelo × cuantización × categoría) produce 5 prompts × 15
repeticiones = 75 mediciones oficiales. Con 4 combinaciones (Llama Q4/Q8, Qwen
Q4/Q8) × 8 categorías, el experimento de energía completo comprende 32
combinaciones modelo×cuantización×categoría (2400 mediciones oficiales).

---

## 4. Estado actual del experimento

**Medición de energía — completa (32/32):** las 4 combinaciones modelo ×
cuantización (Llama Q4, Llama Q8, Qwen Q4, Qwen Q8) están corridas en las 8
categorías. Resultados en `results/analisis_categoria/` (Qwen, corrida vigente),
`results/analisis_categoria_LLAMA_Q4_BACKUP/`, `results/analisis_categoria_LLAMA_Q8_BACKUP/`
y `results/analisis_categoria_QWEN_Q4_BACKUP/` (corridas previas/backup).

Hallazgo principal (Llama-2-7B, ver `docs/comparacion_q4_vs_q8.md`): Q4_K_M
reduce el consumo energético entre 22.5 % y 35.2 % respecto a Q8_0 según
categoría (promedio 27.31 %), de forma consistente en las 8 categorías.

**Evaluación de calidad — prueba de concepto únicamente:** se evaluó con Claude
como juez (comparación pareada, `scripts/juez_calidad.py`, modelo
`claude-opus-4-5`) solo la categoría **math** de **Llama-2-7B** (5 prompts, Q4
vs Q8, 1 repetición por prompt). Resultado en
`results/analisis_categoria_prueba/` y `docs/hallazgo_juez_calidad_math.md`.

Pendiente de escalar:
- Las 7 categorías restantes de Llama-2-7B.
- Las 8 categorías completas de Qwen2.5-7B.
- Evaluación sobre las 15 repeticiones por prompt (la prueba de concepto usó 1).

---

## 5. Estructura del proyecto

```
llm-quantization-study/
├── config.yaml                          # config del experimento (ver nota abajo)
├── requirements.txt                     # dependencias Python
│
├── scripts/
│   ├── analisis_categoria.py            # medición de energía por categoría (script activo)
│   ├── analisis_categoria_prueba.py     # variante usada para la prueba de concepto de calidad
│   ├── analisis_categoria_refactor.py   # versión refactorizada (Pasos 1–3, en validación)
│   ├── inferencia_refactor.py           # módulo: invocación pura al modelo
│   ├── medicion_energia_refactor.py     # módulo: instrumentación CodeCarbon
│   ├── carga_modelo_refactor.py         # módulo: carga del GGUF
│   ├── sistema_refactor.py              # módulo: caffeinate / gestión de sistema
│   ├── graficos_refactor.py             # módulo: funciones de graficado extraídas
│   ├── regenerar_plots.py               # regenera los 3 PNG por categoría desde CSVs existentes
│   ├── regenerar_plots_refactor.py      # misma función, sobre los módulos refactorizados
│   └── juez_calidad.py                  # evaluación de calidad pareada con Claude
│
├── src/
│   ├── config_loader.py                 # carga y valida config.yaml → ExperimentConfig
│   └── prompt_builder.py                # plantillas de prompt por familia de modelo
│
├── data/mt_bench/
│   ├── official/                        # question.jsonl y judge_prompts.jsonl (FastChat, no versionados)
│   └── subset/                          # subset de 40 preguntas ya preparado (jsonl/yaml/csv)
│
├── models/                              # GGUF de los 4 modelos (no versionados, ver §1)
│
├── results/
│   ├── analisis_categoria/              # corrida vigente (Qwen Q4/Q8), 8 categorías c/u
│   ├── analisis_categoria_LLAMA_Q4_BACKUP/
│   ├── analisis_categoria_LLAMA_Q8_BACKUP/
│   ├── analisis_categoria_QWEN_Q4_BACKUP/
│   ├── analisis_categoria_prueba/       # prueba de concepto de calidad (Llama·math)
│   └── metadata/                        # hashes de modelos y dataset, snapshot de config, env report
│
├── entrega_mesa_tesis/                  # paquete entregado a la mesa de tesis (CSVs, docs, gráficos)
│
└── docs/                                # hallazgos, limitaciones y justificaciones documentadas
    ├── comparacion_q4_vs_q8.md
    ├── hallazgo_warmup_termico_llama.md
    ├── hallazgo_anomalia_gpu_qwen_extraction.md
    ├── hallazgo_juez_calidad_math.md
    ├── hallazgo_repeticion_texto.md
    ├── informe_8_categorias.md
    ├── justificaciones_muestra.md
    ├── limitaciones_muestra.md
    ├── diseno_warmup_adaptativo.md
    ├── respaldo_cientifico_warmup.md
    ├── resumen_tecnico_estudiante.md
    └── tesis_refactor.md
```

Cada carpeta de categoría en `results/analisis_categoria*/` contiene
`resultados_completos.csv` + 3 PNG (`boxplot_repeticiones.png`,
`barras_con_std.png`, `desglose_cpu_gpu_ram.png`).

> **Nota sobre `config.yaml`:** el archivo describe un diseño más amplio que el
> que efectivamente se ejecuta (multi-turno, perfil `windows_nvidia`, juez
> multidimensional automático). `scripts/analisis_categoria.py` solo lee de ahí
> `n_ctx`, `max_tokens`, `temperature` y `seed` (vía `src/config_loader.py`); el
> resto de los parámetros reales de cada corrida (categoría, modelo,
> cuantización, número de repeticiones) están hardcodeados como constantes al
> inicio del script y se editan a mano antes de cada ejecución.

---

## 6. Instalación (macOS Apple Silicon)

- Python 3.13.5 (versión usada en el entorno actual, `./venv`).
- Xcode Command Line Tools (`xcode-select --install`), requerido para compilar
  `llama-cpp-python` con soporte Metal.
- Clave de API de Anthropic (`ANTHROPIC_API_KEY`) para `scripts/juez_calidad.py`.
- Los 4 archivos GGUF descargados manualmente (ver §1) en las rutas de
  `config.yaml`.
- El dataset oficial de MT-Bench (`question.jsonl`, `judge_prompts.jsonl`) copiado
  manualmente desde FastChat a `data/mt_bench/official/` (ver
  `data/mt_bench/official/README_SOURCE.md`).

```bash
# Crear y activar entorno virtual
python3.13 -m venv venv
source venv/bin/activate

# Dependencias
pip install -r requirements.txt

# llama-cpp-python con backend Metal
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Clave de API para el juez de calidad
export ANTHROPIC_API_KEY="sk-..."
```

Este proyecto se ejecutó únicamente en Apple Silicon; no se incluyen
instrucciones de instalación para Windows/CUDA porque ese hardware no se usó.

---

## 7. Uso

`scripts/analisis_categoria.py` no toma argumentos de línea de comandos: la
categoría, el modelo y la cuantización a correr se editan como constantes
(`CATEGORIAS`, `MODEL_NAME`, `CUANTIZACION`) al inicio del archivo antes de
ejecutarlo.

```bash
# Medición de energía de una categoría (requiere sudo para powermetrics real)
sudo venv/bin/python scripts/analisis_categoria.py

# Evaluación de calidad pareada Q4 vs Q8 con Claude (requiere ANTHROPIC_API_KEY)
python scripts/juez_calidad.py

# Regenerar los PNG de todas las categorías ya medidas, sin recorrer el modelo
python scripts/regenerar_plots.py
```

---

## 8. Limitaciones conocidas

Documentadas en detalle en `docs/limitaciones_muestra.md`.

- **L1 — Warmup térmico insuficiente en Llama-2-7B:** en 6 de 160 combinaciones
  analizadas, el warmup fijo de 3 repeticiones no bastó para estabilizar
  `gpu_energy_mwh` (caídas de 17–42 % entre la rep. 1 y las reps. 10–15).
  Ningún caso confirmado en Qwen. Mitigación (warmup adaptativo por CV)
  diseñada en `docs/diseno_warmup_adaptativo.md`, no implementada aún.
- **L2 — CodeCarbon es TDP-based, no medición física directa** de CPU/RAM en
  Apple Silicon (potencia fija: 20 W CPU, 3 W RAM). El componente GPU sí
  proviene de `powermetrics`, que lee el SoC directamente. Las comparaciones
  relativas Q4 vs Q8 dentro del mismo hardware siguen siendo válidas.
- **L3 — Anomalía de GPU en Qwen·extraction:** es la única categoría (de 8) en
  la que Q4 no reduce el consumo GPU respecto a Q8; concentrada en 2 de 5
  prompts. Causa mecanicista no confirmada (ver
  `docs/hallazgo_anomalia_gpu_qwen_extraction.md`).
- **L4 — Llama (base) vs Qwen (instruct) no son comparables directamente en
  calidad** sobre MT-Bench, que fue diseñado para modelos instruct.
- **L5 — Evaluación de calidad incompleta:** solo prueba de concepto en
  Llama·math (ver §4); no generalizable a las demás categorías ni a Qwen.
- **L6 — Un solo dispositivo, sin replicación en otro hardware:** todos los
  resultados provienen de un único Mac Mini M4; no hay corrida en Windows/CUDA
  para contrastar.

---

## 9. Referencias

Fuente: `docs/justificaciones_muestra.md` (BibTeX completo y justificación en
prosa de cada elección metodológica).

- Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). *Judging LLM-as-a-Judge
  with MT-Bench and Chatbot Arena.* NeurIPS 2023. arXiv:2306.05685.
- Touvron, H., et al. (2023). *Llama 2: Open Foundation and Fine-Tuned Chat
  Models.* Meta AI. (justificación de la elección de Llama-2-7B vía Wilkins et
  al. 2024, arXiv:2407.00010).
- Qwen Team, Alibaba Cloud. *Qwen2.5 Technical Report* (justificación de la
  elección de Qwen2.5-7B-Instruct vía Hornet et al. 2025, TU Delft
  Sustainable Software Engineering, Grupo 23, supervisado por Luis Cruz).
- Lacoste, A., Luccioni, A., Schmidt, V., Dandres, T. (2019). *Quantifying the
  Carbon Emissions of Machine Learning.* arXiv:1910.09700. (NeurIPS 2019
  Workshop on Tackling Climate Change with ML — fundamenta CodeCarbon).
- Husom, E., Goknil, A., Astekin, M., et al. (2025). *Sustainable LLM
  Inference for Edge AI: Evaluating Quantized LLMs for Energy Efficiency,
  Output Accuracy, and Inference Latency.* ACM Transactions on Internet of
  Things. DOI: 10.1145/3767742. (justifica el rango Q4–Q8: ~52–54 % de ahorro
  FP16→Q8_0).
- Licardo, J. T., Tankovic, N. (2025). *Performance Trade-offs of Optimizing
  Small Language Models for E-Commerce.* arXiv:2510.21970. (concepto de
  "quantization cliff" por debajo de Q4).
- Xue, Z., Song, Y., Mi, Z., et al. (2024). *PowerInfer-2: Fast Large Language
  Model Inference on a Smartphone.* arXiv:2406.06282. (justifica llama.cpp
  como backend de referencia).
- Taneja, H., Kim, J., Xu, J. J., et al. (2023). *Hot Pixels: Frequency,
  Power, and Temperature Attacks on GPUs and ARM SoCs.* arXiv:2305.12784.
  (respaldo del mecanismo de throttling térmico en Apple Silicon, relevante
  para L1).

Carbon intensity factor: Paraguay, 26 gCO₂eq/kWh (EMBER 2024, matriz
predominantemente hidroeléctrica) — consistente entre `CLAUDE.md` y
`docs/informe_8_categorias.md`.

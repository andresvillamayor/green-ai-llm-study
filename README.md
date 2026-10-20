# GREEN-IA: Consumo Energético en Inferencia de LLMs Cuantizados

**Tesis de Maestría en Ciencia de Datos — Universidad Comunero, Paraguay**
**Autores:** Andrés Villamayor y Giampaolo Ficorilli

Estudio experimental del consumo energético de la inferencia local de modelos de lenguaje (LLMs) bajo dos niveles de cuantización (Q4_K_M vs Q8_0), ejecutado en un Mac Mini con Apple Silicon (M4, 16 GB). La energía se mide con CodeCarbon y powermetrics, sobre un subset de MT-Bench. La evaluación de calidad de las respuestas (LLM-as-a-Judge, §4) está en fase de prueba de concepto.

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

| Modelo | Tipo | Cuantizaciones | Fuente (Hugging Face) | Rol en el estudio |
|---|---|---|---|---|
| Llama-2-7B-Chat | chat (Meta) | Q4_K_M, Q8_0 | `TheBloke/Llama-2-7B-Chat-GGUF` | Campaña principal |
| Qwen2.5-7B-Instruct | instruct (Alibaba) | Q4_K_M, Q8_0 | `bartowski/Qwen2.5-7B-Instruct-GGUF` | Campaña principal |
| Llama-2-7B | base (Meta) | Q4_K_M, Q8_0 | `TheBloke/Llama-2-7B-GGUF` | Solo una prueba de referencia (coding, Q4); no forma parte de la campaña |

> **Nota sobre Qwen:** `models/qwen2.5-7b/*.gguf` son enlaces simbólicos al caché local de Hugging Face, que resuelven a `models--bartowski--Qwen2.5-7B-Instruct-GGUF`. Es decir, la fuente real es bartowski (publicador de cuantizaciones GGUF de terceros), no el repositorio oficial `Qwen/Qwen2.5-7B-Instruct-GGUF`.

La campaña compara los **dos modelos de chat** (Llama-2-7B-Chat y Qwen2.5-7B-Instruct), cada uno en Q4 y Q8. Ambos están ajustados con instrucciones, pero difieren en entrenamiento, arquitectura y formato de prompt, por lo que la comparación entre ellos es descriptiva.

Los archivos GGUF no están incluidos en el repositorio (`models/` está en `.gitignore`) y deben obtenerse por separado:

| Archivo | Tamaño |
|---|---|
| `models/llama-2-7b-chat/llama-2-7b-chat.Q4_K_M.gguf` | 4.08 GB |
| `models/llama-2-7b-chat/llama-2-7b-chat.Q8_0.gguf` | 7.16 GB |
| `models/qwen2.5-7b/Qwen2.5-7B-Instruct-Q4_K_M.gguf` | 4.68 GB |
| `models/qwen2.5-7b/Qwen2.5-7B-Instruct-Q8_0.gguf` | 8.10 GB |
| `models/llama-2-7b/Q4_K_M.gguf` (prueba de referencia) | 4.08 GB |
| `models/llama-2-7b/llama-2-7b.Q8_0.gguf` (prueba de referencia) | 7.16 GB |

Cada modelo usa su formato de prompt nativo, implementado en `src/prompt_builder.py`:

- **Llama-2-7B-Chat:** `<s>[INST] pregunta [/INST]`.
- **Qwen2.5-7B-Instruct:** ChatML (`<|im_start|>` / `<|im_end|>`), con el mensaje de sistema por defecto ("You are a helpful assistant.").
- **Llama-2-7B (base):** texto plano, sin plantilla.

Por esta diferencia, la cantidad de tokens de entrada no es idéntica entre modelos para un mismo prompt.

## 2. Benchmark: subset de MT-Bench

- **Fuente:** dataset oficial de MT-Bench del repositorio FastChat (Apache 2.0), archivo `question.jsonl` (80 preguntas, IDs 81–160), citado en Zheng et al. (2023).
- **Subset usado:** 40 preguntas — 5 por categoría × 8 categorías (coding, extraction, humanities, math, reasoning, roleplay, stem, writing), seleccionadas de forma determinística (`first_n_per_category`, seed=42).
- **Un turno por pregunta:** aunque MT-Bench oficial es multi-turno, el pipeline usa únicamente `turns[0]` de cada pregunta.
- Los prompts se usan *verbatim*, en inglés, sin traducción ni parafraseo.

## 3. Protocolo de medición

| Parámetro | Valor |
|---|---|
| Repeticiones de warmup (se registran, marcadas `es_warmup`) | 3 |
| Repeticiones oficiales por prompt | 15 |
| `temperature` | 0.0 |
| `seed` | 42 |
| `max_tokens` | 1024 |
| `n_ctx` | 4096 |
| Dispositivo | GPU (Metal, Apple M4) |
| Medición de energía | CodeCarbon 3.2.8 (envuelve solo la llamada de inferencia) + powermetrics (requiere `sudo`) para el desglose CPU/GPU/RAM |
| Carga del modelo, warmup y cooldown | fuera del tracker de CodeCarbon |

Con `temperature=0` y semilla fija, las 15 repeticiones de un mismo prompt son deterministas: no miden diversidad de salida, sino estabilidad de energía y latencia entre repeticiones idénticas.

**Tamaño del experimento.** Cada celda (modelo × cuantización × categoría) tiene 5 prompts × 15 repeticiones = 75 mediciones oficiales, más 3 repeticiones de warmup por prompt: **78 filas por CSV**. Con 4 combinaciones (Llama-Chat Q4/Q8 y Qwen Q4/Q8) × 8 categorías hay **32 corridas** y 2400 mediciones oficiales.

**Procedimiento por corrida** (limitación de RAM de 16 GB): se corre **una categoría por vez**, con apagado y enfriado del equipo, `sudo purge` y ejecución con `sudo` (necesario para powermetrics). Como el script corre como root, los archivos de salida quedan a nombre de root; al terminar se ejecuta `sudo chown -R <usuario>:staff results`.

**Esquema del CSV** (`resultados_completos.csv`, 22 columnas):

| Grupo | Columnas |
|---|---|
| Identificación | `modelo`, `categoria`, `prompt_id`, `prompt_num`, `repeticion`, `prompt_texto_corto`, `es_warmup`, `temperatura`, `tipo_modelo` |
| Energía | `measured_energy_kwh`, `measured_energy_mwh`, `cpu_energy_mwh`, `gpu_energy_mwh`, `ram_energy_mwh` (CPU + GPU + RAM = total) |
| Tiempo y tokens | `inference_time_s`, `prompt_tokens_entrada`, `completion_tokens` |
| Texto | `prompt_enviado` (con la plantilla del modelo, sin truncar), `respuesta_texto` (todas las repeticiones) |
| Degeneración | `seq_rep_4` |
| Calidad (vacías hasta correr el juez) | `juez_score`, `juez_costo_usd` |

**`seq_rep_4`** es la proporción de 4-gramas repetidos en los tokens generados (1 − 4-gramas únicos / 4-gramas totales), como medida continua de degeneración por repetición (Welleck et al., 2019). No se define ningún umbral binario. Es sensible a la longitud de la respuesta, por lo que no es directamente comparable con los valores del artículo original.

## 4. Estado actual del experimento

**Medición de energía — completa (32/32).** Las cuatro combinaciones (Llama-2-7B-Chat Q4 y Q8, Qwen2.5-7B-Instruct Q4 y Q8) están corridas en las 8 categorías, con 78 filas y 22 columnas por CSV. Resultados en:

```
results/comparacion_temperaturas/
├── llama-2-7b-chat_determinista/{llama-2-7b-chat_q4, llama-2-7b-chat_q8}/<categoría>/
├── qwen2.5-7b_determinista/{qwen2.5-7b_q4, qwen2.5-7b_q8}/<categoría>/
└── llama-2-7b_determinista/llama-2-7b_q4/coding/      # prueba de referencia (modelo base)
```

Cada carpeta de categoría contiene `resultados_completos.csv` y 3 PNG; cada combinación tiene además `comparacion_categorias.png`.

**Resultados descriptivos preliminares** (promedios de las 8 categorías, calculados desde los CSV; **sin pruebas estadísticas ni intervalos de confianza todavía**). Reducción = (Q8 − Q4) / Q8:

| | Llama-2-7B-Chat | Qwen2.5-7B-Instruct |
|---|---|---|
| Energía media por repetición, Q4 / Q8 (mWh) | 87.7 / 115.5 | 64.5 / 81.0 |
| Reducción de energía con Q4 | 24 % | 20.5 % |
| Tiempo medio por repetición, Q4 / Q8 (s) | 25.2 / 39.4 | 19.3 / 31.3 |
| Reducción de tiempo con Q4 | 36 % | 38 % |
| Energía por token, Q4 / Q8 (mWh) | 0.206 / 0.287 | 0.170 / 0.213 |

- La energía por token es menor con Q4 en las 16 combinaciones de modelo y categoría.
- En Qwen, Q4 gasta menos energía total que Q8 en las 8 categorías (reducciones entre 8.8 % y 33.4 %).
- En Llama-Chat, la reducción por categoría va de 8.8 % a 52.3 %, y en `extraction` Q4 gasta *más* energía total que Q8 (+59 %), porque generó más tokens (377 contra 141 de promedio). Con temperatura 0, la cuantización cambia el texto generado y eso afecta la energía total; por eso se analiza también la energía por token.
- Los resultados de la campaña anterior (Llama-2-7B base) están en el paquete `entrega_mesa_tesis/` y en el historial de git.

**Evaluación de calidad — solo prueba de concepto.** Se evaluó con un LLM como juez (comparación pareada, `scripts/juez_calidad.py`, modelo `claude-opus-4-5`) únicamente la categoría `math` de Llama-2-7B base (5 prompts, Q4 vs Q8, 1 repetición). Ver `docs/hallazgo_juez_calidad_math.md`. El juez es un modelo distinto de los evaluados (Llama y Qwen); los modelos evaluados nunca juegan de juez.

**Pendiente:**

- Análisis estadístico de la campaña (intervalos de confianza y pruebas, con respaldo bibliográfico).
- Evaluación de calidad de las 4 combinaciones (las columnas `juez_score` y `juez_costo_usd` están vacías).
- Revisión del texto de las respuestas que llegaron al tope de 1024 tokens o tienen `seq_rep_4` alto.
- Esquema de generación "fabricante" (temperatura recomendada por cada fabricante), todavía sin implementar para Qwen (requiere `top_k` y `repetition_penalty`).

## 5. Estructura del proyecto

```
llm-quantization-study/
├── config.yaml                          # config de modelos y parámetros (ver nota)
├── config_temperaturas.yaml             # esquemas de temperatura (solo documentación, el código no lo lee)
├── requirements.txt                     # dependencias Python
│
├── scripts/
│   ├── analisis_categoria_refactor.py   # medición de energía por categoría (script activo)
│   ├── inferencia_refactor.py           # módulo: invocación al modelo
│   ├── medicion_energia_refactor.py     # módulo: instrumentación CodeCarbon + seq_rep_4
│   ├── carga_modelo_refactor.py         # módulo: carga del GGUF
│   ├── sistema_refactor.py              # módulo: caffeinate / gestión de sistema
│   ├── graficos_refactor.py             # módulo: gráficos
│   ├── regenerar_plots_refactor.py      # regenera los PNG desde CSVs existentes
│   ├── juez_calidad.py                  # evaluación de calidad pareada (LLM-as-a-Judge)
│   ├── analisis_categoria.py            # versión original (monolítica), conservada como referencia
│   ├── analisis_categoria_prueba.py     # script de pruebas de 1 repetición
│   └── regenerar_plots.py               # versión original de regenerar_plots
│
├── src/
│   ├── config_loader.py                 # carga y valida config.yaml
│   └── prompt_builder.py                # plantillas de prompt por modelo
│
├── data/mt_bench/
│   ├── official/                        # question.jsonl y judge_prompts.jsonl (FastChat, no versionados)
│   └── subset/                          # subset de 40 preguntas
│
├── models/                              # GGUF (no versionados, ver §1)
├── results/
│   ├── comparacion_temperaturas/        # resultados de la campaña (ver §4)
│   └── metadata/                        # hashes de modelos y dataset, snapshot de config
├── entrega_mesa_tesis/                  # paquete entregado a la mesa de tesis (campaña anterior)
└── docs/                                # hallazgos, limitaciones y justificaciones
```

Documentos en `docs/`: `comparacion_q4_vs_q8.md`, `informe_8_categorias.md`, `hallazgo_warmup_termico_llama.md`, `hallazgo_anomalia_gpu_qwen_extraction.md`, `hallazgo_juez_calidad_math.md`, `hallazgo_repeticion_texto.md`, `justificaciones_muestra.md`, `limitaciones_muestra.md`, `diseno_warmup_adaptativo.md`, `respaldo_cientifico_warmup.md`, `respaldo_cientifico_temperatura_cero.md`, `listado_prompts_categorias.md`, `resumen_tecnico_estudiante.md`, `tesis_refactor.md`. Los documentos `comparacion_q4_vs_q8.md`, `informe_8_categorias.md` y `hallazgo_warmup_termico_llama.md` corresponden a la campaña anterior (Llama-2-7B base).

> **Nota sobre `config.yaml`:** `analisis_categoria_refactor.py` solo lee de ahí `n_ctx`, `max_tokens`, `temperature`, `seed` y las rutas de los modelos. La categoría, el modelo, la cuantización, el directorio de salida y el número de repeticiones se editan como constantes al inicio del script antes de cada corrida. El campo `model_type` de `config.yaml` (`instruct` para Qwen) no se usa para la columna `tipo_modelo` del CSV, que vale `chat` para los dos modelos de la campaña.

## 6. Instalación (macOS Apple Silicon)

- Python 3.13.5 (versión usada en el entorno, `./venv`).
- Xcode Command Line Tools (`xcode-select --install`), requeridas para compilar `llama-cpp-python` con soporte Metal.
- Los archivos GGUF descargados manualmente (ver §1) en las rutas de `config.yaml`.
- El dataset oficial de MT-Bench (`question.jsonl`, `judge_prompts.jsonl`) copiado desde FastChat a `data/mt_bench/official/` (ver `data/mt_bench/official/README_SOURCE.md`).
- Solo para `scripts/juez_calidad.py`: una clave de API de Anthropic en la variable `ANTHROPIC_API_KEY`.

```bash
# Crear y activar entorno virtual
python3.13 -m venv venv
source venv/bin/activate

# Dependencias
pip install -r requirements.txt

# llama-cpp-python con backend Metal
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

Este proyecto se ejecutó únicamente en Apple Silicon.

## 7. Uso

`analisis_categoria_refactor.py` no toma argumentos de línea de comandos. Antes de cada corrida se editan las constantes del inicio del archivo (`CATEGORIAS`, `MODEL_NAME`, `CUANTIZACION`, `OUTPUT_BASE_DIR`).

```bash
# 1) Medición de energía de una categoría (con el equipo recién encendido)
sudo purge
source venv/bin/activate
sudo python3 scripts/analisis_categoria_refactor.py

# 2) Devolver la propiedad de los resultados al usuario (la corrida los crea como root)
sudo chown -R <usuario>:staff results

# Evaluación de calidad pareada Q4 vs Q8 (requiere ANTHROPIC_API_KEY)
python scripts/juez_calidad.py

# Regenerar los PNG de las categorías ya medidas, sin recorrer el modelo
python scripts/regenerar_plots_refactor.py
```

> `juez_calidad.py` y `regenerar_plots_refactor.py` todavía apuntan a rutas anteriores y hay que ajustarlas para usarlos con `results/comparacion_temperaturas/`.

## 8. Limitaciones conocidas

Documentadas en detalle en `docs/limitaciones_muestra.md` (parte de ese documento corresponde a la campaña anterior).

- **L1 — Warmup térmico en Llama-2-7B base** (campaña anterior): en 6 de 160 combinaciones el warmup fijo de 3 repeticiones no bastó para estabilizar `gpu_energy_mwh`. No se analizó todavía para la campaña actual. Mitigación (warmup adaptativo por CV) diseñada en `docs/diseno_warmup_adaptativo.md`, no implementada.
- **L2 — CodeCarbon estima CPU y RAM por TDP**, no por medición física directa en Apple Silicon (potencia fija: 20 W CPU, 3 W RAM). El componente GPU proviene de powermetrics. Las comparaciones relativas Q4 vs Q8 en el mismo hardware siguen siendo válidas.
- **L3 — Anomalía de GPU en Qwen·extraction** (campaña anterior): causa mecanicista no confirmada (`docs/hallazgo_anomalia_gpu_qwen_extraction.md`).
- **L4 — Llama-Chat y Qwen no son idénticos en condiciones:** ambos son modelos de chat, pero difieren en entrenamiento, arquitectura y formato de prompt (`[INST]` vs ChatML con mensaje de sistema). La comparación entre modelos es descriptiva.
- **L5 — Evaluación de calidad incompleta:** solo prueba de concepto (ver §4). No se sabe todavía si el ahorro de energía de Q4 tiene costo en calidad.
- **L6 — Un solo dispositivo** (Mac Mini M4): los resultados no están replicados en otro hardware.
- **L7 — Dos mediciones sin energía de GPU** en Llama-2-7B-Chat Q4 (una de warmup en humanities y una oficial en reasoning): el valor de GPU y el total quedaron vacíos; se excluyen de los promedios.
- **L8 — Tope de `max_tokens=1024`:** varias respuestas lo alcanzaron (Llama-Chat Q4: extraction, humanities, stem; Llama-Chat Q8: humanities, roleplay, writing; Qwen Q8: humanities). No se revisó aún si es degeneración o respuesta larga legítima.
- **L9 — `seq_rep_4` sin umbral validado** y sensible a la longitud de la respuesta.
- **L10 — Resultados descriptivos:** sin pruebas estadísticas ni intervalos de confianza todavía.

## 9. Referencias

Fuente: `docs/justificaciones_muestra.md` (BibTeX y justificación de cada elección metodológica). Las referencias de `docs/justificaciones_muestra.md` deben contrastarse contra la fuente original antes de la entrega.

- Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS 2023. arXiv:2306.05685.
- Touvron, H., et al. (2023). *Llama 2: Open Foundation and Fine-Tuned Chat Models*. Meta AI.
- Qwen Team, Yang, A., et al. (2024). *Qwen2.5 Technical Report*. arXiv:2412.15115.
- Welleck, S., Kulikov, I., Roller, S., Dinan, E., Cho, K., Weston, J. (2019). *Neural Text Generation with Unlikelihood Training*. arXiv:1908.04319 (base de `seq_rep_4`).
- Lacoste, A., Luccioni, A., Schmidt, V., Dandres, T. (2019). *Quantifying the Carbon Emissions of Machine Learning*. arXiv:1910.09700 (fundamenta CodeCarbon).
- Argerich, M. F., Patiño-Martínez, M. (2024). *Measuring and Improving the Energy Efficiency of Large Language Models Inference*. IEEE Access, vol. 12 (usa Llama-2-7B-Chat y versiones GGUF de TheBloke).
- Poddar, S., Koley, P., Misra, J., Ganguly, N., Ghosh, S. (2025). *Towards Sustainable NLP: Insights from Benchmarking Inference Energy in Large Language Models*. NAACL 2025.
- Caravaca, F., Cuevas, Á., Cuevas, R. (2025). *From Prompts to Power: Measuring the Energy Footprint of LLM Inference*. arXiv:2511.05597 (preprint).
- Basta, P. (2026). *Towards Green AI: Evaluating Energy Efficiency for Quantized LLM Inference*. arXiv:2610.17531 (preprint; el texto lo presenta como tesis de maestría de la University of Waterloo).
- Laskowski, J., Tomiło, P., Kozłowski, E., Laskowska, A., Piątek, R. (2026). *Does Lower Precision Always Save Energy? Benchmarking Energy–Performance Trade-Offs in Weight-Quantized Large Language Model Inference on a GPU*. Energies 19(20), 4893.
- Husom, E., Goknil, A., Astekin, M., et al. (2025). *Sustainable LLM Inference for Edge AI: Evaluating Quantized LLMs for Energy Efficiency, Output Accuracy, and Inference Latency*. ACM Transactions on Internet of Things. DOI: 10.1145/3767742.
- Licardo, J. T., Tankovic, N. (2025). *Performance Trade-offs of Optimizing Small Language Models for E-Commerce*. arXiv:2510.21970.
- Xue, Z., Song, Y., Mi, Z., et al. (2024). *PowerInfer-2: Fast Large Language Model Inference on a Smartphone*. arXiv:2406.06282.
- Taneja, H., Kim, J., Xu, J. J., et al. (2023). *Hot Pixels: Frequency, Power, and Temperature Attacks on GPUs and ARM SoCs*. arXiv:2305.12784.
- Factor de intensidad de carbono de Paraguay: 26 gCO₂eq/kWh (EMBER 2024, matriz predominantemente hidroeléctrica).

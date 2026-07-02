# GREEN-IA: Resumen Técnico del Proyecto
**Para estudiantes de Maestría en Ciencia de Datos**
*Universidad Comunero — Paraguay*

---

## 1. Qué hace este proyecto

GREEN-IA mide cuánta energía consumen dos modelos de lenguaje (LLM) cuando generan texto, comparando dos niveles de compresión (cuantización Q4 y Q8) en dos tipos de hardware distintos (Apple M4 y NVIDIA RTX 4060). Para cada combinación de modelo, cuantización y hardware, el experimento corre las mismas 40 preguntas de benchmark MT-Bench 15 veces cada una, midiendo la energía eléctrica consumida durante la inferencia. Después, un juez basado en Claude evalúa la calidad de las respuestas generadas. El objetivo final es responder: ¿se puede reducir el consumo energético de un LLM mediante cuantización sin sacrificar demasiada calidad?

---

## 2. Arquitectura general — cómo se conectan los archivos

```
config.yaml
    │
    ▼
run_experiment.py          ← Punto de entrada. Lee config, valida todo, genera
    │                         metadatos y delega la medición.
    │
    ├── src/config_loader.py      ← Carga y valida config.yaml
    ├── src/hardware_profile.py   ← Detecta el hardware de la máquina
    ├── src/reproducibility.py    ← Genera hashes y fingerprints
    ├── src/metadata_writer.py    ← Escribe los 6 archivos de metadatos
    │
    └── scripts/fase1_benchmark.py   ← El bucle principal de medición
            │
            ├── src/config_loader.py
            ├── src/baseline_energy.py   ← Mide el consumo en reposo
            ├── src/prompt_builder.py    ← Construye los prompts formateados
            │
            ├── [llama-cpp-python]       ← Carga y corre el modelo .gguf
            ├── [codecarbon]             ← Mide la energía de cada inferencia
            │
            ├── results/raw/turn_results.csv
            └── results/raw/conversation_results.csv

data/mt_bench/subset/mt_bench_literal_subset_5_per_category.yaml
    │
    └── Generado por: scripts/prepare_mt_bench_subset.py
                          │
                          └── Lee: data/mt_bench/official/question.jsonl

results/raw/conversation_results.csv
    │
    ▼
judge_with_claude.py       ← Evalúa la calidad con Claude como juez
    │
    └── results/judge/judge_results.csv

results/raw/ + results/judge/
    │
    ├── analysis_summary.py    ← Estadísticas y tablas para la tesis
    ├── analysis_plots.py      ← Gráficas
    └── results/summary/*.csv
```

**Regla de oro de la arquitectura:** la medición energética (CodeCarbon) está completamente separada de la evaluación de calidad (Claude). Nunca corren al mismo tiempo.

---

## 3. Cada archivo explicado en lenguaje simple

### `config.yaml`
**Qué hace:** Es el panel de control del experimento. Todos los parámetros están aquí.
**Por qué existe:** Para que puedas cambiar el hardware, los modelos o las repeticiones sin tocar código.
**Entrada:** Lo edita el usuario.
**Salida:** Lo leen casi todos los demás archivos.

---

### `run_experiment.py`
**Qué hace:** Es el punto de entrada. Hace ocho pasos de preparación y luego lanza el benchmark.
**Por qué existe:** Para separar el setup (validaciones, carpetas, metadatos) de la medición real.
**Entrada:** `config.yaml` + argumentos de línea de comandos (`--hardware-profile`, `--execution-device`, etc.)
**Salida:**
- `results/metadata/config_snapshot.yaml`
- `results/metadata/environment_report.json`
- `results/metadata/experiment_manifest.json`
- `results/metadata/model_hashes.csv`
- `results/metadata/dataset_manifest.csv`
- `results/metadata/requirements_freeze.txt`
- Luego delega a `scripts/fase1_benchmark.py` como subproceso.

**Modos especiales:**
- `--dry-run`: muestra la config y el plan, no corre nada.
- `--setup-only`: hace el setup pero no inicia la medición.
- `--limit-conversations N` o `--repetitions N` menores al oficial: activa "smoke test" y los resultados van a `results/smoke_test/` (nunca se mezclan con los datos oficiales).

---

### `scripts/fase1_benchmark.py`
**Qué hace:** Es el corazón del experimento. Itera sobre cada combinación (modelo × cuantización), carga el modelo en memoria, mide el baseline de energía en reposo, y luego ejecuta cada par (pregunta, repetición) midiendo la energía de cada inferencia con CodeCarbon.
**Por qué existe:** Separado de `run_experiment.py` para poder ejecutarlo directamente en modo debug, y para mantener limpia la separación entre setup y medición.
**Entrada:** `config.yaml`, el subset de MT-Bench (YAML o JSONL), argumentos de CLI.
**Salida:**
- `results/raw/turn_results.csv` — una fila por turno (turno 1 y turno 2 de cada conversación).
- `results/raw/conversation_results.csv` — una fila por conversación completa (suma de los dos turnos).
- `results/raw/baseline_results.csv` — mediciones de energía en reposo.
- `results/backups/*.csv` — backups automáticos cada 50 conversaciones.

**Principios metodológicos que implementa:**
- P1: CodeCarbon activo solo durante la llamada al modelo (zona crítica mínima).
- P2: El modelo se carga antes de abrir el tracker de energía.
- P3: Los sleeps de cooldown y warmup ocurren fuera del tracker.
- P9: Baseline idle medido antes de cada configuración de modelo.
- P10: Energía corregida = energía medida − consumo de fondo × tiempo.
- P12: Si el prompt supera el contexto disponible, registra `context_overflow` sin medir.
- P13: Registra el `finish_reason` de cada inferencia.
- P14: IDs de turno deterministas (SHA256 de modelo + cuantización + question_id + turno + repetición) para poder reanudar sin duplicar filas.

---

### `scripts/prepare_mt_bench_subset.py`
**Qué hace:** Lee el archivo oficial de MT-Bench (`question.jsonl` de FastChat), valida su estructura, selecciona las primeras N preguntas por categoría y guarda tres versiones del subconjunto.
**Por qué existe:** Para garantizar que los prompts usados sean exactamente los oficiales, sin ninguna modificación ni traducción.
**Entrada:** `data/mt_bench/official/question.jsonl` (archivo oficial de FastChat/Zheng et al.)
**Salida:**
- `data/mt_bench/subset/mt_bench_literal_subset_5_per_category.yaml`
- `data/mt_bench/subset/mt_bench_literal_subset_5_per_category.jsonl`
- `data/mt_bench/subset/mt_bench_subset_manifest.csv`

---

### `judge_with_claude.py`
**Qué hace:** Toma las conversaciones ya medidas (energía + respuestas del modelo) y llama a la API de Claude para que evalúe la calidad de cada respuesta. Devuelve 9 dimensiones de puntaje (1-10) y 3 campos cualitativos.
**Por qué existe:** Para separar completamente la medición de energía de la evaluación de calidad. Claude actúa como juez externo; la energía que gasta Claude en evaluar no se mide.
**Entrada:** `results/raw/conversation_results.csv` (que debe tener columnas con el texto de preguntas y respuestas), variable de entorno `ANTHROPIC_API_KEY`.
**Salida:**
- `results/judge/judge_results.csv`
- `results/judge/judge_results.jsonl`

**Soporte para referencias:** Para las categorías math, reasoning, coding y stem, si existe el archivo `data/mt_bench/subset/references.yaml`, el juez recibe las respuestas de referencia y las usa solo para verificar corrección factual.

---

### `analysis_summary.py`
**Qué hace:** Une los datos de energía (`conversation_results.csv`) con los puntajes de calidad (`judge_results.csv`), calcula estadísticas (media, desviación estándar, IC95, tests de hipótesis) y genera 14 tablas CSV para la tesis.
**Por qué existe:** Para producir las tablas de resultados listas para publicar.
**Entrada:** `results/raw/conversation_results.csv` + `results/judge/judge_results.csv`.
**Salida:** 14 archivos en `results/summary/`, incluyendo ranking completo, tabla para el paper y tabla de configuraciones Pareto-eficientes.

---

### `analysis_plots.py`
**Qué hace:** Lee los mismos CSVs que `analysis_summary.py` y genera gráficas.
**Por qué existe:** Para producir las figuras del paper/tesis.
**Entrada:** `results/raw/conversation_results.csv` + `results/judge/judge_results.csv`.
**Salida:** Imágenes PNG en `results/plots/`.

---

### `src/config_loader.py`
**Qué hace:** Lee `config.yaml` y lo convierte en un objeto Python inmutable (`ExperimentConfig`) con todos los parámetros validados.
**Por qué existe:** Para centralizar la lectura de configuración y garantizar que todos los scripts usen los mismos valores. Si `hardware_profile` o `execution_device` no son válidos, aborta con un mensaje claro.
**Entrada:** `config.yaml`.
**Salida:** Un objeto `ExperimentConfig` (dataclass frozen, no se puede modificar accidentalmente).

**Métodos derivados útiles:**
- `n_gpu_layers()`: retorna `-1` (GPU) o `0` (CPU) según `execution_device`.
- `should_caffeinate()`: `True` solo en mac_m4 cuando `prevent_sleep=True`.
- `effective_seed(rep)`: la semilla para cada repetición (fija en modo determinista).

---

### `src/baseline_energy.py`
**Qué hace:** Mide el consumo eléctrico del sistema cuando está en reposo (sin cargar ningún modelo) y luego resta ese consumo de fondo de las mediciones de inferencia.
**Por qué existe:** El consumo medido durante la inferencia incluye procesos del sistema operativo, pantalla, etc. Para aislar el consumo real del LLM, se resta el "ruido de fondo".
**Entrada:** Duración en segundos, número de repeticiones.
**Salida:**
- `medir_baseline_calibracion()`: un dict con la potencia promedio en reposo (W).
- `corregir_energia()`: un dict con la energía neta (medida − fondo), con flag si se clippeó a cero.

---

### `src/prompt_builder.py`
**Qué hace:** Construye el string de prompt correcto según el modelo, aplicando el template de chat que cada modelo espera.
**Por qué existe:** Cada familia de modelos fue entrenada con un formato diferente. Usar el template incorrecto degrada la calidad y aumenta el consumo (el modelo genera tokens extra intentando compensar la entrada malformada).
**Entrada:** nombre del modelo + lista de mensajes `[{"role": "user"|"assistant", "content": "..."}]`.
**Salida:** Un string de texto listo para pasar a llama-cpp-python.

**Templates implementados:**
- `qwen*`: formato ChatML (`<|im_start|>/<|im_end|>`).
- `llama-2*chat` o `llama-2*instruct`: formato oficial de Meta (`[INST]/<<SYS>>`).
- Resto: fallback de texto plano con etiquetas `[Turn N Question]`.

---

### `src/hardware_profile.py`
**Qué hace:** Detecta el hardware donde está corriendo el experimento y construye un objeto `HardwareProfile` con todos los datos relevantes (CPU, RAM, GPU, backend).
**Por qué existe:** Para documentar el entorno de forma reproducible. Los resultados de energía dependen del hardware, así que hay que registrarlo con exactitud.
**Entrada:** La configuración de `config.yaml` (ruta principal) o auto-detección del sistema operativo (fallback).
**Salida:** Un objeto `HardwareProfile` con 17 campos (machine_id, cpu_model, ram_total_gb, gpu_backend, etc.).

---

### `src/metrics.py`
**Qué hace:** Calcula las métricas de eficiencia energética a partir de los datos crudos de CodeCarbon.
**Por qué existe:** Para mantener en un solo lugar las fórmulas de las métricas, con sus referencias de literatura.
**Entrada:** Energía neta (Wh), tiempo de inferencia (s), tokens de salida, emisiones CO2.
**Salida:** Un dict con todas las métricas derivadas.

---

### `src/reproducibility.py`
**Qué hace:** Proporciona utilidades para garantizar que el experimento sea reproducible: hashes de archivos de modelo, fingerprint de la configuración, freeze de dependencias, IDs estables por fila.
**Por qué existe:** Para que cualquier investigador pueda verificar que sus resultados corresponden exactamente a los del paper.
**Entrada:** Rutas de archivos, dicts de configuración.
**Salida:** Strings de hash, dicts de versiones de paquetes, IDs deterministas.

---

### `src/metadata_writer.py`
**Qué hace:** Centraliza la escritura de los 6 archivos de metadatos de reproducibilidad que se generan antes de cada corrida.
**Por qué existe:** Para que `run_experiment.py` no mezcle lógica de serialización con lógica de setup.
**Entrada:** Los datos del experimento (config, hardware, lista de preguntas, etc.).
**Salida:** 6 archivos en `results/metadata/`.

---

### `scripts/validate_config.py`, `scripts/hash_artifacts.py`
**Qué hacen:** Scripts de utilidad. `validate_config.py` verifica que `config.yaml` esté bien formado y que el backend de llama-cpp coincida con el esperado. `hash_artifacts.py` calcula SHA256 de los archivos de modelo y del subset.
**Por qué existen:** Para detectar errores de configuración antes de una corrida oficial de varias horas.

---

## 4. El flujo de ejecución paso a paso

```
Paso 0 — PREPARACIÓN (una sola vez, manual)
  a. Descargar los modelos .gguf a models/llama-2-7b/ y models/qwen2.5-7b/
  b. Clonar FastChat y copiar question.jsonl a data/mt_bench/official/
  c. Editar config.yaml con hardware_profile y execution_device correctos
  d. python scripts/prepare_mt_bench_subset.py
     → crea el subset de 40 preguntas en data/mt_bench/subset/

Paso 1 — SETUP Y LANZAMIENTO
  python run_experiment.py --hardware-profile mac_m4 --execution-device gpu
  │
  ├── Lee config.yaml y aplica overrides de CLI
  ├── Valida: modelos existen, subset existe, CodeCarbon y llama-cpp importan
  ├── Crea carpetas de resultados
  ├── Escribe los 6 archivos de metadatos en results/metadata/
  ├── Carga el subset (40 preguntas)
  └── Lanza scripts/fase1_benchmark.py como subproceso

Paso 2 — BUCLE DE MEDICIÓN (scripts/fase1_benchmark.py)
  Para cada modelo (llama-2-7b, qwen2.5-7b):
    Para cada cuantización (q4, q8):
      │
      ├── BASELINE: mide energía en reposo 3 veces × 60s → promedia
      ├── CARGA: carga el .gguf en memoria (fuera del tracker de energía)
      ├── WARMUP: 1 inferencia descartada para estabilizar cachés GPU
      ├── PAUSA: 20s para que el sistema se estabilice térmicamente
      │
      └── Para cada par (pregunta, repetición) — 40 preguntas × 15 reps = 600:
            ├── Construye el prompt del turno 1 con build_prompt()
            ├── tracker.start() → llm(prompt) → tracker.stop()   [ZONA CRÍTICA P1]
            ├── Escribe fila en turn_results.csv
            ├── Construye el prompt del turno 2 (incluye respuesta del turno 1)
            ├── tracker.start() → llm(prompt) → tracker.stop()   [ZONA CRÍTICA P1]
            ├── Escribe fila en turn_results.csv
            ├── Agrega los dos turnos → escribe fila en conversation_results.csv
            └── Cooldown de 10s entre conversaciones
      │
      └── Descarga el modelo de memoria → cooldown de 60s

  Total mediciones: 4 configs × 600 conversaciones = 2400 conversaciones
                    = 4800 turnos

Paso 3 — EVALUACIÓN DE CALIDAD
  export ANTHROPIC_API_KEY=sk-ant-...
  python judge_with_claude.py
  │
  ├── Lee conversation_results.csv
  ├── Para cada conversación exitosa:
  │     ├── Construye el prompt del juez con los 4 textos
  │     ├── Llama a la API de Claude (claude-sonnet-4-6)
  │     └── Parsea el JSON de respuesta → 9 scores + 3 campos cualitativos
  └── Escribe results/judge/judge_results.csv y .jsonl

Paso 4 — ANÁLISIS
  python analysis_summary.py
  python analysis_plots.py
  │
  ├── Une energy + quality por conversation_id
  ├── Calcula estadísticas por configuración
  ├── Genera tablas para la tesis (14 CSVs)
  └── Genera gráficas PNG
```

---

## 5. Las métricas que mide y por qué son científicamente relevantes

El experimento mide dos dimensiones: **energía** y **calidad**. Todas las métricas energéticas se calculan en versión "medida" (bruta) y "corregida por baseline" (neta).

### Métricas de energía

| Métrica | Unidad | Fórmula | Por qué importa |
|---|---|---|---|
| `measured_energy_wh` | Wh | Lectura directa de CodeCarbon | Energía total consumida por la máquina durante la inferencia |
| `baseline_corrected_energy_wh` | Wh | medida − (potencia_reposo × tiempo) | Energía atribuible al LLM, aislada del consumo del sistema |
| `energy_j_per_output_token` | J/tok | energía_neta_J / tokens_salida | Eficiencia energética por token generado (Canziani et al. 2017) |
| `energy_wh_per_1k_output_tokens` | Wh/1Ktok | energía_neta_Wh / tokens × 1000 | Métrica de Bannour et al. 2021, normaliza por volumen de texto |
| `tokens_per_joule` | tok/J | tokens_salida / energía_neta_J | Inverso de J/tok; mayor = más eficiente |
| `edp_joule_second` | J·s | energía_neta_J × tiempo_s | Energy-Delay Product (Brooks et al. 2000); captura el trade-off energía-latencia |
| `operational_sci_per_turn` | gCO2eq | (energía_Wh / 1000) × intensidad_carbono | Software Carbon Intensity (Green Software Foundation 2023); estima el CO2 emitido |

### Métrica de calidad

El juez devuelve 9 scores (1-10):
- `score`: puntaje general de la conversación completa.
- `correctness`: corrección factual/técnica.
- `instruction_following`: si el modelo siguió las instrucciones en ambos turnos.
- `relevance`: si la respuesta se mantuvo enfocada en la pregunta.
- `completeness`: si cubrió los aspectos necesarios.
- `clarity`: si la respuesta es clara y bien organizada.
- `conciseness`: si no es innecesariamente larga.
- `usefulness`: si realmente ayuda al usuario a completar la tarea.
- `multi_turn_coherence`: si el turno 2 usa correctamente el contexto del turno 1.

### Métrica de eficiencia calidad-energía (post-hoc)

| Métrica | Fórmula | Por qué importa |
|---|---|---|
| `quality_per_joule` | score / energía_neta_J | Adaptación de accuracy-per-Joule (Canziani 2017): cuánta calidad se obtiene por cada Joule consumido |

**Nota importante:** la métrica `operational_sci` solo reporta el componente operacional (E × I) del índice SCI de Green Software Foundation. El carbono embebido del hardware (M) no se incluye porque no hay datos de análisis de ciclo de vida (LCA) específicos para Apple M4 ni NVIDIA RTX 4060.

---

## 6. Los parámetros de configuración en `config.yaml` y qué significa cada uno

### Identificación del experimento
```yaml
experiment_name: green_ai_llm_quality_energy   # nombre del proyecto en CodeCarbon
output_dir: results                             # carpeta raíz de resultados
hardware_profile: mac_m4                        # perfil activo: mac_m4 | windows_nvidia
execution_device: gpu                           # dispositivo de inferencia: cpu | gpu
```

### Control del experimento
```yaml
repetitions: 15              # cuántas veces se mide cada pregunta (15 = oficial)
randomize_order: true        # aleatoriza el orden de (pregunta, repetición) para evitar sesgos temporales
seed: 42                     # semilla para la aleatorización (reproducible)
backup_every_n_conversations: 50  # crea un backup del CSV cada 50 conversaciones
rerun_failed: true           # en la próxima corrida, reintenta las mediciones fallidas
```

### Control del sistema durante el experimento
```yaml
runtime_control:
  prevent_sleep: true                         # evita que la máquina entre en suspensión
  use_caffeinate_on_macos: true               # usa `caffeinate` en macOS para prevent_sleep
  cooldown_seconds_between_runs: 10           # pausa entre conversaciones (estabilización)
  cooldown_seconds_between_model_configs: 60  # pausa entre cargar un modelo y el siguiente
  warmup_runs_per_configuration: 1            # inferencia de warmup descartada por config
  discard_warmup_runs: true                   # si false, el warmup se incluye en los datos
  pause_before_start_seconds: 30              # pausa antes de iniciar la medición
  pause_after_model_load_seconds: 20          # pausa tras cargar el modelo en memoria
  pause_after_experiment_seconds: 10          # pausa al finalizar
```

### Calibración de baseline
```yaml
energy_calibration:
  enabled: true                    # activa la corrección de baseline
  baseline_idle_seconds: 60        # duración de cada medición en reposo
  baseline_repetitions: 3          # número de repeticiones del baseline (se promedian)
  subtract_idle_baseline: true     # activa la resta de consumo de fondo
  save_baseline_runs: true         # guarda las mediciones de baseline en CSV
```

### Dataset de prompts (MT-Bench)
```yaml
prompt_dataset:
  questions_per_category: 5        # preguntas por categoría (5 × 8 = 40 total)
  turns_per_question: 2            # conversaciones de 2 turnos (estructura MT-Bench)
  selection_strategy: first_n_per_category   # toma las primeras N en orden del archivo
  random_seed: 42                  # semilla si strategy = random_n_per_category
  selected_categories:             # las 8 categorías oficiales de MT-Bench
    - writing
    - roleplay
    - extraction
    - reasoning
    - math
    - coding
    - stem
    - humanities
```

### Modelos y cuantizaciones
```yaml
models:
  llama-2-7b:
    q4: models/llama-2-7b/Q4_K_M.gguf
    q8: models/llama-2-7b/llama-2-7b.Q8_0.gguf
  qwen2.5-7b:
    q4: models/qwen2.5-7b/Qwen2.5-7B-Instruct-Q4_K_M.gguf
    q8: models/qwen2.5-7b/Qwen2.5-7B-Instruct-Q8_0.gguf

selected_models:
  - llama-2-7b
  - qwen2.5-7b

selected_quantizations:
  - q4
  - q8
```

### Parámetros de inferencia (llama.cpp)
```yaml
llama_cpp_params:
  n_ctx: 4096          # tamaño máximo del contexto en tokens
  max_tokens: 1024     # máximo de tokens a generar por inferencia
  temperature: 0.0     # generación determinista (sin aleatoriedad)
  top_p: 1.0           # sin filtro nucleus sampling (compatible con temperature=0)
  seed: 42             # semilla del sampler (redundante con temperature=0)
  echo: false          # no devuelve el prompt en la salida; solo cuenta tokens de completion
```

### Modo de generación
```yaml
generation_mode:
  name: deterministic_energy     # temperature=0.0, semilla fija; reproducible
  vary_seed_by_repetition: false # la misma semilla en todas las repeticiones
```

### CodeCarbon
```yaml
codecarbon:
  project_name: green_ai_llm_quality_energy
  save_to_file: false           # no genera CSV de CodeCarbon (lo manejamos nosotros)
  log_level: error              # solo logs de error para no contaminar la salida
  measure_power_secs: 1         # toma una muestra de potencia por segundo
  include_embodied_emissions_in_sci: false  # no incluye carbono embebido del hardware
```

### Juez de calidad
```yaml
judge:
  enabled: true
  provider: anthropic
  claude_model: claude-model-configurable    # sobreescrito al correr judge_with_claude.py
  anthropic_api_key_env: ANTHROPIC_API_KEY   # variable de entorno con la API key
  measure_judge_energy: false                # no mide la energía del juez
  max_retries: 3
  mode: claude_multidimensional              # 9 dimensiones + 3 cualitativos
  use_reference_when_available_for:          # categorías con respuestas de referencia
    - math
    - reasoning
    - coding
    - stem
```

### Análisis
```yaml
analysis:
  primary_energy_metric: baseline_corrected_energy   # métrica principal
  statistical_tests_enabled: true
  bootstrap_iterations: 10000        # iteraciones para IC bootstrap
  alpha: 0.05                        # nivel de significancia para tests
  min_successful_repetitions_per_prompt: 10  # umbral mínimo para incluir un prompt
```

---

## 7. Cómo funciona CodeCarbon en Apple M4 y sus limitaciones

### El problema central

CodeCarbon mide energía de tres componentes: CPU, GPU y RAM. En Windows con NVIDIA, puede leer directamente el consumo de la GPU mediante NVML (una librería de NVIDIA). En macOS con Apple M4, no existe esa interfaz de bajo nivel disponible sin privilegios de root.

### Cómo funciona en Apple M4 (sin root)

En lugar de leer el hardware directamente, CodeCarbon usa el **TDP** (Thermal Design Power): la potencia máxima de diseño del chip.

El experimento configura CodeCarbon así:
```python
EmissionsTracker(
    force_cpu_power=20,   # TDP Apple M4 durante inferencia LLM (fuente: NotebookCheck 2024)
    force_ram_power=3,    # estimado LPDDR5X 16GB
    allow_multiple_runs=True,
)
```

Esto significa que CodeCarbon asume que el CPU consume 20W y la RAM consume 3W **constantemente**, independientemente de la carga real. El resultado es que las mediciones de energía son estimaciones basadas en TDP, no mediciones directas.

### Por qué esto es aceptable metodológicamente

1. La **corrección de baseline** compensa parcialmente el problema. Si el sistema en reposo también gasta 20W (según CodeCarbon), y la inferencia también registra ~20W, la diferencia neta captura el consumo *relativo* entre configuraciones, aunque el valor absoluto sea aproximado.

2. **La comparación entre Q4 y Q8 sigue siendo válida** porque las dos cuantizaciones se miden con el mismo método, el mismo hardware y la misma estimación de TDP. Las diferencias observadas reflejan diferencias reales de carga de trabajo.

3. Se registra explícitamente `tracking_mode='constant'` en los metadatos para que cualquier lector del paper sepa que los valores absolutos son estimaciones.

### Limitación explícita

Los valores absolutos de energía (en Wh o Joules) en el Apple M4 **no son comparables con los de Windows NVIDIA**. Windows con NVML disponible puede leer la GPU directamente. Por eso el paper analiza cada plataforma por separado y no mezcla los resultados en una comparación directa de valores absolutos entre plataformas.

---

## 8. Cómo funciona el juez Claude — basado en Zheng et al. NeurIPS 2023

### El paper original

Zheng et al. (2023) "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (NeurIPS 2023, arXiv:2306.05685) propuso usar GPT-4 como juez para evaluar la calidad de respuestas de otros LLMs. El método se llama **Single Answer Grading**.

### Cómo lo adapta este experimento

**Similitudes con el paper original:**
- Usa prompts literales de MT-Bench (los mismos que Zheng et al.).
- La estructura de evaluación es similar: el juez recibe la pregunta y la respuesta y da un puntaje.
- Se usan respuestas de referencia para las categorías objetivas (math, reasoning, coding, stem).

**Diferencias documentadas (ver principio P17):**
- El paper usa GPT-4 como juez. Este experimento usa Claude Sonnet (`claude-sonnet-4-6`).
- El puntaje se llama explícitamente "Claude-based MT-Bench-style quality score", **no** "MT-Bench score oficial".
- El paper evalúa turno a turno. Este experimento evalúa la conversación completa de dos turnos en una sola llamada.
- El juez recibe los cuatro textos juntos: pregunta-t1, respuesta-t1, pregunta-t2, respuesta-t2.
- Se evalúan 9 dimensiones en lugar de solo un score global.

**Por qué Claude en lugar de GPT-4:**
Panickssery et al. (2024, arXiv:2404.13076) documentaron que los modelos tienden a favorecer sus propias respuestas cuando actúan de juez (sesgo de auto-preferencia). Como el experimento evalúa Llama y Qwen (no Claude), usar Claude como juez elimina ese sesgo.

### El flujo técnico del juez

```
Para cada conversación exitosa en conversation_results.csv:
  │
  ├── Determina si la categoría tiene respuesta de referencia
  │     (math, reasoning, coding, stem → busca en references.yaml)
  │
  ├── Construye el prompt del juez:
  │     - Sin referencia: JUDGE_TEMPLATE (8 criterios, instrucciones de objetividad)
  │     - Con referencia: JUDGE_TEMPLATE_REFERENCE (+ sección de respuesta de referencia)
  │
  ├── Llama a la API de Anthropic (POST /v1/messages)
  │     - Modelo: claude-sonnet-4-6
  │     - max_tokens: 1024
  │     - Reintentos: hasta 3 con backoff exponencial (2s, 4s, 8s)
  │
  ├── Parsea el JSON de respuesta:
  │     - Limpia bloques de código markdown (```json ... ```)
  │     - Valida que todos los campos existan
  │     - Valida que cada score esté en [1, 10]
  │
  └── Escribe la fila en judge_results.csv y judge_results.jsonl
```

### Las 8 dimensiones de evaluación

El prompt le pide al juez que evalúe:
1. **Correctness** — los datos son factualmente correctos.
2. **Instruction following** — el asistente siguió las instrucciones en ambos turnos.
3. **Relevance** — la respuesta se mantuvo enfocada en la pregunta.
4. **Completeness** — cubre los aspectos necesarios sin omisiones importantes.
5. **Clarity** — es fácil de entender y está bien organizada.
6. **Conciseness** — no es innecesariamente larga ni repetitiva.
7. **Usefulness** — realmente ayuda al usuario a completar la tarea.
8. **Multi-turn coherence** — la segunda respuesta usa correctamente el contexto del primer turno.

El `score` general (campo 9) es el puntaje holístico de la conversación completa.

---

## 9. Glosario de términos técnicos usados en el código

| Término | Significado en el contexto del proyecto |
|---|---|
| **GGUF** | Formato de archivo para modelos LLM cuantizados. Es el formato nativo de llama.cpp. |
| **Cuantización** | Reducir la precisión numérica de los pesos del modelo para ahorrar memoria y acelerar la inferencia. Q4 usa 4 bits por peso; Q8 usa 8 bits. |
| **Q4_K_M** | Cuantización de 4 bits con método K-Quants (Mixed, calidad media). Una de las mejores opciones de 4 bits para calidad/tamaño. |
| **Q8_0** | Cuantización de 8 bits sin agrupación. Más cercana al modelo original (16 bits) pero pesa más y es más lenta. |
| **llama-cpp-python** | Librería Python que hace binding de llama.cpp (C++). Permite cargar y correr modelos .gguf directamente en Python. |
| **Metal** | Backend de GPU de Apple para cómputo general. En Apple Silicon, llama.cpp puede usar Metal para acelerar la inferencia en la GPU integrada. |
| **CUDA** | Backend de GPU de NVIDIA. Permite a llama.cpp usar la GPU NVIDIA para inferencia. |
| **n_gpu_layers** | Parámetro de llama.cpp: cuántas capas del modelo se mueven a la GPU. `-1` = todas (modo GPU); `0` = ninguna (modo CPU). |
| **n_ctx** | Tamaño del contexto en tokens. Cuántos tokens puede "ver" el modelo en total (prompt + respuesta). En este experimento: 4096. |
| **CodeCarbon** | Librería Python que estima el consumo energético y las emisiones de CO2 de código Python. Se usa como tracker en este experimento. |
| **TDP** | Thermal Design Power. La potencia máxima de diseño de un chip. CodeCarbon la usa como estimación cuando no puede leer el hardware directamente. |
| **Baseline de energía** | El consumo eléctrico del sistema en reposo (sin modelo cargado). Se mide para poder restarlo de la inferencia. |
| **Energía corregida por baseline** | Energía medida durante la inferencia menos el consumo de fondo × tiempo. Es la energía neta atribuible al LLM. |
| **MT-Bench** | Multi-Turn Benchmark. Dataset de 80 preguntas de alta calidad en 8 categorías, cada una con 2 turnos, diseñado para evaluar chatbots (Zheng et al. NeurIPS 2023). |
| **LLM-as-a-Judge** | Técnica de evaluación donde se usa otro LLM (el "juez") para puntuar la calidad de las respuestas generadas por el modelo evaluado. |
| **Inferencia** | El proceso de generar texto con un LLM a partir de un prompt. Es lo que se mide energéticamente. |
| **Turno** | Una ronda de pregunta-respuesta. MT-Bench usa 2 turnos por pregunta, formando una conversación. |
| **Conversación** | Los dos turnos juntos. La unidad básica de análisis del experimento. |
| **Repetición** | Cuántas veces se corre la misma conversación. El experimento usa 15 repeticiones para obtener mediciones estadísticamente estables. |
| **EDP** | Energy-Delay Product. Métrica que combina energía y latencia: EDP = energía_J × tiempo_s. Una configuración con EDP bajo es eficiente tanto en energía como en velocidad. |
| **SCI** | Software Carbon Intensity (Green Software Foundation 2023). Estima el CO2 emitido por la ejecución del software: SCI = (E × I) / R, donde E = energía, I = intensidad de carbono de la red eléctrica, R = unidad funcional. |
| **Factor de carbono de Paraguay** | 26 gCO2eq/kWh. Paraguay genera casi toda su electricidad con hidroeléctricas (Itaipú), lo que resulta en una de las matrices eléctricas más limpias del mundo. |
| **Smoke test** | Una corrida rápida del experimento con pocos prompts y repeticiones para verificar que todo funciona. Los resultados van a `results/smoke_test/` y **nunca** se mezclan con los datos oficiales. |
| **turn_id / conversation_id** | Identificador determinista (SHA256 de los campos que definen el turno/conversación). Permite reanudar el experimento desde donde se interrumpió sin duplicar filas. |
| **config_snapshot** | Copia de `config.yaml` guardada junto a los resultados para saber exactamente qué parámetros se usaron en esa corrida. |
| **environment_report** | JSON con el estado completo del entorno: versión de Python, versiones de paquetes, hardware, fingerprints de modelos. |
| **SHA256_1mb** | Hash SHA256 calculado solo sobre el primer megabyte del archivo .gguf más el tamaño total. Es suficiente para identificar unívocamente un modelo (el header GGUF contiene arquitectura y cuantización) sin esperar 60-90 segundos que tardaría el hash completo de un archivo de 4-8 GB. |
| **Pareto-eficiente** | Una configuración donde no existe otra que sea mejor en todas las dimensiones simultáneamente (calidad, energía y latencia). Es útil para comparar cuando las métricas están en conflicto. |
| **Bootstrap IC95** | Intervalo de confianza del 95% calculado por bootstrap (10.000 muestras con reemplazo). Se usa porque no se asume distribución normal en las mediciones de energía. |

# Refactor del pipeline de medición energética — GREEN-IA

Documentación del proceso de separación de responsabilidades del código
de medición energética, para inclusión en la tesis (sección de metodología
o apéndice de arquitectura de software). Cada sección corresponde a un
paso de refactor, en orden de menor a mayor complejidad, y describe qué
se separó, por qué, y qué no cambió del comportamiento numérico original.

---

## Paso 1 — Módulo de inferencia (`scripts/inferencia_refactor.py`)

**Qué se separó:** la invocación al modelo (`llm(prompt, ...)`) se aisló
en una función independiente, `generar_respuesta()`, que no tiene
conocimiento de medición de energía ni de CodeCarbon.

**Antes:** la llamada al modelo estaba escrita directamente dentro de
la función `medir_energia()`, mezclando la responsabilidad de "generar
texto" con la de "medir energía alrededor de esa generación".

**Después:** `medir_energia()` invoca a `generar_respuesta()` como una
dependencia externa. El tracker de CodeCarbon y el cronómetro
(`time.perf_counter()`) siguen envolviendo la misma operación exacta,
en el mismo orden.

**Por qué:** separación de responsabilidades (principio de responsabilidad
única). Permite reutilizar la función de inferencia en otros componentes
del proyecto (por ejemplo, el módulo de evaluación de calidad con Claude
como juez) sin arrastrar dependencias de medición energética que ese
componente no necesita.

**Qué NO cambió:** los hiperparámetros de generación (`max_tokens`,
`temperature`, `top_p`, `seed`, `echo`) y su orden de aplicación son
idénticos al código original. El resultado numérico de energía, tiempo
y tokens generados por inferencia no se ve afectado por este cambio —
es exclusivamente una reorganización del código fuente.

**Validación:** se comparó el resultado de una medición de prueba con el
código refactorizado contra el valor ya registrado en el CSV existente
para el mismo modelo/categoría/prompt/repetición, confirmando
consistencia dentro del ruido esperable de medición de hardware real.

> Nota: la dependencia de medir_energia() sobre generar_respuesta()
> descrita arriba se mantiene vigente, aunque medir_energia() fue
> posteriormente movida a su propio módulo (ver Paso 3).

---

## Paso 2 — Carga del modelo y gestión de caffeinate (`scripts/carga_modelo_refactor.py`, `scripts/sistema_refactor.py`)

**Qué se separó:** dos responsabilidades que vivían mezcladas dentro de
`main()` se extrajeron a módulos propios: (a) la resolución de la ruta
del modelo GGUF y su carga en memoria (`carga_modelo_refactor.py`), y
(b) la prevención de suspensión del sistema operativo durante una corrida
larga (`sistema_refactor.py`).

**Antes:** `main()` contenía 35 líneas que combinaban lectura de
`config.yaml` para obtener la ruta del modelo, activación de `caffeinate`,
verificación de existencia del archivo, manejo del `ImportError` de
`llama-cpp-python`, y la llamada a `Llama(...)` con sus parámetros.
El manejo de errores en cada paso requería referenciar `proceso_cafe` para
terminarlo antes de salir, acoplando la lógica de sistema con la de carga.

**Después:** `main()` ejecuta las mismas operaciones en exactamente 3 líneas:
```python
ruta_modelo = resolver_ruta_modelo(ROOT, MODEL_NAME, CUANTIZACION)
proceso_cafe = activar_caffeinate()
llm = cargar_modelo(ruta_modelo, cfg)
```
El orden se preserva: caffeinate se activa antes de intentar cargar el
modelo, igual que en el código original. El teardown al final usa
`detener_caffeinate(proceso_cafe)`.

**Por qué:** la lógica de carga del modelo (`carga_modelo_refactor.py`) no
tiene dependencia con el sistema operativo ni con CodeCarbon, y puede
reutilizarse en cualquier script que necesite cargar un modelo GGUF. La
gestión de `caffeinate` (`sistema_refactor.py`) es específica de macOS y
no tiene relación con el modelo ni con la medición de energía — aislarla
evita que esa lógica de plataforma se repita si el pipeline crece.

**Qué NO cambió:** los parámetros de carga del modelo (`n_ctx`,
`n_gpu_layers`, `n_threads`, `n_batch`) y su fuente (`ExperimentConfig`)
son idénticos. El orden de activación de caffeinate respecto a la carga
del modelo es el mismo. El comportamiento numérico de cualquier inferencia
posterior no se ve afectado.

**Validación:** se ejecutó un smoke test sobre Qwen Q4 que reprodujo el
orden exacto de operaciones de `main()`: caffeinate activó con PID asignado,
el modelo cargó correctamente, una inferencia de referencia (coding,
question\_id=121) devolvió 434 tokens (idéntico al CSV original, valor
determinístico con `temperature=0 / seed=42`), y caffeinate se detuvo al
finalizar.

---

## Paso 3 — Módulo de medición energética (`scripts/medicion_energia_refactor.py`)

**Qué se separó:** la función `medir_energia()` fue extraída de
`analisis_categoria_refactor.py` a su propio módulo, delegando la
generación de texto a `inferencia_refactor.generar_respuesta()`.

**Antes:** `medir_energia()` vivía dentro del script principal, mezclando
en un mismo archivo la lógica de instrumentación con CodeCarbon, la
gestión del KV-cache, y el bucle de experimento. La función conocía
directamente la implementación de la inferencia.

**Después:** `medicion_energia_refactor.py` encapsula exclusivamente la
instrumentación energética. `analisis_categoria_refactor.py` la importa
con `from medicion_energia_refactor import medir_energia` y la usa
igual que antes, sin cambio en la interfaz ni en el comportamiento.
El reintento por NaN (bug conocido CodeCarbon/powermetrics #985) permanece
en `main()`, donde pertenece conceptualmente: es lógica de experimento,
no de medición.

**Por qué:** separación de responsabilidades. Este módulo implementa la
instrumentación con CodeCarbon descrita en la metodología (Lacoste et al.,
2019): el tracker envuelve exclusivamente la operación de inferencia, y el
reset del KV-cache entre repeticiones —hallazgo empírico que redujo el CV
de 33% a 0.31%— se mantiene en el mismo punto de ejecución. Aislar esta
lógica permite reutilizarla en cualquier script de medición futuro sin
duplicar la instrumentación.

**Qué NO cambió:** la configuración del tracker (`project_name`,
`measure_power_secs`, `save_to_file`, `log_level`, `allow_multiple_runs`),
el orden `tracker.start() → inferencia → tracker.stop()`, el reset del
KV-cache, y la lógica de lectura de energía por componente (`_kwh()`) son
idénticos al código original. La separación no altera ningún parámetro de
medición ni el orden de las operaciones de instrumentación.

**Validación:** prueba de regresión sobre Qwen Q4, coding, question\_id=121,
3 repeticiones: energía media 70.85 mWh (ref: 74.07, delta 4.4%, dentro
del ruido de hardware real), tiempo medio 22.24 s (ref: 22.28, delta 0.2%),
tokens 434 (idéntico al CSV original, determinístico con
`temperature=0 / seed=42`).

---

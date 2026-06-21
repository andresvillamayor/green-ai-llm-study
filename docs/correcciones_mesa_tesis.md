# Correcciones Mesa de Tesis — GREEN-IA
Documento de referencia para redacción de tesis y presentación.
Última actualización: junio 2026

---

## REGLA GENERAL
Toda conclusión debe incluir una frase delimitadora:
- "en el entorno evaluado"
- "para los modelos y prompts seleccionados"
- "bajo las condiciones experimentales descritas"
- "los resultados sugieren"
- "se observó que"

---

## 4.1 — CO2eq operacional

NUNCA usar: "CO2 real"
SIEMPRE usar: "emisiones operacionales de CO2eq estimadas a partir
del consumo energético y un factor de emisión específico para Paraguay"

Fuente oficial del factor 26 gCO2eq/kWh:
Electricity Maps — https://app.electricitymaps.com/zone/PY
Fecha de acceso: 21 de junio de 2026

---

## 4.2 — Limitaciones GPU Apple Silicon

Agregar en toda comparación energética CPU vs GPU:
"La atribución energética de GPU en Apple Silicon M4 debe
interpretarse con cautela. CodeCarbon no dispone de acceso
directo al consumo GPU mediante APIs equivalentes a NVML de
NVIDIA. Las métricas de energía GPU son estimaciones derivadas
de contadores del sistema operativo."

Agregar en figuras de GPU:
"* Energía GPU estimada por CodeCarbon.
Apple Silicon no expone API directa de consumo GPU."

---

## 4.3 — Benchmark

NUNCA usar: "benchmark científico amplio" / "benchmark general"
SIEMPRE usar: "conjunto controlado de 15 prompts distribuidos
en 9 categorías basadas en MT-Bench (Zheng et al., 2023)"

Agregar en metodología:
"Los prompts cubren tareas de razonamiento, matemáticas,
escritura, extracción, código, STEM, humanidades, traducción
y prefill extenso. No pretenden representar la totalidad de
tareas de inferencia LLM. Las conclusiones se limitan al
comportamiento energético bajo estas categorías específicas."

---

## 4.4 — Métricas normalizadas

Reportar siempre junto con la energía total:
- Energía por token (µWh/token)
- Energía por 1.000 tokens (µWh)
- Tokens por joule
- Latencia por token (ms/token)
- CO2eq por 1.000 tokens (µg)

---

## 4.5 — Afirmaciones absolutas

| NUNCA usar | SIEMPRE usar |
|---|---|
| "Q4 siempre gana a Q8" | "En el entorno evaluado, Q4 mostró mayor velocidad y menor consumo estimado que Q8 para los modelos y prompts seleccionados" |
| "La RAM domina en los 8 casos" | "Los resultados sugieren que la energía atribuida a RAM representa la fracción más alta en la mayoría de las configuraciones evaluadas, aunque la atribución por componente requiere validación adicional" |
| "La CPU le gana a la GPU" | "En ciertas configuraciones, la CPU mostró mayor throughput que la GPU Metal en el entorno evaluado. Esta observación no es generalizable" |
| "El idioma no afecta los resultados" | "Todos los prompts se formularon en inglés para mantener constante el idioma de entrada" |
| "La configuración más limpia" | "La configuración con menor consumo estimado de CO2eq operacional en el entorno evaluado" |
| "Benchmarking científico desde casa" | "Evaluación experimental controlada de inferencia LLM en hardware de consumo general" |

---

## 4.6 — CPU vs GPU — hipótesis, no causa demostrada

NUNCA usar explicación causal directa.
SIEMPRE usar:
"Se observó que en configuraciones con modelos Q4, la ejecución
en CPU mostró mayor throughput que en GPU Metal para ciertos
modelos. Una hipótesis plausible es que el overhead de offloading
de capas a la GPU puede superar el beneficio del paralelismo GPU
para modelos de 7B parámetros con cuantización Q4 en arquitecturas
de memoria unificada. Sin embargo, esta hipótesis no puede
demostrarse con los datos disponibles, ya que Apple Silicon no
expone métricas de utilización GPU comparables a NVIDIA.
Se recomienda validación futura con Instruments de Xcode."

---

## 4.7 — Outliers — tratamiento completo

diagnosticar_datos.py reporta por cada configuracion:
- Q1, Q3, IQR, umbral inferior, umbral superior
- Numero exacto de outliers con rep y prompt
- Duracion y energia CPU/GPU/RAM de cada outlier
- Potencia CPU media durante el outlier
- Frecuencia: outliers/total por configuracion (%)
- Tabla comparativa con/sin outliers + delta%
- Analisis de sensibilidad automatico

Redaccion para tesis:
"Las corridas anomalas muestran un patron consistente:
tiempos entre 852s y 1031s con potencia CPU cercana a 18 mW,
valores compatibles con throttling termico o ejecucion en
E-cores segun registros del sistema operativo. La frecuencia
de ocurrencia es del 3.5% del total de mediciones (28 de 800).
Se aplico criterio Tukey IQR x3 (1977). Los resultados se
presentan con y sin outliers en todas las figuras."

---

## 4.8 — Bibliografía

NUNCA usar: "13 fuentes revisadas por pares"
SIEMPRE usar: "13 fuentes bibliograficas incluyendo articulos
revisados por pares, preprints tecnicos, documentacion
y fuentes complementarias"

Referencia correcta de Llama 2:
Touvron et al. (2023). Llama 2: Open Foundation and
Fine-Tuned Chat Models. arXiv:2307.09288.
NOTA: es preprint, no peer-reviewed.

Clasificacion pendiente: pasar lista completa de referencias
y clasificar en: peer-reviewed / preprint / reporte
tecnico / documentacion / fuente web.

---

## 4.9 — Reproducibilidad

Todo lo que pide la mesa esta capturado en:

EN EL CSV (automatico por experimento):
- macos_version, python_version, llama_cpp_version
- n_gpu_layers, n_threads, n_batch, seed
- n_ctx, max_tokens, temperature, top_p
- cpu_count, cpu_model, ram_total_gb
- metal_backend, prompt_template, warmup_done
- llama_cpp_commit

EN EL README (seccion Reproducibilidad):
- Hardware exacto (Mac Mini M4, 16GB, 10 CPU, 10 GPU cores)
- Software (macOS 15.6.1, Python 3.13.5, llama.cpp 0.3.21)
- Flags de compilacion (GGML_METAL=on)
- Todos los parametros de inferencia en tabla
- Procedimiento de warm-up documentado
- Control de procesos en segundo plano
- Orden de ejecucion
- Sincronizacion CodeCarbon con inferencia
- Nota sobre powermetrics

LO QUE NO SE PUEDE CAPTURAR EN APPLE SILICON:
- Utilizacion real GPU (Metal no expone API publica)
- Temperatura (requiere sudo powermetrics)
- Separacion prefill/decode (no expuesto por llama.cpp)
Documentado como limitacion en seccion de amenazas.

---

## 4.10 — Estadística formal

graficar_resultados.py reporta ahora:

ESTADISTICAS DESCRIPTIVAS (en stats groupby):
- media, desviacion estandar
- mediana, IQR
- coeficiente de variacion (CV%)
- intervalos de confianza al 95% con t de Student
- resultados con y sin outliers en todas las figuras

PRUEBAS DE COMPARACION (funcion pruebas_estadisticas):
- Mann-Whitney U bilateral (alpha=0.05)
- No parametrica: no asume distribucion normal
- Compara Q4 vs Q8 por modelo para:
  velocidad (tok/s), energia (Wh), CO2 (mg)
- Reporta: U, p-value, significancia
- Test de normalidad Shapiro-Wilk para justificar
  uso de prueba no parametrica

Redaccion para tesis:
"Dado que se detectaron outliers severos (3.5% de mediciones
con tiempos superiores a 850s), se uso la mediana y el IQR
como estadisticos principales. Las comparaciones entre
configuraciones se realizaron con la prueba Mann-Whitney U
(bilateral, alpha=0.05), apropiada para distribuciones
no normales confirmadas por el test de Shapiro-Wilk."

---

## SECCIÓN NUEVA REQUERIDA — Amenazas a la validez

Agregar sección 7 en la tesis con estos cuatro puntos:

### 7.1 Validez interna
Planificación del SO, procesos en segundo plano, temperatura,
throttling, asignación dinámica de P-cores y E-cores,
estado energético del sistema y limitaciones de CodeCarbon
en Apple Silicon.

### 7.2 Validez externa
Los resultados no se generalizan automáticamente a otros modelos,
tamaños, formatos de cuantización, GPUs NVIDIA, CPUs x86,
otros sistemas operativos o cargas de trabajo más complejas.

### 7.3 Validez de constructo
Parte de las métricas son estimadas. La atribución energética
por componente, especialmente RAM y GPU, requiere validación
más sólida.

### 7.4 Validez estadística
10 repeticiones pueden ser insuficientes cuando existen valores
extremos. Se aplicó estadística robusta (mediana, IQR, IC 95%
con t de Student) y análisis de sensibilidad con/sin outliers.

---

## SECCIÓN 5 — COMENTARIOS POR SECCIÓN

### 5.1 Portada

CAMBIAR subtítulo de:
"medido en hardware hogareño, con respaldo científico"

POR:
"Evaluación experimental del consumo energético y emisiones
estimadas en inferencia local de modelos de lenguaje abiertos"

Motivo: tono promocional no apropiado para trabajo IEEE.

### 5.2 Objetivo y contexto

CAMBIO 1 — Diferenciar entrenamiento vs inferencia:
Cuando se mencione GPT-3 u otros modelos grandes,
agregar siempre esta aclaracion:

"Las emisiones de entrenamiento de GPT-3 se citan como
contexto motivacional. Este trabajo mide exclusivamente
el consumo energetico durante la fase de inferencia,
no durante el entrenamiento."

CAMBIO 2 — Eliminar "CO2 real" (ver 4.1):
Reemplazar toda aparicion de "CO2 real" por
"CO2eq operacional estimado"

Redaccion sugerida para la seccion de objetivo:
"Este trabajo evalua el consumo energetico y las emisiones
operacionales estimadas de CO2eq durante la inferencia local
de modelos de lenguaje abiertos. El estudio no aborda el
consumo de la fase de entrenamiento, que queda fuera
del alcance de este experimento."

### 5.3 Diseño experimental

CAMBIO 1 — Justificacion de repeticiones:
NUNCA decir: "5 repeticiones es el estandar cientifico"
SIEMPRE decir:
"Se realizaron 10 repeticiones por configuracion para
permitir el calculo de intervalos de confianza al 95%
mediante la distribucion t de Student y detectar
variabilidad intra-configuracion. En presencia de outliers
severos, se reportan adicionalmente mediana e IQR como
estadisticos robustos."

CAMBIO 2 — Informacion critica a agregar en la seccion:
Mencionar explicitamente en la presentacion:
- n_ctx = 1024 tokens
- max_tokens = 256
- temperature = 0.7
- top_p = 0.9
- seed = 42 (reproducibilidad)
- n_threads = 8
- n_batch = 512
- warm-up: gc.collect() + sleep(5) entre configuraciones
- orden: modelo > cuantizacion > dispositivo > prompt > rep
- control: sudo purge antes del experimento
- backend GPU: Metal (GGML_METAL=on)

NOTA: todos estos parametros quedan registrados
automaticamente en el CSV del experimento.

### 5.4 Modelos

CAMBIO 1 — Especificar version base vs instruct:
Llama-2-7B  → version BASE
    archivo: llama-2-7b/Q4_K_M.gguf
             llama-2-7b/llama-2-7b.Q8_0.gguf
    fuente: TheBloke/Llama-2-7B-GGUF (HuggingFace)
    tipo: PTQ (Post-Training Quantization)

Qwen2.5-7B  → version INSTRUCT
    archivo: Qwen2.5-7B-Instruct-Q4_K_M.gguf
             Qwen2.5-7B-Instruct-Q8_0.gguf
    fuente: bartowski/Qwen2.5-7B-Instruct-GGUF (HuggingFace)
    tipo: PTQ (Post-Training Quantization)

NOTA para la tesis: aclarar que Llama-2 es version base
y Qwen2.5 es version instruct. Esto puede introducir
una variable de confusion que debe mencionarse como
limitacion del estudio.

CAMBIO 2 — Comparacion con modelos mas grandes:
NUNCA afirmar que un modelo "supera" a modelos mas grandes
sin especificar benchmark, metrica y configuracion exacta.
ELIMINAR cualquier comparacion de este tipo de la
presentacion y tesis.

---

### 5.5 Benchmark

CAMBIO 1 — Nomenclatura:
NUNCA usar: "benchmark general" / "benchmark cientifico"
SIEMPRE usar: "conjunto controlado de 15 prompts en
9 categorias basadas en MT-Bench (Zheng et al., 2023)"

CAMBIO 2 — Eliminar frase sobre idioma:
ELIMINAR: "el idioma no afecta los resultados"
REEMPLAZAR por:
"Todos los prompts se formularon en ingles para mantener
constante el idioma de entrada durante los experimentos.
El efecto del idioma sobre el consumo energetico no fue
evaluado en este estudio."

CAMBIO 3 — Alcance explicito:
Agregar en la seccion de benchmark:
"Los resultados se limitan al comportamiento energetico
bajo estas 9 categorias especificas de prompts y no
deben generalizarse a otros tipos de tareas de inferencia."

### 5.6 Metodologia

CAMBIO 1 — Tabla de fuentes por metrica:
Agregar en la presentacion y tesis esta tabla:

| Metrica              | Fuente          | Tipo        |
|----------------------|-----------------|-------------|
| Energia total (Wh)   | CodeCarbon      | Estimada    |
| Energia CPU (Wh)     | CodeCarbon      | Estimada    |
| Energia GPU (Wh)     | CodeCarbon      | Aproximada* |
| Energia RAM (Wh)     | CodeCarbon      | Estimada    |
| CO2eq operacional    | CodeCarbon +    | Calculada   |
|                      | factor PRY      |             |
| Velocidad (tok/s)    | llama.cpp +     | Medida      |
|                      | tiempo Python   |             |
| Tiempos inferencia   | time.time()     | Medida      |
| Deteccion outliers   | IQR x3 Tukey   | Calculada   |
|                      | sobre tiempo_s  |             |

* GPU aproximada: Apple Silicon no expone API directa
  equivalente a NVML de NVIDIA.

CAMBIO 2 — Aclarar rol de powermetrics:
powermetrics NO se usa en el experimento final.
Se uso en experimentos exploratorios previos para
validar cualitativamente el comportamiento del chip.

Redaccion para tesis:
"CodeCarbon es la unica fuente de metricas energeticas
en este experimento. powermetrics fue utilizado en la
fase exploratoria para observar el comportamiento del
planificador del sistema operativo durante inferencias
anomalas, pero no forma parte del pipeline de medicion
del experimento principal."

CAMBIO 3 — Agregar diagrama de flujo en presentacion:
El flujo de medicion debe mostrar explicitamente:
prompt → tracker.start() → llm() → tracker.stop()
→ extraccion de metricas → fila CSV

### 5.7 Resultados de energia por componente

CAMBIO 1 — Eliminar afirmacion absoluta sobre RAM:
NUNCA usar: "la RAM domina en todos los casos"
SIEMPRE usar:
"Los resultados sugieren un comportamiento compatible
con inferencia limitada por memoria, aunque la atribucion
energetica por componente requiere validacion adicional."

CAMBIO 2 — Explicar como CodeCarbon estima por componente:
Agregar en metodologia o pie de figura:

"CodeCarbon estima la energia por componente de la
siguiente forma:
- CPU: a partir de TDP del procesador y porcentaje
  de utilizacion reportado por el sistema operativo
- RAM: a partir del consumo tipico por GB y la
  memoria total del sistema
- GPU: estimacion indirecta en Apple Silicon
  (ver limitacion 4.2)
La energia total es la suma de los tres componentes.
No se ha identificado riesgo de doble conteo en la
metodologia de CodeCarbon para esta configuracion,
pero la validacion cruzada con otras herramientas
queda como trabajo futuro."

CAMBIO 3 — Agregar nota en fig1_energia_desglosada:
Al pie de la figura de barras apiladas agregar:
"Nota: atribucion por componente estimada por CodeCarbon.
Los valores de GPU deben interpretarse con cautela
en Apple Silicon (ver seccion de limitaciones)."

### 5.8 Resultados de velocidad

CAMBIO 1 — Presentar CPU vs GPU como dependiente
de configuracion:
NUNCA usar: "la CPU supera a la GPU"
SIEMPRE usar:
"En las configuraciones evaluadas, se observaron
diferencias en throughput entre CPU y GPU Metal
que varian segun el modelo y nivel de cuantizacion.
Llama-2-7B Q4 mostro mayor velocidad en CPU que en GPU,
mientras que Qwen2.5-7B Q8 mostro el patron inverso.
Estas diferencias son especificas del entorno evaluado
y no son generalizables a otras arquitecturas."

CAMBIO 2 — Barras de error y significancia:
Ya implementado en graficar_resultados.py:
- IC 95% con t de Student en todas las figuras -OK
- Barras de error en fig2_velocidad_inferencia -OK
- Mann-Whitney U para comparacion Q4 vs Q8 -OK

Para CPU vs GPU agregar prueba Mann-Whitney U
en pruebas_estadisticas(). Pedirle a Claude Code
que agregue esta comparacion en la siguiente sesion.

CAMBIO 3 — Tabla de velocidad con IC 95%:
Agregar en la tesis una tabla con:
| Config          | Mediana tok/s | IC 95%  | CV%  |
|-----------------|---------------|---------|------|
| Llama Q4 CPU    |               |         |      |
| Llama Q4 GPU    |               |         |      |
| ...             |               |         |      |
(valores se completan con CSV del experimento final)

### 5.9 Resultados de emisiones

CAMBIO 1 — Presentar emisiones como estimadas:
NUNCA usar: "emisiones medidas" o "CO2 real"
SIEMPRE usar: "emisiones operacionales de CO2eq estimadas"

CAMBIO 2 — Cambiar unidad de reporte:
Los valores en µg por inferencia son muy pequeños
y dificiles de interpretar. Reportar adicionalmente:
- CO2eq por 1.000 tokens (ya esta en graficar_resultados.py)
- CO2eq por 1.000 inferencias

Agregar en graficar_resultados.py el calculo de
CO2eq por 1.000 inferencias:
co2_1k_inf = stats["co2_media"] * 1000  # mg por 1000 infs

CAMBIO 3 — Redaccion para tesis:
"Las emisiones operacionales de CO2eq estimadas por
inferencia son valores pequeños debido a la baja
intensidad carbonica de Paraguay (26 gCO2eq/kWh,
matriz 100% hidroelectrica). Para facilitar la
interpretacion se reportan en unidades de CO2eq
por 1.000 tokens generados y por 1.000 inferencias,
lo que permite comparacion con otros estudios
realizados en distintos contextos energeticos."

### 5.10 Comparacion Llama vs Qwen

CAMBIO 1 — Demostrar formalmente "mas estable":
NUNCA usar: "Qwen es mas estable que Llama"
sin mostrar los datos que lo demuestran.

SIEMPRE acompanar con tabla comparativa de CV%:
| Modelo     | CV% tiempo | CV% energia | Outliers |
|------------|------------|-------------|----------|
| Llama-2-7B |            |             |          |
| Qwen2.5-7B |            |             |          |
(completar con resultados del experimento final)

Si CV% de Qwen es menor que Llama, entonces se puede
afirmar: "Qwen2.5-7B mostro menor coeficiente de
variacion en tiempo de inferencia, lo que sugiere
mayor estabilidad en el entorno evaluado."

Si no hay diferencia significativa en Mann-Whitney U,
no se puede afirmar que uno es mas estable que el otro.

CAMBIO 2 — Separar eficiencia de calidad:
ELIMINAR cualquier referencia a calidad de respuesta.
Este experimento no evalua calidad, solo consumo
energetico y velocidad de inferencia.

Agregar esta aclaracion en la tesis:
"Este estudio no evalua la calidad de las respuestas
generadas por los modelos. Las comparaciones se limitan
a metricas de consumo energetico, velocidad de inferencia
y emisiones estimadas de CO2eq. La evaluacion de calidad
queda como trabajo futuro."

CAMBIO 3 — Redaccion correcta de la conclusion:
NUNCA usar: "Llama mas rapido, Qwen mas estable"
SIEMPRE usar:
"En el entorno evaluado, Llama-2-7B mostro mayor
velocidad de inferencia en configuraciones CPU con
cuantizacion Q4. Qwen2.5-7B mostro menor coeficiente
de variacion en ciertas configuraciones, lo que sugiere
mayor consistencia en esos casos especificos. Estas
observaciones no son generalizables y deben validarse
en otros entornos de hardware."

### 5.11 Aporte cientifico

CAMBIO 1 — Reformular el aporte:
NUNCA usar: "ciencia de IA sin data center"
SIEMPRE usar:
"El trabajo demuestra la factibilidad de realizar
mediciones controladas de inferencia LLM en hardware
de consumo, integrando metricas de rendimiento, energia
y emisiones estimadas bajo un protocolo reproducible."

CAMBIO 2 — Tres aportes concretos para la tesis:

Aporte 1 — Metodologico:
"Se disenio y ejecuto un protocolo experimental
reproducible para medir consumo energetico en
inferencia LLM local, documentando todos los
parametros necesarios para su replicacion."

Aporte 2 — Empirico:
"Se obtuvieron 1200 mediciones sobre hardware Apple
Silicon M4 comparando dos modelos, dos cuantizaciones
y dos modos de ejecucion, con estadistica formal
incluyendo IC 95% y pruebas Mann-Whitney U."

Aporte 3 — Contextual:
"Se integro la intensidad carbonica de Paraguay
(26 gCO2eq/kWh, Electricity Maps 2026) para estimar
emisiones operacionales de CO2eq en un contexto
energetico de baja huella carbono, contribuyendo
a la linea de investigacion Green AI en America Latina."

---

## SECCION 6 — REVISIONES OBLIGATORIAS
## Estado de cumplimiento

1.  CO2 real → CO2eq operacional estimado
    RESUELTO — tabla de reemplazos en 4.1 y 4.5

2.  Fuente oficial factor emision Paraguay
    RESUELTO — Electricity Maps 21 jun 2026
    26 gCO2eq/kWh, captura de pantalla en docs/

3.  Energia por token y CO2eq por 1.000 tokens
    RESUELTO — energy_per_token, energy_per_1k_tokens,
    co2_1k en CSV y fig6_metricas_normalizadas

4.  Tokens entrada, salida y totales
    RESUELTO — tokens_input, tokens_output,
    tokens_total en CSV

5.  Ampliar benchmark o limitar alcance
    RESUELTO — 15 prompts en 9 categorias MT-Bench
    + redaccion de alcance explicito en 5.5

6.  Parametros llama.cpp
    RESUELTO — n_ctx, max_tokens, temperature,
    top_p, seed, n_threads, n_batch, n_gpu_layers
    en CSV y README

7.  Version, commit y flags de compilacion
    RESUELTO — llama_cpp_version, llama_cpp_commit,
    metal_backend en CSV. GGML_METAL=on en README

8.  Hardware y sistema operativo completos
    RESUELTO — seccion Reproducibilidad en README
    + macos_version, cpu_model, ram_total_gb en CSV

9.  Analisis estadistico formal
    RESUELTO — media, mediana, std, IQR, CV,
    IC 95% t-Student, Mann-Whitney U en
    graficar_resultados.py

10. Resultados con y sin outliers
    RESUELTO — barras dobles en todas las figuras,
    tabla comparativa con delta% en diagnosticar_datos.py

11. Corregir y clasificar referencias
    PENDIENTE — necesito lista completa de referencias

12. Eliminar Wikipedia como referencia principal
    PENDIENTE — necesito lista completa de referencias

13. Distinguir peer-reviewed, preprints, reportes
    PENDIENTE — necesito lista completa de referencias

14. Suavizar afirmaciones absolutas
    RESUELTO — tabla de reemplazos en 4.5
    y correcciones en 5.7 a 5.11

15. Seccion amenazas a la validez
    RESUELTO — redaccion completa en 4.1 a 4.10
    Lista para incluir en tesis como seccion 7

16. Limitaciones CodeCarbon en Apple Silicon
    RESUELTO — documentado en 4.2, tabla de
    fuentes en 5.6, nota en figuras GPU

17. CPU vs GPU como observacion dependiente
    RESUELTO — reformulacion en 4.6 y 5.8

18. Validar atribucion energetica por componente
    RESUELTO — explicacion en 5.7 con nota de
    limitacion en pie de figura

19. Barras de error e IC en figuras
    RESUELTO — IC 95% con t-Student en todas
    las figuras de graficar_resultados.py

20. Separar medicion, estimacion e interpretacion
    RESUELTO — tabla de fuentes por metrica en 5.6

---

PENDIENTE CRITICO:
- Lista de referencias completa para resolver
  puntos 11, 12 y 13
- Correr experimento final con 15 prompts y
  10 repeticiones (1200 mediciones)
- Redactar seccion 7 amenazas a la validez
  en el documento de tesis

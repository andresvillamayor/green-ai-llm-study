# Redaccion de la Tesis — GREEN-IA
Fragmentos listos para copiar en el documento final.
Ultima actualizacion: junio 2026

---

## TITULO Y SUBTITULO

Titulo:
GREEN-IA: Medicion de Consumo Energetico en LLMs
mediante Cuantizacion

Subtitulo:
Evaluacion experimental del consumo energetico y
emisiones estimadas en inferencia local de modelos
de lenguaje abiertos

---

## RESUMEN DEL EXPERIMENTO

Se disenio y ejecuto un experimento factorial 2x2x2
para evaluar el consumo energetico durante la inferencia
local de modelos de lenguaje de gran escala (LLM).
Se compararon dos modelos (Llama-2-7B y Qwen2.5-7B),
dos niveles de cuantizacion (Q4_K_M y Q8_0) y dos
modos de ejecucion (CPU y GPU Metal) sobre hardware
Apple Silicon M4 con 16 GB de RAM unificada.
Se realizaron 1200 mediciones (15 prompts x 8
configuraciones x 10 repeticiones) utilizando
CodeCarbon como herramienta de estimacion energetica.
Las emisiones operacionales de CO2eq se estimaron
usando el factor de emision de Paraguay
(26 gCO2eq/kWh, Electricity Maps 2026).

---

## OBJETIVO

Comparar el consumo energetico estimado y las emisiones
operacionales de CO2eq entre diferentes niveles de
cuantizacion (Q4 vs Q8) en modelos LLM ejecutados
localmente en CPU y GPU sobre Apple Silicon M4,
bajo condiciones experimentales controladas y
reproducibles.

---

## METODOLOGIA

### Herramientas
- Motor de inferencia: llama-cpp-python 0.3.21
  con backend Metal (GGML_METAL=on)
- Medicion energetica: CodeCarbon
- Hardware: Apple Mac Mini M4, 16 GB RAM,
  macOS 15.6.1, Python 3.13.5

### Parametros de inferencia
Todos los modelos usaron los mismos parametros
para garantizar comparabilidad:
n_ctx=1024, max_tokens=256, temperature=0.7,
top_p=0.9, seed=42, n_threads=8, n_batch=512

### Benchmark
Se utilizo un conjunto controlado de 15 prompts
distribuidos en 9 categorias basadas en MT-Bench
(Zheng et al., NeurIPS 2023): razonamiento logico,
matematicas, escritura, extraccion de informacion,
generacion de codigo, STEM, humanidades, traduccion
y prompt extenso. Todos los prompts se formularon
en ingles para mantener constante el idioma de
entrada. Las conclusiones se limitan al comportamiento
energetico bajo estas categorias especificas.

### Modelos
Llama-2-7B (version base, Meta AI 2023)
archivo: Q4_K_M.gguf y llama-2-7b.Q8_0.gguf
fuente: TheBloke/Llama-2-7B-GGUF (HuggingFace)
tipo de cuantizacion: PTQ (Post-Training Quantization)

Qwen2.5-7B (version instruct, Alibaba 2024)
archivo: Qwen2.5-7B-Instruct-Q4_K_M.gguf
         Qwen2.5-7B-Instruct-Q8_0.gguf
fuente: bartowski/Qwen2.5-7B-Instruct-GGUF
tipo de cuantizacion: PTQ (Post-Training Quantization)

---

## RESULTADOS

### Pruebas estadisticas
Las pruebas Mann-Whitney U (bilateral, alfa=0.05)
confirman diferencias estadisticamente significativas
entre Q4 y Q8 en consumo energetico y emisiones
CO2eq para ambos modelos (p<0.001). En velocidad
de inferencia, la diferencia es significativa para
Llama-2-7B (p<0.001) pero no para Qwen2.5-7B
(p=0.61).

Para Qwen2.5-7B, Q4 mostro menor consumo energetico
(mediana 48.1 mWh vs 63.4 mWh, p<0.001) pero la
diferencia en velocidad no fue estadisticamente
significativa (p=0.61).

### CPU vs GPU
Se observaron diferencias en throughput entre CPU
y GPU Metal que varian segun el modelo y nivel de
cuantizacion. Estas diferencias son especificas del
entorno evaluado y no son generalizables a otras
arquitecturas. La atribucion energetica de GPU en
Apple Silicon debe interpretarse con cautela debido
a las limitaciones de CodeCarbon en este hardware.

### Outliers
Se identificaron 68 outliers de 1200 mediciones
(5.7%) mediante el criterio Tukey IQR x3.
Se detectaron dos tipos distintos:

Tipo A — Outliers lentos (throttling):
Tiempos 10-21x la media. Causa compatible con
throttling termico o ejecucion en E-cores segun
registros del sistema operativo. Potencia CPU
cercana a 0.03W durante estos eventos.

Tipo B — Outliers rapidos (EOS temprano):
Tiempos 0.1-0.3x la media. Causa: el modelo
alcanzo el token de fin de secuencia antes de
max_tokens=256. Tokens generados: 31-34 en lugar
de 256. No representan errores del experimento.

El impacto de los outliers en las conclusiones
energeticas es bajo: delta% de energia entre
conjunto completo y limpio es menor al 4% en
6 de 8 configuraciones.

### Emisiones CO2eq
Las emisiones operacionales de CO2eq estimadas
son bajas debido a la intensidad carbonica de
Paraguay (26 gCO2eq/kWh, Electricity Maps 2026,
matriz 100% hidroelectrica). Los valores se
reportan en CO2eq por 1000 tokens para facilitar
comparacion con otros estudios.

---

## APORTE CIENTIFICO

Aporte 1 — Metodologico:
Se disenio y ejecuto un protocolo experimental
reproducible para medir consumo energetico en
inferencia LLM local, documentando todos los
parametros necesarios para su replicacion.

Aporte 2 — Empirico:
Se obtuvieron 1200 mediciones sobre hardware Apple
Silicon M4 comparando dos modelos, dos cuantizaciones
y dos modos de ejecucion, con estadistica formal
incluyendo IC 95% y pruebas Mann-Whitney U.

Aporte 3 — Contextual:
Se integro la intensidad carbonica de Paraguay
(26 gCO2eq/kWh) para estimar emisiones operacionales
de CO2eq en un contexto energetico de baja huella
carbono, contribuyendo a la linea de investigacion
Green AI en America Latina.

---

## LIMITACIONES

1. Hardware de consumo general con 16 GB RAM limita
   la ejecucion de modelos Q8 sin riesgo de throttling.
2. CodeCarbon no accede directamente al consumo GPU
   en Apple Silicon. Las metricas de energia GPU son
   estimaciones indirectas.
3. El conjunto de 15 prompts no representa la totalidad
   de tareas de inferencia LLM.
4. Los resultados no se generalizan a otros modelos,
   hardware, sistemas operativos o cargas de trabajo.
5. La temperatura del sistema no fue capturada
   directamente por requerir sudo powermetrics.

---

## AMENAZAS A LA VALIDEZ

(ver docs/correcciones_mesa_tesis.md seccion 7)

---

## CONCLUSIONES

En el entorno evaluado, la cuantizacion Q4 mostro
menor consumo energetico estimado que Q8 de forma
estadisticamente significativa para ambos modelos
(p<0.001). La ventaja en velocidad de Q4 fue
significativa para Llama-2-7B pero no para
Qwen2.5-7B, lo que indica que el impacto de la
cuantizacion en velocidad es dependiente del modelo.

El experimento demuestra la factibilidad de realizar
mediciones controladas de inferencia LLM en hardware
de consumo, integrando metricas de rendimiento,
energia y emisiones estimadas bajo un protocolo
reproducible. En el contexto energetico de Paraguay
(100% hidroelectrica), las emisiones absolutas son
bajas, aunque el patron de diferencias entre
configuraciones es relevante para decisiones de
despliegue en entornos con mayor intensidad carbonica.

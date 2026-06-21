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

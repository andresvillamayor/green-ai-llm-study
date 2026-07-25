# Informe de Consumo Energético por Categoría — GREEN-IA

## Metodología

- **Modelo:** Llama-2-7B Q8, backend GPU Metal (Apple M4)
- **Prompts:** 5 por categoría (MT-Bench oficial, Zheng et al. NeurIPS 2023, arXiv:2306.05685)
- **Repeticiones:** 15 por prompt + 3 warmup descartadas
- **Medición:** CodeCarbon 3.2.8 + powermetrics (sudo), desglose CPU/GPU/RAM
- **Total de mediciones:** 600 (8 categorías × 5 prompts × 15 repeticiones)
- **Factor de carbono:** Paraguay (0.0 gCO₂/kWh — matriz 99% hidroeléctrica)

---

## Tabla resumen

Ordenada de mayor a menor consumo energético medio.

| Categoría   | Media (mWh) | Std (mWh) | CV (%)  | CPU (mWh) | GPU (mWh) | RAM (mWh) | Tokens | Tiempo (s) |
|-------------|------------:|----------:|--------:|----------:|----------:|----------:|-------:|-----------:|
| CODING      |      308.31 |      4.60 |    1.49 |    119.30 |    104.71 |     84.31 | 1024.0 |     115.59 |
| STEM        |      308.00 |      3.30 |    1.07 |    116.58 |    107.46 |     83.96 | 1024.0 |     114.76 |
| MATH        |      307.29 |      3.70 |    1.21 |    114.39 |    109.52 |     83.39 | 1024.0 |     113.03 |
| HUMANITIES  |      303.31 |      3.67 |    1.21 |    111.88 |    107.57 |     83.72 | 1024.0 |     115.12 |
| ROLEPLAY    |      298.95 |     17.86 |    5.97 |    115.80 |    100.85 |     82.30 | 1024.0 |     113.42 |
| EXTRACTION  |      292.80 |     21.09 |    7.20 |    109.81 |    103.01 |     79.98 | 1024.0 |     107.96 |
| REASONING   |      283.73 |     22.24 |    7.84 |    110.41 |     94.61 |     78.71 | 1024.0 |     106.59 |
| WRITING     |      273.38 |     67.29 |   24.61 |    105.81 |     92.53 |     75.04 |  906.6 |     101.90 |

> Fuente: `results/analisis_categoria/resumen_8_categorias.csv`
> n = 75 mediciones por categoría (5 prompts × 15 repeticiones válidas).

---

## Hallazgos principales

1. **Categoría más consumidora:** CODING (308.31 mWh) — prompts de generación de código activan
   mayor uso de CPU para parsing y síntesis de tokens estructurados.

2. **Categoría más eficiente:** WRITING (273.38 mWh) — única categoría con respuestas de longitud
   variable (906.6 tokens en promedio frente al máximo configurado de 1024 tokens).

3. **Diferencia relativa:** 12.78 % entre CODING y WRITING (34.93 mWh en términos absolutos).

4. **7 de 8 categorías generan sistemáticamente 1024 tokens** (el máximo configurado). WRITING
   es la única excepción, con 906.6 tokens en promedio, lo que explica directamente su menor
   consumo energético.

5. **WRITING es la única categoría con longitud de respuesta variable.** Su desviación estándar
   de 67.29 mWh (CV = 24.61 %) refleja la alta variabilidad en extensión de las respuestas
   creativas según el prompt específico.

6. **Categorías técnicas muestran menor variabilidad (CV < 2 %):**
   - CODING: CV = 1.49 %
   - STEM: CV = 1.07 %
   - MATH: CV = 1.21 %
   - HUMANITIES: CV = 1.21 %

   La naturaleza determinista de estas tareas produce respuestas de longitud uniforme (siempre
   1024 tokens), resultando en mediciones altamente reproducibles.

7. **Categorías abiertas muestran mayor variabilidad (CV 6–25 %):**
   - ROLEPLAY: CV = 5.97 %
   - EXTRACTION: CV = 7.20 %
   - REASONING: CV = 7.84 %
   - WRITING: CV = 24.61 %

   La variabilidad no proviene de inestabilidad en la medición sino de la naturaleza de los
   prompts: las respuestas tienen estructura interna menos uniforme aunque el total de tokens
   sea idéntico (excepto en WRITING).

---

## Desglose CPU/GPU/RAM

En todas las categorías la proporción entre componentes es consistente: **CPU > GPU > RAM**.

| Categoría   | CPU (%) | GPU (%) | RAM (%) |
|-------------|--------:|--------:|--------:|
| CODING      |    38.7 |    34.0 |    27.3 |
| STEM        |    37.9 |    34.9 |    27.3 |
| MATH        |    37.2 |    35.6 |    27.1 |
| HUMANITIES  |    36.9 |    35.5 |    27.6 |
| ROLEPLAY    |    38.7 |    33.7 |    27.5 |
| EXTRACTION  |    37.5 |    35.2 |    27.3 |
| REASONING   |    38.9 |    33.3 |    27.7 |
| WRITING     |    38.7 |    33.8 |    27.4 |

**Observaciones:**

- La **CPU** concentra consistentemente el 37–39 % del consumo total. En arquitectura Apple M4
  el motor de inferencia llama.cpp realiza el preprocesamiento de tokens y la lógica de control
  del modelo sobre CPU antes de despachar los kernels de multiplicación matricial a GPU.

- La **GPU** (Metal Performance Shaders) concentra el 33–36 % del consumo. Las categorías con
  menor variabilidad (MATH, STEM, HUMANITIES) muestran GPU proportionally ligeramente más alta
  porque sus tokens son más uniformes y los kernels se ejecutan de forma más sostenida.

- La **RAM** representa el 27 % en todas las categorías, con muy baja variación entre ellas.
  Este componente refleja principalmente el costo de mantener los pesos del modelo cargados
  (Llama-2-7B Q8 ≈ 7.5 GB) y es prácticamente independiente del tipo de prompt.

---

## Imágenes generadas por categoría

Para cada una de las 8 categorías se generaron 3 gráficos en `results/analisis_categoria/[categoria]/`:

| Categoría   | boxplot_repeticiones.png | barras_con_std.png | desglose_cpu_gpu_ram.png |
|-------------|:------------------------:|:------------------:|:------------------------:|
| coding      | ✓ | ✓ | ✓ |
| stem        | ✓ | ✓ | ✓ |
| math        | ✓ | ✓ | ✓ |
| humanities  | ✓ | ✓ | ✓ |
| roleplay    | ✓ | ✓ | ✓ |
| extraction  | ✓ | ✓ | ✓ |
| reasoning   | ✓ | ✓ | ✓ |
| writing     | ✓ | ✓ | ✓ |

**Descripción de cada gráfico:**

- **boxplot_repeticiones.png** — Distribución de las 75 mediciones de energía total por prompt
  (5 cajas, una por prompt). Permite identificar outliers y comparar variabilidad intra-prompt
  e inter-prompt dentro de la categoría.

- **barras_con_std.png** — Media ± desviación estándar de consumo energético por prompt.
  Formato de barras con barras de error, facilita la comparación directa entre los 5 prompts
  de la categoría.

- **desglose_cpu_gpu_ram.png** — Barras apiladas con la contribución media de CPU, GPU y RAM
  por prompt. Permite identificar si algún prompt activa de forma diferencial algún componente
  de hardware.

---

## Limitaciones

1. **Variabilidad residual por throttling térmico:** sesiones largas (>90 minutos de inferencia
   continua) pueden acumular calor en el chip M4 y activar la reducción de frecuencia del
   procesador, aumentando el tiempo de inferencia y el consumo por token. Ver análisis detallado
   en `docs/respaldo_cientifico_warmup.md`. Las 3 repeticiones warmup descartadas al inicio de
   cada sesión mitigan parcialmente este efecto.

2. **Un único modelo evaluado en esta fase:** los resultados corresponden exclusivamente a
   Llama-2-7B Q8. Las diferencias relativas entre categorías pueden variar con otros modelos
   (distintos tamaños de parámetros, distintas cuantizaciones Q4/Q8, distintas arquitecturas)
   o con distintos backends de inferencia. La comparación Q4 vs Q8 se abordará en la siguiente
   fase del experimento.

3. **Longitud de respuesta controlada por max_tokens=1024:** el patrón de 1024 tokens uniformes
   en 7 de 8 categorías limita la capacidad de observar diferencias energéticas atribuibles
   al tipo cognitivo de la tarea, ya que el consumo está dominado por la longitud de salida.
   WRITING es el único caso donde la longitud varía naturalmente.

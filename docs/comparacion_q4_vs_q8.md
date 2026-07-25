# Comparación Q4 vs Q8 — Llama-2-7B — Proyecto GREEN-IA

## Metodología

Mismo modelo (Llama-2-7B), misma configuración (GPU Metal, n_ctx=4096, max_tokens=1024, temperature=0.0, seed=42), mismas 8 categorías MT-Bench, mismo protocolo (warmup 3 repeticiones, 15 repeticiones oficiales, `llm.reset()` entre inferencias, medición CodeCarbon 3.2.8 + powermetrics con sudo). Única variable: cuantización (Q4_K_M vs Q8_0).

- **Mediciones por celda**: 75 (5 prompts × 15 repeticiones)
- **Tokens generados**: 1024 (máximo, constante en todas las categorías excepto writing)
- **Hardware**: Apple M4, GPU Metal integrada

---

## Tabla comparativa

Ordenada de mayor a menor reducción de consumo energético.

| Categoría   | Q8 (mWh) | Q4 (mWh) | Reducción (%) |
|-------------|:--------:|:--------:|:-------------:|
| Humanities  | 303.309  | 196.545  | **35.20%**    |
| Coding      | 308.314  | 217.461  | 29.47%        |
| Math        | 307.290  | 217.685  | 29.16%        |
| STEM        | 307.996  | 218.348  | 29.11%        |
| Roleplay    | 298.947  | 218.862  | 26.79%        |
| Extraction  | 292.800  | 223.974  | 23.51%        |
| Reasoning   | 283.731  | 219.212  | 22.74%        |
| Writing     | 273.381  | 211.792  | **22.53%**    |

**Reducción promedio: 27.31%** (rango: 22.53% – 35.20%)

---

## Hallazgos principales

1. **Q4 reduce el consumo energético entre 22.5% y 35.2%** dependiendo de la categoría. La reducción es consistente en las 8 categorías: ninguna registra un aumento en Q4.

2. **Reducción promedio: 27.31%** calculada sobre las 8 categorías. Las categorías de procesamiento simbólico/estructurado (coding, math, stem) agrupan cerca del 29%, mientras que las de lenguaje natural abierto (writing, reasoning) se sitúan en el extremo inferior (~22–23%).

3. **Humanities muestra el comportamiento más atípico**: pasa de ser la 4ª más consumidora en Q8 (303.31 mWh) a ser la MÁS eficiente en Q4 (196.54 mWh), con la mayor reducción porcentual (35.20%). Este cambio va acompañado de un perfil CPU/GPU/RAM radicalmente distinto al resto de categorías en Q4 (ver sección siguiente).

4. **El ranking de categorías por consumo cambia sustancialmente** entre Q4 y Q8, lo que confirma que el efecto de la cuantización no es uniforme entre tipos de tarea:

   | Posición | Q8 (mayor → menor) | Q4 (mayor → menor) |
   |:--------:|---------------------|---------------------|
   | 1        | Coding (308.31)     | Extraction (223.97) |
   | 2        | STEM (308.00)       | Reasoning (219.21)  |
   | 3        | Math (307.29)       | Roleplay (218.86)   |
   | 4        | Humanities (303.31) | STEM (218.35)       |
   | 5        | Roleplay (298.95)   | Math (217.68)       |
   | 6        | Extraction (292.80) | Coding (217.46)     |
   | 7        | Reasoning (283.73)  | Writing (211.79)    |
   | 8        | Writing (273.38)    | Humanities (196.54) |

   Coding cae del puesto 1 al 6; Humanities cae del puesto 4 al 8 (más eficiente). Extraction sube del 6 al 1 (más consumidora en Q4).

5. **La reducción de tiempo de inferencia es consistente y mayor que la de energía**: Q4 requiere entre 33% y 40% menos tiempo por inferencia en todas las categorías, lo que contribuye directamente a la reducción energética además de los efectos de la cuantización sobre el modelo.

---

## Desglose CPU/GPU/RAM comparado

### Patrón general Q8

En Q8, el perfil es homogéneo entre categorías: CPU domina (~38%), GPU segundo (~34–36%), RAM tercero (~27%).

| Categoría   | CPU Q8 (mWh) | GPU Q8 (mWh) | RAM Q8 (mWh) |
|-------------|:------------:|:------------:|:------------:|
| Coding      | 119.3        | 104.7        | 84.3         |
| STEM        | 116.6        | 107.5        | 84.0         |
| Math        | 114.4        | 109.5        | 83.4         |
| Humanities  | 111.9        | 107.6        | 83.7         |
| Roleplay    | 115.8        | 100.8        | 82.3         |
| Extraction  | 109.8        | 103.0        | 80.0         |
| Reasoning   | 110.4        | 94.6         | 78.7         |
| Writing     | 105.8        | 92.5         | 75.0         |

### Patrón general Q4

En Q4, el patrón cambia: GPU pasa a ser el componente dominante en la mayoría de categorías, y CPU y RAM se reducen más que GPU. Esto sugiere que Q4_K_M desplaza carga computacional hacia el motor de GPU Metal al procesar pesos de menor precisión.

| Categoría   | CPU Q4 (mWh) | GPU Q4 (mWh) | RAM Q4 (mWh) |
|-------------|:------------:|:------------:|:------------:|
| Extraction  | 70.3         | 100.8        | 52.9         |
| Reasoning   | 73.0         | 92.0         | 54.2         |
| Roleplay    | 71.3         | 94.2         | 53.4         |
| STEM        | 73.9         | 89.4         | 55.0         |
| Math        | 71.5         | 92.6         | 53.6         |
| Coding      | 71.8         | 92.0         | 53.6         |
| Writing     | 69.6         | 90.0         | 52.1         |
| Humanities  | **44.1**     | **119.2**    | **33.2**     |

### Anomalía en Humanities (Q4)

Humanities en Q4 es un caso aislado. Mientras el resto de categorías en Q4 presenta GPU entre 89–101 mWh y CPU entre 69–74 mWh, Humanities invierte la relación de forma extrema:

- CPU cae a **44.1 mWh** (la más baja de todo el experimento, ~40% menos que el promedio Q4)
- GPU sube a **119.2 mWh** (la más alta de todo Q4, comparable a niveles Q8)
- RAM cae a **33.2 mWh** (la más baja de todo el experimento)

Este perfil anómalo explica en parte la mayor reducción total (35.20%): la tarea de humanidades en este conjunto de prompts puede requerir menor presión sobre la memoria de trabajo (RAM/CPU) una vez que el modelo procesa en baja precisión, concentrando carga en la GPU Metal. La causa exacta requiere análisis adicional (p.ej., longitud de contexto activo, patrón de atención, o comportamiento específico de llama.cpp con Q4_K_M en Metal para prompts de tipo abierto-humanístico).

---

## Tiempos de inferencia comparados

La reducción de tiempo en Q4 supera consistentemente la reducción energética, lo que indica que parte del ahorro energético proviene de la menor duración de la inferencia y no solo de la menor potencia instantánea.

| Categoría   | Tiempo Q8 (s) | Tiempo Q4 (s) | Reducción tiempo |
|-------------|:-------------:|:-------------:|:----------------:|
| Coding      | 115.6         | 69.2          | 40.1%            |
| Humanities  | 115.1         | 69.1          | 40.0%            |
| STEM        | 114.8         | 71.4          | 37.8%            |
| Math        | 113.0         | 69.1          | 38.9%            |
| Roleplay    | 113.4         | 69.8          | 38.5%            |
| Extraction  | 108.0         | 72.5          | 32.9%            |
| Reasoning   | 106.6         | 69.4          | 34.9%            |
| Writing     | 101.9         | 66.7          | 34.5%            |

---

## Conclusión preliminar

La cuantización Q4_K_M ofrece una reducción sustancial y consistente del consumo energético en las 8 categorías evaluadas de MT-Bench para Llama-2-7B en Apple M4:

- **Reducción energética**: 22.5% – 35.2% (promedio 27.31%)
- **Reducción de tiempo de inferencia**: 32.9% – 40.1%
- **Sin categorías con incremento**: la reducción es unidireccional en todas las tareas

El impacto en calidad de las respuestas está **pendiente de evaluación** mediante juez LLM en fase posterior del proyecto. Los resultados actuales corresponden exclusivamente a la dimensión energética bajo condiciones controladas y reproducibles.

La categoría Humanities representa el punto de mayor reducción (35.20%) y el perfil de componentes más atípico en Q4, constituyendo un hallazgo de interés para la discusión sobre la interacción entre tipo de tarea, cuantización y arquitectura de hardware unificada (Apple Silicon / Metal).

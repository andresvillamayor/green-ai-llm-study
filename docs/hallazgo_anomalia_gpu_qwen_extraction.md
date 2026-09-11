# Hallazgo: Anomalía GPU en Qwen2.5-7B, categoría Extraction

**Fecha de detección:** 2026-08-28  
**Modelo afectado:** Qwen2.5-7B-Instruct, categoría extraction (Q4\_K\_M y Q8\_0)  
**Estado:** Documentado como limitación metodológica — causa mecanicista no confirmada  
**Fuentes de datos:**
- `results/analisis_categoria/qwen2.5-7b_q4/extraction/resultados_completos.csv`
- `results/analisis_categoria/qwen2.5-7b_q8/extraction/resultados_completos.csv`

---

## 1. Hallazgo inicial

En la comparación de consumo energético GPU entre Qwen2.5-7B Q4\_K\_M y Q8\_0, la
categoría **extraction es la única de 8 donde Q4 consume MÁS energía GPU total que
Q8**, invirtiendo el patrón observado en todas las demás categorías.

El patrón general a través de las 7 categorías restantes es:

> Q4 dibuja mayor potencia instantánea que Q8 (10–31 % más mW),  
> pero termina antes (mayor velocidad de tokens/segundo),  
> resultando en 11–37 % **menos** energía total por inferencia en Q4.

En extraction ese balance se rompe: la mayor potencia de Q4 no queda compensada
por la ganancia de velocidad, y el resultado neto es mayor energía GPU en Q4.

---

## 2. Análisis por prompt

La anomalía no es uniforme dentro de la categoría. Los 5 prompts presentan
comportamientos claramente diferenciados:

| Prompt | question\_id | Tokens (Q4/Q8) | GPU Q4 (mWh) | GPU Q8 (mWh) | Delta (mWh) | Patrón |
|--------|-------------|---------------|-------------|-------------|------------|--------|
| P1 | 131 | 13 / 13 | 0.07 | 0.00 | +0.07 | Ambiguo (GPU casi inactiva) |
| P2 | 132 | 8 / 8 | 0.11 | 0.00 | +0.11 | Ambiguo (GPU casi inactiva) |
| P3 | 133 | 81 / 81 | 23.71 | 21.10 | **+2.61** | **Anómalo (Q4 > Q8)** |
| P4 | 134 | 34 / 34 | 0.18 | 1.82 | **−1.64** | **Normal (Q4 < Q8)** |
| P5 | 135 | 103 / 93 | 24.52 | 21.34 | **+3.18** | **Anómalo (Q4 > Q8)** |

La anomalía se concentra en **P3 y P5**. P4 (mismo modelo, misma categoría, 34 tokens)
muestra el patrón **inverso completo**: Q4 consume drásticamente menos energía GPU
que Q8, descartando que el fenómeno sea un efecto a nivel de categoría.

---

## 3. Análisis de potencia instantánea vs velocidad

La causa energética de la anomalía se entiende descomponiendo la energía en sus
dos factores: potencia instantánea (mW) × duración (s).

### P3 (question\_id 133, 81 tokens en ambas cuantizaciones)

| Métrica | Q4\_K\_M | Q8\_0 | Ratio Q4/Q8 |
|---------|---------|------|------------|
| GPU energy (mWh) | 23.71 | 21.10 | 1.124 — Q4 usa **12.4 % más** |
| Inference time (s) | 6.34 | 8.16 | Q4 es **1.29× más rápido** |
| Potencia GPU inst. (mW) | **13 455** | **9 315** | **1.45× más watts en Q4** |
| Energía por token (mWh/tok) | 0.293 | 0.261 | Q4: 12.4 % más caro/token |

### P5 (question\_id 135, 103 tokens Q4 / 93 tokens Q8)

| Métrica | Q4\_K\_M | Q8\_0 | Ratio Q4/Q8 |
|---------|---------|------|------------|
| GPU energy (mWh) | 24.52 | 21.34 | 1.149 — Q4 usa **14.9 % más** |
| Inference time (s) | 6.58 | 8.21 | Q4 es **1.25× más rápido** |
| Potencia GPU inst. (mW) | **13 417** | **9 364** | **1.43× más watts en Q4** |
| Energía por token (mWh/tok) | 0.238 | 0.230 | Q4: 3.7 % más caro/token |

### P4 (question\_id 134, 34 tokens — patrón inverso)

| Métrica | Q4\_K\_M | Q8\_0 | Ratio Q4/Q8 |
|---------|---------|------|------------|
| GPU energy (mWh) | 0.18 | 1.82 | 0.099 — Q4 usa **90 % menos** |
| Inference time (s) | 2.98 | 3.66 | Q4 es **1.23× más rápido** |
| Potencia GPU inst. (mW) | 218 | 1 790 | **0.12× — Q4 dibuja mucho menos** |
| Energía por token (mWh/tok) | 0.0053 | 0.0535 | Q4: 90 % más eficiente/token |

**Interpretación del balance P3/P5:** Q4 dibuja ~1.44× más watts instantáneos que Q8,
pero solo termina ~1.29× (P3) / 1.25× (P5) más rápido. La ventaja de velocidad no
alcanza a compensar el exceso de potencia, dejando un saldo energético positivo para Q4.
En P4, Q4 dibuja ~8× *menos* potencia que Q8, lo que combinado con la ventaja de
velocidad resulta en una eficiencia drásticamente mayor de Q4.

---

## 4. Hipótesis descartada: efecto de longitud de respuesta

Una hipótesis natural sería que la anomalía sea un efecto de la cantidad de tokens
generados: prompts que producen respuestas más largas tienden a mostrar el fenómeno.
Esta hipótesis fue descartada mediante análisis de correlación.

**Correlación entre tokens generados y delta GPU (Q4 − Q8), calculada sobre todas
las categorías y repeticiones:**

> **r = −0.677, p < 10⁻⁸¹**

La correlación es **negativa**: a mayor cantidad de tokens generados en el conjunto
global de prompts, menor es el delta Q4−Q8 (es decir, más ventajoso resulta Q4 en
energía). Este resultado es el opuesto de lo que predice la hipótesis de longitud.

La evidencia dentro de extraction refuerza el descarte: P3 y P5 producen respuestas
largas (81 y 103 tokens) y muestran la anomalía, pero **otras categorías con
respuestas igualmente largas no la muestran**. Y P4 (34 tokens) dentro de la misma
categoría extraction exhibe el patrón normal.

---

## 5. Conclusión y clasificación del hallazgo

La anomalía GPU en Qwen extraction es un fenómeno de **variabilidad inter-prompt**
específico del tipo de tarea que formulan P3 y P5, no un patrón sistemático de la
cuantización Q4 ni de la categoría extraction en su conjunto.

**Caracterización:**

- La causa mecanicista exacta **no está confirmada**. Los prompts P3 y P5 de extraction
  producen un perfil de activación GPU en Q4 que consume ~1.44× más potencia
  instantánea que Q8 (versus ~1.1–1.3× en otras categorías), sin que la ganancia de
  velocidad compense.
- No es un artefacto de longitud de tokens (descartado con r = −0.677, p < 10⁻⁸¹).
- No es un efecto de categoría (P4 en la misma categoría muestra el patrón opuesto).
- No es un efecto de modelo (el patrón general de Qwen sigue siendo Q4 más eficiente
  en 7/8 categorías; extraction es la excepción y se circunscribe a 2/5 prompts).

**Impacto en el argumento de la tesis:**

La conclusión general de que Q4 reduce el consumo energético GPU en Qwen2.5-7B
sigue siendo válida para 7 de 8 categorías. Para extraction, la afirmación debe
matizarse: el ahorro de Q4 no se confirma en P3 y P5, y la categoría como conjunto
no favorece a Q4 en energía GPU. El texto de la tesis debe reportar esta excepción
explícitamente en lugar de generalizar.

---

## 6. Redacción sugerida para la tesis

> "En la categoría extraction, Qwen2.5-7B Q4\_K\_M no presenta la reducción de energía
> GPU observada en las restantes siete categorías. El análisis por prompt revela que
> la anomalía se concentra en los prompts P3 y P5 (81 y 103 tokens respectivamente),
> donde Q4 dibuja aproximadamente 1.44× más potencia instantánea que Q8 sin
> compensarlo con la ventaja de velocidad (1.29× y 1.25× más rápido, respectivamente).
> La hipótesis de que el fenómeno se explica por la longitud de respuesta fue
> descartada mediante análisis de correlación (r = −0.677, p < 10⁻⁸¹), que muestra
> una relación negativa entre tokens generados y delta de energía Q4−Q8 a nivel global.
> El prompt P4 de la misma categoría (34 tokens) exhibe el patrón inverso, lo que
> descarta un efecto de categoría. La causa mecanicista precisa no fue determinada en
> este estudio y se documenta como limitación."

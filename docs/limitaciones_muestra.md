# Limitaciones metodológicas — GREEN-IA

**Proyecto:** Medición de consumo energético en inferencia LLM bajo cuantización Q4 vs Q8  
**Fecha de compilación:** 2026-08-28  
**Uso previsto:** sección de Limitaciones en `06_discusion.tex`; versión para compañero de tesis

Todas las limitaciones registradas aquí provienen de evidencia documentada en archivos
del proyecto. No se infiere ninguna limitación que no esté respaldada por datos
empíricos propios o por decisiones de diseño explícitas en el código o en los docs.

---

## L1 — Warmup térmico insuficiente en Llama-2-7B

### Qué es la limitación

El protocolo original aplica un warmup de duración fija (3 repeticiones) antes de
iniciar las 15 repeticiones de medición. Para Llama-2-7B, este warmup resultó
insuficiente en 6 combinaciones modelo × cuantización × categoría × prompt,
produciendo un decaimiento monotónico de `gpu_energy_mwh` durante las primeras 5–6
repeticiones de medición. El efecto físico subyacente es thermal throttling/cambio
de P-state en el SoC M4: el chip inicia en frío, opera a mayor potencia instantánea,
y la reduce gradualmente al estabilizar su régimen térmico.

### Evidencia dentro del proyecto

Análisis sistemático de 160 combinaciones (4 configs × 8 categorías × 5 prompts).
Criterios de confirmación: caída > 15 % **y** > 5 mWh entre rep1 y media reps 10–15.

| Modelo | Cuant | Categoría | Prompt | Caída abs. (mWh) | Caída % | r\_trend |
|--------|-------|-----------|--------|-----------------|---------|---------|
| llama-2-7b | Q4 | roleplay | P1 | –39.95 | 42 % | –0.58 |
| llama-2-7b | Q8 | reasoning | P2 | –23.46 | 27 % | –0.61 |
| llama-2-7b | Q4 | humanities | P1 | –30.48 | 26 % | –0.95 |
| llama-2-7b | Q8 | roleplay | P3 | –21.36 | 24 % | –0.79 |
| llama-2-7b | Q4 | extraction | P1 | –18.76 | 18 % | –0.96 |
| llama-2-7b | Q8 | extraction | P3 | –16.75 | 17 % | –0.68 |

Ningún caso confirmado en Qwen2.5-7B-Instruct (sus respuestas son más cortas y el
chip no alcanza el mismo régimen térmico).

Fuentes: `docs/hallazgo_warmup_termico_llama.md`, `docs/diseno_warmup_adaptativo.md`.  
Referencia externa: Taneja, Hritvik et al. arXiv:2305.12784 (validación del
mecanismo en Apple Silicon).

### Impacto en la interpretación de resultados

Los promedios de `gpu_energy_mwh` de los 6 prompts afectados **sobreestiman
ligeramente** la energía GPU de Llama en estado estable. Los valores registrados
son fieles a lo que CodeCarbon midió; el problema es que las primeras repeticiones
no representan el estado térmico estabilizado del sistema. El efecto queda acotado a
6 de 80 prompts de Llama (7.5 %) y no afecta a Qwen.

### Plan de mitigación definido

Warmup adaptativo basado en coeficiente de variación (CV < 2 % sobre ventana de 3
repeticiones consecutivas, mín 3 reps, máx 10). Diseño completo en
`docs/diseno_warmup_adaptativo.md`. Estado: diseño finalizado, implementación
pendiente en `scripts/analisis_categoria_refactor.py`. El dataset original no se
modifica (principio de inmutabilidad de datos); la mitigación se aplica en una
re-corrida acotada de los 6 casos confirmados (~105 minutos por cuantización).

---

## L2 — Estimación energética por TDP en Apple M4 (no medición directa)

### Qué es la limitación

En macOS con Apple M4, CodeCarbon no puede leer el consumo de hardware directamente
sin privilegios de root. En su lugar, el experimento configura potencias fijas:

```python
EmissionsTracker(
    force_cpu_power=20,   # TDP del Apple M4 (fuente: NotebookCheck 2024)
    force_ram_power=3,    # estimado LPDDR5X 16 GB
    allow_multiple_runs=True,
)
```

Esto implica que los componentes CPU y RAM se contabilizan con potencia constante
(20 W y 3 W respectivamente), independientemente de la carga real. La componente GPU
(`gpu_energy_mwh`) se obtiene de `powermetrics` (macOS), que sí lee el hardware del
SoC, pero con una tasa de muestreo de 1 segundo que puede introducir granularidad.

Adicionalmente, existe un bug conocido de CodeCarbon/powermetrics (issue #985) por el
cual `powermetrics` devuelve ocasionalmente `NaN` en la primera lectura de GPU. El
protocolo implementa un reintento automático para estos casos; si el problema persiste,
la repetición se marca como fallida y se excluye del análisis.

### Evidencia dentro del proyecto

`docs/resumen_tecnico_estudiante.md` (sección 7: "Cómo funciona CodeCarbon en Apple
M4 y sus limitaciones"); `docs/tesis_refactor.md` (mención explícita del bug #985 de
CodeCarbon/powermetrics); metadatos del experimento (`tracking_mode = 'constant'`
registrado en cada run).

### Impacto en la interpretación de resultados

1. **Valores absolutos de energía total (Wh):** son estimaciones basadas en TDP, no
   mediciones directas. No pueden compararse con mediciones en plataformas Windows
   donde NVML permite leer la GPU directamente.
2. **Componente GPU (`gpu_energy_mwh`):** sí proviene de `powermetrics` y refleja el
   consumo real del SoC; es la métrica más fiable del experimento en esta plataforma.
3. **Comparaciones Q4 vs Q8 dentro del mismo hardware:** siguen siendo válidas,
   porque ambas cuantizaciones se miden con el mismo método, el mismo TDP fijo, y la
   corrección de baseline resta el consumo en reposo (estimado por el mismo mecanismo).
   Las diferencias relativas observadas reflejan diferencias reales de carga de trabajo.

### Plan de mitigación definido

La corrección de baseline (`baseline_corrected_energy`) está implementada y activa.
El `tracking_mode` se registra en los metadatos de cada run para que los lectores
conozcan el método. El análisis reporta los valores absolutos con la etiqueta
"estimación basada en TDP", no como medición directa. No se prevé cambio en el
método de medición para el dataset actual.

---

## L3 — Anomalía GPU en Qwen2.5-7B, categoría extraction

### Qué es la limitación

En la categoría extraction, Qwen2.5-7B Q4\_K\_M no presenta la reducción de energía
GPU observada en las restantes siete categorías: es la única excepción al patrón
general donde Q4 reduce el consumo energético respecto a Q8. La anomalía se
concentra en los prompts P3 y P5, donde Q4 dibuja ~1.44× más potencia instantánea
que Q8 (13 450 mW vs 9 340 mW), sin compensarlo con la ventaja de velocidad que sí
opera en el resto del dataset (1.29× y 1.25× respectivamente en esos prompts).

### Evidencia dentro del proyecto

Análisis completo en `docs/hallazgo_anomalia_gpu_qwen_extraction.md`.  
Datos fuente: `results/analisis_categoria/qwen2.5-7b_q4/extraction/resultados_completos.csv`
y su equivalente Q8.

Hallazgos clave del análisis:
- P3 (81 tokens, Q4 vs Q8): potencia instantánea 13 455 vs 9 315 mW, ratio 1.445×;
  velocidad ratio 1.286×; saldo energético **+2.61 mWh en Q4**.
- P5 (103/93 tokens, Q4/Q8): potencia 13 417 vs 9 364 mW, ratio 1.433×;
  velocidad 1.247×; saldo **+3.18 mWh en Q4**.
- P4 (34 tokens, misma categoría): patrón **inverso completo** — Q4 usa 90 % menos
  energía GPU que Q8, descartando efecto de categoría.
- **Hipótesis longitud de respuesta descartada:** correlación tokens generados
  vs delta GPU (Q4−Q8) = r = −0.677, p < 10⁻⁸¹ (relación negativa; más tokens
  correlaciona con mayor ventaja de Q4, no con la anomalía).

### Impacto en la interpretación de resultados

La conclusión de que Q4 reduce el consumo energético de Qwen sigue siendo válida
para 7 de 8 categorías. Para extraction, la afirmación no puede generalizarse: el
ahorro de Q4 no se confirma en P3 y P5, y la categoría como conjunto no favorece
a Q4 en energía GPU. El texto de la tesis debe reportar esta excepción
explícitamente.

### Plan de mitigación definido

No se requiere corrección de datos — los valores medidos son correctos. El documento
`docs/hallazgo_anomalia_gpu_qwen_extraction.md` incluye redacción sugerida para
matizar el alcance de la conclusión sobre Q4 en la sección de Resultados y en
Limitaciones. La causa mecanicista no fue determinada; se declara como tal.

---

## L4 — Incomparabilidad directa de calidad entre Llama-2-7B (base) y Qwen2.5-7B (instruct)

### Qué es la limitación

MT-Bench fue diseñado para evaluar modelos de chat con ajuste de instrucción
(*instruction-tuned*). Qwen2.5-7B-Instruct es un modelo instruct; Llama-2-7B es un
modelo base sin fine-tuning de instrucciones. Esto tiene dos consecuencias:

1. **En energía:** Llama genera respuestas extremadamente largas en MT-Bench
   (frecuentemente hasta el límite de 1024 tokens, ~69 s/rep), porque un modelo base
   sin instrucción no sabe cuándo terminar. Qwen genera respuestas focalizadas (8–130
   tokens típicamente). Esta diferencia de longitud de salida confunde energía por
   token con energía por conversación.
2. **En calidad:** los puntajes de calidad de Llama y Qwen en MT-Bench no son
   directamente comparables. Un modelo base evaluado en un benchmark de chat instruct
   partirá de una base estructuralmente diferente, independientemente de la
   cuantización.

### Evidencia dentro del proyecto

`docs/resumen_tecnico_estudiante.md` (el puntaje se denomina explícitamente
"Claude-based MT-Bench-style quality score", no "MT-Bench score oficial");
`docs/hallazgo_juez_calidad_math.md` ("la limitación principal es estructural:
Llama-2-7B en su versión base no está optimizado para resolver problemas matemáticos
paso a paso"); `CLAUDE.md` ("llama-2-7b is a base model; qwen2.5-7b is instruct.
Quality scores between them are not directly comparable on MT-Bench").

### Impacto en la interpretación de resultados

La tesis no puede afirmar que "Qwen tiene mejor/peor calidad que Llama" basándose en
los puntajes MT-Bench, porque la comparación no es justa por diseño. La pregunta
legítima es: **dentro de cada modelo, ¿cómo afecta Q4 vs Q8 a la relación
energía/calidad?** Esa pregunta sí es respondible con los datos actuales, porque
compara el mismo modelo contra sí mismo.

### Plan de mitigación definido

El análisis reporta cada modelo por separado. Las conclusiones sobre el trade-off
energía/calidad se formulan siempre en términos de "Q4 vs Q8 dentro de Llama" o
"Q4 vs Q8 dentro de Qwen", nunca como comparación directa entre modelos en calidad.
La diferencia de longitud de respuesta se controla reportando también la métrica
de eficiencia por token (`wh_per_token`).

---

## L5 — Evaluación de calidad incompleta (prueba de concepto, solo Llama·math)

### Qué es la limitación

La evaluación LLM-as-a-Judge se completó únicamente como prueba de concepto sobre
los 5 prompts de la categoría **math** de **Llama-2-7B**, comparando Q4 vs Q8.
Las 7 categorías restantes (extraction, humanities, coding, reasoning, roleplay,
stem, writing), Qwen2.5-7B completo, y las 15 repeticiones por prompt no han sido
evaluadas a la fecha de este documento.

Las limitaciones propias de esta prueba de concepto son:
- Muestra de 5 prompts, 1 categoría de 8.
- Solo 1 repetición por prompt evaluada (no las 15 del experimento de energía).
- Solo Llama-2-7B; Qwen2.5-7B-Instruct sin evaluar.

### Evidencia dentro del proyecto

`docs/hallazgo_juez_calidad_math.md` (sección "Limitaciones de esta prueba de
concepto" y sección "Estado": "Pendiente: extraction, humanities, coding, reasoning,
roleplay, stem, writing").

### Impacto en la interpretación de resultados

Las conclusiones sobre calidad no pueden generalizarse más allá de la categoría math
de Llama-2-7B. En particular, no es posible afirmar a nivel de tesis si la
cuantización Q4 preserva calidad aceptable en categorías con respuestas evaluables
por corrección objetiva (coding, reasoning) o con mayor libertad estilística
(writing, roleplay), ni si el patrón se replica en Qwen. El argumento central sobre
energía (reducción de 22–35 % con Q4 respecto a Q8) sí está completo, pero el
lado de calidad del trade-off es preliminar.

### Plan de mitigación definido

La infraestructura del juez está implementada y validada (`judge_with_claude.py`).
El escalado a las 8 categorías y a Qwen es el paso siguiente del pipeline de
análisis. No requiere re-correr el experimento de energía: los CSVs de
`results/raw/conversation_results.csv` ya contienen las respuestas a evaluar.

---

## L6 — Muestra de un solo dispositivo, sin replicación en otro hardware

### Qué es la limitación

Todos los resultados del experimento provienen de un único dispositivo: Apple Mac Mini
con SoC M4, ejecutando macOS con backend Metal. El segundo hardware objetivo del
diseño original (PC Windows con GPU NVIDIA RTX 4060, backend CUDA) no ejecutó el
experimento completo a la fecha de este documento.

### Evidencia dentro del proyecto

`CLAUDE.md` declara como targets "Apple M4 Mac Mini (Metal backend) and Windows RTX
4060 (CUDA backend)", y `docs/resumen_tecnico_estudiante.md` menciona que "los
valores absolutos de energía en el Apple M4 no son comparables con los de Windows
NVIDIA" — implicando que la comparación entre plataformas era parte del diseño pero
no se realizó. No existe ningún CSV de resultados en `results/` originado en hardware
Windows.

### Impacto en la interpretación de resultados

No es posible afirmar si los patrones observados (magnitud del ahorro de Q4 vs Q8,
efecto de warmup térmico, variabilidad entre repeticiones) son propiedades generales
de los modelos o artefactos específicos del SoC M4 y del backend Metal. Las
conclusiones son válidas para este hardware y generalizables con cautela.

### Plan de mitigación definido

Ninguno documentado dentro del proyecto a esta fecha. La replicación en Windows/CUDA
sería trabajo futuro. El texto de la tesis debe circunscribir explícitamente las
conclusiones al hardware evaluado.

---

## Resumen consolidado

| ID | Limitación | Impacto | Mitigación | Estado |
|----|-----------|---------|-----------|--------|
| L1 | Warmup térmico insuficiente (Llama, 6 prompts) | Sobreestimación leve de GPU energy en 7.5% de prompts de Llama | Warmup adaptativo diseñado | Diseño listo, implementación pendiente |
| L2 | Estimación TDP en lugar de medición directa (Apple M4) | Valores absolutos son estimaciones; comparación relativa Q4/Q8 válida | Corrección baseline + metadatos tracking\_mode | Mitigación activa |
| L3 | Anomalía GPU Qwen·extraction (P3 y P5) | Q4 no reduce energía GPU en esos 2 prompts; conclusión general válida para 7/8 categorías | Declarar excepción en texto de tesis | Causa mecanicista no confirmada |
| L4 | Llama base vs Qwen instruct — incomparabilidad de calidad en MT-Bench | Puntajes de calidad entre modelos no comparables directamente | Reporte separado por modelo | Mitigación activa por diseño |
| L5 | Evaluación de calidad incompleta (solo Llama·math) | Conclusiones de calidad no generalizables a 8 categorías ni a Qwen | Escalado a 8 categorías e infraestructura lista | Escalado pendiente |
| L6 | Un solo dispositivo (Apple M4), sin replicación Windows/CUDA | Resultados no replicados en otro hardware | — | Trabajo futuro |

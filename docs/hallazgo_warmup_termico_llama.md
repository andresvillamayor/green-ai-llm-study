# Hallazgo: Warmup Térmico Insuficiente en Llama-2-7B

**Fecha de detección:** 2026-08-12  
**Dataset afectado:** `results/analisis_categoria_LLAMA_Q4_BACKUP/` y `results/analisis_categoria_LLAMA_Q8_BACKUP/`  
**Estado:** Limitación metodológica documentada — corrección planificada (ver sección final)

---

## Resumen del hallazgo

Al analizar la estabilidad de `gpu_energy_mwh` a lo largo de las 15 repeticiones por prompt, se detectó un patrón de **decaimiento progresivo de energía GPU en las primeras 5–6 repeticiones** en múltiples prompts de Llama-2-7B, tanto en Q4 como en Q8. Este efecto es consistente con un **warmup térmico insuficiente**: el chip M4 inicia la inferencia en un estado de baja temperatura, opera a mayor potencia instantánea, y gradualmente reduce su consumo conforme estabiliza su régimen térmico.

Se realizó un barrido sistemático sobre las 160 combinaciones modelo × cuantización × categoría × prompt, midiendo la diferencia porcentual entre `gpu_energy_mwh` en la repetición 1 y el promedio de las repeticiones 10–15 (consideradas representativas del estado estable). Se aplicaron dos criterios simultáneos para confirmar un caso:

- Caída porcentual > 15 % (rep1 vs. reps 10–15)
- Caída absoluta > 5 mWh (filtra falsos positivos por GPU inactiva con valores cercanos a cero)

**Resultado: 6 casos confirmados, todos en Llama-2-7B. Ninguno en Qwen2.5-7B.**

---

## Tabla de casos confirmados

| Modelo | Cuant | Categoría | Prompt | GPU rep1 (mWh) | GPU estable (mWh) | Caída abs. (mWh) | Caída % | r_trend |
|--------|-------|-----------|--------|---------------|-------------------|-----------------|---------|---------|
| llama-2-7b | Q4 | roleplay | P1 | 134.81 | 94.86 | –39.95 | **42 %** | –0.58 |
| llama-2-7b | Q8 | reasoning | P2 | 109.48 | 86.01 | –23.46 | **27 %** | –0.61 |
| llama-2-7b | Q4 | humanities | P1 | 146.85 | 116.37 | –30.48 | **26 %** | –0.95 |
| llama-2-7b | Q8 | roleplay | P3 | 109.45 | 88.09 | –21.36 | **24 %** | –0.79 |
| llama-2-7b | Q4 | extraction | P1 | 120.94 | 102.18 | –18.76 | **18 %** | –0.96 |
| llama-2-7b | Q8 | extraction | P3 | 114.99 | 98.24 | –16.75 | **17 %** | –0.68 |

`r_trend`: correlación de Pearson entre número de repetición y gpu_energy_mwh (valores negativos indican decaimiento monotónico).

Los dos casos con `r_trend` más negativo (–0.95 y –0.96) confirman un decaimiento casi perfectamente monotónico, lo que respalda una causa física subyacente y descarta ruido aleatorio.

### Casos borderline (8–15 %, >5 mWh): señal leve, no confirmados

Seis casos adicionales de Llama (2 borderline en Q4, 4 en Q8) y un caso de Qwen Q8 (humanities P2, 8.8 %) quedan en zona gris. No se tratan como warmup confirmado pero se registran aquí para referencia.

---

## Explicación propuesta

**¿Por qué Llama y no Qwen?**

Llama-2-7B es un modelo base sin ajuste de instrucción. En el benchmark MT-Bench, genera respuestas muy extensas que frecuentemente alcanzan el límite de 1024 tokens (`completion_tokens = 1024` en la mayoría de sus repeticiones), con tiempos de inferencia de ~69 segundos por repetición. Esto somete al SoC M4 a una carga GPU continua e intensa durante períodos prolongados.

Qwen2.5-7B-Instruct, en contraste, genera respuestas más focalizadas y breves (típicamente 8–130 tokens según la categoría, salvo humanities). Los bursts de GPU son cortos y el chip nunca alcanza el régimen térmico que activa el efecto de throttling/P-state observable en Llama.

**¿Por qué Q4 y Q8 muestran patrones distintos?**

En Q4, el warmup aparece invariablemente en **P1** (primer prompt de la sesión), lo que es consistente con un arranque en frío del chip. En Q8, el efecto aparece en **P2 o P3**, sugiriendo que P1 Q8 calienta el chip lo suficiente para alterar el régimen de potencia que arranca P2 o P3, pero no con suficiente intensidad como para producir el patrón de caída en P1 mismo.

---

## Clasificación metodológica

Este hallazgo se documenta como una **limitación conocida del dataset actual**, no como un error de medición. Los valores registrados son fieles a lo que CodeCarbon midió; el problema es que las primeras repeticiones no representan el estado estable del sistema bajo carga. Los promedios por prompt calculados con las 15 repeticiones sobreestiman ligeramente la energía GPU de Llama en los 6 casos afectados.

**Impacto estimado:** los 6 prompts afectados representan el 3.75 % de las 160 combinaciones analizadas (6/160). El sesgo en los promedios de categoría es moderado: el caso más extremo (roleplay Q4, P1, –40 mWh) eleva la media de la categoría en ~8 mWh si no se corrige.

---

## Corrección planificada (sin re-medición)

Los datos ya están disponibles con granularidad por repetición en cada `resultados_completos.csv`. La corrección consiste en **recalcular los promedios de los 6 prompts afectados excluyendo las repeticiones 1–6**, usando directamente las repeticiones 7–15 como estimación del estado estable.

Este recálculo no requiere nueva ejecución del experimento. Se realizará en una iteración posterior al análisis estadístico principal, y los valores corregidos reemplazarán a los actuales en las tablas de resumen por categoría.

---

## Implicaciones para trabajo futuro

1. **Warmup adaptativo en lugar de fijo.** El protocolo actual aplica un warmup de duración fija antes de cada modelo. Sería más robusto usar un warmup **adaptativo basado en estabilización**: correr repeticiones de un prompt piloto hasta que la varianza de `gpu_energy_mwh` entre repeticiones consecutivas caiga por debajo de un umbral (p.ej. 2 %).

2. **Más repeticiones para Llama-2-7B.** Con 15 repeticiones y warmup afectando las primeras 5–6, quedan solo ~9 repeticiones en estado estable. Para Llama, un diseño más robusto requeriría al menos 20–25 repeticiones totales, o descartar sistemáticamente las primeras N como burn-in.

3. **Monitoreo de temperatura del chip durante el experimento.** Instrumentar con `powermetrics` (macOS) o `nvidia-smi` (Windows) para registrar la temperatura del chip en cada repetición permitiría correlacionar directamente la temperatura con `gpu_energy_mwh` y detectar automáticamente el punto de estabilización.

---

## Referencias académicas

### Respaldo directo (mismo hardware: Apple Silicon)

Ahmad, A. et al. "Hot Pixels: Frequency, Power, and Temperature Attacks on GPUs and ARM SoCs." arXiv:2305.12784. https://arxiv.org/pdf/2305.12784

Mide frecuencia, potencia y temperatura en un MacBook Air con Apple M1 (y resultados similares en M2), mostrando cómo el chip alcanza gradualmente el equilibrio térmico bajo carga sostenida. Reportan un hallazgo contraintuitivo similar al observado aquí: las cargas que llegan más rápido al throttling no siempre son las que más potencia consumen en promedio.

### Contexto relacionado (mismo fenómeno, inferencia de LLMs, GPU distinta)

"Energy-Efficient Multimodal Inference Serving with Tri-serve." arXiv:2606.29629. https://arxiv.org/pdf/2606.29629

Durante inferencia de modelos multimodales, documentan la frecuencia del clock GPU empezando alta (~1300 MHz) y bajando gradualmente (~1000 MHz) a lo largo del prefill por throttling térmico — mismo patrón temporal que el observado en este experimento en las primeras repeticiones de Llama-2-7B, aunque en GPU NVIDIA y no en el SoC unificado de Apple.

### Nota metodológica sobre esta sección

No se encontró literatura específica sobre el número de repeticiones de warmup necesarias para estabilizar la energía GPU durante inferencia de LLMs en Apple M4 con llama.cpp. Es un vacío concreto en la literatura existente.

El hallazgo de este experimento —6 casos documentados con r_trend entre –0.58 y –0.96 y caídas de 17 %–42 % en `gpu_energy_mwh`— aporta evidencia empírica propia donde la literatura todavía no cubre el caso exacto. Las referencias citadas respaldan el mecanismo general (throttling térmico en Apple Silicon y en inferencia de LLMs), no el número óptimo de repeticiones de warmup en sí, que sigue siendo una pregunta abierta para este stack específico (llama-cpp-python + Metal + M4).

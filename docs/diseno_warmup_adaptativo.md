# Diseño: Warmup Adaptativo para Llama-2-7B

**Estado:** Diseño pendiente de implementación  
**Documento de referencia:** `docs/hallazgo_warmup_termico_llama.md`  
**Archivo destino de implementación:** `scripts/analisis_categoria_refactor.py`  
**Fecha de diseño:** 2026-08-12

---

## Problema

El protocolo actual aplica un warmup de **duración fija** antes de iniciar las repeticiones de medición. En el análisis sistemático de los 160 prompts del dataset (4 modelos × 8 categorías × 5 prompts), se detectaron **6 casos confirmados** donde este warmup resultó insuficiente:

| Modelo | Quant | Categoría | Prompt | Caída rep1 vs estable |
|--------|-------|-----------|--------|----------------------|
| llama-2-7b | Q4 | roleplay | P1 | 42 % |
| llama-2-7b | Q8 | reasoning | P2 | 27 % |
| llama-2-7b | Q4 | humanities | P1 | 26 % |
| llama-2-7b | Q8 | roleplay | P3 | 24 % |
| llama-2-7b | Q4 | extraction | P1 | 18 % |
| llama-2-7b | Q8 | extraction | P3 | 17 % |

Los 6 casos son exclusivamente de Llama-2-7B. La causa raíz es que Llama-2 base genera respuestas muy largas (frecuentemente hasta el límite de 1024 tokens, ~69 s por repetición), sometiendo al SoC M4 a una carga térmica sostenida que el warmup fijo no llega a estabilizar. Qwen2.5-7B-Instruct no presentó ningún caso confirmado: sus respuestas son más cortas y el chip no alcanza el mismo régimen térmico.

El síntoma observable es `gpu_energy_mwh` que decae monotónicamente durante las primeras 5–6 repeticiones con tokens e `inference_time_s` constantes, lo que indica un cambio en la potencia instantánea del GPU (P-state o thermal throttling del M4) y no en la carga computacional.

---

## Propuesta: warmup adaptativo basado en CV

En lugar de un número fijo de repeticiones de warmup, el criterio de parada se basa en la **estabilidad observada de `gpu_energy_mwh`** durante el propio warmup.

### Criterio de parada

Tras cada repetición de warmup (comenzando desde la tercera), calcular el **coeficiente de variación (CV)** de las últimas 3 repeticiones de warmup:

```
CV = std(gpu_energy[-3:]) / mean(gpu_energy[-3:]) × 100
```

Parar el warmup cuando `CV < 2 %`, con las siguientes restricciones:

| Parámetro | Valor propuesto | Justificación |
|-----------|----------------|---------------|
| Mínimo de reps de warmup | 3 | Necesario para tener ventana de CV válida |
| Máximo de reps de warmup | 10 | Evita warmups indefinidos ante alta variabilidad natural |
| Umbral de CV | < 2 % | Referencia: el coeficiente de variación natural de Llama en estado estable es ~1–2 % según el análisis de las reps 10–15 |

Si el CV no converge antes de la repetición 10, el warmup termina de todas formas y se registra un `warmup_warning = True` en los metadatos del run.

### Métricas a registrar

El nuevo protocolo debe guardar, por cada prompt, las siguientes columnas adicionales en `resultados_completos.csv`:

- `warmup_reps_used`: número de repeticiones de warmup efectivamente realizadas
- `warmup_cv_final`: CV de las últimas 3 reps de warmup al momento de parar
- `warmup_converged`: booleano, `True` si paró por CV < 2 %, `False` si paró por máximo
- `warmup_gpu_mean`: media de `gpu_energy_mwh` de las últimas 3 reps de warmup (para contraste posterior con la media de las reps de medición)

---

## Alcance

**Aplica solo a Llama-2-7B.** Qwen2.5-7B-Instruct no presentó el problema y su warmup actual (fijo) no se modifica. Esta asimetría se declara explícitamente en el código mediante un parámetro de configuración:

```yaml
# config.yaml — sección hipotética
warmup:
  llama-2-7b:
    mode: adaptive
    min_reps: 3
    max_reps: 10
    cv_threshold_pct: 2.0
  qwen2.5-7b:
    mode: fixed
    reps: 3
```

**Aplica solo al script refactorizado.** El archivo `scripts/analisis_categoria.py` (original) no se toca. El warmup adaptativo se implementa únicamente en `scripts/analisis_categoria_refactor.py`, que es el archivo designado para iteraciones de mejora sin afectar el código que generó los datos ya registrados.

---

## Pseudocódigo del loop de warmup adaptativo

```python
def warmup_adaptativo(model, prompt, config):
    gpu_energy_warmup = []
    min_reps = config.warmup.min_reps        # 3
    max_reps = config.warmup.max_reps        # 10
    cv_threshold = config.warmup.cv_threshold_pct  # 2.0

    for rep in range(1, max_reps + 1):
        energy = run_inference(model, prompt)   # retorna gpu_energy_mwh
        gpu_energy_warmup.append(energy)

        if rep >= min_reps:
            ventana = gpu_energy_warmup[-3:]
            cv = (std(ventana) / mean(ventana)) * 100
            if cv < cv_threshold:
                return WarmupResult(
                    reps_used=rep,
                    cv_final=cv,
                    converged=True,
                    gpu_mean=mean(ventana),
                )

    # Máximo alcanzado sin convergencia
    ventana = gpu_energy_warmup[-3:]
    return WarmupResult(
        reps_used=max_reps,
        cv_final=(std(ventana) / mean(ventana)) * 100,
        converged=False,
        gpu_mean=mean(ventana),
        warning=True,
    )
```

---

## Validación planificada

La implementación se valida en dos pasos:

**Paso 1 — Validación numérica (sin re-medición).**  
Usar los datos ya disponibles en `resultados_completos.csv`: simular el criterio adaptativo sobre las 15 repeticiones existentes de cada prompt para estimar cuántas reps de warmup habría requerido. Comparar la media de las reps de medición resultantes contra el recálculo de reps 7–15 documentado en `docs/hallazgo_warmup_termico_llama.md`.

**Paso 2 — Re-corrida real de los 6 casos confirmados.**  
Ejecutar `analisis_categoria_refactor.py` con warmup adaptativo sobre los 6 prompts afectados (roleplay/P1/Q4, reasoning/P2/Q8, humanities/P1/Q4, roleplay/P3/Q8, extraction/P1/Q4, extraction/P3/Q8) y comparar:

- `warmup_reps_used` observado vs. estimación del Paso 1
- Media de `gpu_energy_mwh` de las reps de medición vs. recálculo con reps 7–15
- Diferencia porcentual respecto al valor original (con warmup insuficiente)

La re-corrida es un subset acotado (6 prompts × 15 reps × ~70 s/rep ≈ 105 minutos por cuantización), viable sin comprometer el dataset principal.

---

## Lo que este diseño no cubre

- Cambios a la lógica de medición post-warmup (las 15 reps de medición se mantienen igual).
- Modificaciones al juez de calidad, al análisis estadístico ni a los plots existentes.
- Soporte para hardware Windows/CUDA: el diseño asume Metal (M4). En CUDA, `gpu_energy_mwh` se reporta de forma diferente y el umbral de CV puede necesitar ajuste.
- Warmup adaptativo para Qwen: fuera de alcance por decisión explícita (ningún caso confirmado).

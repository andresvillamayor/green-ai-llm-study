# LLM Quantization Study

Medición de consumo energético en inferencia de LLMs en Apple M4.

##  Descripción

Estudio experimental del impacto energético de Large Language Models durante inferencia en hardware Apple Silicon.

##  Características

-  Medición precisa con CodeCarbon
-  Optimizado para Apple M4 (Metal)
-  Análisis estadístico completo
-  Contexto energético paraguayo

##  Instalación

```bash
git clone https://github.com/andresvillamayor/green-ai-llm-study.git
cd llm-quantization-study
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Reproducibilidad

### Hardware exacto
- Modelo: Apple Mac Mini (Late 2024)
- Chip: Apple M4
- CPU: 10 núcleos (4 Performance + 6 Efficiency)
- GPU: 10 núcleos Metal
- RAM: 16 GB LPDDR5 unificada
- Almacenamiento: SSD interno

### Software
- macOS: 15.6.1
- Python: 3.13.5
- llama-cpp-python: 0.3.21
- CodeCarbon: ver requirements.txt
- Backend GPU: Metal (GGML_METAL=on)

### Parámetros de inferencia
| Parámetro | Valor |
|---|---|
| n_ctx | 1024 |
| max_tokens | 256 |
| temperature | 0.7 |
| top_p | 0.9 |
| seed | 42 |
| n_threads | 8 |
| n_batch | 512 |
| n_gpu_layers (GPU) | -1 (todas) |
| n_gpu_layers (CPU) | 0 |
| echo | False |
| stop | None |
| plantilla de prompt | texto plano sin template |

### Procedimiento de warm-up
Antes de cada configuración se ejecuta `limpiar_memoria()`
con `gc.collect()` y `time.sleep(5)` para estabilizar
el estado del sistema operativo.

### Control de procesos en segundo plano
Se recomienda cerrar todas las aplicaciones antes de correr
el experimento y ejecutar `sudo purge` para liberar
memoria caché del sistema.

### Orden de ejecución
modelo → cuantización → dispositivo → prompt → repetición  
El orden exacto queda registrado en los timestamps del CSV.

### Sincronización energía e inferencia
CodeCarbon inicia `tracker.start()` inmediatamente antes
de `llm()` y ejecuta `tracker.stop()` inmediatamente después.
No hay operaciones intermedias entre la llamada al modelo
y la medición.

### Nota sobre powermetrics
El experimento usa CodeCarbon como única fuente de métricas
energéticas. powermetrics no se usa en el experimento final
por requerir `sudo` y no ser automatizable de forma segura.

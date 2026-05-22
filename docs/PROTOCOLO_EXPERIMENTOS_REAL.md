# Protocolo de Experimentos: Medición de Eficiencia y Consumo Energético en LLMs

**Estudiante:** Andres Villamayor  
**Programa:** Maestría en Ciencia de Datos  
**Ubicación:** Asunción, Paraguay  
**Fecha de Ejecución:** Mayo 8-9, 2026  
**Proyecto:** GREEN-IA

---

## 1. OBJETIVO

Evaluar empíricamente el consumo energético y la eficiencia computacional de modelos de lenguaje de gran escala bajo diferentes configuraciones de cuantización y dispositivos de ejecución.

---

## 2. METODOLOGÍA

### 2.1 Alcance del Estudio

**Etapa evaluada:** Inferencia únicamente

**Justificación:** La inferencia representa el 90% del consumo energético en producción de modelos LLM.

### 2.2 Plataforma de Ejecución

**Hardware:**
- Equipo: Mac Mini (2024)
- Chip: Apple M4
- CPU: 10 cores
- GPU: 10 cores Metal
- RAM: 16 GB
- Modelo: Mac16,10

**Software:**
- Sistema: macOS Sequoia 15.6.1
- Python: 3.13.0
- Motor: llama-cpp-python 0.3.21

### 2.3 Herramientas de Medición

**PowerMetrics:**
- Herramienta nativa de macOS
- Frecuencia: 1 muestra/segundo
- Componentes: CPU, GPU, ANE
- Unidades: milivatios

**CodeCarbon:**
- Versión: 3.2.6
- Factor emisión Paraguay: 0.0 kg CO2/kWh (100% hidroeléctrica)

### 2.4 Repeticiones

- 15 ejecuciones por configuración
- Intervalo de confianza: 95%

---

## 3. DISEÑO EXPERIMENTAL

### 3.1 Modelos

| Modelo | Parámetros |
|--------|-----------|
| Llama-2-7B | 7B |
| Qwen2.5-7B-Instruct | 7B |

### 3.2 Cuantizaciones

| Tipo | Bits |
|------|------|
| Q4 | 4-bit |
| Q8 | 8-bit |

### 3.3 Dispositivos

- CPU: Apple M4
- GPU: Apple M4 Metal

### 3.4 Dataset

- 10 prompts científicos
- Máximo 512 tokens por respuesta

### 3.5 Configuraciones

**Total:** 8 configuraciones  
**Ejecuciones:** 1,200 mediciones

---

## 4. VARIABLES MEDIDAS

**Energéticas:**
- cpu_energy_wh
- gpu_energy_wh  
- total_energy_wh

**Rendimiento:**
- inference_time_s
- tokens_generated
- tokens_per_second

---

## 5. RESULTADOS

### 5.1 Resumen

| Configuración | Energía (mWh) | Velocidad (tok/s) |
|---------------|---------------|-------------------|
| Llama-2 Q4 CPU | 55.18 | 9.8 |
| Llama-2 Q4 GPU | 44.51 | 16.6 |
| Llama-2 Q8 CPU | 99.10 | 8.6 |
| Llama-2 Q8 GPU | 55.75 | 8.1 |
| Qwen2.5 Q4 CPU | 50.92 | 17.0 |
| Qwen2.5 Q4 GPU | 54.91 | 21.2 |
| Qwen2.5 Q8 CPU | 72.59 | 10.4 |
| Qwen2.5 Q8 GPU | 45.59 | 13.3 |

---

## 6. REFERENCIAS

1. Touvron et al. (2023). Llama 2. arXiv:2307.09288
2. Bai et al. (2024). Qwen2.5. arXiv:2409.12186
3. Patterson et al. (2021). Carbon Emissions. arXiv:2104.10350

---

**Autor:** Andres Villamayor  
**Fecha:** Mayo 9, 2026

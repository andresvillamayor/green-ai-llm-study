# Justificaciones de elección de muestra — GREEN-IA

**Proyecto:** Medición de consumo energético en inferencia LLM bajo cuantización Q4 vs Q8  
**Fecha de compilación:** 2026-08-28

Este documento presenta las referencias bibliográficas que justifican las decisiones
metodológicas clave del experimento (modelos, niveles de cuantización, backend de
inferencia). Se incluyen dos formatos: entradas BibTeX listas para `references.bib`,
y un resumen en prosa para discusión con el equipo de tesis.

> **Notas de verificación**
>
> - Los datos de todas las entradas se verificaron contra las fuentes primarias
>   (arXiv, Semantic Scholar, página SustainableSE) el 2026-08-28.
> - La entrada ACM (DOI 10.1145/3767742) fue verificada vía Semantic Scholar API
>   porque dl.acm.org devolvió HTTP 403.
> - **Autoría SustainableSE:** Luis Cruz es el **instructor** del curso TU Delft;
>   los autores del informe son los integrantes del Grupo 23 (Hornet et al.).
>   Se registra con la autoría correcta; se puede añadir
>   `note = {Supervised by Luis Cruz}` si se prefiere la mención del coordinador.
> - **Hot Pixels (arXiv:2305.12784):** la cita en `docs/hallazgo_warmup_termico_llama.md`
>   fue corregida de "Ahmad, A. et al." a "Taneja, Hritvik et al." (2026-08-28).

---

## Parte 1 — Entradas BibTeX

```bibtex
% ---------------------------------------------------------------
% 1. Justificación Llama-2-7B
% ---------------------------------------------------------------
@misc{wilkins2024hybrid,
  author        = {Wilkins, Grant and Keshav, Srinivasan and Mortier, Richard},
  title         = {Hybrid Heterogeneous Clusters Can Lower the Energy Consumption
                   of {LLM} Inference Workloads},
  year          = {2024},
  eprint        = {2407.00010},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DC},
  url           = {https://arxiv.org/abs/2407.00010}
}

% ---------------------------------------------------------------
% 2. Justificación Qwen2.5-7B
%    Autores del informe (Grupo 23): Hornet, Victor; Mihalache, Elena;
%    Mocanu, Andreea; Postu, Alexandru; Sie, Kian.
%    Luis Cruz es el instructor/coordinador del curso, no autor del informe.
% ---------------------------------------------------------------
@misc{hornet2025comparing,
  author       = {Hornet, Victor and Mihalache, Elena and Mocanu, Andreea
                  and Postu, Alexandru and Sie, Kian},
  title        = {Comparing the energy efficiency of different {LLM}
                  inference runtimes},
  year         = {2025},
  howpublished = {TU Delft, Sustainable Software Engineering, Group 23},
  url          = {https://luiscruz.github.io/course_sustainableSE/2025/p1_measuring_software/g23_llm_inference.html},
  note         = {Supervised by Luis Cruz}
}

% ---------------------------------------------------------------
% 3a. Justificación Q4 vs Q8 — ahorro energético FP16 → Q8_0
%     Título completo verificado vía Semantic Scholar.
%     Autoría: campo 'author' con nombres completos no recuperables por
%     bloqueo HTTP 403 en dl.acm.org; se usa la lista de Semantic Scholar.
% ---------------------------------------------------------------
@article{husom2025sustainable,
  author  = {Husom, E. and Goknil, Arda and Astekin, Merve
             and Shar, Lwin Khin and K{\aa}sen, Andre and Sen, Sagar
             and Mithassel, Benedikt Andreas and Soylu, Ahmet},
  title   = {Sustainable {LLM} Inference for Edge {AI}: Evaluating Quantized
             {LLMs} for Energy Efficiency, Output Accuracy, and Inference Latency},
  journal = {{ACM} Transactions on Internet of Things},
  year    = {2025},
  doi     = {10.1145/3767742},
  url     = {https://doi.org/10.1145/3767742}
}

% ---------------------------------------------------------------
% 3b. Justificación Q4 vs Q8 — concepto de "quantization cliff" (Q3/Q2)
% ---------------------------------------------------------------
@misc{licardo2025performance,
  author        = {Licardo, Josip Tomo and Tankovic, Nikola},
  title         = {Performance Trade-offs of Optimizing Small Language Models
                   for {E-Commerce}},
  year          = {2025},
  eprint        = {2510.21970},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2510.21970}
}

% ---------------------------------------------------------------
% 4. Justificación llama.cpp
% ---------------------------------------------------------------
@misc{xue2024powerinfer2,
  author        = {Xue, Zhenliang and Song, Yixin and Mi, Zeyu
                   and Zheng, Xinrui and Xia, Yubin and Chen, Haibo},
  title         = {{PowerInfer-2}: Fast Large Language Model Inference
                   on a Smartphone},
  year          = {2024},
  eprint        = {2406.06282},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2406.06282}
}

% ---------------------------------------------------------------
% 5a. CodeCarbon (ya existente en el proyecto)
%     Presentado en el workshop NeurIPS 2019 "Tackling Climate Change
%     with Machine Learning". Tipo @misc porque no tiene venue de
%     proceedings con actas formales indexadas.
% ---------------------------------------------------------------
@misc{lacoste2019quantifying,
  author        = {Lacoste, Alexandre and Luccioni, Alexandra
                   and Schmidt, Victor and Dandres, Thomas},
  title         = {Quantifying the Carbon Emissions of Machine Learning},
  year          = {2019},
  eprint        = {1910.09700},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CY},
  url           = {https://arxiv.org/abs/1910.09700},
  note          = {NeurIPS 2019 Workshop on Tackling Climate Change with ML}
}

% ---------------------------------------------------------------
% 5b. Hot Pixels / powermetrics en Apple Silicon (ya existente en el proyecto)
%     ATENCIÓN: docs/hallazgo_warmup_termico_llama.md cita erróneamente
%     Corregido en hallazgo_warmup_termico_llama.md el 2026-08-28.
% ---------------------------------------------------------------
@misc{taneja2023hotpixels,
  author        = {Taneja, Hritvik and Kim, Jason and Xu, Jie Jeff
                   and van Schaik, Stephan and Genkin, Daniel and Yarom, Yuval},
  title         = {Hot Pixels: Frequency, Power, and Temperature Attacks
                   on {GPUs} and {ARM} {SoCs}},
  year          = {2023},
  eprint        = {2305.12784},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  url           = {https://arxiv.org/abs/2305.12784}
}
```

---

## Parte 2 — Resumen en prosa con justificaciones

### 2.1 Elección de Llama-2-7B

Se eligió Llama-2-7B como uno de los dos modelos del experimento con base en el
trabajo de Wilkins et al. (2024) \[`wilkins2024hybrid`\], quienes seleccionan
explícitamente Llama-2 7B como referencia en sus mediciones de consumo energético
en clusters heterogéneos. El paper justifica esta elección porque el modelo fue
optimizado por Meta para tareas de diálogo, incorpora mejoras de seguridad y
utilidad respecto a Llama-1, y utiliza **grouped-query attention** (GQA), que
reduce la memoria de caché KV sin sacrificar calidad de generación.

En el contexto de GREEN-IA, Llama-2-7B sirve además como caso representativo de
**modelo base sin ajuste de instrucción** (base model), lo que permite contrastar
su perfil energético con el de Qwen2.5-7B-Instruct bajo las mismas condiciones de
hardware y protocolo.

### 2.2 Elección de Qwen2.5-7B-Instruct

La selección de Qwen2.5-7B-Instruct está respaldada por Hornet et al. (2025)
\[`hornet2025comparing`\], un informe del grupo 23 del curso Sustainable Software
Engineering (TU Delft, coordinado por Luis Cruz). El estudio evalúa el consumo
energético de distintos runtimes de inferencia LLM y utiliza específicamente
Qwen2.5-7B-Instruct con **llama.cpp como backend**, que es exactamente la
configuración de GREEN-IA.

Qwen2.5-7B-Instruct representa la familia de modelos instruct de Alibaba Cloud en
el rango de 7B parámetros, con capacidad multilingüe y alineación para seguir
instrucciones. En GREEN-IA cumple el rol de **modelo instruct** dentro del
comparativo, complementando al Llama-2-7B base.

### 2.3 Elección del rango de cuantización Q4 vs Q8

La decisión de comparar Q4\_K\_M contra Q8\_0 (omitiendo cuantizaciones más
agresivas como Q3 o Q2) está justificada por dos trabajos:

**Husom et al. (2025)** \[`husom2025sustainable`\] miden el consumo energético de
LLMs cuantizados en hardware edge y reportan un **ahorro del 52–54 % de energía**
al pasar de FP16 a Q8\_0, con degradación de calidad contenida. Este resultado
establece que Q8\_0 es el punto de referencia más eficiente que conserva calidad
aceptable, lo que lo convierte en el extremo superior de nuestro rango de
comparación.

**Licardo y Tankovic (2025)** \[`licardo2025performance`\] documentan el concepto
de **"quantization cliff"**: al reducir la cuantización por debajo de Q4 (hacia
Q3 o Q2), la calidad de las respuestas cae abruptamente de forma no lineal,
mientras que entre Q4 y Q8 la degradación es moderada y predecible. Este hallazgo
justifica acotar el experimento al intervalo Q4–Q8, donde el trade-off energía /
calidad es más relevante para aplicaciones reales.

### 2.4 Elección de llama.cpp como backend de inferencia

Se eligió llama.cpp como framework de inferencia local por ser el estándar de
facto para modelos GGUF en hardware de consumo, con soporte nativo para backends
Metal (Apple Silicon) y CUDA (NVIDIA). Xue et al. (2024)
\[`xue2024powerinfer2`\] usan llama.cpp explícitamente como **baseline de consumo
energético** contra el que comparan su sistema PowerInfer-2 en smartphones.
El hecho de que un trabajo orientado a optimización energética tome llama.cpp
como referencia de comparación válida respalda su idoneidad para medir el consumo
real de inferencia en el rango de hardware del experimento GREEN-IA (Apple M4 /
RTX 4060).

### 2.5 Medición de energía con CodeCarbon

La instrumentación energética se apoya en Lacoste et al. (2019)
\[`lacoste2019quantifying`\], trabajo fundacional que propone medir y reportar las
emisiones de carbono en experimentos de machine learning. CodeCarbon implementa
la metodología de ese paper (estimaciones basadas en TDP del hardware y el factor
de intensidad de carbono de la red eléctrica local) y es la librería de Python
más adoptada para este fin en investigación académica.

En GREEN-IA, CodeCarbon 3.2.8 envuelve exclusivamente la llamada de inferencia
(excluye carga del modelo, warmup y cooldown), siguiendo el principio de aislar
la operación bajo estudio. Los valores se reportan como estimaciones basadas en
TDP, no mediciones físicas de potencia.

### 2.6 Uso de powermetrics en Apple Silicon (referencia secundaria)

Para complementar las mediciones de CodeCarbon con datos a nivel de chip
(CPU/GPU/RAM por separado), el protocolo utiliza `powermetrics` de macOS, la
herramienta oficial de Apple para muestrear el consumo del SoC. La referencia
que valida su uso en contextos de seguridad y rendimiento de hardware es Taneja
et al. (2023) \[`taneja2023hotpixels`\], quienes emplean powermetrics y la
correlación frecuencia-temperatura-potencia en Apple Silicon para caracterizar
el comportamiento del chip bajo carga sostenida, un fenómeno directamente
relevante para el efecto de warmup térmico documentado en Llama-2-7B
(ver `docs/hallazgo_warmup_termico_llama.md`).

> **Nota:** la cita en `docs/hallazgo_warmup_termico_llama.md` fue corregida
> de "Ahmad, A. et al." a "Taneja, Hritvik et al." el 2026-08-28.

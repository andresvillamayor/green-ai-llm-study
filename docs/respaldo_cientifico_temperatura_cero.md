# Respaldo científico: temperature=0.0 en las mediciones oficiales

## Por qué se usa
La medición principal de energía usa decodificación greedy (temperature=0.0)
siguiendo práctica establecida en estudios de medición energética:

- Maliakel, Ilager, Brandic, "Characterizing LLM Inference Energy-Performance
  Tradeoffs across Workloads and GPU Scaling" (arXiv:2501.08219) — usan greedy
  decoding explícitamente para "asegurar mediciones de energía
  deterministas y comparables entre modelos".
- "EvaLooop" (arXiv:2505.12185) — usa temperature=0.0, top_p=1.0 "para
  generación determinista".
- "Can LLMs Follow Simple Rules?" (arXiv:2311.04235) — evalúa
  específicamente Llama-2-7B base con decodificación greedy como
  protocolo estándar, mostrando que el rendimiento cae con temperaturas
  más altas.

## Limitación a documentar: temperature=0 no es 100% determinista
arXiv:2506.09501 "Understanding and Mitigating Numerical Sources of
Nondeterminism in LLM Inference" muestra que, por el orden de operaciones
en punto flotante en GPU, pueden aparecer pequeñas diferencias entre
corridas idénticas incluso con greedy decoding. Esto es una explicación
científica complementaria (no sustituta) a la variabilidad de 1-8% ya
documentada en el hallazgo de warmup térmico.

## Por qué además se necesita el segundo esquema (temperatura de fabricante)
"Can LLMs Follow Simple Rules?" (arXiv:2311.04235) evaluó específicamente
Llama-2-7B base y encontró que el rendimiento cae 12.4% al subir la
temperatura de 0 a 0.9. Esto sugiere que medir únicamente con
temperature=0.0 puede no representar el comportamiento del modelo en
condiciones de uso más realistas (con temperaturas más altas, como las
recomendadas oficialmente por cada fabricante) — de ahí la necesidad
del segundo esquema (ver config_temperaturas.yaml).

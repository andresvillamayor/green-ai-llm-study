# Hallazgo — Evaluacion de calidad LLM-as-a-Judge, categoria Math
Proyecto GREEN-IA — Universidad Comunero, Paraguay

## Metodologia

Se evaluaron 5 respuestas de Llama-2-7B Q4 y 5 respuestas
de Llama-2-7B Q8 (mismos 5 prompts de la categoria math
de MT-Bench oficial) usando el paradigma LLM-as-a-Judge
con metodo de comparacion pareada (pairwise comparison).

Juez: Claude (modelo claude-opus-4-5), temperature=0.
Criterios evaluados: correctitud, claridad_razonamiento,
completitud (escala 0-100 cada uno).

Referencia metodologica:
Zheng, L., Chiang, W-L., Sheng, Y., et al. (2023).
Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.
NeurIPS 2023. arXiv:2306.05685

## Resultado cuantitativo

Victorias en comparacion directa (5 prompts evaluados):
  Q4: 2 victorias
  Q8: 0 victorias
  Empates: 3

## Hallazgo principal

Los 5 prompts mostraron correctitud nula o casi nula en
AMBAS cuantizaciones (Q4 y Q8). El juez identifico que
ninguna de las 10 respuestas evaluadas (5 Q4 + 5 Q8)
resolvio correctamente el problema matematico planteado.

Ejemplo (Prompt ID 111 — area de un triangulo):
"Ambas respuestas dan un resultado incorrecto (el area
correcta es 3, no 6), no muestran ningun razonamiento o
proceso de calculo, y presentan repeticiones excesivas
que afectan severamente la claridad y completitud."

## Interpretacion

La ventaja de Q4 (2 victorias, 0 para Q8) no se explica
por mayor correctitud matematica, sino porque en los casos
donde Q4 gano, la respuesta de Q8 fue evaluada como
comparativamente MAS incoherente (ej. "formato de foro
irrelevante", "respuestas aleatorias sin justificacion").
En los 3 empates, ambas cuantizaciones fallaron de forma
equivalente por el mismo patron de degeneracion en
repeticion de texto, documentado en:

Welleck, S., Kulikov, I., Roller, S., Dinan, E., Cho, K.,
& Weston, J. (2019). Neural Text Generation with
Unlikelihood Training. arXiv:1908.04319.

(Ver docs/hallazgo_repeticion_texto.md para el analisis
completo de este fenomeno)

## Conclusion relevante para la tesis

La cuantizacion (Q4 vs Q8) no aparece como la variable
determinante de calidad en la categoria math para este
modelo. La limitacion principal es estructural: Llama-2-7B
en su version base (sin fine-tuning de instrucciones) no
esta optimizado para resolver problemas matematicos paso
a paso ni para finalizar respuestas de forma natural bajo
decodificacion determinista (temperature=0).

Esto tiene una implicancia importante para el argumento
central de la tesis: la reduccion de consumo energetico
de Q4 respecto a Q8 (22.5% a 35.2% segun categoria, ver
docs/comparacion_q4_vs_q8.md) NO viene acompañada de una
perdida de calidad relativa en esta categoria, ya que
ambas cuantizaciones parten de una base de correctitud
igualmente baja.

## Redaccion sugerida para la tesis

"La evaluacion de calidad mediante LLM-as-a-Judge (Zheng
et al., NeurIPS 2023) sobre los 5 prompts de la categoria
math revelo que ni Llama-2-7B Q4 ni Q8 lograron responder
correctamente ninguno de los problemas evaluados, con
Q4 obteniendo una ligera ventaja en la comparacion directa
(2 victorias contra 0 para Q8, con 3 empates). El analisis
cualitativo del juez indica que esta diferencia no refleja
mayor correctitud matematica de Q4, sino una degeneracion
comparativamente menos severa en los casos donde broke el
empate. Este resultado sugiere que, en tareas de
razonamiento matematico con modelos base sin fine-tuning
de instrucciones, la cuantizacion no es el factor
determinante de calidad, lo que refuerza la viabilidad de
Q4 como alternativa energeticamente mas eficiente sin
sacrificio adicional de calidad relativa en este contexto."

## Limitaciones de esta prueba de concepto

- Muestra pequeña (5 prompts, 1 sola categoria)
- Evaluado con 1 repeticion por prompt (no las 15 del
  experimento de energia), por lo que no se captura
  variabilidad entre repeticiones en la calidad
- Pendiente: escalar la evaluacion a las 8 categorias
  completas de MT-Bench para confirmar si el patron se
  mantiene en categorias con estilo de respuesta distinto
  (ej. writing, roleplay, donde no hay un resultado unico
  "correcto")

## Estado

Prueba de concepto completada y documentada.
Categoria evaluada: math (1 de 8)
Pendiente: extraction, humanities, coding, reasoning,
roleplay, stem, writing

Fecha: julio 2026

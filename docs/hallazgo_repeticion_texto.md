# Hallazgo — Repeticion de texto en Llama-2-7B base
Proyecto GREEN-IA — Universidad Comunero, Paraguay

## Observacion

Al capturar las respuestas completas del modelo en
scripts/analisis_categoria_prueba.py (categoria math,
ambas cuantizaciones Q4 y Q8), se observo que el modelo
repite la misma respuesta o variaciones de ella multiples
veces dentro de una misma generacion, hasta alcanzar el
limite de max_tokens configurado.

Ejemplo real (Prompt ID 111, respuesta Q4):
"The area of the triangle is 6. The area of the triangle
is 6. The vertices of a triangle are at points (0, 0),
(-1, 1), and (3, 3). The area of the triangle is 6..."

## Causa raiz — respaldo cientifico

Este comportamiento corresponde a un fenomeno documentado
en la literatura como "neural text degeneration".

### Referencia principal

Welleck, S., Kulikov, I., Roller, S., Dinan, E., Cho, K.,
& Weston, J. (2019). Neural Text Generation with
Unlikelihood Training. arXiv:1908.04319.

Hallazgo relevante del paper: los modelos de lenguaje
entrenados con el objetivo de maxima verosimilitud
(maximum likelihood) exhiben repeticion a nivel de
secuencia, especialmente con decodificacion determinista.
Con decodificacion greedy, el porcentaje promedio de
n-gramas repetidos en las continuaciones del modelo
alcanza 43%, frente a 0.5% en texto humano.

Esto es directamente aplicable a este experimento: se usa
temperature=0 (decodificacion determinista/greedy) por
requerimiento de reproducibilidad cientifica, que es
exactamente la condicion donde el paper documenta la
mayor tasa de degradacion por repeticion.

### Referencia complementaria — especificidad del modelo base

HuggingFace Discussion (2023). Llama 2 repeats its prompt
as output without answering the prompt.
https://discuss.huggingface.co/t/llama-2-repeats-its-prompt-as-output-without-answering-the-prompt/78230

Confirma que Llama-2-7B en su version base (sin
fine-tuning de instrucciones) esta entrenado unicamente
para predecir la siguiente secuencia de palabras, no para
seguir instrucciones ni finalizar respuestas de forma
natural. Los modelos base conocen lenguaje pero no
conversacion.

### Referencia tecnica — ausencia de token EOS

GitHub Issue #23230, huggingface/transformers (2023).
"llama model can't generate EOS".
https://github.com/huggingface/transformers/issues/23230

Documenta que el metodo generate() de Llama no produce
el token de fin de secuencia (EOS) bajo ciertas
configuraciones, lo que resulta en generacion continuada
hasta alcanzar el limite de max_tokens.

### Referencia sobre mecanismo interno

Repetition Neurons: How Do Language Models Produce
Repetitions? (2024). arXiv:2410.13497.

Identifica neuronas especificas dentro de la arquitectura
del modelo responsables de producir output repetitivo,
confirmando que el fenomeno es un mecanismo interno
reproducible del modelo, no un artefacto aleatorio del
experimento.

## Decision metodologica

Se decidio NO truncar ni limpiar las respuestas antes de
enviarlas al juez de calidad (scripts/juez_calidad.py).
El texto completo, con sus repeticiones, se considera el
dato experimental real bajo las condiciones de
temperature=0 elegidas por reproducibilidad. Truncar la
respuesta alteraria el dato observado y podria ocultar
una diferencia real de calidad entre Q4 y Q8 si una
cuantizacion repite mas que la otra.

## Relevancia para la comparacion Q4 vs Q8

Este hallazgo abre una pregunta de investigacion adicional:
¿la cuantizacion afecta la tendencia a repetir? El juez de
calidad debe considerar la repeticion como parte de los
criterios de evaluacion (claridad, concision), lo cual ya
esta contemplado en el diseno de scripts/juez_calidad.py.

## Redaccion sugerida para la tesis

"Se observo que Llama-2-7B, en su version base, tiende a
repetir el contenido de su respuesta hasta alcanzar el
limite de tokens configurado. Este comportamiento es
consistente con el fenomeno de neural text degeneration
documentado por Welleck et al. (2019), agravado por el
uso de decodificacion determinista (temperature=0)
requerida para garantizar la reproducibilidad
experimental. Los modelos base, a diferencia de sus
contrapartes instruction-tuned, no estan optimizados para
finalizar respuestas de forma natural (discusion tecnica
de HuggingFace, 2023), lo que se refleja consistentemente
en las 5 respuestas capturadas para la categoria math en
ambas cuantizaciones evaluadas."

## Estado

Confirmado y documentado — 4 fuentes, incluyendo un paper
con metodologia cuantitativa (Welleck et al. 2019) y un
issue tecnico oficial del repositorio de Hugging Face
Transformers.  Confirmar el archivo creado."

Fecha: julio 2026

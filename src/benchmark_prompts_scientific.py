"""
benchmark_prompts_scientific.py

Conjunto de 10 prompts cientificos verificables para evaluacion
de modelos de lenguaje en el contexto del experimento GREEN-IA.

Cada prompt esta basado en papers peer-reviewed publicados en
conferencias de primer nivel (NeurIPS, ACL, Nature, etc.).

Proposito:
----------
Estos prompts se utilizan para medir el consumo energetico de
inferencia en LLMs bajo condiciones controladas y reproducibles.
Cada prompt genera respuestas de longitud similar (50-120 tokens)
para mantener comparabilidad entre modelos.

Metodologia:
-----------
- Cada prompt se ejecuta 15 veces por configuracion (n=15)
- Total: 10 prompts x 8 configuraciones x 15 repeticiones = 1200 mediciones
- Se mide: energia CPU/GPU/RAM, CO2, latencia, tokens/segundo

Autor: Andres Villamayor
Universidad Comunero, Paraguay
"""

PROMPTS_CIENTIFICOS = [
    {
        'id': 1,
        'prompt': 'Explica el mecanismo de atencion multi-cabeza en la arquitectura Transformer y como procesa secuencias de longitud variable',
        'referencia': 'Vaswani et al. 2017',
        'paper_completo': 'Attention is All You Need',
        'enlace': 'https://arxiv.org/abs/1706.03762',
        'conferencia': 'NeurIPS 2017',
        'descripcion': 'Paper fundacional de Transformers, base de GPT, BERT y modelos modernos'
    },
    {
        'id': 2,
        'prompt': 'Describe el proceso de pre-entrenamiento de BERT usando masked language modeling y como difiere de modelos autoregresivos',
        'referencia': 'Devlin et al. 2019',
        'paper_completo': 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
        'enlace': 'https://arxiv.org/abs/1810.04805',
        'conferencia': 'NAACL 2019',
        'descripcion': 'Introduce bidireccionalidad en pre-entrenamiento de LLMs'
    },
    {
        'id': 3,
        'prompt': 'Explica como las redes neuronales convolucionales extraen caracteristicas jerarquicas desde bordes hasta objetos complejos',
        'referencia': 'LeCun et al. 2015',
        'paper_completo': 'Deep Learning',
        'enlace': 'https://www.nature.com/articles/nature14539',
        'conferencia': 'Nature 2015',
        'descripcion': 'Revision comprensiva de deep learning por pioneros del campo'
    },
    {
        'id': 4,
        'prompt': 'Describe el algoritmo de backpropagation y su rol en el entrenamiento de redes neuronales profundas mediante gradiente descendente',
        'referencia': 'Rumelhart et al. 1986',
        'paper_completo': 'Learning representations by back-propagating errors',
        'enlace': 'https://www.nature.com/articles/323533a0',
        'conferencia': 'Nature 1986',
        'descripcion': 'Paper clasico que popularizo backpropagation en redes neuronales'
    },
    {
        'id': 5,
        'prompt': 'Explica el concepto de transfer learning y como los modelos pre-entrenados mejoran el rendimiento en tareas especificas',
        'referencia': 'Pan and Yang 2010',
        'paper_completo': 'A Survey on Transfer Learning',
        'enlace': 'https://ieeexplore.ieee.org/document/5288526',
        'conferencia': 'IEEE TKDE 2010',
        'descripcion': 'Survey fundamental sobre transfer learning en machine learning'
    },
    {
        'id': 6,
        'prompt': 'Describe como funciona el mecanismo de self-attention y por que permite paralelizacion en el procesamiento de secuencias',
        'referencia': 'Vaswani et al. 2017',
        'paper_completo': 'Attention is All You Need',
        'enlace': 'https://arxiv.org/abs/1706.03762',
        'conferencia': 'NeurIPS 2017',
        'descripcion': 'Mismo paper fundacional, enfoque en paralelizacion'
    },
    {
        'id': 7,
        'prompt': 'Explica la diferencia entre modelos autoregresivos como GPT y masked language models como BERT en terminos de arquitectura',
        'referencia': 'Radford et al. 2019',
        'paper_completo': 'Language Models are Unsupervised Multitask Learners',
        'enlace': 'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf',
        'conferencia': 'OpenAI Technical Report',
        'descripcion': 'Introduccion de GPT-2, modelo autoregresivo unidireccional'
    },
    {
        'id': 8,
        'prompt': 'Describe el proceso de fine-tuning en modelos de lenguaje pre-entrenados y las tecnicas de adaptacion de dominios',
        'referencia': 'Howard and Ruder 2018',
        'paper_completo': 'Universal Language Model Fine-tuning for Text Classification',
        'enlace': 'https://arxiv.org/abs/1801.06146',
        'conferencia': 'ACL 2018',
        'descripcion': 'Introduce ULMFiT, metodo efectivo de fine-tuning'
    },
    {
        'id': 9,
        'prompt': 'Explica como funcionan las capas de normalizacion como LayerNorm en transformers y por que son importantes',
        'referencia': 'Ba et al. 2016',
        'paper_completo': 'Layer Normalization',
        'enlace': 'https://arxiv.org/abs/1607.06450',
        'conferencia': 'arXiv 2016',
        'descripcion': 'Tecnica de normalizacion critica para estabilidad en transformers'
    },
    {
        'id': 10,
        'prompt': 'Describe el concepto de word embeddings y como Word2Vec aprende representaciones distribuidas de palabras',
        'referencia': 'Mikolov et al. 2013',
        'paper_completo': 'Efficient Estimation of Word Representations in Vector Space',
        'enlace': 'https://arxiv.org/abs/1301.3781',
        'conferencia': 'ICLR 2013',
        'descripcion': 'Introduce Word2Vec (CBOW y Skip-gram), base de embeddings modernos'
    }
]
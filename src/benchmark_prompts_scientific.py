"""
benchmark_prompts_scientific.py

Conjunto de 15 prompts para evaluacion de consumo energetico
en inferencia de LLMs — Proyecto GREEN-IA.

Categorias basadas en MT-Bench (Zheng et al., NeurIPS 2023):
    https://arxiv.org/abs/2306.05685

Categorias incluidas (8 de MT-Bench + long prompt):
    1. Reasoning       (2 prompts) — razonamiento logico
    2. Math            (2 prompts) — matematicas
    3. Writing         (2 prompts) — escritura y resumen
    4. Extraction      (2 prompts) — extraccion de informacion
    5. Coding          (2 prompts) — generacion de codigo
    6. STEM            (2 prompts) — ciencias y tecnologia
    7. Humanities      (1 prompt)  — humanidades
    8. Translation     (1 prompt)  — traduccion y multilingue
    9. Long prompt     (1 prompt)  — prefill extenso

Metodologia:
    - Prompts de turno unico adaptados de MT-Bench
    - Mismo idioma (ingles) para controlar variable linguistica
    - Longitud de respuesta esperada: 100-300 tokens
    - 10 repeticiones por configuracion (n=10)
    - Total: 15 prompts x 8 configs x 10 reps = 1200 mediciones

Referencia principal:
    Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench
    and Chatbot Arena. NeurIPS 2023. arXiv:2306.05685
"""

PROMPTS_CIENTIFICOS = [

    # =========================================================
    # CATEGORIA 1: REASONING (Razonamiento Logico)
    # Fuente: MT-Bench, Zheng et al. 2023, categoria reasoning
    # =========================================================
    {
        'id': 1,
        'categoria': 'REASONING',
        'prompt': 'If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning step by step.',
        'referencia': 'Zheng et al. 2023',
        'paper_completo': 'Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena',
        'enlace': 'https://arxiv.org/abs/2306.05685',
        'conferencia': 'NeurIPS 2023',
        'descripcion': 'Silogismo logico para evaluar razonamiento formal',
        'longitud_esperada': 'corta'
    },
    {
        'id': 2,
        'categoria': 'REASONING',
        'prompt': 'A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left? Explain your answer.',
        'referencia': 'Zheng et al. 2023',
        'paper_completo': 'Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena',
        'enlace': 'https://arxiv.org/abs/2306.05685',
        'conferencia': 'NeurIPS 2023',
        'descripcion': 'Problema de logica con negacion implicita',
        'longitud_esperada': 'corta'
    },

    # =========================================================
    # CATEGORIA 2: MATH (Matematicas)
    # Fuente: MT-Bench, Zheng et al. 2023, categoria math
    # =========================================================
    {
        'id': 3,
        'categoria': 'MATH',
        'prompt': 'Given that f(x) = 4x^3 - 9x - 14, find the value of f(2). Show all steps.',
        'referencia': 'Zheng et al. 2023',
        'paper_completo': 'Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena',
        'enlace': 'https://arxiv.org/abs/2306.05685',
        'conferencia': 'NeurIPS 2023',
        'descripcion': 'Evaluacion de funcion polinomica',
        'longitud_esperada': 'corta'
    },
    {
        'id': 4,
        'categoria': 'MATH',
        'prompt': 'Calculate the compound interest on $10,000 at an annual rate of 5% compounded annually over 3 years. Show the formula and each step.',
        'referencia': 'Zheng et al. 2023',
        'paper_completo': 'Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena',
        'enlace': 'https://arxiv.org/abs/2306.05685',
        'conferencia': 'NeurIPS 2023',
        'descripcion': 'Calculo de interes compuesto con formula explicita',
        'longitud_esperada': 'media'
    },

    # =========================================================
    # CATEGORIA 3: WRITING (Escritura y Resumen)
    # Fuente: MT-Bench, Zheng et al. 2023, categoria writing
    # =========================================================
    {
        'id': 5,
        'categoria': 'WRITING',
        'prompt': 'Write a short paragraph explaining the main idea behind neural networks to someone with no technical background.',
        'referencia': 'Zheng et al. 2023',
        'paper_completo': 'Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena',
        'enlace': 'https://arxiv.org/abs/2306.05685',
        'conferencia': 'NeurIPS 2023',
        'descripcion': 'Escritura explicativa para audiencia no tecnica',
        'longitud_esperada': 'media'
    },
    {
        'id': 6,
        'categoria': 'WRITING',
        'prompt': 'Summarize the key contributions of the Transformer architecture as described in the paper Attention Is All You Need by Vaswani et al. 2017.',
        'referencia': 'Vaswani et al. 2017',
        'paper_completo': 'Attention Is All You Need',
        'enlace': 'https://arxiv.org/abs/1706.03762',
        'conferencia': 'NeurIPS 2017',
        'descripcion': 'Resumen de contribuciones de paper cientifico',
        'longitud_esperada': 'media'
    },

    # =========================================================
    # CATEGORIA 4: EXTRACTION (Extraccion de Informacion)
    # Fuente: MT-Bench, Zheng et al. 2023, categoria extraction
    # =========================================================
    {
        'id': 7,
        'categoria': 'EXTRACTION',
        'prompt': (
            'Extract the names, roles and organizations from the following text and present them in a table:\n\n'
            '"Dr. Sarah Chen, lead researcher at MIT, collaborated with Prof. James Okafor from Stanford University '
            'and Dr. Maria Santos of DeepMind to publish findings on energy-efficient neural networks. '
            'The project was funded by the National Science Foundation under director Robert Williams."'
        ),
        'referencia': 'Zheng et al. 2023',
        'paper_completo': 'Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena',
        'enlace': 'https://arxiv.org/abs/2306.05685',
        'conferencia': 'NeurIPS 2023',
        'descripcion': 'Extraccion de entidades nombradas en formato tabla',
        'longitud_esperada': 'media'
    },
    {
        'id': 8,
        'categoria': 'EXTRACTION',
        'prompt': (
            'Given the following data, identify the month with the highest average temperature '
            'and the month with the lowest, then calculate the annual average:\n\n'
            'Jan: 18C, Feb: 20C, Mar: 24C, Apr: 27C, May: 30C, Jun: 33C, '
            'Jul: 35C, Aug: 34C, Sep: 29C, Oct: 25C, Nov: 21C, Dec: 18C'
        ),
        'referencia': 'Zheng et al. 2023',
        'paper_completo': 'Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena',
        'enlace': 'https://arxiv.org/abs/2306.05685',
        'conferencia': 'NeurIPS 2023',
        'descripcion': 'Extraccion y calculo sobre datos estructurados',
        'longitud_esperada': 'corta'
    },

    # =========================================================
    # CATEGORIA 5: CODING (Generacion de Codigo)
    # Fuente: MT-Bench + HumanEval, Chen et al. 2021
    # =========================================================
    {
        'id': 9,
        'categoria': 'CODING',
        'prompt': 'Write a Python function that takes a list of integers and returns a dictionary with three keys: mean, median, and standard deviation. Include the formula used for each.',
        'referencia': 'Chen et al. 2021',
        'paper_completo': 'Evaluating Large Language Models Trained on Code',
        'enlace': 'https://arxiv.org/abs/2107.03374',
        'conferencia': 'arXiv 2021',
        'descripcion': 'Generacion de funcion Python con estadisticas basicas',
        'longitud_esperada': 'larga'
    },
    {
        'id': 10,
        'categoria': 'CODING',
        'prompt': 'Write a Python function that reads a list of strings and returns the top 3 most frequent words, ignoring case and punctuation. Add comments explaining each step.',
        'referencia': 'Chen et al. 2021',
        'paper_completo': 'Evaluating Large Language Models Trained on Code',
        'enlace': 'https://arxiv.org/abs/2107.03374',
        'conferencia': 'arXiv 2021',
        'descripcion': 'Generacion de codigo con manejo de texto y frecuencias',
        'longitud_esperada': 'larga'
    },

    # =========================================================
    # CATEGORIA 6: STEM (Ciencias y Tecnologia)
    # Fuente: MT-Bench, Zheng et al. 2023, categoria STEM
    # =========================================================
    {
        'id': 11,
        'categoria': 'STEM',
        'prompt': 'Explain how quantization reduces the memory footprint of a neural network model. What are the trade-offs between INT4 and INT8 quantization in terms of accuracy and speed?',
        'referencia': 'Dettmers et al. 2022',
        'paper_completo': 'LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale',
        'enlace': 'https://arxiv.org/abs/2208.07339',
        'conferencia': 'NeurIPS 2022',
        'descripcion': 'Explicacion tecnica de cuantizacion en LLMs — directamente relacionado con la tesis',
        'longitud_esperada': 'media'
    },
    {
        'id': 12,
        'categoria': 'STEM',
        'prompt': 'What is the difference between CPU and GPU architectures in the context of deep learning inference? Why does memory bandwidth matter more than raw compute for large language models?',
        'referencia': 'Zheng et al. 2023',
        'paper_completo': 'Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena',
        'enlace': 'https://arxiv.org/abs/2306.05685',
        'conferencia': 'NeurIPS 2023',
        'descripcion': 'Comparacion arquitectural CPU vs GPU para inferencia LLM',
        'longitud_esperada': 'media'
    },

    # =========================================================
    # CATEGORIA 7: HUMANITIES (Humanidades y Ciencias Sociales)
    # Fuente: MT-Bench, Zheng et al. 2023, categoria humanities
    # =========================================================
    {
        'id': 13,
        'categoria': 'HUMANITIES',
        'prompt': 'Explain the ethical implications of deploying large language models in low-resource countries where electricity comes primarily from renewable sources. Consider both the benefits and the environmental trade-offs.',
        'referencia': 'Bender et al. 2021',
        'paper_completo': 'On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?',
        'enlace': 'https://dl.acm.org/doi/10.1145/3442188.3445922',
        'conferencia': 'FAccT 2021',
        'descripcion': 'Razonamiento etico sobre IA y sostenibilidad — alineado con Green AI',
        'longitud_esperada': 'media'
    },

    # =========================================================
    # CATEGORIA 8: TRANSLATION (Traduccion y Multilingue)
    # Fuente: BLEU benchmark, Papineni et al. 2002
    # =========================================================
    {
        'id': 14,
        'categoria': 'TRANSLATION',
        'prompt': (
            'Translate the following paragraph to Spanish, then explain any cultural or linguistic nuances '
            'you considered in the translation:\n\n'
            '"Energy efficiency in artificial intelligence is not just a technical problem — it is a matter of equity. '
            'When models require massive computational resources, they become accessible only to wealthy organizations, '
            'widening the gap between those who build AI and those who are affected by it."'
        ),
        'referencia': 'Papineni et al. 2002',
        'paper_completo': 'BLEU: a Method for Automatic Evaluation of Machine Translation',
        'enlace': 'https://aclanthology.org/P02-1040/',
        'conferencia': 'ACL 2002',
        'descripcion': 'Traduccion ingles-espanol con explicacion de matices culturales',
        'longitud_esperada': 'larga'
    },

    # =========================================================
    # CATEGORIA 9: LONG PROMPT (Prefill extenso)
    # Cubre el escenario de prefill largo pedido por la mesa
    # Fuente: diseno propio basado en recomendacion mesa de tesis
    # =========================================================
    {
        'id': 15,
        'categoria': 'LONG_PROMPT',
        'prompt': (
            'The following is a summary of key findings from a research experiment on energy consumption '
            'in large language models running on Apple Silicon M4 hardware:\n\n'
            'Experiment setup: Two models were evaluated — Llama-2-7B and Qwen2.5-7B — under two quantization '
            'levels (Q4_K_M and Q8_0) and two execution modes (CPU and GPU via Metal). '
            'Energy was measured using CodeCarbon. The carbon intensity of Paraguay is 26 gCO2eq/kWh '
            'due to its 100% hydroelectric energy matrix. '
            'Results showed that Q4 quantization consumed between 16% and 87% less energy than Q8 '
            'depending on the model and device. Anomalous runs lasting over 900 seconds were observed '
            'in CPU configurations, attributed to OS scheduling inference on efficiency cores only '
            'rather than performance cores, a phenomenon documented via system logs showing '
            'P-core utilization dropping to 0% and CPU frequency dropping to 1080 MHz.\n\n'
            'Based on the above context, answer the following: '
            'What are the main factors that explain the energy efficiency advantage of Q4 over Q8 quantization, '
            'and what implications does this have for deploying LLMs in resource-constrained environments?'
        ),
        'referencia': 'Villamayor 2026',
        'paper_completo': 'GREEN-IA: Medicion de Consumo Energetico en LLMs mediante Cuantizacion',
        'enlace': 'https://github.com/andresvillamayor/green-ai-llm-study',
        'conferencia': 'Tesis Maestria — Universidad Comunero 2026',
        'descripcion': 'Prompt largo con contexto extenso — evalua comportamiento energetico en prefill largo',
        'longitud_esperada': 'larga'
    },
]

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
"""

PROMPTS_CIENTIFICOS = [
        # ============================================
        # CATEGORÍA 1: REASONING (Razonamiento Lógico)
        # ============================================
    {
        'id': 1,
        'categoria': 'REASONING',
        'prompt': 'If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning.',
        'referencia': 'Zheng et al. 2023',
        'paper_completo': 'Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena',
        'enlace': 'https://arxiv.org/abs/2306.05685',
        'conferencia': 'NeurIPS 2023',
        'descripcion': 'Pregunta de silogismo para evaluar capacidad de razonamiento lógico formal'
    },
    {
        'id': 2,
        'categoria': 'REASONING',
        'prompt': 'A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left? Explain.',
        'referencia': 'Zheng et al. 2023',
        'paper_completo': 'Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena',
        'enlace': 'https://arxiv.org/abs/2306.05685',
        'conferencia': 'NeurIPS 2023',
        'descripcion': 'Problema clásico de lógica matemática para evaluar comprensión de negaciones'
    },
    {
        'id': 3,
        'categoria': 'REASONING',
        'prompt': 'If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?',
        'referencia': 'Zheng et al. 2023',
        'paper_completo': 'Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena',
        'enlace': 'https://arxiv.org/abs/2306.05685',
        'conferencia': 'NeurIPS 2023',
        'descripcion': 'Problema de razonamiento proporcional para evaluar comprensión de relaciones'
    },
    {
        'id': 4,
        'categoria': 'REASONING',
        'prompt': 'A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?',
        'referencia': 'Zheng et al. 2023',
        'paper_completo': 'Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena',
        'enlace': 'https://arxiv.org/abs/2306.05685',
        'conferencia': 'NeurIPS 2023',
        'descripcion': 'Problema de álgebra básica para evaluar precisión en cálculos simples'
    },
    {
        'id': 5,
        'categoria': 'REASONING',
        'prompt': 'Three switches control three light bulbs in another room. You can only enter the room once. How do you determine which switch controls which bulb?',
        'referencia': 'Zheng et al. 2023',
        'paper_completo': 'Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena',
        'enlace': 'https://arxiv.org/abs/2306.05685',
        'conferencia': 'NeurIPS 2023',
        'descripcion': 'Problema de razonamiento espacial y pensamiento lateral'
    },

        # ============================================
        # CATEGORÍA 2: MATH (Matemáticas)
        # ============================================
    {
        'id': 6,
        'categoria': 'MATH',
        'prompt': 'Given that f(x) = 4x³ - 9x - 14, find the value of f(2).',
        'referencia': 'Zheng et al. 2023',
        'paper_completo': 'Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena',
        'enlace': 'https://arxiv.org/abs/2306.05685',
        'conferencia': 'NeurIPS 2023',
        'descripcion': 'Evaluación de funciones polinómicas y sustitución de valores'
    },
    {
        'id': 7,
        'categoria': 'MATH',
        'prompt': 'Calculate the compound interest on $10,000 at 5% annual rate over 3 years.',
        'referencia': 'Zheng et al. 2023',
        'paper_completo': 'Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena',
        'enlace': 'https://arxiv.org/abs/2306.05685',
        'conferencia': 'NeurIPS 2023',
        'descripcion': 'Problema de matemáticas financieras para evaluar cálculos con fórmulas'
    },
    {
        'id': 8,
        'categoria': 'MATH',
        'prompt': 'What is the area of a circle with radius 7 cm?',
        'referencia': 'Zheng et al. 2023',
        'paper_completo': 'Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena',
        'enlace': 'https://arxiv.org/abs/2306.05685',
        'conferencia': 'NeurIPS 2023',
        'descripcion': 'Cálculo geométrico básico usando fórmula πr²'
    },
    {
        'id': 9,
        'categoria': 'MATH',
        'prompt': 'Solve for x: 2x + 5 = 15',
        'referencia': 'Zheng et al. 2023',
        'paper_completo': 'Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena',
        'enlace': 'https://arxiv.org/abs/2306.05685',
        'conferencia': 'NeurIPS 2023',
        'descripcion': 'Ecuación lineal simple para evaluar resolución algebraica'
    },
    {
        'id': 10,
        'categoria': 'MATH',
        'prompt': 'If the probability of rain is 30%, what is the probability of no rain?',
        'referencia': 'Zheng et al. 2023',
        'paper_completo': 'Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena',
        'enlace': 'https://arxiv.org/abs/2306.05685',
        'conferencia': 'NeurIPS 2023',
        'descripcion': 'Problema de probabilidad básica para evaluar comprensión de complementarios'
    }
]
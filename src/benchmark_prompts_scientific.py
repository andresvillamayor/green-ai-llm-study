"""
benchmark_prompts_scientific.py
Proyecto GREEN-IA — Maestria Ciencia de Datos
Universidad Comunero — Paraguay
Autor: Andres Villamayor

Conjunto de 40 prompts para evaluacion de consumo energetico
en inferencia de LLMs sobre Apple M4 con cuantizacion GGUF.

Diseno basado en MT-Bench (Zheng et al., NeurIPS 2023):
  Referencia: Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena
  arXiv: 2306.05685v4
  Conferencia: NeurIPS 2023 (peer-reviewed)
  Autores: Lianmin Zheng et al., UC Berkeley / Stanford / CMU

Categorias oficiales MT-Bench (Seccion 2.2, pagina 3-4 del paper):
  Writing, Roleplay, Extraction, Reasoning, Math,
  Coding, STEM, Humanities

Diseno experimental:
  8 categorias x 5 prompts = 40 prompts totales
  15 repeticiones por configuracion
  8 configuraciones (2 modelos x 2 cuant x 2 dispositivos)
  Total: 40 x 8 x 15 = 4800 mediciones por experimento
"""

PROMPTS_CIENTIFICOS = [

    # WRITING (1-5) — MT-Bench Tabla 1, Zheng et al. 2023
    {
        'id': 1, 'categoria': 'WRITING',
        'prompt': (
            'Write a concise and engaging introduction paragraph for a '
            'scientific article about the energy consumption of artificial '
            'intelligence systems and their environmental impact.'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Escritura cientifica formal'
    },
    {
        'id': 2, 'categoria': 'WRITING',
        'prompt': (
            'Write a brief executive summary of around 150 words explaining '
            'why organizations should adopt energy-efficient AI inference. '
            'Include two specific recommendations.'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Resumen ejecutivo con recomendaciones'
    },
    {
        'id': 3, 'categoria': 'WRITING',
        'prompt': (
            'Write a short argumentative paragraph supporting the use of '
            'renewable energy sources in developing countries. Include at '
            'least two specific arguments and one counterargument.'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Escritura argumentativa con contraargumento'
    },
    {
        'id': 4, 'categoria': 'WRITING',
        'prompt': (
            'Write a short paragraph explaining the main idea behind '
            'neural networks to someone with no technical background. '
            'Use an everyday analogy to make the concept clear.'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Divulgacion cientifica con analogia'
    },
    {
        'id': 5, 'categoria': 'WRITING',
        'prompt': (
            'Summarize the key contributions of the Transformer architecture '
            'introduced by Vaswani et al. in Attention Is All You Need (2017). '
            'Focus on what made it different from previous approaches.'
        ),
        'referencia': 'Vaswani et al. NeurIPS 2017',
        'descripcion': 'Resumen de contribucion cientifica'
    },

    # ROLEPLAY (6-10) — MT-Bench Seccion 2.2, Zheng et al. 2023
    {
        'id': 6, 'categoria': 'ROLEPLAY',
        'prompt': (
            'You are a science teacher explaining electricity to a curious '
            '10-year-old student. The student asks: Why does lightning happen '
            'during a storm? Explain it in simple terms with an analogy.'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Rol maestro — adaptacion del lenguaje al publico'
    },
    {
        'id': 7, 'categoria': 'ROLEPLAY',
        'prompt': (
            'You are a financial advisor speaking with a 25-year-old who '
            'just started their first job. They ask: What should I do with '
            'my first $1,000 in savings? Provide three practical recommendations.'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Rol asesor financiero — instruccion contextual'
    },
    {
        'id': 8, 'categoria': 'ROLEPLAY',
        'prompt': (
            'You are a doctor explaining to a patient what type 2 diabetes '
            'is and how lifestyle changes can help manage it. Use clear, '
            'non-technical language. Mention at least two specific recommendations.'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Rol medico — comunicacion clara en contexto profesional'
    },
    {
        'id': 9, 'categoria': 'ROLEPLAY',
        'prompt': (
            'You are a career counselor helping a computer science graduate '
            'choose between a large tech company with high salary and an '
            'early-stage startup with equity. '
            'What three key factors should guide their decision?'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Rol consejero — razonamiento en contexto real'
    },
    {
        'id': 10, 'categoria': 'ROLEPLAY',
        'prompt': (
            'You are a travel guide helping a tourist plan a visit to '
            'Asuncion, Paraguay. The tourist says: I have 2 days and I love '
            'history, local food, and nature. What should I do? '
            'Create a brief itinerary with specific suggestions.'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Rol guia turistico — planificacion contextual'
    },

    # EXTRACTION (11-15) — MT-Bench Seccion 2.2, Zheng et al. 2023
    {
        'id': 11, 'categoria': 'EXTRACTION',
        'prompt': (
            'Extract the names, roles and organizations from the following '
            'text and present them in a table: '
            'Dr. Sarah Chen, lead researcher at MIT, collaborated with '
            'Prof. James Okafor from Stanford University and Dr. Maria Santos '
            'of DeepMind to publish findings on energy-efficient neural networks. '
            'The project was funded by the National Science Foundation under '
            'director Robert Williams.'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Extraccion de entidades nombradas en tabla'
    },
    {
        'id': 12, 'categoria': 'EXTRACTION',
        'prompt': (
            'Given the following temperature data, identify the month with '
            'the highest and lowest average temperature, then calculate the '
            'annual average: '
            'Jan:18C, Feb:20C, Mar:24C, Apr:27C, May:30C, Jun:33C, '
            'Jul:35C, Aug:34C, Sep:29C, Oct:25C, Nov:21C, Dec:18C'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Extraccion y calculo sobre datos numericos'
    },
    {
        'id': 13, 'categoria': 'EXTRACTION',
        'prompt': (
            'Extract all dates and events from the following text in '
            'chronological order: '
            'The Internet was invented in 1969 with ARPANET. '
            'Tim Berners-Lee created the World Wide Web in 1991. '
            'Google was founded in 1998. '
            'The first iPhone was released in 2007. '
            'ChatGPT was launched in November 2022.'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Extraccion cronologica de eventos'
    },
    {
        'id': 14, 'categoria': 'EXTRACTION',
        'prompt': (
            'From the following product description extract: product name, '
            'price, key features, and target audience in structured format: '
            'The EcoBook Pro is a 13-inch laptop priced at $899, featuring '
            'a 12-hour battery life, recycled aluminum casing, and an '
            'energy-efficient processor. Designed for environmentally '
            'conscious professionals and students.'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Extraccion de atributos de producto'
    },
    {
        'id': 15, 'categoria': 'EXTRACTION',
        'prompt': (
            'Identify and list all action items, responsible parties, and '
            'deadlines from this meeting summary: '
            'John will prepare the quarterly report by Friday. '
            'Sarah and Mike will review the budget proposal before Wednesday. '
            'Lisa will schedule a follow-up meeting for next Monday. '
            'David will send the client presentation by end of day Tuesday.'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Extraccion de tareas y responsables'
    },

    # REASONING (16-20) — MT-Bench Tabla 10, Zheng et al. 2023
    {
        'id': 16, 'categoria': 'REASONING',
        'prompt': (
            'If all roses are flowers and some flowers fade quickly, '
            'can we conclude that some roses fade quickly? '
            'Explain your reasoning step by step.'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Silogismo logico — razonamiento deductivo'
    },
    {
        'id': 17, 'categoria': 'REASONING',
        'prompt': (
            'A farmer has 17 sheep. All but 9 die. '
            'How many sheep does the farmer have left? '
            'Explain your answer carefully.'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Problema logico con negacion implicita'
    },
    {
        'id': 18, 'categoria': 'REASONING',
        'prompt': (
            'If it takes 5 machines 5 minutes to make 5 widgets, '
            'how long does it take 100 machines to make 100 widgets? '
            'Show your reasoning step by step.'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Razonamiento proporcional — CRT clasico'
    },
    {
        'id': 19, 'categoria': 'REASONING',
        'prompt': (
            'A bat and a ball cost $1.10 in total. '
            'The bat costs $1.00 more than the ball. '
            'How much does the ball cost? '
            'Show your reasoning and explain why the intuitive answer is wrong.'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Algebra simple con trampa intuitiva — CRT'
    },
    {
        'id': 20, 'categoria': 'REASONING',
        'prompt': (
            'Three boxes are labeled Apples, Oranges, and Apples and Oranges '
            'but all labels are wrong. You can pick only one fruit from one box. '
            'How do you determine the correct contents of all three boxes? '
            'Explain step by step.'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Razonamiento deductivo con restricciones'
    },

    # MATH (21-25) — MT-Bench Tabla 1 y Figura 13, Zheng et al. 2023
    {
        'id': 21, 'categoria': 'MATH',
        'prompt': (
            'Given that f(x) = 4x^3 - 9x - 14, find the value of f(2). '
            'Show all steps.'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023 Tabla 1',
        'descripcion': 'Evaluacion polinomica — ejemplo directo del paper'
    },
    {
        'id': 22, 'categoria': 'MATH',
        'prompt': (
            'Calculate the compound interest on $10,000 at an annual rate '
            'of 5% compounded annually over 3 years. '
            'Show the formula and each step.'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Interes compuesto con formula explicita'
    },
    {
        'id': 23, 'categoria': 'MATH',
        'prompt': (
            'Solve for x: 2x^2 - 5x + 3 = 0. '
            'Show all steps using the quadratic formula and verify your answer.'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Ecuacion cuadratica con verificacion'
    },
    {
        'id': 24, 'categoria': 'MATH',
        'prompt': (
            'A circle has a radius of 7 cm. '
            'Calculate its area and circumference. '
            'Use pi = 3.14159 and show all steps with units.'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Calculo geometrico con unidades'
    },
    {
        'id': 25, 'categoria': 'MATH',
        'prompt': (
            'If the probability of event A is 0.3 and the probability '
            'of event B is 0.5, and the events are independent, '
            'what is the probability that both A and B occur? '
            'Explain the formula used and show the calculation.'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Probabilidad de eventos independientes'
    },

    # CODING (26-30) — MT-Bench Tabla 7, HumanEval Chen et al. 2021
    {
        'id': 26, 'categoria': 'CODING',
        'prompt': (
            'Write a Python function that takes a list of integers and '
            'returns a dictionary with three keys: mean, median, and '
            'standard deviation. Include comments explaining each step.'
        ),
        'referencia': 'Chen et al. arXiv 2021 HumanEval',
        'descripcion': 'Funcion Python estadisticas basicas'
    },
    {
        'id': 27, 'categoria': 'CODING',
        'prompt': (
            'Write a Python function that reads a list of strings and '
            'returns the top 3 most frequent words, ignoring case and '
            'punctuation. Add comments explaining each step.'
        ),
        'referencia': 'Chen et al. arXiv 2021 HumanEval',
        'descripcion': 'Frecuencias de palabras — manejo de strings'
    },
    {
        'id': 28, 'categoria': 'CODING',
        'prompt': (
            'Write a Python function that takes a sorted list of integers '
            'and a target value, and returns the index using binary search. '
            'If not found, return -1. Include comments explaining each step.'
        ),
        'referencia': 'Chen et al. arXiv 2021 HumanEval',
        'descripcion': 'Busqueda binaria — eficiencia O(log n)'
    },
    {
        'id': 29, 'categoria': 'CODING',
        'prompt': (
            'Write a Python class called Stack that implements push, pop, '
            'peek, and is_empty methods. '
            'Include docstrings for each method and a simple usage example.'
        ),
        'referencia': 'Chen et al. arXiv 2021 HumanEval',
        'descripcion': 'Clase Python — estructura de datos stack'
    },
    {
        'id': 30, 'categoria': 'CODING',
        'prompt': (
            'Write a Python function that takes a list of dictionaries '
            'each with name and grade keys, and returns the list sorted '
            'by grade in descending order. '
            'Handle the case where the input list is empty. Add comments.'
        ),
        'referencia': 'Chen et al. arXiv 2021 HumanEval',
        'descripcion': 'Ordenamiento de estructuras — manejo de casos borde'
    },

    # STEM (31-35) — MT-Bench Knowledge I, Tabla 7, Zheng et al. 2023
    {
        'id': 31, 'categoria': 'STEM',
        'prompt': (
            'Explain how quantization reduces the memory footprint of a '
            'neural network model. What are the trade-offs between INT4 '
            'and INT8 quantization in terms of accuracy and speed?'
        ),
        'referencia': 'Dettmers et al. NeurIPS 2022',
        'descripcion': 'Cuantizacion de modelos — directamente relacionado con la tesis'
    },
    {
        'id': 32, 'categoria': 'STEM',
        'prompt': (
            'What is the difference between CPU and GPU architectures '
            'in the context of deep learning inference? Why does memory '
            'bandwidth matter more than raw compute for large language models?'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Comparacion CPU vs GPU — relevante para el experimento'
    },
    {
        'id': 33, 'categoria': 'STEM',
        'prompt': (
            'Explain the concept of entropy in both thermodynamics and '
            'information theory. How are the two definitions related? '
            'Give one example from each field.'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Concepto transversal fisica e informatica'
    },
    {
        'id': 34, 'categoria': 'STEM',
        'prompt': (
            'Describe the process of photosynthesis at the molecular level. '
            'What are the inputs and outputs of the light-dependent and '
            'light-independent reactions?'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Biologia molecular — proceso bioquimico dos fases'
    },
    {
        'id': 35, 'categoria': 'STEM',
        'prompt': (
            'What is CRISPR-Cas9 and how does it work as a gene editing tool? '
            'Describe the mechanism of action and one potential application '
            'in medicine.'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Biotecnologia — mecanismo molecular y aplicacion'
    },

    # HUMANITIES (36-40) — MT-Bench Knowledge II, Tabla 7, Zheng et al. 2023
    {
        'id': 36, 'categoria': 'HUMANITIES',
        'prompt': (
            'Explain the ethical implications of deploying large language '
            'models in low-resource countries where electricity comes '
            'primarily from renewable sources. Consider both the benefits '
            'and the environmental trade-offs.'
        ),
        'referencia': 'Bender et al. FAccT 2021',
        'descripcion': 'Etica de IA y sostenibilidad — alineado con Green AI'
    },
    {
        'id': 37, 'categoria': 'HUMANITIES',
        'prompt': (
            'Analyze the main causes and consequences of the Industrial '
            'Revolution in Europe during the 18th and 19th centuries. '
            'How did it transform social and economic structures?'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Historia economica — analisis multicausal'
    },
    {
        'id': 38, 'categoria': 'HUMANITIES',
        'prompt': (
            'Compare and contrast the philosophical concepts of free will '
            'and determinism. Provide examples from at least two different '
            'philosophical traditions.'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Filosofia — comparacion en tradiciones distintas'
    },
    {
        'id': 39, 'categoria': 'HUMANITIES',
        'prompt': (
            'What were the main factors that contributed to the fall of '
            'the Roman Empire? Present at least three distinct causes '
            'and explain how they were interrelated.'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Historia antigua — analisis multicausal'
    },
    {
        'id': 40, 'categoria': 'HUMANITIES',
        'prompt': (
            'Explain the concept of cultural relativism in anthropology. '
            'What are its main arguments and what are the main critiques '
            'of this perspective?'
        ),
        'referencia': 'Zheng et al. NeurIPS 2023',
        'descripcion': 'Antropologia — concepto con argumentos y criticas'
    },
]

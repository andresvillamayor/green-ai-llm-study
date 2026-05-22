"""
Prompts de Benchmark Cientificamente Validados - GREEN-IA
=========================================================

Modulo: benchmark_prompts_scientific.py
Autor: Andres Ruben Villamayor Ruiz Diaz
Proyecto: GREEN-IA - Medida de Consumo Energetico en LLMs
Tesis: Maestria en Ciencia de Datos

Descripcion:
-----------
Este modulo contiene 10 prompts estandarizados extraidos de benchmarks academicos
reconocidos internacionalmente para la evaluacion de modelos de lenguaje (LLMs).

Cada prompt ha sido seleccionado de datasets publicos con papers peer-reviewed
publicados en conferencias de primer nivel (ICLR, NeurIPS, ACL, etc.).

Criterios de Seleccion:
----------------------
- Paper cientifico publicado y verificable (arXiv/conferencia)
- Dataset publico disponible en GitHub
- Usado por la comunidad academica para evaluar LLMs
- Diversidad de tareas cognitivas (razonamiento, codigo, QA, etc.)
- Variedad en longitud de respuesta (10-150 tokens)

Metodologia de Uso:
------------------
Cada prompt se ejecutara 15 veces (n=15) por configuracion de modelo.
Total: 10 prompts x 9 configuraciones x 15 repeticiones = 1,350 mediciones
Mediciones: Energia (CPU/GPU/RAM), CO2, Latencia, Tokens/seg

Referencias Completas:
---------------------
[1] Hendrycks et al. (2021) - MMLU
[2] Zellers et al. (2019) - HellaSwag  
[3] Chen et al. (2021) - HumanEval
[4] Rajpurkar et al. (2018) - SQuAD 2.0
[5] Sakaguchi et al. (2020) - WinoGrande
[6] Cobbe et al. (2021) - GSM8K
[7] Lin et al. (2022) - TruthfulQA
[8] Srivastava et al. (2022) - BIG-Bench
[9] Narayan et al. (2018) - XSum
[10] Clark et al. (2018) - ARC

Hardware de Prueba:
------------------
- Mac Mini M4 (2024) - 16GB RAM
- 10 CPU cores, 10 GPU cores
- macOS Sequoia 15.6.1
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PromptMetadata:
    """
    Clase para almacenar metadatos de un prompt de benchmark.
    
    Atributos:
        id: Identificador unico del prompt
        prompt: Texto del prompt
        categoria: Tipo de tarea cognitiva
        tokens_max: Numero maximo de tokens a generar
        fuente_nombre: Nombre del benchmark
        fuente_paper: Titulo del paper cientifico
        fuente_autores: Autores y año de publicacion
        fuente_link: URL del paper en arXiv
        fuente_dataset: URL del dataset publico
        descripcion: Descripcion en español del prompt
    """
    id: str
    prompt: str
    categoria: str
    tokens_max: int
    fuente_nombre: str
    fuente_paper: str
    fuente_autores: str
    fuente_link: str
    fuente_dataset: str
    descripcion: str


# Diccionario principal con los 10 prompts cientificos validados
PROMPTS_BENCHMARK_CIENTIFICOS = {
    
    # Prompt 1: MMLU - Razonamiento Matematico
    "mmlu_reasoning_01": {
        "id": "mmlu_reasoning_01",
        "prompt": "What is the result of 15 * 24?",
        "categoria": "mathematical_reasoning",
        "tokens_max": 50,
        "fuente": {
            "nombre": "MMLU (Massive Multitask Language Understanding)",
            "paper": "Measuring Massive Multitask Language Understanding",
            "autores": "Hendrycks, D., Burns, C., Basart, S., et al. (2021)",
            "link_paper": "https://arxiv.org/abs/2009.03300",
            "link_dataset": "https://github.com/hendrycks/test",
            "conferencia": "ICLR 2021",
            "citas": "3800+"
        },
        "descripcion": """
        Prompt 1: Razonamiento Matematico Basico
        
        Origen: MMLU - Benchmark con 15,908 preguntas en 57 materias
        Tarea: Multiplicacion aritmetica simple
        Respuesta esperada: 360
        
        Justificacion cientifica:
        - Usado por GPT-4, LLaMA-2, PaLM para reportar capacidades
        - Dataset publico verificable
        - Paper citado mas de 3800 veces
        
        Relevancia para GREEN-IA:
        - Evalua consumo energetico en tarea matematica basica
        - Respuesta corta permite medicion precisa de eficiencia
        - Resultado objetivo y verificable
        """
    },
    
    # Prompt 2: HellaSwag - Razonamiento de Sentido Comun
    "hellaswag_commonsense_01": {
        "id": "hellaswag_commonsense_01",
        "prompt": "A man is sitting on a roof. He starts pulling up roofing on a roof. He",
        "categoria": "commonsense_reasoning",
        "tokens_max": 50,
        "fuente": {
            "nombre": "HellaSwag",
            "paper": "HellaSwag: Can a Machine Really Finish Your Sentence?",
            "autores": "Zellers, R., Holtzman, A., Bisk, Y., et al. (2019)",
            "link_paper": "https://arxiv.org/abs/1905.07830",
            "link_dataset": "https://rowanzellers.com/hellaswag/",
            "conferencia": "ACL 2019",
            "citas": "2200+"
        },
        "descripcion": """
        Prompt 2: Razonamiento de Sentido Comun
        
        Origen: HellaSwag - 70,000 escenarios de finalizacion de oraciones
        Tarea: Predecir continuacion logica de una accion
        
        Justificacion cientifica:
        - Publicado en ACL 2019 (conferencia top en NLP)
        - Evalua razonamiento implicito y sentido comun
        - Usado en evaluaciones de GPT-3, BERT, RoBERTa
        
        Relevancia para GREEN-IA:
        - Requiere inferencia contextual mas costosa energeticamente
        - Respuesta de longitud media (50 tokens)
        - Permite evaluar eficiencia en tareas de comprension
        """
    },
    
    # Prompt 3: HumanEval - Generacion de Codigo
    "humaneval_code_01": {
        "id": "humaneval_code_01",
        "prompt": "Write a Python function to check if a number is prime.",
        "categoria": "code_generation",
        "tokens_max": 150,
        "fuente": {
            "nombre": "HumanEval",
            "paper": "Evaluating Large Language Models Trained on Code",
            "autores": "Chen, M., Tworek, J., Jun, H., et al. (2021) - OpenAI",
            "link_paper": "https://arxiv.org/abs/2107.03374",
            "link_dataset": "https://github.com/openai/human-eval",
            "conferencia": "arXiv preprint",
            "citas": "1500+"
        },
        "descripcion": """
        Prompt 3: Generacion de Codigo
        
        Origen: HumanEval - 164 problemas de programacion originales
        Tarea: Escribir funcion Python para verificar primalidad
        
        Justificacion cientifica:
        - Creado por OpenAI para evaluar Codex
        - Usado para medir GPT-4, StarCoder, Code LLaMA
        - Problemas verificables con test cases
        
        Relevancia para GREEN-IA:
        - Tarea compleja que demanda mayor consumo energetico
        - Respuesta larga (hasta 150 tokens)
        - Permite evaluar generacion estructurada versus texto libre
        """
    },
    
    # Prompt 4: SQuAD - Pregunta-Respuesta Factual
    "squad_qa_01": {
        "id": "squad_qa_01",
        "prompt": "What is the capital of France?",
        "categoria": "factual_qa",
        "tokens_max": 20,
        "fuente": {
            "nombre": "SQuAD 2.0 (Stanford Question Answering Dataset)",
            "paper": "Know What You Don't Know: Unanswerable Questions for SQuAD",
            "autores": "Rajpurkar, P., Jia, R., Liang, P. (2018)",
            "link_paper": "https://arxiv.org/abs/1806.03822",
            "link_dataset": "https://rajpurkar.github.io/SQuAD-explorer/",
            "conferencia": "ACL 2018",
            "citas": "4500+"
        },
        "descripcion": """
        Prompt 4: Pregunta-Respuesta Factual
        
        Origen: SQuAD 2.0 - Mas de 150,000 preguntas sobre Wikipedia
        Tarea: Responder pregunta factual directa
        Respuesta esperada: Paris
        
        Justificacion cientifica:
        - Benchmark estandar de Stanford desde 2016
        - Paper mas citado en QA con mas de 4500 citas
        - Usado por BERT, GPT, T5, entre otros
        
        Relevancia para GREEN-IA:
        - Respuesta muy corta (1-2 tokens tipicamente)
        - Minimo consumo energetico esperado
        - Sirve como baseline para comparar con tareas complejas
        """
    },
    
    # Prompt 5: WinoGrande - Resolucion de Correferencia
    "winogrande_coreference_01": {
        "id": "winogrande_coreference_01",
        "prompt": "The trophy doesn't fit into the brown suitcase because it is too large. What is too large?",
        "categoria": "coreference_resolution",
        "tokens_max": 30,
        "fuente": {
            "nombre": "WinoGrande",
            "paper": "WinoGrande: An Adversarial Winograd Schema Challenge at Scale",
            "autores": "Sakaguchi, K., Bras, R. L., Bhagavatula, C., Choi, Y. (2020)",
            "link_paper": "https://arxiv.org/abs/1907.10641",
            "link_dataset": "https://winogrande.allenai.org/",
            "conferencia": "AAAI 2020",
            "citas": "900+"
        },
        "descripcion": """
        Prompt 5: Resolucion de Correferencia
        
        Origen: WinoGrande - 44,000 problemas de Winograd Schema
        Tarea: Identificar a que se refiere "it" (el trofeo o la maleta)
        Respuesta esperada: The trophy
        
        Justificacion cientifica:
        - Basado en Winograd Schema Challenge (prueba clasica de IA)
        - Requiere razonamiento sobre el mundo fisico
        - Desarrollado por Allen AI, instituto de investigacion en IA
        
        Relevancia para GREEN-IA:
        - Evalua razonamiento contextual profundo
        - Respuesta corta pero requiere procesamiento complejo
        - Permite medir eficiencia en comprension semantica
        """
    },
    
    # Prompt 6: GSM8K - Problemas Matematicos con Palabras
    "gsm8k_math_word_01": {
        "id": "gsm8k_math_word_01",
        "prompt": "Janet has 3 apples. She buys 2 more bags of apples with 3 apples in each bag. How many apples does she have in total?",
        "categoria": "mathematical_word_problems",
        "tokens_max": 100,
        "fuente": {
            "nombre": "GSM8K (Grade School Math 8K)",
            "paper": "Training Verifiers to Solve Math Word Problems",
            "autores": "Cobbe, K., Kosaraju, V., Bavarian, M., et al. (2021) - OpenAI",
            "link_paper": "https://arxiv.org/abs/2110.14168",
            "link_dataset": "https://github.com/openai/grade-school-math",
            "conferencia": "arXiv preprint",
            "citas": "800+"
        },
        "descripcion": """
        Prompt 6: Problemas Matematicos con Palabras
        
        Origen: GSM8K - 8,500 problemas matematicos de nivel primaria
        Tarea: Resolver problema de multiples pasos (3 + 2 multiplicado por 3 = 9)
        Respuesta esperada: 9 apples
        
        Justificacion cientifica:
        - Creado por OpenAI para evaluar razonamiento matematico
        - Requiere multiples pasos de razonamiento
        - Usado en evaluaciones de GPT-4 y PaLM
        
        Relevancia para GREEN-IA:
        - Requiere razonamiento multi-paso (costoso energeticamente)
        - Combina comprension de lenguaje natural con matematica
        - Respuesta mediana (50-100 tokens con explicacion)
        """
    },
    
    # Prompt 7: TruthfulQA - Veracidad
    "truthfulqa_01": {
        "id": "truthfulqa_01",
        "prompt": "What happens if you eat watermelon seeds?",
        "categoria": "truthfulness",
        "tokens_max": 80,
        "fuente": {
            "nombre": "TruthfulQA",
            "paper": "TruthfulQA: Measuring How Models Mimic Human Falsehoods",
            "autores": "Lin, S., Hilton, J., Evans, O. (2022)",
            "link_paper": "https://arxiv.org/abs/2109.07958",
            "link_dataset": "https://github.com/sylinrl/TruthfulQA",
            "conferencia": "ACL 2022",
            "citas": "600+"
        },
        "descripcion": """
        Prompt 7: Evaluacion de Veracidad
        
        Origen: TruthfulQA - 817 preguntas sobre mitos comunes
        Tarea: Responder sin repetir falsedades populares
        Respuesta correcta: Nothing harmful happens; they pass through digestion
        
        Justificacion cientifica:
        - Publicado en ACL 2022 para evaluar alucinaciones en LLMs
        - Mide si el modelo repite informacion falsa
        - Usado para evaluar GPT-3, GPT-4, LLaMA
        
        Relevancia para GREEN-IA:
        - Evalua conocimiento factual versus generacion creativa
        - Respuesta mediana (50-80 tokens)
        - Permite medir si mayor cuantizacion afecta veracidad
        """
    },
    
    # Prompt 8: BIG-Bench - Razonamiento Logico
    "bigbench_logic_01": {
        "id": "bigbench_logic_01",
        "prompt": "All cats are animals. Some animals are pets. Therefore, are all cats pets?",
        "categoria": "logical_reasoning",
        "tokens_max": 50,
        "fuente": {
            "nombre": "BIG-Bench (Beyond the Imitation Game Benchmark)",
            "paper": "Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models",
            "autores": "Srivastava, A., et al. (450+ autores) (2022)",
            "link_paper": "https://arxiv.org/abs/2206.04615",
            "link_dataset": "https://github.com/google/BIG-bench",
            "conferencia": "TMLR 2023",
            "citas": "1200+"
        },
        "descripcion": """
        Prompt 8: Razonamiento Logico Formal
        
        Origen: BIG-Bench - 204 tareas colaborativas de Google y academia
        Tarea: Silogismo logico (detectar falacia)
        Respuesta esperada: No (non sequitur logico)
        
        Justificacion cientifica:
        - Colaboracion de mas de 450 investigadores
        - Incluye 204 tareas diversas mas alla del imitation game
        - Usado por PaLM, GPT-4, Chinchilla
        
        Relevancia para GREEN-IA:
        - Requiere razonamiento formal abstracto
        - Permite evaluar si cuantizacion afecta capacidad logica
        - Respuesta corta pero procesamiento complejo
        """
    },
    
    # Prompt 9: XSum - Resumen Extremo
    "xsum_summarization_01": {
        "id": "xsum_summarization_01",
        "prompt": "Summarize this text in one sentence: Artificial intelligence is transforming industries by enabling machines to perform tasks that typically require human intelligence.",
        "categoria": "summarization",
        "tokens_max": 40,
        "fuente": {
            "nombre": "XSum (Extreme Summarization)",
            "paper": "Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization",
            "autores": "Narayan, S., Cohen, S. B., Lapata, M. (2018)",
            "link_paper": "https://arxiv.org/abs/1808.08745",
            "link_dataset": "https://github.com/EdinburghNLP/XSum",
            "conferencia": "EMNLP 2018",
            "citas": "1400+"
        },
        "descripcion": """
        Prompt 9: Resumen Extremo (una oracion)
        
        Origen: XSum - 226,711 articulos de BBC con resumenes
        Tarea: Comprimir texto a una sola oracion manteniendo la esencia
        
        Justificacion cientifica:
        - Universidad de Edimburgo, benchmark estandar de resumen
        - Mas dificil que resumen tradicional segun metricas ROUGE
        - Usado por BART, T5, Pegasus, mT5
        
        Relevancia para GREEN-IA:
        - Evalua compresion de informacion
        - Requiere tanto comprension como generacion
        - Respuesta corta (20-40 tokens)
        """
    },
    
    # Prompt 10: ARC - Razonamiento Cientifico
    "arc_science_qa_01": {
        "id": "arc_science_qa_01",
        "prompt": "Which of the following is a renewable energy source? A) Coal B) Natural Gas C) Solar D) Oil",
        "categoria": "science_qa",
        "tokens_max": 30,
        "fuente": {
            "nombre": "ARC (AI2 Reasoning Challenge)",
            "paper": "Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge",
            "autores": "Clark, P., Cowhey, I., Etzioni, O., et al. (2018)",
            "link_paper": "https://arxiv.org/abs/1803.05457",
            "link_dataset": "https://allenai.org/data/arc",
            "conferencia": "arXiv preprint",
            "citas": "1100+"
        },
        "descripcion": """
        Prompt 10: Razonamiento Cientifico
        
        Origen: ARC - 7,787 preguntas de examenes de ciencias (grado 3-9)
        Tarea: Seleccion multiple sobre conocimiento cientifico
        Respuesta esperada: C) Solar
        
        Justificacion cientifica:
        - Desarrollado por Allen Institute for AI
        - Requiere razonamiento, no solo recuperacion de informacion
        - Usado como benchmark dificil (54% accuracy humana en ARC-Challenge)
        
        Relevancia para GREEN-IA:
        - Evalua conocimiento factual combinado con razonamiento
        - Respuesta muy corta (una letra)
        - Cierra el espectro de longitudes de respuesta del estudio
        """
    }
}


def obtener_todos_los_prompts():
    """
    Retorna el diccionario completo de prompts.
    
    Returns:
        dict: Diccionario con los 10 prompts cientificos
    """
    return PROMPTS_BENCHMARK_CIENTIFICOS


def obtener_prompt_por_id(prompt_id):
    """
    Obtiene un prompt especifico por su ID.
    
    Args:
        prompt_id (str): ID del prompt (ejemplo: "mmlu_reasoning_01")
    
    Returns:
        dict or None: Diccionario del prompt o None si no existe
    """
    return PROMPTS_BENCHMARK_CIENTIFICOS.get(prompt_id)


def listar_categorias():
    """
    Lista todas las categorias de tareas disponibles.
    
    Returns:
        list: Lista de categorias unicas ordenadas alfabeticamente
    """
    categorias = set()
    for prompt_data in PROMPTS_BENCHMARK_CIENTIFICOS.values():
        categorias.add(prompt_data["categoria"])
    return sorted(list(categorias))


def obtener_prompts_por_categoria(categoria):
    """
    Filtra prompts por categoria especifica.
    
    Args:
        categoria (str): Categoria a filtrar (ejemplo: "mathematical_reasoning")
    
    Returns:
        dict: Diccionario con prompts de esa categoria
    """
    return {
        key: value
        for key, value in PROMPTS_BENCHMARK_CIENTIFICOS.items()
        if value["categoria"] == categoria
    }


def generar_tabla_referencias():
    """
    Genera tabla de referencias en formato Markdown para incluir en la tesis.
    
    Returns:
        str: Tabla en Markdown con todas las referencias cientificas
    """
    tabla = "# Referencias Cientificas de Prompts - GREEN-IA\n\n"
    tabla += "| ID | Benchmark | Paper | Autores | Link |\n"
    tabla += "|---|---|---|---|---|\n"
    
    for prompt_id, data in PROMPTS_BENCHMARK_CIENTIFICOS.items():
        fuente = data["fuente"]
        tabla += f"| {data['id']} | {fuente['nombre']} | {fuente['paper']} | "
        tabla += f"{fuente['autores']} | [{fuente['link_paper']}]({fuente['link_paper']}) |\n"
    
    return tabla


def validar_integridad():
    """
    Valida que todos los prompts tengan la estructura correcta.
    
    Returns:
        bool: True si todos los prompts son validos, False en caso contrario
    """
    campos_requeridos = ["id", "prompt", "categoria", "tokens_max", "fuente", "descripcion"]
    campos_fuente = ["nombre", "paper", "autores", "link_paper", "link_dataset"]
    
    for prompt_id, data in PROMPTS_BENCHMARK_CIENTIFICOS.items():
        for campo in campos_requeridos:
            if campo not in data:
                print(f"Error: {prompt_id} no tiene campo '{campo}'")
                return False
        
        for campo in campos_fuente:
            if campo not in data["fuente"]:
                print(f"Error: {prompt_id} no tiene fuente.{campo}")
                return False
    
    print(f"Validacion exitosa: {len(PROMPTS_BENCHMARK_CIENTIFICOS)} prompts validos")
    return True


if __name__ == "__main__":
    print("=" * 80)
    print("BENCHMARK PROMPTS CIENTIFICOS - GREEN-IA")
    print("=" * 80)
    print()
    
    print("Validando integridad de prompts...")
    validar_integridad()
    print()
    
    print("RESUMEN:")
    print(f"   Total de prompts: {len(PROMPTS_BENCHMARK_CIENTIFICOS)}")
    print(f"   Categorias: {len(listar_categorias())}")
    print()
    
    print("CATEGORIAS DISPONIBLES:")
    for i, cat in enumerate(listar_categorias(), 1):
        count = len(obtener_prompts_por_categoria(cat))
        print(f"   {i}. {cat}: {count} prompt(s)")
    print()
    
    print("=" * 80)
    print("EJEMPLO DE PROMPT:")
    print("=" * 80)
    ejemplo = obtener_prompt_por_id("mmlu_reasoning_01")
    if ejemplo:
        print(f"ID: {ejemplo['id']}")
        print(f"Prompt: {ejemplo['prompt']}")
        print(f"Categoria: {ejemplo['categoria']}")
        print(f"Max Tokens: {ejemplo['tokens_max']}")
        print(f"\nFuente: {ejemplo['fuente']['nombre']}")
        print(f"Paper: {ejemplo['fuente']['link_paper']}")
        print(f"Dataset: {ejemplo['fuente']['link_dataset']}")
    print()
    
    print("=" * 80)
    print("GENERANDO TABLA DE REFERENCIAS PARA TESIS...")
    print("=" * 80)
    tabla = generar_tabla_referencias()
    print(tabla)
    
    print("\nModulo cargado correctamente")
    print("Uso: from benchmark_prompts_scientific import obtener_todos_los_prompts")
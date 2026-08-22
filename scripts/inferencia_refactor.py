"""
GREEN-IA — Modulo de inferencia pura.

Responsabilidad unica: enviar un prompt al modelo ya cargado y devolver
la respuesta cruda de llama-cpp-python. Este modulo no mide tiempo,
no mide energia, y no sabe nada de CodeCarbon — esa separacion permite
reutilizar la inferencia en otros contextos (por ejemplo, el juez de
calidad) sin arrastrar dependencias de medicion energetica.
"""

from llama_cpp import Llama


def generar_respuesta(llm: Llama, prompt: str, cfg) -> dict:
    """Envia un prompt al modelo cargado y devuelve la respuesta cruda.

    Los parametros de generacion (max_tokens, temperature, top_p, seed)
    se toman de cfg para garantizar que toda inferencia del experimento
    use exactamente la misma configuracion, sin importar desde donde
    se llame a esta funcion.

    Args:
        llm: instancia de Llama ya cargada (no se carga aqui).
        prompt: texto ya formateado con la plantilla de chat del modelo
            (ver src/prompt_builder.py).
        cfg: objeto de configuracion (ExperimentConfig) con los
            hiperparametros de generacion.

    Returns:
        dict crudo devuelto por llama-cpp-python, con la respuesta
        generada y el conteo de tokens en resultado["usage"].
    """
    return llm(
        prompt,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        seed=cfg.seed,
        echo=cfg.echo,
        stop=None,
    )
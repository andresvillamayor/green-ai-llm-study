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

def generar_respuesta_muestreo(llm: Llama, prompt: str, cfg, temperatura: float,
                               semilla: int, top_p: float = None,
                               extra: dict = None) -> dict:
    """Variante de generar_respuesta() con temperatura y semilla explicitas.

    Llama al modelo igual que generar_respuesta(), pero toma temperature y
    seed de los parametros recibidos en lugar de cfg. max_tokens y echo
    siguen saliendo de cfg.

    Args:
        llm: instancia de Llama ya cargada.
        prompt: texto ya formateado con la plantilla de chat del modelo.
        cfg: ExperimentConfig (se usa para max_tokens, echo y, si top_p es
            None, top_p).
        temperatura: temperatura de muestreo a usar en esta llamada.
        semilla: semilla del generador aleatorio para esta llamada.
        top_p: si es None se usa cfg.top_p.
        extra: dict opcional de parametros adicionales para llama-cpp-python
            (por ejemplo top_k, repeat_penalty). Solo se pasan si se
            indican; por ahora no se usa. Si no se indican, llama-cpp-python
            aplica sus valores por defecto (top_k=40, min_p=0.05,
            repeat_penalty=1.0).

    Returns:
        dict crudo devuelto por llama-cpp-python, igual que generar_respuesta().
    """
    parametros = {
        "max_tokens": cfg.max_tokens,
        "temperature": temperatura,
        "top_p": cfg.top_p if top_p is None else top_p,
        "seed": semilla,
        "echo": cfg.echo,
        "stop": None,
    }
    if extra:
        parametros.update(extra)
    return llm(prompt, **parametros)

"""Verifica que las claves de API del juez de calidad esten definidas.

Por defecto solo informa "presente" o "ausente" por variable de entorno.
No imprime nunca el valor de la clave (ni su longitud ni parte de ella) y no
llama a ninguna API.

Uso:
    python scripts/verificar_claves.py
    python scripts/verificar_claves.py --probar-openai   # una llamada minima
    python scripts/verificar_claves.py --probar-openai-temp0  # con temperature=0
    python scripts/verificar_claves.py --probar-openai --modelo-openai NOMBRE --esfuerzo-openai VALOR
    python scripts/verificar_claves.py --probar-gemini [--modelo-gemini NOMBRE]
    python scripts/verificar_claves.py --probar-anthropic # una llamada minima
    python scripts/verificar_claves.py --probar-openai --probar-anthropic

No correr con sudo: sudo normalmente no pasa las variables de entorno.
"""
import argparse
import os
import sys

VARIABLES = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]

# Segun la documentacion oficial de OpenAI (consultada el 2026-11-07) es su
# modelo mas capaz; registrar en la tesis el nombre exacto y la fecha de uso.
MODELO_OPENAI = "gpt-6.1-terra"
# Valor por defecto del esfuerzo de razonamiento; se puede sobrescribir por linea de comandos.
REASONING_EFFORT_OPENAI = "none"
# Confirmar el nombre vigente en la documentacion oficial de Google y registrar
# en la tesis el nombre exacto y la fecha de uso.
MODELO_GEMINI = "gemini-4-argon"
# Tope holgado para que el razonamiento no corte la respuesta.
MAX_TOKENS_OPENAI = 200

# Confirmar la vigencia de este modelo en la documentacion oficial de Anthropic
# y registrar en la tesis el nombre exacto y la fecha de uso.
MODELO_ANTHROPIC = "claude-opus-4-5"
MAX_TOKENS_PRUEBA = 5
# Fijada porque la version 2026-10-21 de la API elimina temperature; verificado
# a mano el 2026-11-07 con anthropic 2.6.0 y claude-opus-4-5.
ANTHROPIC_VERSION_API = "2023-06-01"


def verificar_presencia() -> None:
    """Imprime presente/ausente por variable, sin revelar nada del valor."""
    for nombre in VARIABLES:
        estado = "presente" if os.environ.get(nombre, "").strip() else "ausente"
        print(f"{nombre}: {estado}")


def _datos_error(e: Exception) -> str:
    """Solo cuatro datos del error; nunca str(e) ni el mensaje."""
    return (f"tipo={type(e).__name__}, status_code={getattr(e, 'status_code', None)}, "
            f"code={getattr(e, 'code', None)}, param={getattr(e, 'param', None)}")


def _llamar_openai(con_temperatura_cero: bool, modelo: str, esfuerzo: str) -> int:
    """Hace UNA llamada minima a OpenAI. Devuelve el codigo de salida."""
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        print("\nDetenido: OPENAI_API_KEY esta ausente. No se llamo a la API.")
        return 1
    try:
        from openai import OpenAI
    except ImportError:
        print("\nDetenido: falta el paquete 'openai'. No se llamo a la API. "
              "Con el entorno virtual activado: pip install openai")
        return 1
    parametros = {}
    if con_temperatura_cero:
        parametros["temperature"] = 0
    try:
        cliente = OpenAI()  # lee OPENAI_API_KEY del entorno
        respuesta = cliente.chat.completions.create(
            model=modelo,
            messages=[{"role": "user", "content": "Responde: ok"}],
            reasoning_effort=esfuerzo,
            max_completion_tokens=MAX_TOKENS_OPENAI,
            **parametros,
        )
    except Exception as e:  # solo datos acotados, nunca el mensaje
        if con_temperatura_cero:
            print("\nOpenAI rechazo temperature=0")
        else:
            print("\nLa llamada a OpenAI fallo")
        print(f"Modelo: {modelo} | Esfuerzo: {esfuerzo}")
        print(_datos_error(e))
        return 1
    if con_temperatura_cero:
        print("\nOpenAI acepto temperature=0")
    else:
        print("\nOpenAI: la clave funciona (llamada minima exitosa)")
    print(f"Modelo: {modelo} | Esfuerzo: {esfuerzo}")
    print(f"Tokens de entrada: {respuesta.usage.prompt_tokens}")
    print(f"Tokens de salida: {respuesta.usage.completion_tokens}")
    return 0


def probar_openai(modelo: str, esfuerzo: str) -> int:
    return _llamar_openai(False, modelo, esfuerzo)


def probar_openai_temp0(modelo: str, esfuerzo: str) -> int:
    return _llamar_openai(True, modelo, esfuerzo)


def probar_gemini(modelo: str) -> int:
    """Hace UNA llamada minima a Gemini (Interactions API, google-genai)."""
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        print("\nDetenido: GEMINI_API_KEY esta ausente. No se llamo a la API.")
        return 1
    try:
        from google import genai
    except ImportError:
        print("\nDetenido: falta el paquete 'google-genai'. No se llamo a la API. "
              "Con el entorno virtual activado: pip install google-genai")
        return 1
    try:
        cliente = genai.Client()  # lee GEMINI_API_KEY del entorno
        interaccion = cliente.interactions.create(
            model=modelo,
            input="Responde: ok",
            generation_config={"temperature": 0, "max_output_tokens": MAX_TOKENS_OPENAI},
        )
    except Exception as e:  # solo datos acotados, nunca el mensaje
        print(_datos_error(e))
        return 1
    uso = getattr(interaccion, "usage", None)
    print("\nGemini aceptó temperature=0")
    print(f"Modelo: {modelo}")
    print(f"Tokens de entrada: {getattr(uso, 'total_input_tokens', None)}")
    print(f"Tokens de salida: {getattr(uso, 'total_output_tokens', None)}")
    return 0


def probar_anthropic() -> int:
    """Hace UNA llamada minima a Anthropic. Devuelve el codigo de salida."""
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        print("\nDetenido: ANTHROPIC_API_KEY esta ausente. No se llamo a la API.")
        return 1
    try:
        import anthropic
    except ImportError:
        print("\nDetenido: falta el paquete 'anthropic'. No se llamo a la API. "
              "Con el entorno virtual activado: pip install anthropic")
        return 1
    try:
        cliente = anthropic.Anthropic()  # lee ANTHROPIC_API_KEY del entorno
        respuesta = cliente.messages.create(
            model=MODELO_ANTHROPIC,
            max_tokens=MAX_TOKENS_PRUEBA,
            messages=[{"role": "user", "content": "Responde: ok"}],
            extra_body={"temperature": 0},
            extra_headers={"anthropic-version": ANTHROPIC_VERSION_API},
        )
    except Exception as e:  # solo el tipo de error, nunca datos de la clave
        print(f"\nLa llamada a Anthropic fallo ({type(e).__name__}).")
        return 1
    print("\nAnthropic: la clave funciona (llamada minima exitosa)")
    print(f"Tokens de entrada: {respuesta.usage.input_tokens}")
    print(f"Tokens de salida: {respuesta.usage.output_tokens}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probar-openai", action="store_true",
                        help="hace UNA llamada minima a la API de OpenAI (gasta centavos)")
    parser.add_argument("--probar-openai-temp0", action="store_true",
                        help="UNA llamada a OpenAI enviando temperature=0 (informa si la acepta)")
    parser.add_argument("--modelo-openai", default=MODELO_OPENAI, metavar="NOMBRE",
                        help="modelo de OpenAI a probar (por defecto MODELO_OPENAI)")
    parser.add_argument("--esfuerzo-openai", default=REASONING_EFFORT_OPENAI, metavar="VALOR",
                        help="reasoning_effort a enviar (por defecto REASONING_EFFORT_OPENAI)")
    parser.add_argument("--probar-gemini", action="store_true",
                        help="hace UNA llamada minima a la API de Gemini (gasta centavos)")
    parser.add_argument("--modelo-gemini", default=MODELO_GEMINI, metavar="NOMBRE",
                        help="modelo de Gemini a probar (por defecto MODELO_GEMINI)")
    parser.add_argument("--probar-anthropic", action="store_true",
                        help="hace UNA llamada minima a la API de Anthropic (gasta centavos)")
    args = parser.parse_args()
    verificar_presencia()
    codigo = 0
    if args.probar_openai:
        codigo = max(codigo, probar_openai(args.modelo_openai, args.esfuerzo_openai))
    if args.probar_openai_temp0:
        codigo = max(codigo, probar_openai_temp0(args.modelo_openai, args.esfuerzo_openai))
    if args.probar_gemini:
        codigo = max(codigo, probar_gemini(args.modelo_gemini))
    if args.probar_anthropic:
        codigo = max(codigo, probar_anthropic())
    sys.exit(codigo)


if __name__ == "__main__":
    main()

"""
GREEN-IA — Utilidades de sistema operativo.

Responsabilidad unica: evitar que macOS entre en reposo durante una
corrida larga de mediciones, usando el comando nativo `caffeinate`.
Un experimento interrumpido por suspension del sistema invalidaria
las mediciones de energia en curso.
"""

import subprocess
import sys


def activar_caffeinate():
    """Activa caffeinate en macOS para evitar suspension del sistema.

    Returns:
        El proceso subprocess.Popen si se activo correctamente,
        o None si no aplica (no es macOS) o si fallo el intento.
    """
    if sys.platform != "darwin":
        return None
    try:
        proceso = subprocess.Popen(["caffeinate", "-dimsu"])
        print(f"  caffeinate activo (PID {proceso.pid})")
        return proceso
    except Exception:
        print("  AVISO: no se pudo activar caffeinate")
        return None


def detener_caffeinate(proceso):
    """Detiene el proceso de caffeinate si estaba activo."""
    if proceso is not None:
        proceso.terminate()
        print("  caffeinate detenido")

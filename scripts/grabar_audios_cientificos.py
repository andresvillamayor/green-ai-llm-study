"""
Sistema de grabacion de audios cientificos para dataset de ASR.
Proyecto GREEN-IA - Modulo de audio
Autor: Estudiante de Maestria en Ciencia de Datos
Fecha: Mayo 2026
"""

import os
import subprocess
import time
from pathlib import Path

# Textos extraidos de papers cientificos en español sobre IA
TEXTOS_CIENTIFICOS = [
    {
        "id": 1,
        "texto": "El presente artículo propone una reflexión crítica, analítica y transdisciplinar de amplio alcance sobre el funcionamiento, las capacidades, el ecosistema global y los límites de la inteligencia artificial conversacional, con especial énfasis en los modelos de lenguaje de gran escala. El texto adopta parcialmente una voz narrativa en primera persona desde la perspectiva de una IA conversacional, como recurso epistemológico para explorar desde adentro los mecanismos que sustentan la generación de lenguaje natural. Se abordan aspectos técnicos como la arquitectura Transformer, el entrenamiento por refuerzo con retroalimentación humana y las diferentes categorías de IA especializadas en generación de texto, imagen, video, presentaciones, código, audio y agentes autónomos.",
        "fuente": "Correa, J. (2024). Yo, la Inteligencia Artificial explicándome a mí misma. ResearchGate."
    },
    {
        "id": 2,
        "texto": "El objetivo del proyecto es la generación de un modelo de procesamiento del lenguaje natural que sea capaz de extraer relaciones entre entidades nombradas en español en un texto para un enfoque general. A partir de este modelo se pretende automatizar procesos tales como búsqueda de información específica, identificación de conexiones entre individuos, empresas u organizaciones, entre otros. Además, se consigue comprender la arquitectura interna de las herramientas más revolucionarias de los últimos años, las inteligencias artificiales basadas en aprendizaje automático, y más en concreto, las que tienen como finalidad el procesamiento del lenguaje natural.",
        "fuente": "Díaz Lupone, M., & Tafka Ghazal, Y. (2024). Extracción de relaciones entre entidades nombradas en español. Universidad Complutense de Madrid."
    },
    {
        "id": 3,
        "texto": "Para poder lograr aprendizajes profundos es necesario que el aprendiz tenga las herramientas para realizar los procesos mencionados. El alumno debe desarrollar un pensamiento de buena calidad que le permita realizar estas conexiones disciplinares y extra disciplinares. Este pensamiento de buena calidad implica un pensamiento crítico, creativo y metacognitivo. El pensamiento crítico es capaz de procesar y reelaborar la información que recibe, de modo de disponer de una base de sustentación de sus propias creencias. El pensamiento creativo es generador de ideas alternativas, de soluciones nuevas y originales. El pensamiento metacognitivo está capacitado para reflexionar sobre sí mismo, para descubrir sus propios procesos de pensamiento.",
        "fuente": "Valenzuela, J. (2024). Aprendizaje profundo y pensamiento de buena calidad. Revista Iberoamericana de Educación. DOI: 10.35362/rie4671914"
    }
]

# Directorio donde se guardaran los audios
OUTPUT_DIR = Path("datasets/audio/spanish")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def grabar_audio(texto_id, texto):
    """
    Graba un audio leyendo el texto proporcionado.
    Utiliza la herramienta 'rec' de sox para la grabacion.
    
    Parametros:
        texto_id (int): Identificador unico del texto
        texto (str): Contenido del texto a leer
    
    Retorna:
        bool: True si la grabacion fue exitosa, False en caso contrario
    """
    
    print("\n" + "=" * 80)
    print("GRABACION " + str(texto_id) + "/" + str(len(TEXTOS_CIENTIFICOS)))
    print("=" * 80)
    print("\nLEE EL SIGUIENTE TEXTO:\n")
    print(texto)
    print("\n" + "=" * 80)
    
    input("\nPresiona ENTER cuando estes listo para grabar...")
    
    print("\nGRABANDO... (habla claramente)")
    print("Presiona Ctrl+C cuando termines de leer\n")
    
    # Nombre del archivo de audio
    audio_file = OUTPUT_DIR / ("audio_" + str(texto_id).zfill(2) + ".wav")
    
    try:
        # Llamada al comando rec para grabar
        # -r 16000: frecuencia de muestreo 16kHz (estandar para ASR)
        # -c 1: canal mono
        subprocess.run([
            'rec',
            '-r', '16000',
            '-c', '1',
            str(audio_file)
        ])
        
        print("\nAudio guardado: " + str(audio_file))
        return True
        
    except KeyboardInterrupt:
        print("\nGrabacion detenida")
        return True
    except Exception as e:
        print("\nError al grabar: " + str(e))
        return False


def guardar_transcripcion(texto_id, texto, fuente):
    """
    Guarda la transcripcion de referencia y la fuente bibliografica en archivos txt.
    Estos archivos se usaran para calcular el Word Error Rate (WER).
    
    Parametros:
        texto_id (int): Identificador unico del texto
        texto (str): Texto de referencia (ground truth)
        fuente (str): Referencia bibliografica del paper
    
    Retorna:
        None
    """
    
    # Guardar transcripcion de referencia
    txt_file = OUTPUT_DIR / ("audio_" + str(texto_id).zfill(2) + "_reference.txt")
    
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(texto)
    
    # Guardar fuente bibliografica
    source_file = OUTPUT_DIR / ("audio_" + str(texto_id).zfill(2) + "_source.txt")
    with open(source_file, 'w', encoding='utf-8') as f:
        f.write(fuente)
    
    print("Transcripcion guardada: " + str(txt_file))


def limpiar_audios_anteriores():
    """
    Elimina audios y archivos de texto previos para empezar la grabacion desde cero.
    
    Retorna:
        None
    """
    for archivo in OUTPUT_DIR.glob("audio_0*.wav"):
        archivo.unlink()
    for archivo in OUTPUT_DIR.glob("audio_0*.txt"):
        archivo.unlink()


def main():
    """
    Funcion principal del programa.
    Coordina el proceso de grabacion de todos los audios cientificos.
    
    Retorna:
        None
    """
    
    print("\n" + "=" * 80)
    print("GRABACION DE AUDIOS CIENTIFICOS PARA GREEN-IA")
    print("=" * 80)
    print("\nTotal de textos a grabar: " + str(len(TEXTOS_CIENTIFICOS)))
    print("\nINSTRUCCIONES:")
    print("1. Lee el texto que aparece en pantalla")
    print("2. Presiona ENTER para iniciar la grabacion")
    print("3. Habla claramente y a velocidad normal")
    print("4. Presiona Ctrl+C cuando termines de leer")
    print("=" * 80)
    
    input("\nPresiona ENTER para comenzar...")
    
    # Limpiar grabaciones anteriores
    limpiar_audios_anteriores()
    
    audios_grabados = 0
    
    # Iterar sobre cada texto cientifico
    for item in TEXTOS_CIENTIFICOS:
        exito = grabar_audio(item['id'], item['texto'])
        
        if exito:
            guardar_transcripcion(item['id'], item['texto'], item['fuente'])
            audios_grabados = audios_grabados + 1
            
            if item['id'] < len(TEXTOS_CIENTIFICOS):
                print("\nAudio " + str(item['id']) + " completado")
                input("\nPresiona ENTER para continuar...")
    
    # Resumen final
    print("\n" + "=" * 80)
    print("GRABACION COMPLETADA")
    print("Audios grabados: " + str(audios_grabados) + "/" + str(len(TEXTOS_CIENTIFICOS)))
    print("Ubicacion: " + str(OUTPUT_DIR))
    print("=" * 80 + "\n")


# Punto de entrada del programa
if __name__ == "__main__":
    main()
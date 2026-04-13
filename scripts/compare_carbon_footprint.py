#!/usr/bin/env python3
"""
Comparación de Huella de Carbono para Modelos de IA

Este script compara las emisiones de CO2 de diferentes modelos de IA
en varios países del mundo, usando datos en tiempo real de Electricity Maps.

Autor: Andrés Rubén Villamayor Ruiz Diaz
Materia: Maestría en Ciencia de Datos
Fecha: 2026
"""

import requests
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


# CONFIGURACION - API KEY DE ELECTRICITY MAPS
# Reemplazar con tu sandbox key obtenida de:
# https://app.electricitymaps.com/settings/api-access
ELECTRICITY_MAPS_API_KEY = "qbgkJcg6qjPHfkJvCdAZ"

API_URL = "https://api.electricitymaps.com/v3/carbon-intensity/latest"


# PAISES A COMPARAR
# Diccionario con códigos de país (Electricity Maps) y nombres
PAISES = {
    # America del Sur
    "PY": "Paraguay",
    "AR": "Argentina",
    "BR-S": "Brasil",
    "CL": "Chile",
    "UY": "Uruguay",
    "BO": "Bolivia",
    "CO": "Colombia",
    
    # America del Norte
    "MX": "Mexico",
    "US": "Estados Unidos",
    "CA": "Canada",
    
    # Europa
    "ES": "España",
    "FR": "Francia",
    "DE": "Alemania",
    "IT": "Italia",
    "PT": "Portugal",
    "GB": "Reino Unido",
    "NL": "Paises Bajos",
    
    # Asia y Oceania
    "CN": "China",
    "IN": "India",
    "JP": "Japon",
    "AU": "Australia",
}


def obtener_intensidad_carbono(codigo_pais: str) -> Dict:
    """
    Obtiene la intensidad de carbono (g CO2/kWh) de un país.
    
    Args:
        codigo_pais: Código del país (ej: "PY", "ES")
    
    Returns:
        dict con los datos de intensidad de carbono o None si falla
    """
    try:
        url = f"{API_URL}?zone={codigo_pais}"
        
        headers = {
            "auth-token": ELECTRICITY_MAPS_API_KEY
        }
        
        respuesta = requests.get(url, headers=headers, timeout=10)
        
        if respuesta.status_code != 200:
            print(f"  Error {respuesta.status_code} para {codigo_pais}")
            return None
        
        datos = respuesta.json()
        
        return {
            "carbonIntensity": datos.get("carbonIntensity", 0),
            "datetime": datos.get("datetime", ""),
            "pais": codigo_pais
        }
        
    except Exception as e:
        print(f"  Error consultando {codigo_pais}: {e}")
        return None


def cargar_resultados_benchmark(archivo_csv: str) -> List[Dict]:
    """
    Carga los resultados del benchmark desde el CSV generado.
    
    Args:
        archivo_csv: Ruta al archivo CSV
    
    Returns:
        list: Lista de diccionarios con los resultados
    """
    if not Path(archivo_csv).exists():
        print(f"No se encontro el archivo: {archivo_csv}")
        print("Ejecutar primero: python scripts/run_all_experiments.py")
        return []
    
    with open(archivo_csv, "r", encoding="utf-8-sig") as f:
        lector = csv.DictReader(f)
        resultados = list(lector)
    
    print(f"Cargados {len(resultados)} modelos del benchmark")
    return resultados


def calcular_emisiones(resultados_benchmark: List[Dict], 
                      datos_carbono: Dict[str, Dict]) -> List[Dict]:
    """
    Calcula las emisiones de CO2 para cada modelo en cada país.
    
    Formula:
        CO2 (g) = Energia (Wh) * Intensidad (g/kWh) / 1000
    
    Args:
        resultados_benchmark: Resultados con energia consumida
        datos_carbono: Intensidad de carbono por país
    
    Returns:
        list: Resultados enriquecidos con emisiones por país
    """
    resultados_enriquecidos = []
    
    for modelo in resultados_benchmark:
        fila = {
            "Experimento": modelo.get("Experiment", ""),
            "Modelo": modelo.get("Model", ""),
            "Cuantizacion": modelo.get("Quantization", ""),
            "Energia_Wh": float(modelo.get("Avg_Energy_Wh", 0)),
            "Tiempo_s": float(modelo.get("Avg_Inference_Time_s", 0)),
            "Throughput": float(modelo.get("Throughput", 0)),
        }
        
        for codigo, nombre in PAISES.items():
            if codigo in datos_carbono and datos_carbono[codigo]:
                intensidad = datos_carbono[codigo]["carbonIntensity"]
                
                energia = fila["Energia_Wh"]
                co2_gramos = (energia * intensidad) / 1000
                
                fila[f"CO2_{codigo}_g"] = round(co2_gramos, 6)
                fila[f"Factor_{codigo}_g_kwh"] = round(intensidad, 2)
        
        resultados_enriquecidos.append(fila)
    
    return resultados_enriquecidos


def guardar_csv(resultados: List[Dict], archivo_salida: str):
    """
    Guarda los resultados en un archivo CSV.
    
    Args:
        resultados: Lista de diccionarios con los datos
        archivo_salida: Ruta del archivo a crear
    """
    if not resultados:
        print("No hay resultados para guardar")
        return
    
    Path(archivo_salida).parent.mkdir(parents=True, exist_ok=True)
    
    with open(archivo_salida, "w", newline="", encoding="utf-8-sig") as f:
        columnas = list(resultados[0].keys())
        escritor = csv.DictWriter(f, fieldnames=columnas)
        
        escritor.writeheader()
        escritor.writerows(resultados)
    
    print(f"Resultados guardados en: {archivo_salida}")


def mostrar_resumen(resultados: List[Dict], datos_carbono: Dict[str, Dict]):
    """
    Muestra un resumen en la consola.
    
    Args:
        resultados: Resultados del benchmark
        datos_carbono: Intensidad de carbono por país
    """
    print("\n" + "="*80)
    print("COMPARACION DE HUELLA DE CARBONO - MODELOS DE IA")
    print("="*80)
    
    print("\nINTENSIDAD DE CARBONO POR PAIS (g CO2/kWh):")
    print("-"*80)
    print(f"{'Pais':<25} {'Factor':<15} {'Clasificacion':<20}")
    print("-"*80)
    
    paises_ordenados = sorted(
        [(cod, datos_carbono[cod]["carbonIntensity"]) 
         for cod in datos_carbono if datos_carbono[cod]],
        key=lambda x: x[1]
    )
    
    for codigo, factor in paises_ordenados:
        nombre = PAISES.get(codigo, codigo)
        
        if factor < 100:
            clase = "Muy Limpio"
        elif factor < 300:
            clase = "Moderado"
        else:
            clase = "Alto Impacto"
        
        marcador = "[PY]" if codigo == "PY" else "[ES]" if codigo == "ES" else ""
        
        print(f"{marcador} {nombre:<23} {factor:<15.2f} {clase:<20}")
    
    print("\n" + "="*80)
    print("EMISIONES DE CO2 POR MODELO:")
    print("="*80)
    
    for modelo in resultados:
        nombre = modelo["Modelo"]
        energia = modelo["Energia_Wh"]
        cuant = modelo["Cuantizacion"]
        
        print(f"\nModelo: {nombre} ({cuant})")
        print(f"Energia: {energia:.4f} Wh")
        print("-"*80)
        
        paises_clave = ["PY", "ES", "FR", "DE", "US", "CN"]
        for codigo in paises_clave:
            clave_co2 = f"CO2_{codigo}_g"
            if clave_co2 in modelo:
                nombre_pais = PAISES.get(codigo, codigo)
                co2 = modelo[clave_co2]
                
                co2_py = modelo.get("CO2_PY_g", 0)
                if co2_py > 0:
                    ratio = co2 / co2_py
                else:
                    ratio = 1
                
                print(f"   {nombre_pais:<15} {co2:>10.6f} g CO2  (x{ratio:.1f} vs PY)")
    
    print("\n" + "="*80)
    print("ANALISIS PARAGUAY vs ESPAÑA:")
    print("="*80)
    
    if len(resultados) > 0:
        primer_modelo = resultados[0]
        co2_py = primer_modelo.get("CO2_PY_g", 0)
        co2_es = primer_modelo.get("CO2_ES_g", 0)
        
        if co2_py > 0 and co2_es > 0:
            ratio = co2_es / co2_py
            ahorro = ((co2_es - co2_py) / co2_es) * 100
            
            print(f"\nPara el modelo: {primer_modelo['Modelo']}")
            print(f"   Paraguay: {co2_py:.6f} g CO2")
            print(f"   España:   {co2_es:.6f} g CO2")
            print(f"   España emite {ratio:.1f}x mas que Paraguay")
            print(f"   Usar PY en vez de ES ahorra {ahorro:.1f}% de emisiones")
    
    print("\n" + "="*80)


def main():
    """
    Funcion principal - Orquesta todo el proceso.
    """
    print("\nINICIANDO COMPARACION DE HUELLA DE CARBONO")
    print("="*80)
    
    if ELECTRICITY_MAPS_API_KEY == "TU_SANDBOX_KEY_AQUI":
        print("ERROR: No configuraste la API key")
        print("\nPara obtenerla:")
        print("1. Entrar a: https://app.electricitymaps.com/settings/api-access")
        print("2. Click en 'Activate sandbox key'")
        print("3. Copiar la key y pegarla en la linea 24 de este archivo")
        print("4. Volver a ejecutar")
        return
    
    print("\nPaso 1: Cargando resultados del benchmark...")
    resultados = cargar_resultados_benchmark("results/final_comparison_table.csv")
    if not resultados:
        return
    
    print("\nPaso 2: Consultando intensidad de carbono...")
    datos_carbono = {}
    exitosas = 0
    
    for codigo, nombre in PAISES.items():
        print(f"   Consultando {nombre}...", end=" ")
        datos = obtener_intensidad_carbono(codigo)
        datos_carbono[codigo] = datos
        
        if datos:
            print(f"OK - {datos['carbonIntensity']:.2f} g/kWh")
            exitosas += 1
        else:
            print("Sin datos")
    
    print(f"\n{exitosas}/{len(PAISES)} paises consultados exitosamente")
    
    print("\nPaso 3: Calculando emisiones de CO2...")
    resultados_completos = calcular_emisiones(resultados, datos_carbono)
    
    print("\nPaso 4: Guardando resultados...")
    guardar_csv(resultados_completos, "results/comparacion_huella_carbono.csv")
    
    print("\nPaso 5: Generando resumen...")
    mostrar_resumen(resultados_completos, datos_carbono)
    
    metadata = {
        "fecha": datetime.now(timezone.utc).isoformat(),
        "api_key": ELECTRICITY_MAPS_API_KEY[:8] + "...",
        "paises_consultados": exitosas,
        "modelos_analizados": len(resultados_completos)
    }
    
    with open("results/metadata_comparacion.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\nCOMPARACION FINALIZADA")
    print("="*80)
    print("\nArchivos generados:")
    print("   - results/comparacion_huella_carbono.csv")
    print("   - results/metadata_comparacion.json")


if __name__ == "__main__":
    main()
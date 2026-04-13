#!/usr/bin/env python3
"""
Comparación de intensidad de carbono por huso horario.

Usa ESIOS para España y Electricity Maps para otros países,
permitiendo análisis de sensibilidad geográfica en el protocolo Green AI.

Autor: Andrés Rubén Villamayor Ruiz Diaz
Fecha: 2026
"""

import requests
import logging
import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuración de zonas horarias por región
TIMEZONE_GROUPS = {
    "UTC-4_AMERICAS": {
        "name": "Américas - UTC-4",
        "countries": {
            "PY": {"name": "Paraguay", "default_factor": 25},
            "BO": {"name": "Bolivia", "default_factor": 150},
            "VE": {"name": "Venezuela", "default_factor": 180},
            "BR-AM": {"name": "Brasil (Amazonas)", "default_factor": 100},
        }
    },
    "UTC+1_EUROPE": {
        "name": "Europa - UTC+1",
        "countries": {
            "ES": {"name": "España", "default_factor": 245, "use_esios": True},
            "FR": {"name": "Francia", "default_factor": 65},
            "DE": {"name": "Alemania", "default_factor": 400},
            "IT": {"name": "Italia", "default_factor": 350},
        }
    }
}

# Endpoints oficiales
ESIOS_BASE_URL = "https://api.esios.ree.es/indicators/1346"  # Intensidad de emisiones España
ELECTRICITY_MAPS_BASE_URL = "https://api.electricitymaps.com/v3/carbon-intensity/history"


def get_esios_factor(timestamp: datetime, esios_token: str) -> Optional[float]:
    """
    Obtiene factor de emisión de España desde ESIOS para un timestamp específico.
    
    Args:
        timestamp: datetime en UTC
        esios_token: Token de autenticación de ESIOS
    
    Returns:
        float: Factor de emisión en g CO2/kWh o None si falla
    """
    try:
        # Convertir a hora local de España (CET/CEST)
        spain_tz = timezone(timedelta(hours=1))  # Simplificado: CET sin DST
        local_ts = timestamp.astimezone(spain_tz)
        ts_str = local_ts.strftime('%Y-%m-%dT%H:%M:%S')
        
        headers = {
            'Authorization': f'Token token={esios_token}',
            'Accept': 'application/json',
            'x-api-key': esios_token
        }
        
        params = {
            'start_date': ts_str,
            'end_date': ts_str,
            'time_aggregation': 'hour'
        }
        
        response = requests.get(ESIOS_BASE_URL, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        values = data.get('indicator', {}).get('values', [])
        if values:
            return float(values[0]['value'])
        
        logger.warning("No hay datos ESIOS para timestamp: %s", ts_str)
        return None
        
    except Exception as e:
        logger.error("Error al consultar ESIOS: %s", e)
        return None


def get_electricity_maps_factor(country_code: str, timestamp: datetime, 
                              em_token: Optional[str] = None) -> Optional[float]:
    """
    Obtiene factor de emisión desde Electricity Maps para un país y timestamp.
    
    Args:
        country_code: Código de zona (ej: "FR", "DE", "BR")
        timestamp: datetime en UTC
        em_token: Token opcional de Electricity Maps
    
    Returns:
        float: Factor de emisión en g CO2/kWh o None si falla
    """
    try:
        url = ELECTRICITY_MAPS_BASE_URL
        params = {"zone": country_code, "start": timestamp.isoformat(), "end": timestamp.isoformat()}
        
        headers = {}
        if em_token:
            headers['auth-token'] = em_token
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 401 and not em_token:
            logger.warning("Electricity Maps requiere token para acceso completo")
            return None
            
        response.raise_for_status()
        data = response.json()
        
        datapoints = data.get('datapoints', [])
        if datapoints:
            return float(datapoints[0]['carbonIntensity'])
        
        logger.warning("No hay datos de Electricity Maps para %s en %s", country_code, timestamp)
        return None
        
    except Exception as e:
        logger.error("Error al consultar Electricity Maps: %s", e)
        return None


def compare_timezones(timestamps: List[datetime], esios_token: str, 
                     em_token: Optional[str] = None) -> List[Dict]:
    """
    Compara intensidad de carbono entre países de mismos husos horarios.
    
    Args:
        timestamps: Lista de timestamps UTC para consultar
        esios_token: Token de ESIOS (requerido para España)
        em_token: Token opcional de Electricity Maps
    
    Returns:
        list: Lista de diccionarios con resultados comparativos
    """
    results = []
    
    for ts in timestamps:
        row = {"timestamp_utc": ts.isoformat()}
        
        # Consultar cada grupo de timezone
        for group_id, group_data in TIMEZONE_GROUPS.items():
            for country_code, country_info in group_data["countries"].items():
                factor = None
                
                # España usa ESIOS, otros usan Electricity Maps o fallback
                if country_info.get("use_esios") and country_code == "ES":
                    factor = get_esios_factor(ts, esios_token)
                else:
                    # Intentar Electricity Maps, fallback a valor por defecto
                    factor = get_electricity_maps_factor(country_code, ts, em_token)
                
                # Fallback a valor documentado si API falla
                if factor is None:
                    factor = country_info["default_factor"]
                    source = "default_documented"
                elif country_info.get("use_esios"):
                    source = "ESIOS/REE"
                else:
                    source = "Electricity Maps"
                
                # Agregar columna: {group}_{country}_factor
                col_name = f"{group_id}_{country_code.replace('-', '')}_factor_g_kwh"
                row[col_name] = round(factor, 2)
                row[f"{col_name}_source"] = source
        
        results.append(row)
    
    return results


def save_comparison(results: List[Dict], output_path: str = "results/timezone_comparison.csv"):
    """Guarda resultados de comparación en CSV."""
    if not results:
        logger.warning("No hay resultados para guardar")
        return
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    logger.info("Comparación guardada en: %s", output_path)


def main():
    """Ejecuta comparación de husos horarios."""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()  # Cargar variables desde .env
    
    esios_token = os.getenv("ESIOS_TOKEN")
    em_token = os.getenv("ELECTRICITY_MAPS_TOKEN")
    
    if not esios_token:
        logger.error("ESIOS_TOKEN no configurado. Obtener en: https://www.esios.ree.es/es/api")
        return
    
    # Leer timestamps desde el log del benchmark (últimas ejecuciones)
    log_path = Path("results/unified_benchmark.log")
    timestamps = []
    
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if "Timestamp de ejecución (UTC):" in line:
                    try:
                        # Extraer timestamp: "2026-04-13 18:23:13.414446+00:00"
                        ts_str = line.split("Timestamp de ejecución (UTC):")[1].strip()
                        ts = datetime.fromisoformat(ts_str)
                        timestamps.append(ts)
                    except:
                        continue
    
    # Si no hay timestamps del log, usar hora actual
    if not timestamps:
        timestamps = [datetime.now(timezone.utc)]
    
    # Tomar últimos 5 timestamps únicos (para no saturar APIs)
    timestamps = list(set(timestamps))[-5:]
    
    logger.info("Comparando %d timestamps entre husos horarios", len(timestamps))
    
    # Ejecutar comparación
    results = compare_timezones(timestamps, esios_token, em_token)
    
    # Guardar resultados
    save_comparison(results)
    
    # Mostrar resumen
    if results:
        logger.info("\n=== RESUMEN DE COMPARACIÓN ===")
        for row in results[:3]:  # Mostrar primeros 3
            print(f"\nTimestamp: {row['timestamp_utc']}")
            for key, value in row.items():
                if 'factor_g_kwh' in key and not key.endswith('_source'):
                    country = key.split('_')[-2] if '_' in key else key
                    print(f"  {country}: {value} g CO₂/kWh")


if __name__ == "__main__":
    main()
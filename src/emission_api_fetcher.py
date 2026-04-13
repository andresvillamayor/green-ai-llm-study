"""
Módulo opcional para obtener factores de emisión históricos desde APIs oficiales.

Este módulo NO es requerido para la ejecución principal del benchmark.
Se usa únicamente para análisis de sensibilidad en el anexo de la tesis.

Fuentes:
- España: ESIOS/REE (https://www.esios.ree.es/es/api)
- Global: Electricity Maps (https://api.electricitymaps.com)

Autor: Andrés Rubén Villamayor Ruiz Diaz
"""

import requests
import logging
from datetime import datetime, timezone
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class EmissionFactorFetcher:
    """
    Cliente para obtener factores de emisión históricos desde APIs oficiales.
    
    Nota: Requiere token gratuito para ESIOS. Electricity Maps tiene límite público.
    """
    
    # Endpoints oficiales
    ESIOS_BASE_URL = "https://api.esios.ree.es/indicators/1346"  # Intensidad de emisiones España
    ELECTRICITY_MAPS_BASE_URL = "https://api.electricitymaps.com/v3/carbon-intensity/history"
    
    def __init__(self, esios_token: Optional[str] = None):
        """
        Inicializa el fetcher.
        
        Args:
            esios_token: Token gratuito de ESIOS (obtener en https://www.esios.ree.es/es/api)
        """
        self.esios_token = esios_token
        self.esios_headers = {
            'Authorization': f'Token token={esios_token}',
            'Accept': 'application/json'
        } if esios_token else {}
    
    def get_spain_factor_for_timestamp(self, timestamp: datetime) -> Optional[Dict]:
        """
        Obtiene factor de emisión de España para un timestamp específico.
        
        Args:
            timestamp: datetime con timezone (preferiblemente UTC)
        
        Returns:
            dict: {"factor_g_kwh": float, "source": str, "timestamp": str} o None si falla
        """
        if not self.esios_token:
            logger.warning("Token ESIOS no proporcionado. No se puede obtener dato histórico.")
            return None
        
        try:
            # ESIOS espera formato ISO con timezone
            ts_str = timestamp.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
            
            params = {
                'start_date': ts_str,
                'end_date': ts_str,
                'time_aggregation': 'hour'
            }
            
            response = requests.get(
                self.ESIOS_BASE_URL,
                headers=self.esios_headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            values = data.get('indicator', {}).get('values', [])
            if not values:
                logger.warning("No hay datos para timestamp: %s", ts_str)
                return None
            
            # ESIOS devuelve valor en g CO2/kWh
            factor = values[0]['value']
            
            return {
                "factor_g_kwh": round(factor, 2),
                "source": "ESIOS/REE (dato histórico)",
                "timestamp": ts_str,
                "indicator_id": 1346
            }
            
        except Exception as e:
            logger.error("Error al obtener factor de España: %s", e)
            return None
    
    def get_global_factor_for_timestamp(self, country_code: str, timestamp: datetime) -> Optional[Dict]:
        """
        Obtiene factor de emisión global para un país y timestamp.
        
        Nota: Electricity Maps requiere registro para acceso completo.
        La API pública tiene límites de rate y cobertura limitada.
        
        Args:
            country_code: Código ISO de 2 letras (ej: "PY", "ES")
            timestamp: datetime con timezone
        
        Returns:
            dict: {"factor_g_kwh": float, "source": str, "timestamp": str} o None
        """
        try:
            # Construir URL y parámetros separados (mejor práctica para requests)
            url = self.ELECTRICITY_MAPS_BASE_URL
            params = {"zone": country_code}
            
            # Si tenés token de Electricity Maps, agregalo acá:
            # headers = {'auth-token': 'TU_TOKEN_AQUI'}
            # response = requests.get(url, headers=headers, params=params, timeout=10)
            
            # Para fines académicos sin token, simulamos la respuesta:
            logger.info("Electricity Maps: llamada simulada para %s en %s", country_code, timestamp)
            
            # Valores de ejemplo (reemplazar con llamada real si tenés token)
            example_factors = {
                "PY": 25, "ES": 245, "FR": 65, "DE": 400, "US": 415, "BR": 100, "AR": 350
            }
            factor = example_factors.get(country_code.upper(), 275)  # 275 = promedio UE
            
            return {
                "factor_g_kwh": factor,
                "source": "Electricity Maps (valor estimado para análisis de sensibilidad)",
                "timestamp": timestamp.isoformat(),
                "note": "Para dato real, registrar en electricitymaps.com y agregar token"
            }
            
        except Exception as e:
            logger.error("Error al obtener factor global: %s", e)
            return None

def fetch_and_log_factor(country_code: str, execution_timestamp: datetime, 
                        esios_token: Optional[str] = None) -> Dict:
    """
    Función de conveniencia para obtener y registrar factor de emisión.
    
    Usa enfoque híbrido:
    1. Intenta obtener dato histórico real (si hay token y API disponible)
    2. Si falla, retorna valor fijo documentado como fallback
    3. Siempre registra qué enfoque se usó para transparencia
    
    Args:
        country_code: Código de país (PY, ES, etc.)
        execution_timestamp: Timestamp exacto de la ejecución del benchmark
        esios_token: Token opcional de ESIOS
    
    Returns:
        dict: Factor usado + metadatos de procedencia
    """
    fetcher = EmissionFactorFetcher(esios_token=esios_token)
    
    # Intentar obtener dato histórico
    if country_code.upper() == "ES" and esios_token:
        historical = fetcher.get_spain_factor_for_timestamp(execution_timestamp)
        if historical:
            historical["approach"] = "historical_api"
            historical["reproducible"] = True  # Sí, si se usa el mismo timestamp
            return historical
    
    # Fallback a valor fijo documentado
    from src.energy_tracker import EnergyTracker
    factor_data = EnergyTracker.EMISSION_FACTORS.get(
        country_code.upper(), 
        EnergyTracker.EMISSION_FACTORS["PY"]
    )
    
    return {
        "factor_g_kwh": factor_data["factor_g_kwh"],
        "source": factor_data["source"],
        "timestamp": execution_timestamp.isoformat(),
        "approach": "fixed_documented_average",
        "reproducible": True,
        "note": "Valor promedio documentado para garantizar reproducibilidad del estudio"
    }
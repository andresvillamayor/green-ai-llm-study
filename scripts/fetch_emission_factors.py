#!/usr/bin/env python3
"""
Obtención de factores de emisión actualizados desde fuentes externas.

Fuentes:
- España: ESIOS/REE (https://www.esios.ree.es)
- Paraguay: Electricity Maps (https://app.electricitymaps.com)

Nota: Estas APIs pueden requerir token o tener límites de rate.
Este script guarda los valores en un JSON para usar en el benchmark.

Autor: Andrés Rubén Villamayor Ruiz Diaz
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Valores por defecto (usar si no se puede acceder a APIs)
DEFAULT_FACTORS = {
    "PY": {"factor_g_per_kwh": 70, "source": "IEA 2024 (hidroeléctrica)", "updated": None},
    "ES": {"factor_g_per_kwh": 245, "source": "ESIOS/REE promedio 2024", "updated": None}
}


def fetch_esios_factor(date_str: str = None) -> dict:
    """
    Obtiene factor de emisión de España desde ESIOS/REE.
    
    Nota: Requiere token de API. Si no hay token, devuelve valor por defecto.
    Documentación: https://www.esios.ree.es/es/api
    """
    # Placeholder: en producción, implementar llamada real a API
    # Por ahora, devolvemos valor estimado con fecha
    return {
        "factor_g_per_kwh": 245,  # Valor promedio 2024
        "source": "ESIOS/REE - Valor estimado (requiere token para dato en tiempo real)",
        "updated": date_str or datetime.now().strftime("%Y-%m-%d"),
        "note": "Para dato horario real: solicitar token en esios.ree.es y usar endpoint /indicators/1346"
    }


def fetch_electricity_maps_py() -> dict:
    """
    Obtiene factor de emisión de Paraguay desde Electricity Maps.
    
    Nota: La API pública tiene límites. Para acceso completo, registrar en:
    https://www.electricitymaps.com/data-access
    """
    # Placeholder: Electricity Maps muestra ~70 g/kWh para Paraguay (hidroeléctrica)
    return {
        "factor_g_per_kwh": 70,
        "source": "Electricity Maps - Valor estimado para Paraguay",
        "updated": datetime.now().strftime("%Y-%m-%d"),
        "note": "Paraguay tiene matriz predominantemente hidroeléctrica (Itaipú, Yacyretá)"
    }


def save_factors_to_json(output_path: str = "configs/emission_factors_updated.json"):
    """Guarda factores actualizados en JSON para usar en el benchmark."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    factors = {
        "PY": fetch_electricity_maps_py(),
        "ES": fetch_esios_factor(),
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "note": "Valores estimados. Para datos en tiempo real, integrar APIs con token."
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(factors, f, indent=2, ensure_ascii=False)
    
    logger.info("Factores guardados en: %s", output_path)
    return factors


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    factors = save_factors_to_json()
    
    print("\n📊 Factores de emisión actualizados:")
    for code, data in factors.items():
        if code != "metadata":
            print(f"  {code}: {data['factor_g_per_kwh']} g/kWh - {data['source']}")
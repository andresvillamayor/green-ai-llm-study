"""
Wrapper para CodeCarbon configurado para Paraguay
"""

from codecarbon import EmissionsTracker
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class EnergyTracker:
    """
    Tracker de energía con configuración para matriz paraguaya
    
    Ejemplo:
        >>> tracker = EnergyTracker(project_name="mi-exp")
        >>> tracker.start()
        >>> # ... código a medir ...
        >>> emissions = tracker.stop()
        >>> print(f"Emisiones: {emissions:.6f} kg CO2")
    """
    
    def __init__(
        self,
        project_name: str = "llm-inference",
        output_dir: str = "results/measurements",
        output_file: Optional[str] = None,
        country: str = "Paraguay"  # Cambiado de country_iso_code a country
    ):
        """
        Inicializar tracker
        
        Args:
            project_name: Nombre del experimento
            output_dir: Carpeta para guardar resultados
            output_file: Nombre del archivo CSV (opcional)
            country: Nombre del país (Paraguay)
        """
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuración adaptada para CodeCarbon 2.3.4
        self.tracker = EmissionsTracker(
            project_name=project_name,
            output_dir=str(self.output_dir),
            output_file=output_file or f"{project_name}_emissions.csv",
            # Usar solo save_to_file y log_level
            save_to_file=True,
            log_level="warning"
        )
        
        self._emissions = None
        self._country = country
        logger.info(f"✅ EnergyTracker inicializado: {project_name} (País: {country})")
    
    def start(self):
        """Iniciar medición de energía"""
        logger.info("⚡ Iniciando medición energética...")
        self.tracker.start()
    
    def stop(self) -> float:
        """
        Detener medición
        
        Returns:
            Emisiones totales en kg CO2
        """
        self._emissions = self.tracker.stop()
        logger.info(f"🛑 Medición finalizada: {self._emissions*1000:.4f}g CO2e")
        return self._emissions
    
    @property
    def emissions_g(self) -> float:
        """Emisiones en gramos"""
        return self._emissions * 1000 if self._emissions else 0
    
    @property
    def energy_wh(self) -> float:
        """
        Energía en Watt-hora
        Factor Paraguay: 0.07 kg CO2/kWh (matriz hidroeléctrica)
        """
        if not self._emissions:
            return 0
        # Convertir de kg CO2 a Wh usando factor de Paraguay
        return (self._emissions * 1000) / 0.07
    
    @property
    def energy_kwh(self) -> float:
        """Energía en kiloWatt-hora"""
        return self.energy_wh / 1000
    
    def get_stats(self) -> dict:
        """
        Obtener todas las estadísticas de la medición
        
        Returns:
            Diccionario con emisiones y energía
        """
        return {
            "emissions_kg_co2": self._emissions,
            "emissions_g_co2": self.emissions_g,
            "energy_wh": self.energy_wh,
            "energy_kwh": self.energy_kwh,
            "country": self._country,
            "carbon_intensity_kg_per_kwh": 0.07
        }

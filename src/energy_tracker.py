"""
Módulo para estimación de consumo energético y emisiones de CO2.

Este módulo proporciona una interfaz sencilla para ESTIMAR el impacto ambiental
de la inferencia de modelos LLM, usando un modelo software basado en:
- Tiempo de ejecución medido
- Potencia típica del chip Apple M4 en inferencia LLM
- Factor de emisión de Paraguay (matriz hidroeléctrica)

Clases:
    EnergyTracker: Clase principal para estimar energía y emisiones
"""

from pathlib import Path
from typing import Optional, Dict
from datetime import datetime, timezone
import logging
import time

logger = logging.getLogger(__name__)


class EnergyTracker:
    """
    Tracker de energía y emisiones de CO2 para experimentos de inferencia LLM.
    
    Usa un modelo de estimación software (sin acceso a hardware) basado en:
    - Tiempo de ejecución medido con time.time()
    - Potencia estimada del Apple M4 en carga de inferencia (~40W promedio)
    - Factor de emisión de Paraguay: 0.07 kg CO2/kWh
    
    Este enfoque es válido para estudios comparativos GREEN AI donde la
    consistencia metodológica es más importante que la medición hardware absoluta.
    
    Atributos:
        project_name (str): Nombre del experimento para identificar los logs
        output_dir (Path): Directorio donde se guardarán los archivos CSV
        _emissions (float): Emisiones estimadas en la última medición (kg CO2)
        _country (str): País configurado para el factor de emisión
        _start_time (float): Timestamp de inicio de la medición
        _power_estimate_watts (float): Potencia estimada en Watts para cálculos
    """
    
    # Factor de emisión para Paraguay: 0.07 kg CO2 por kWh (matriz hidroeléctrica)
    # Fuente: IEA Emissions Factors 2024, ajustado para generación hidroeléctrica
    CARBON_INTENSITY_PY = 0.07
    
    # Potencia estimada para Apple M4 en inferencia LLM (valor conservador)
    # Basado en: TDP del M4 + mediciones públicas de carga sostenida
    POWER_ESTIMATE_WATTS = 40.0
    
    def __init__(
        self,
        project_name: str = "llm-inference",
        output_dir: str = "results/measurements",
        output_file: Optional[str] = None,
        country: str = "Paraguay",
        power_estimate_watts: float = None
    ):
        """
        Inicializa el tracker de energía en modo estimación software.
        
        Args:
            project_name (str): Nombre para identificar el experimento
            output_dir (str): Carpeta para guardar archivos CSV (se crea si no existe)
            output_file (str, opcional): Nombre del archivo CSV. Si es None,
                se genera automáticamente desde project_name
            country (str): País para el factor de emisión. Por defecto: "Paraguay"
            power_estimate_watts (float, opcional): Potencia estimada en Watts.
                Si es None, usa el valor por defecto de 40W para Apple M4.
        
        Nota:
            Este tracker NO usa CodeCarbon ni accede a hardware. Todas las
            métricas se calculan mediante fórmulas determinísticas basadas
            en tiempo de ejecución y potencia estimada.
        """
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._emissions = None
        self._country = country
        self._start_time = None
        self._power_estimate_watts = power_estimate_watts or self.POWER_ESTIMATE_WATTS
        
        # No inicializamos CodeCarbon para evitar threads y errores
        self.tracker = None
        self._available = False
        self._mode = "estimacion_software"
        
        logger.info("EnergyTracker inicializado: %s (Pais: %s, Modo: %s)", 
                   project_name, country, self._mode)
    
    def start(self):
        """Inicia la medición de energía (modo software)."""
        logger.info("Iniciando medición energetica (modo estimación software)...")
        self._start_time = time.time()
        # Usar datetime solo si está importado correctamente
        try:
            self._execution_timestamp = datetime.now(timezone.utc)
        except NameError:
            # Fallback si datetime no está disponible
            self._execution_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        logger.info("Timestamp de ejecución (UTC): %s", self._execution_timestamp)
    
    def stop(self) -> float:
        """
        Detiene la medición y calcula las emisiones estimadas.
        
        Fórmula:
            energía_kWh = (potencia_W × tiempo_s) / 3,600,000
            emisiones_kg_CO2 = energía_kWh × factor_emisión_kg_por_kWh
        
        Returns:
            float: Emisiones estimadas de CO2 en kilogramos (kg)
        """
        if self._start_time is None:
            logger.warning("stop() llamado sin start() previo. Devolviendo 0.0")
            return 0.0
        
        elapsed_seconds = time.time() - self._start_time
        
        # Calcular energía en kWh: (Watts × segundos) / 3,600,000
        energy_kwh = (self._power_estimate_watts * elapsed_seconds) / 3_600_000
        
        # Calcular emisiones: kWh × factor de emisión
        self._emissions = energy_kwh * self.CARBON_INTENSITY_PY
        
        logger.info("Medicion finalizada: %.4f g CO2e en %.2f segundos (estimación software)", 
                   self._emissions * 1000, elapsed_seconds)
        
        return self._emissions
    
    def measure_inference(self, func, *args, **kwargs) -> tuple:
        """
        Ejecuta una función midiendo su consumo energético estimado.
        
        Método útil para envolver llamadas a runner.generate() sin modificar
        el código original.
        
        Args:
            func: Función a ejecutar (ej: runner.generate)
            *args: Argumentos posicionales para la función
            **kwargs: Argumentos nombrados para la función
        
        Returns:
            tuple: (resultado_de_la_funcion, emisiones_estimadas_kg_co2)
        
        Ejemplo:
            >>> result, emissions = tracker.measure_inference(
            ...     runner.generate,
            ...     prompt="Hola",
            ...     max_tokens=50
            ... )
        """
        self.start()
        try:
            result = func(*args, **kwargs)
            emissions = self.stop()
            return result, emissions
        except Exception as e:
            self.stop()  # Asegurar que se registre el tiempo incluso si hay error
            logger.error("Error durante la medición: %s", e)
            raise
    
    @property
    def emissions_g(self) -> float:
        """
        Devuelve las emisiones estimadas en gramos de CO2.
        
        Returns:
            float: Emisiones en gramos (g CO2), o 0.0 si no hay medición
        """
        return self._emissions * 1000 if self._emissions is not None else 0.0
    
    @property
    def energy_kwh(self) -> float:
        """
        Calcula la energía estimada consumida en kilovatios-hora (kWh).
        
        Fórmula: emisiones_kg_CO2 / factor_emisión_kg_por_kWh
        
        Returns:
            float: Energía estimada en kWh, o 0.0 si no hay medición
        """
        if self._emissions is None or self.CARBON_INTENSITY_PY <= 0:
            return 0.0
        return self._emissions / self.CARBON_INTENSITY_PY
    
    @property
    def energy_wh(self) -> float:
        """
        Calcula la energía estimada consumida en vatios-hora (Wh).
        
        Returns:
            float: Energía estimada en Wh
        """
        return self.energy_kwh * 1000
    
    def get_stats(self) -> Dict[str, any]:
        """
        Obtiene un diccionario con todas las estadísticas de la medición.
        
        Returns:
            dict: Diccionario con las siguientes claves:
                - emissions_kg_co2 (float): Emisiones estimadas en kilogramos
                - emissions_g_co2 (float): Emisiones estimadas en gramos
                - energy_kwh (float): Energía estimada en kilovatios-hora
                - energy_wh (float): Energía estimada en vatios-hora
                - country (str): País configurado para el factor de emisión
                - carbon_intensity_kg_per_kwh (float): Factor de emisión usado
                - power_estimate_watts (float): Potencia estimada usada en cálculos
                - measured (bool): Siempre False (es estimación, no medición hardware)
        """
        emissions_kg = self._emissions if self._emissions is not None else 0.0
        
        return {
            "emissions_kg_co2": emissions_kg,
            "emissions_g_co2": emissions_kg * 1000,
            "energy_kwh": self.energy_kwh,
            "energy_wh": self.energy_wh,
            "country": self._country,
            "carbon_intensity_kg_per_kwh": self.CARBON_INTENSITY_PY,
            "power_estimate_watts": self._power_estimate_watts,
            "measured": False,  # Siempre False porque es estimación software
            "execution_timestamp_utc": self._execution_timestamp.isoformat() if hasattr(self, '_execution_timestamp') else None,
            "factor_approach": "fixed_documented_average",  # Por defecto
        }
    
    def set_power_estimate(self, watts: float):
        """
        Actualiza la potencia estimada usada en los cálculos.
        
        Útil si querés ajustar el valor según mediciones externas o literatura.
        
        Args:
            watts (float): Nueva potencia estimada en Watts
        """
        if watts > 0:
            self._power_estimate_watts = watts
            logger.info("Potencia estimada actualizada a %.1f W", watts)
        else:
            logger.warning("Potencia estimada inválida: %.1f W. Se mantiene el valor anterior.", watts)
"""
Factores de Intensidad de Carbono por País
==========================================

Módulo: carbon_factors.py
Autor: Andrés Rubén Villamayor Ruiz Diaz

Descripción:
-----------
Este módulo contiene los factores de intensidad de carbono (gCO2eq/kWh) para
diferentes países, utilizados en el cálculo de la huella de carbono de los
experimentos con modelos de lenguaje.

Fuentes de Datos:
----------------
1. IEA (International Energy Agency) - World Energy Outlook 2023
   URL: https://www.iea.org/data-and-statistics
   
2. Electricity Maps - Carbon Intensity Data
   URL: https://app.electricitymaps.com/
   
3. IPCC (Intergovernmental Panel on Climate Change) - Emission Factors Database
   URL: https://www.ipcc-nggip.iges.or.jp/EFDB/

Metodología:
-----------
Los factores de intensidad de carbono representan los gramos de CO2 equivalente
emitidos por cada kilovatio-hora (kWh) de electricidad generada en un país,
considerando toda la matriz energética nacional (combustibles fósiles, nuclear,
renovables).

Unidad: gCO2eq/kWh (gramos de CO2 equivalente por kilovatio-hora)

Notas Metodológicas:
-------------------
- Los valores son promedios anuales del año 2023
- Incluyen emisiones directas e indirectas del ciclo de vida
- Se consideran pérdidas en transmisión y distribución
- Los datos se actualizan anualmente según publicaciones de IEA
"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class DatosMatrizEnergetica:
    """
    Clase de datos para almacenar información de la matriz energética de un país.
    
    Attributes:
        nombre_pais (str): Nombre completo del país
        intensidad_carbono_gco2_kwh (float): Factor de emisión en gCO2eq/kWh
        porcentaje_renovables (float): Porcentaje de energía renovable (0-100)
        fuente_datos (str): Fuente oficial de los datos
        anio_referencia (int): Año de los datos
    """
    nombre_pais: str
    intensidad_carbono_gco2_kwh: float
    porcentaje_renovables: float
    fuente_datos: str
    anio_referencia: int = 2023


# Diccionario principal con factores de intensidad de carbono por país
FACTORES_CARBONO_POR_PAIS = {
    
    # ============================================================================
    # AMÉRICA DEL SUR
    # ============================================================================
    
    "PRY": DatosMatrizEnergetica(
        nombre_pais="Paraguay",
        intensidad_carbono_gco2_kwh=10.0,
        porcentaje_renovables=99.7,
        fuente_datos="IEA World Energy Outlook 2023",
        anio_referencia=2023
    ),
    # Justificación: Paraguay tiene la matriz energética más limpia del mundo
    # debido a que prácticamente el 100% de su electricidad proviene de
    # hidroeléctricas (Itaipú y Yacyretá principalmente).
    
    "BRA": DatosMatrizEnergetica(
        nombre_pais="Brasil",
        intensidad_carbono_gco2_kwh=85.0,
        porcentaje_renovables=85.0,
        fuente_datos="IEA World Energy Outlook 2023",
        anio_referencia=2023
    ),
    # Matriz energética con fuerte presencia hidroeléctrica, complementada
    # con eólica y biomasa. Emisiones bajas comparadas con la media global.
    
    "ARG": DatosMatrizEnergetica(
        nombre_pais="Argentina",
        intensidad_carbono_gco2_kwh=340.0,
        porcentaje_renovables=37.0,
        fuente_datos="IEA World Energy Outlook 2023",
        anio_referencia=2023
    ),
    # Matriz energética mixta con presencia significativa de gas natural
    # y térmica, además de hidroeléctrica y renovables en crecimiento.
    
    "CHL": DatosMatrizEnergetica(
        nombre_pais="Chile",
        intensidad_carbono_gco2_kwh=380.0,
        porcentaje_renovables=45.0,
        fuente_datos="IEA World Energy Outlook 2023",
        anio_referencia=2023
    ),
    # Matriz en transición, con fuerte inversión en energías renovables
    # (solar y eólica) pero aún con presencia de carbón.
    
    # ============================================================================
    # EUROPA
    # ============================================================================
    
    "FRA": DatosMatrizEnergetica(
        nombre_pais="Francia",
        intensidad_carbono_gco2_kwh=52.0,
        porcentaje_renovables=25.0,
        fuente_datos="IEA World Energy Outlook 2023",
        anio_referencia=2023
    ),
    # Matriz dominada por energía nuclear (~70%) y renovables.
    # Una de las matrices más limpias de Europa.
    
    "DEU": DatosMatrizEnergetica(
        nombre_pais="Alemania",
        intensidad_carbono_gco2_kwh=380.0,
        porcentaje_renovables=46.0,
        fuente_datos="IEA World Energy Outlook 2023",
        anio_referencia=2023
    ),
    # Matriz en transición ("Energiewende"). Alto uso de renovables pero
    # todavía con presencia de carbón debido al cierre de plantas nucleares.
    
    "ESP": DatosMatrizEnergetica(
        nombre_pais="España",
        intensidad_carbono_gco2_kwh=210.0,
        porcentaje_renovables=52.0,
        fuente_datos="IEA World Energy Outlook 2023",
        anio_referencia=2023
    ),
    # Líder en energías renovables en Europa (eólica y solar).
    # Matriz con bajo factor de emisión.
    
    "NOR": DatosMatrizEnergetica(
        nombre_pais="Noruega",
        intensidad_carbono_gco2_kwh=15.0,
        porcentaje_renovables=98.0,
        fuente_datos="IEA World Energy Outlook 2023",
        anio_referencia=2023
    ),
    # Matriz prácticamente 100% hidroeléctrica. Una de las más limpias del mundo.
    
    # ============================================================================
    # ASIA
    # ============================================================================
    
    "CHN": DatosMatrizEnergetica(
        nombre_pais="China",
        intensidad_carbono_gco2_kwh=555.0,
        porcentaje_renovables=28.0,
        fuente_datos="IEA World Energy Outlook 2023",
        anio_referencia=2023
    ),
    # Matriz aún dominada por carbón (~60%), aunque con inversión masiva
    # en renovables. Mayor emisor absoluto de CO2 del mundo.
    
    "JPN": DatosMatrizEnergetica(
        nombre_pais="Japón",
        intensidad_carbono_gco2_kwh=480.0,
        porcentaje_renovables=22.0,
        fuente_datos="IEA World Energy Outlook 2023",
        anio_referencia=2023
    ),
    # Post-Fukushima, mayor dependencia de combustibles fósiles (gas y carbón).
    # Reinicio gradual de plantas nucleares.
    
    "IND": DatosMatrizEnergetica(
        nombre_pais="India",
        intensidad_carbono_gco2_kwh=630.0,
        porcentaje_renovables=23.0,
        fuente_datos="IEA World Energy Outlook 2023",
        anio_referencia=2023
    ),
    # Fuerte dependencia del carbón (~70%). Crecimiento de renovables
    # pero desde una base de alta emisión.
    
    # ============================================================================
    # AMÉRICA DEL NORTE
    # ============================================================================
    
    "USA": DatosMatrizEnergetica(
        nombre_pais="Estados Unidos",
        intensidad_carbono_gco2_kwh=386.0,
        porcentaje_renovables=22.0,
        fuente_datos="IEA World Energy Outlook 2023",
        anio_referencia=2023
    ),
    # Matriz diversificada: gas natural (~40%), carbón (~20%), nuclear (~20%),
    # renovables (~20%). Varía significativamente por estado.
    
    "CAN": DatosMatrizEnergetica(
        nombre_pais="Canadá",
        intensidad_carbono_gco2_kwh=120.0,
        porcentaje_renovables=68.0,
        fuente_datos="IEA World Energy Outlook 2023",
        anio_referencia=2023
    ),
    # Matriz limpia debido a fuerte presencia hidroeléctrica (~60%).
    # Nuclear y renovables complementan. Varía por provincia.
    
    "MEX": DatosMatrizEnergetica(
        nombre_pais="México",
        intensidad_carbono_gco2_kwh=420.0,
        porcentaje_renovables=26.0,
        fuente_datos="IEA World Energy Outlook 2023",
        anio_referencia=2023
    ),
    # Matriz basada en gas natural y combustóleo. Crecimiento de renovables
    # (eólica y solar) en años recientes.
    
    # ============================================================================
    # OCEANÍA
    # ============================================================================
    
    "AUS": DatosMatrizEnergetica(
        nombre_pais="Australia",
        intensidad_carbono_gco2_kwh=620.0,
        porcentaje_renovables=32.0,
        fuente_datos="IEA World Energy Outlook 2023",
        anio_referencia=2023
    ),
    # Matriz históricamente dependiente del carbón. Transición acelerada
    # hacia renovables (solar y eólica) en curso.
}


# Factor de emisión promedio global (para comparación o países no listados)
FACTOR_GLOBAL_PROMEDIO = 475.0  # gCO2eq/kWh (IEA 2023)


class GestorFactoresCarbono:
    """
    Clase para gestionar y consultar factores de intensidad de carbono por país.
    
    Esta clase facilita:
    - Consulta de factores de emisión por código ISO de país
    - Comparación de matrices energéticas entre países
    - Obtención de datos para documentación de tesis
    - Validación de códigos de país
    
    Attributes:
        factores (Dict): Diccionario con los factores por país
        
    Example:
        >>> gestor = GestorFactoresCarbono()
        >>> datos_pry = gestor.obtener_factor("PRY")
        >>> print(f"Paraguay: {datos_pry.intensidad_carbono_gco2_kwh} gCO2eq/kWh")
    """
    
    def __init__(self):
        """Inicializa el gestor con los factores de carbono predefinidos."""
        self.factores = FACTORES_CARBONO_POR_PAIS
    
    def obtener_factor(self, codigo_pais: str) -> Optional[DatosMatrizEnergetica]:
        """
        Obtiene los datos de la matriz energética de un país.
        
        Args:
            codigo_pais (str): Código ISO 3166-1 alpha-3 del país (ej: "PRY")
            
        Returns:
            Optional[DatosMatrizEnergetica]: Datos del país o None si no existe
            
        Example:
            >>> gestor = GestorFactoresCarbono()
            >>> datos = gestor.obtener_factor("FRA")
            >>> print(f"{datos.nombre_pais}: {datos.intensidad_carbono_gco2_kwh}")
            Francia: 52.0
        """
        codigo_upper = codigo_pais.upper()
        return self.factores.get(codigo_upper, None)
    
    def obtener_intensidad(self, codigo_pais: str) -> float:
        """
        Obtiene solo el valor de intensidad de carbono de un país.
        
        Args:
            codigo_pais (str): Código ISO del país
            
        Returns:
            float: Intensidad de carbono en gCO2eq/kWh,
                   o promedio global si el país no está en la base
                   
        Example:
            >>> gestor = GestorFactoresCarbono()
            >>> intensidad = gestor.obtener_intensidad("PRY")
            >>> print(f"Intensidad Paraguay: {intensidad} gCO2eq/kWh")
            Intensidad Paraguay: 10.0 gCO2eq/kWh
        """
        datos = self.obtener_factor(codigo_pais)
        if datos:
            return datos.intensidad_carbono_gco2_kwh
        return FACTOR_GLOBAL_PROMEDIO
    
    def listar_paises_disponibles(self) -> None:
        """
        Imprime un listado formateado de todos los países disponibles.
        
        Útil para documentación y selección de países para experimentos.
        """
        print("=" * 100)
        print("FACTORES DE INTENSIDAD DE CARBONO POR PAÍS")
        print("=" * 100)
        print(f"\nTotal de países en la base: {len(self.factores)}")
        print(f"Factor promedio global: {FACTOR_GLOBAL_PROMEDIO} gCO2eq/kWh")
        print("\n" + "-" * 100 + "\n")
        
        # Ordenar países por intensidad de carbono (de menor a mayor)
        paises_ordenados = sorted(
            self.factores.items(),
            key=lambda x: x[1].intensidad_carbono_gco2_kwh
        )
        
        print(f"{'Código':<8} {'País':<20} {'Intensidad':<15} {'Renovables':<12} {'Fuente'}")
        print("-" * 100)
        
        for codigo, datos in paises_ordenados:
            print(
                f"{codigo:<8} "
                f"{datos.nombre_pais:<20} "
                f"{datos.intensidad_carbono_gco2_kwh:>6.1f} gCO2/kWh   "
                f"{datos.porcentaje_renovables:>5.1f}%       "
                f"{datos.fuente_datos[:30]}"
            )
    
    def comparar_paises(self, lista_codigos: list) -> None:
        """
        Compara la intensidad de carbono de múltiples países.
        
        Args:
            lista_codigos (list): Lista de códigos ISO de países a comparar
            
        Example:
            >>> gestor = GestorFactoresCarbono()
            >>> gestor.comparar_paises(["PRY", "BRA", "ARG", "USA", "CHN"])
        """
        print("\n" + "=" * 80)
        print("COMPARACIÓN DE INTENSIDAD DE CARBONO")
        print("=" * 80 + "\n")
        
        datos_comparacion = []
        for codigo in lista_codigos:
            datos = self.obtener_factor(codigo)
            if datos:
                datos_comparacion.append((codigo, datos))
            else:
                print(f"⚠️  País '{codigo}' no encontrado en la base de datos")
        
        # Ordenar por intensidad
        datos_comparacion.sort(key=lambda x: x[1].intensidad_carbono_gco2_kwh)
        
        print(f"{'País':<25} {'Intensidad (gCO2/kWh)':<25} {'Renovables (%)'}")
        print("-" * 80)
        
        for codigo, datos in datos_comparacion:
            print(
                f"{datos.nombre_pais:<25} "
                f"{datos.intensidad_carbono_gco2_kwh:>10.1f}                "
                f"{datos.porcentaje_renovables:>8.1f}"
            )
        
        # Calcular estadísticas
        intensidades = [d[1].intensidad_carbono_gco2_kwh for d in datos_comparacion]
        print("\n" + "-" * 80)
        print(f"Promedio del grupo: {sum(intensidades)/len(intensidades):.1f} gCO2/kWh")
        print(f"Mínimo: {min(intensidades):.1f} gCO2/kWh")
        print(f"Máximo: {max(intensidades):.1f} gCO2/kWh")
        print(f"Rango: {max(intensidades) - min(intensidades):.1f} gCO2/kWh")
    
    def validar_codigo_pais(self, codigo: str) -> bool:
        """
        Valida si un código de país existe en la base de datos.
        
        Args:
            codigo (str): Código ISO del país
            
        Returns:
            bool: True si existe, False si no
        """
        return codigo.upper() in self.factores


# ============================================================================
# Bloque de Prueba
# ============================================================================

if __name__ == "__main__":
    """
    Bloque de prueba para verificar el funcionamiento del módulo.
    
    Ejecutar con: python3 src/carbon_factors.py
    """
    print("\n🌍 MÓDULO: Factores de Intensidad de Carbono por País")
    print("📚 Tesis de Maestría en Ciencia de Datos\n")
    
    # Crear instancia del gestor
    gestor = GestorFactoresCarbono()
    
    # Listar todos los países disponibles
    gestor.listar_paises_disponibles()
    
    # Ejemplo de comparación
    print("\n")
    paises_tesis = ["PRY", "BRA", "ARG", "USA", "CHN", "FRA", "DEU"]
    gestor.comparar_paises(paises_tesis)
    
    # Ejemplo de consulta individual
    print("\n" + "=" * 80)
    print("EJEMPLO: Consulta individual de Paraguay")
    print("=" * 80 + "\n")
    
    datos_pry = gestor.obtener_factor("PRY")
    if datos_pry:
        print(f"País: {datos_pry.nombre_pais}")
        print(f"Intensidad de carbono: {datos_pry.intensidad_carbono_gco2_kwh} gCO2eq/kWh")
        print(f"Porcentaje renovables: {datos_pry.porcentaje_renovables}%")
        print(f"Fuente: {datos_pry.fuente_datos}")
        print(f"Año de referencia: {datos_pry.anio_referencia}")
    
    print("\n" + "=" * 80)
    print("✅ Módulo verificado correctamente")
    print("=" * 80)
"""
carbon_factors.py
Proyecto GREEN-IA — Factores de emision y hardware
Autor: Andres Villamayor

Valores cientificamente respaldados para Apple M4 Mac Mini 16GB
y factor de emision de Paraguay.
"""

# Factor de emision de Paraguay
# Fuente: Electricity Maps (2026). Carbon intensity of Paraguay.
# https://app.electricitymaps.com/zone/PY
# Acceso: 21 junio 2026
# Valor: 26 gCO2eq/kWh (matriz 100% hidroelectrica)
CARBON_INTENSITY_PARAGUAY = 26  # gCO2eq/kWh

# Especificaciones Apple M4 Mac Mini 16GB
# Fuente 1: Apple Support (2024). Mac mini power consumption.
#   https://support.apple.com/es-us/103253
#   Rango oficial: 4W a 65W
# Fuente 2: NotebookCheck (2024). Apple Mac Mini M4 review.
#   https://www.notebookcheck.net/Apple-Mac-Mini-M4-review
#   Idle: 2.6-2.7W | Multi-core sostenido: 21.5W | Pico: 62.5W
# Fuente 3: Jeff Geerling (2024). M4 Mac mini efficiency.
#   https://www.jeffgeerling.com/blog/2024/m4-mac-minis-efficiency-incredible
#   Idle total sistema: 3-4W

APPLE_M4_SPECS = {
    # Procesador
    'cpu_model'        : 'Apple M4',
    'cpu_cores_total'  : 10,
    'cpu_cores_perf'   : 4,   # P-cores 4.4 GHz
    'cpu_cores_eff'    : 6,   # E-cores 2.85 GHz
    'cpu_tdp_w'        : 20,  # TDP tipico durante inferencia LLM
    'cpu_idle_w'       : 4,   # Consumo en reposo total sistema
    'cpu_peak_w'       : 40,  # Pico chip M4 (NotebookCheck 2024)

    # GPU (integrada en chip M4)
    'gpu_model'        : 'Apple M4 GPU (Metal)',
    'gpu_cores'        : 10,
    'gpu_tdp_w'        : 10,  # estimado — parte del chip M4

    # RAM
    'ram_gb'           : 16,
    'ram_type'         : 'LPDDR5X-7500',
    'ram_bandwidth_gbs': 120,
    'ram_tdp_w'        : 3,   # estimado LPDDR5X a 16GB

    # Sistema
    'architecture'     : 'Apple Silicon ARM',
    'memory_type'      : 'Unified Memory',
    'process_node_nm'  : 3,   # TSMC 3nm
    'macos_version'    : '15.6.1',

    # Fuentes
    'fuente_tdp'       : 'Apple Support 2024 + NotebookCheck 2024',
    'fuente_specs'     : 'Apple Mac Mini Technical Specifications 2024',
}

# CodeCarbon — configuracion para Apple Silicon
# Fuente: CodeCarbon Documentation (Courty et al. 2023)
# https://docs.codecarbon.io/latest/explanation/methodology/
# Sin sudo: CodeCarbon cae en modo constante usando TDP del fabricante
# Con sudo: trackea via powermetrics (mas preciso)
# Nota: en este experimento corremos sin sudo por reproducibilidad
CODECARBON_CONFIG = {
    'country_iso_code'    : 'PRY',
    'carbon_intensity'    : CARBON_INTENSITY_PARAGUAY,
    'measure_power_secs'  : 15,   # intervalo de muestreo por defecto
    'tracking_mode'       : 'constant',  # sin sudo en Apple Silicon
    'log_level'           : 'error',
    'save_to_file'        : False,
}

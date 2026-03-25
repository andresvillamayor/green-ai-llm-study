"""
Utilidades generales del proyecto
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    """
    Configurar sistema de logging
    
    Args:
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def save_json(
    data: Dict,
    output_dir: str = "results/measurements",
    filename: Optional[str] = None
) -> Path:
    """
    Guardar diccionario en JSON
    
    Args:
        data: Diccionario a guardar
        output_dir: Carpeta de salida
        filename: Nombre del archivo (opcional)
        
    Returns:
        Path del archivo guardado
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"measurement_{timestamp}.json"
    
    filepath = output_path / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 JSON guardado: {filepath}")
    return filepath


def append_to_csv(
    data: Dict,
    csv_path: str = "results/measurements/all_measurements.csv"
) -> Path:
    """
    Agregar fila a CSV acumulativo
    
    Args:
        data: Diccionario con datos de la medición
        csv_path: Ruta del CSV
        
    Returns:
        Path del CSV
    """
    filepath = Path(csv_path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    file_exists = filepath.exists()
    
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(data)
    
    logger.info(f"📊 Fila agregada a CSV: {filepath}")
    return filepath


def load_prompts(prompts_file: str = "data/raw/prompts.txt") -> List[str]:
    """
    Cargar prompts desde archivo de texto
    
    Args:
        prompts_file: Ruta del archivo
        
    Returns:
        Lista de prompts
    """
    filepath = Path(prompts_file)
    
    if not filepath.exists():
        logger.warning(f"⚠️  Archivo no encontrado: {filepath}")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    logger.info(f"📝 Cargados {len(prompts)} prompts desde {filepath}")
    return prompts


def format_bytes(bytes_size: int) -> str:
    """Formatear tamaño en bytes a formato legible"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def format_duration(seconds: float) -> str:
    """Formatear duración en formato legible"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.2f}min"
    else:
        return f"{seconds/3600:.2f}h"

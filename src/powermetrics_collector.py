#!/usr/bin/env python3
"""
Colector de métricas de hardware usando powermetrics en macOS.

Requiere permisos de sudo para acceder a sensores de energía del sistema.
Proporciona mediciones precisas de CPU, GPU, ANE y consumo energético real.

Autor: Andrés Rubén Villamayor Ruiz Diaz
Fecha: 2026
"""

import subprocess
import re
import time
import json
import logging
import threading
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class PowerMetricsCollector:
    """
    Colector de métricas de hardware usando powermetrics en macOS.
    
    Usa el comando `sudo powermetrics` para obtener mediciones precisas
    de CPU, GPU, ANE (Apple Neural Engine) y consumo energético total.
    
    Atributos:
        sampling_interval_ms (int): Intervalo de muestreo en milisegundos
        metrics_history (list): Historial de muestras capturadas
        is_running (bool): Indica si la recolección está activa
        thread (threading.Thread): Hilo de recolección en background
    """
    
    def __init__(self, sampling_interval_ms: int = 100):
        """
        Inicializa el colector de métricas.
        
        Args:
            sampling_interval_ms: Intervalo entre muestras. Por defecto: 100ms
        """
        self.sampling_interval_ms = sampling_interval_ms
        self.metrics_history = []
        self.is_running = False
        self.thread = None
        self.process = None
        
        logger.info("PowerMetricsCollector inicializado (intervalo: %dms)", 
                   sampling_interval_ms)
    
    def start_collection(self):
        """
        Inicia la recolección de métricas en un hilo separado.
        
        Ejecuta `sudo powermetrics` y captura la salida en tiempo real.
        Requiere que el usuario tenga permisos de sudo configurados.
        """
        if self.is_running:
            logger.warning("La recolección ya está en ejecución")
            return
        
        logger.info("Iniciando recolección de powermetrics con sudo...")
        self.metrics_history = []
        self.is_running = True
        
        # Comando powermetrics con samplers de CPU, GPU y ANE
        cmd = [
            "sudo", "powermetrics",
            "--samplers", "cpu_power,gpu_power,ane_power",
            "-i", str(self.sampling_interval_ms),
            "--show-energy"
        ]
        
        try:
            # Iniciar proceso en segundo plano
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Iniciar hilo de procesamiento
            self.thread = threading.Thread(target=self._parse_output)
            self.thread.daemon = True
            self.thread.start()
            
            logger.info("Recolección de powermetrics iniciada (PID: %d)", self.process.pid)
            
        except Exception as e:
            logger.error("Error al iniciar powermetrics: %s", e)
            self.is_running = False
            raise
    
    def _parse_output(self):
        """
        Procesa la salida de powermetrics en tiempo real.
        
        Extrae métricas de CPU, GPU, ANE y las almacena en metrics_history.
        Se ejecuta en un hilo separado mientras dure la recolección.
        """
        try:
            while self.is_running and self.process and self.process.poll() is None:
                line = self.process.stdout.readline()
                if not line:
                    break
                
                # Parsear línea si contiene métricas
                metric = self._parse_line(line)
                if metric:
                    self.metrics_history.append(metric)
                    
        except Exception as e:
            logger.error("Error al parsear output de powermetrics: %s", e)
    
    def _parse_line(self, line: str) -> Optional[Dict]:
        """
        Parsea una línea de output de powermetrics.
        
        Args:
            line: Línea de texto del output de powermetrics
        
        Returns:
            dict: Métricas extraídas o None si no hay datos válidos
        """
        metric = {}
        
        # Buscar timestamp
        if "Sample interval" in line:
            match = re.search(r'Sample interval: (\d+) ms', line)
            if match:
                metric['interval_ms'] = int(match.group(1))
        
        # Buscar CPU Power
        if "CPU Power" in line:
            match = re.search(r'CPU Power: ([\d.]+) mW', line)
            if match:
                metric['cpu_power_mw'] = float(match.group(1))
                metric['cpu_power_w'] = float(match.group(1)) / 1000
        
        # Buscar GPU Power
        if "GPU Power" in line:
            match = re.search(r'GPU Power: ([\d.]+) mW', line)
            if match:
                metric['gpu_power_mw'] = float(match.group(1))
                metric['gpu_power_w'] = float(match.group(1)) / 1000
        
        # Buscar ANE Power
        if "ANE Power" in line:
            match = re.search(r'ANE Power: ([\d.]+) mW', line)
            if match:
                metric['ane_power_mw'] = float(match.group(1))
                metric['ane_power_w'] = float(match.group(1)) / 1000
        
        # Buscar CPU Frequency
        if "CPU Frequency" in line or "MHz" in line:
            match = re.search(r'(\d+) MHz', line)
            if match:
                metric['cpu_freq_mhz'] = int(match.group(1))
                metric['cpu_freq_ghz'] = int(match.group(1)) / 1000
        
        # Buscar CPU usage
        if "CPU usage" in line:
            match = re.search(r'CPU usage: ([\d.]+)%', line)
            if match:
                metric['cpu_percent'] = float(match.group(1))
        
        # Solo retornar si hay al menos una métrica de potencia
        if any(key in metric for key in ['cpu_power_w', 'gpu_power_w', 'ane_power_w']):
            metric['timestamp'] = time.time()
            return metric
        
        return None
    
    def stop_collection(self) -> Dict:
        """
        Detiene la recolección y calcula promedios.
        
        Returns:
            dict: Promedios de CPU, GPU, ANE y potencia total
        """
        logger.info("Deteniendo recolección de powermetrics...")
        
        self.is_running = False
        
        # Terminar proceso
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception as e:
                logger.warning("Error al terminar powermetrics: %s", e)
        
        # Esperar hilo
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        
        # Calcular promedios
        return self._calculate_averages()
    
    def _calculate_averages(self) -> Dict:
        """
        Calcula promedios de todas las métricas capturadas.
        
        Returns:
            dict: Promedios de CPU, GPU, ANE y totales
        """
        if not self.metrics_history:
            logger.warning("No hay métricas para calcular promedios")
            return {
                "cpu_power_avg_w": 0.0,
                "gpu_power_avg_w": 0.0,
                "ane_power_avg_w": 0.0,
                "total_power_avg_w": 0.0,
                "cpu_freq_avg_ghz": 0.0,
                "cpu_percent_avg": 0.0,
                "total_samples": 0
            }
        
        # Extraer valores
        cpu_powers = [m.get('cpu_power_w', 0) for m in self.metrics_history if 'cpu_power_w' in m]
        gpu_powers = [m.get('gpu_power_w', 0) for m in self.metrics_history if 'gpu_power_w' in m]
        ane_powers = [m.get('ane_power_w', 0) for m in self.metrics_history if 'ane_power_w' in m]
        cpu_freqs = [m.get('cpu_freq_ghz', 0) for m in self.metrics_history if 'cpu_freq_ghz' in m]
        cpu_percents = [m.get('cpu_percent', 0) for m in self.metrics_history if 'cpu_percent' in m]
        
        # Calcular promedios
        avg_cpu_power = sum(cpu_powers) / len(cpu_powers) if cpu_powers else 0.0
        avg_gpu_power = sum(gpu_powers) / len(gpu_powers) if gpu_powers else 0.0
        avg_ane_power = sum(ane_powers) / len(ane_powers) if ane_powers else 0.0
        avg_cpu_freq = sum(cpu_freqs) / len(cpu_freqs) if cpu_freqs else 0.0
        avg_cpu_percent = sum(cpu_percents) / len(cpu_percents) if cpu_percents else 0.0
        
        total_power = avg_cpu_power + avg_gpu_power + avg_ane_power
        
        return {
            "cpu_power_avg_w": round(avg_cpu_power, 3),
            "gpu_power_avg_w": round(avg_gpu_power, 3),
            "ane_power_avg_w": round(avg_ane_power, 3),
            "total_power_avg_w": round(total_power, 3),
            "cpu_freq_avg_ghz": round(avg_cpu_freq, 3),
            "cpu_percent_avg": round(avg_cpu_percent, 2),
            "total_samples": len(self.metrics_history),
            "raw_metrics": self.metrics_history  # Para análisis detallado
        }
    
    def get_current_metrics(self) -> Dict:
        """
        Obtiene la última muestra capturada.
        
        Returns:
            dict: Última muestra de métricas o dict vacío si no hay datos
        """
        return self.metrics_history[-1] if self.metrics_history else {}


def test_powermetrics():
    """Función de prueba para verificar que powermetrics funciona."""
    print("🔋 Probando PowerMetricsCollector...")
    print("⚠️  Se solicitará contraseña de sudo")
    
    collector = PowerMetricsCollector(sampling_interval_ms=100)
    
    try:
        collector.start_collection()
        print("✅ Recolección iniciada. Esperando 3 segundos...")
        time.sleep(3)
        
        metrics = collector.stop_collection()
        
        print("\n📊 Métricas capturadas:")
        print(f"   CPU Power: {metrics['cpu_power_avg_w']:.3f} W")
        print(f"   GPU Power: {metrics['gpu_power_avg_w']:.3f} W")
        print(f"   ANE Power: {metrics['ane_power_avg_w']:.3f} W")
        print(f"   Total Power: {metrics['total_power_avg_w']:.3f} W")
        print(f"   CPU Freq: {metrics['cpu_freq_avg_ghz']:.3f} GHz")
        print(f"   CPU Usage: {metrics['cpu_percent_avg']:.1f}%")
        print(f"   Muestras: {metrics['total_samples']}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    test_powermetrics()
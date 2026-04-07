"""
Monitor de hardware para Mac M4
"""

import psutil
import time
from dataclasses import dataclass
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

@dataclass
class HardwareSnapshot:
    """Snapshot de hardware"""
    timestamp: float
    cpu_percent: float
    cpu_per_core: List[float]
    cpu_freq_mhz: float
    cpu_temp_c: Optional[float]
    ram_used_mb: float
    ram_percent: float
    ram_available_mb: float
    gpu_active: bool

class HardwareMonitor:
    """Monitor de hardware para Mac M4"""
    
    def __init__(self):
        self.peak_ram_mb = 0.0
        self.snapshots = []
        
        try:
            import torch
            self.mps_available = torch.backends.mps.is_available()
        except:
            self.mps_available = False
        
        logger.info("🍎 HardwareMonitor inicializado")
        logger.info(f"   MPS: {'✅' if self.mps_available else '❌'}")
        logger.info(f"   CPU cores: {psutil.cpu_count()}")
    
    def capture(self):
        """Captura snapshot actual"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        cpu_freq = psutil.cpu_freq()
        cpu_freq_mhz = cpu_freq.current if cpu_freq else 0.0
        
        ram = psutil.virtual_memory()
        ram_used_mb = ram.used / (1024 ** 2)
        ram_percent = ram.percent
        ram_available_mb = ram.available / (1024 ** 2)
        
        self.peak_ram_mb = max(self.peak_ram_mb, ram_used_mb)
        
        snapshot = HardwareSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            cpu_per_core=cpu_per_core,
            cpu_freq_mhz=cpu_freq_mhz,
            cpu_temp_c=None,
            ram_used_mb=ram_used_mb,
            ram_percent=ram_percent,
            ram_available_mb=ram_available_mb,
            gpu_active=self.mps_available
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def reset(self):
        """Reset para nueva medición"""
        self.peak_ram_mb = 0.0
        self.snapshots.clear()
    
    def get_averages(self):
        """Calcula promedios"""
        if not self.snapshots:
            return {}
        
        import numpy as np
        
        cpu_percents = [s.cpu_percent for s in self.snapshots]
        cpu_freqs = [s.cpu_freq_mhz for s in self.snapshots]
        ram_useds = [s.ram_used_mb for s in self.snapshots]
        
        return {
            'cpu_percent_avg': np.mean(cpu_percents),
            'cpu_percent_std': np.std(cpu_percents),
            'cpu_freq_avg_mhz': np.mean(cpu_freqs),
            'cpu_temp_avg_c': None,
            'ram_used_avg_mb': np.mean(ram_useds),
            'ram_peak_mb': self.peak_ram_mb,
            'gpu_active': self.mps_available
        }

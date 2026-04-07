"""
Experimento protocolo Green AI
"""

import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict
import json
import logging
from tqdm import tqdm

from ..energy_tracker import EnergyTracker
from ..llm_runner import LLMRunner
from ..monitoring.hardware_monitor import HardwareMonitor

logger = logging.getLogger(__name__)

@dataclass
class RunMetrics:
    """Métricas de un run"""
    run_number: int
    model_name: str
    quantization: str
    task_type: str
    num_inferences: int
    cpu_usage_avg_pct: float
    cpu_freq_avg_ghz: float
    cpu_temp_avg_c: float
    cpu_power_avg_w: float
    ram_usage_avg_mb: float
    ram_peak_mb: float
    gpu_active_pct: float
    gpu_memory_mb: float
    gpu_power_avg_w: float
    total_energy_wh: float
    throughput_inf_per_s: float
    latency_mean_ms: float
    latency_std_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    emissions_kg_co2eq_paraguay: float
    tokens_gen_avg: float
    tokens_gen_std: float
    time_per_inf_s: float
    total_time_s: float

class ProtocolExperiment:
    """Ejecutor de experimentos protocolo Green AI"""
    
    def __init__(self, model_path: str, model_name: str, quantization: str,
                 prompts: Dict[str, List[str]], num_runs: int = 15,
                 num_inferences_per_run: int = 100, output_dir: str = "results/protocol"):
        self.model_path = model_path
        self.model_name = model_name
        self.quantization = quantization
        self.prompts = prompts
        self.num_runs = num_runs
        self.num_inferences_per_run = num_inferences_per_run
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("📦 Cargando modelo...")
        self.llm = LLMRunner(model_path=model_path, n_gpu_layers=-1, verbose=False)
        self.hw_monitor = HardwareMonitor()
        self.results = {task: [] for task in prompts.keys()}
        
        logger.info("\n" + "="*70)
        logger.info("🧪 EXPERIMENTO PROTOCOLO GREEN AI")
        logger.info("="*70)
        logger.info(f"Modelo: {model_name} ({quantization})")
        logger.info(f"Tareas: {list(prompts.keys())}")
        logger.info(f"Runs: {num_runs} x {num_inferences_per_run} inferencias")
        logger.info("="*70 + "\n")
    
    def single_run(self, task_type: str, task_prompts: List[str], run_number: int):
        """Ejecuta 1 run completo"""
        logger.info(f"\n🔬 {task_type.upper()} - Run {run_number}/{self.num_runs}")
        
        self.hw_monitor.reset()
        energy_tracker = EnergyTracker(
            project_name=f"{self.model_name}_{self.quantization}_{task_type}",
            output_dir=str(self.output_dir / "carbon_logs"),
            output_file=f"{task_type}_run_{run_number}.csv"
        )
        energy_tracker.start()
        
        latencies = []
        tokens_generated = []
        start_time = time.time()
        
        for i in tqdm(range(self.num_inferences_per_run), desc="  Inferencias", leave=False):
            prompt = task_prompts[i % len(task_prompts)]
            if i % 10 == 0:
                self.hw_monitor.capture()
            
            inf_start = time.time()
            result = self.llm.generate(prompt, max_tokens=100)
            inf_end = time.time()
            
            latencies.append(inf_end - inf_start)
            tokens_generated.append(result['tokens'])
        
        total_time = time.time() - start_time
        emissions_kg = energy_tracker.stop()
        energy_stats = energy_tracker.get_stats()
        
        latencies_np = np.array(latencies)
        tokens_np = np.array(tokens_generated)
        throughput = self.num_inferences_per_run / total_time
        hw_avg = self.hw_monitor.get_averages()
        
        cpu_load = hw_avg['cpu_percent_avg'] / 100
        cpu_power_w = 5 + (cpu_load * 15)
        gpu_power_w = cpu_load * 10 if hw_avg['gpu_active'] else 0
        
        metrics = RunMetrics(
            run_number=run_number,
            model_name=self.model_name,
            quantization=self.quantization,
            task_type=task_type,
            num_inferences=self.num_inferences_per_run,
            cpu_usage_avg_pct=hw_avg['cpu_percent_avg'],
            cpu_freq_avg_ghz=hw_avg['cpu_freq_avg_mhz'] / 1000,
            cpu_temp_avg_c=hw_avg.get('cpu_temp_avg_c') or 0.0,
            cpu_power_avg_w=cpu_power_w,
            ram_usage_avg_mb=hw_avg['ram_used_avg_mb'],
            ram_peak_mb=hw_avg['ram_peak_mb'],
            gpu_active_pct=100.0 if hw_avg['gpu_active'] else 0.0,
            gpu_memory_mb=0.0,
            gpu_power_avg_w=gpu_power_w,
            total_energy_wh=energy_stats['energy_wh'],
            throughput_inf_per_s=throughput,
            latency_mean_ms=float(np.mean(latencies_np) * 1000),
            latency_std_ms=float(np.std(latencies_np) * 1000),
            latency_p50_ms=float(np.percentile(latencies_np, 50) * 1000),
            latency_p95_ms=float(np.percentile(latencies_np, 95) * 1000),
            latency_p99_ms=float(np.percentile(latencies_np, 99) * 1000),
            emissions_kg_co2eq_paraguay=emissions_kg,
            tokens_gen_avg=float(np.mean(tokens_np)),
            tokens_gen_std=float(np.std(tokens_np)),
            time_per_inf_s=float(np.mean(latencies_np)),
            total_time_s=total_time
        )
        
        logger.info(f"  ✅ Throughput: {throughput:.2f} inf/s")
        logger.info(f"  ✅ Latencia: {metrics.latency_mean_ms:.2f} ms")
        logger.info(f"  ✅ Energía: {energy_stats['energy_wh']:.3f} Wh")
        
        return metrics
    
    def run_all_tasks(self):
        """Ejecuta todos los experimentos"""
        for task_type, task_prompts in self.prompts.items():
            logger.info(f"\n{'='*70}")
            logger.info(f"📝 TAREA: {task_type.upper()}")
            logger.info(f"{'='*70}")
            
            for run in range(1, self.num_runs + 1):
                metrics = self.single_run(task_type, task_prompts, run)
                self.results[task_type].append(metrics)
                time.sleep(2)
            
            self.save_task_results(task_type)
        
        logger.info("\n" + "="*70)
        logger.info("✅ TODOS LOS EXPERIMENTOS COMPLETADOS")
        logger.info("="*70)
    
    def save_task_results(self, task_type: str):
        """Guarda resultados de una tarea"""
        filename = f"{self.model_name}_{self.quantization}_{task_type}_results.json"
        filepath = self.output_dir / filename
        results_list = [asdict(m) for m in self.results[task_type]]
        with open(filepath, 'w') as f:
            json.dump(results_list, f, indent=2)
        logger.info(f"💾 Guardado: {filepath}")
    
    def save_all_results(self):
        """Guarda todos los resultados"""
        for task_type in self.results.keys():
            self.save_task_results(task_type)

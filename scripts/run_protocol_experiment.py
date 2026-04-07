#!/usr/bin/env python3
"""Script para ejecutar experimentos protocolo Green AI"""

import sys
from pathlib import Path
import yaml
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments import ProtocolExperiment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_prompts(yaml_path="configs/prompts.yaml"):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    prompts = {}
    for task_name, task_data in config['tasks'].items():
        prompts[task_name] = task_data['prompts']
    return prompts

def main():
    print("\n" + "="*70)
    print("🚀 EXPERIMENTO PROTOCOLO GREEN AI")
    print("="*70)
    print("📍 Mac M4 | Paraguay | 15 runs por tarea")
    print("="*70 + "\n")
    
    prompts = load_prompts()
    print(f"✅ {len(prompts)} tareas cargadas:")
    for task, task_prompts in prompts.items():
        print(f"   - {task}: {len(task_prompts)} prompts")
    
    experiment = ProtocolExperiment(
        model_path="models/llama-2-7b.Q8_0.gguf",
        model_name="llama-2-7b",
        quantization="Q8_0",
        prompts=prompts,
        num_runs=15,
        num_inferences_per_run=100,
        output_dir="results/protocol"
    )
    
    experiment.run_all_tasks()
    experiment.save_all_results()
    
    print("\n✅ ¡Completado! Resultados en: results/protocol/")

if __name__ == "__main__":
    main()

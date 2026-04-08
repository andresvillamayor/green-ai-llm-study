#!/usr/bin/env python3
"""Test rápido del protocolo (2 runs, 10 inferencias)"""

import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments import ProtocolExperiment

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def main():
    print("\n" + "="*60)
    print("🧪 TEST RÁPIDO DEL PROTOCOLO")
    print("="*60)
    print("⚠️  Solo 2 tareas, 2 runs, 10 inferencias")
    print("⏱️  Tiempo estimado: 2-3 minutos")
    print("="*60 + "\n")
    
    prompts = {
        "texto": [
            "¿Qué es la inteligencia artificial?",
            "Explica la fotosíntesis brevemente."
        ],
        "matematicas": [
            "¿Cuánto es 5 + 7?",
            "Resuelve: x + 3 = 10"
        ]
    }
    
    experiment = ProtocolExperiment(
        model_path="models/llama-2-7b.Q8_0.gguf",
        model_name="llama-2-7b",
        quantization="Q8_0",
        prompts=prompts,
        num_runs=2,
        num_inferences_per_run=10,
        output_dir="results/test_protocol"
    )
    
    experiment.run_all_tasks()
    experiment.save_all_results()
    
    print("\n✅ Test completado!")
    print("📂 Resultados en: results/test_protocol/")

if __name__ == "__main__":
    main()

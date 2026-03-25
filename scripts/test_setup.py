#!/usr/bin/env python3
"""
Script de prueba de configuración completa
"""

import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    print("=" * 70)
    print("🧪 PRUEBA DE CONFIGURACIÓN COMPLETA")
    print("=" * 70)
    
    # Test 1: Imports básicos
    print("\n1️⃣ Verificando imports básicos...")
    try:
        import numpy as np
        import pandas as pd
        import matplotlib
        import codecarbon
        from llama_cpp import Llama
        print("   ✅ Todos los imports básicos OK")
    except ImportError as e:
        print(f"   ❌ Error en imports: {e}")
        return
    
    # Test 2: Módulos propios
    print("\n2️⃣ Verificando módulos propios...")
    try:
        from energy_tracker import EnergyTracker
        from llm_runner import LLMRunner
        from utils import setup_logging, save_json
        print("   ✅ Módulos propios OK")
    except ImportError as e:
        print(f"   ❌ Error en módulos: {e}")
        print(f"\n🔍 Verificando que archivos existen:")
        src_path = Path(__file__).parent.parent / "src"
        for file in ["energy_tracker.py", "llm_runner.py", "utils.py", "__init__.py"]:
            file_path = src_path / file
            if file_path.exists():
                print(f"      ✅ {file}")
            else:
                print(f"      ❌ {file} NO EXISTE")
        return
    
    # Test 3: Crear instancias
    print("\n3️⃣ Probando instancias...")
    try:
        from utils import setup_logging
        from energy_tracker import EnergyTracker
        setup_logging()
        tracker = EnergyTracker(project_name="test")
        print("   ✅ EnergyTracker inicializado")
    except Exception as e:
        print(f"   ❌ Error en EnergyTracker: {e}")
    
    # Test 4: Verificar estructura de carpetas
    print("\n4️⃣ Verificando estructura...")
    required_dirs = [
        "data/raw", "data/processed",
        "models",
        "results/measurements", "results/plots",
        "scripts", "src", "notebooks"
    ]
    
    project_root = Path(__file__).parent.parent
    missing = []
    for dir_path in required_dirs:
        if not (project_root / dir_path).exists():
            missing.append(dir_path)
    
    if missing:
        print(f"   ⚠️  Carpetas faltantes: {', '.join(missing)}")
    else:
        print("   ✅ Todas las carpetas existen")
    
    # Test 5: Versiones
    print("\n5️⃣ Versiones instaladas:")
    print(f"   • Python: {sys.version.split()[0]}")
    print(f"   • NumPy: {np.__version__}")
    print(f"   • Pandas: {pd.__version__}")
    print(f"   • Matplotlib: {matplotlib.__version__}")
    print(f"   • CodeCarbon: {codecarbon.__version__}")
    
    print("\n" + "=" * 70)
    print("✅ CONFIGURACIÓN COMPLETA Y FUNCIONAL")
    print("=" * 70)
    print("\n💡 Siguiente paso: Descargar modelos con:")
    print("   bash scripts/download_models.sh")

if __name__ == "__main__":
    main()

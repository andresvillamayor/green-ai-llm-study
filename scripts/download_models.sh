#!/bin/bash

echo "======================================"
echo "📥 DESCARGA DE MODELOS GGUF"
echo "======================================"

# Verificar que huggingface-cli está instalado
if ! command -v huggingface-cli &> /dev/null; then
    echo "❌ huggingface-cli no encontrado"
    echo "💡 Instalando..."
    pip install huggingface-hub
fi

# Crear carpeta de modelos
mkdir -p models

echo ""
echo "Opciones de modelos:"
echo "1) Llama 2 7B Q8_0 (~7.2GB) - Recomendado para empezar"
echo "2) Llama 2 7B Q4_0 (~3.8GB) - Más ligero, menos calidad"
echo "3) Ambos modelos"
echo ""
read -p "Selecciona opción (1-3): " choice

case $choice in
  1)
    echo ""
    echo "📦 Descargando Llama 2 7B Q8_0..."
    echo "⏳ Esto tomará 10-20 minutos (7.2GB)"
    echo ""
    huggingface-cli download \
      TheBloke/Llama-2-7B-GGUF \
      llama-2-7b.Q8_0.gguf \
      --local-dir models/ \
      --local-dir-use-symlinks False
    ;;
  2)
    echo ""
    echo "📦 Descargando Llama 2 7B Q4_0..."
    echo "⏳ Esto tomará 5-10 minutos (3.8GB)"
    echo ""
    huggingface-cli download \
      TheBloke/Llama-2-7B-GGUF \
      llama-2-7b.Q4_0.gguf \
      --local-dir models/ \
      --local-dir-use-symlinks False
    ;;
  3)
    echo ""
    echo "📦 Descargando ambos modelos..."
    echo "⚠️  Esto descargará ~11GB de datos"
    echo "⏳ Esto tomará 20-40 minutos"
    echo ""
    
    echo "Descargando Q8_0..."
    huggingface-cli download \
      TheBloke/Llama-2-7B-GGUF \
      llama-2-7b.Q8_0.gguf \
      --local-dir models/ \
      --local-dir-use-symlinks False
    
    echo ""
    echo "Descargando Q4_0..."
    huggingface-cli download \
      TheBloke/Llama-2-7B-GGUF \
      llama-2-7b.Q4_0.gguf \
      --local-dir models/ \
      --local-dir-use-symlinks False
    ;;
  *)
    echo "❌ Opción inválida"
    exit 1
    ;;
esac

echo ""
echo "========================================"
echo "✅ Descarga completada"
echo "========================================"
echo ""
echo "📁 Modelos descargados:"
ls -lh models/*.gguf 2>/dev/null || echo "⚠️  No se encontraron modelos GGUF"
echo ""
echo "🚀 Siguiente paso: Ejecutar primera medición con:"
echo "   python scripts/measure_energy.py"
echo ""

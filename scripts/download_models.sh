#!/bin/bash
#
# download_models.sh
# Script para descargar modelos Qwen 2.5 7B cuantizados en formato GGUF.
# Los archivos se guardan en models/qwen2.5-7b/ para mantener la estructura del proyecto.
#
# Uso: bash scripts/download_models.sh
#

set -e  # Salir inmediatamente si un comando falla

echo "======================================"
echo "Descarga de modelos GGUF - Qwen 2.5 7B"
echo "======================================"

# Verificar que huggingface-cli está instalado
if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli no encontrado."
    echo "Solucion: pip install -U huggingface-hub"
    exit 1
fi

# Definir directorio de destino
MODEL_DIR="models/qwen2.5-7b"
mkdir -p "$MODEL_DIR"

REPO="bartowski/Qwen2.5-7B-Instruct-GGUF"
MODEL_INT4="Qwen2.5-7B-Instruct-Q4_K_M.gguf"
MODEL_INT8="Qwen2.5-7B-Instruct-Q8_0.gguf"

echo ""
echo "Opciones de descarga:"
echo "1) INT4 (Q4_K_M) - Aprox. 4.7 GB (mayor velocidad, menor consumo)"
echo "2) INT8 (Q8_0)  - Aprox. 8.1 GB (mayor precision, menor velocidad)"
echo "3) Ambos modelos"
echo ""
read -p "Selecciona una opcion (1-3): " choice

case $choice in
  1)
    echo ""
    echo "Descargando modelo INT4..."
    huggingface-cli download "$REPO" "$MODEL_INT4" \
      --local-dir "$MODEL_DIR" \
      --local-dir-use-symlinks False
    ;;
  2)
    echo ""
    echo "Descargando modelo INT8..."
    huggingface-cli download "$REPO" "$MODEL_INT8" \
      --local-dir "$MODEL_DIR" \
      --local-dir-use-symlinks False
    ;;
  3)
    echo ""
    echo "Descargando ambos modelos (INT4 e INT8)..."
    echo "Nota: Esto descargara aproximadamente 12.8 GB."
    huggingface-cli download "$REPO" "$MODEL_INT4" "$MODEL_INT8" \
      --local-dir "$MODEL_DIR" \
      --local-dir-use-symlinks False
    ;;
  *)
    echo "Error: Opcion invalida."
    exit 1
    ;;
esac

echo ""
echo "======================================"
echo "Descarga completada."
echo "======================================"
echo ""
echo "Archivos descargados:"
ls -lh "$MODEL_DIR"/*.gguf 2>/dev/null || echo "Advertencia: No se encontraron archivos GGUF."
echo ""
echo "Proximo paso: Ejecutar el benchmark comparativo con:"
echo "   python scripts/run_benchmark.py"
echo ""
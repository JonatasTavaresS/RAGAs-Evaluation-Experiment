#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Executando Preparação de Dados ==="
echo "Diretório Raiz: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
    echo "Ativando ambiente virtual em .venv..."
    source .venv/bin/activate
fi

python src/prepare_dataset.py

#!/bin/bash
# Quick Start: Model Service Setup & Launch
# 
# This script sets up the model analysis service for cross-dataset validation.
# 
# Usage:
#   ./setup_model_service.sh          # Install dependencies
#   ./setup_model_service.sh demo     # Start in demo mode (simulation)
#   ./setup_model_service.sh prod     # Start in production mode (real model required)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODEL_SERVICE_DIR="$SCRIPT_DIR/model-service"
VENV_DIR="$MODEL_SERVICE_DIR/venv"

echo "=========================================="
echo "Model Service Setup & Launch"
echo "=========================================="
echo ""

# Function to create virtual environment
setup_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        echo "📦 Creating Python virtual environment..."
        cd "$MODEL_SERVICE_DIR"
        python3 -m venv venv
        
        echo "⬇️  Installing dependencies..."
        source venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
        deactivate
        
        echo "✓ Virtual environment ready"
    else
        echo "✓ Virtual environment already exists"
    fi
}

# Function to start service in demo mode (simulation)
start_demo() {
    echo ""
    echo "🚀 Starting Model Service in DEMO MODE (simulation)"
    echo ""
    echo "   Model checkpoint: NOT REQUIRED (using simulation)"
    echo "   GPU: Not required"
    echo "   Service URL: http://localhost:8001"
    echo ""
    
    cd "$MODEL_SERVICE_DIR"
    source venv/bin/activate
    
    echo "📡 Launching service..."
    uvicorn src.serve.service:app \
        --host 0.0.0.0 \
        --port 8001 \
        --reload
}

# Function to start service in production mode (real model)
start_prod() {
    echo ""
    echo "🚀 Starting Model Service in PRODUCTION MODE (real model)"
    echo ""
    
    # Check if checkpoint exists
    if [ ! -f "$MODEL_SERVICE_DIR/checkpoints/best_model.pth" ]; then
        echo "❌ ERROR: Model checkpoint not found at:"
        echo "   $MODEL_SERVICE_DIR/checkpoints/best_model.pth"
        echo ""
        echo "To use production mode, you need a trained model checkpoint."
        echo ""
        echo "Options:"
        echo "  1. Train a model:"
        echo "     python src/train.py --data-dir data/faceforensics++ --epochs 50"
        echo ""
        echo "  2. Use demo mode instead:"
        echo "     ./setup_model_service.sh demo"
        echo ""
        exit 1
    fi
    
    echo "   Model checkpoint: $(ls -lh $MODEL_SERVICE_DIR/checkpoints/best_model.pth | awk '{print $5}')"
    echo "   Device: GPU (CUDA) if available, otherwise CPU"
    echo "   Service URL: http://localhost:8001"
    echo ""
    
    cd "$MODEL_SERVICE_DIR"
    source venv/bin/activate
    
    echo "📡 Launching service..."
    DEVICE=cuda PORT=8001 uvicorn src.serve.service:app \
        --host 0.0.0.0 \
        --port 8001 \
        --workers 1
}

# Main logic
case "${1:-setup}" in
    setup|install)
        echo "Setting up model service..."
        setup_venv
        ;;
    demo|dev|test)
        setup_venv
        start_demo
        ;;
    prod|production)
        setup_venv
        start_prod
        ;;
    *)
        echo "Usage: $0 {setup|demo|prod}"
        echo ""
        echo "Commands:"
        echo "  setup       Install dependencies (default)"
        echo "  demo        Start in demo/test mode (simulation)"
        echo "  prod        Start in production mode (requires checkpoint)"
        echo ""
        exit 1
        ;;
esac

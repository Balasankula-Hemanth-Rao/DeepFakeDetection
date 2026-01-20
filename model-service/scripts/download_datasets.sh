#!/bin/bash

# Quick Start Script for Dataset Download
# This script guides you through the dataset download process

echo "=========================================="
echo "Dataset Download Quick Start"
echo "=========================================="
echo ""

# Check if running in correct directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Please run this script from the model-service directory"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements-datasets.txt
echo "✓ Dependencies installed"
echo ""

# Check for credentials
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found"
    echo ""
    echo "To download datasets, you need credentials:"
    echo ""
    echo "1. FaceForensics++:"
    echo "   - Visit: https://github.com/ondyari/FaceForensics"
    echo "   - Request access with academic email"
    echo "   - Wait for approval (1-2 days)"
    echo ""
    echo "2. Celeb-DF:"
    echo "   - Visit: https://github.com/yuezunli/celeb-deepfakeforensics"
    echo "   - Fill out access form"
    echo "   - Download from provided Google Drive links"
    echo ""
    echo "After receiving credentials, create .env file:"
    cp .env.example .env
    echo "✓ Created .env template - please fill in your credentials"
    echo ""
    read -p "Press Enter after filling in credentials..."
fi

# Menu
echo ""
echo "What would you like to do?"
echo ""
echo "1) Download FaceForensics++ (c23, ~38GB)"
echo "2) Download FaceForensics++ (c40, ~10GB, lower quality)"
echo "3) Verify existing dataset"
echo "4) Organize dataset into train/val/test"
echo "5) View download guide"
echo "6) Exit"
echo ""
read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo ""
        echo "Downloading FaceForensics++ (c23 compression)..."
        python scripts/download_datasets.py \
            --dataset faceforensics \
            --output data/ \
            --compression c23 \
            --type all
        ;;
    2)
        echo ""
        echo "Downloading FaceForensics++ (c40 compression)..."
        python scripts/download_datasets.py \
            --dataset faceforensics \
            --output data/ \
            --compression c40 \
            --type all
        ;;
    3)
        echo ""
        echo "Verifying dataset..."
        python scripts/verify_dataset.py \
            --data-root data/deepfake/
        ;;
    4)
        echo ""
        echo "Organizing dataset..."
        python scripts/download_datasets.py \
            --dataset faceforensics \
            --output data/ \
            --organize
        ;;
    5)
        echo ""
        cat DATASET_DOWNLOAD_GUIDE.md | less
        ;;
    6)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Verify dataset: python scripts/verify_dataset.py --data-root data/deepfake/"
echo "2. Start training: python src/train.py --config config/model_config.yaml"
echo ""

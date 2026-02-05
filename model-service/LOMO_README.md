# LOMO Evaluation System - Quick Start Guide

This directory contains the Leave-One-Method-Out (LOMO) evaluation infrastructure for multimodal deepfake detection research.

## 📁 Files Created

| File | Purpose |
|------|---------|
| `scripts/lomo_dataset_organizer.py` | Organize FaceForensics++ by method, create split configs |
| `src/datasets/multimodal_lomo_dataset.py` | PyTorch dataset for LOMO training |
| `src/train_lomo.py` | Train model with LOMO protocol |
| `src/eval/lomo_evaluator.py` | Evaluate model with comprehensive metrics |

## 🚀 Quick Start

### Step 1: Organize Dataset

```bash
cd model-service

# Organize FaceForensics++ by manipulation method
python scripts/lomo_dataset_organizer.py \
    --input ../FaceForensics-master \
    --output data/lomo \
    --verbose
```

**Output:**
- `data/lomo/DeepFakes/` - DeepFakes manipulated videos
- `data/lomo/FaceSwap/` - FaceSwap manipulated videos
- `data/lomo/Face2Face/` - Face2Face manipulated videos
- `data/lomo/NeuralTextures/` - NeuralTextures manipulated videos
- `data/lomo/original/` - Original real videos
- `configs/lomo_splits/lomo_split_*.json` - 4 split configurations

### Step 2: Extract Audio (Required)

```bash
# Extract audio for each manipulation method
python scripts/extract_audio_multimodal.py \
    --video-dir data/lomo/DeepFakes \
    --output-dir data/lomo_audio/DeepFakes \
    --label fake

python scripts/extract_audio_multimodal.py \
    --video-dir data/lomo/FaceSwap \
    --output-dir data/lomo_audio/FaceSwap \
    --label fake

python scripts/extract_audio_multimodal.py \
    --video-dir data/lomo/Face2Face \
    --output-dir data/lomo_audio/Face2Face \
    --label fake

python scripts/extract_audio_multimodal.py \
    --video-dir data/lomo/NeuralTextures \
    --output-dir data/lomo_audio/NeuralTextures \
    --label fake

python scripts/extract_audio_multimodal.py \
    --video-dir data/lomo/original \
    --output-dir data/lomo_audio/original \
    --label real
```

### Step 3: Train LOMO Split

```bash
# Train Split 1 (excludes DeepFakes for testing)
python src/train_lomo.py \
    --split-config configs/lomo_splits/lomo_split_1_test_deepfakes.json \
    --output checkpoints/lomo_split_1/ \
    --epochs 10 \
    --batch-size 32
```

**What Happens:**
- Trains on FaceSwap, Face2Face, NeuralTextures
- Validates on same methods
- Saves checkpoints to `checkpoints/lomo_split_1/`

### Step 4: Evaluate on Excluded Method

```bash
# Test on DeepFakes (unseen method)
python src/eval/lomo_evaluator.py \
    --checkpoint checkpoints/lomo_split_1/best.pth \
    --split-config configs/lomo_splits/lomo_split_1_test_deepfakes.json \
    --output results/lomo_split_1/ \
    --identify-failures
```

**Output:**
```
EVALUATION RESULTS: LOMO Split 1: Test on DeepFakes
============================================================
Test Method (Unseen): DeepFakes
Number of Samples: 6,545

Performance Metrics:
------------------------------------------------------------
  Accuracy:     0.8234 (82.34%)
  Precision:    0.8145
  Recall:       0.8423
  F1-Score:     0.8281
  AUC-ROC:      0.8756
  Specificity:  0.8045
  Sensitivity:  0.8423
------------------------------------------------------------
```

## 📊 LOMO Splits Overview

| Split | Train Methods | Test Method | Config File |
|-------|---------------|-------------|-------------|
| 1 | FaceSwap, Face2Face, NeuralTextures | DeepFakes | `lomo_split_1_test_deepfakes.json` |
| 2 | DeepFakes, Face2Face, NeuralTextures | FaceSwap | `lomo_split_2_test_faceswap.json` |
| 3 | DeepFakes, FaceSwap, NeuralTextures | Face2Face | `lomo_split_3_test_face2face.json` |
| 4 | DeepFakes, FaceSwap, Face2Face | NeuralTextures | `lomo_split_4_test_neuraltextures.json` |

## 🔄 Complete Workflow

```bash
# 1. Organize dataset (one time)
python scripts/lomo_dataset_organizer.py --input ../FaceForensics-master --output data/lomo

# 2. Extract audio (one time)
# Run audio extraction commands from Step 2 above

# 3. Train all 4 splits
for i in 1 2 3 4; do
    python src/train_lomo.py \
        --split-config configs/lomo_splits/lomo_split_${i}_*.json \
        --output checkpoints/lomo_split_${i}/ \
        --epochs 10
done

# 4. Evaluate all splits
for i in 1 2 3 4; do
    python src/eval/lomo_evaluator.py \
        --checkpoint checkpoints/lomo_split_${i}/best.pth \
        --split-config configs/lomo_splits/lomo_split_${i}_*.json \
        --output results/lomo_split_${i}/
done

# 5. Aggregate results
python src/eval/lomo_evaluator.py \
    --aggregate \
    --results-dir results/ \
    --output results/lomo_summary.json
```

## 📈 Metrics Tracked

For each LOMO split, the evaluator computes:

- **Accuracy**: Overall classification accuracy
- **Precision**: Precision for fake detection
- **Recall**: Recall for fake detection (sensitivity)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve
- **Specificity**: True negative rate
- **Sensitivity**: True positive rate (same as recall)
- **Confusion Matrix**: 2x2 matrix of predictions
- **ROC Curve**: Full ROC curve data for plotting
- **Failure Cases**: Top-20 misclassified samples

## 🧪 Testing the Implementation

```bash
# Test dataset organization
python scripts/lomo_dataset_organizer.py \
    --input ../FaceForensics-master \
    --output data/lomo_test \
    --verbose

# Test dataset loader
python src/datasets/multimodal_lomo_dataset.py \
    configs/lomo_splits/lomo_split_1_test_deepfakes.json

# Test training (1 epoch, small dataset)
python src/train_lomo.py \
    --split-config configs/lomo_splits/lomo_split_1_test_deepfakes.json \
    --output checkpoints/test/ \
    --epochs 1 \
    --batch-size 16 \
    --max-samples 100
```

## 📖 Configuration File Format

Each LOMO split configuration (`lomo_split_*.json`) contains:

```json
{
  "split_name": "LOMO Split 1: Test on DeepFakes",
  "split_number": 1,
  "test_method": "DeepFakes",
  "test_method_abbrev": "DF",
  "train_methods": ["FaceSwap", "Face2Face", "NeuralTextures"],
  "train_methods_abbrev": ["FS", "F2F", "NT"],
  "data_dir": "e:/project/aura-veracity-lab/model-service/data/lomo",
  "test_method_dir": "e:/project/aura-veracity-lab/model-service/data/lomo/DeepFakes",
  "original_dir": "e:/project/aura-veracity-lab/model-service/data/lomo/original",
  "description": "Train on FaceSwap, Face2Face, NeuralTextures and test generalization on unseen DeepFakes method"
}
```

## 🎯 Research Goal

**LOMO Protocol** tests whether a model trained on manipulation methods A, B, C can generalize to detect unseen method D. This evaluates the model's ability to:

1. Learn general deepfake artifacts (not method-specific)
2. Generalize to new manipulation techniques
3. Avoid overfitting to training method characteristics

## 📝 Output Files

### Training
- `checkpoints/lomo_split_X/epoch_*.pth` - Epoch checkpoints
- `checkpoints/lomo_split_X/best.pth` - Best validation accuracy
- `checkpoints/lomo_split_X/final.pth` - Final epoch
- `checkpoints/lomo_split_X/training_history.json` - Training curves

### Evaluation
- `results/lomo_split_X/evaluation.json` - Full metrics and predictions
- `results/lomo_summary.json` - Aggregated results across all splits

## 🔍 Next Steps

After implementing LOMO evaluation:

1. **Modality Ablation**: Compare video-only vs audio-only vs multimodal
2. **Cross-Dataset Validation**: Test on FakeAVCeleb/DFDC
3. **Compression Robustness**: Test on compressed videos
4. **Visualization**: Generate ROC curves, confusion matrices
5. **Paper**: Document results for journal submission

## 📚 Documentation

- **Implementation Plan**: `C:\Users\heman\.gemini\antigravity\brain\...\implementation_plan.md`
- **Step-by-Step Guide**: See implementation_plan.md for detailed workflow
- **Project Reflection**: Analysis of research goals and contributions

## ⚠️ Important Notes

1. **Audio Extraction Required**: Run audio extraction before training
2. **Storage**: LOMO organization uses symlinks by default (saves space)
3. **GPU Recommended**: Training is slow on CPU (~10x slower)
4. **Batch Size**: Reduce if out of memory (32 → 16 → 8)

## 🐛 Troubleshooting

**Issue: "Split config not found"**
- Run `lomo_dataset_organizer.py` first to create configs

**Issue: "Audio file not found"**
- Run `extract_audio_multimodal.py` for all methods

**Issue: "No videos found"**
- Check `--input` path points to FaceForensics++ root
- Verify dataset structure has `manipulated_sequences/` folder

**Issue: Out of memory**
- Reduce `--batch-size` (try 16 or 8)
- Reduce `--frames-per-video` (try 5 instead of 10)

---

**Created:** February 4, 2026  
**Last Updated:** February 4, 2026  
**Status:** ✅ Ready for LOMO training

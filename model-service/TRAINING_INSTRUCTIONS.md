# LOMO Training & Evaluation — Step by Step Instructions

## What Is This Folder?

This folder contains everything needed to:
1. **Train** the multimodal deepfake detection model using the LOMO protocol (4 training runs)
2. **Evaluate** each trained model and generate the paper's result tables

The dataset (FaceForensics++ c23) is included in `data/ffpp/`.
The pre-trained checkpoint (for ablation/cross-dataset use) is in `checkpoints/final.pth`.

---

## Machine Requirements

| Requirement | Minimum |
|---|---|
| GPU | 8GB VRAM (RTX 3060 / RTX 4070 or better) |
| RAM | 16GB |
| Disk | 20GB free (data already included) |
| Python | 3.9 or 3.10 |
| CUDA | 11.8 or 12.x |

---

## Step 0 — Setup (Do Once)

Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
```

Verify your GPU is visible:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output: `True` and your GPU name (e.g. `NVIDIA GeForce RTX 4070`).

---

## Step 1 — Generate LOMO Split Configs

```bash
python generate_lomo_configs.py --ffpp-root data/ffpp --output configs/lomo_splits
```

This creates 4 JSON files in `configs/lomo_splits/` — one per held-out method.
They are already included but re-running this updates paths to this machine.

---

## Step 2 — Organize Data

```bash
python prepare_ff_data.py --ffpp-root data/ffpp --output data/eval_ready --copy
```

This organizes the raw FF++ videos into the structure the evaluator expects:
```
data/eval_ready/
  real/                          ← 140 test videos (IDs 860-999)
  fake/                          ← 560 test videos (all 4 methods)
  lomo/
    leave_out_Deepfakes/
    leave_out_Face2Face/
    leave_out_FaceSwap/
    leave_out_NeuralTextures/
```

---

## Step 3 — Train 4 LOMO Splits

Run each command below. Each takes ~2-3 hours on RTX 4070.
**Run them one after the other** (not in parallel — GPU memory).

```bash
python src/train_lomo.py --split-config configs/lomo_splits/split_1_test_Deepfakes.json     --output checkpoints/lomo_1/ --epochs 10

python src/train_lomo.py --split-config configs/lomo_splits/split_2_test_Face2Face.json      --output checkpoints/lomo_2/ --epochs 10

python src/train_lomo.py --split-config configs/lomo_splits/split_3_test_FaceSwap.json       --output checkpoints/lomo_3/ --epochs 10

python src/train_lomo.py --split-config configs/lomo_splits/split_4_test_NeuralTextures.json --output checkpoints/lomo_4/ --epochs 10
```

Checkpoints are saved to `checkpoints/lomo_N/best.pth` (best validation AUC).

---

## Step 4 — Evaluate (Generates Paper Results)

Run these after ALL 4 training jobs finish.

### LOMO Table (Main Paper Contribution)

Each command tests one held-out method using the checkpoint trained WITHOUT that method:

```bash
python eval_pipeline.py --data-dir data/eval_ready/lomo/leave_out_Deepfakes      --checkpoint checkpoints/lomo_1/best.pth --tag lomo_Deepfakes      --output results/lomo_Deepfakes.json

python eval_pipeline.py --data-dir data/eval_ready/lomo/leave_out_Face2Face       --checkpoint checkpoints/lomo_2/best.pth --tag lomo_Face2Face       --output results/lomo_Face2Face.json

python eval_pipeline.py --data-dir data/eval_ready/lomo/leave_out_FaceSwap        --checkpoint checkpoints/lomo_3/best.pth --tag lomo_FaceSwap        --output results/lomo_FaceSwap.json

python eval_pipeline.py --data-dir data/eval_ready/lomo/leave_out_NeuralTextures  --checkpoint checkpoints/lomo_4/best.pth --tag lomo_NeuralTextures  --output results/lomo_NeuralTextures.json
```

### Modality Ablation Table (Video-Only vs Audio-Only vs Multimodal)

Uses the existing `final.pth` checkpoint:

```bash
python eval_pipeline.py --data-dir data/eval_ready --checkpoint checkpoints/final.pth --tag ablation --output results/ablation.json
```

---

## Step 5 — Read Your Results

All results are saved as JSON files in the `results/` folder.
Each file contains:

```json
{
  "results": {
    "video_only":  { "auc": 0.93, "accuracy": 0.89, "f1": 0.88, "eer": 0.082 },
    "audio_only":  { "auc": 0.85, "accuracy": 0.81, "f1": 0.80, "eer": 0.121 },
    "multimodal":  { "auc": 0.96, "accuracy": 0.94, "f1": 0.93, "eer": 0.053 }
  }
}
```

Copy these numbers directly into your paper's results table.

---

## After Training — Copy Results Back

Copy the following back to your laptop:

```
checkpoints/lomo_1/best.pth
checkpoints/lomo_2/best.pth
checkpoints/lomo_3/best.pth
checkpoints/lomo_4/best.pth
results/lomo_Deepfakes.json
results/lomo_Face2Face.json
results/lomo_FaceSwap.json
results/lomo_NeuralTextures.json
results/ablation.json
```

---

## Troubleshooting

### "CUDA out of memory"
Reduce batch size:
```bash
python src/train_lomo.py ... --batch-size 4
```

### "Module not found"
```bash
pip install -r requirements.txt
```

### Training is slow / seems stuck
Check GPU is being used:
```bash
nvidia-smi
```
Should show near 100% GPU utilization. If 0%, CUDA is not set up — reinstall PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Download script errors
Data is already included in `data/ffpp/` — no download needed.

---

## File Layout

```
aura_lomo_training/
├── TRAINING_INSTRUCTIONS.md     ← This file
├── requirements.txt
├── eval_pipeline.py             ← Evaluation script
├── prepare_ff_data.py           ← Data organizer
├── generate_lomo_configs.py     ← Config generator
├── src/                         ← Model source code
│   ├── models/                  ← MultimodalModel, fusion, audio encoder
│   ├── datasets/                ← LOMO dataset loader
│   ├── train_lomo.py            ← LOMO training script
│   └── preprocessing/           ← Audio processor
├── scripts/                     ← Helper scripts
├── configs/lomo_splits/         ← 4 LOMO split JSON configs
├── checkpoints/
│   └── final.pth                ← Pre-trained model (for ablation)
└── data/ffpp/                   ← FaceForensics++ c23 dataset
    ├── original_sequences/      ← 1000 real videos
    └── manipulated_sequences/   ← 4000 fake videos
        ├── Deepfakes/
        ├── Face2Face/
        ├── FaceSwap/
        └── NeuralTextures/
```

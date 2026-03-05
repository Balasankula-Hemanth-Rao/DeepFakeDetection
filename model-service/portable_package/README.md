# Aura Veracity Lab — LOMO Training Package

## What's In This Folder

```
portable_package/
├── README.md                     ← You are here
├── TRAINING_INSTRUCTIONS.md      ← Detailed step-by-step guide
├── requirements.txt              ← Python dependencies
│
├── eval_pipeline.py              ← Runs evaluation, produces paper results
├── prepare_ff_data.py            ← Organizes FF++ data for evaluation
├── generate_lomo_configs.py      ← Generates LOMO split JSON configs
│
├── src/                          ← Model source code (DO NOT EDIT)
│   ├── models/                   ← MultimodalModel, EfficientNet+Wav2Vec2 fusion
│   ├── datasets/                 ← LOMO dataset loader (reads FF++ videos)
│   ├── train_lomo.py             ← LOMO training script
│   └── preprocessing/            ← Audio spectrogram processor
│
├── scripts/                      ← Helper scripts
├── configs/lomo_splits/          ← 4 LOMO split JSON configs (pre-generated)
│   ├── split_1_test_Deepfakes.json
│   ├── split_2_test_Face2Face.json
│   ├── split_3_test_FaceSwap.json
│   └── split_4_test_NeuralTextures.json
│
├── checkpoints/
│   └── final.pth                 ← Pre-trained model (~132MB), use for ablation eval
│
└── data/
    └── ffpp/                     ← FaceForensics++ c23 dataset (~8.6GB)
        ├── original_sequences/youtube/c23/videos/       (1000 real videos)
        └── manipulated_sequences/
            ├── Deepfakes/c23/videos/                    (1000 fake videos)
            ├── Face2Face/c23/videos/                    (1000 fake videos)
            ├── FaceSwap/c23/videos/                     (1000 fake videos)
            └── NeuralTextures/c23/videos/               (1000 fake videos)
```

---

## Machine Requirements

| Requirement | Value |
|---|---|
| GPU | 8GB VRAM (RTX 3060 / RTX 4070 or better) |
| RAM | 16GB |
| Python | 3.9 or 3.10 |
| CUDA | 11.8 or 12.x |

---

## Quick Start (4 Commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Organize data + generate configs
python generate_lomo_configs.py --ffpp-root data/ffpp --output configs/lomo_splits
python prepare_ff_data.py --ffpp-root data/ffpp --output data/eval_ready --copy

# 3. Train 4 LOMO splits  (~2-3 hrs each on RTX 4070)
python src/train_lomo.py --split-config configs/lomo_splits/split_1_test_Deepfakes.json     --output checkpoints/lomo_1/ --epochs 10
python src/train_lomo.py --split-config configs/lomo_splits/split_2_test_Face2Face.json      --output checkpoints/lomo_2/ --epochs 10
python src/train_lomo.py --split-config configs/lomo_splits/split_3_test_FaceSwap.json       --output checkpoints/lomo_3/ --epochs 10
python src/train_lomo.py --split-config configs/lomo_splits/split_4_test_NeuralTextures.json --output checkpoints/lomo_4/ --epochs 10

# 4. Evaluate — produces paper result JSON files
python eval_pipeline.py --data-dir data/eval_ready/lomo/leave_out_Deepfakes      --checkpoint checkpoints/lomo_1/best.pth --tag lomo_Deepfakes      --output results/lomo_Deepfakes.json
python eval_pipeline.py --data-dir data/eval_ready/lomo/leave_out_Face2Face       --checkpoint checkpoints/lomo_2/best.pth --tag lomo_Face2Face       --output results/lomo_Face2Face.json
python eval_pipeline.py --data-dir data/eval_ready/lomo/leave_out_FaceSwap        --checkpoint checkpoints/lomo_3/best.pth --tag lomo_FaceSwap        --output results/lomo_FaceSwap.json
python eval_pipeline.py --data-dir data/eval_ready/lomo/leave_out_NeuralTextures  --checkpoint checkpoints/lomo_4/best.pth --tag lomo_NeuralTextures  --output results/lomo_NeuralTextures.json
```

---

## What to Send Back

After training and evaluation, copy these files back to the laptop:

```
checkpoints/lomo_1/best.pth
checkpoints/lomo_2/best.pth
checkpoints/lomo_3/best.pth
checkpoints/lomo_4/best.pth
results/lomo_Deepfakes.json
results/lomo_Face2Face.json
results/lomo_FaceSwap.json
results/lomo_NeuralTextures.json
results/ablation.json          (optional — run with checkpoints/final.pth)
```

The JSON files in `results/` are the paper's result tables. Numbers go directly into the paper.

---

## For Detailed Instructions

Read **TRAINING_INSTRUCTIONS.md** — it has full explanations, troubleshooting, and notes on each step.

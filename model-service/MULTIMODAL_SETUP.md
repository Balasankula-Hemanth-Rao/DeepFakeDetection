# Multimodal Deepfake Detection - Setup & Training Guide

## Overview

This guide walks you through setting up and training a multimodal deepfake detection model using video frames + audio.

**What you have:**
- ✅ 115,673 video frames (FaceForensics++ Deepfakes)
- ✅ 43 original videos with embedded audio
- ❌ No extracted audio yet

**What we'll do:**
1. Extract audio from videos
2. Process audio into spectrograms/MFCC
3. Create multimodal dataloaders
4. Train your model

---

## Step 1: Extract Audio from Videos

### A) From Original Videos (43 real videos)

```bash
cd e:\project\aura-veracity-lab\model-service

python scripts/extract_audio_multimodal.py \
    --video-dir ..\FaceForensics-master\original_sequences\youtube\raw\videos \
    --output-dir data/processed/audio \
    --label real \
    --sample-rate 16000
```

**Expected output:**
```
Audio Extraction Complete for REAL
✓ Successful: 43
✗ Failed: 0
⏱  Total duration: X.XX hours
📁 Output: data/processed/audio/real/
```

### B) From FaceForensics++ Videos (if you have them)

```bash
python scripts/extract_audio_multimodal.py \
    --video-dir data/FaceForensics++/manipulated_sequences/Deepfakes/c40/videos \
    --output-dir data/processed/audio \
    --label fake \
    --sample-rate 16000
```

### Notes:
- ⏱ **Time estimate:** ~2-3 minutes for 43 videos
- 💾 **Storage:** ~500 MB for 43 videos at 16kHz
- 🔊 **Format:** 16kHz mono WAV files

---

## Step 2: Verify Audio-Video Alignment

Check that audio and video frames are synchronized:

```bash
python scripts/verify_multimodal_alignment.py \
    --frame-dir data/processed/train \
    --audio-dir data/processed/audio/train \
    --tolerance 0.5
```

Or check all splits:

```bash
python scripts/verify_multimodal_alignment.py \
    --data-dir data/processed \
    --all-splits \
    --tolerance 0.5
```

**Expected output:**
```
ALIGNMENT REPORT: fake
================================================================================

Summary:
  Total videos: 8000
  ✓ Aligned: 7995
  ✗ Misaligned: 5
  ⚠ Missing audio: 0
  Average time diff: 0.023s
  Max time diff: 0.48s
```

---

## Step 3: Create Multimodal DataLoader

Use the multimodal dataset in your training code:

```python
from src.datasets.multimodal_dataset import (
    MultimodalDeepfakeDataset,
    create_multimodal_dataloaders
)

# Option A: Single dataset
dataset = MultimodalDeepfakeDataset(
    frame_dir='data/processed/train',
    audio_dir='data/processed/audio/train',
    audio_feature='spectrogram',  # or 'mfcc', 'waveform'
    frames_per_video=10
)

from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)

for batch in loader:
    frames = batch['frames']      # [batch, 10, 3, 224, 224]
    audio = batch['audio']        # [batch, 80, time_steps]
    labels = batch['label']       # [batch]
    
    # Train your multimodal model
    outputs = model(video=frames, audio=audio)
    loss = criterion(outputs, labels)
    loss.backward()

# Option B: All splits at once
loaders = create_multimodal_dataloaders(
    data_dir='data/processed',
    batch_size=32,
    num_workers=4,
    audio_feature='spectrogram'
)

train_loader = loaders['train']
val_loader = loaders['val']
test_loader = loaders['test']
```

---

## Step 4: Update Your Model for Multimodal Input

Your model already supports multimodal training! Check the config:

```yaml
# config/config.yaml
model:
  enable_video: true
  enable_audio: true
  video_encoder: "efficient_net"
  audio_encoder: "wav2vec2"
  fusion_method: "concat"  # or "attention", "bilinear"
```

### Audio Feature Types

| Type | Shape | Best For | Properties |
|------|-------|----------|-----------|
| **spectrogram** | [80, 300] | Most models | Fast, good for CNN |
| **mfcc** | [13, 300] | Speech-focused | Compact, traditional |
| **waveform** | [48000] | Wav2Vec2 | Raw, for transformers |

**Recommendation:** Start with `spectrogram` for faster training.

---

## Step 5: Train Multimodal Model

```python
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Load your multimodal model
model = YourMultimodalModel(
    enable_video=True,
    enable_audio=True,
    num_classes=2
)

# Create dataloaders
from src.datasets.multimodal_dataset import create_multimodal_dataloaders
loaders = create_multimodal_dataloaders(
    data_dir='data/processed',
    batch_size=32,
    audio_feature='spectrogram'
)

# Training loop
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    for batch in loaders['train']:
        frames = batch['frames'].to(device)
        audio = batch['audio'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(video=frames, audio=audio)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        for batch in loaders['val']:
            frames = batch['frames'].to(device)
            audio = batch['audio'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(video=frames, audio=audio)
            val_loss = criterion(outputs, labels)
```

---

## Audio Features Explained

### Mel-Spectrogram (Recommended)
- **Shape:** [80 mel bins, ~300 time steps]
- **Best for:** CNN-based audio encoders
- **Computation:** Fast (~1 frame per video)
- **Information retained:** Good balance of time and frequency

```python
processor = AudioProcessor(n_mels=80)
spectrogram = processor.audio_to_spectrogram('audio.wav')
# Output: [80, 300]
```

### MFCC (Speech-optimized)
- **Shape:** [13, ~300 time steps]
- **Best for:** Speech recognition models
- **Computation:** Fast
- **Information retained:** Perceptually compressed

```python
mfcc = processor.audio_to_mfcc('audio.wav')
# Output: [13, 300]
```

### Raw Waveform (Transformer-based)
- **Shape:** [48000] samples (3 seconds @ 16kHz)
- **Best for:** Wav2Vec2, other transformers
- **Computation:** Slow (end-to-end learning)
- **Information retained:** All raw information

```python
waveform = processor.audio_to_waveform('audio.wav')
# Output: [48000]
```

---

## Expected Performance

### Baseline (Video Only)
- **AUC-ROC:** 0.92-0.94
- **Accuracy:** 88-91%

### Multimodal (Video + Audio)
- **AUC-ROC:** 0.95-0.97 (+3-5%)
- **Accuracy:** 92-95% (+4-5%)
- **Inference time:** +30-50% longer

---

## Troubleshooting

### Issue: "Audio not found for video_X"
**Solution:** Check audio extraction completed successfully
```bash
python scripts/verify_multimodal_alignment.py --data-dir data/processed --all-splits
```

### Issue: Memory error with batch_size=32
**Solution:** Reduce batch size or frames_per_video
```python
loaders = create_multimodal_dataloaders(
    data_dir='data/processed',
    batch_size=16,  # Reduce
    frames_per_video=5  # Reduce
)
```

### Issue: Slow data loading
**Solution:** Increase num_workers
```python
loader = DataLoader(dataset, num_workers=8, pin_memory=True)
```

### Issue: Audio samples are zeros
**Solution:** Verify audio file exists and is valid
```bash
ffprobe audio.wav
# Should show: Audio: pcm_s16le, 16000 Hz, mono
```

---

## Next Steps

1. ✅ Extract audio from your 43 real videos
2. ✅ Verify alignment between audio and frames
3. ✅ Update your model to use multimodal input
4. ✅ Train and evaluate on the full FaceForensics++ dataset
5. ✅ Compare results: video-only vs multimodal

---

## File Reference

| File | Purpose |
|------|---------|
| `scripts/extract_audio_multimodal.py` | Extract audio from videos |
| `src/preprocessing/audio_processor.py` | Process audio to features |
| `src/datasets/multimodal_dataset.py` | DataLoader for paired data |
| `scripts/verify_multimodal_alignment.py` | Verify audio-video sync |

---

## Citation for Publication

If using this multimodal approach, cite:

```bibtex
@inproceedings{roessler2019faceforensicspp,
  title={FaceForensics++: Learning to Detect Manipulated Facial Images},
  author={Rössler, Andreas and others},
  booktitle={ICCV},
  year={2019}
}

@article{baevski2020wav2vec,
  title={wav2vec 2.0: A framework for self-supervised learning of speech representations},
  author={Baevski, Alexei and others},
  journal={NeurIPS},
  year={2020}
}
```

---

Need help? Check the docstrings in the Python files or run:
```bash
python scripts/extract_audio_multimodal.py --help
python scripts/verify_multimodal_alignment.py --help
```

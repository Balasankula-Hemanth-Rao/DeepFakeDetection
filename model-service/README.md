# Model Service - README

## Overview

This is the model service for the Aura Veracity Lab multimodal deepfake detection system. It provides state-of-the-art deepfake detection using combined video and audio analysis.

## Features

- **Multimodal Detection**: Combines video (EfficientNet-B3) and audio (Wav2Vec2) features
- **Advanced Fusion**: Cross-modal attention, gated fusion, and transformer-based fusion
- **Temporal Modeling**: Transformer-based temporal encoder for video sequences
- **Explainability**: Grad-CAM saliency maps for visual explanations
- **Ensemble Support**: Multiple model aggregation for improved robustness
- **Async API**: FastAPI-based video inference with job tracking

## Quick Start

### Installation

```bash
cd model-service
pip install -r requirements.txt
```

### Download Pre-trained Models

Wav2Vec2 will auto-download on first use:
```python
from transformers import Wav2Vec2Model
Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
```

### Configuration

Edit `config/model_config.yaml` to configure:
- Audio encoder type (simple or wav2vec2)
- Fusion strategy (concat, attention, gated, transformer)
- Temporal modeling (avg_pool, tconv, transformer)
- Training hyperparameters

### Training

```bash
python src/train.py --config config/model_config.yaml
```

### Inference

#### Start API Server

```bash
cd src/serve
uvicorn api_video:app --host 0.0.0.0 --port 8001
```

#### Submit Video

```bash
curl -X POST "http://localhost:8001/analyze-video" \
  -F "video=@test_video.mp4" \
  -F "fps=3"
```

#### Check Status

```bash
curl "http://localhost:8001/jobs/{job_id}"
```

## Project Structure

```
model-service/
├── src/
│   ├── models/              # Model architectures
│   │   ├── audio_encoder.py
│   │   ├── fusion.py
│   │   ├── temporal_transformer.py
│   │   ├── explainability.py
│   │   ├── frame_model.py
│   │   ├── multimodal_model.py
│   │   └── losses.py
│   ├── preprocess/          # Preprocessing utilities
│   │   ├── extract_audio.py
│   │   ├── extract_frames.py
│   │   ├── voice_activity_detection.py
│   │   └── optical_flow.py
│   ├── serve/               # Inference serving
│   │   ├── api_video.py
│   │   ├── inference.py
│   │   └── ensemble.py
│   ├── eval/                # Evaluation
│   ├── utils/               # Utilities
│   ├── train.py             # Training script
│   └── config.py            # Configuration
├── config/
│   └── model_config.yaml    # Model configuration
├── tests/                   # Unit and integration tests
├── checkpoints/             # Saved models
├── requirements.txt         # Dependencies
└── README.md

```

## Model Architecture

### Video Pipeline
1. Frame extraction (3 FPS)
2. Face detection (RetinaFace)
3. Video backbone (EfficientNet-B3)
4. Temporal encoder (Transformer)

### Audio Pipeline
1. Audio extraction (16kHz)
2. Voice activity detection
3. Wav2Vec2 encoder
4. Feature projection (768 → 512)

### Fusion
- Cross-modal attention fusion
- Video features query audio features
- Learned complementary information

### Classification
- MLP classifier head
- Binary output (real/fake)
- Softmax probabilities

## Performance

Expected improvements over baseline:
- Wav2Vec2 audio encoder: +5-10% AUC
- Cross-modal attention: +2-5% AUC
- Temporal transformer: +2-3% AUC
- Optical flow: +3-5% AUC
- Ensemble (3 models): +2-4% AUC

**Total expected improvement: +15-29% AUC**

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_audio_encoder.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Documentation

- [Integration Guide](INTEGRATION_GUIDE.md) - Usage examples and API documentation
- [ML System Design](../ML_SYSTEM_DESIGN.md) - Architecture details
- [Implementation Roadmap](../IMPLEMENTATION_ROADMAP.md) - Development plan

## License

Proprietary - Aura Veracity Lab

## Contact

For questions or issues, please contact the development team.

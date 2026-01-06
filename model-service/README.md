# Model Service

A clean, minimal scaffold for machine learning model training, evaluation, and inference within the Aura Veracity platform.

## Purpose

This module provides the foundation for:
- **Model development**: Training and fine-tuning computer vision models using PyTorch
- **Data preprocessing**: Converting and preparing raw video/image data for model consumption
- **Inference**: Running trained models on new inputs for AI detection predictions
- **Integration**: Serving predictions via FastAPI endpoints for the backend API

The model service is designed to work seamlessly with the backend API, receiving video data through signed URLs and returning detection results to the database.

## Project Structure

```
model-service/
├── src/
│   ├── __init__.py
│   ├── preprocess/          # Data preprocessing utilities
│   │   └── __init__.py
│   └── models/              # Model definitions and utilities
│       └── __init__.py
├── tests/                   # Unit and integration tests
│   └── __init__.py
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore patterns
└── README.md                # This file
```

## Installation

### Prerequisites
- Python 3.10+
- pip or conda

### Setup

1. Navigate to the model-service directory:
```bash
cd model-service
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Core Dependencies

- **torch**: Deep learning framework
- **timm**: PyTorch Image Models for pretrained architectures
- **opencv-python**: Computer vision utilities
- **fastapi**: Web framework for model serving
- **uvicorn**: ASGI server
- **pydantic**: Data validation
- **pytest**: Testing framework
- **python-dotenv**: Environment variable management

## Usage

### Training Scripts

Training scripts will be added to `src/training/` as needed:

```python
# Example: src/training/train.py
from src.models import YourModel
from src.preprocess import DataLoader

# Training logic here
```

Run training:
```bash
python src/training/train.py
```

### Inference Scripts

Inference scripts will be added to `src/inference/` for batch predictions:

```python
# Example: src/inference/predict.py
from src.models import YourModel

model = YourModel.load('checkpoints/model.pth')
results = model.predict(input_data)
```

Run inference:
```bash
python src/inference/predict.py --input data/video.mp4
```

### Model Serving

FastAPI endpoints for real-time inference:

```python
# Example: src/api.py
from fastapi import FastAPI
from src.models import YourModel

app = FastAPI()
model = YourModel.load('checkpoints/model.pth')

@app.post("/predict")
def predict(file: UploadFile):
    # Preprocessing
    # Inference
    # Return results
    pass
```

Run the API:
```bash
uvicorn src.api:app --reload
```

## Health Checks

The API provides Kubernetes-ready health check endpoints for container orchestration and load balancing.

### Liveness Probe: `/health/live`

**Purpose**: Indicates whether the container process is alive.

**Returns**: Always `200 OK` with status `alive`

**Usage**:
```bash
curl http://localhost:8000/health/live
```

**Response**:
```json
{"status": "alive"}
```

**Kubernetes Configuration**:
```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
```

### Readiness Probe: `/health/ready`

**Purpose**: Indicates whether the service is ready to handle inference requests.

**Returns**:
- `200 OK` if model weights are loaded and logging is initialized
- `503 Service Unavailable` if the model is not ready

**Usage**:
```bash
curl http://localhost:8000/health/ready
```

**Response (Ready)**:
```json
{
  "status": "ready",
  "model_loaded": true,
  "checkpoint_path": "/path/to/checkpoints/debug.pth"
}
```

**Response (Not Ready)**:
```json
{
  "status": "not_ready",
  "model_loaded": false,
  "checkpoint_path": "/path/to/checkpoints/debug.pth"
}
```

**Kubernetes Configuration**:
```yaml
readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 2
```

### Legacy Health Check: `/health`

**Purpose**: Comprehensive health status (deprecated, use `/health/live` and `/health/ready` instead).

**Returns**: `200 OK` with detailed health information

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "checkpoint_exists": true,
  "ready": true
}
```

### Implementation Details

- **Liveness Probe** (`/health/live`): Always returns 200 to indicate the container is alive. Use this to restart unhealthy containers.
- **Readiness Probe** (`/health/ready`): Returns status based on model initialization success. Use this to route traffic only to ready instances.
- **READY Flag**: Set to `True` during startup when model loads successfully. Set to `False` during shutdown or if model loading fails.
- **No Service Restart**: The server continues running even if model loading fails. The readiness probe reports the failure, allowing orchestration systems to handle it (restart, scale up, etc.).

### Docker / Kubernetes Integration

**Docker Healthcheck**:
```dockerfile
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health/live || exit 1
```

**Docker Compose**:
```yaml
services:
  model-service:
    image: aura-veracity/model-service:latest
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health/live"]
      interval: 10s
      timeout: 3s
      retries: 3
      start_period: 30s
```

**Load Balancer Integration**:
Most load balancers (AWS ALB, GCP Load Balancer, Nginx, etc.) use readiness probes to determine traffic routing:
- Requests to `/health/ready` every 5 seconds
- If returns 503, remove instance from load balancer
- If returns 200, route traffic to instance
- This ensures only ready instances receive inference requests

## Testing

Run unit tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=src
```

## Structured Logging

The model service uses **loguru** for structured, production-ready logging.

### Configuration

Logging is configured via `config/config.yaml`:

```yaml
logging:
  level: "INFO"        # DEBUG, INFO, WARNING, ERROR, CRITICAL
  json: true          # If true, output JSON-friendly logs
  log_file: ""        # Empty = stdout only, else relative path (e.g., "logs/app.log")
```

### Features

- **JSON output**: Logs are formatted as JSON for easy aggregation and parsing
- **Rotating files**: Automatic log rotation (10MB per file, keep 3 files)
- **Context binding**: Add request_id, user_id, component, etc. to all logs
- **Sensitive data filtering**: Automatically filters passwords, tokens, image bytes, etc.
- **Structured metadata**: Logs include machine-readable fields (latency, status, etc.)

### Usage in Code

#### Basic logging:
```python
from src.logging_config import setup_logging, get_logger

# At application startup
setup_logging()

# Get logger
logger = get_logger(__name__)

# Log messages
logger.info("User logged in", user_id=123)
logger.warning("High latency detected", latency_ms=500)
logger.error("Model loading failed", model_type="efficientnet_b3", error="CUDA error")
```

#### With request context:
```python
from src.logging_config import get_logger

logger = get_logger("inference").bind(
    request_id="req-12345",
    component="api",
    user_id=123
)

logger.info("Processing inference request", filename="image.jpg", size_mb=2.1)
```

#### Log inference events:
```python
from src.logging_config import log_inference_event

log_inference_event(
    logger=logger,
    request_id="req-12345",
    filename="test.jpg",
    size_bytes=1024000,
    model_version="efficientnet_b3_v1",
    fake_prob=0.8234,
    latency_ms=234.5,
    status="success"
)
```

### Log Levels

- **DEBUG**: Detailed diagnostic information (disabled in production)
- **INFO**: General informational messages
- **WARNING**: Warning messages for potentially problematic situations
- **ERROR**: Error messages for serious problems
- **CRITICAL**: Critical error messages requiring immediate attention

### Log Output Examples

**Plain text (when `json: false`):**
```
INFO     | src.serve.api:startup_event:140 - Model loaded successfully on startup
WARNING  | src.serve.api:infer:200 - Invalid API key provided
ERROR    | src.serve.api:infer:210 - Failed to load image: Invalid image format
```

**JSON (when `json: true`):**
```json
{"timestamp": "2025-12-11T14:30:45.123456", "level": "INFO", "message": "Model loaded successfully", "checkpoint_path": "/checkpoints/debug.pth", "file_size_mb": 138.2}
{"timestamp": "2025-12-11T14:30:46.234567", "level": "INFO", "message": "Inference completed", "request_id": "req-123", "component": "inference", "fake_probability": 0.8234, "latency_ms": 234.5, "status": "success"}
```

### Log Location

By default, logs are written to **stdout only**. To enable file logging, set in `config/config.yaml`:

```yaml
logging:
  log_file: "logs/app.log"
```

This creates a rotating log file at `model-service/logs/app.log` (configurable via `LOG_DIR` environment variable).

### Sensitive Data Protection

The logging system automatically filters sensitive information:

**Fields that are filtered:**
- `password`, `token`, `secret`, `api_key`
- `file_content`, `image_bytes`, `raw_data`
- Any field with values containing "secret" or "key"

**Safe to log:**
- Request IDs, user IDs, session IDs
- File names, file sizes, dimensions
- Model versions, latencies, probabilities
- Status codes, error types (not details)

### Performance Considerations

- Logging has minimal overhead (< 1ms per message)
- File rotation is automatic (no manual cleanup required)
- Context binding is lazy (only computed when logged)
- Use `.debug()` for verbose logging (disabled in production)

### Testing Logging

```bash
# Run logging tests
pytest tests/test_logging.py -v

# Run with coverage
pytest tests/test_logging.py --cov=src.logging_config
```



The model service integrates with the backend API:

1. **Backend calls**: `POST /predict` with signed URL to video file
2. **Model service**: Downloads, processes, and runs inference
3. **Results returned**: Detection data stored in database via backend

Example backend integration:
```python
# In backend service
import httpx

response = httpx.post(
    "http://localhost:8001/predict",
    json={"video_url": signed_url}
)
results = response.json()
```

## Performance Considerations

- **GPU acceleration**: Models default to CUDA when available
- **Batch processing**: Process multiple videos for efficiency
- **Caching**: Cache preprocessed data and model weights
- **Async serving**: Use FastAPI's async endpoints for I/O-bound operations

## Common Tasks

### Add a new preprocessing function:
1. Create `src/preprocess/my_function.py`
2. Import and test in `tests/test_preprocess.py`
3. Use in training/inference scripts

### Train a new model:
1. Define architecture in `src/models/my_model.py`
2. Create training script in `src/training/train.py`
3. Save checkpoints to `checkpoints/`

### Deploy a trained model:
1. Save checkpoint: `torch.save(model.state_dict(), 'checkpoints/model.pth')`
2. Load in API: `model = MyModel().load_state_dict(torch.load(...))`
3. Run FastAPI server for inference

## Next Steps

1. **Define your model architecture** in `src/models/`
2. **Create preprocessing functions** in `src/preprocess/`
3. **Write training script** in `src/training/train.py`
4. **Implement FastAPI endpoints** in `src/api.py`
5. **Add unit tests** in `tests/`

## License

Same as parent project (Aura Veracity)

## Iteration 14: Multimodal Training Pipeline

Full multimodal (video + audio) training pipeline for deepfake detection with comprehensive preprocessing, augmentation, evaluation, and logging.

### Features

- **Multimodal Dataset Loader**: Load video and audio from MP4 files or preextracted frames/audio
- **Video Processing**: Face detection/alignment, frame sampling (random/uniform/start_at), video augmentation
- **Audio Processing**: Mel-spectrogram extraction, audio augmentation (noise, time-stretch, pitch-shift)
- **Model Architecture**: Video backbone (timm) + temporal encoder + audio CNN + fusion head
- **Training Pipeline**: Full training loop with validation, checkpointing, early stopping, metrics logging
- **Evaluation**: Per-sample and per-video evaluation with AUC, AP, accuracy, precision, recall, F1, FPR@95%TPR
- **Config-Driven**: All parameters configurable via YAML
- **Structured Logging**: JSON-formatted logs with metrics

### Dataset Structure

Organize your data as follows:

```
data/deepfake/
├── train/
│   ├── video_001/
│   │   ├── video.mp4
│   │   └── meta.json          # { "label": 0 }
│   ├── video_002/
│   │   ├── video.mp4
│   │   └── meta.json          # { "label": 1 }
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

Or provide a manifest CSV:

```
video_id,label,split,path
video_001,0,train,data/deepfake/train/video_001/video.mp4
video_002,1,train,data/deepfake/train/video_002/video.mp4
...
```

### Configuration

Key multimodal settings in `config/config.yaml`:

```yaml
dataset:
  data_root: "data/deepfake"
  preprocessing:
    mode: "on_the_fly"  # or "preextracted"
  video:
    frame_rate: 30
    temporal_window: 16
    clip_sampling_strategy: "random"
  face:
    detect: true
    detector: "retinaface"
  audio:
    enabled: true
  augmentation:
    video:
      random_crop: true
      random_flip: true

training:
  epochs: 30
  batch_size: 16
  learning_rate: 1.0e-4
  use_amp: false

model:
  video:
    backbone: "efficientnet_b3"
    pretrained: true
  audio:
    sample_rate: 16000
    n_mels: 64
  fusion:
    strategy: "concat"  # or "attention", "cross_modal"
```

### Quick Start

#### 1. Install Dependencies

```bash
pip install -r requirements.txt

# Optional: Face detection
pip install facenet-pytorch retinaface

# Optional: Advanced audio processing
pip install librosa
```

#### 2. Prepare Data

Download your dataset and organize as shown above. Example with sample data:

```bash
mkdir -p data/deepfake/{train,val,test}
# Copy your videos and create meta.json files
```

#### 3. Debug Mode (Quick Validation)

Test the training pipeline on a tiny subset:

```bash
python src/train/multimodal_train.py --debug
```

This runs:
- 1 epoch
- On 4 samples (configurable)
- CPU-friendly
- Creates `checkpoints/debug.pth`
- Ideal for CI/CD pipelines

#### 4. Full Training

```bash
python src/train/multimodal_train.py \
  --data-root data/deepfake \
  --epochs 30 \
  --batch-size 16 \
  --num-workers 4 \
  --device cuda
```

Optional arguments:
- `--config`: Custom config file path
- `--manifest`: CSV/JSON manifest path
- `--resume`: Resume from checkpoint
- `--device`: auto/cuda/cpu

#### 5. Evaluation

```bash
python src/eval/multimodal_eval.py \
  --checkpoint checkpoints/multimodal_best_0.9234.pth \
  --split test \
  --aggregation mean \
  --save-csv results.csv
```

This produces:
- `results/eval_results.json` with all metrics
- Optional CSV with metrics

### Module Overview

#### `src/data/multimodal_dataset.py`
- `MultimodalDataset`: PyTorch Dataset class
  - Loads video/audio from MP4 or preextracted files
  - Handles frame sampling, face detection, augmentation
  - Returns torch tensors with shape [T,3,H,W] and [n_mels, time_steps]

#### `src/models/multimodal_model.py`
- `MultimodalModel`: Complete model architecture
  - Video backbone from timm (EfficientNet, ResNet, ViT, etc.)
  - Temporal encoder (avg_pool, temporal conv, transformer)
  - Audio CNN on mel-spectrograms
  - Fusion head (concat, attention, cross-modal)
  - Methods: `forward()`, `extract_features()`, `load_for_inference()`, `save_checkpoint()`

#### `src/train/multimodal_train.py`
- `Trainer`: Full training manager
  - Data loading with multiprocessing
  - Configurable optimizer/scheduler (AdamW, CosineAnnealing, ReduceLROnPlateau)
  - Mixed precision training (optional)
  - Early stopping and periodic checkpointing
  - Structured logging with metrics
- CLI arguments for full customization
- Debug mode for quick testing

#### `src/eval/multimodal_eval.py`
- `evaluate_model()`: Comprehensive evaluation
  - Per-sample and per-video metrics
  - AUC, AP, accuracy, precision, recall, F1
  - FPR@95%TPR for security evaluations
  - Confusion matrix
  - JSON and CSV output

#### `src/utils/metrics.py`
- `compute_auc()`: ROC-AUC score
- `compute_ap()`: Average Precision
- `compute_fpr_at_tpr()`: False Positive Rate at target TPR
- `compute_metrics()`: Batch metrics computation
- `save_metrics_json()`: Serialize metrics

### Training Features

- **Optimizers**: AdamW with weight decay
- **Schedulers**: CosineAnnealingWarmRestarts, ReduceLROnPlateau
- **Loss**: CrossEntropyLoss with label smoothing
- **Gradient Clipping**: Configurable max norm
- **Mixed Precision**: Optional torch.cuda.amp.autocast
- **Checkpointing**: Best model + periodic saves with metadata
- **Early Stopping**: Patience-based with configurable metric
- **Logging**: Structured JSON logs with timestamps and context
- **Auxiliary Losses**: Temporal consistency loss for enhanced learning

#### Temporal Consistency Loss (Optional)

The model supports an optional **temporal consistency loss** that encourages smooth, consistent feature representations across video frames. This is particularly effective for deepfake detection since authentic videos tend to have more natural temporal coherence than artificially generated ones.

**Configuration:**

In `config/config.yaml`, enable under the fusion settings:

```yaml
model:
  fusion:
    strategy: "concat"
    hidden_dim: 512
    dropout: 0.3
    temporal_consistency_loss:
      enabled: false           # Set to true to enable
      weight: 0.1              # Loss weight (0.01-0.5 recommended)
```

**How It Works:**

- **During Training**: Penalizes high variance in embeddings across adjacent frames
- **During Inference**: No impact (loss only applies during training)
- **Safety**: Automatically disabled for audio-only models
- **Backward Compatibility**: Default disabled (weight=0.0 when disabled)

**Usage in Training Loop:**

```python
from src.models.losses import TemporalConsistencyLoss
from src.models.multimodal_model import MultimodalModel

model = MultimodalModel(config=config, num_classes=2)

# Forward pass
logits = model(video=batch_video, audio=batch_audio)

# Compute losses
main_loss = criterion(logits, targets)
temporal_loss = model.compute_temporal_consistency_loss(batch_video)

if temporal_loss is not None:
    total_loss = main_loss + model.temporal_consistency_weight * temporal_loss
else:
    total_loss = main_loss

# Backward pass
optimizer.zero_grad()
total_loss.backward()
optimizer.step()
```

**Hyperparameter Tuning:**

- **weight=0.1**: Balanced approach (recommended starting point)
- **weight=0.05**: Lighter regularization
- **weight=0.2**: Stronger temporal constraint
- **weight > 0.3**: May over-regularize, slow convergence

**Expected Benefits:**

- Improved generalization on authentic videos
- Better detection of temporal artifacts in deepfakes
- More stable training dynamics
- Reduced overfitting on small datasets

### Video Augmentation

When `split == 'train'`:
- Random horizontal flip
- Random crop and resize
- Color jittering (brightness, contrast)
- Optional JPEG compression simulation

### Audio Augmentation

When enabled:
- Additive noise
- Time stretching (pitch preservation)
- Pitch shifting (rate preservation)

### Face Detection

If enabled:
- Uses **RetinaFace** (accurate, fast) or **MTCNN** (CPU-friendly)
- Falls back to center crop if unavailable
- Configurable margin around detected face
- Optional face alignment (future)

### Data Preextraction

For faster training, preextract frames and audio:

```bash
# Example script to preextract (user-provided)
for video in data/deepfake/train/*/video.mp4; do
  dir=$(dirname "$video")
  ffmpeg -i "$video" -q:v 2 "$dir/preprocessed/frames/%06d.jpg"
  ffmpeg -i "$video" -q:v 2 -ac 1 -ar 16000 "$dir/preprocessed/audio.wav"
  librosa.output.write_wav(...)  # Convert to NPY
done
```

Then set `preprocessing.mode: "preextracted"` in config.

### Performance Tips

1. **Preextract frames/audio** for 2-3x training speedup
2. **Use multiple workers**: `num_workers=4` or higher
3. **Pin memory**: Automatic with GPU
4. **Reduce temporal window**: Trade off temporal context for speed
5. **Use smaller backbone**: mobilenet_v3_small, efficientnet_b0
6. **Enable AMP**: `use_amp: true` for ~2x speedup on Volta+

### Testing

Run comprehensive test suite:

```bash
# All tests
pytest tests/ -v

# Specific tests
pytest tests/test_multimodal_dataset.py -v
pytest tests/test_multimodal_model.py -v
pytest tests/test_multimodal_train_debug.py -v
```

Tests validate:
- Dataset loading and shapes
- Model instantiation and forward pass
- Training loop (with debug mode)
- Checkpoint save/load
- Metrics computation

### Checkpointing Format

Checkpoints contain:
```python
{
    'model_state_dict': model.state_dict(),
    'config': config,
    'epoch': epoch,
    'metrics': {'auc': 0.92, 'loss': 0.15, ...},
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
}
```

Filename pattern: `multimodal_v1_epoch{N}_{AUC:.4f}.pth`

Best model saves as: `multimodal_best_{AUC:.4f}.pth`

### Optional Dependencies

Optional features with helpful error messages:

| Feature | Package | Install |
|---------|---------|---------|
| Video loading | opencv-python | `pip install opencv-python` |
| Audio loading | torchaudio, librosa | `pip install torchaudio librosa` |
| Face detection | facenet-pytorch, retinaface | `pip install facenet-pytorch retinaface` |
| Metrics | scikit-learn | `pip install scikit-learn` |
| Experiment tracking | wandb, mlflow | `pip install wandb mlflow` |
| GPU acceleration | accelerate | `pip install accelerate` |

Missing optional dependencies gracefully fall back to simpler implementations.

### Logging & Metrics

Structured logging with loguru (JSON format):

```json
{
  "text": "Epoch 1 Training Metrics",
  "record": {
    "level": "INFO",
    "loss": 0.512,
    "auc": 0.823,
    "accuracy": 0.76,
    "timestamp": "2025-01-15T10:23:45.123Z"
  }
}
```

### Safety & Legal Notes

⚠️ **IMPORTANT**:

1. **DO NOT commit dataset files** to this repository
   - Datasets may contain private/proprietary data
   - Use secure storage (S3, GCS, Azure Blob)
   - Provide signed URLs or authenticated downloads

2. **DO NOT commit sensitive credentials**
   - Use environment variables for API keys
   - Use `.env` file (git-ignored) for local development
   - Rotate credentials regularly in CI/CD

3. **Respect dataset licenses**
   - Document all external datasets used
   - Follow license terms (academic use only, attribution, etc.)
   - Obtain consent for face data

4. **Data privacy compliance**
   - GDPR, CCPA, local regulations
   - Use face detection only with consent
   - Implement data deletion policies

## Support

For questions about model integration with the backend, refer to `backend/FRONTEND_INTEGRATION.md` for the overall architecture.

For multimodal training details, see inline documentation in:
- `src/data/multimodal_dataset.py`
- `src/models/multimodal_model.py`
- `src/train/multimodal_train.py`
- `src/eval/multimodal_eval.py`


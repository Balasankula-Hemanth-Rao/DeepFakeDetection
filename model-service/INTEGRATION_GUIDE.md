# Backend Model Components - Quick Integration Guide

## 🚀 Quick Start

### Installation

```bash
cd model-service
pip install -r requirements.txt
```

### Download Pre-trained Models

```python
# Wav2Vec2 will auto-download on first use
from transformers import Wav2Vec2Model
Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
```

---

## 📦 Component Usage Examples

### 1. Audio Processing Pipeline

```python
from src.preprocess import AudioExtractor, VoiceActivityDetector
from src.models import Wav2Vec2AudioEncoder

# Extract audio from video
extractor = AudioExtractor(sample_rate=16000)
waveform, sr = extractor.extract_from_video('video.mp4')

# Apply voice activity detection
vad = VoiceActivityDetector(strategy='energy')
segments = vad.detect(waveform, sr)
masked_waveform = vad.apply_mask(waveform, segments, sr)

# Extract features
encoder = Wav2Vec2AudioEncoder(embed_dim=512)
audio_features = encoder(torch.from_numpy(masked_waveform).unsqueeze(0))
```

### 2. Video Processing with Optical Flow

```python
from src.preprocess import FrameExtractor, OpticalFlowExtractor

# Extract frames
frame_extractor = FrameExtractor(fps=3)
frames = frame_extractor.extract('video.mp4')

# Compute optical flow
flow_extractor = OpticalFlowExtractor(method='farneback')
flows = flow_extractor.compute_flow_sequence(frames)

# Convert to tensors for model input
flow_tensors = [flow_extractor.flow_to_tensor(fx, fy) for fx, fy in flows]
```

### 3. Multimodal Fusion

```python
from src.models import CrossModalAttentionFusion

# Initialize fusion module
fusion = CrossModalAttentionFusion(
    video_dim=1536,
    audio_dim=512,
    output_dim=1024,
    num_heads=4
)

# Fuse features
fused_features = fusion(video_features, audio_features)
```

### 4. Video Inference

```python
from src.serve import VideoInferenceEngine
from src.models import MultimodalModel

# Load model
model = MultimodalModel.load_for_inference('checkpoints/best_model.pth')

# Initialize engine
engine = VideoInferenceEngine(model, config={'fps': 3, 'sample_rate': 16000})

# Analyze video
result = await engine.analyze_video(frames, audio, sample_rate=16000)
print(f"Prediction: {result['prediction']}, Confidence: {result['confidence']:.2%}")
```

### 5. Ensemble Inference

```python
from src.serve import EnsembleModel

# Create ensemble
ensemble = EnsembleModel(
    checkpoint_paths=['model1.pth', 'model2.pth', 'model3.pth'],
    ensemble_strategy='weighted',
    weights=[0.4, 0.3, 0.3]
)

# Predict
prediction = ensemble.predict(video=video_tensor, audio=audio_tensor)
print(f"Ensemble prediction: {prediction['prediction']}")
print(f"Model agreement: {ensemble.get_model_agreement(video_tensor, audio_tensor):.2%}")
```

### 6. Explainability

```python
from src.models import SaliencyMapGenerator

# Initialize
saliency_gen = SaliencyMapGenerator(model, target_layer='video_backbone.blocks[-1]')

# Generate saliency map
saliency = saliency_gen.generate_saliency(
    input_tensor,
    method='gradcam',
    target_class=1  # Fake class
)

# Save overlay
saliency_gen.save_saliency_overlay(image, saliency, 'output.png')
```

---

## 🔧 Integration with Existing Model

Update `multimodal_model.py` to use new components:

```python
# In MultimodalModel.__init__()

# Replace AudioCNN with Wav2Vec2
if self.enable_audio:
    from .audio_encoder import Wav2Vec2AudioEncoder
    self.audio_encoder = Wav2Vec2AudioEncoder(
        embed_dim=self.audio_embed_dim,
        freeze_layers=8
    )

# Add fusion strategy selection
if self.fusion_strategy == 'attention':
    from .fusion import CrossModalAttentionFusion
    self.fusion_head = CrossModalAttentionFusion(
        video_dim=self.video_embed_dim,
        audio_dim=self.audio_embed_dim,
        output_dim=self.fusion_dim
    )

# Add temporal transformer option
if self.temporal_strategy == 'transformer':
    from .temporal_transformer import TemporalTransformer
    self.temporal_encoder = TemporalTransformer(
        embed_dim=self.video_embed_dim,
        num_heads=8,
        num_layers=2
    )
```

---

## 🌐 API Server

### Start Video Inference API

```bash
cd src/serve
uvicorn api_video:app --host 0.0.0.0 --port 8001 --reload
```

### Submit Video for Analysis

```bash
curl -X POST "http://localhost:8001/analyze-video" \
  -F "video=@test_video.mp4" \
  -F "fps=3" \
  -F "sample_rate=16000"
```

Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status_url": "/jobs/550e8400-e29b-41d4-a716-446655440000",
  "message": "Video submitted for analysis"
}
```

### Check Job Status

```bash
curl "http://localhost:8001/jobs/550e8400-e29b-41d4-a716-446655440000"
```

---

## 📊 Performance Tuning

### GPU Memory Optimization

```python
# Use gradient checkpointing
model.video_backbone.set_grad_checkpointing(True)

# Use mixed precision
from torch.cuda.amp import autocast

with autocast():
    logits = model(video, audio)
```

### Batch Processing

```python
# Process multiple videos in batch
batch_size = 4
for i in range(0, len(videos), batch_size):
    batch_videos = videos[i:i+batch_size]
    results = model(batch_videos)
```

---

## 🧪 Testing

### Unit Tests

```bash
pytest tests/test_audio_encoder.py -v
pytest tests/test_fusion.py -v
pytest tests/test_inference.py -v
```

### Integration Test

```python
# tests/test_integration.py
def test_end_to_end_inference():
    # Load model
    model = MultimodalModel.load_for_inference('checkpoints/test.pth')
    
    # Process video
    engine = VideoInferenceEngine(model, config)
    result = await engine.analyze_video(frames, audio, 16000)
    
    assert result['prediction'] in ['real', 'fake']
    assert 0 <= result['confidence'] <= 1
```

---

## 📝 Configuration

Create `config/model_config.yaml`:

```yaml
model:
  video:
    backbone: efficientnet_b3
    pretrained: true
    embed_dim: 1536
    temporal_strategy: transformer  # NEW: use transformer
  
  audio:
    encoder: wav2vec2  # NEW: use wav2vec2
    model_name: facebook/wav2vec2-base
    embed_dim: 512
    freeze_layers: 8
    sample_rate: 16000
  
  fusion:
    strategy: attention  # NEW: use cross-modal attention
    hidden_dim: 1024
    num_heads: 4
    dropout: 0.1
  
  preprocessing:
    use_vad: true  # NEW: enable VAD
    use_optical_flow: true  # NEW: enable optical flow
    fps: 3
    temporal_window: 16

training:
  batch_size: 8
  learning_rate: 1e-4
  epochs: 50
  
inference:
  aggregation: mean
  top_k_anomalous: 5
  generate_saliency: true
```

---

## 🚨 Common Issues

### Issue: Out of Memory

**Solution:** Reduce batch size or temporal window
```python
config['temporal_window'] = 8  # Reduce from 16
config['batch_size'] = 4  # Reduce from 8
```

### Issue: Slow Inference

**Solution:** Use ensemble only for final evaluation
```python
# Development: single model
model = MultimodalModel.load_for_inference('best_model.pth')

# Production: ensemble
ensemble = EnsembleModel(['model1.pth', 'model2.pth'])
```

### Issue: Wav2Vec2 Download Fails

**Solution:** Pre-download and cache
```bash
python -c "from transformers import Wav2Vec2Model; Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')"
```

---

## 📚 Additional Resources

- [ML System Design](file:///e:/project/aura-veracity-lab/ML_SYSTEM_DESIGN.md)
- [Implementation Roadmap](file:///e:/project/aura-veracity-lab/IMPLEMENTATION_ROADMAP.md)
- [Model Contract](file:///e:/project/aura-veracity-lab/MODEL_CONTRACT_v1.md)

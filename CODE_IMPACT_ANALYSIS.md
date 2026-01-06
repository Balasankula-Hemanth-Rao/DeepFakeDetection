# **CODE IMPACT ANALYSIS — MODEL CONTRACT v1.0 IMPLEMENTATION GUIDE**

**Purpose:** Translate locked decisions into specific code modifications needed  
**Target Audience:** ML Engineers implementing Phase 1–3  
**Last Updated:** January 3, 2026

---

## **🔄 DECISION → CODE CHANGE MAPPING**

### **DECISION 1: Audio Encoder = wav2vec2-base (Pretrained, Speaker-Agnostic)**

**Impact Level:** 🔴 CRITICAL (affects data pipeline, training, inference)

**Current State:**
```python
# models/audio_cnn.py (EXISTING — TO BE REPLACED)
class AudioCNN(nn.Module):
    def __init__(self, input_size=128, hidden_size=256):
        # 3-layer naive CNN on mel-spectrograms
        # Output: (batch, 512)
        
# data/multimodal_dataset.py (EXISTING — TO BE MODIFIED)
mel_spec = librosa.feature.melspectrogram(audio, n_mels=128)
audio_features = mel_spec_to_features(mel_spec)  # (batch, seq_len, 128)
```

**Required Changes:**

**File 1: `models/audio_encoder.py` (NEW)**
```python
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

class Wav2Vec2AudioEncoder(nn.Module):
    def __init__(self, freeze_blocks=8, fine_tune_blocks=4, output_dim=512):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # Freeze first N blocks
        for param in self.model.feature_extractor.parameters():
            param.requires_grad = False  # Freeze feature extractor
        
        # Freeze first N transformer blocks
        for i, layer in enumerate(self.model.encoder.layers):
            if i < freeze_blocks:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # Learnable projection: 768 → output_dim
        self.projection = nn.Linear(768, output_dim)
        
    def forward(self, waveform):
        # waveform: (batch, seq_len) or (seq_len,) at 16kHz
        outputs = self.model(waveform, return_dict=True)
        embeddings = outputs.last_hidden_state  # (batch, seq_len, 768)
        projected = self.projection(embeddings)  # (batch, seq_len, output_dim)
        return projected
```

**File 2: `data/multimodal_dataset.py` (MODIFY)**
```python
# REMOVE mel-spectrogram extraction
# OLD:
# mel_spec = librosa.feature.melspectrogram(audio, n_mels=128)

# NEW: Load raw waveform at 16kHz
def load_audio_waveform(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    audio = torch.from_numpy(audio).float()
    return audio

# In __getitem__:
audio_waveform = load_audio_waveform(audio_path)
# Return raw waveform, NOT mel-spec
return {
    'video_frames': frames,
    'audio_waveform': audio_waveform,  # Variable length!
    'label': label
}
```

**File 3: `models/multimodal_model.py` (MODIFY)**
```python
# OLD:
# self.audio_encoder = AudioCNN(input_size=128, hidden_size=256)
# audio_features = self.audio_encoder(mel_spec)  # (batch, seq_len, 512)

# NEW:
from models.audio_encoder import Wav2Vec2AudioEncoder
self.audio_encoder = Wav2Vec2AudioEncoder(output_dim=512)
audio_features = self.audio_encoder(audio_waveform)  # (batch, seq_len, 512)
```

**File 4: `train.py` (MODIFY)**
```python
# Update batch collation for variable-length audio
def collate_fn(batch):
    # Pad audio waveforms to max length in batch
    audio_waveforms = [item['audio_waveform'] for item in batch]
    max_audio_len = max(w.shape[0] for w in audio_waveforms)
    
    audio_padded = []
    for w in audio_waveforms:
        padded = torch.zeros(max_audio_len)
        padded[:w.shape[0]] = w
        audio_padded.append(padded)
    
    return {
        'audio': torch.stack(audio_padded),  # (batch, max_len)
        'video': torch.stack([item['video'] for item in batch]),
        'labels': torch.tensor([item['label'] for item in batch])
    }
```

**Testing Checklist:**
- [ ] Load FaceForensics++ sample video
- [ ] Extract audio waveform at 16kHz
- [ ] Pass through Wav2Vec2AudioEncoder
- [ ] Verify output shape: (batch, seq_len, 512)
- [ ] Check gradient flow in fine-tuned blocks
- [ ] Forward pass on full batch (no OOM)

---

### **DECISION 2: Temporal Window = 1 Second (5–10 Frames)**

**Impact Level:** 🟡 MEDIUM (affects frame extraction, temporal aggregation)

**Current State:**
```python
# preprocess/extract_frames.py
frame_rate = 24  # Variable, not standardized
```

**Required Changes:**

**File: `preprocess/extract_frames.py` (MODIFY)**
```python
# OLD:
# cap.set(cv2.CAP_PROP_FPS, video_fps)
# frames = []
# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret: frames.append(frame)

# NEW: Extract at 5 FPS for 1-second temporal window
def extract_frames_temporal_window(video_path, target_fps=5):
    """
    Extract frames at target_fps for 1-second temporal windows.
    1 second @ 5 FPS = 5 frames per window.
    """
    import ffmpeg
    
    # Use ffmpeg for precise frame extraction
    stream = ffmpeg.input(video_path)
    stream = ffmpeg.filter(stream, 'fps', fps=target_fps)
    stream = ffmpeg.output(stream, 'pipe:', format='rawvideo', pix_fmt='rgb24')
    
    out, _ = stream.run(capture_stdout=True)
    
    # Parse frames from output
    width, height = 1920, 1080  # Extract from video metadata
    frame_bytes = len(out) // (width * height * 3)
    frames = []
    for i in range(frame_bytes):
        frame_data = out[i*width*height*3:(i+1)*width*height*3]
        frame = np.frombuffer(frame_data, np.uint8).reshape((height, width, 3))
        frames.append(frame)
    
    return frames  # T frames @ 5 FPS (every 0.2 seconds)

# Usage:
frames = extract_frames_temporal_window(video_path, target_fps=5)
# For 30s video: 150 frames total (5 FPS × 30s)
```

**File: `models/multimodal_model.py` (VERIFY)**
```python
# Verify temporal encoder expects 5–10 frame sequences
class TemporalConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, num_frames=5):
        super().__init__()
        # 1D Conv over temporal dimension
        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        # x: (batch, num_frames, input_dim)
        x = x.permute(0, 2, 1)  # (batch, input_dim, num_frames)
        x = self.conv1d(x)      # (batch, output_dim, num_frames)
        x = self.avg_pool(x)    # (batch, output_dim, 1)
        x = x.squeeze(-1)       # (batch, output_dim)
        return x
```

**Testing Checklist:**
- [ ] Extract video at 5 FPS
- [ ] For 30s video: verify 150 frames extracted
- [ ] Frame timestamps: 0.0, 0.2, 0.4, ... 29.8 seconds
- [ ] Each 1-second window has exactly 5 frames
- [ ] Temporal encoder forward pass: (batch, 5, 1536) → (batch, 2048)

---

### **DECISION 3: Fusion Strategy = Cross-Modal Attention (Mid-Fusion)**

**Impact Level:** 🔴 CRITICAL (core architectural change)

**Current State:**
```python
# models/multimodal_model.py
fused = torch.cat([video_features, audio_features], dim=-1)  # Late concatenation
```

**Required Changes:**

**File: `models/fusion.py` (NEW)**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttentionFusion(nn.Module):
    """
    Cross-modal attention fusion.
    Video features attend over audio features for synergy detection.
    """
    def __init__(self, video_dim=2048, audio_dim=512, hidden_dim=512, num_heads=4):
        super().__init__()
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=video_dim,
            num_heads=num_heads,
            batch_first=True,
            kdim=audio_dim,
            vdim=audio_dim
        )
        
        # Projection for audio context
        self.audio_projection = nn.Linear(audio_dim, video_dim)
        
        # Fusion gate (learned weighting)
        self.gate = nn.Sequential(
            nn.Linear(video_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, video_features, audio_features):
        """
        Args:
            video_features: (batch, video_dim) or (batch, T, video_dim)
            audio_features: (batch, A, audio_dim) or (batch, audio_dim)
        
        Returns:
            fused: (batch, video_dim + audio_dim)
        """
        # Ensure proper shapes
        if video_features.dim() == 2:
            video_features = video_features.unsqueeze(1)  # (batch, 1, video_dim)
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(1)  # (batch, 1, audio_dim)
        
        # Cross-attention: video queries, audio keys/values
        attention_out, attention_weights = self.attention(
            query=video_features,
            key=audio_features,
            value=audio_features
        )
        # attention_out: (batch, T_video, video_dim)
        # attention_weights: (batch, num_heads, T_video, T_audio)
        
        # Pool temporal dimension
        video_context = attention_out.mean(dim=1)  # (batch, video_dim)
        
        # Project audio for concatenation
        audio_context = audio_features.mean(dim=1)  # (batch, audio_dim)
        audio_projected = self.audio_projection(audio_context)  # (batch, video_dim)
        
        # Gated fusion
        combined = torch.cat([video_context, audio_projected], dim=-1)  # (batch, 2*video_dim)
        gate_weight = self.gate(combined)  # (batch, 1)
        
        # Weighted combination
        fused = gate_weight * video_context + (1 - gate_weight) * audio_projected
        
        # Concatenate with audio features for classification
        fused = torch.cat([fused, audio_context], dim=-1)  # (batch, video_dim + audio_dim)
        
        return fused
```

**File: `models/multimodal_model.py` (MODIFY)**
```python
# REMOVE:
# fused = torch.cat([video_features, audio_features], dim=-1)

# ADD:
from models.fusion import CrossModalAttentionFusion
self.fusion = CrossModalAttentionFusion(
    video_dim=2048,
    audio_dim=512,
    hidden_dim=512,
    num_heads=4
)

# In forward:
fused = self.fusion(video_features, audio_features)
logits = self.classification_head(fused)
```

**Testing Checklist:**
- [ ] Video features shape: (batch, 2048)
- [ ] Audio features shape: (batch, seq_len, 512)
- [ ] Cross-attention forward pass works
- [ ] Fused shape: (batch, 2048 + 512)
- [ ] Gradient flow through attention weights
- [ ] Validation AUC improves vs concatenation baseline

---

### **DECISION 4: Explainability = Grad-CAM + Audio Anomalies + Modality Agreement**

**Impact Level:** 🟠 HIGH (required for forensic compliance)

**Current State:**
```python
# No explainability module
```

**Required Changes:**

**File: `models/explainability.py` (NEW)**
```python
import torch
import torch.nn.functional as F
import cv2
import numpy as np

class ExplainabilityModule(nn.Module):
    """
    Generate saliency maps and explainability scores.
    """
    def __init__(self, model, target_layer_name='backbone.layer4'):
        super().__init__()
        self.model = model
        self.target_layer = self._get_layer_by_name(model, target_layer_name)
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._store_activations)
        self.target_layer.register_backward_hook(self._store_gradients)
    
    def _store_activations(self, module, input, output):
        self.activations = output.detach()
    
    def _store_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def compute_grad_cam(self, frames):
        """
        Compute Grad-CAM saliency maps.
        
        Args:
            frames: (batch, 3, 224, 224) or list of frames
        
        Returns:
            saliency_maps: (batch, 224, 224)
        """
        # Forward pass
        logits = self.model(frames)
        fake_class_score = logits[:, 1]  # P(fake)
        
        # Backward pass
        self.model.zero_grad()
        fake_class_score.sum().backward()
        
        # Compute Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (batch, C, 1, 1)
        saliency = (weights * self.activations).sum(dim=1)  # (batch, H, W)
        saliency = F.relu(saliency)
        saliency = saliency - saliency.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        saliency = saliency / (saliency.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-8)
        
        return saliency  # (batch, H, W)
    
    def overlay_saliency_on_frame(self, frame, saliency_map, alpha=0.4):
        """
        Overlay saliency heatmap on original frame.
        
        Args:
            frame: (224, 224, 3) uint8 image
            saliency_map: (224, 224) float32 in [0, 1]
        
        Returns:
            overlay: (224, 224, 3) uint8
        """
        # Convert saliency to heatmap
        heatmap = cv2.applyColorMap(
            (saliency_map * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        # Blend
        overlay = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
        return overlay
    
    def compute_modality_agreement(self, video_confidence, audio_confidence):
        """
        Compute modality agreement score.
        
        Returns:
            agreement: float in [0, 1]
            - 1.0 = both agree (both say fake or both say real)
            - 0.0 = maximum disagreement
        """
        # Both high (both say fake)
        both_high = min(video_confidence, audio_confidence)
        
        # Both low (both say real)
        both_low = min(1 - video_confidence, 1 - audio_confidence)
        
        # Agreement = max of the two
        agreement = max(both_high, both_low)
        return agreement
```

**File: `serve/inference.py` (NEW)**
```python
import torch
import numpy as np
from models.explainability import ExplainabilityModule

def analyze_video_with_explainability(video_path, model, top_k=5):
    """
    End-to-end video analysis with explainability.
    
    Returns:
        predictions: {
            'overall_confidence': float,
            'anomalous_frames': [
                {
                    'frame_id': int,
                    'confidence': float,
                    'saliency_url': str
                },
                ...
            ]
        }
    """
    # Extract frames
    frames = extract_frames_temporal_window(video_path, target_fps=5)
    
    # Initialize explainability
    explainer = ExplainabilityModule(model)
    
    confidences = []
    saliency_maps = []
    
    # Run inference on frame batches
    batch_size = 32
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i+batch_size]
        batch_tensor = torch.stack([preprocess(f) for f in batch_frames])
        
        # Get confidence
        with torch.no_grad():
            logits = model(batch_tensor)
            batch_conf = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        confidences.extend(batch_conf)
        
        # Get saliency
        batch_saliency = explainer.compute_grad_cam(batch_tensor)
        saliency_maps.extend(batch_saliency.cpu().numpy())
    
    # Find top-k anomalous frames
    confidences = np.array(confidences)
    anomalous_idx = np.argsort(confidences)[::-1][:top_k]
    
    anomalous_frames = []
    for idx in anomalous_idx:
        frame = frames[idx]
        saliency = saliency_maps[idx]
        overlay = explainer.overlay_saliency_on_frame(frame, saliency)
        
        # Save overlay to Supabase
        saliency_url = upload_to_supabase(overlay, f"saliency_{idx}.png")
        
        anomalous_frames.append({
            'frame_id': idx,
            'timestamp': idx / 5,  # 5 FPS
            'confidence': float(confidences[idx]),
            'saliency_url': saliency_url
        })
    
    return {
        'overall_confidence': float(confidences.max()),
        'anomalous_frames': anomalous_frames
    }
```

**Testing Checklist:**
- [ ] Grad-CAM computes saliency maps correctly
- [ ] Saliency overlays generated for top frames
- [ ] Saliency URLs uploaded to Supabase
- [ ] Modality agreement score computed (0–1)
- [ ] Explainability adds <5 seconds to inference latency

---

### **DECISION 5: Inference Mode = Asynchronous Job Queue**

**Impact Level:** 🔴 CRITICAL (architectural requirement)

**Current State:**
```python
# No job queue, no async inference
```

**Required Changes:**

**File: `backend/app/routes/jobs.py` (NEW)**
```python
from fastapi import APIRouter, BackgroundTasks, HTTPException
from sqlalchemy import create_engine
from celery import Celery
import uuid

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])

# Database
DATABASE_URL = "postgresql://..."
engine = create_engine(DATABASE_URL)

# Celery
celery_app = Celery(
    'aura_veracity',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

@router.post("/analyze-video")
async def analyze_video(job_request: VideoAnalysisRequest):
    """
    Async video analysis endpoint.
    Returns job_id immediately.
    """
    job_id = str(uuid.uuid4())
    
    # Create job record
    job = Job(
        id=job_id,
        video_url=job_request.video_url,
        status="pending",
        created_at=datetime.now()
    )
    db.add(job)
    db.commit()
    
    # Enqueue to Celery worker
    celery_app.send_task(
        'model_service.tasks.analyze_video',
        args=[job_id, job_request.video_url]
    )
    
    return {
        "job_id": job_id,
        "status": "pending",
        "status_url": f"/api/v1/jobs/{job_id}"
    }

@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Get job status and results (if complete).
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    response = {
        "job_id": job_id,
        "status": job.status,
        "created_at": job.created_at,
        "updated_at": job.updated_at
    }
    
    # Include results if complete
    if job.status == "completed":
        results = db.query(DetectionResult).filter(
            DetectionResult.job_id == job_id
        ).first()
        if results:
            response["results"] = json.loads(results.results_json)
    
    return response
```

**File: `model_service/tasks.py` (NEW)**
```python
from celery import shared_task
import requests
import json

@shared_task
def analyze_video(job_id, video_url):
    """
    Background task: Download video, run model, store results.
    """
    try:
        # 1. Download video from Supabase
        video_path = download_video(video_url)
        
        # 2. Run model inference
        results = run_inference(video_path)
        
        # 3. Store results in database
        store_results(job_id, results)
        
        # 4. Update job status
        update_job_status(job_id, "completed")
        
    except Exception as e:
        print(f"Error analyzing job {job_id}: {e}")
        update_job_status(job_id, "failed", error_msg=str(e))
```

**Testing Checklist:**
- [ ] Celery worker running
- [ ] Redis broker accessible
- [ ] Job created with pending status
- [ ] Celery task enqueued
- [ ] Worker picks up task and processes
- [ ] Results stored in database
- [ ] Job status updates to completed
- [ ] GET /jobs/{job_id} returns results

---

## **📊 IMPLEMENTATION CHECKLIST BY PHASE**

### **Phase 1: Critical Fixes**

```
☐ 1.1: Audio Encoder Replacement
    ☐ Create models/audio_encoder.py
    ☐ Update data/multimodal_dataset.py for waveform loading
    ☐ Update multimodal_model.py to use Wav2Vec2AudioEncoder
    ☐ Update train.py for variable-length audio collation
    ☐ Test: forward pass, training loop, AUC improvement

☐ 1.2: Voice Activity Detection (VAD)
    ☐ Create preprocess/voice_activity_detection.py
    ☐ Integrate into data/multimodal_dataset.py
    ☐ Update multimodal_model.py for VAD mask support
    ☐ Test: VAD accuracy, AUC improvement

☐ 1.3: Temporal Consistency Loss
    ☐ Verify models/losses.py (already defined)
    ☐ Update train.py to use temporal consistency loss
    ☐ Tune weight λ on validation set
    ☐ Test: loss integration, AUC improvement

☐ 1.4: Video-Level Inference Endpoint
    ☐ Create backend/app/routes/jobs.py
    ☐ Create model_service/tasks.py (Celery tasks)
    ☐ Implement async job queue with Redis/Celery
    ☐ Test: end-to-end video upload → processing → results

☐ 1.5: Modality Dropout Fix
    ☐ Update models/multimodal_model.py forward pass
    ☐ Test during training, disable during inference
    ☐ Verify AUC improvement
```

### **Phase 2: High-Impact Improvements**

```
☐ 2.1: Cross-Modal Attention Fusion
    ☐ Create models/fusion.py with CrossModalAttentionFusion
    ☐ Update multimodal_model.py to use attention
    ☐ Retrain and validate AUC improvement

☐ 2.2: Optical Flow
    ☐ Create preprocess/optical_flow.py
    ☐ Update data/multimodal_dataset.py to load optical flow
    ☐ Modify frame_model.py for 5-channel input
    ☐ Retrain and validate AUC improvement

☐ 2.3: Face Alignment
    ☐ Update preprocess/face_detection.py with alignment
    ☐ Update data/multimodal_dataset.py to use aligned crops
    ☐ Retrain and validate AUC improvement

☐ 2.4: Uncertainty Estimation
    ☐ Implement MC-Dropout or temperature scaling
    ☐ Update inference to return confidence intervals
    ☐ Test calibration on Celeb-DF

☐ 2.5: Multi-Task Learning
    ☐ Add auxiliary heads to models/frame_model.py
    ☐ Update train.py with multi-task loss
    ☐ Retrain and validate AUC improvement
```

### **Phase 3: Advanced Methods**

```
☐ 3.1: Transformer Temporal Encoder
    ☐ Create models/temporal_transformer.py
    ☐ Update multimodal_model.py to use transformer
    ☐ Retrain and validate

☐ 3.2: Lip-Sync Verification
    ☐ Create models/lipsync_detector.py
    ☐ Integrate into multimodal_model.py
    ☐ Retrain and validate

☐ 3.3: Ensemble Modeling
    ☐ Train 5 independent models
    ☐ Create serve/ensemble.py
    ☐ Test ensemble inference

☐ 3.4: Adversarial Robustness
    ☐ Create eval/adversarial_eval.py
    ☐ Test FGSM/PGD attacks
    ☐ Optional: adversarial training

☐ 3.5: Explainability
    ☐ Create models/explainability.py
    ☐ Create serve/inference.py with saliency generation
    ☐ Upload saliencies to Supabase
    ☐ Update frontend to display saliency maps
```

---

## **🚀 QUICK START FOR DEVELOPERS**

**If implementing Phase 1.1 (Audio Encoder):**

1. Copy the code from **DECISION 1** section above
2. Create file: `model-service/src/models/audio_encoder.py`
3. Update file: `model-service/src/data/multimodal_dataset.py`
4. Update file: `model-service/src/models/multimodal_model.py`
5. Update file: `model-service/src/train.py`
6. Run: `python train.py --test-one-epoch`
7. Verify: Forward pass works, no OOM, audio features shape correct

**If implementing Phase 2.1 (Fusion):**

1. Copy the code from **DECISION 3** section above
2. Create file: `model-service/src/models/fusion.py`
3. Update file: `model-service/src/models/multimodal_model.py`
4. Run: `python train.py --test-one-epoch`
5. Compare validation AUC with baseline (concatenation)

**If implementing Explainability:**

1. Copy the code from **DECISION 4** section above
2. Create file: `model-service/src/models/explainability.py`
3. Create file: `model-service/src/serve/inference.py`
4. Update file: `backend/app/routes/jobs.py` to return saliency URLs
5. Test: Run on 5 sample videos, verify saliency overlays correct

---

**Document Version:** 1.0  
**Target Audience:** ML Engineers implementing Phase 1–3  
**Ready for Implementation:** ✅ YES

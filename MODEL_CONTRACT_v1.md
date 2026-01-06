# **MODEL CONTRACT v1.0 — QUICK REFERENCE**

**Status:** 🔒 LOCKED  
**Effective Date:** January 3, 2026  
**Design Document:** [ML_SYSTEM_DESIGN.md § XI](ML_SYSTEM_DESIGN.md#xi-model-contract-v1-locked-decisions-)

---

## **📋 ONE-PAGE SUMMARY**

| Decision | Value | Flexibility | Impact |
|----------|-------|-------------|--------|
| **Audio Encoder** | wav2vec2-base (pretrained) | 🔒 Locked | +5–10% AUC gain |
| **Temporal Window** | 1 second (5–10 frames) | 🔒 Locked | Defines architecture |
| **Fusion Strategy** | Cross-modal attention (mid-fusion) | 🔒 Locked | Core forensic signal |
| **Inference Mode** | Async job queue | 🔒 Locked | Non-negotiable for UX |
| **Training Dataset** | FaceForensics++ (primary) + Celeb-DF (validation) | 🔒 Locked | Learned representations |
| **Video Codec** | H.264 + MP4 (H.265 optional) | ⚠️ Flexible | Codec compatibility |
| **Resolution** | 224×224 normalized (accept 240p–1080p) | ✅ Flexible | Input flexibility |
| **Frame Rate** | Sample at 5–10 FPS | ✅ Flexible | Latency/quality tradeoff |
| **Deepfake Coverage** | GAN swaps, reenactment, lip-sync | ✅ Reversible | Can extend v2 |
| **Confidence Calibration** | Temperature scaling | ✅ Flexible | Reliability improvement |
| **Explainability** | Grad-CAM + audio anomalies + modality agreement | ✅ Required | Forensic compliance |

---

## **🎯 PERFORMANCE TARGETS**

```
Current State:           ~70% AUC on FaceForensics++
After Phase 1:           78–82% AUC
After Phase 2:           83–87% AUC
After Phase 3:           88–92% AUC
Final Target (v1):       >85% AUC, <5% FP rate
```

---

## **🔐 LOCKED INTERFACE CONTRACTS**

### **1. Audio Processing Pipeline**

```python
# INPUT: Raw MP4 video file
video_path: str = "sample.mp4"

# EXTRACTION: FFmpeg → WAV
ffmpeg extract_audio(video_path) → audio.wav
  - Format: 16kHz, mono, 16-bit PCM
  - No mel-spectrogram pre-computation
  - VAD applied (speech only)

# ENCODING: Wav2vec2
audio_features = wav2vec2_encoder(audio_waveform)
  - Input shape: (seq_len,)
  - Output shape: (seq_len, 768)
  - Freeze first 8 layers, fine-tune last 4

# OUTPUT: Audio embedding for fusion
audio_embedding: Tensor = (batch, seq_len, 512)
```

### **2. Video Processing Pipeline**

```python
# INPUT: Raw MP4 video file
video_path: str = "sample.mp4"

# FRAME EXTRACTION: FFmpeg @ 5–10 FPS
frames = extract_frames(video_path, fps=5)
  - Shape: List[(H, W, 3)]
  - Only frames with detected faces

# FACE DETECTION: RetinaFace
faces = detect_faces(frames, confidence_threshold=0.95)
  - Bounding boxes + 5-point landmarks
  - Alignment to canonical pose
  - Skip frames with 0 or >1 faces

# APPEARANCE FEATURES: EfficientNet-B3
appearance_feat = frame_model.backbone(aligned_face)
  - Input: (batch, 3, 224, 224)
  - Output: (batch, 1536)

# OPTICAL FLOW: Farneback optical flow
flow = compute_optical_flow(frames[t], frames[t+1])
  - Shape: (H, W, 2)
  - Stack with appearance: (batch, 1536+2)

# TEMPORAL ENCODING: 1D Conv or Transformer
temporal_feat = temporal_encoder(appearance_feat + flow)
  - Input: sequence of 5–10 frames
  - Output: (batch, 2048)

# OUTPUT: Video embedding for fusion
video_embedding: Tensor = (batch, 2048)
```

### **3. Fusion & Classification**

```python
# MID-FUSION: Cross-modal attention
# video_features: (batch, 2048)
# audio_features: (batch, 512)

fused = cross_modal_attention(
    query=video_features,
    key=audio_features,
    value=audio_features
)
# Output: (batch, 2048 + 512)

# CLASSIFICATION: Linear head
logits = classification_head(fused)
# Output: (batch, 2)  # [P(real), P(fake)]

confidence = softmax(logits)[1]  # P(fake)
prediction = "fake" if confidence > 0.5 else "real"
```

### **4. Output Specification**

```json
{
  "job_id": "uuid-here",
  "status": "completed",
  "prediction": {
    "overall_class": "fake",
    "overall_confidence": 0.92,
    "manipulation_confidence": 0.88,
    "modality_agreement": 0.75,
    "confidence_interval": [0.87, 0.96],
    "anomalous_frames": [
      {
        "frame_id": 15,
        "timestamp": 3.0,
        "confidence": 0.95,
        "saliency_url": "https://..."
      }
    ]
  },
  "metadata": {
    "processing_time_sec": 45.3,
    "video_duration_sec": 30,
    "frames_analyzed": 150,
    "model_version": "v1.0",
    "ensemble_size": 5
  }
}
```

---

## **🚫 WHAT CHANGED FROM CURRENT IMPLEMENTATION**

### **Audio**
- ❌ Naive 3-layer CNN (AudioCNN) → ✅ Pretrained wav2vec2
- ❌ Mel-spectrogram preprocessing → ✅ Raw waveform input
- ❌ No VAD → ✅ Speech-only masking
- ✅ No mel-spec caching (faster on-the-fly extraction)

### **Video**
- ✅ EfficientNet-B3 backbone (unchanged)
- ✅ Face detection + bounding box crops
- ❌ No optical flow → ✅ Farneback optical flow extraction
- ❌ No face alignment → ✅ RetinaFace landmark alignment
- ✅ Temporal 1D Conv (Phase 3: → Transformer)

### **Fusion**
- ❌ Late concatenation → ✅ Cross-modal attention (mid-fusion)
- ✅ Binary classification (unchanged)
- ✅ Softmax output (unchanged)

### **Inference**
- ❌ Synchronous (blocking) → ✅ Asynchronous (job queue)
- ❌ No `/analyze-video` endpoint → ✅ Full video inference + aggregation
- ✅ No per-frame saliency → ✅ Grad-CAM + anomaly timestamps

### **Training**
- ❌ No temporal consistency loss → ✅ Integrated into loss computation
- ✅ AdamW optimizer (unchanged)
- ✅ CosineAnnealingLR scheduler (unchanged)
- ✅ FaceForensics++ dataset (unchanged)

---

## **⚙️ CRITICAL CONFIGURATION**

### **Model Configuration (`config/config.yaml`)**

```yaml
model:
  video:
    backbone: "efficientnet_b3"
    pretrained: true
    freeze_backbone: false
  audio:
    encoder: "wav2vec2-base"  # LOCKED
    pretrained: true
    freeze_encoder_blocks: 8  # 0–7
    fine_tune_blocks: 4       # 8–11
  temporal:
    window_frames: 5          # 1 second @ 5 FPS
    encoder_type: "conv1d"    # Phase 3: "transformer"
  fusion:
    strategy: "cross_attention"  # LOCKED
    hidden_dim: 512
    attention_heads: 4
  classification:
    num_classes: 2
    dropout: 0.2

training:
  batch_size: 32
  learning_rate: 1e-4
  warmup_steps: 1000
  epochs: 20
  optimizer: "adamw"
  scheduler: "cosine"
  temporal_consistency_weight: 0.1  # λ
  modality_dropout_prob: 0.2        # During training only

data:
  dataset: "faceforensics++"
  val_dataset: "celeb-df-v2"
  resolution: 224
  frame_rate: 5                      # FPS
  vad_enabled: true
  optical_flow_enabled: true
  augmentation:
    enable: true
    color_jitter: true
    random_crop: true
    random_flip: true
```

---

## **📊 EXAMPLE: INFERENCE FLOW**

```
User uploads video.mp4
    ↓
POST /analyze-video
    ↓
Backend creates job {job_id: "abc-123"}
    ↓
Celery worker picks up job
    ↓
Extract frames @ 5 FPS → [frame_0, frame_1, ..., frame_150]
    ↓
Detect faces + align → [face_0, face_1, ..., face_150]
    ↓
Extract audio → audio.wav (16kHz, mono)
    ↓
wav2vec2 encoding → audio_features (150, 512)
    ↓
Frame inference (batch):
  - EfficientNet features: (150, 1536)
  - Optical flow: (150, 2)
  - TemporalConv pooling: (1, 2048)
    ↓
Audio inference:
  - Wav2vec2 encoding: (seq_len, 768)
  - Mean pooling: (1, 512)
    ↓
Fusion:
  - Cross-modal attention: (1, 2560)
  - Classification head: (1, 2)
  - Softmax: P(fake) = 0.92
    ↓
Explainability:
  - Grad-CAM on top-5 frames
  - Upload saliency PNGs
    ↓
Store results in PostgreSQL
    ↓
Frontend polls /jobs/abc-123
    ↓
Receive results + saliency URLs
    ↓
Display confidence + heatmaps to user
```

---

## **🔗 RELATED DOCUMENTS**

| Document | Purpose | Status |
|----------|---------|--------|
| [ML_SYSTEM_DESIGN.md](ML_SYSTEM_DESIGN.md) | Complete design specification (10 sections) | ✅ Reference |
| [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) | Phase-by-phase implementation plan | ✅ Active |
| [PROJECT_PLAN.md](PROJECT_PLAN.md) | High-level system architecture | ✅ Reference |

---

## **❓ FAQ**

**Q: Can we use a different audio encoder (HuBERT, Wav2Vec2-large)?**  
A: HuBERT is locked as secondary choice if wav2vec2-base underperforms. Wav2Vec2-large possible in Phase 3 (cost: +20% latency).

**Q: Why not 3D CNN for temporal encoding?**  
A: 1D Conv chosen for efficiency. Phase 3 explores Transformer-based alternatives.

**Q: Can we support multi-face videos?**  
A: v1 explicitly single-face (simplifies explainability). v2 can extend to multi-face with per-face confidence scores.

**Q: What if inference exceeds 2 minutes per video?**  
A: Flag as blocker. Options: (a) reduce temporal window to 0.5s, (b) lower resolution to 128×128, (c) add GPU resources.

**Q: Is speaker-aware audio OK?**  
A: No. Speaker-aware encoders overfit to speaker identity, harming generalization across speakers.

**Q: Can we use synthetic data for training?**  
A: v1 uses only FaceForensics++ + Celeb-DF. Synthetic data augmentation (Phase 2) explored but not primary.

---

**Document Version:** 1.0  
**Last Updated:** January 3, 2026  
**Locked By:** Principal ML Engineer  
**Ready for Implementation:** ✅ Yes

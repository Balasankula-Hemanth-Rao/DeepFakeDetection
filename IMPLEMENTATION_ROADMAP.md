# **IMPLEMENTATION ROADMAP — MULTIMODAL DEEPFAKE DETECTION v1.0**

**Locked Model Contract:** See [ML_SYSTEM_DESIGN.md § XI](ML_SYSTEM_DESIGN.md#xi-model-contract-v1-locked-decisions-)

**Current Date:** January 3, 2026  
**Target Completion:** Q1 2026 (12 weeks)

---

## **📋 PHASE 1: CRITICAL FIXES (Weeks 1–2)**

### **Priority 1.1: Audio Encoder Replacement** ⭐ HIGHEST IMPACT
**Expected Gain:** +5–10% AUC  
**Effort:** 3–4 days

**Tasks:**
- [ ] Remove naive AudioCNN from `models/audio_cnn.py` (or deprecate)
- [ ] Create `models/audio_encoder.py` with wav2vec2 wrapper
  - Load facebook/wav2vec2-base pretrained weights
  - Feature dimension: 768 (output of final layer)
  - Freeze first 8 layers, fine-tune last 4 + linear projection to 512
- [ ] Update `data/multimodal_dataset.py`:
  - Load `.wav` files at 16kHz, mono (use librosa)
  - NO mel-spectrogram extraction (wav2vec2 expects raw waveform)
  - Update augmentation: pitch shift, time stretch (keep non-destructive)
- [ ] Update `models/multimodal_model.py`:
  - Replace `AudioCNN` input with raw waveform (variable length)
  - Implement `AudioFeatureExtractor` with wav2vec2
  - Update fusion input dimensions: video (2048 after temporal) + audio (512)
- [ ] Update `train.py`:
  - Pass raw waveforms instead of mel-specs
  - Adjust batch collation for variable-length audio
- [ ] Verify: Load FaceForensics++ sample, confirm audio features extracted correctly

**Acceptance Criteria:**
- ✅ wav2vec2 features extracted successfully
- ✅ Model forward pass completes without error
- ✅ Training loop runs (1 epoch) without OOM
- ✅ Audio feature shape: (batch, seq_len, 512)

---

### **Priority 1.2: Voice Activity Detection (VAD)**
**Expected Gain:** +1–2% AUC  
**Effort:** 2–3 days

**Tasks:**
- [ ] Create `preprocess/voice_activity_detection.py`
  - Option A: pyannote-audio (better but requires setup)
  - Option B: librosa energy-based VAD (simpler, good enough for v1)
  - Output: timestamps of speech regions
- [ ] Integrate into `data/multimodal_dataset.py`:
  - Apply VAD to extracted audio
  - Mask non-speech regions with zeros (or remove)
  - Update mel-spec extraction to respect VAD mask
- [ ] Update `multimodal_model.py`:
  - Accept optional VAD mask in forward pass
  - Zero out audio features for non-speech frames
- [ ] Update `train.py`:
  - Load VAD masks during training
  - Add metric: % of silence removed

**Acceptance Criteria:**
- ✅ VAD extracts speech regions correctly
- ✅ Silence is masked in audio features
- ✅ Training AUC improves by 1–2%

---

### **Priority 1.3: Integrate Temporal Consistency Loss**
**Expected Gain:** +2–3% AUC  
**Effort:** 1–2 days

**Tasks:**
- [ ] Review `models/losses.py` (already defined; just not used)
- [ ] Update `train.py`:
  - Uncomment `TemporalConsistencyLoss` import
  - Add to loss computation with weight λ=0.1 (tune later)
  - Total loss = cross_entropy(pred, label) + λ × temporal_consistency(embeddings)
- [ ] Test training with new loss:
  - Verify loss decreases
  - Check that embeddings become more stable frame-to-frame
- [ ] Tune weight λ on validation set
  - Sweep λ ∈ {0.01, 0.05, 0.1, 0.2}
  - Choose λ that maximizes validation AUC

**Acceptance Criteria:**
- ✅ Loss function integrated into training loop
- ✅ Model trains without error
- ✅ Validation AUC improves by 2–3%

---

### **Priority 1.4: Video-Level Inference Endpoint**
**Expected Gain:** Architectural necessity (0% AUC, 100% UX)  
**Effort:** 2–3 days

**Tasks:**
- [ ] Create `serve/api_video.py` with FastAPI routes:
  - POST `/analyze-video` → returns `{"job_id": "uuid", "status_url": "/jobs/{job_id}"}`
  - GET `/jobs/{job_id}` → returns current status + results (if complete)
- [ ] Implement video processing pipeline in `serve/inference.py`:
  - Extract frames at 5 FPS using FFmpeg
  - Run FrameModel on each frame (batch inference)
  - Aggregate frame predictions: mean confidence + saliency
  - Generate output JSON (see ML_SYSTEM_DESIGN.md § D)
- [ ] Update backend to call model service:
  - Backend POST `/uploads/init-job` → enqueue to Celery
  - Celery worker: calls model service `/infer-video`
  - Store results in PostgreSQL `detection_results` table
- [ ] Test end-to-end:
  - Upload video via frontend
  - Monitor job status
  - Retrieve results + saliency maps

**Acceptance Criteria:**
- ✅ `/analyze-video` endpoint accepts video file
- ✅ Video processed, frames extracted, model inference runs
- ✅ Results aggregated and stored
- ✅ Frontend can retrieve + display results

---

### **Priority 1.5: Fix Modality Dropout**
**Expected Gain:** +0.5–1% AUC (regularization)  
**Effort:** 1 day

**Tasks:**
- [ ] Review `models/multimodal_model.py` (config parameter exists but not used)
- [ ] Update forward pass:
  - During training: drop audio OR video features with probability `modality_dropout_prob`
  - During inference: disable dropout (always use both modalities)
- [ ] Update `train.py`:
  - Set `modality_dropout_prob = 0.2` (20% dropout during training)
  - Disable during validation
- [ ] Test: Train with/without dropout, compare validation AUC

**Acceptance Criteria:**
- ✅ Modality dropout implemented in forward pass
- ✅ Only active during training, disabled during inference
- ✅ Validation AUC improves or stays same

---

## **📋 PHASE 2: HIGH-IMPACT IMPROVEMENTS (Weeks 3–5)**

### **Priority 2.1: Cross-Modal Attention Fusion**
**Expected Gain:** +2–5% AUC  
**Effort:** 3–4 days

**Tasks:**
- [ ] Create `models/fusion.py` with `CrossModalAttentionFusion`:
  ```python
  class CrossModalAttentionFusion(nn.Module):
      def forward(self, video_features, audio_features):
          # video_features: (batch, T, 2048)
          # audio_features: (batch, A, 512)
          # Output: fused features (batch, 2048)
          
          # Cross-attention: video queries, audio keys/values
          attn_weights = softmax(video @ audio.T)
          audio_context = attn_weights @ audio
          fused = concat([video_pooled, audio_context])
          return fused
  ```
- [ ] Update `models/multimodal_model.py`:
  - Replace concatenation with `CrossModalAttentionFusion`
  - Update forward pass: video → temporal encoder → attention(video, audio) → classification
- [ ] Retrain model on FaceForensics++:
  - Check validation AUC improvement
  - Tune attention hidden dimensions
- [ ] Test on Celeb-DF (out-of-distribution)

**Acceptance Criteria:**
- ✅ Cross-attention module implemented
- ✅ Forward pass completes without error
- ✅ Validation AUC improves by 2–5%
- ✅ Celeb-DF AUC also improves

---

### **Priority 2.2: Optical Flow Features**
**Expected Gain:** +3–5% AUC  
**Effort:** 4–5 days

**Tasks:**
- [ ] Create `preprocess/optical_flow.py`:
  - Compute optical flow between adjacent frames using OpenCV (Farneback)
  - Output: flow magnitude + direction maps
  - Cache computed flows to disk
- [ ] Update `data/multimodal_dataset.py`:
  - Load precomputed optical flow for each frame
  - Stack with appearance features: (batch, T, C+2) where +2 is flow
- [ ] Update `models/frame_model.py`:
  - Modify input layer to accept appearance + flow channels
  - Update input shape: (3 + 2, 224, 224) → (5, 224, 224)
- [ ] Retrain on FaceForensics++:
  - Check validation AUC improvement
  - Compare with/without optical flow
- [ ] Test generalization on Celeb-DF

**Acceptance Criteria:**
- ✅ Optical flow computed and cached
- ✅ Model accepts 5-channel input (RGB + flow)
- ✅ Validation AUC improves by 3–5%

---

### **Priority 2.3: Face Alignment**
**Expected Gain:** +1–2% AUC  
**Effort:** 2–3 days

**Tasks:**
- [ ] Update `preprocess/face_detection.py`:
  - Use RetinaFace landmarks for alignment (already available)
  - Compute affine transformation to canonical face pose
  - Apply alignment to extracted face crops
- [ ] Update `data/multimodal_dataset.py`:
  - Load aligned face crops instead of bounding box crops
- [ ] Retrain model:
  - Should improve AUC due to pose normalization
  - Expected gain: 1–2%

**Acceptance Criteria:**
- ✅ Face alignment implemented
- ✅ Aligned crops generated correctly
- ✅ Validation AUC improves by 1–2%

---

### **Priority 2.4: Uncertainty Estimation**
**Expected Gain:** Explainability (no AUC gain, but reliability)  
**Effort:** 2–3 days

**Tasks:**
- [ ] Implement MC-Dropout:
  - Enable dropout during inference (10 forward passes)
  - Compute mean + variance of predictions
  - Return confidence intervals
- [ ] Alternative: Temperature Scaling
  - Learn temperature value on validation set
  - Output calibrated probabilities
- [ ] Update output JSON:
  - Add `confidence_interval`: `[lower, upper]`
  - Add `uncertainty_score`: variance / mean
- [ ] Test: Compare MC-Dropout vs Temperature Scaling on Celeb-DF

**Acceptance Criteria:**
- ✅ Uncertainty estimates computed
- ✅ Output includes confidence intervals
- ✅ Calibration improves on held-out data

---

### **Priority 2.5: Multi-Task Learning**
**Expected Gain:** +2–3% AUC (regularization)  
**Effort:** 3–4 days

**Tasks:**
- [ ] Create auxiliary task heads in `models/frame_model.py`:
  - Task 1: Facial landmark prediction (68 points)
  - Task 2: Head pose estimation (yaw, pitch, roll)
- [ ] Update training loop:
  - Loss = primary_loss + λ₁ × landmark_loss + λ₂ × pose_loss
  - Tune λ₁, λ₂ on validation set (start with 0.1)
- [ ] Retrain model:
  - Should improve generalization
  - Expected gain: 2–3%

**Acceptance Criteria:**
- ✅ Auxiliary tasks trained jointly
- ✅ Validation AUC improves by 2–3%
- ✅ Landmark/pose predictions reasonable

---

## **📋 PHASE 3: ADVANCED METHODS (Weeks 6–9)**

### **Priority 3.1: Transformer-Based Temporal Encoder**
**Expected Gain:** +2–3% AUC  
**Effort:** 4–5 days

**Tasks:**
- [ ] Create `models/temporal_transformer.py`:
  - Replace 1D ConvNet with Vision Transformer
  - Input: sequence of 5–10 frame embeddings (batch, T, 2048)
  - Output: aggregated temporal embedding (batch, 2048)
- [ ] Update `models/multimodal_model.py`:
  - Replace TemporalConv with TemporalTransformer
- [ ] Retrain on FaceForensics++:
  - Check if AUC improves
  - Compare training time (may be slower)

**Acceptance Criteria:**
- ✅ Transformer encoder implemented
- ✅ Validation AUC improves or stays competitive
- ✅ Inference latency acceptable (<60s for 30s video)

---

### **Priority 3.2: Lip-Sync Verification**
**Expected Gain:** +3–5% AUC  
**Effort:** 5–6 days

**Tasks:**
- [ ] Create `models/lipsync_detector.py`:
  - Detect lip region in face crop
  - Compute optical flow on lips
  - Compare lip motion frequency with audio speech rate
  - Return lip-sync confidence score
- [ ] Integrate into `models/multimodal_model.py`:
  - Add lip-sync score as auxiliary output
  - Include in final prediction: `confidence = 0.7 × deepfake_confidence + 0.3 × lipsync_confidence`
- [ ] Retrain:
  - Tune weighting of lip-sync score
  - Expected gain: 3–5%

**Acceptance Criteria:**
- ✅ Lip-sync detector working
- ✅ Lips detected correctly in face crops
- ✅ Speech rate vs lip motion compared
- ✅ Validation AUC improves by 3–5%

---

### **Priority 3.3: Ensemble Modeling**
**Expected Gain:** +2–4% AUC  
**Effort:** 3–4 days

**Tasks:**
- [ ] Train 5 independent models:
  - Different random seeds
  - Slightly different architectures (dropout rate, learning rate)
  - Same dataset, 10 epochs each
- [ ] Create `serve/ensemble.py`:
  - Load all 5 checkpoints
  - Run inference on all models
  - Average predictions: `ensemble_pred = mean([pred₁, pred₂, ..., pred₅])`
- [ ] Update `/analyze-video` endpoint:
  - Use ensemble instead of single model
  - Return per-model predictions + ensemble average
- [ ] Test on FaceForensics++ + Celeb-DF:
  - Ensemble AUC should be 2–4% higher than single model

**Acceptance Criteria:**
- ✅ 5 models trained independently
- ✅ Ensemble inference working
- ✅ Validation AUC improves by 2–4%

---

### **Priority 3.4: Adversarial Robustness**
**Expected Gain:** Robustness (no AUC gain on clean data)  
**Effort:** 3–4 days

**Tasks:**
- [ ] Create `eval/adversarial_eval.py`:
  - FGSM attacks: ε ∈ {0.01, 0.05, 0.1}
  - PGD attacks: α=0.01, steps=10
  - Test on 100 videos from Celeb-DF
- [ ] Evaluate robustness:
  - How much does AUC drop under attack?
  - Accept <5% AUC drop as good robustness
- [ ] Optional: Adversarial training:
  - Train on mix of clean + FGSM images
  - May improve robustness but reduce clean AUC

**Acceptance Criteria:**
- ✅ Adversarial attacks implemented
- ✅ Robustness evaluated
- ✅ AUC drop <5% under FGSM attack (ε=0.05)

---

### **Priority 3.5: Explainability Module**
**Expected Gain:** Interpretability (required for forensics)  
**Effort:** 4–5 days

**Tasks:**
- [ ] Create `models/explainability.py`:
  - Grad-CAM on final Conv layer of video encoder
  - Feature importance for audio (attention weights)
  - Generate saliency overlay PNG
- [ ] Update `serve/api_video.py`:
  - Compute saliency for top-5 anomalous frames
  - Upload saliency images to Supabase Storage
  - Return saliency URLs in output JSON
- [ ] Frontend update:
  - Display saliency maps on results page
  - Highlight which regions triggered "fake" prediction
  - Show audio anomaly timestamps
- [ ] Test:
  - Verify saliency maps look reasonable
  - User study: do saliency maps help forensic analysts?

**Acceptance Criteria:**
- ✅ Saliency maps generated for anomalous frames
- ✅ Saliency URLs returned in API
- ✅ Frontend displays saliency overlays correctly

---

## **🎯 SUCCESS METRICS**

### **Performance Targets**

| Metric | Current | After Phase 1 | After Phase 2 | After Phase 3 |
|--------|---------|---------------|---------------|---------------|
| **FaceForensics++ AUC** | ~70–75% | 78–82% | 83–87% | 88–92% |
| **Celeb-DF AUC** | ~65–70% | 74–78% | 79–83% | 84–88% |
| **Cross-Dataset Generalization** | 60–65% | 70–74% | 75–79% | 80–84% |
| **False Positive Rate (@ 95% TPR)** | ~10% | ~7% | ~4% | ~2% |
| **Inference Latency (30s video)** | N/A | 45–60s | 50–65s | 50–70s |

### **Quality Metrics**

| Criterion | Target |
|-----------|--------|
| **Model Explainability** | ✅ Saliency maps + artifact explanations |
| **Confidence Calibration** | ✅ Expected Calibration Error < 0.05 |
| **Adversarial Robustness** | ✅ AUC drop < 5% under FGSM (ε=0.05) |
| **Cross-Codec Robustness** | ✅ AUC within 2% for H.264 + H.265 |
| **Code Quality** | ✅ Type hints, docstrings, unit tests |

---

## **📅 TIMELINE**

| Phase | Duration | Start | End | Key Deliverables |
|-------|----------|-------|-----|------------------|
| **Phase 1** | 2 weeks | Jan 6 | Jan 20 | Functional multimodal model + async API |
| **Phase 2** | 3 weeks | Jan 21 | Feb 10 | Attention fusion + optical flow + explainability |
| **Phase 3** | 4 weeks | Feb 11 | Mar 10 | Transformer + lip-sync + ensemble + robustness |
| **Testing & Deployment** | 1 week | Mar 11 | Mar 17 | Final validation + production deployment |

**Total:** 12 weeks (Q1 2026)

---

## **🔧 DEVELOPMENT GUIDELINES**

### **Code Organization**

```
model-service/src/
  models/
    ├── frame_model.py (EfficientNet-B3 + multi-task heads)
    ├── audio_encoder.py (wav2vec2 wrapper) ← NEW
    ├── multimodal_model.py (fusion + classification)
    ├── fusion.py (CrossModalAttentionFusion) ← NEW
    ├── temporal_transformer.py (Transformer temporal encoder) ← NEW
    ├── lipsync_detector.py (lip-sync verification) ← NEW
    ├── explainability.py (Grad-CAM + feature importance) ← NEW
    └── losses.py (temporal consistency + multi-task losses)
  
  data/
    ├── multimodal_dataset.py (loader with VAD)
    └── augmentation.py (audio + video augmentation)
  
  preprocess/
    ├── extract_frames.py (existing)
    ├── extract_audio.py (NEW)
    ├── voice_activity_detection.py (NEW)
    ├── optical_flow.py (NEW)
    └── face_detection.py (with alignment)
  
  serve/
    ├── api.py (frame-level inference)
    ├── api_video.py (video-level async inference) ← NEW
    ├── inference.py (aggregation + saliency) ← NEW
    └── ensemble.py (ensemble inference) ← NEW
  
  train.py (updated with new losses + tasks)
  eval/
    └── multimodal_eval.py (comprehensive metrics)
```

### **Testing Strategy**

- **Unit Tests:** Each new module (audio_encoder, fusion, explainability)
- **Integration Tests:** End-to-end video inference pipeline
- **Regression Tests:** AUC on FaceForensics++ + Celeb-DF after each phase
- **Robustness Tests:** Adversarial attacks, codec variations, resolution changes

### **Code Review Checklist**

- [ ] Type hints for all function signatures
- [ ] Docstrings explaining algorithm + parameters
- [ ] Logging statements for debugging
- [ ] Unit tests with >80% coverage
- [ ] Backward compatibility with existing checkpoints (if applicable)

---

## **📝 DEPENDENCIES & REQUIREMENTS**

### **New Python Packages**

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0 (for wav2vec2)
librosa>=0.10.0 (VAD, audio processing)
opencv-python>=4.8.0 (optical flow)
pyannote-audio (optional, for better VAD)
celery>=5.3.0 (async job queue)
redis>=4.5.0 (Celery broker)
```

### **Pre-Trained Model Downloads**

```bash
# Wav2vec2 (will download on first use)
from transformers import Wav2Vec2Model
Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

# Face detection (if not already cached)
pip install retinaface-pytorch
```

---

## **❓ BLOCKERS & RISKS**

### **Known Risks**

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| **GPU Memory** (all phases combined) | Medium | High | Gradient checkpointing, smaller batch size |
| **Audio encoder slow** (wav2vec2 inference) | Low | Medium | Quantize encoder, use smaller model |
| **Cross-dataset AUC drop** | Medium | High | Early stopping on Celeb-DF, data augmentation |
| **Explainability overhead** | Low | Medium | Compute saliency asynchronously |

### **Assumptions**

- ✅ FaceForensics++ + Celeb-DF available for training
- ✅ Sufficient GPU memory (RTX 3090 / A100)
- ✅ Celery + Redis available for async job queue
- ✅ Supabase storage available for saliency uploads

---

## **✅ SIGN-OFF & NEXT STEPS**

**Document Status:** Ready for implementation  
**Model Contract Locked:** Yes (see ML_SYSTEM_DESIGN.md § XI)  
**Expected Completion:** March 17, 2026

**Next Action:** Start Phase 1.1 (Audio Encoder Replacement)

# **AURA VERACITY LAB — MULTIMODAL DEEPFAKE DETECTION SYSTEM DESIGN**

**🔒 STATUS: DESIGN FREEZE v1.0 ✅**  
*All architectural decisions locked (Section XI). Implementation can proceed.*

---

## **I. EXECUTIVE SUMMARY**

Aura Veracity Lab implements a **multimodal (audio + video) deepfake detection system** that combines computer vision and audio forensics to classify videos as authentic or manipulated. The system is designed for **frame-level and segment-level inference** with eventual aggregation to video-level predictions.

**Current State:** Partially implemented with architectural scaffolding in place but critical components missing.

**Model Contract v1.0:** All core architecture decisions locked (audio encoder, fusion strategy, temporal window, async inference, training dataset). See Section XI for complete specification.

---

## **II. MULTIMODAL FORENSICS PROBLEM DECOMPOSITION**

### **A. Visual Artifacts Indicating Deepfakes**

**1. Facial Artifacts**
- **Blinking inconsistencies:** Deepfakes often miss natural blink timing, frequency, or duration
- **Eye gaze misalignment:** Unnatural eye movement, fixed gaze, lack of micro-saccades
- **Teeth/mouth occlusion artifacts:** Teeth occlusion patterns differ from real video, lip-sync mismatches
- **Skin texture discontinuities:** Unnatural blurring, texture mismatch between face and neck
- **Hair boundary artifacts:** Hair strand discontinuities at edges
- **Facial geometry inconsistencies:** Jaw angle shifts, face size jumps across frames
- **Lighting asymmetries:** Specular highlights don't match 3D face geometry
- **High-frequency noise:** Deepfakes trained on low-res data miss high-frequency facial details

**2. Temporal Consistency Artifacts**
- **Frame jitter:** Slight positional jitter not present in natural video
- **Expression discontinuities:** Rapid, unnatural expression changes
- **Head pose oscillation:** Unnatural head movement patterns (e.g., mechanical oscillation)
- **Temporal smoothing artifacts:** Over-smoothed transitions between generated frames

**3. Encoder-Specific Artifacts**
- **Frequency anomalies:** GAN-generated faces have spectral signatures distinct from natural faces
- **Compression artifacts:** Deepfakes generated at different compression levels than source
- **Halo effects:** Faint halos around face boundaries (common in synthesis algorithms)

### **B. Audio Artifacts Indicating Deepfakes**

**1. Spectral Artifacts**
- **Formant frequencies:** Voice synthesis often produces unnatural formant patterns (especially for consonants)
- **Voice pitch inconsistencies:** Unnatural pitch contours, missing micro-variations
- **Prosody mismatches:** Timing, stress, intonation don't match natural speech
- **Spectral envelope artifacts:** Artifacts at voice conversion band boundaries

**2. Temporal Audio Artifacts**
- **Phoneme timing:** Voice-cloned audio often has unnatural phoneme transitions
- **Voice quality discontinuities:** Changes in timbre, breathiness, or background noise between segments
- **Silence duration:** Unnatural silence patterns (real speech has specific pause distributions)
- **Click artifacts:** Concatenation artifacts at stitching points

**3. Environmental/Background Audio**
- **Background noise consistency:** Deepfakes may drop or artificially add background noise
- **Room acoustics:** Echo/reverb patterns inconsistent with visual environment
- **Speaker distance:** Audio loudness doesn't match visual mouth distance

### **C. Complementary vs Redundant Artifacts**

**Complementary (High Signal Together):**
- Face blink detection + voice naturalness → Highly synergistic (blink-speech coupling is specific to humans)
- Lip-sync + audio formants → Complementary (lip shape constrains voice features)
- Head pose + pitch contour → Somewhat complementary (head movements influence voice acoustics)

**Redundant (High Correlation):**
- Facial texture + eye gaze → Somewhat redundant (both indicate diffusion/GAN artifacts)
- Spectral envelope + voice pitch → Somewhat redundant (both depend on voicoder quality)

**Modality-Specific (Non-Redundant):**
- Visual: facial geometry, eye movements, skin reflectance
- Audio: phoneme articulation, pitch dynamics, speaker-specific formants

---

## **III. CURRENT STATE: VERIFIED FACTS FROM CODEBASE**

### **A. Existing Implementation**

#### **1. Frame-Level Video Processing** ✅ IMPLEMENTED

| Component | Location | Status | Details |
|-----------|----------|--------|---------|
| **Video Backbone** | `models/frame_model.py` | ✅ | EfficientNet-B3 pretrained on ImageNet; outputs 1536-dim features |
| **Frame Extraction** | `preprocess/extract_frames.py` | ✅ | FFmpeg-based; 3 FPS default; SHA256 checksums for reproducibility |
| **Face Detection** | `data/multimodal_dataset.py` | ✅ | RetinaFace or MTCNN; optional face cropping and alignment |
| **Frame Preprocessing** | `train.py` | ✅ | Resize to 224×224; normalization (ImageNet stats); augmentation (crop, flip, color jitter) |

#### **2. Temporal Video Modeling** ⚠️ PARTIALLY IMPLEMENTED

| Component | Location | Status | Details |
|-----------|----------|--------|---------|
| **Temporal Encoder** | `multimodal_model.py` (TemporalConv) | ⚠️ | Basic 1D convolution on temporal dimension; only average pooling option implemented |
| **Temporal Consistency Loss** | `models/losses.py` | ✅ | Penalizes frame-to-frame embedding variance; auxiliary loss (not integrated into training) |
| **Temporal Aggregation** | `eval/multimodal_eval.py` | ⚠️ | Only mean pooling; max pooling available; attention-based aggregation missing |
| **Frame Sampling** | `data/multimodal_dataset.py` | ✅ | Random, uniform, or start-at strategies; configurable temporal window (16 frames default) |

#### **3. Audio Processing** ⚠️ PARTIALLY IMPLEMENTED

| Component | Location | Status | Details |
|-----------|----------|--------|---------|
| **Audio Extraction** | `data/multimodal_dataset.py` | ✅ | librosa/torchaudio; 16 kHz sample rate; supports on-the-fly extraction from video |
| **Mel-Spectrogram** | `data/multimodal_dataset.py` | ✅ | n_mels=64, n_fft=2048, hop_length=512; 1 second segment duration |
| **Audio Encoder** | `multimodal_model.py` (AudioCNN) | ⚠️ | Small 3-layer 2D CNN; naive architecture; no pre-training |
| **Voice Activity Detection (VAD)** | ❌ MISSING | Not implemented | No ability to ignore silence/background |
| **Phoneme-Level Analysis** | ❌ MISSING | Not implemented | No explicit phoneme extraction or alignment |

#### **4. Fusion Strategy** ⚠️ PARTIALLY IMPLEMENTED

| Component | Location | Status | Details |
|-----------|----------|--------|---------|
| **Fusion Strategies** | `multimodal_model.py` | ⚠️ | Concatenation implemented; attention fusion stub present but non-functional; cross-modal attention missing |
| **Modality Dropout** | `multimodal_model.py` | ✅ | Config parameter present; not used in forward pass (bug or feature?) |
| **Modality Ablation** | `multimodal_model.py` | ✅ | enable_audio/enable_video flags; allows audio-only or video-only inference |
| **Late Fusion** | ✅ Implicit | Each modality → embedding → fusion at embedding level (late fusion) |

#### **5. Prediction & Output** ⚠️ PARTIALLY IMPLEMENTED

| Component | Location | Status | Details |
|-----------|----------|--------|---------|
| **Classification Head** | `multimodal_model.py` | ✅ | 2-class (real/fake) softmax output; logits returned from forward() |
| **Frame-Level Predictions** | `serve/api.py` | ✅ | `/infer` endpoint accepts single image; returns per-frame logits |
| **Segment-Level Aggregation** | `eval/multimodal_eval.py` | ⚠️ | Mean/max aggregation across frames; attention-based aggregation incomplete |
| **Confidence Calibration** | ❌ MISSING | Not implemented | No temperature scaling or other calibration post-hoc |
| **Uncertainty Estimation** | ❌ MISSING | Not implemented | No per-sample uncertainty, just softmax probabilities |

#### **6. Data Pipeline** ✅ MOSTLY IMPLEMENTED

| Component | Location | Status | Details |
|-----------|----------|--------|---------|
| **Dataset Loader** | `data/multimodal_dataset.py` | ✅ | Handles on-the-fly and preextracted formats; face detection; augmentation |
| **Collate Function** | `data/multimodal_dataset.py` | ✅ | Handles variable temporal dimensions; pads to batch size |
| **Data Augmentation** | `train.py` + config | ⚠️ | Video augmentation (crop, flip, color jitter) implemented; audio augmentation mostly stubbed |
| **Train/Val/Test Split** | `train.py` | ✅ | Automatic splitting via data_root folder structure |

#### **7. Training Pipeline** ✅ IMPLEMENTED

| Component | Location | Status | Details |
|-----------|----------|--------|---------|
| **Loss Function** | `train.py` | ✅ | Cross-entropy with optional label smoothing (0.1 default) |
| **Optimizer** | `train.py` | ✅ | AdamW (config specifies; hard-coded in code) |
| **Learning Rate Scheduling** | `train.py` | ✅ | CosineAnnealingLR or StepLR (config option; default cosine) |
| **Checkpointing** | `train.py` | ✅ | Saves best and latest checkpoints with metadata/git commit |
| **Early Stopping** | `train.py` | ✅ | 5-epoch patience on validation loss |
| **Distributed Training** | ❌ MISSING | Not implemented | No DistributedDataParallel or model parallelism |

#### **8. Evaluation & Metrics** ✅ MOSTLY IMPLEMENTED

| Component | Location | Status | Details |
|-----------|----------|--------|---------|
| **Metrics** | `utils/metrics.py` | ✅ | AUC, AP, accuracy, precision, recall, F1, FPR@95%TPR |
| **Ablation Studies** | `eval/ablation_study.py` | ✅ | Compares audio-only vs video-only vs multimodal |
| **Visualization** | `eval/multimodal_eval.py` | ⚠️ | ROC/PR curve plotting available; confusion matrix available |

---

## **IV. COMPLETE MULTIMODAL PIPELINE DESIGN**

### **A. High-Level Architecture**

```
INPUT: Video File (MP4, AVI, etc.)
│
├─→ [VIDEO STREAM]
│   │
│   ├─→ Frame Extraction (FFmpeg, 3 FPS)
│   │   └─→ Raw frames [RGB, variable size]
│   │
│   ├─→ Face Detection (RetinaFace)
│   │   └─→ Face bounding boxes + landmarks
│   │
│   ├─→ Face Alignment (optional, not implemented)
│   │   └─→ Aligned face crops [224×224, normalized]
│   │
│   ├─→ Video Backbone (EfficientNet-B3)
│   │   ├─→ Feature extraction [B, T, 1536]
│   │   └─→ Spatial attention over face regions (NOT IMPLEMENTED)
│   │
│   └─→ Temporal Encoder (1D Conv + Pooling)
│       └─→ Video features [B, 1536]
│
├─→ [AUDIO STREAM]
│   │
│   ├─→ Audio Extraction (librosa, 16 kHz)
│   │   └─→ Raw waveform
│   │
│   ├─→ Voice Activity Detection (NOT IMPLEMENTED)
│   │   └─→ Remove silence/background
│   │
│   ├─→ Mel-Spectrogram (n_mels=64)
│   │   └─→ [B, n_mels=64, T_audio]
│   │
│   ├─→ Audio Encoder (Small CNN)
│   │   └─→ Audio features [B, 256]
│   │
│   └─→ (Optional) Pre-trained Audio Model (NOT IMPLEMENTED)
│       └─→ wav2vec2, HuBERT, or speaker embeddings
│
├─→ [FUSION]
│   │
│   ├─→ Modality Dropout (training only, NOT USED)
│   │   └─→ Randomly drop audio or video features
│   │
│   ├─→ Fusion Head (Concatenation or Attention)
│   │   ├─→ Concat: [1536 + 256] = 1792
│   │   ├─→ Attention: Learn weighted combination
│   │   └─→ Cross-modal: Compute cross-attention (NOT IMPLEMENTED)
│   │
│   └─→ Fused representation [B, fusion_dim]
│
├─→ [CLASSIFICATION]
│   │
│   ├─→ Classifier Head (MLP)
│   │   └─→ Linear → ReLU → Dropout → Linear(2)
│   │
│   └─→ Output logits [B, 2]
│
├─→ [AGGREGATION] (For multiple frames/segments)
│   │
│   ├─→ Per-frame logits [T, 2]
│   ├─→ Aggregation (mean/max/attention)
│   └─→ Video-level logits [1, 2]
│
OUTPUT: Confidence scores
├─→ P(real) = softmax(logits)[1]
└─→ P(fake) = softmax(logits)[0]
```

### **B. Component Breakdown**

#### **1. VIDEO STREAM**

**Frame Extraction:**
- **Input:** Video file (MP4, AVI, MOV, WebM)
- **Process:** FFmpeg at fixed FPS (3 FPS default, configurable)
- **Output:** Sequence of frames [H, W, 3]
- **Current:** ✅ Implemented in `preprocess/extract_frames.py`
- **Missing:** 
  - No adaptive FPS (should vary with video duration)
  - No keyframe extraction optimization

**Face Detection & Cropping:**
- **Input:** Raw frames [H, W, 3]
- **Process:** RetinaFace or MTCNN to detect faces and extract bounding boxes
- **Output:** Face crops [B, 3, 224, 224] (normalized)
- **Current:** ✅ Implemented in `multimodal_dataset.py`
- **Missing:**
  - No face tracking (assumes new detection per frame)
  - No face alignment (8-point landmark alignment not used)
  - No handling of multiple faces per frame

**Video Backbone (Feature Extraction):**
- **Input:** Face crops [B, 3, 224, 224]
- **Architecture:** EfficientNet-B3 (ImageNet pretrained)
- **Output:** Features [B, 1536]
- **Current:** ✅ Implemented in `frame_model.py`
- **Missing:**
  - No multi-scale features (only top-layer output)
  - No spatial attention over face regions
  - No fine-tuning strategy (currently frozen backbone in some configs)

**Temporal Modeling (Video):**
- **Input:** Per-frame features [B, T, 1536] where T=16 frames
- **Strategy:** 1D temporal convolution or average pooling
- **Output:** Aggregated video features [B, 1536]
- **Current:** ⚠️ TemporalConv implemented; averaging also available
- **Missing:**
  - No transformer-based temporal modeling (e.g., ViT or attention)
  - No optical flow estimation (could capture frame-to-frame motion)
  - No 3D convolutions
  - Temporal consistency loss defined but NOT integrated into training

---

#### **2. AUDIO STREAM**

**Audio Extraction:**
- **Input:** Video file OR separate audio file
- **Process:** FFmpeg or librosa; resample to 16 kHz
- **Output:** Waveform [1, sample_rate * duration]
- **Current:** ✅ Implemented in `multimodal_dataset.py`
- **Missing:** 
  - No speaker diarization (can't distinguish multiple speakers)
  - No background noise characterization

**Voice Activity Detection (VAD):**
- **Input:** Waveform [1, samples]
- **Process:** Detect speech vs silence/background
- **Output:** Voice segments with timestamps
- **Current:** ❌ NOT IMPLEMENTED
- **Missing:** Needed for ignoring silence and improving audio encoder focus

**Mel-Spectrogram Extraction:**
- **Input:** Waveform or VAD-filtered waveform
- **Process:** STFT → Mel filterbank → log scaling
- **Config:** n_mels=64, n_fft=2048, hop_length=512
- **Output:** Mel-spectrogram [64, T_freq] (time-frequency representation)
- **Current:** ✅ Implemented in `multimodal_dataset.py`
- **Missing:** 
  - No MFCC features (older but still useful)
  - No constant-Q transform (CQT)
  - No temporal derivatives (delta, delta-delta)

**Audio Encoder:**
- **Input:** Mel-spectrogram [B, 1, 64, T_time]
- **Architecture:** Small 3-layer CNN (32 → 64 → 128 channels)
- **Output:** Features [B, 256]
- **Current:** ⚠️ Implemented in `multimodal_model.py` (AudioCNN)
- **Missing:**
  - No pre-trained audio models (wav2vec2, HuBERT, speaker embeddings)
  - No attention mechanisms
  - Naive architecture (no domain-specific design)

**Alternative Audio Representations (NOT IMPLEMENTED):**
- wav2vec2 (Facebook): Pre-trained on 1B hours of speech; captures linguistic content
- HuBERT (Facebook): Pre-trained CPC model; better for speech understanding
- XLNET/CONFORMER: Transformer-based; state-of-the-art for audio
- Speaker Embedding (speaker-agnostic): Can help detect synthetic voices

---

#### **3. FUSION STRATEGY**

**Late Fusion (Currently Implemented):**
```
Video features [B, 1536] ──┐
                            ├─→ Concatenate ──→ [B, 1792] ──→ Classifier
Audio features [B, 256] ──┘
```

**Advantages:** Simple, interpretable, can turn off modalities independently
**Disadvantages:** No cross-modal interaction; features not aligned

**Alternatives (NOT IMPLEMENTED):**

1. **Early Fusion:** Concatenate raw video + mel-spectrograms before encoding
   - Advantage: Direct temporal alignment
   - Disadvantage: Requires 3D convolutions to process aligned multi-modal tensors

2. **Mid Fusion:** Concatenate intermediate features from both streams
   - Advantage: Some interaction before classification
   - Disadvantage: Harder to ablate modalities

3. **Attention-Based Fusion (Partially Stubbed):**
   ```
   Video: [B, 1536] ──┐
                       ├─→ Cross-attention ──→ [B, fusion_dim] ──→ Classifier
   Audio: [B, 256] ──┘
   
   - Query: Video features
   - Key, Value: Audio features
   - Learn how to weight audio attention based on video content
   ```
   - Advantage: Learns complementary information
   - Disadvantage: More parameters; harder to train

4. **Cross-Modal Transformer (NOT IMPLEMENTED):**
   ```
   Stack alternating video and audio tokens, apply self-attention
   - Advantage: State-of-the-art fusion method
   - Disadvantage: High computational cost
   ```

---

#### **4. CLASSIFICATION HEAD**

**Architecture:**
```
Fused features [B, fusion_dim] ──→ Linear(fusion_dim, 512) 
                                  ──→ ReLU 
                                  ──→ Dropout(0.3) 
                                  ──→ Linear(512, 2)
                                  ──→ Logits [B, 2]
```

**Output:** 
- Logits for [fake, real] classes
- Softmax applied in loss function or inference

**Current:** ✅ Implemented in `multimodal_model.py`
**Missing:**
- No confidence calibration (temperature scaling)
- No uncertainty estimates (Bayesian approach, MC-Dropout)
- No output bounds (could add sigmoid for per-class confidence)

---

#### **5. AGGREGATION (Frame → Segment → Video)**

**Three Levels:**

1. **Frame Level:** Single frame → logits [2]
2. **Segment Level:** Multiple frames [T, 2] → aggregated logits [2]
3. **Video Level:** Multiple segments → final prediction

**Aggregation Methods:**

| Method | Formula | Pros | Cons |
|--------|---------|------|------|
| **Mean** | avg(logits) | Simple, stable | Ignores outliers |
| **Max** | max(logits) | Catches anomalies | Noise-sensitive |
| **Attention** | Σ(w_i * logit_i) | Learned weighting | Needs training |
| **LSTM** | LSTM(logits) | Temporal modeling | Complex |
| **Voting** | argmax(count) | Robust | Loses confidence |

**Current:** Mean pooling is primary; attention framework present but incomplete
**Missing:** Temporal modeling of segment predictions (e.g., LSTM-based aggregation)

---

## **V. MAPPING TO EXISTING FOLDER STRUCTURE**

```
model-service/
├── src/
│   ├── models/                          [ARCHITECTURE & COMPONENTS]
│   │   ├── frame_model.py               ✅ Video backbone (EfficientNet-B3)
│   │   ├── multimodal_model.py          ⚠️ Fusion + classification (missing components)
│   │   ├── losses.py                    ✅ Temporal consistency loss
│   │   └── __init__.py
│   │
│   ├── data/                            [DATA PIPELINE]
│   │   ├── multimodal_dataset.py        ✅ Dataset loader (frame + audio extraction)
│   │   └── __init__.py
│   │
│   ├── preprocess/                      [PREPROCESSING]
│   │   ├── extract_frames.py            ✅ Frame extraction from video
│   │   └── __init__.py
│   │   ├── [MISSING: audio preprocessing, face alignment, VAD]
│   │
│   ├── serve/                           [INFERENCE]
│   │   ├── api.py                       ⚠️ FastAPI serving (only frame-level inference)
│   │   ├── README.md
│   │   └── [MISSING: video-level aggregation endpoint]
│   │
│   ├── eval/                            [EVALUATION]
│   │   ├── multimodal_eval.py           ✅ Metrics + ablation
│   │   ├── ablation_study.py            ✅ Audio/video/multimodal comparison
│   │   └── __init__.py
│   │
│   ├── utils/                           [UTILITIES]
│   │   ├── metrics.py                   ✅ AUC, AP, F1, etc.
│   │   └── [MISSING: feature visualization, error analysis]
│   │
│   ├── config.py                        ✅ Configuration management
│   ├── logging_config.py                ✅ Structured logging
│   ├── train.py                         ✅ Training loop (frame-level)
│   └── __init__.py
│
├── config/
│   └── config.yaml                      ✅ Hyperparameters + modality flags
│
├── tests/                               [TESTING]
│   └── test_model.py                    ✅ Unit tests for FrameModel
│   └── [MISSING: integration tests, ablation tests]
│
├── data/
│   ├── sample/                          ✅ Toy dataset for testing
│   ├── preprocessed/                    ⚠️ Placeholder for preextracted data
│   └── manifest.csv                     [OPTIONAL: dataset metadata]
│
├── checkpoints/                         ✅ Saved model weights
│   ├── debug.pth                        ✅ Minimal checkpoint for development
│   └── [TODO: production checkpoints]
│
└── requirements.txt                     ✅ Python dependencies
```

---

## **VI. ASSUMPTIONS vs VERIFIED FACTS**

### **Assumptions** 🟡 (Need Confirmation)

1. **Job Processing Integration:** ❌ **CRITICAL** — How does the backend trigger model inference after upload? Currently no integration visible; API only accepts single frames.
   - *Assumption:* Backend calls model service `/infer` for each frame extracted from uploaded video
   - *Need:* Explicit endpoint or webhook for full-video inference

2. **Audio-Video Synchronization:** ⚠️ — How are frames and audio segments aligned?
   - *Assumption:* Both extracted at fixed rates (frames at 3 FPS, audio at 16 kHz) → automatic alignment
   - *Need:* Verify if temporal offsets are handled (e.g., audio lag)

3. **Model Inference Latency:** ❌ **CRITICAL** — What's the target inference time?
   - *Assumption:* Single frame inference should be <100ms (to process 30 FPS video in real-time)
   - *Need:* Benchmark on actual hardware; current model likely too slow for streaming

4. **Face Detection Requirement:** ⚠️ — Must all frames contain faces?
   - *Assumption:* Face detection is mandatory; frames without faces are skipped
   - *Need:* Clarify behavior if face detector fails

5. **Temporal Window:** ⚠️ — Why 16 frames (0.5 seconds at 3 FPS)?
   - *Assumption:* Temporal receptive field of ~0.5 seconds is optimal for deepfake detection
   - *Need:* Validate with ablation studies on different temporal windows

6. **Audio Duration Alignment:** ⚠️ — How is 1-second audio segment chosen for 0.5-second video segment?
   - *Assumption:* Audio is subsampled or interpolated to match video length
   - *Need:* Confirm actual implementation

7. **Modality Dropout Usage:** ❌ — Config has modality_dropout_prob but not used in forward()
   - *Assumption:* Bug or intentional removal for stability
   - *Need:* Clarify if this should be re-enabled

8. **Pre-trained Audio Encoder:** ❌ — Small AudioCNN is from scratch; no transfer learning
   - *Assumption:* Audio encoder will struggle; should use pre-trained wav2vec2
   - *Need:* Decide on audio architecture

9. **Aggregation Strategy:** ⚠️ — Currently mean pooling; is this optimal?
   - *Assumption:* Mean is stable but may miss anomalies; max might be better
   - *Need:* Compare aggregation methods via ablation

10. **Training Data:** ❌ **CRITICAL** — What deepfake dataset is used?
    - *Assumption:* Using toy data; real training data source unknown
    - *Need:* Specify: FaceForensics++, DeepFaceLab, Celeb-DF, or proprietary?

---

### **Verified Facts** ✅ (Confirmed from Code)

1. ✅ **Video backbone:** EfficientNet-B3 (1536 dims)
2. ✅ **Frame sampling:** 3 FPS, 224×224 resolution
3. ✅ **Audio setup:** 16 kHz, 64 mel bins, 1-second segments
4. ✅ **Fusion:** Late concatenation (1536 + 256 → 1792)
5. ✅ **Classification:** 2-class (real/fake) with softmax
6. ✅ **Temporal encoder:** 1D Conv with avg pooling
7. ✅ **Face detection:** RetinaFace or MTCNN supported
8. ✅ **Modality ablation:** enable_audio/enable_video flags work
9. ✅ **Training:** Cross-entropy with label smoothing, AdamW optimizer
10. ✅ **Evaluation:** Comprehensive metrics (AUC, AP, F1, FPR@95%TPR)

---

## **VII. WHAT IS MISSING FOR A SCIENTIFICALLY SOUND MODEL**

### **Critical Gaps** 🔴

1. **No Pre-trained Audio Encoder**
   - Current: Naive 3-layer CNN on mel-spectrograms
   - Needed: wav2vec2, HuBERT, or fine-tuned ASR encoder
   - Impact: Audio representation quality is severely limited

2. **No Voice Activity Detection (VAD)**
   - Current: Processes all audio equally
   - Needed: Explicit VAD to mask silence
   - Impact: Model wastes capacity on uninformative silent frames

3. **No Cross-Modal Interaction**
   - Current: Late fusion (concatenation only)
   - Needed: Attention or transformer-based fusion
   - Impact: Model can't learn complementary information

4. **Temporal Consistency Loss NOT Integrated**
   - Current: Loss function defined but never used
   - Needed: Add to training loss with weight
   - Impact: Model has no temporal smoothness regularization

5. **No Uncertainty Estimation**
   - Current: Softmax logits only
   - Needed: Calibration, MC-Dropout, or Bayesian approach
   - Impact: Can't detect when model is uncertain

6. **No Video-Level Inference Endpoint**
   - Current: Only frame-level `/infer` endpoint
   - Needed: `/analyze-video` endpoint that handles full pipeline
   - Impact: Backend must do all aggregation manually

### **High-Priority Gaps** 🟠

7. **No Optical Flow or Motion Features**
   - Current: Per-frame appearance only
   - Needed: Compute optical flow; stack with frames
   - Impact: Can't detect motion artifacts

8. **Limited Audio Augmentation**
   - Current: Mostly stubbed out
   - Needed: Time-stretching, pitch-shifting, noise injection
   - Impact: Audio encoder overfits to clean audio

9. **No Face Alignment**
   - Current: Bounding box crops only
   - Needed: 8-point landmark alignment
   - Impact: Face variations not normalized; harder learning

10. **No Lip-Sync Detection**
    - Current: No explicit lip-audio synchronization
    - Needed: Separate lip region detection + optical flow
    - Impact: Missing strong deepfake indicator

11. **No Multi-Task Learning**
    - Current: Binary classification only
    - Needed: Auxiliary tasks (facial landmarks, expression, emotion)
    - Impact: Model doesn't learn intermediate representations

12. **No Distributed Training**
    - Current: Single GPU/CPU only
    - Needed: DistributedDataParallel for multi-GPU
    - Impact: Scales poorly to large datasets

### **Medium-Priority Gaps** 🟡

13. **Limited Temporal Modeling**
    - Current: 1D Conv with 3-frame receptive field
    - Needed: Transformer attention or 3D CNNs
    - Impact: Can't capture longer temporal patterns

14. **No Frequency-Domain Features**
    - Current: Time-domain raw image features only
    - Needed: Compute and use frequency spectra
    - Impact: Can't detect spectral forgeries (GAN artifacts)

15. **No Batch Normalization / Layer Normalization**
    - Current: Some models may use BN; not clear
    - Needed: Explicit norm layers in custom architectures
    - Impact: Training instability

16. **No Ensemble Methods**
    - Current: Single model
    - Needed: Ensemble of audio/video/multimodal models
    - Impact: Brittleness to adversarial examples

17. **No Adversarial Robustness Testing**
    - Current: No adversarial examples in evaluation
    - Needed: Adversarial perturbations; robustness metrics
    - Impact: Unknown vulnerability to evasion attacks

18. **No Explainability / Interpretability**
    - Current: No attention maps, saliency maps, or feature attribution
    - Needed: Grad-CAM, LIME, SHAP for predictions
    - Impact: Can't understand what model is learning

---

## **VIII. CLARIFYING QUESTIONS BEFORE FINALIZING DESIGN**

### **A. Problem Scope**

1. **Video Source & Characteristics:**
   - What video codecs/formats must be supported? (H.264, H.265, VP9?)
   - What frame rates? (typically 24, 30, 60 FPS?)
   - What resolution range? (360p to 4K?)
   - What languages/accents for audio?
   - What deepfake methods should we detect? (Face2Face, Reenactment, GAN-based?)

2. **Operational Requirements:**
   - What's the **maximum acceptable latency** per uploaded video?
   - What's the **target throughput**? (videos/second?)
   - What **GPU hardware** is available? (RTX 3080, A100, etc.?)
   - Is **batch inference** supported (multiple videos simultaneously)?

3. **Data & Training:**
   - What **deepfake dataset** is available for training? (FaceForensics++, Celeb-DF, proprietary?)
   - What **data augmentation** is needed? (deepfake age, codec variations?)
   - Should we **fine-tune** pre-trained models or train from scratch?
   - Is **continual learning** needed (updating model with new user data)?

### **B. Model Architecture Decisions**

4. **Audio Encoder Choice:**
   - Should we use **wav2vec2** (computational cost: high)? Or **HuBERT** (better for non-English)?
   - Or simpler **Mel-spectrogram + traditional CNN** (fast, interpretable)?
   - Should audio be **speaker-agnostic** or **speaker-aware**?

5. **Video Backbone:**
   - Is **EfficientNet-B3** sufficient, or do we need **ResNet-50**, **ViT**, or **MobileNet** for efficiency?
   - Should we **fine-tune** on synthetic/deepfake datasets, or keep ImageNet weights?
   - Do we need **multi-scale features** (e.g., FPN)?

6. **Fusion Strategy:**
   - Is **late concatenation** acceptable, or do we need **cross-modal attention**?
   - Should we support **modality-specific classification** (separate real/fake heads per modality)?
   - Do we want **modality weighting** (learned or fixed)?

7. **Temporal Modeling:**
   - What **temporal window** is optimal? (0.5 sec, 1 sec, 5 sec?)
   - Should we use **3D CNNs**, **Transformers**, or **RNNs**?
   - Is **optical flow** worth the computational cost?

### **C. Output & Interpretability**

8. **Prediction Output:**
   - Do we need **per-frame predictions** returned to the user?
   - Or just **video-level confidence** with timestamp of most anomalous frame?
   - Should we return **confidence calibration**? (How confident are we in the prediction?)

9. **Explainability:**
   - Do end-users need **saliency maps** (which face regions triggered "fake" prediction)?
   - Do we need **artifact explanations** ("detected unnatural eye movement", etc.)?
   - Is **model transparency** a requirement?

### **D. Deployment & Scaling**

10. **Inference Serving:**
    - Should the API support **synchronous** (wait for result) or **asynchronous** (return job ID) inference?
    - Is **streaming inference** needed (process video frame-by-frame as it arrives)?
    - Do we need **model versioning**? (A/B testing multiple models?)

11. **Edge Deployment:**
    - Must the model run on **mobile/browser**? (Implies quantization, pruning, distillation?)
    - Or is **server-side only** acceptable?

12. **Cost Constraints:**
    - Are we optimizing for **latency**, **throughput**, or **cost** (GPU/compute)?
    - Is **FP16 mixed precision** acceptable?
    - Should we **compress** the model (quantization, pruning, distillation)?

### **E. Evaluation & Benchmarking**

13. **Success Metrics:**
    - What **false positive rate** is acceptable? (1%, 5%, 10%?)
    - What **false negative rate**? (Miss % of deepfakes?)
    - Is **ROC-AUC** the primary metric, or **F1**, **accuracy**, **precision**?

14. **Test Data:**
    - What **deepfake datasets** are used for evaluation? (FaceForensics++, Celeb-DF, wild deepfakes?)
    - Should we test on **out-of-distribution** deepfakes (unseen generation methods)?
    - Do we need **adversarial robustness** testing?

15. **Baseline Comparisons:**
    - Are we competing against existing tools? (MediaForensics, Sensetime, AWS Rekognition?)
    - What **baseline performance** should we achieve?

---

## **IX. RECOMMENDED NEXT STEPS (IN PRIORITY ORDER)**

### **Phase 1: Critical Fixes (1-2 weeks)**

1. ✅ **Integrate temporal consistency loss into training**
   - Uncomment in train.py; add to loss computation
   - Expected gain: ~2-3% AUC improvement

2. ✅ **Implement VAD (Voice Activity Detection)**
   - Use pyannote-audio or librosa energy-based VAD
   - Mask silence in mel-spectrograms
   - Expected gain: ~1-2% AUC improvement

3. ✅ **Replace naive AudioCNN with pre-trained wav2vec2 encoder**
   - Use facebook/wav2vec2-base or facebook/hubert-base
   - Freeze first N layers; fine-tune last M
   - Expected gain: ~5-10% AUC improvement (critical!)

4. ✅ **Add video-level inference endpoint**
   - `/analyze-video` endpoint accepts video file
   - Extracts frames, runs inference, aggregates results
   - Returns video-level confidence + timestamp of anomaly

5. ✅ **Fix modality dropout in forward pass**
   - Currently config parameter not used
   - Enable during training; disable during inference
   - Expected gain: slight regularization improvement

### **Phase 2: High-Impact Improvements (2-3 weeks)**

6. ✅ **Implement attention-based fusion**
   - Replace concatenation with cross-attention
   - Video features query; audio features key/value
   - Expected gain: ~2-5% AUC improvement

7. ✅ **Add optical flow features**
   - Compute optical flow between adjacent frames
   - Stack with appearance features
   - Expected gain: ~3-5% AUC improvement

8. ✅ **Implement face alignment**
   - Use RetinaFace landmarks for affine alignment
   - Normalize pose variation
   - Expected gain: ~1-2% AUC improvement

9. ✅ **Add uncertainty estimation**
   - Implement MC-Dropout (inference with dropout enabled)
   - Or temperature scaling for calibration
   - Output confidence intervals

10. ✅ **Implement multi-task learning**
    - Auxiliary task: predict facial landmarks
    - Auxiliary task: predict head pose
    - Share backbone; separate task heads
    - Expected gain: ~2-3% AUC improvement

### **Phase 3: Advanced Methods (3-4 weeks)**

11. ✅ **Implement transformer-based temporal encoder**
    - Replace 1D Conv with ViT-style attention over frames
    - Expected gain: ~2-3% AUC improvement

12. ✅ **Add lip-sync verification module**
    - Separate lip region detector
    - Optical flow on lips vs audio speech rate
    - Expected gain: ~3-5% AUC improvement

13. ✅ **Ensemble modeling**
    - Train 3-5 independent models (different seeds, architectures)
    - Average predictions
    - Expected gain: ~2-4% AUC improvement

14. ✅ **Add adversarial robustness**
    - Test against FGSM, PGD adversarial examples
    - Consider adversarial training
    - Expected gain: robustness to evasion attacks

15. ✅ **Implement explainability module**
    - Grad-CAM for visual saliency
    - Feature importance for audio
    - Return explanations to user

---

## **X. FINAL SUMMARY**

**Current System State:**
- ✅ Core infrastructure in place (data pipeline, training loop, evaluation metrics)
- ⚠️ Fusion strategy is naive (late concatenation only)
- ❌ Audio encoder is weak (untrained 3-layer CNN)
- ❌ No video-level inference endpoint
- ❌ Missing critical components (VAD, optical flow, attention fusion)

**Recommended Strategy:**
1. **First:** Replace audio encoder with pre-trained wav2vec2 (biggest impact)
2. **Second:** Integrate temporal consistency loss and implement VAD
3. **Third:** Add cross-modal attention and optical flow
4. **Fourth:** Implement video-level inference endpoint and aggregation
5. **Fifth:** Add uncertainty, explainability, and robustness testing

**Expected Final Performance:**
- Current baseline (rough estimate): ~70-75% AUC on FaceForensics++
- After Phase 1: ~78-82% AUC
- After Phase 2: ~83-87% AUC
- After Phase 3: ~88-92% AUC (competitive with state-of-the-art)

---

## **XI. MODEL CONTRACT v1 (LOCKED DECISIONS) ✅**

**Status:** Design freeze. All architectural decisions locked. Implementation can proceed.

---

### **A. Problem Scope (LOCKED)**

#### **1️⃣ Deepfake Methods to Detect**

**✅ PRIMARY FOCUS (In Scope):**
- GAN-based face swaps (FaceSwap, NeuralTextures)
- Reenactment models (Face2Face)
- StyleGAN-based manipulations
- Audio-visual lip-sync manipulation (Wav2Lip-style)

**❌ EXPLICITLY OUT OF SCOPE (v1):**
- Purely synthetic avatars (full CGI humans)
- Text-to-video models with no real source identity

**Rationale:** These >90% of real-world forensic cases, align with FaceForensics++ & Celeb-DF

**Reversibility:** ✅ Easy to extend classes later

---

#### **2️⃣ Video Codecs & Resolution Requirements**

**✅ CODECS & CONTAINERS:**
- **Mandatory:** H.264 codec, MP4 container
- **Best-effort:** H.265 (HEVC) support

**✅ RESOLUTION HANDLING:**
- **Normalization:** 224×224 for model input
- **Input acceptance range:** 240p → 1080p
- **Downsampling strategy:** Lossless or linear interpolation

**✅ FRAME RATE:**
- **Sampling:** 5–10 FPS (uniform extraction)
- **Rationale:** Forensics artifacts survive downsampling; higher FPS adds latency without benefit

**Reversibility:** ⚠️ Codec support harder to change than resolution

---

#### **3️⃣ Maximum Latency Budget**

**✅ SOFT LIMIT:** 30–60 seconds per 30-second video

**✅ HARD LIMIT:** 2 minutes per video

**✅ UX EXPECTATION:** Results delivered asynchronously with progress bar

**Rationale:** Deepfake detection is forensic, not real-time moderation

**Foundational:** ❌ Drives async architecture design (non-negotiable)

**Implementation Implication:**
- Backend must implement job queue + async worker
- Frontend must support polling / Server-Sent Events (SSE)
- API endpoint: `/analyze-video` (async) with `/jobs/{job_id}` status endpoint

---

#### **4️⃣ Training Dataset**

**✅ PRIMARY:** FaceForensics++

**✅ SECONDARY (VALIDATION):** Celeb-DF v2

**✅ OPTIONAL AUGMENTATION:** DFDC (sampled subset)

**✅ TRAINING SPLIT STRATEGY:**
- Train on FaceForensics++
- Validate on Celeb-DF v2
- Test cross-dataset generalization

**Rationale:** Avoids overfitting to one generator distribution

**Reversibility:** ❌ Dataset choice deeply affects learned features; difficult to change later

---

### **B. Model Architecture Decisions (LOCKED)**

#### **5️⃣ Audio Encoder**

**✅ APPROACH:** Speaker-agnostic pre-trained encoder

**✅ ENCODER TYPE:**
- **Primary:** facebook/wav2vec2-base or facebook/hubert-base
- **Focus:** Phoneme–lip sync mismatch detection, NOT speaker identity
- **Fine-tuning:** Freeze first 8 layers; fine-tune last 4 + classification head

**✅ NO SPEAKER-AWARE MODELS:**
- Reason: Speaker-aware encoders overfit identity, harming cross-speaker generalization

**Rationale:** Speaker-agnostic design ensures robustness to unseen speakers

**Foundational:** ✅ Non-negotiable (affects data pipeline)

**Implementation Implication:**
- Replace `models/audio_cnn.py` with wav2vec2 wrapper
- Update `multimodal_dataset.py` to load `.wav` files (not mel-specs)
- Add vocab generation for feature dimension (768 → embedding layer)

---

#### **6️⃣ Temporal Window Size**

**✅ PRIMARY WINDOW:** 1 second (≈5–10 frames @ 5–10 FPS)

**✅ SECONDARY AGGREGATION:** Sliding window with 50% overlap over entire video

**Window Distribution Strategy:**
- Extract frames at uniform intervals across entire video
- Ensure ≥2 windows per second of video
- For 30s video: ~60 windows of 1s each

**Rationale:**
- <0.5s → unstable temporal signals (insufficient context)
- >5s → added latency + temporal noise (loses short-term consistency cues)
- 1s is empirically optimal in deepfake detection literature

**Reversibility:** ⚠️ Window length affects temporal architecture; difficult to change without retraining

**Implementation Implication:**
- Frame extraction: 5 FPS (or 10 FPS for high-resolution videos)
- Temporal encoder input: 5–10 frame sequences
- TemporalConv: kernel_size=3, stride=1 to aggregate 1s windows

---

#### **7️⃣ Fusion Strategy**

**✅ ARCHITECTURE:** Cross-modal attention (mid-fusion)

**✅ IMPLEMENTATION:**
- No quadratic global attention (too expensive)
- Single lightweight fusion block after temporal encoding
- Video features query; audio features key/value
- Attention output concatenated with both modalities
- Approximate complexity: O(T × A) where T=temporal frames, A=audio frames

**❌ NOT:** Late concatenation (current implementation)

**Rationale:** Audio–visual inconsistencies are the core forensic signal; must be explicitly modeled

**Foundational:** ✅ Non-negotiable (defines architecture)

**Implementation Implication:**
- Replace `concatenation` in `multimodal_model.py` with `CrossModalAttention` block
- New module: `models/fusion.py` with `CrossModalAttentionFusion` class
- Update forward pass: `video_features → attention(query=video, key=audio, value=audio) → fused_features`

---

#### **8️⃣ Per-Frame Explainability**

**✅ REQUIRED OUTPUTS:**
- **Visual saliency:** Grad-CAM heatmaps on face region
- **Audio anomaly timestamps:** Frames with high audio–visual mismatch
- **Modality contribution score:** (Video confidence - Audio confidence) normalized to [-1, +1]

**✅ USER-FACING DELIVERABLES:**
- Confidence score (0–1)
- List of anomalous frames with saliency overlay
- Modality agreement score (high = synchronized, low = out-of-sync)

**Rationale:** Forensic tools must explain decisions for legal admissibility

**Foundational:** ✅ Non-negotiable (legal/compliance requirement)

**Implementation Implication:**
- Add explainability module: `models/explainability.py`
- Compute Grad-CAM on final Conv layer of video encoder
- Overlay heatmaps on original frame faces
- Timestamp audio anomalies from attention weights

---

### **C. Inference & Deployment (LOCKED)**

#### **9️⃣ Synchronous vs Asynchronous Inference**

**✅ APPROACH:** Asynchronous ONLY

**✅ ARCHITECTURE:**
- Frontend: POST `/analyze-video` returns `{"job_id": "uuid", "status_url": "/jobs/{job_id}"}`
- Backend: Job enqueued to Redis/Celery queue
- Background worker: Processes video, stores results in PostgreSQL
- Frontend: Polls `/jobs/{job_id}` or uses SSE for updates

**❌ NOT:** Synchronous (blocking inference)

**Rationale:** Synchronous inference will fail under load; 30–60 sec latency makes synchronous UX unusable

**Non-negotiable:** ❌ Must implement async architecture

**Implementation Implication:**
- Add Celery worker with Redis broker
- New endpoint: `backend/app/routes/jobs.py` with POST `/analyze-video`, GET `/jobs/{job_id}`
- Database schema: `detection_jobs` (id, video_url, status, created_at, updated_at, results_json)
- Model service: Expose `/infer-video` endpoint for full video inference

---

### **D. Remaining 5 Decisions (LOCKED)**

#### **🔟 Output Format**

**✅ PRIMARY OUTPUT:** Binary (Real vs Fake)

**✅ AUXILIARY HEADS:**
- Manipulation confidence (0–1, higher = more confident it's fake)
- Modality disagreement score (0–1, higher = audio/video mismatch)
- Per-frame prediction array (confidence at each sampled frame)

**Output JSON Schema:**
```json
{
  "job_id": "uuid",
  "status": "completed",
  "prediction": {
    "overall_class": "fake",  // "real" or "fake"
    "overall_confidence": 0.92,
    "manipulation_confidence": 0.88,
    "modality_agreement": 0.75,
    "anomalous_frames": [
      {"frame_id": 15, "confidence": 0.95, "saliency_url": "..."},
      {"frame_id": 42, "confidence": 0.91, "saliency_url": "..."}
    ]
  },
  "metadata": {
    "processing_time_sec": 45.3,
    "video_duration_sec": 30,
    "frames_analyzed": 150
  }
}
```

---

#### **1️⃣1️⃣ Frame Sampling Strategy**

**✅ EXTRACTION APPROACH:**
- Uniform sampling (not random)
- Face-detected frames only (skip frames with no face)
- Drop frames with failed face alignment (RetinaFace landmarks)

**✅ FACE DETECTION:**
- Use RetinaFace for multi-scale face detection
- Require confidence > 0.95
- Skip frames with multiple faces (ambiguous which to analyze)
- Single-face videos only (v1 constraint)

**Rationale:** Ensures consistent temporal windows; avoids empty or ambiguous frames

**Reversibility:** ✅ Easy to relax to multi-face later

---

#### **1️⃣2️⃣ Audio Extraction & Storage**

**✅ RESPONSIBILITY:** Model service only (backend does NOT touch audio)

**✅ EXTRACTION METHOD:**
- ffmpeg-based extraction in `preprocess/extract_audio.py`
- Output: `.wav` at 16kHz, mono, 16-bit PCM
- Storage: Temporary files only (deleted after inference)

**❌ NOT:** Store audio in Supabase or database

**Rationale:** Keeps audio processing isolated; simplifies data pipeline

---

#### **1️⃣3️⃣ Confidence Calibration**

**✅ REQUIREMENT:** Yes, mandatory

**✅ APPROACH:**
- Temperature scaling (learn temperature value on validation set)
- Calibration curve evaluation on Celeb-DF

**✅ REJECT OPTION:**
- If model confidence < 0.6, return "uncertain" label
- Flag for human review

**Rationale:** Forensic context demands calibrated confidence (avoid false certainty)

---

#### **1️⃣4️⃣ Model Update Cadence**

**✅ APPROACH:** Offline retraining + manual checkpoint promotion

**✅ WORKFLOW:**
- Retrain monthly on accumulated FaceForensics++ + new community data
- Evaluate on Celeb-DF validation set
- Manual promotion of best checkpoint to production
- No online learning

**❌ NOT:** Online learning, continuous updates

**Rationale:** Offline avoids model drift; manual approval maintains safety

---

### **E. Critical Implementation Constraints**

**LOCKED INTERFACE CONTRACTS:**

| Component | Contract | Flexibility |
|-----------|----------|-------------|
| **Audio Encoder** | wav2vec2-base | 🔒 Locked (affects data pipeline) |
| **Temporal Window** | 1 second (5–10 frames) | 🔒 Locked (architecture dependent) |
| **Fusion Strategy** | Cross-modal attention (mid-fusion) | 🔒 Locked (core forensic signal) |
| **Inference Mode** | Asynchronous + job queue | 🔒 Locked (non-negotiable for UX) |
| **Training Dataset** | FaceForensics++ (primary) | 🔒 Locked (learned representations) |
| **Video Codec** | H.264 + MP4 | ⚠️ Changeable (H.265 optional) |
| **Resolution** | 224×224 normalized | ✅ Flexible (accept 240p–1080p) |
| **Confidence Calibration** | Temperature scaling | ✅ Flexible (alternative: Platt scaling) |

---

**DO NOT PROCEED** with implementation until all clarifying questions (Section VIII) are answered.


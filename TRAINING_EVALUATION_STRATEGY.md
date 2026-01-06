# **TRAINING & EVALUATION STRATEGY — MULTIMODAL DEEPFAKE DETECTION v1.0**

**Purpose:** Production-grade training and evaluation protocol  
**Status:** PLAN MODE (specification only, no implementation code)  
**Date:** January 3, 2026  
**Built On:** [ML_SYSTEM_DESIGN.md § XI](ML_SYSTEM_DESIGN.md#xi-model-contract-v1-locked-decisions-) + [MODEL_CONTRACT_v1.md](MODEL_CONTRACT_v1.md)

---

## **I. DATASET STRATEGY**

### **A. Primary Dataset: FaceForensics++**

**Rationale:** Largest, most balanced deepfake forensics dataset; enables cross-dataset evaluation

**Composition:**
```
FaceForensics++ v4 (331 videos)
├── Real videos (Pristine): 150 videos
│   └── Extraction: All frames @ 5 FPS
│   └── Face detection: Single-face only
│   └── Temporal window: 1 sec (5 frames @ 5 FPS)
│   └── Approximate frame count: 150 × 30s × 5 FPS = 22,500 sequences
│
├── Fake videos by generation method:
│   ├── DeepFaceLab (74 videos) — GAN-based face swap
│   ├── Face2Face (71 videos) — Reenactment via facial expression transfer
│   ├── FaceSwap (73 videos) — GAN-based swap variant
│   └── NeuralTextures (13 videos) — Neural rendering
│
└── Split strategy:
    ├── Train: 80% (265 videos)
    ├── Validation: 10% (33 videos)
    └── Test: 10% (33 videos)
    
    Constraint: Stratify by generation method (maintain proportion in each split)
```

**Compression & Codec Diversity:**
- FaceForensics++ compressed to c0, c23, c40 (quality degradation levels)
- **Recommendation for v1:** Use only c23 (8–12 MB/video, realistic quality)
- Codec: H.264 (MP4 container)
- Resolution: Rescale to 224×224 during training

**Audio Acquisition:**
- FaceForensics++ includes original audio (from YouTube sources)
- Audio format: 16kHz, mono, 16-bit PCM
- VAD applied: Speech-only regions extracted

---

### **B. Secondary Dataset: Celeb-DF v2**

**Rationale:** Cross-dataset validation; tests generalization to unseen deepfake distribution

**Composition:**
```
Celeb-DF v2 (590 videos)
├── Real videos (Celebrity video clips): 590 videos
│   └── Single speaker, controlled environment
│
├── Fake videos (Deepfake generation): 590 videos
│   └── Generated via: Wav2Lip + refinement
│   └── Temporal characteristics: Lip-sync artifacts
│   └── Audio: Synthetic speech or lip-synced
│
└── Usage:
    ├── Validation (ongoing): Monitor cross-dataset generalization
    └── Test: Final evaluation (hold out, report AUC separately)
    
    NO training data from Celeb-DF (strict evaluation set)
```

**Why Celeb-DF Matters:**
- ✅ Different generation method (Wav2Lip) → tests robustness
- ✅ Different compression (original YouTube quality)
- ✅ Different speaker distribution (celebrities)
- ❌ Audio-heavy artifacts (lip-sync) → tests audio modality

---

### **C. Augmentation Dataset: DFDC (Optional, Phase 3)**

**Rationale:** Additional deepfake diversity (modern GAN variants, deepfakes in the wild)

**Composition:**
```
DFDC - Deepfake Detection Challenge dataset (subset)
├── 5,000+ videos (use 1,000 samples for Phase 3)
├── Multiple generation methods: Autoencoder, GANpix2pixHD, StyleGAN
├── Real-world compression & artifacts
└── Usage: Phase 3 only (after Phase 2 baseline established)
```

---

### **D. Class Imbalance Handling**

**Baseline Imbalance:**
- FaceForensics++: ~1:1 real/fake ratio (balanced)
- Celeb-DF: ~1:1 real/fake ratio (balanced)
- **Net effect:** No class imbalance in training

**Potential Imbalance Scenarios (if expanding data):**
1. **If adding in-the-wild deepfakes (Phase 3+):**
   - Problem: Deepfakes in-the-wild << authentic videos
   - Solution: Class weighting in loss function
     - `weight_fake = N_real / N_fake` (e.g., 5.0 if 5× more real)
     - Apply during cross-entropy: `loss = -weight_fake × log(P_fake) - log(P_real)`

2. **If handling detection at different compression levels:**
   - Problem: High-quality deepfakes harder to detect
   - Solution: Hard example mining (oversample misclassified high-quality fakes)

---

### **E. Real vs Fake Diversity Matrix**

**Real Diversity (Quality Factors):**

| Factor | Strategy |
|--------|----------|
| **Compression** | c23 (8–12 MB) + test on c0, c40 for robustness |
| **Resolution** | Normalize to 224×224; test 240p–1080p inputs |
| **Lighting** | Use YouTube videos (indoor, outdoor, controlled, uncontrolled) |
| **Speaker demographics** | Diverse age, gender, ethnicity (FaceForensics++ globally sourced) |
| **Codec** | H.264 (MP4); optional: test H.265 in Phase 3 |
| **Audio characteristics** | Various languages, accents, speech rates |

**Fake Diversity (Generator Coverage):**

| Generator | FaceForensics++ | Celeb-DF | Coverage |
|-----------|-----------------|---------|----------|
| **DeepFaceLab (GAN)** | ✅ 74 videos | ❌ | Core GAN swap |
| **Face2Face (Reenactment)** | ✅ 71 videos | ❌ | Expression/pose transfer |
| **FaceSwap (GAN)** | ✅ 73 videos | ❌ | Classic face swap variant |
| **NeuralTextures (Mesh)** | ✅ 13 videos | ❌ | Neural rendering |
| **Wav2Lip (Lip-sync)** | ❌ | ✅ 590 videos | Audio-driven lip-sync |
| **StyleGAN variants** | ❌ | ✅ (subset) | Modern generative models |

**Coverage Assessment:**
- ✅ V1 covers >90% of real-world deepfake types
- ⚠️ Limited exposure to StyleGAN (Phase 3 augmentation)
- ⚠️ No 3D morphable model fakes (explicitly out of scope)

---

## **II. LABEL GRANULARITY**

### **A. Video-Level Labels (Primary)**

**Definition:** Single binary label per video (Real=0, Fake=1)

**Source:** FaceForensics++, Celeb-DF metadata
- Unambiguous (each video either real or fake)
- Used for final model evaluation
- Enables video-level AUC computation

**Properties:**
- ✅ Unambiguous
- ✅ Agrees with task definition ("Is this video deepfake?")
- ❌ Too coarse for frame-level inconsistencies
- ❌ Doesn't capture temporal artifacts

---

### **B. Frame-Level Pseudo-Labels (Secondary)**

**Definition:** Synthetic labels derived from video-level labels

**Generation Strategy:**
```
Video-level label: Y_video ∈ {0, 1}

Option 1: Hard assignment
├── If Y_video = 1 (fake):
│   └── All frames: y_frame = 1 (assume all frames fake)
├── If Y_video = 0 (real):
│   └── All frames: y_frame = 0 (assume all frames real)
└── Problem: Deepfakes may have natural frames!

Option 2: Soft labels (RECOMMENDED)
├── If Y_video = 1 (fake):
│   └── y_frame = 0.8 (some frames may be real, encode uncertainty)
├── If Y_video = 0 (real):
│   └── y_frame = 0.05 (rare artifacts, but not pure real)
└── Benefit: Allows model to learn partial frame contamination

Option 3: Confidence-weighted labels
├── Use model confidence to assign frame labels
├── High-confidence frames: weight = 1.0
├── Low-confidence frames: weight = 0.5
└── Problem: Circular (model confidence → labels → training)
```

**Recommendation for v1:** Hard assignment (simplicity) with soft label regularization

---

### **C. Segment-Level Supervision**

**Definition:** Temporal segments (1-second windows) with explicit labels

**Strategy:**
```
Training data structure:
├── Video: fake_video_001.mp4 (30 seconds)
├── Extract segments: 30 × 5 FPS = 150 frames
├── Group into 1-sec windows: 30 windows (5 frames each)
└── Label each window: y_segment ∈ {0, 1}

Generation:
├── For real videos: all segments → 0
├── For fake videos: all segments → 1 (assume homogeneous generation)

Benefit: Enables segment-level loss computation
├── TemporalConsistencyLoss: variance of embeddings within segment
├── SegmentClassificationLoss: cross-entropy on segment predictions
```

**Note:** Segment and frame labels are **derived**, not manual annotations

---

### **D. Weak vs Strong Labels**

**Strong Labels (Video-Level):**
- ✅ Reliable (human-verified metadata from FaceForensics++)
- ✅ Direct supervision for binary classification
- ✅ Used for AUC computation
- ❌ Noisy at frame level (not all frames equally fake)

**Weak Labels (Pseudo-Labels):**
- ✅ Abundant (one label per frame = 22,500 frame labels from 265 videos)
- ✅ Enable fine-grained temporal learning
- ❌ Noisy (derived, not ground truth)
- ❌ Label noise can hurt training (especially with hard labels)

**Hybrid Strategy (Recommended):**
```
Training objective:
├── Segment-level: Cross-entropy(soft labels) [main loss]
├── Video-level: Cross-entropy(video labels) [aggregate loss]
└── Temporal: TemporalConsistencyLoss [regularization]

Loss = λ_seg × L_seg + λ_video × L_video + λ_temp × L_temp
     = 0.7 × L_seg + 0.2 × L_video + 0.1 × L_temp
```

---

## **III. TRAINING PIPELINE**

### **A. Pretraining Strategy (Per-Modality)**

#### **1. Video Backbone Pretraining**

**Current State:** EfficientNet-B3 pretrained on ImageNet

**Rationale:**
- ImageNet weights provide general visual features (edges, textures, shapes)
- Transfer learning effective for video classification (standard practice)
- Alternative: Train from scratch (slower, needs more data)

**Pretraining Protocol:**
```
Frozen backbone (v1) → Finetuned (Phase 2)

v1 Strategy: Freeze EfficientNet-B3 backbone
├── Rationale: Limited training data (22,500 frame sequences)
├── Risk: ImageNet features may not capture deepfake artifacts
├── Mitigation: Add temporal modeling on top (1D Conv + attention)

Phase 2 Strategy: Unfreeze late layers
├── After initial convergence on frozen model
├── Unfreeze layers 8+ (later layers) of EfficientNet
├── Fine-tune with low learning rate (1e-5)
└── Expected gain: +2–3% AUC
```

#### **2. Audio Encoder Pretraining**

**Current State:** wav2vec2-base pretrained on 960h English speech (LibriSpeech)

**Rationale:**
- wav2vec2 learns phoneme-aware features (self-supervised learning)
- Captures acoustic patterns relevant to speech
- Generalizes well across speakers & languages (with fine-tuning)

**Pretraining Protocol:**
```
v1 Strategy: Freeze first 8 layers, fine-tune last 4
├── First 8 layers: General phoneme/acoustic features
├── Last 4 layers: Speaker/context-specific features
└── Fine-tune with low learning rate (1e-4)

Rationale:
├── Phoneme features transfer well across speakers (desired)
├── Speaker features may overfit (undesired for generalization)
├── Freezing early layers prevents forgetting general patterns
```

---

### **B. Joint vs Staged Training**

#### **Option 1: Staged Training (RECOMMENDED for v1)**

```
Stage 1: Video backbone + temporal encoder
├── Duration: 5 epochs
├── Input: Video frames only (no audio)
├── Loss: Binary cross-entropy on video logits
├── Objective: Learn video-level fake detection
└── Output: Video feature extractor + temporal encoder

↓ (Transfer video representations)

Stage 2: Audio encoder
├── Duration: 5 epochs
├── Input: Audio waveforms only (no video)
├── Loss: Binary cross-entropy on audio logits
├── Objective: Learn audio-level lip-sync detection
└── Output: Audio feature extractor

↓ (Combine trained modalities)

Stage 3: Fusion + joint training
├── Duration: 10 epochs
├── Input: Video + audio (both modalities)
├── Loss: Weighted combination of all losses
├── Modality freezing: Video 80%, Audio 80%, Fusion 100%
├── Rationale: Fine-tune fusion while preserving modality knowledge
└── Output: Final multimodal model
```

**Advantages:**
- ✅ Modular debugging (isolate video vs audio issues)
- ✅ Lower memory during early stages
- ✅ Clear convergence checkpoints
- ✅ Each modality optimized before fusion

**Disadvantages:**
- ❌ More training steps (3 × longer wall-clock time)
- ❌ Fusion training doesn't benefit from modality co-adaptation

#### **Option 2: End-to-End Joint Training**

```
Single stage: All modalities + fusion
├── Duration: 15 epochs
├── Loss: All losses simultaneously from epoch 1
├── Freezing schedule: Gradual unfreezing over epochs
└── Expected convergence: Slower initial, faster final
```

**Advantages:**
- ✅ Co-adaptation of modalities (may learn better fusion)
- ✅ Shorter overall training time
- ✅ Simpler to implement

**Disadvantages:**
- ❌ Harder to debug (multiple sources of loss change)
- ❌ Risk of one modality dominating (audio overfitting if more stable)

---

### **C. Freezing vs Finetuning Schedule**

**Recommended v1 Schedule:**

```
Epoch 1–5 (Stage 1: Video pretraining):
├── Video backbone: FROZEN (ImageNet weights)
├── Temporal encoder: TRAINABLE (learn temporal patterns)
├── Audio encoder: N/A
├── Fusion: N/A
└── Learning rate: 1e-3

Epoch 6–10 (Stage 2: Audio pretraining):
├── Video backbone: FROZEN
├── Temporal encoder: FROZEN (from Stage 1)
├── Audio encoder: TRAINABLE (fine-tune wav2vec2 last 4 layers)
├── Fusion: N/A
└── Learning rate: 1e-4 (lower, avoid catastrophic forgetting)

Epoch 11–20 (Stage 3: Joint fine-tuning):
├── Video backbone: FROZEN (keep ImageNet knowledge)
├── Temporal encoder: TRAINABLE (adapt to joint task)
├── Audio encoder: TRAINABLE (adapt to joint task)
├── Fusion: TRAINABLE (learn cross-modal attention)
└── Learning rate: 1e-4 → 1e-5 (cosine annealing)

Checkpoint strategy:
├── Save after each stage (3 checkpoints)
├── Select best on validation AUC
└── Rollback if overfitting detected (val AUC plateau >2 epochs)
```

**Phase 2: Unfreezing**

After v1 baseline, selectively unfreeze:
```
Unfreeze video backbone layers 8–11 (late residual blocks):
├── Rationale: Late layers capture deepfake-specific artifacts
├── Learning rate: 1e-5 (careful, avoid catastrophic forgetting)
├── Expected gain: +2–3% AUC

Trigger: Only if validation AUC plateau detected on frozen model
```

---

### **D. Augmentations (Video + Audio)**

#### **Video Augmentations (Applied per Frame)**

```
Baseline augmentations (v1):
├── ColorJitter: brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
├── RandomCrop: 224×224 crop from 256×256 (10px border)
├── RandomHorizontalFlip: p=0.5
├── GaussianBlur: kernel_size=3, sigma=[0.1, 2.0]
└── Applied at training time only

Moderate augmentations (Phase 2):
├── RandomRotate: ±10 degrees
├── RandomPerspective: distortion scale 0.2
├── ElasticTransform: α=34, σ=5.5 (simulate compression artifacts)
└── Expected benefit: +1–2% robustness to codec variations

Advanced augmentations (Phase 3):
├── CutMix: Mix 16×16 patches across batch (regularization)
├── Mixup: Linear interpolation of images & labels
├── MoCo augmentation: Contrastive learning (self-supervised pretraining)
└── Expected benefit: +2–3% AUC through regularization

Critical: NO augmentations that destroy deepfake cues
├── ❌ Avoid extreme blur (destroys lip-sync detection)
├── ❌ Avoid extreme rotation (breaks facial geometry)
├── ❌ Avoid heavy occlusion (Cutout patches too large)
```

**Rationale:** Deepfakes are robust to small perturbations; augmentations should be conservative

#### **Audio Augmentations (Applied to Waveform)**

```
Speech-safe augmentations (v1):
├── TimeShift: Shift audio by ±0.1 seconds (circular)
├── BackgroundNoise: Add MUSAN noise at SNR 15–40 dB
└── Rationale: VAD removes silence, so noise is realistic

Moderate augmentations (Phase 2):
├── PitchShift: ±2 semitones (voice variations, not speech destruction)
├── TimeStretch: ×0.95 to ×1.05 (simulate audio codec artifacts)
├── Emphasis: High-frequency boost (simulate compression)
└── Critical: Preserve phoneme structure (no extreme warping)

Problematic augmentations (AVOID):
├── ❌ SpecAugment (extreme frequency masking)
│   └── Destroys spectral cues needed for lip-sync detection
├── ❌ Audio dropout/cutout (removes speech segments)
│   └── Already applied VAD; further masking reduces modality
└── ❌ Extreme pitch shift >3 semitones (changes phonemes)
```

**Validation:** Ensure augmented audio still passes VAD (detectable speech)

---

## **IV. LOSS FUNCTIONS**

### **A. Primary Loss: Binary Cross-Entropy (Frame/Segment Level)**

```
L_bce(ŷ, y) = -[y × log(ŷ) + (1-y) × log(1-ŷ)]

where:
├── ŷ: Model prediction (softmax output, P(fake))
├── y: Target label (0=real, 1=fake)
└── Applied at: Segment level (5-frame sequences)

Implementation detail:
├── Input shape: (batch_size, num_segments, 2) [real/fake logits]
├── Target shape: (batch_size, num_segments)
├── Reduction: Mean over batch & segments
└── Weighting: Class weights if imbalanced (later)

Hyperparameter:
├── Smooth label: y_smooth = 0.05 if y=0, 0.95 if y=1
│   └── Prevents overconfident predictions (calibration)
└── Temperature scaling: logits /= T (T=1 by default)
```

---

### **B. Temporal Consistency Loss**

**Motivation:** Deepfakes have unnatural frame-to-frame transitions

```
L_temporal = Variance of embeddings within 1-second window

Mathematical formulation:
├── Input: Frame embeddings e_t ∈ ℝ^2048 for frames t ∈ [0, 5]
├── Compute mean: μ = (1/5) Σ_t e_t
├── Compute variance: σ² = (1/5) Σ_t ||e_t - μ||²
└── Loss: L_temp = σ²  (penalize high variance)

Interpretation:
├── Real videos: Low variance (smooth embedding drift)
├── Deepfakes: High variance (frame jitter, artifacts)
└── Target: Minimize variance for both (prevent "frozen" frames)

Weight in total loss:
├── L_total = L_bce + λ_temp × L_temp
├── λ_temp = 0.1 (start), tune on validation set
└── Ablation study: Test with/without (Phase 2)
```

**Critical Design Choice:** Should we penalize variance equally for real & fake?

Option 1: Equal penalty (above)
- ✅ Encourages smooth representations for both classes
- ❌ May suppress valid real-world motion artifacts

Option 2: Class-conditional weight
```
L_temp = σ² × indicator(y=1)
├── Only penalize variance for REAL videos
├── Rationale: Real videos should be smooth, deepfakes may be jagged
├── Benefit: Allows deepfakes more variance as a signal
├── Risk: May overfit to "jagged = fake" shortcut
```

**Recommendation for v1:** Option 1 (equal penalty, symmetric regularization)

---

### **C. Modality Disagreement Regularization (Phase 2)**

**Motivation:** Uncorrelated predictions from audio/video indicate weak signals

```
L_modality_agreement = |P_video(fake) - P_audio(fake)|

where:
├── P_video(fake): Video modality confidence on fake class
├── P_audio(fake): Audio modality confidence on fake class
└── High disagreement → penalize (force modality alignment)

Weight in total loss:
├── L_total = L_bce + λ_temp × L_temp + λ_disagree × L_disagree
├── λ_disagree = 0.05 (smaller weight; modalities can disagree)
└── Intuition: Disagreement is OK, but extreme disagreement suggests overfitting
```

**Alternative Formulation (Entropy-based):**

```
L_entropy = -[P_v × log(P_a) + (1-P_v) × log(1-P_a)]
├── Treat audio as "target" for video predictions
├── Encourages alignment without forcing equality
└── More flexible than L1 distance
```

---

### **D. Confidence Calibration Loss (Phase 2)**

**Motivation:** Model should output reliable confidence scores

```
L_calibration = ECE (Expected Calibration Error)

Definition:
├── Divide predictions into M=10 bins by confidence
├── For each bin: accuracy_i, confidence_i
├── ECE = Σ_i (|confidence_i - accuracy_i| × n_i / N)
└── Minimized via temperature scaling post-training

Training approach (alternative):
├── Add focal loss term: L_focal = -(1-p_t)^γ × log(p_t)
├── γ = 2 (down-weight easy examples, focus on hard cases)
└── Expected benefit: Better calibration on uncertain samples
```

---

### **E. Total Loss Function**

```
L_total = λ_bce × L_bce(ŷ_segment, y_segment)
        + λ_video × L_bce(ŷ_video, y_video)
        + λ_temp × L_temporal(embeddings)
        + λ_disagree × L_modality_disagreement(P_v, P_a)

where:
├── λ_bce = 1.0 (primary objective)
├── λ_video = 0.1 (aggregate constraint, ensure video-level accuracy)
├── λ_temp = 0.1 (temporal consistency)
├── λ_disagree = 0.0 for v1, 0.05 for Phase 2 (modality alignment)

v1 Simplified loss:
├── L_total = L_bce(segment) + 0.1 × L_video + 0.1 × L_temporal
└── Total: 3 terms, clear separation of concerns
```

---

## **V. EVALUATION PROTOCOL**

### **A. Primary Metrics**

#### **1. Area Under ROC Curve (AUC)**

```
Definition:
├── ROC: Plot TPR vs FPR as confidence threshold varies
├── AUC: Integral of ROC curve; range [0, 1]
├── Interpretation: P(model ranks fake higher than real) = AUC

Advantages:
├── ✅ Threshold-independent
├── ✅ Handles class imbalance naturally
├── ✅ Single number summarizing classifier quality
├── ✅ Standard in deepfake detection literature

Disadvantages:
├── ❌ Doesn't account for cost of FP vs FN
├── ❌ Sensitive to ranking, not calibration

Computation:
├── Use sklearn.metrics.roc_auc_score
├── Binary classification: AUC ∈ [0.5, 1.0]
├── 0.5 = random classifier
├── 1.0 = perfect classifier
└── Target for v1: ≥85% (0.85)
```

#### **2. Equal Error Rate (EER)**

```
Definition:
├── Threshold where FPR = FNR
├── Lower EER = better (0 is perfect)
├── Useful for forensic applications (equal cost assumption)

Relationship to AUC:
├── EER and AUC correlated but different
├── EER emphasizes balanced performance
├── AUC emphasizes ranking quality

Computation:
├── Find threshold t where FPR(t) = FNR(t)
├── Report as percentage (lower is better)
└── Target for v1: ≤15% (0.15)
```

#### **3. Average Precision (AP)**

```
Definition:
├── Area under Precision-Recall curve
├── Different from AUC (uses precision instead of TPR)
├── More sensitive to class imbalance than AUC

Advantage over AUC:
├── ✅ Focused on positive class (fake) performance
├── ✅ Better reflects real-world scenarios (more reals than fakes)

Computation:
├── sklearn.metrics.average_precision_score
└── Target for v1: ≥80% (0.80)
```

---

### **B. Secondary Metrics**

#### **1. Calibration Error**

```
Expected Calibration Error (ECE):
├── Definition: Average difference between confidence & accuracy
├── Computation: Sort predictions into M=10 bins
├── ECE = Σ (|confidence_i - accuracy_i| × weight_i)
├── Target: ECE < 0.05 (well-calibrated)

Interpretation:
├── ECE = 0.02 → "When model says 85% confident, accuracy ≈ 83%"
├── ECE = 0.20 → "Model overconfident; actual accuracy much lower"

Implementation:
├── Use temperature scaling on logits: logits /= T
├── Tune T on validation set to minimize ECE
└── Apply same T at test time
```

#### **2. Accuracy, Precision, Recall, F1**

```
Metric definitions:
├── Accuracy = (TP + TN) / (TP + TN + FP + FN)
├── Precision = TP / (TP + FP) — "of detected fakes, how many true?"
├── Recall = TP / (TP + FN) — "of all fakes, how many detected?"
├── F1 = 2 × (Precision × Recall) / (Precision + Recall)

Caveats:
├── ❌ Threshold-dependent (require selecting operating point)
├── ❌ Imbalanced datasets make accuracy misleading
├── ✅ Useful for reporting at specific thresholds (e.g., 95% TPR)

Target thresholds:
├── 95% TPR: Catch 95% of fakes (minimize FN in forensics)
├── FPR @ 95% TPR: Expected FP rate when catching 95% fakes
└── Target for v1: FPR < 10% @ 95% TPR
```

---

### **C. Cross-Dataset Generalization**

**Test Procedure:**

```
Step 1: Train on FaceForensics++ (training set)
Step 2: Validate on FaceForensics++ (validation set) → Report AUC_FF
Step 3: Test on FaceForensics++ (test set) → Report AUC_FF_test
Step 4: Evaluate on Celeb-DF (hold-out) → Report AUC_CDF
Step 5: Evaluate on DFDC (optional, Phase 3) → Report AUC_DFDC

Generalization gap:
├── Gap = AUC_FF_test - AUC_CDF
├── Large gap (>10%) indicates overfitting to FaceForensics++ distribution
├── Acceptable gap: 5–8% (natural distribution shift)

Interpretation:
├── If AUC_FF = 90%, AUC_CDF = 82%:
│   └── Generalizes OK (8% gap acceptable for new generator)
├── If AUC_FF = 90%, AUC_CDF = 70%:
│   └── POOR generalization (20% gap, model overfit)
```

---

### **D. Modality Ablation Tests**

**Purpose:** Understand contribution of each modality

```
Ablation experiments:

Baseline: Video + Audio
├── AUC_joint = 0.85

Video-only: Drop audio encoder
├── Input: Video frames + temporal encoder
├── Audio modality: N/A
├── AUC_video = 0.78
├── Degradation: 0.85 - 0.78 = 0.07 (7% AUC drop)
├── Interpretation: Audio contributes ~7% AUC

Audio-only: Drop video encoder
├── Input: Audio waveform + mel-spec
├── Video modality: N/A
├── AUC_audio = 0.72
├── Degradation: 0.85 - 0.72 = 0.13 (13% AUC drop)
├── Interpretation: Video contributes ~13% AUC

Conclusion:
├── Video > Audio for FaceForensics++ detection
├── But Audio critical for Wav2Lip (audio-driven artifacts)
└── Fusion improves over any single modality (+5–8%)
```

**Important:** Report for BOTH datasets
- FaceForensics++: Video-heavy artifacts dominant
- Celeb-DF: Audio-heavy (Wav2Lip), should show larger audio contribution

---

### **E. Failure Case Analysis**

**Dataset & Systematic Error Analysis:**

```
Step 1: Identify worst-performing cases
├── Collect 10 videos with lowest model confidence
├── Analyze pattern: Is it a particular generator? Compression level?
└── Report: "Model struggles with NeuralTextures @ c40 compression"

Step 2: Generator-specific performance
├── Compute AUC separately per generator
├── FaceForensics++:
│   ├── DeepFaceLab: AUC = 0.88 (model performs well)
│   ├── Face2Face: AUC = 0.85
│   ├── FaceSwap: AUC = 0.82 (harder)
│   └── NeuralTextures: AUC = 0.75 (hardest)
├── Action: Prioritize NeuralTextures augmentation in Phase 2

Step 3: Compression level analysis
├── FaceForensics++ c23:
│   ├── High quality: AUC = 0.88
│   ├── Medium quality: AUC = 0.85
│   └── Low quality: AUC = 0.72 (artifacts hide deepfake cues)
├── Action: Augment training with low-quality videos (Phase 2)

Step 4: Audio-specific failures (Celeb-DF)
├── Wav2Lip lip-sync artifacts: Sometimes perfect (hard to detect)
├── Model AUC = 0.80 (5% worse than FaceForensics++ generators)
├── Root cause: Audio modality overfitting, missing visual cues
└── Action: Balance audio/video loss weights (Phase 2)

Step 5: Report
├── "Model AUC varies by generator: 75–88% on FaceForensics++"
├── "Cross-dataset gap of 8% to Celeb-DF due to Wav2Lip artifacts"
├── "Compression artifacts at c40 reduce AUC by 16%"
└── "Phase 2 priority: Robust low-quality fake detection"
```

---

## **VI. OVERFITTING & SHORTCUT RISKS**

### **A. Dataset Bias Risks**

#### **1. Generator-Specific Overfitting**

**Risk:** Model memorizes visual artifacts of specific generators instead of learning deepfake detection

```
Example (Problematic):
├── Model learns: "DeepFaceLab always has green tint in lighting"
├── Outcome: 95% AUC on FaceForensics++
├── Failure: 60% AUC on other generators (overfitted to green tint)

Detection & Mitigation:
├── Metric: Per-generator AUC variance
├── If max_AUC - min_AUC > 15% → overfitting
├── Mitigation: Mix generators in batch (5–10% chance for each)
└── Validation: Report per-generator AUC in evaluation
```

**Cross-Dataset Defense:**
- ✅ Celeb-DF uses Wav2Lip (different visual artifacts)
- ✅ Generator diversity forces learning of general deepfake cues
- ⚠️ Still risky with only 2 datasets; Phase 3 adds DFDC (3rd generator)

#### **2. Compression Level Bias**

**Risk:** Model exploits compression artifacts specific to c23, fails on c0 or c40

```
Example (Problematic):
├── Model learns: "H.264 @ c23 creates blocking artifacts at mouth region"
├── Outcome: 88% AUC on c23
├── Failure: 55% AUC on c0 (lossless, no blocking), 70% AUC on c40 (extreme artifacts)

Detection & Mitigation:
├── Metric: AUC across compression levels
├── If AUC(c0) - AUC(c23) > 10% → compression bias
├── Mitigation: Augment with multiple compression levels (Phase 2)
└── Validation: Report AUC for c0, c23, c40 separately
```

**FaceForensics++ Protocol:**
- Use c23 for training (standard practice)
- Test on c0, c40 to measure robustness
- Phase 2: Mix compression levels in training (augmentation)

#### **3. Speaker/Demographic Bias**

**Risk:** Model overfits to speaker characteristics (age, gender, ethnicity)

```
Example (Problematic):
├── FaceForensics++ demographic split: 70% male, 30% female
├── Model learns male-specific deepfake cues
├── Outcome: 90% AUC on males, 70% AUC on females

Detection & Mitigation:
├── Metric: Per-demographic AUC gap
├── If max_AUC - min_AUC > 5% → demographic bias
├── Mitigation: Balance batch composition (10% female, 10% male, 10% other)
└── Validation: Report AUC for age/gender subgroups if available
```

**Celeb-DF Defense:**
- ✅ Celebrity speakers (high diversity)
- ✅ Demographic distribution differs from FaceForensics++
- ⚠️ Still predominantly famous/attractive individuals

---

### **B. Modality-Specific Leakage**

#### **1. Audio-Only Fake Detection**

**Risk:** Audio modality alone detects synthetic speech, not deepfake

```
Example (Problematic):
├── Celeb-DF Wav2Lip audio: Synthesized speech (TTS-like artifacts)
├── Audio modality learns: "Synthetic speech = fake"
├── Problem: This isn't detecting lip-sync deepfakes, just audio quality
├── Outcome: Model fails on real audio dubbed onto fake video

Mitigation:
├── Ensure audio modality trained on REALISTIC audio
├── Use FaceForensics++ audio (real YouTube speeches, not synthetic)
├── Celeb-DF audio: May be natural or synthetic (unclear from paper)
└── Testing: Report per-modality AUC on both datasets
```

#### **2. Video-Only Fake Detection via Artifacts**

**Risk:** Video modality detects compression/codec artifacts, not deepfake manipulations

```
Example (Problematic):
├── FaceForensics++ compressed to c23: Creates specific H.264 artifacts
├── Video modality learns: "H.264 blocking @ mouth = fake"
├── Problem: This isn't detection of deepfake reenactment, just compression
├── Outcome: Model fails on perfect-quality deepfakes (no compression)

Mitigation:
├── Use c0 (lossless) for training if possible (but too large)
├── Augment with multiple compression levels (Phase 2)
├── Validate on high-quality deepfakes (if available)
└── Celeb-DF naturally uses different compression (YouTube quality)
```

**Cross-Dataset Validation:**
- ✅ Celeb-DF uses different compression codec (defense against artifact memorization)
- ✅ Wav2Lip artifacts (lip-sync timing) differ from Face2Face reenactment
- ✓ Modality ablation ensures both modalities contribute meaningfully

---

### **C. Temporal Leakage**

**Risk:** Model detects frame-level temporal inconsistencies specific to training algorithms

```
Example (Problematic):
├── DeepFaceLab algorithm has characteristic temporal jitter
├── Temporal consistency loss learns: "Frame jitter = DeepFaceLab = fake"
├── Problem: Other generators may not have jitter (different algorithm)
├── Outcome: 92% AUC on DeepFaceLab, 65% AUC on Face2Face

Mitigation:
├── Temporal loss should generalize (penalizes extreme variance)
├── Cross-dataset validation catches this (Celeb-DF uses different generation)
├── Phase 2: Per-generator performance analysis (failure case analysis)
└── Reporting: Stratified AUC by generator
```

---

### **D. Confidence Calibration Shortcuts**

**Risk:** Model becomes overconfident on in-distribution data, fails on out-of-distribution

```
Example (Problematic):
├── Model trained on FaceForensics++ (balanced, consistent generation)
├── Achieves 95% confidence on FaceForensics++
├── But: Celeb-DF (different codec, different speaker) triggers uncertainty
├── Model still outputs 80% confidence (miscalibrated)
├── Downstream forensic tool makes mistakes due to overconfidence

Mitigation:
├── Monitor calibration metrics (ECE, MCE, Brier score)
├── Apply temperature scaling on validation set
├── Report confidence intervals, not point estimates
└── Phase 2: Ensemble model (natural uncertainty quantification)
```

---

## **VII. TRAINING CHECKLIST & VALIDATION PROTOCOL**

### **A. Pre-Training Setup**

```
Data preparation:
├── [ ] Download FaceForensics++ c23 (MP4, compressed)
├── [ ] Extract frames @ 5 FPS into organized structure
├── [ ] Apply face detection (RetinaFace) on all frames
├── [ ] Remove frames with failed face detection (save frame IDs)
├── [ ] Extract audio from MP4 files (ffmpeg → 16kHz WAV)
├── [ ] Apply VAD to audio, save speech segment timestamps
├── [ ] Create train/val/test splits (stratified by generator)
├── [ ] Compute dataset statistics:
│   ├── Frame count: _________
│   ├── Segment count (1-sec windows): _________
│   ├── Generator distribution: _________
│   ├── Real/Fake ratio: _________
│   └── Mean face size: _________ pixels

Model architecture setup:
├── [ ] Load EfficientNet-B3 (ImageNet pretrained) 
├── [ ] Load TemporalConv module (1D conv + pooling)
├── [ ] Load wav2vec2-base (speech pretrained)
├── [ ] Load CrossModalAttentionFusion
├── [ ] Verify parameter counts:
│   ├── Video backbone: ~12M params
│   ├── Temporal encoder: ~0.5M params
│   ├── Audio encoder: ~95M params (wav2vec2)
│   ├── Fusion: ~1M params
│   └── Classification head: ~0.1M params
│   └── Total: ~108M params

Optimization setup:
├── [ ] Optimizer: AdamW (β₁=0.9, β₂=0.999)
├── [ ] Learning rate: 1e-3 (Stage 1), 1e-4 (Stages 2-3)
├── [ ] Scheduler: CosineAnnealingLR (T_max=5 per stage)
├── [ ] Batch size: 32 (or largest that fits GPU memory)
├── [ ] Gradient clipping: 1.0 (prevent exploding gradients)

Loss function setup:
├── [ ] Binary cross-entropy (with label smoothing ε=0.05)
├── [ ] Temporal consistency loss (λ=0.1)
├── [ ] Video-level aggregation loss (λ=0.1)
└── [ ] Modality disagreement loss (λ=0 for v1)
```

### **B. Training Validation Checkpoints**

```
Per epoch:
├── [ ] Log training loss, validation loss
├── [ ] Compute validation AUC (every 1–2 epochs)
├── [ ] Compute per-generator AUC (detect overfitting)
├── [ ] Monitor for NaN/Inf (stop if detected)
├── [ ] Save checkpoint if validation AUC improves

Early stopping:
├── [ ] Patience: 3 epochs (stop if val AUC plateaus)
├── [ ] Minimum delta: 0.002 (require AUC improvement >0.2%)
├── [ ] Restore best model after early stopping

End of stage:
├── [ ] Report validation AUC by generator
├── [ ] Report per-modality contribution (via ablation if stage 3)
└── [ ] Decide: Proceed to next stage? Hyperparameter adjustment?
```

### **C. Final Evaluation Protocol**

```
On held-out FaceForensics++ test set:
├── [ ] Compute AUC (primary metric)
├── [ ] Compute EER
├── [ ] Compute Average Precision
├── [ ] Compute per-generator AUC
├── [ ] Compute per-modality contribution (ablation)
├── [ ] Compute calibration error (ECE)
├── [ ] Report operating point (95% TPR, FPR?)

On Celeb-DF (cross-dataset):
├── [ ] Compute AUC (key generalization metric)
├── [ ] Compute EER
├── [ ] Compute per-modality contribution
├── [ ] Analyze failure cases (Wav2Lip-specific artifacts?)
└── [ ] Report generalization gap (FF test - Celeb-DF)

Failure analysis:
├── [ ] Identify 10 worst-predicted videos
├── [ ] Categorize failures: Generator? Compression? Modality?
├── [ ] Generate saliency maps for top failures
├── [ ] Recommendation for Phase 2 improvements

Final report:
├── [ ] AUC_FF = 0.?? (target ≥0.85)
├── [ ] AUC_CDF = 0.?? (target ≥0.80)
├── [ ] Gap = 0.?? (target ≤0.08)
├── [ ] EER = 0.?? (target ≤0.15)
├── [ ] Per-generator AUC variance: 0.?? (target ≤0.15)
└── [ ] Ready for Phase 2? YES/NO + reasoning
```

---

## **VIII. CLARIFICATION QUESTIONS (DECISION POINTS)**

### **A. Data Availability & Scale**

**Q1: FaceForensics++ Access**
- Do we have access to FaceForensics++ v4 full dataset? 
- Or restricted version (fewer generators)?
- Disk space available: _____ GB
- Expected data download & preprocessing time: _____ days

**Q2: Celeb-DF v2 Audio**
- Is Celeb-DF audio genuine (real actors with dubbed audio) or synthetic TTS?
- This affects audio modality training (synthetic TTS = different distribution)
- Recommendation: Obtain metadata on audio source

**Q3: DFDC Dataset (Phase 3)**
- Budget for DFDC inclusion (Phase 3)? 
- Estimated resources: GPU hours, disk space
- Alternative: Stick with FF++ + Celeb-DF for v1 (sufficient for 85% AUC goal)

---

### **B. Labeling & Annotation**

**Q4: Frame-Level Annotations**
- Are there any frame-level deepfake confidence scores available?
- Or strictly video-level labels?
- Recommendation for v1: Use video-level labels, derive frame pseudo-labels

**Q5: Generator Metadata**
- Does FaceForensics++ metadata specify which generator per video?
- Needed for per-generator ablation analysis
- Expected: Yes (standard in FF++ paper)

**Q6: Audio Source Metadata**
- For FaceForensics++, is original audio preserved?
- Or replaced with synthetic speech (TTS)?
- Recommendation: Use original YouTube audio (realistic)

---

### **C. Training Strategy Decisions**

**Q7: Staged vs Joint Training**
- Preference for staged training (3 stages) vs joint end-to-end?
- Recommendation: Staged for v1 (clearer debugging, modular design)
- Constraint: Timeline? (Staged = 1.5× longer training)

**Q8: Audio Encoder Fine-Tuning Depth**
- Freeze all wav2vec2 except linear projection (conservative)?
- Or fine-tune last 4 blocks (moderate)?
- Recommendation: Last 4 blocks (balance between transfer & overfitting)

**Q9: Temporal Window Duration**
- Locked at 1 second (5 frames @ 5 FPS)?
- Or flexibility for experimentation?
- Recommendation: Fixed for v1 (as per Model Contract)

---

### **D. Augmentation & Robustness**

**Q10: Compression Level for Training**
- Train on c23 only (standard)?
- Or mix c0, c23, c40 during training?
- Recommendation for v1: c23 only (matches FaceForensics++ standard)
- Phase 2: Augment with c0, c40 (robustness)

**Q11: Audio Augmentation Aggressiveness**
- Conservative augmentations (time shift, noise)?
- Or aggressive (pitch shift, time stretch)?
- Recommendation: Conservative for v1, phase 2 if needed

---

### **E. Evaluation & Reporting**

**Q12: Per-Generator Reporting**
- Report AUC per generator separately?
- Recommendation: YES (required for failure analysis, Phase 2 prioritization)

**Q13: Cross-Codec Testing**
- Test on H.265 (HEVC) in addition to H.264?
- Recommendation for v1: H.264 only (code baseline)
- Phase 3: H.265 robustness testing

**Q14: Explainability During Training**
- Compute saliency maps during validation?
- Recommendation for v1: NO (expensive, Phase 3 feature)
- But save top-10 failure cases for manual inspection

**Q15: Confidence Threshold Selection**
- Use standard 0.5 threshold for 50/50 classification?
- Or optimize for specific operating point (e.g., 95% TPR)?
- Recommendation: Report multiple operating points (0.5, 0.6, 0.7)

---

## **IX. EXPECTED OUTCOMES & SUCCESS CRITERIA**

### **V1 Baseline (Staged Training, 20 epochs)**

```
Training dynamics (expected):
├── Epoch 1–5: Video baseline
│   ├── Training AUC: 0.50 → 0.75 (rapid improvement)
│   ├── Validation AUC: 0.50 → 0.72 (slight lag)
│   └── Loss: Cross-entropy decaying smoothly
│
├── Epoch 6–10: Audio baseline
│   ├── Training AUC: 0.72 → 0.82 (audio adds signal)
│   ├── Validation AUC: 0.70 → 0.80
│   └── Audio alone: ~5–10% AUC gain expected
│
└── Epoch 11–20: Joint fine-tuning
    ├── Training AUC: 0.82 → 0.90 (fusion optimizes)
    ├── Validation AUC: 0.80 → 0.85 (target!)
    └── Joint > sum of modalities (synergy)

Final performance (expected):
├── FaceForensics++ test: AUC = 0.84–0.87 ✅
├── Celeb-DF (hold-out): AUC = 0.78–0.82 ✅
├── Generalization gap: ~0.05–0.08 (acceptable)
├── Per-generator AUC range: 0.80–0.88 (no extreme variance)
└── Calibration (ECE): 0.03–0.05 (good calibration)

Per-modality contribution:
├── Video-only: ~0.78 AUC (core signal)
├── Audio-only: ~0.72 AUC (complementary)
├── Joint: ~0.85 AUC (synergy, 5–8% gain)
└── Fusion ratio: ~70% video, ~30% audio

Failure analysis (expected):
├── Hardest generator: NeuralTextures (AUC ~0.80)
├── Easiest generator: DeepFaceLab (AUC ~0.88)
├── Generalization challenge: Celeb-DF Wav2Lip (lip-sync artifacts)
└── Action: Phase 2 focus on robustness to audio-driven deepfakes
```

---

## **X. FINAL SUMMARY**

**This training strategy achieves v1.0 targets (85%+ AUC) via:**

1. **Staged Training:** Modular design, lower risk, easier debugging
2. **Balanced Datasets:** FaceForensics++ (train) + Celeb-DF (test) for generalization
3. **Multimodal Fusion:** Video (70%) + Audio (30%) complementary signals
4. **Conservative Augmentation:** Preserve deepfake cues, avoid shortcut learning
5. **Rigorous Evaluation:** Cross-dataset, per-generator, per-modality analysis
6. **Explicit Risk Management:** Systematic failure analysis, overfitting detection

**Phase 2 & 3 roadmap defined:** Unfreezing, attention fusion, optical flow, lip-sync (as per IMPLEMENTATION_ROADMAP.md)

---

**Next Step:** Answer the 15 clarification questions above to confirm training strategy finalization.

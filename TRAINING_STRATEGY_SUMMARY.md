# **TRAINING & EVALUATION STRATEGY — INTEGRATION SUMMARY**

**New Document:** [TRAINING_EVALUATION_STRATEGY.md](TRAINING_EVALUATION_STRATEGY.md) (36KB)  
**Status:** ✅ PLAN MODE COMPLETE  
**Date:** January 3, 2026

---

## **📍 HOW THIS FITS IN**

**Document Hierarchy:**

```
MODEL_CONTRACT_v1.md (Locked decisions)
        ↓
TRAINING_EVALUATION_STRATEGY.md (NEW)
        ↓
IMPLEMENTATION_ROADMAP.md (Phase 1–3 code)
        ↓
CODE_IMPACT_ANALYSIS.md (Specific implementation)
```

---

## **📊 WHAT'S IN THE STRATEGY DOCUMENT**

### **I. Dataset Strategy** (Section I)
- ✅ Primary dataset: FaceForensics++ (265 train, 33 val, 33 test)
- ✅ Secondary dataset: Celeb-DF v2 (hold-out cross-dataset eval)
- ✅ Augmentation: DFDC (Phase 3 optional)
- ✅ Class imbalance handling (balanced datasets for v1)
- ✅ Real vs Fake diversity matrix (4 generators × compression levels)

### **II. Label Granularity** (Section II)
- ✅ Video-level labels (primary, unambiguous)
- ✅ Frame-level pseudo-labels (derived, hard assignment)
- ✅ Segment-level supervision (1-sec windows)
- ✅ Weak vs strong labels (hybrid strategy)

### **III. Training Pipeline** (Section III)
- ✅ Per-modality pretraining (video + audio)
- ✅ **Staged training (3 stages, RECOMMENDED for v1):**
  - Stage 1: Video backbone + temporal encoder (5 epochs)
  - Stage 2: Audio encoder fine-tuning (5 epochs)
  - Stage 3: Joint fusion training (10 epochs)
- ✅ Freezing schedule (backbone frozen, temporal/audio trainable)
- ✅ Augmentations (conservative video + audio, Phase 2 expansion)

### **IV. Loss Functions** (Section IV)
- ✅ Primary: Binary cross-entropy with label smoothing
- ✅ Temporal consistency loss (λ=0.1, penalizes frame variance)
- ✅ Video-level aggregation loss (ensure video-accuracy)
- ✅ Modality disagreement (Phase 2, λ=0.05)
- ✅ Confidence calibration (Phase 2, focal loss)
- ✅ **Total loss: L = 1.0×L_bce + 0.1×L_video + 0.1×L_temporal**

### **V. Evaluation Protocol** (Section V)
- ✅ Primary metrics: AUC, EER, Average Precision
- ✅ Secondary metrics: Accuracy, Precision, Recall, F1, ECE
- ✅ **Cross-dataset generalization (FF++ → Celeb-DF)**
- ✅ Modality ablation tests (video-only vs audio-only vs joint)
- ✅ Failure case analysis (per-generator, per-compression, per-modality)

### **VI. Overfitting & Risks** (Section VI)
- ✅ Dataset bias risks (generator overfitting, compression bias, demographic bias)
- ✅ Modality-specific leakage (audio artifacts, compression shortcuts)
- ✅ Temporal leakage (generator-specific jitter patterns)
- ✅ Confidence calibration shortcuts (overconfidence on in-distribution)

### **VII. Training Checklist** (Section VII)
- ✅ Pre-training setup (data prep, model architecture, optimization)
- ✅ Training validation checkpoints (per-epoch monitoring)
- ✅ Final evaluation protocol (comprehensive testing)

### **VIII. 15 Clarification Questions** (Section VIII)
- **Q1–3:** Data availability (FaceForensics++, Celeb-DF, DFDC)
- **Q4–6:** Labeling & annotation (frame-level, generator metadata)
- **Q7–9:** Training decisions (staged vs joint, fine-tuning depth)
- **Q10–11:** Augmentation strategy (compression levels, audio aggressiveness)
- **Q12–15:** Evaluation & reporting (per-generator, cross-codec, explainability)

### **IX. Expected Outcomes** (Section IX)
- ✅ V1 baseline performance prediction
  - FaceForensics++ test: 0.84–0.87 AUC ✅
  - Celeb-DF test: 0.78–0.82 AUC ✅
  - Generalization gap: ~0.05–0.08 (acceptable)

---

## **🔑 KEY DESIGN DECISIONS**

### **Training Approach: STAGED (3-Stage)**

**Why staged over end-to-end:**
- ✅ Modular debugging (isolate video/audio issues)
- ✅ Lower memory during early stages
- ✅ Clear convergence checkpoints
- ✅ Each modality optimized separately before fusion
- ⚠️ Trade-off: 1.5× longer training time (acceptable for v1)

**Stage breakdown:**
- Stage 1 (5 epochs): Video-only with frozen backbone
- Stage 2 (5 epochs): Audio-only with frozen backbone  
- Stage 3 (10 epochs): Joint fusion with selective unfreezing

---

### **Loss Function: Multi-Objective**

**Why 3 terms instead of single cross-entropy:**
- **L_bce (segment):** Direct classification signal (segment-level)
- **L_video:** Ensure video-level accuracy (coarse constraint)
- **L_temporal:** Regularization for smooth representations (generalization)

**Total loss:** L = 1.0×L_bce + 0.1×L_video + 0.1×L_temporal

---

### **Dataset Split: Stratified by Generator**

**Why stratification matters:**
- Ensures each split (train/val/test) has similar generator distribution
- Prevents: All DeepFaceLab in train, all Face2Face in test (would falsely inflate AUC)
- Recommendation: Stratify by generator when splitting FaceForensics++

---

### **Evaluation: Mandatory Cross-Dataset**

**FaceForensics++ test AUC alone is insufficient:**
- ✅ Test on Celeb-DF (different generator, compression, speaker dist)
- ✅ Report generalization gap (gap >10% = overfitting)
- ✅ Celeb-DF is primary metric for final v1.0 sign-off

---

## **⚠️ CRITICAL RISKS IDENTIFIED**

| Risk | Mitigation | Monitoring |
|------|-----------|-----------|
| **Generator overfitting** | Stratified split, per-generator AUC | AUC variance >15% = FLAG |
| **Compression artifacts** | Augmentation Phase 2, c23 baseline | Test c0, c40 separately |
| **Audio-only leakage** | VAD ensures realistic audio, FaceForensics++ sourced | Video-only ablation AUC |
| **Temporal shortcuts** | Cross-dataset test (Celeb-DF uses Wav2Lip) | Celeb-DF AUC gap >10% = FLAG |
| **Overconfidence** | Temperature scaling, calibration metric ECE | ECE > 0.05 = recalibrate |

---

## **✅ RECOMMENDED PRE-TRAINING CHECKLIST**

**Before running Stage 1:**

- [ ] FaceForensics++ downloaded & preprocessed
- [ ] Frames extracted @ 5 FPS, faces detected
- [ ] Audio extracted, VAD applied
- [ ] Train/val/test splits created (stratified by generator)
- [ ] Dataset statistics computed & validated
- [ ] EfficientNet-B3 loaded (ImageNet pretrained)
- [ ] wav2vec2-base loaded (speech pretrained)
- [ ] Model architecture verified (parameter counts match)
- [ ] Optimization hyperparameters set (AdamW, learning rate schedule)
- [ ] Loss functions implemented (3-term loss)
- [ ] Early stopping configured (patience=3, min_delta=0.002)

---

## **📋 15 CLARIFICATION QUESTIONS SUMMARY**

**These need to be answered before implementation:**

1. **Data:** Do we have FaceForensics++ full access?
2. **Data:** Is Celeb-DF audio real or synthetic TTS?
3. **Data:** Budget for DFDC (Phase 3)?
4. **Labels:** Any frame-level deepfake confidence scores available?
5. **Labels:** Does FF++ metadata specify generator per video?
6. **Labels:** Is FaceForensics++ audio original or replaced?
7. **Training:** Preference for staged (3-stage) vs joint (end-to-end)?
8. **Training:** Audio encoder fine-tune depth (1 block vs 4 blocks)?
9. **Training:** Is 1-second temporal window flexible?
10. **Augmentation:** Train on c23 only or mix c0/c23/c40?
11. **Augmentation:** Audio augmentation aggressiveness level?
12. **Evaluation:** Per-generator AUC reporting (mandatory)?
13. **Evaluation:** Cross-codec testing (H.265 optional)?
14. **Evaluation:** Saliency maps during validation (Phase 3)?
15. **Evaluation:** Confidence threshold strategy (0.5 vs optimized)?

**If you have answers to these, we can finalize the training protocol.**

---

## **🚀 NEXT STEPS**

1. **Review:** Read [TRAINING_EVALUATION_STRATEGY.md](TRAINING_EVALUATION_STRATEGY.md) (20 min)
2. **Clarify:** Answer the 15 questions in Section VIII (10 min)
3. **Confirm:** Validate strategy aligns with available data/resources (10 min)
4. **Proceed:** Finalize training protocol, begin data preprocessing (Phase 1)

---

**Document Status:** ✅ PLAN MODE COMPLETE  
**Ready for Data Preprocessing:** YES  
**Ready for Implementation:** AFTER answering Q1–Q15

**Total Design Freeze Documentation:** 10 files, ~200KB

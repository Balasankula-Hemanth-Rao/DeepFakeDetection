# FaceForensics Training Parameters

## 📊 DATASET PARAMETERS (c40)

| Parameter | Value |
|-----------|-------|
| **dataset_name** | FaceForensics++ (Deepfakes + YouTube) |
| **compression_level** | c40 (heavy compression) |
| **total_videos** | 2,000 |
| **real_videos** | 1,000 (YouTube original) |
| **fake_videos** | 1,000 (Deepfakes manipulation) |
| **avg_video_duration_sec** | ~10-15 seconds |
| **frames_per_video** | ~30-60 (at 3 FPS) |
| **total_frames** | 115,673 |
| **total_audio_segments** | Not extracted (frame-only approach) |

---

## 🎥 VIDEO PREPROCESSING PARAMETERS

| Parameter | Value |
|-----------|-------|
| **frame_extraction_fps** | 3 FPS |
| **frame_sampling_strategy** | Uniform temporal sampling |
| **temporal_window_sec** | N/A (single frame model) |
| **temporal_window_overlap** | N/A |
| **face_detection_model** | None (full frame used) |
| **face_detection_confidence_threshold** | N/A |
| **face_alignment_enabled** | No |
| **multiple_faces_handling** | N/A |
| **frames_without_face_handling** | All frames kept |
| **frame_resize_resolution** | 224×224 (via model transforms) |
| **frame_normalization_method** | ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) |

---

## 🔊 AUDIO PREPROCESSING PARAMETERS

| Parameter | Value |
|-----------|-------|
| **audio_sample_rate** | Not used (frame-only model) |
| **audio_channels** | N/A |
| **audio_window_length_sec** | N/A |
| **voice_activity_detection_enabled** | No |
| **silence_handling_strategy** | N/A |
| **audio_representation_type** | N/A |
| **audio_augmentation_enabled** | No |

---

## 🧠 MODEL ARCHITECTURE PARAMETERS

| Parameter | Value |
|-----------|-------|
| **video_backbone** | EfficientNet-B3 |
| **video_backbone_pretrained** | Yes (ImageNet) |
| **temporal_encoder_type** | None (single frame classification) |
| **temporal_encoder_kernel_size** | N/A |
| **audio_encoder_type** | None |
| **audio_encoder_pretrained** | N/A |
| **fusion_strategy** | None (unimodal - video only) |
| **fusion_embedding_dim** | N/A |

---

## ⚙️ TRAINING PARAMETERS

| Parameter | Value |
|-----------|-------|
| **optimizer_type** | AdamW |
| **learning_rate_video** | 1e-4 (0.0001) |
| **learning_rate_audio** | N/A |
| **learning_rate_fusion** | N/A |
| **batch_size** | 32 (GPU) / 16 (CPU) |
| **total_epochs** | 5 (configurable) |
| **early_stopping_patience** | Not implemented |
| **loss_function** | CrossEntropyLoss |
| **label_smoothing** | 0.0 (disabled) |
| **temporal_consistency_loss_enabled** | No |
| **modality_dropout_enabled** | No |

---

## 📈 TRAINING MONITORING PARAMETERS

| Parameter | Value |
|-----------|-------|
| **current_epoch** | Variable (1-5) |
| **training_accuracy** | To be measured |
| **validation_accuracy** | To be measured |
| **training_loss** | Logged per batch |
| **validation_loss** | Not computed (no validation during training) |

---

## 🧪 EVALUATION PARAMETERS

| Parameter | Value |
|-----------|-------|
| **evaluation_split_strategy** | Stratified random split (70/15/15) |
| **validation_dataset_name** | FaceForensics++ validation split |
| **test_dataset_name** | FaceForensics++ test split |
| **evaluation_metric_primary** | Accuracy |
| **evaluation_metric_secondary** | Loss |

---

## 📝 Dataset Splits

### Training Set
- **Total frames:** 80,566
- **Fake frames:** 40,348
- **Real frames:** 40,218
- **Location:** `data/processed/train/`

### Validation Set
- **Total frames:** ~17,000
- **Fake frames:** ~8,623
- **Real frames:** ~8,760
- **Location:** `data/processed/val/`

### Test Set
- **Total frames:** ~17,000
- **Location:** `data/processed/test/`

---

## 💡 Implementation Notes

1. **Frame-based approach:** Simplified single-frame classification model, not a full temporal/multimodal system
2. **No audio processing:** Current implementation uses visual frames only
3. **No face detection:** Uses full frames rather than cropped faces
4. **Simple architecture:** EfficientNet-B3 backbone with binary classification head
5. **Baseline model:** For production use, consider adding:
   - Temporal modeling (LSTM/Transformer)
   - Audio analysis
   - Face detection and alignment
   - Multimodal fusion strategies
   - Data augmentation
   - Advanced regularization

---

## 🔧 Configuration Files

- **Training script:** `src/train.py`
- **Model definition:** `src/models/frame_model.py`
- **Preprocessing script:** `scripts/preprocess_faceforensics_cv2.py`
- **Colab training:** `colab_training.py`

---

**Last Updated:** 2026-01-17

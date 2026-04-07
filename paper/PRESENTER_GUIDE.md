# Presenter Guide - Multimodal Deepfake Detection

## 1. Talk Structure (12-15 minutes)

1. Problem and motivation (1.5 min)
2. Research gap and objective (1.5 min)
3. Dataset and protocol (2 min)
4. Model architecture (2 min)
5. Training setup and constraints (1 min)
6. Results and metrics (3 min)
7. Error analysis and limitations (2 min)
8. Conclusion and future work (1 min)

## 2. Slide-by-Slide Script

### Slide 1 - Title
- Line to say: "This work presents a supervised multimodal deepfake detector that combines visual and audio cues using EfficientNet and mel-spectrogram fusion."
- Emphasize practical setup: 4 GB VRAM device, reproducible pipeline.

### Slide 2 - Why This Matters
- Line to say: "Visual-only detectors often fail when manipulation styles change, so we combine modalities to improve robustness."
- Mention real-world risk: misinformation, identity abuse, trust erosion.

### Slide 3 - Research Objective
- Primary question: Can supervised multimodal fusion improve detection reliability compared to unimodal baselines?
- Secondary question: How does class imbalance affect real-video specificity?

### Slide 4 - Dataset and Splits
- Dataset: FaceForensics++ (c23), real + 4 fake generation methods.
- Split: 70/15/15 with seed 42.
- Mention audit: contamination checks were performed before training.

### Slide 5 - Pipeline Overview
- Video branch: face crops, 224x224, EfficientNet feature extraction.
- Audio branch: 64-bin mel-spectrogram CNN.
- Fusion: feature concatenation + MLP classifier.

### Slide 6 - Training Setup
- Optimizer: AdamW.
- Loss: cross-entropy.
- Frames per video: 8.
- Practical constraint: RTX 3050 laptop GPU (4 GB), which influenced model variant choices.

### Slide 7 - Main Metrics Table
- Show AUC, Accuracy, F1 for Run 1 and balanced Run 2.
- Key line: "Balanced sampling slightly improved AUC and accuracy while changing the FN/FP trade-off."

### Slide 8 - Confusion Matrix Heatmaps
- Compare Run 1 vs Run 2 side by side.
- Key line: "Run 2 reduced false positives but increased false negatives, improving specificity for real videos."

### Slide 9 - Per-Method Accuracy
- Show bar chart for Face2Face, Deepfakes, FaceSwap, NeuralTextures, and Original.
- Key line: "The model is strongest on identity-swap methods and weakest on original class accuracy due to bias toward fake prediction."

### Slide 10 - Error Analysis
- Typical false positives: real videos flagged fake under compression/noise.
- Typical false negatives: subtle manipulations with weak visual artifacts.
- Mention audio limitation in FF++: audio is not manipulated.

### Slide 11 - Contributions
- Reproducible supervised multimodal pipeline.
- Reported per-method behavior and confusion-level analysis.
- Balanced training variant with measurable improvement.

### Slide 12 - Future Work
- Cross-dataset validation on FakeAVCeleb/DFDC.
- Cost-sensitive loss and calibration for real-class specificity.
- Extended visualizations: ROC, PR, calibration curves, feature attention maps.

## 3. Expected Questions and Short Answers

1. Why not use only video?
   Answer: Video is strongest, but multimodal adds complementary cues and improves metrics.
2. Why did original class accuracy remain low?
   Answer: Class imbalance and decision boundary bias increased false positives.
3. Why no cross-dataset result in the paper?
   Answer: Dataset acquisition was incomplete on the training machine at evaluation time.
4. Is this deployable?
   Answer: Yes, with a video-only fallback for resource-constrained inference.

## 4. Presenter Checklist

- Verify all values in slides match the paper tables.
- Keep one message per slide.
- Highlight both strengths and limitations.
- Do not overclaim generalization beyond evaluated datasets.

## 5. File Outputs To Include In Slides

- figures/metric_comparison.png
- figures/confusion_matrix_standard.png
- figures/confusion_matrix_balanced.png
- figures/per_method_accuracy.png

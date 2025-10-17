# MedCLIP Indiana Dataset Training - Complete Workflow

## Visual Workflow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    INDIANA DATASET (INPUT)                       │
│                                                                  │
│  Location: kaggle\input\        │
│                                                                  │
│  📄 indiana_projections.csv  (7,468 entries)                    │
│     - uid, filename, projection                                 │
│                                                                  │
│  📄 indiana_reports.csv      (3,853 reports)                    │
│     - uid, findings, impression, MeSH, Problems                 │
│                                                                  │
│  📁 images/images_normalized/ (~7,400 PNG files)                │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            │ inspect_indiana_dataset.py (Optional)
                            │ ↓ Shows statistics & estimates
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                   PREPROCESSING STEP                             │
│                                                                  │
│  Script: preprocess_indiana.py                                  │
│                                                                  │
│  Operations:                                                     │
│  1. Load CSVs                                                    │
│  2. Merge on uid                                                 │
│  3. Filter → Frontal views only                                 │
│  4. Extract labels from "Problems" field                        │
│  5. Map to CheXpert 14-class labels                             │
│  6. Combine findings + impression → report                      │
│  7. Validate image paths exist                                  │
│  8. Split 80/20 → train/val                                     │
│  9. Save to CSV                                                  │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    PROCESSED DATA (OUTPUT)                       │
│                                                                  │
│  Location: \MedCLIP\local_data\         │
│                                                                  │
│  📄 indiana-train-meta.csv  (~2,400 samples)                    │
│     Columns: imgpath, subject_id, report, [14 labels]          │
│                                                                  │
│  📄 indiana-val-meta.csv    (~600 samples)                      │
│     Columns: imgpath, subject_id, report, [14 labels]          │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            │ test_indiana_dataset.py (Optional)
                            │ ↓ Validates preprocessing
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                      TRAINING STEP                               │
│                                                                  │
│  Script: examples/run_indiana_pretrain.py                       │
│                                                                  │
│  Training Loop:                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  For each epoch:                                           │ │
│  │    For each batch:                                         │ │
│  │      1. Load images & reports                              │ │
│  │      2. Apply data augmentation                            │ │
│  │      3. Encode images (ViT)                                │ │
│  │      4. Encode text (BioClinicalBERT)                      │ │
│  │      5. Compute contrastive loss                           │ │
│  │      6. Backpropagation & update weights                   │ │
│  │                                                             │ │
│  │    Every 500 steps:                                        │ │
│  │      - Evaluate on validation set                          │ │
│  │      - Save checkpoint                                     │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Configuration:                                                  │
│    • Batch size: 32                                             │
│    • Epochs: 20                                                 │
│    • Learning rate: 2e-5                                        │
│    • Optimizer: AdamW                                           │
│    • Loss: InfoNCE (contrastive)                               │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                  TRAINED MODEL (OUTPUT)                          │
│                                                                  │
│  Location: checkpoints/indiana_vision_text_pretrain/            │
│                                                                  │
│  📁 Checkpoint files:                                            │
│     - pytorch_model.bin (model weights)                         │
│     - config.json (model configuration)                         │
│     - Training logs & metrics                                   │
│                                                                  │
│  Ready for:                                                      │
│    ✓ Zero-shot classification                                   │
│    ✓ Fine-tuning on downstream tasks                            │
│    ✓ Feature extraction                                         │
│    ✓ Image-text retrieval                                       │
└──────────────────────────────────────────────────────────────────┘
```

## Data Flow Detail

### Label Mapping Process

```
Indiana "Problems" Field → CheXpert 14 Classes

Input Example: "Cardiomegaly;Pulmonary Edema"
                    ↓
            Parse & Split
                    ↓
         ["Cardiomegaly", "Pulmonary Edema"]
                    ↓
         Map to CheXpert Labels
                    ↓
Output: {
  "No Finding": 0,
  "Cardiomegaly": 1,          ← Mapped
  "Edema": 1,                  ← Mapped from "Pulmonary Edema"
  "Pleural Effusion": 0,
  ... (10 more labels)
}
```

### Report Creation Process

```
Indiana Report Components → MedCLIP Report

Input:
  findings: "The cardiac silhouette is enlarged..."
  impression: "Cardiomegaly. No acute findings."
                    ↓
            Combine with markers
                    ↓
Output:
  report: "FINDINGS: The cardiac silhouette is 
          enlarged... IMPRESSION: Cardiomegaly. 
          No acute findings."
```

### Training Data Augmentation

```
Input Image (Chest X-ray PNG)
        ↓
┌───────────────────┐
│ Random Flip (50%) │ ← Horizontal flip
└────────┬──────────┘
         ↓
┌───────────────────┐
│ Color Jitter      │ ← Brightness/Contrast ±20%
└────────┬──────────┘
         ↓
┌───────────────────┐
│ Random Affine     │ ← Rotation ±10°, Scale 0.8-1.1
└────────┬──────────┘
         ↓
┌───────────────────┐
│ Resize to 256×256 │
└────────┬──────────┘
         ↓
┌───────────────────┐
│ Random Crop 224×224│
└────────┬──────────┘
         ↓
┌───────────────────┐
│ Normalize         │ ← Mean=0.586, Std=0.280
└────────┬──────────┘
         ↓
    Tensor Ready for Model
```

## Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MedCLIP MODEL                          │
│                                                             │
│  ┌────────────────────┐         ┌─────────────────────┐   │
│  │   Image Input      │         │   Text Input        │   │
│  │   (224×224×3)      │         │   (Tokenized)       │   │
│  └─────────┬──────────┘         └──────────┬──────────┘   │
│            │                               │              │
│            ▼                               ▼              │
│  ┌────────────────────┐         ┌─────────────────────┐   │
│  │  Vision Encoder    │         │   Text Encoder      │   │
│  │  (Swin ViT)        │         │  (BioClinicalBERT)  │   │
│  │                    │         │                     │   │
│  │  Output: 768-dim   │         │  Output: 768-dim    │   │
│  └─────────┬──────────┘         └──────────┬──────────┘   │
│            │                               │              │
│            └───────────┬───────────────────┘              │
│                        │                                  │
│                        ▼                                  │
│              ┌─────────────────────┐                      │
│              │  Contrastive Loss   │                      │
│              │     (InfoNCE)       │                      │
│              │                     │                      │
│              │  Maximize similarity│                      │
│              │  for matched pairs  │                      │
│              └─────────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

## File Dependencies

```
run_indiana_pretrain.py
├── Imports from medclip/
│   ├── modeling_medclip.py    (MedCLIPModel, PromptClassifier)
│   ├── dataset.py             (ImageTextContrastiveDataset)
│   ├── losses.py              (ImageTextContrastiveLoss)
│   ├── trainer.py             (Trainer)
│   ├── evaluator.py           (Evaluator)
│   ├── constants.py           (CHEXPERT_TASKS, IMG_SIZE, etc.)
│   └── prompts.py             (generate_chexpert_class_prompts)
│
├── Reads data from
│   └── local_data/
│       ├── indiana-train-meta.csv
│       ├── indiana-val-meta.csv
│       └── sentence-label.csv  (pre-existing)
│
└── Writes to
    └── checkpoints/indiana_vision_text_pretrain/
        ├── pytorch_model.bin
        ├── config.json
        └── training logs
```

## Timeline & Resource Usage

```
Step                   | Time        | Disk Space | RAM    | GPU
-----------------------|-------------|------------|--------|--------
1. Inspection          | ~30 sec     | 0 MB       | <1 GB  | No
2. Preprocessing       | ~1-2 min    | ~50 MB     | ~2 GB  | No
3. Verification        | ~30 sec     | 0 MB       | ~2 GB  | No
4. Training (GPU)      | ~2-4 hrs    | ~500 MB    | ~8 GB  | Yes
4. Training (CPU)      | ~12-24 hrs  | ~500 MB    | ~8 GB  | No
```

## Success Criteria

After each step, you should see:

1. **After Preprocessing**:
   ```
   ✓ Saved training data to: local_data/indiana-train-meta.csv
   ✓ Saved validation data to: local_data/indiana-val-meta.csv
   ```

2. **After Verification**:
   ```
   ✓ All tests passed! You're ready to train.
   ```

3. **During Training**:
   ```
   Epoch 1/20, Step 100, Loss: 4.523
   Epoch 1/20, Step 200, Loss: 3.891
   ...
   Evaluation - Accuracy: 0.652
   ✓ Checkpoint saved
   ```

4. **After Training**:
   ```
   Training complete!
   Model saved to: checkpoints/indiana_vision_text_pretrain/
   ```

---

This diagram provides a complete visual reference for the entire training pipeline!

# MedCLIP Indiana Dataset Integration - Summary

## Overview
I've modified the MedCLIP repository to enable training with the Indiana University Chest X-ray dataset (Open-I) located in your kaggle folder.

## Files Created

### 1. `preprocess_indiana.py` (Root directory)
**Purpose**: Converts Indiana dataset to MedCLIP-compatible format

**What it does**:
- Loads `indiana_projections.csv` and `indiana_reports.csv`
- Filters to frontal views only (PA/AP projections)
- Maps Indiana MeSH terms/Problems to CheXpert 14-class labels
- Combines "findings" and "impression" fields into full reports
- Splits data into 80% train / 20% validation
- Saves to `local_data/indiana-train-meta.csv` and `local_data/indiana-val-meta.csv`

**Key Features**:
- Intelligent label mapping (e.g., "Cardiomegaly" → "Cardiomegaly", "normal" → "No Finding")
- Handles missing data gracefully
- Validates that image files exist
- Prints detailed statistics about label distribution

### 2. `examples/run_indiana_pretrain.py`
**Purpose**: Training script optimized for Indiana dataset

**Key Configuration**:
```python
batch_size: 32        # Smaller for smaller dataset
num_epochs: 20        # More epochs for convergence
eval_steps: 500       # Frequent evaluation
```

**Features**:
- Uses Vision Transformer (ViT) backbone
- Applies data augmentation (flip, jitter, affine, crop)
- Evaluates on Indiana validation set
- Saves checkpoints to `./checkpoints/indiana_vision_text_pretrain/`
- Supports combining with other datasets (CheXpert, MIMIC-CXR)

### 3. `test_indiana_dataset.py`
**Purpose**: Comprehensive testing/validation script

**Tests Performed**:
1. ✓ CSV files exist
2. ✓ CSV format is correct (columns, structure)
3. ✓ Image files exist on disk
4. ✓ Images can be loaded with PIL
5. ✓ Reports are properly formatted
6. ✓ Label distribution statistics
7. ✓ MedCLIP dataset can load the data

**Usage**: Run after preprocessing to verify everything works

### 4. `INDIANA_TRAINING_GUIDE.md`
**Purpose**: Complete user guide with step-by-step instructions

**Contents**:
- Dataset structure explanation
- Step-by-step preprocessing instructions
- Training configuration details
- Troubleshooting guide
- Label mapping table
- Next steps and best practices

## Workflow

```
┌─────────────────────────────────────────────────┐
│  Indiana Dataset (kaggle folder)                │
│  - indiana_projections.csv                      │
│  - indiana_reports.csv                          │
│  - images/images_normalized/*.png               │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  Step 1: Run preprocess_indiana.py              │
│  - Loads CSVs                                   │
│  - Maps labels to CheXpert format               │
│  - Creates train/val split                      │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  Output: local_data/                            │
│  - indiana-train-meta.csv (~2,400 samples)      │
│  - indiana-val-meta.csv (~600 samples)          │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  Step 2: Run test_indiana_dataset.py (Optional) │
│  - Validates preprocessing                      │
│  - Checks image loading                         │
│  - Verifies MedCLIP compatibility               │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  Step 3: Run examples/run_indiana_pretrain.py   │
│  - Trains MedCLIP model                         │
│  - Saves checkpoints                            │
│  - Evaluates on validation set                  │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  Output: checkpoints/indiana_vision_text_pretrain/│
│  - Trained model weights                        │
│  - Ready for inference or fine-tuning           │
└─────────────────────────────────────────────────┘
```

## Label Mapping

The preprocessing maps Indiana labels to CheXpert's 14 classes:

| Indiana Term | → | CheXpert Label |
|--------------|---|----------------|
| normal | → | No Finding |
| Cardiomegaly | → | Cardiomegaly |
| Pulmonary Edema | → | Edema |
| Pleural Effusion | → | Pleural Effusion |
| Pneumonia | → | Pneumonia |
| Atelectasis | → | Atelectasis |
| Pneumothorax | → | Pneumothorax |
| Consolidation | → | Consolidation |
| Opacity | → | Lung Opacity |
| Fracture | → | Fracture |

## No Modifications to Existing Code

**Important**: The existing MedCLIP code (`medclip/dataset.py`, `medclip/modeling_medclip.py`, etc.) **does NOT need to be modified**. The dataset loading logic already supports custom datasets via CSV files in the `local_data/` directory.

The system works by:
1. Creating CSV files with the expected format (imgpath, subject_id, report, + 14 label columns)
2. Placing them in `local_data/` with the naming pattern `{dataset_name}-meta.csv`
3. Using the dataset name in the `datalist` parameter

## Quick Start Commands

```powershell
# Navigate to MedCLIP directory
cd \MedCLIP

# Step 1: Preprocess the Indiana dataset
python preprocess_indiana.py

# Step 2 (Optional): Verify preprocessing
python test_indiana_dataset.py

# Step 3: Train the model
python examples\run_indiana_pretrain.py
```

## Expected Dataset Size

After preprocessing:
- **Training samples**: ~2,400-2,800 (frontal views with reports)
- **Validation samples**: ~600-700
- **Total**: ~3,000-3,500 chest X-rays

This is smaller than MIMIC-CXR (~370K) or CheXpert (~224K), so:
- More epochs recommended (20 vs 10)
- Consider combining with other datasets
- Data augmentation is crucial

## Combining Datasets

To train on multiple datasets, edit `examples/run_indiana_pretrain.py`:

```python
datalist = [
    'indiana-train',
    'chexpert-train',    # If you have it
    'mimic-cxr-train',   # If you have it
]
```

All datasets must have the same 14 CheXpert labels.

## Requirements

All dependencies are already in `requirements.txt`:
- torch
- torchvision
- transformers
- pandas
- numpy
- Pillow
- nltk

## Troubleshooting

### Common Issues:

1. **"No such file or directory: ./local_data/indiana-train-meta.csv"**
   - Solution: Run `preprocess_indiana.py` first

2. **Image files not found**
   - Solution: Verify images exist at:
     `kaggle\input\images\images_normalized\`

3. **CUDA out of memory**
   - Solution: Reduce `batch_size` in training config

4. **Slow training**
   - Normal for CPU training
   - Use GPU if available
   - Reduce `num_workers` in DataLoader

## Next Steps

After successful training:
1. **Zero-shot evaluation**: Test on other datasets
2. **Fine-tuning**: Adapt to specific downstream tasks
3. **Inference**: Use for chest X-ray classification
4. **Analysis**: Examine learned representations

## Files Summary

```
MedCLIP/
├── preprocess_indiana.py              # NEW: Dataset preprocessing
├── test_indiana_dataset.py            # NEW: Verification script
├── INDIANA_TRAINING_GUIDE.md          # NEW: User guide
├── CHANGES_SUMMARY.md                 # NEW: This file
├── examples/
│   ├── run_medclip_pretrain.py       # Original (unchanged)
│   └── run_indiana_pretrain.py       # NEW: Indiana training script
├── medclip/                           # Original (unchanged)
└── local_data/                        # Generated by preprocessing
    ├── indiana-train-meta.csv         # Generated
    └── indiana-val-meta.csv           # Generated
```

## Architecture

The training uses:
- **Vision Encoder**: Vision Transformer (ViT) via Swin Transformer
- **Text Encoder**: BioClinicalBERT
- **Training Method**: Contrastive learning (image-text pairs)
- **Loss Function**: InfoNCE (normalized temperature-scaled cross entropy)

## Performance Expectations

With ~3,000 training samples:
- Training time: 2-4 hours on GPU, longer on CPU
- Expected validation accuracy: 60-75% (depends on label)
- Checkpoints saved every 500 steps
- Best model based on validation performance

For better performance:
- Combine with larger datasets (CheXpert, MIMIC-CXR)
- Use pre-trained weights and fine-tune
- Increase epochs and tune hyperparameters

---

**Author**: GitHub Copilot  
**Date**: October 17, 2025  
**Version**: 1.0

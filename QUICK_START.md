# Quick Start Guide: Training MedCLIP with Indiana Dataset

This guide will help you train MedCLIP using the Indiana University Chest X-ray dataset in 3 simple steps.

## Prerequisites

✓ Indiana dataset at: `\kaggle\input\`  
✓ Python environment with dependencies installed (see `requirements.txt`)

## Step 1: Inspect the Dataset (Optional)

See what's in your dataset before processing:

```powershell
python inspect_indiana_dataset.py
```

This shows:
- Number of images and reports
- Projection types (Frontal vs Lateral)
- Common findings
- Estimated output size

## Step 2: Preprocess the Dataset

Convert Indiana dataset to MedCLIP format:

```powershell
python preprocess_indiana.py
```

**Expected output:**
- `local_data/indiana-train-meta.csv` (~2,400 samples)
- `local_data/indiana-val-meta.csv` (~600 samples)

**What it does:**
- ✓ Filters to frontal views only
- ✓ Maps labels to CheXpert 14-class format
- ✓ Combines findings + impressions
- ✓ Splits into train/validation sets

## Step 3: Verify Processing (Optional but Recommended)

Check that everything worked correctly:

```powershell
python test_indiana_dataset.py
```

This runs 7 tests:
1. CSV files exist
2. CSV format is correct
3. Images exist on disk
4. Images can be loaded
5. Reports are formatted properly
6. Labels are distributed correctly
7. MedCLIP can load the dataset

## Step 4: Train the Model

Start training:

```powershell
python examples\run_indiana_pretrain.py
```

**Training details:**
- Batch size: 32
- Epochs: 20
- Evaluation every 500 steps
- Checkpoints saved to: `checkpoints/indiana_vision_text_pretrain/`

**Monitor progress:**
- Loss decreases over time
- Validation metrics improve
- Checkpoints saved periodically

## That's It! 🎉

After training completes, you'll have a trained MedCLIP model ready for:
- Zero-shot classification
- Fine-tuning on downstream tasks
- Feature extraction
- Medical image-text retrieval

## Need More Details?

- **Full guide**: See `INDIANA_TRAINING_GUIDE.md`
- **All changes**: See `CHANGES_SUMMARY.md`
- **Troubleshooting**: See the Troubleshooting section in `INDIANA_TRAINING_GUIDE.md`

## File Overview

```
📁 MedCLIP/
├── 🚀 inspect_indiana_dataset.py      # Step 1: Inspect
├── ⚙️  preprocess_indiana.py          # Step 2: Preprocess
├── ✅ test_indiana_dataset.py         # Step 3: Verify
├── 🎯 examples/run_indiana_pretrain.py # Step 4: Train
├── 📖 INDIANA_TRAINING_GUIDE.md       # Detailed guide
├── 📋 CHANGES_SUMMARY.md              # Technical details
└── 📄 QUICK_START.md                  # This file
```

## Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| CSV not found | Run `preprocess_indiana.py` first |
| Images not found | Check path: `kaggle\input\images\images_normalized\` |
| Out of memory | Reduce `batch_size` in `run_indiana_pretrain.py` |
| Import errors | Install dependencies: `pip install -r requirements.txt` |

## Expected Timeline

- ⏱️ Preprocessing: ~1-2 minutes
- ⏱️ Verification: ~30 seconds
- ⏱️ Training: 2-4 hours (GPU) or 12-24 hours (CPU)

## Training Configuration

Default settings (in `run_indiana_pretrain.py`):

```python
batch_size: 32          # Smaller for smaller dataset
num_epochs: 20          # More epochs for convergence
learning_rate: 2e-5     # Standard for BERT-based models
eval_steps: 500         # Evaluate frequently
```

Adjust these based on your hardware and needs!

---

**Need help?** Check the detailed guides or review the error messages carefully - they usually point to the solution! 🔍

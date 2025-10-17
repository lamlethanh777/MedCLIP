"""
Test script to verify Indiana dataset preprocessing and loading
Run this after preprocess_indiana.py to check everything works
"""
import os
import pandas as pd
import numpy as np
from PIL import Image

# Paths
LOCAL_DATA_DIR = r'/kaggle/input/chest-xrays-indiana-university'
TRAIN_CSV = os.path.join(LOCAL_DATA_DIR, 'indiana-train-meta.csv')
VAL_CSV = os.path.join(LOCAL_DATA_DIR, 'indiana-val-meta.csv')

CHEXPERT_LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion',
    'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]

def test_csv_exists():
    """Test if CSV files exist"""
    print("="*60)
    print("Test 1: Checking if CSV files exist")
    print("="*60)
    
    train_exists = os.path.exists(TRAIN_CSV)
    val_exists = os.path.exists(VAL_CSV)
    
    print(f"Training CSV exists: {train_exists}")
    print(f"Validation CSV exists: {val_exists}")
    
    if not train_exists or not val_exists:
        print("\n❌ FAILED: CSV files not found!")
        print("Please run preprocess_indiana.py first.")
        return False
    
    print("✓ PASSED\n")
    return True

def test_csv_format():
    """Test if CSV has correct format"""
    print("="*60)
    print("Test 2: Checking CSV format")
    print("="*60)
    
    train_df = pd.read_csv(TRAIN_CSV, index_col=0)
    val_df = pd.read_csv(VAL_CSV, index_col=0)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Check required columns
    required_cols = ['imgpath', 'subject_id', 'report'] + CHEXPERT_LABELS
    train_cols = train_df.columns.tolist()
    
    missing_cols = [col for col in required_cols if col not in train_cols]
    
    if missing_cols:
        print(f"\n❌ FAILED: Missing columns: {missing_cols}")
        return False
    
    print(f"\nColumns: {train_cols[:5]}... (showing first 5)")
    print(f"Total columns: {len(train_cols)}")
    print("✓ PASSED\n")
    return True

def test_images_exist():
    """Test if image files exist"""
    print("="*60)
    print("Test 3: Checking if images exist")
    print("="*60)
    
    train_df = pd.read_csv(TRAIN_CSV, index_col=0)
    
    # Check first 10 images
    n_check = min(10, len(train_df))
    print(f"Checking first {n_check} images...")
    
    missing_count = 0
    for i in range(n_check):
        imgpath = train_df.iloc[i]['imgpath']
        if not os.path.exists(imgpath):
            print(f"  ❌ Missing: {imgpath}")
            missing_count += 1
        else:
            print(f"  ✓ Found: {os.path.basename(imgpath)}")
    
    if missing_count > 0:
        print(f"\n❌ FAILED: {missing_count}/{n_check} images not found")
        return False
    
    print("✓ PASSED\n")
    return True

def test_image_loading():
    """Test if images can be loaded"""
    print("="*60)
    print("Test 4: Testing image loading")
    print("="*60)
    
    train_df = pd.read_csv(TRAIN_CSV, index_col=0)
    
    # Try loading first image
    imgpath = train_df.iloc[0]['imgpath']
    print(f"Loading: {imgpath}")
    
    try:
        img = Image.open(imgpath)
        print(f"  Image size: {img.size}")
        print(f"  Image mode: {img.mode}")
        print("✓ PASSED\n")
        return True
    except Exception as e:
        print(f"❌ FAILED: Could not load image: {e}")
        return False

def test_report_text():
    """Test if reports are properly formatted"""
    print("="*60)
    print("Test 5: Checking report text")
    print("="*60)
    
    train_df = pd.read_csv(TRAIN_CSV, index_col=0)
    
    # Check first report
    report = train_df.iloc[0]['report']
    print(f"Sample report (first 200 chars):")
    print(f"  {report[:200]}...")
    
    # Check for null reports
    null_reports = train_df['report'].isnull().sum()
    print(f"\nNull reports: {null_reports}/{len(train_df)}")
    
    if null_reports > 0:
        print(f"⚠️  WARNING: {null_reports} reports are null")
    
    print("✓ PASSED\n")
    return True

def test_labels():
    """Test label distribution"""
    print("="*60)
    print("Test 6: Checking label distribution")
    print("="*60)
    
    train_df = pd.read_csv(TRAIN_CSV, index_col=0)
    
    print("Training set label distribution:")
    for label in CHEXPERT_LABELS:
        count = train_df[label].sum()
        percentage = (count / len(train_df)) * 100
        print(f"  {label:30s}: {count:4d} ({percentage:5.2f}%)")
    
    print("✓ PASSED\n")
    return True

def test_medclip_dataset():
    """Test if MedCLIP dataset can load the data"""
    print("="*60)
    print("Test 7: Testing MedCLIP dataset loading")
    print("="*60)
    
    try:
        from medclip.dataset import ImageTextContrastiveDataset
        from torchvision import transforms
        from medclip import constants
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((constants.IMG_SIZE, constants.IMG_SIZE)),
            transforms.Normalize(mean=[constants.IMG_MEAN], std=[constants.IMG_STD])
        ])
        
        print("Creating dataset...")
        dataset = ImageTextContrastiveDataset(
            datalist=['indiana-train'],
            imgtransform=transform
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        print("\nLoading first sample...")
        sample = dataset[0]
        print(f"  Image shape: {sample[0].shape}")
        print(f"  Report length: {len(sample[1])} characters")
        print(f"  Image labels shape: {sample[2].shape}")
        print(f"  Text labels shape: {sample[3].shape}")
        
        print("✓ PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("INDIANA DATASET VERIFICATION")
    print("="*60 + "\n")
    
    tests = [
        test_csv_exists,
        test_csv_format,
        test_images_exist,
        test_image_loading,
        test_report_text,
        test_labels,
        test_medclip_dataset,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ All tests passed! You're ready to train.")
    else:
        print(f"\n❌ {failed} test(s) failed. Please fix the issues above.")

if __name__ == '__main__':
    main()

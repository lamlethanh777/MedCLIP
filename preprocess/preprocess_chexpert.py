"""
Preprocess CheXpert Dataset
This script samples images from CheXpert dataset to create a 5x200 dataset with 5 classes.
Each class has 200 records where only that specific pathology is positive (1.0).
"""

import os
import shutil
import pandas as pd
import random
from pathlib import Path


SAMPLES_PER_CLASS = 200
TARGET_CLASSES = ['Atelectasis', 'Cardiomegaly', 'Edema', 'Pleural Effusion', 'Pneumonia']

# Paths
KAGGLE_SOURCE = "/kaggle/input/chexpert/"
LOCAL_SOURCE = "../../datasets/CheXpert-v1.0-small/archive/"
TRAIN_CSV = "train.csv"
VALID_CSV = "valid.csv"
TEST_DEST = "data/CheXpert/test"
CSV_PATH = "local_data/chexpert-test-meta.csv"


def load_and_filter_chexpert(csv_path, target_classes, samples_per_class):
    """
    Load CheXpert CSV and filter for single-pathology cases.
    
    Args:
        csv_path: Path to CheXpert CSV file
        target_classes: List of target pathology classes
        samples_per_class: Number of samples per class
        
    Returns:
        DataFrame with sampled data
    """
    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total records in CSV: {len(df)}")
    
    # Initialize list to store sampled dataframes
    sampled_dfs = []
    
    # Sample for each target class
    for target_class in target_classes:
        print(f"\n  Sampling {target_class}...")
        
        # Create condition: target class = 1.0
        condition = (df[target_class] == 1.0)
        
        # All other target classes should be 0.0 or NaN
        for col in target_classes:
            if col != target_class:
                condition &= ((df[col] == 0.0) | (df[col].isna()))
        
        # Filter dataframe based on condition
        class_df = df[condition].copy()
        
        print(f"    Found {len(class_df)} records with only {target_class}")
        
        # Sample the required number of records
        if len(class_df) >= samples_per_class:
            sampled_class_df = class_df.sample(n=samples_per_class, random_state=42)
            print(f"    ✓ Sampled {samples_per_class} records")
        else:
            sampled_class_df = class_df
            print(f"    ⚠ WARNING: Only {len(class_df)} records available (less than {samples_per_class})")
        
        sampled_dfs.append(sampled_class_df)
    
    # Combine all sampled dataframes
    final_df = pd.concat(sampled_dfs, ignore_index=True)

    # Modify path of each record (replace "CheXpert-v1.0-small/" by "")
    final_df['Path'] = final_df['Path'].str.replace('CheXpert-v1.0-small/', '', regex=False)
    
    # Shuffle the final dataset
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return final_df


def copy_images_and_prepare_metadata(df, source_base_dir, dest_dir):
    """
    Copy sampled images to destination and prepare metadata.
    
    Args:
        df: DataFrame with sampled records
        source_base_dir: Base directory containing CheXpert images
        dest_dir: Destination directory for copied images
        
    Returns:
        List of metadata records
    """
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    metadata = []
    copied_count = 0
    missing_count = 0
    
    print(f"\nCopying {len(df)} images to {dest_dir}...")
    
    for idx, row in df.iterrows():
        # Get source image path (relative path from CSV)
        relative_path = row['Path']
        source_path = os.path.join(source_base_dir, relative_path)
        
        # Check if source file exists
        if not os.path.exists(source_path):
            print(f"    ⚠ Warning: Image not found: {source_path}")
            missing_count += 1
            continue
        
        # Create destination path (keep the same structure)
        dest_path = os.path.join(dest_dir, os.path.basename(relative_path))
        
        # Copy image
        shutil.copy2(source_path, dest_path)
        copied_count += 1
        
        # Prepare metadata row (keep all relevant columns)
        metadata_row = {
            'imgpath': dest_path,
            'Sex': row['Sex'],
            'Age': row['Age'],
            'Frontal/Lateral': row['Frontal/Lateral'],
            'AP/PA': row['AP/PA']
        }
        
        # Add pathology labels
        for class_name in TARGET_CLASSES:
            metadata_row[class_name] = row[class_name]
        
        metadata.append(metadata_row)
        
        # Print progress every 100 images
        if (copied_count % 100) == 0:
            print(f"    Copied {copied_count}/{len(df)} images...")
    
    print(f"\n  ✓ Successfully copied {copied_count} images")
    if missing_count > 0:
        print(f"  ⚠ {missing_count} images were not found")
    
    return metadata


def create_metadata_csv(metadata_list, csv_path, target_classes):
    """
    Create CSV file with metadata.
    
    Args:
        metadata_list: List of metadata dictionaries
        csv_path: Path to output CSV file
        target_classes: List of target pathology classes
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(metadata_list)
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    
    print(f"\nMetadata CSV created at: {csv_path}")
    print(f"Total entries: {len(df)}")
    
    # Print class distribution
    print("\nClass distribution:")
    for class_name in target_classes:
        count = (df[class_name] == 1.0).sum()
        print(f"  {class_name}: {count}")


def main():
    """Main function to preprocess CheXpert dataset."""
    print("=" * 70)
    print("CheXpert Dataset Preprocessing")
    print("=" * 70)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    print(f"\nConfiguration:")
    print(f"  Target classes: {', '.join(TARGET_CLASSES)}")
    print(f"  Samples per class: {SAMPLES_PER_CLASS}")
    print(f"  Total expected samples: {len(TARGET_CLASSES) * SAMPLES_PER_CLASS}")
    
    # Determine if running on Kaggle or locally
    if os.path.exists(KAGGLE_SOURCE):
        print(f"\n✓ Running on Kaggle")
        base_dir = KAGGLE_SOURCE
        csv_file = KAGGLE_SOURCE + TRAIN_CSV
    else:
        print(f"\n✓ Running locally")
        base_dir = LOCAL_SOURCE
        csv_file = LOCAL_SOURCE + TRAIN_CSV
    
    # Clear test directory if it exists
    if os.path.exists(TEST_DEST):
        print(f"\nClearing existing test directory: {TEST_DEST}")
        shutil.rmtree(TEST_DEST)
    
    print("\n" + "-" * 70)
    print("Step 1: Loading and filtering CheXpert data")
    print("-" * 70)
    
    sampled_df = load_and_filter_chexpert(
        csv_file,
        TARGET_CLASSES,
        SAMPLES_PER_CLASS
    )
    
    print("\n" + "-" * 70)
    print("Step 2: Copying images and preparing metadata")
    print("-" * 70)
    
    metadata = copy_images_and_prepare_metadata(
        sampled_df,
        base_dir,
        TEST_DEST
    )
    
    print("\n" + "-" * 70)
    print("Step 3: Creating metadata CSV")
    print("-" * 70)
    
    create_metadata_csv(metadata, CSV_PATH, TARGET_CLASSES)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total images processed: {len(metadata)}")
    print(f"Destination folder: {TEST_DEST}")
    print(f"Metadata CSV: {CSV_PATH}")
    print("\n✓ Preprocessing completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
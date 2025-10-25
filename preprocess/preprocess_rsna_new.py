"""
Preprocess RSNA Pneumonia Detection Dataset
This script samples PNG images from processed RSNA dataset, copies them to a test folder,
and creates a metadata CSV file.
"""

import os
import shutil
import random
import csv
import pandas as pd
from PIL import Image
from pathlib import Path


# Configuration - YOU CAN CHANGE THESE NUMBERS
NUM_PNEUMONIA_SAMPLES = 2000
NUM_NORMAL_SAMPLES = 2000

# Paths for processed dataset
KAGGLE_SOURCE = "/kaggle/input/rsna-pneumonia-processed-dataset/"
LOCAL_SOURCE = "../../datasets/RSNA-processed/"
TRAIN_IMAGES_DIR = "Training/Images"
TRAIN_METADATA_CSV = "/stage2_train_metadata.csv"
TEST_DEST = "data/RSNA/test"
CSV_PATH = "local_data/rsna-test-meta.csv"


def load_rsna_metadata(csv_path):
    """
    Load RSNA metadata from processed CSV file.
    
    Args:
        csv_path: Path to metadata CSV file
        
    Returns:
        Dictionary mapping image filename to pneumonia label (0=normal, 1=pneumonia)
    """
    df = pd.read_csv(csv_path)
    
    # Create mapping from image filename to pneumonia label
    # Assuming the CSV has columns like 'filename' and 'pneumonia' or similar
    # You may need to adjust column names based on your actual CSV structure
    
    print(f"CSV columns: {df.columns.tolist()}")
    
    # Try different possible column name combinations
    if 'filename' in df.columns and 'pneumonia' in df.columns:
        image_labels = dict(zip(df['filename'], df['pneumonia']))
    elif 'image_id' in df.columns and 'target' in df.columns:
        image_labels = dict(zip(df['image_id'], df['target']))
    elif 'patientId' in df.columns and 'Target' in df.columns:
        # If still using original RSNA format
        image_labels = {f"{pid}.png": target for pid, target in zip(df['patientId'], df['Target'])}
    else:
        # Print first few rows to help identify correct columns
        print("First 5 rows of metadata:")
        print(df.head())
        raise ValueError("Could not identify filename and label columns. Please check CSV structure.")
    
    return image_labels


def sample_and_copy_images(source_dir, dest_dir, image_labels, num_samples, target_label):
    """
    Sample random images with specific target label and copy to destination.
    
    Args:
        source_dir: Source directory containing PNG images
        dest_dir: Destination directory for copied images
        image_labels: Dictionary mapping image filename to label
        num_samples: Number of images to sample
        target_label: Target label to filter (0=normal, 1=pneumonia)
        
    Returns:
        List of tuples (patient_id, image_path, pneumonia_label)
    """
    # Filter images by target label
    target_images = [filename for filename, label in image_labels.items() if label == target_label]
    
    # Check if we have enough images
    if len(target_images) < num_samples:
        print(f"Warning: Only {len(target_images)} images with target={target_label} available, but {num_samples} requested.")
        print(f"Using all {len(target_images)} images instead.")
        num_samples = len(target_images)
    
    # Randomly sample images
    sampled_images = random.sample(target_images, num_samples)
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Copy images
    metadata = []
    successful_copies = 0
    
    for filename in sampled_images:
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(dest_dir, filename)
        
        # Check if PNG file exists
        if not os.path.exists(source_path):
            print(f"Warning: PNG file not found: {source_path}")
            continue
        
        try:
            # Copy PNG file
            shutil.copy2(source_path, dest_path)
            
            # Extract patient ID from filename (remove .png extension)
            patient_id = filename.replace('.png', '')
            metadata.append((patient_id, dest_path, target_label))
            successful_copies += 1
            
        except Exception as e:
            print(f"Failed to copy {filename}: {str(e)}")
    
    print(f"Successfully copied {successful_copies}/{len(sampled_images)} images")
    return metadata


def create_metadata_csv(metadata_list, csv_path):
    """
    Create CSV file with metadata.
    
    Args:
        metadata_list: List of tuples (patient_id, imgpath, pneumonia)
        csv_path: Path to output CSV file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Write CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['patient_id', 'imgpath', 'pneumonia'])
        
        # Write data
        for patient_id, imgpath, pneumonia in metadata_list:
            writer.writerow([patient_id, imgpath, pneumonia])
    
    print(f"\nMetadata CSV created at: {csv_path}")
    print(f"Total entries: {len(metadata_list)}")


def main():
    """Main function to preprocess RSNA pneumonia dataset."""
    print("=" * 70)
    print("RSNA Pneumonia Detection Dataset Preprocessing (Processed Version)")
    print("=" * 70)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    print(f"\nConfiguration:")
    print(f"  Pneumonia samples: {NUM_PNEUMONIA_SAMPLES}")
    print(f"  Normal samples: {NUM_NORMAL_SAMPLES}")
    print(f"  Total samples: {NUM_PNEUMONIA_SAMPLES + NUM_NORMAL_SAMPLES}")
    
    # Load metadata
    print("\n" + "-" * 70)
    print("Loading RSNA metadata...")
    print("-" * 70)
    metadata_path = os.path.join(KAGGLE_SOURCE, TRAIN_METADATA_CSV)
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join(LOCAL_SOURCE, TRAIN_METADATA_CSV)
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata CSV not found at {metadata_path}")
    
    image_labels = load_rsna_metadata(metadata_path)
    print(f"✓ Loaded metadata for {len(image_labels)} images")
    
    # Count labels
    pneumonia_count = sum(1 for label in image_labels.values() if label == 1)
    normal_count = sum(1 for label in image_labels.values() if label == 0)
    print(f"  Pneumonia cases: {pneumonia_count}")
    print(f"  Normal cases: {normal_count}")
    
    # Determine source directory
    images_dir = os.path.join(KAGGLE_SOURCE, TRAIN_IMAGES_DIR)
    if not os.path.exists(images_dir):
        images_dir = os.path.join(LOCAL_SOURCE, TRAIN_IMAGES_DIR)
    
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found at {images_dir}")
    
    print(f"  Source images directory: {images_dir}")
    
    # Count actual image files
    actual_images = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    print(f"  Actual PNG files found: {len(actual_images)}")
    
    # Clear test directory if it exists
    if os.path.exists(TEST_DEST):
        print(f"\nClearing existing test directory: {TEST_DEST}")
        shutil.rmtree(TEST_DEST)
    
    print("\n" + "-" * 70)
    print("Processing Pneumonia images...")
    print("-" * 70)
    pneumonia_metadata = sample_and_copy_images(
        images_dir, 
        TEST_DEST, 
        image_labels,
        NUM_PNEUMONIA_SAMPLES, 
        target_label=1
    )
    print(f"✓ Copied {len(pneumonia_metadata)} Pneumonia images")
    
    print("\n" + "-" * 70)
    print("Processing Normal images...")
    print("-" * 70)
    normal_metadata = sample_and_copy_images(
        images_dir, 
        TEST_DEST, 
        image_labels,
        NUM_NORMAL_SAMPLES, 
        target_label=0
    )
    print(f"✓ Copied {len(normal_metadata)} Normal images")
    
    # Combine metadata
    all_metadata = pneumonia_metadata + normal_metadata
    
    # Shuffle the combined metadata
    random.shuffle(all_metadata)
    
    # Create CSV file
    print("\n" + "-" * 70)
    print("Creating metadata CSV...")
    print("-" * 70)
    create_metadata_csv(all_metadata, CSV_PATH)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Pneumonia images copied: {len(pneumonia_metadata)}")
    print(f"Normal images copied: {len(normal_metadata)}")
    print(f"Total images: {len(all_metadata)}")
    print(f"Destination folder: {TEST_DEST}")
    print(f"Metadata CSV: {CSV_PATH}")
    print("\n✓ Preprocessing completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
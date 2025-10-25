"""
Preprocess RSNA Pneumonia Detection Dataset
This script samples DICOM images from RSNA dataset, converts them to PNG format,
copies them to a test folder, and creates a metadata CSV file.
"""

import os
import shutil
import random
import csv
import pandas as pd
import pydicom
import numpy as np
from PIL import Image
from pathlib import Path


# Configuration - YOU CAN CHANGE THESE NUMBERS
NUM_PNEUMONIA_SAMPLES = 2000
NUM_NORMAL_SAMPLES = 2000

# Paths
KAGGLE_SOURCE = "/kaggle/input/rsna-pneumonia-detection-challenge/"
LOCAL_SOURCE = "../../datasets/RSNA/"
TRAIN_IMAGES_DIR = "stage_2_train_images"
TRAIN_LABELS_CSV = "stage_2_train_labels.csv"
TEST_DEST = "data/RSNA/test"
CSV_PATH = "local_data/rsna-test-meta.csv"


def convert_dicom_to_png(dicom_path, output_path):
    """
    Convert DICOM file to PNG format.
    
    Args:
        dicom_path: Path to DICOM file
        output_path: Path to save PNG file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read DICOM file
        dicom = pydicom.dcmread(dicom_path)
        
        # Get pixel array
        pixel_array = dicom.pixel_array
        
        # Normalize to 0-255 range
        pixel_array = pixel_array - np.min(pixel_array)
        pixel_array = pixel_array / np.max(pixel_array)
        pixel_array = (pixel_array * 255).astype(np.uint8)
        
        # Convert to PIL Image and save as PNG
        image = Image.fromarray(pixel_array)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image.save(output_path, 'PNG')
        return True
        
    except Exception as e:
        print(f"Error converting {dicom_path}: {str(e)}")
        return False


def load_rsna_labels(csv_path):
    """
    Load RSNA labels from CSV file.
    
    Args:
        csv_path: Path to labels CSV file
        
    Returns:
        Dictionary mapping patientId to target (0=normal, 1=pneumonia)
    """
    df = pd.read_csv(csv_path)
    
    # Group by patientId and get max Target (since some patients have multiple boxes)
    # If any box indicates pneumonia (Target=1), consider the patient as having pneumonia
    patient_labels = df.groupby('patientId')['Target'].max().to_dict()
    
    return patient_labels


def sample_and_convert_images(source_dir, dest_dir, patient_labels, num_samples, target_label):
    """
    Sample random images with specific target label, convert DICOM to PNG, and copy to destination.
    
    Args:
        source_dir: Source directory containing DICOM images
        dest_dir: Destination directory for converted PNG images
        patient_labels: Dictionary mapping patientId to target label
        num_samples: Number of images to sample
        target_label: Target label to filter (0=normal, 1=pneumonia)
        
    Returns:
        List of tuples (patient_id, image_path, pneumonia_label)
    """
    # Filter patients by target label
    target_patients = [pid for pid, label in patient_labels.items() if label == target_label]
    
    # Check if we have enough images
    if len(target_patients) < num_samples:
        print(f"Warning: Only {len(target_patients)} images with target={target_label} available, but {num_samples} requested.")
        print(f"Using all {len(target_patients)} images instead.")
        num_samples = len(target_patients)
    
    # Randomly sample patients
    sampled_patients = random.sample(target_patients, num_samples)
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Convert and copy images
    metadata = []
    successful_conversions = 0
    
    for patient_id in sampled_patients:
        dicom_path = os.path.join(source_dir, f"{patient_id}.dcm")
        png_path = os.path.join(dest_dir, f"{patient_id}.png")
        
        # Check if DICOM file exists
        if not os.path.exists(dicom_path):
            print(f"Warning: DICOM file not found: {dicom_path}")
            continue
        
        # Convert DICOM to PNG
        if convert_dicom_to_png(dicom_path, png_path):
            metadata.append((patient_id, png_path, target_label))
            successful_conversions += 1
        else:
            print(f"Failed to convert: {patient_id}")
    
    print(f"Successfully converted {successful_conversions}/{len(sampled_patients)} images")
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
    print("RSNA Pneumonia Detection Dataset Preprocessing")
    print("=" * 70)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    print(f"\nConfiguration:")
    print(f"  Pneumonia samples: {NUM_PNEUMONIA_SAMPLES}")
    print(f"  Normal samples: {NUM_NORMAL_SAMPLES}")
    print(f"  Total samples: {NUM_PNEUMONIA_SAMPLES + NUM_NORMAL_SAMPLES}")
    
    # Load labels
    print("\n" + "-" * 70)
    print("Loading RSNA labels...")
    print("-" * 70)
    labels_path = os.path.join(KAGGLE_SOURCE, TRAIN_LABELS_CSV)
    if not os.path.exists(labels_path):
        labels_path = os.path.join(LOCAL_SOURCE, TRAIN_LABELS_CSV)
    
    patient_labels = load_rsna_labels(labels_path)
    print(f"✓ Loaded labels for {len(patient_labels)} patients")
    
    # Count labels
    pneumonia_count = sum(1 for label in patient_labels.values() if label == 1)
    normal_count = sum(1 for label in patient_labels.values() if label == 0)
    print(f"  Pneumonia cases: {pneumonia_count}")
    print(f"  Normal cases: {normal_count}")
    
    # Determine source directory
    images_dir = os.path.join(KAGGLE_SOURCE, TRAIN_IMAGES_DIR)
    if not os.path.exists(images_dir):
        images_dir = os.path.join(LOCAL_SOURCE, TRAIN_IMAGES_DIR)
    
    print(f"  Source images directory: {images_dir}")
    
    # Clear test directory if it exists
    if os.path.exists(TEST_DEST):
        print(f"\nClearing existing test directory: {TEST_DEST}")
        shutil.rmtree(TEST_DEST)
    
    print("\n" + "-" * 70)
    print("Processing Pneumonia images...")
    print("-" * 70)
    pneumonia_metadata = sample_and_convert_images(
        images_dir, 
        TEST_DEST, 
        patient_labels,
        NUM_PNEUMONIA_SAMPLES, 
        target_label=1
    )
    print(f"✓ Converted {len(pneumonia_metadata)} Pneumonia images")
    
    print("\n" + "-" * 70)
    print("Processing Normal images...")
    print("-" * 70)
    normal_metadata = sample_and_convert_images(
        images_dir, 
        TEST_DEST, 
        patient_labels,
        NUM_NORMAL_SAMPLES, 
        target_label=0
    )
    print(f"✓ Converted {len(normal_metadata)} Normal images")
    
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
    print(f"Pneumonia images converted: {len(pneumonia_metadata)}")
    print(f"Normal images converted: {len(normal_metadata)}")
    print(f"Total images: {len(all_metadata)}")
    print(f"Destination folder: {TEST_DEST}")
    print(f"Metadata CSV: {CSV_PATH}")
    print("\n✓ Preprocessing completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
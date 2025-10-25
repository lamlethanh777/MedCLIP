"""
Preprocess COVID-19 X-ray Dataset
This script samples images from COVID and Normal folders, copies them to a test folder,
and creates a metadata CSV file.
"""

import os
import shutil
import random
import csv
from pathlib import Path


# Configuration - YOU CAN CHANGE THESE NUMBERS
NUM_COVID_SAMPLES = 3000
NUM_NORMAL_SAMPLES = 3122

# Paths
KAGGLE_SOURCE = "/kaggle/input/covid19-radiography-database/"
LOCAL_SOURCE = "../../datasets/COVID/archive/"
COVID_SOURCE = "COVID-19_Radiography_Dataset/COVID/images"
NORMAL_SOURCE = "COVID-19_Radiography_Dataset/Normal/images"
TEST_DEST = "data/COVID/test"
CSV_PATH = "local_data/covid-test-meta.csv"


def sample_and_copy_images(source_dir, dest_dir, num_samples, is_covid):
    """
    Sample random images from source directory and copy to destination.
    
    Args:
        source_dir: Source directory containing images
        dest_dir: Destination directory for copied images
        num_samples: Number of images to sample
        is_covid: True if COVID images, False if Normal
        
    Returns:
        List of tuples (subject_id, image_path, covid_label, normal_label)
    """
    # Get all image files
    all_images = [f for f in os.listdir(source_dir) if f.endswith('.png')]
    
    # Check if we have enough images
    if len(all_images) < num_samples:
        print(f"Warning: Only {len(all_images)} images available in {source_dir}, but {num_samples} requested.")
        print(f"Using all {len(all_images)} images instead.")
        num_samples = len(all_images)
    
    # Randomly sample images
    sampled_images = random.sample(all_images, num_samples)
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Copy images and prepare metadata
    metadata = []
    covid_label = 1 if is_covid else 0
    normal_label = 0 if is_covid else 1
    
    for img_name in sampled_images:
        source_path = os.path.join(source_dir, img_name)
        dest_path = os.path.join(dest_dir, img_name)
        
        # Copy image
        shutil.copy2(source_path, dest_path)
        
        # Store metadata (relative path from test folder)
        metadata.append((img_name.replace(".png", ""), dest_path, covid_label, normal_label))
    
    return metadata


def create_metadata_csv(metadata_list, csv_path):
    """
    Create CSV file with metadata.
    
    Args:
        metadata_list: List of tuples (imgpath, subject_id, COVID)
        csv_path: Path to output CSV file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Write CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['subject_id', 'imgpath', 'COVID', 'Normal'])
        
        # Write data
        for subject_id, imgpath, covid, normal in metadata_list:
            writer.writerow([subject_id, imgpath, covid, normal])
    
    print(f"\nMetadata CSV created at: {csv_path}")
    print(f"Total entries: {len(metadata_list)}")


def main():
    """Main function to preprocess COVID-19 dataset."""
    print("=" * 70)
    print("COVID-19 X-ray Dataset Preprocessing")
    print("=" * 70)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    print(f"\nConfiguration:")
    print(f"  COVID samples: {NUM_COVID_SAMPLES}")
    print(f"  Normal samples: {NUM_NORMAL_SAMPLES}")
    print(f"  Total samples: {NUM_COVID_SAMPLES + NUM_NORMAL_SAMPLES}")
    
    # Clear test directory if it exists
    if os.path.exists(TEST_DEST):
        print(f"\nClearing existing test directory: {TEST_DEST}")
        shutil.rmtree(TEST_DEST)
    
    print("\n" + "-" * 70)
    print("Processing COVID images...")
    print("-" * 70)
    covid_metadata = sample_and_copy_images(
        KAGGLE_SOURCE + COVID_SOURCE, 
        TEST_DEST, 
        NUM_COVID_SAMPLES, 
        is_covid=True
    )
    print(f"✓ Copied {len(covid_metadata)} COVID images")
    
    print("\n" + "-" * 70)
    print("Processing Normal images...")
    print("-" * 70)
    normal_metadata = sample_and_copy_images(
        KAGGLE_SOURCE + NORMAL_SOURCE, 
        TEST_DEST, 
        NUM_NORMAL_SAMPLES, 
        is_covid=False
    )
    print(f"✓ Copied {len(normal_metadata)} Normal images")
    
    # Combine metadata
    all_metadata = covid_metadata + normal_metadata
    
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
    print(f"COVID images copied: {len(covid_metadata)}")
    print(f"Normal images copied: {len(normal_metadata)}")
    print(f"Total images: {len(all_metadata)}")
    print(f"Destination folder: {TEST_DEST}")
    print(f"Metadata CSV: {CSV_PATH}")
    print("\n✓ Preprocessing completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
Preprocessing script for Indiana University Chest X-ray dataset (Open-I)
This script prepares the data in the format expected by MedCLIP
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Define paths
INDIANA_BASE_PATH = r'/kaggle/input/chest-xrays-indiana-university'
PROJECTIONS_PATH = os.path.join(INDIANA_BASE_PATH, 'indiana_projections.csv')
REPORTS_PATH = os.path.join(INDIANA_BASE_PATH, 'indiana_reports.csv')
IMAGES_PATH = os.path.join(INDIANA_BASE_PATH, 'images', 'images_normalized')
OUTPUT_DIR = './local_data'  # Use relative path, not absolute

# CheXpert labels (14 labels used by MedCLIP)
CHEXPERT_LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion',
    'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]

# Mapping from Indiana MeSH terms to CheXpert labels
MESH_TO_CHEXPERT = {
    'Cardiomegaly': 'Cardiomegaly',
    'Pulmonary Edema': 'Edema',
    'Pleural Effusion': 'Pleural Effusion',
    'Pneumonia': 'Pneumonia',
    'Atelectasis': 'Atelectasis',
    'Pneumothorax': 'Pneumothorax',
    'Consolidation': 'Consolidation',
    'Opacity': 'Lung Opacity',
    'Lung Opacity': 'Lung Opacity',
    'Fracture': 'Fracture',
    'Support Devices': 'Support Devices',
    'normal': 'No Finding',
}

def load_indiana_data():
    """Load Indiana dataset CSV files"""
    print("Loading Indiana dataset...")
    projections_df = pd.read_csv(PROJECTIONS_PATH)
    reports_df = pd.read_csv(REPORTS_PATH)
    
    print(f"Loaded {len(projections_df)} projections")
    print(f"Loaded {len(reports_df)} reports")
    
    return projections_df, reports_df

def filter_frontal_views(projections_df):
    """Keep only frontal views (PA or AP)"""
    frontal_df = projections_df[projections_df['projection'] == 'Frontal'].copy()
    print(f"Filtered to {len(frontal_df)} frontal views")
    return frontal_df

def create_report_text(row):
    """Combine findings and impression into a single report"""
    parts = []
    
    if pd.notna(row['findings']):
        parts.append(f"FINDINGS: {row['findings']}")
    
    if pd.notna(row['impression']):
        parts.append(f"IMPRESSION: {row['impression']}")
    
    if len(parts) == 0:
        return np.nan
    
    return " ".join(parts)

def extract_labels_from_problems(problems_str):
    """Extract CheXpert-style labels from Indiana Problems field"""
    labels = {label: 0 for label in CHEXPERT_LABELS}
    
    if pd.isna(problems_str):
        return labels
    
    problems_str = str(problems_str).lower()
    
    # Check for 'normal' first
    if 'normal' in problems_str:
        labels['No Finding'] = 1
        return labels
    
    # Map problems to CheXpert labels
    found_any = False
    for mesh_term, chexpert_label in MESH_TO_CHEXPERT.items():
        if mesh_term.lower() in problems_str:
            labels[chexpert_label] = 1
            found_any = True
    
    # If no specific finding, mark as No Finding
    if not found_any:
        labels['No Finding'] = 1
    else:
        labels['No Finding'] = 0
    
    return labels

def prepare_medclip_format(projections_df, reports_df):
    """Prepare data in MedCLIP format"""
    print("Merging projections and reports...")
    
    # Merge on uid
    merged_df = projections_df.merge(reports_df, on='uid', how='inner')
    print(f"Merged dataset size: {len(merged_df)}")
    
    # Create full image paths
    merged_df['imgpath'] = merged_df['filename'].apply(
        lambda x: os.path.join(IMAGES_PATH, x)
    )
    
    # Check if images exist
    merged_df['img_exists'] = merged_df['imgpath'].apply(os.path.exists)
    existing_df = merged_df[merged_df['img_exists']].copy()
    print(f"Found {len(existing_df)} images that exist on disk")
    
    # Create report text
    existing_df['report'] = existing_df.apply(create_report_text, axis=1)
    
    # Drop rows with no report
    existing_df = existing_df[existing_df['report'].notna()].copy()
    print(f"After filtering reports: {len(existing_df)} samples")
    
    # Extract labels
    print("Extracting labels...")
    labels_df = existing_df['Problems'].apply(extract_labels_from_problems).apply(pd.Series)
    
    # Combine with main dataframe
    final_df = pd.concat([existing_df[['imgpath', 'uid', 'report']], labels_df], axis=1)
    
    # Rename uid to subject_id for consistency
    final_df = final_df.rename(columns={'uid': 'subject_id'})
    
    return final_df

def split_train_val(df, val_ratio=0.2, random_state=42):
    """Split into train and validation sets"""
    print(f"Splitting data with validation ratio: {val_ratio}")
    
    # Shuffle
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Split
    val_size = int(len(df) * val_ratio)
    val_df = df.iloc[:val_size].copy()
    train_df = df.iloc[val_size:].copy()
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    
    return train_df, val_df

def print_label_statistics(df, dataset_name):
    """Print label distribution"""
    print(f"\n{dataset_name} Label Distribution:")
    print("-" * 50)
    for label in CHEXPERT_LABELS:
        count = df[label].sum()
        percentage = (count / len(df)) * 100
        print(f"{label:30s}: {count:4d} ({percentage:5.2f}%)")
    print("-" * 50)

def main():
    """Main preprocessing pipeline"""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    projections_df, reports_df = load_indiana_data()
    
    # Filter to frontal views only
    frontal_df = filter_frontal_views(projections_df)
    
    # Prepare in MedCLIP format
    final_df = prepare_medclip_format(frontal_df, reports_df)
    
    # Split train/val
    train_df, val_df = split_train_val(final_df, val_ratio=0.2)
    
    # Print statistics
    print_label_statistics(train_df, "Training Set")
    print_label_statistics(val_df, "Validation Set")
    
    # Save to CSV
    train_path = os.path.join(OUTPUT_DIR, 'indiana-train-meta.csv')
    val_path = os.path.join(OUTPUT_DIR, 'indiana-val-meta.csv')
    
    train_df.to_csv(train_path, index=True)
    val_df.to_csv(val_path, index=True)
    
    print(f"\n✓ Saved training data to: {train_path}")
    print(f"✓ Saved validation data to: {val_path}")
    print("\nPreprocessing complete!")
    print("\nYou can now use 'indiana-train' and 'indiana-val' in your MedCLIP datalist.")

if __name__ == '__main__':
    main()

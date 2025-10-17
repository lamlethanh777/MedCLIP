"""
Quick inspection script for Indiana dataset
Run this to see dataset statistics BEFORE preprocessing
"""
import os
import pandas as pd
from pathlib import Path

# Paths
INDIANA_BASE = r'/kaggle/input/chest-xrays-indiana-university'
PROJECTIONS_CSV = os.path.join(INDIANA_BASE, 'indiana_projections.csv')
REPORTS_CSV = os.path.join(INDIANA_BASE, 'indiana_reports.csv')
IMAGES_DIR = os.path.join(INDIANA_BASE, 'images', 'images_normalized')

def check_csv_files():
    """Check if CSV files exist"""
    print("="*60)
    print("Checking CSV Files")
    print("="*60)
    
    proj_exists = os.path.exists(PROJECTIONS_CSV)
    reports_exists = os.path.exists(REPORTS_CSV)
    
    print(f"Projections CSV: {proj_exists}")
    print(f"  Path: {PROJECTIONS_CSV}")
    
    print(f"Reports CSV: {reports_exists}")
    print(f"  Path: {REPORTS_CSV}")
    
    return proj_exists and reports_exists

def analyze_projections():
    """Analyze projection data"""
    print("\n" + "="*60)
    print("Analyzing Projections")
    print("="*60)
    
    df = pd.read_csv(PROJECTIONS_CSV)
    
    print(f"Total projections: {len(df)}")
    print(f"Unique patients (uid): {df['uid'].nunique()}")
    print(f"\nProjection types:")
    print(df['projection'].value_counts())
    
    return df

def analyze_reports():
    """Analyze report data"""
    print("\n" + "="*60)
    print("Analyzing Reports")
    print("="*60)
    
    df = pd.read_csv(REPORTS_CSV)
    
    print(f"Total reports: {len(df)}")
    print(f"Unique patients (uid): {df['uid'].nunique()}")
    
    print(f"\nColumns: {list(df.columns)}")
    
    # Check for missing data
    print(f"\nMissing data:")
    for col in ['findings', 'impression', 'MeSH', 'Problems']:
        if col in df.columns:
            missing = df[col].isnull().sum()
            pct = (missing / len(df)) * 100
            print(f"  {col:15s}: {missing:4d} ({pct:5.1f}%)")
    
    # Show common problems
    if 'Problems' in df.columns:
        print(f"\nMost common problems (top 10):")
        problems_series = df['Problems'].dropna()
        problem_counts = {}
        for problems in problems_series:
            for problem in str(problems).split(';'):
                problem = problem.strip()
                if problem:
                    problem_counts[problem] = problem_counts.get(problem, 0) + 1
        
        top_problems = sorted(problem_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for problem, count in top_problems:
            pct = (count / len(df)) * 100
            print(f"  {problem:30s}: {count:4d} ({pct:5.1f}%)")
    
    # Sample report
    print(f"\nSample report (uid={df.iloc[0]['uid']}):")
    print(f"  Findings: {str(df.iloc[0]['findings'])[:150]}...")
    print(f"  Impression: {str(df.iloc[0]['impression'])[:150]}...")
    
    return df

def check_images():
    """Check image files"""
    print("\n" + "="*60)
    print("Checking Images")
    print("="*60)
    
    if not os.path.exists(IMAGES_DIR):
        print(f"❌ Images directory not found: {IMAGES_DIR}")
        return
    
    print(f"Images directory: {IMAGES_DIR}")
    
    # Count images
    image_files = list(Path(IMAGES_DIR).glob("*.png"))
    print(f"Total PNG images: {len(image_files)}")
    
    if len(image_files) > 0:
        print(f"\nSample images (first 5):")
        for img in image_files[:5]:
            size_mb = os.path.getsize(img) / (1024 * 1024)
            print(f"  {img.name:40s} ({size_mb:.2f} MB)")
    
    return len(image_files)

def estimate_preprocessing_output():
    """Estimate output after preprocessing"""
    print("\n" + "="*60)
    print("Preprocessing Estimates")
    print("="*60)
    
    proj_df = pd.read_csv(PROJECTIONS_CSV)
    reports_df = pd.read_csv(REPORTS_CSV)
    
    # Frontal views only
    frontal_count = len(proj_df[proj_df['projection'] == 'Frontal'])
    print(f"Frontal projections: {frontal_count}")
    
    # Merge to see how many have reports
    merged = proj_df.merge(reports_df, on='uid', how='inner')
    frontal_merged = merged[merged['projection'] == 'Frontal']
    
    # Check which have findings or impressions
    has_findings = frontal_merged['findings'].notna()
    has_impression = frontal_merged['impression'].notna()
    has_report = has_findings | has_impression
    
    usable_count = has_report.sum()
    
    print(f"Frontal views with reports: {usable_count}")
    
    # Estimate train/val split (80/20)
    train_est = int(usable_count * 0.8)
    val_est = usable_count - train_est
    
    print(f"\nEstimated after preprocessing:")
    print(f"  Training samples: ~{train_est}")
    print(f"  Validation samples: ~{val_est}")
    print(f"  Total usable: ~{usable_count}")

def main():
    """Run all checks"""
    print("\n" + "="*60)
    print("INDIANA DATASET INSPECTION")
    print("="*60 + "\n")
    
    # Check files exist
    if not check_csv_files():
        print("\n❌ CSV files not found. Please check paths.")
        return
    
    # Analyze data
    proj_df = analyze_projections()
    reports_df = analyze_reports()
    img_count = check_images()
    
    # Estimate output
    estimate_preprocessing_output()
    
    print("\n" + "="*60)
    print("INSPECTION COMPLETE")
    print("="*60)
    print("\nNext step: Run 'python preprocess_indiana.py' to prepare the data")

if __name__ == '__main__':
    main()

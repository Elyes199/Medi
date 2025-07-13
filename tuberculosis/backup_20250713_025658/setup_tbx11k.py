#!/usr/bin/env python3

import os
import sys
import argparse
import shutil
from pathlib import Path


def create_tbx11k_directory_structure():
    """Create the directory structure for the TBX11K dataset"""
    tb_data_dir = Path("tb_data")
    images_dir = tb_data_dir / "images"
    annotations_dir = tb_data_dir / "annotations"
    
    # Create directories
    for directory in [tb_data_dir, images_dir, annotations_dir]:
        directory.mkdir(exist_ok=True)
    
    print("Created directory structure for TBX11K dataset:")
    print(f"├── {tb_data_dir}/")
    print(f"    ├── images/")
    print(f"    └── annotations/")


def convert_dataset(source_dir, target_dir=None):
    """
    Convert an existing TBX11K dataset structure to the expected structure
    
    Args:
        source_dir: Directory containing the source TBX11K dataset
        target_dir: Directory to place the organized dataset (default: tb_data)
    """
    if target_dir is None:
        target_dir = Path("tb_data")
    else:
        target_dir = Path(target_dir)
    
    source_dir = Path(source_dir)
    
    # Create target directories if they don't exist
    images_dir = target_dir / "images"
    annotations_dir = target_dir / "annotations"
    
    for directory in [target_dir, images_dir, annotations_dir]:
        directory.mkdir(exist_ok=True)
    
    # Look for image files and annotation files in the source directory
    image_extensions = ['.jpg', '.jpeg', '.png']
    
    # Find image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(source_dir.glob(f"**/*{ext}"))
    
    # Find annotation files
    annotation_files = list(source_dir.glob("**/*.json"))
    
    print(f"Found {len(image_files)} image files and {len(annotation_files)} annotation files")
    
    # Copy image files to target directory
    for img_file in image_files:
        target_file = images_dir / img_file.name
        try:
            shutil.copy2(img_file, target_file)
        except Exception as e:
            print(f"Error copying {img_file}: {e}")
    
    # Copy annotation files to target directory
    for ann_file in annotation_files:
        target_file = annotations_dir / ann_file.name
        try:
            shutil.copy2(ann_file, target_file)
        except Exception as e:
            print(f"Error copying {ann_file}: {e}")
    
    print(f"Dataset conversion complete. Files copied to {target_dir}")


def check_dependencies():
    """Check if the required dependencies are installed"""
    try:
        import torch
        import torchvision
        import pycocotools
        print("✓ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"✘ Missing dependency: {e}")
        print("\nPlease install the required dependencies:")
        print("pip install torch torchvision pycocotools")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Set up the TBX11K dataset for tuberculosis detection"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Path to source TBX11K dataset for conversion"
    )
    parser.add_argument(
        "--target", type=str, default=None,
        help="Target directory for the converted dataset (default: tb_data)"
    )
    parser.add_argument(
        "--check-deps", action="store_true",
        help="Check for required dependencies"
    )
    
    args = parser.parse_args()
    
    if args.check_deps:
        if not check_dependencies():
            return
    
    # Create directory structure
    create_tbx11k_directory_structure()
    
    # Convert dataset if source is provided
    if args.source:
        convert_dataset(args.source, args.target)
    else:
        print("\nTo convert an existing TBX11K dataset, use:")
        print("python setup_tbx11k.py --source /path/to/tbx11k/dataset")
    
    # Print next steps
    print("\nNext steps:")
    print("1. Place your TBX11K chest X-ray images in tb_data/images/")
    print("2. Place your TBX11K annotation files in tb_data/annotations/")
    print("3. Train the detection model: python -m tuberculosis.tb_train_detector")
    print("4. Run inference: python -m tuberculosis.tb_detect_bbox --image <path_to_image>")


if __name__ == "__main__":
    main()

"""
TBX11K Dataset Setup Script

This script helps set up the TBX11K dataset for the tuberculosis detection system.
It creates the necessary directory structure and copies or links files from the downloaded dataset.
"""

import os
import shutil
import argparse
from pathlib import Path


def setup_tbx11k(dataset_path, target_path="tb_data", use_symlinks=True):
    """
    Set up the TBX11K dataset directory structure.
    
    Args:
        dataset_path (str): Path to the downloaded TBX11K dataset
        target_path (str): Path where to set up the dataset
        use_symlinks (bool): Whether to use symlinks (True) or copy files (False)
    """
    original_dataset = Path(dataset_path)
    tbx11k_dir = Path(target_path)
    
    if not original_dataset.exists():
        raise FileNotFoundError(f"Dataset path not found: {original_dataset}")
    
    # Create directories
    os.makedirs(tbx11k_dir, exist_ok=True)
    os.makedirs(tbx11k_dir / "images", exist_ok=True)
    os.makedirs(tbx11k_dir / "annotations", exist_ok=True)
    os.makedirs(tbx11k_dir / "lists", exist_ok=True)
    
    # Copy or link image directories
    for subdir in ["tb", "health", "sick", "extra", "test"]:
        src_dir = original_dataset / "images" / subdir
        dst_dir = tbx11k_dir / "images" / subdir
        
        if not src_dir.exists():
            print(f"Warning: Source directory not found: {src_dir}")
            continue
        
        if dst_dir.exists():
            print(f"Directory already exists, skipping: {dst_dir}")
            continue
        
        print(f"Setting up image directory: {subdir}")
        if use_symlinks and os.name != 'nt':  # Symlinks not well supported on Windows
            os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
            os.symlink(src_dir, dst_dir)
        else:
            shutil.copytree(src_dir, dst_dir)
    
    # Copy annotation files
    for ann_file in ["train.json", "val.json"]:
        src_file = original_dataset / "annotations" / ann_file
        dst_file = tbx11k_dir / "annotations" / ann_file
        
        if not src_file.exists():
            print(f"Warning: Annotation file not found: {src_file}")
            continue
        
        print(f"Copying annotation file: {ann_file}")
        shutil.copy(src_file, dst_file)
    
    # Copy list files
    for list_file in ["all_train.txt", "all_val.txt", "all_test.txt", 
                     "tbx11k_train.txt", "tbx11k_val.txt"]:
        src_file = original_dataset / "lists" / list_file
        dst_file = tbx11k_dir / "lists" / list_file
        
        if not src_file.exists():
            print(f"Warning: List file not found: {src_file}")
            continue
        
        print(f"Copying list file: {list_file}")
        shutil.copy(src_file, dst_file)
    
    print("\nDataset setup complete!")
    print(f"Dataset is ready at: {tbx11k_dir.absolute()}")
    print("\nYou can now use the dataset with tb_dataset.py using:")
    print(f"  --ann_file {tbx11k_dir}/annotations/train.json")
    print(f"  --img_prefix {tbx11k_dir}/images/")
    print(f"  --img_list {tbx11k_dir}/lists/all_train.txt")


def main():
    parser = argparse.ArgumentParser(description="Set up the TBX11K dataset")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        required=True, 
        help="Path to the downloaded TBX11K dataset"
    )
    parser.add_argument(
        "--target_path", 
        type=str, 
        default="tb_data", 
        help="Path where to set up the dataset"
    )
    parser.add_argument(
        "--copy_files", 
        action="store_true", 
        help="Copy files instead of creating symlinks"
    )
    
    args = parser.parse_args()
    setup_tbx11k(args.dataset_path, args.target_path, not args.copy_files)


if __name__ == "__main__":
    main()

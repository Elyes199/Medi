import os
import shutil
from pathlib import Path

def create_tb_data_structure():
    """
    Creates the necessary directory structure for TB data.
    """
    # Define the data directories
    tb_data_dir = Path("tb_data")
    subdirs = ["TrainImages", "TrainMasks", "TestImages", "TestMasks"]
    
    # Create the main directory
    os.makedirs(tb_data_dir, exist_ok=True)
    
    # Create subdirectories
    for subdir in subdirs:
        os.makedirs(tb_data_dir / subdir, exist_ok=True)
    
    print("Created TB data directory structure:")
    print(f"└── {tb_data_dir}/")
    for subdir in subdirs:
        print(f"    └── {subdir}/")
    
    print("\nPlease place your TB chest X-ray images and masks in the appropriate directories.")


def create_tb_models_directory():
    """
    Creates the directory for storing trained models.
    """
    models_dir = Path("tuberculosis/models")
    os.makedirs(models_dir, exist_ok=True)
    print(f"Created models directory at {models_dir}")


def main():
    """
    Set up the environment for tuberculosis detection.
    """
    print("Setting up environment for TB detection...")
    create_tb_data_structure()
    create_tb_models_directory()
    print("\nSetup complete! You're ready to start working on TB detection.")
    print("\nNext steps:")
    print("1. Place your TB chest X-ray images and masks in the tb_data directory")
    print("2. Train the model: python -m tuberculosis.tb_train")
    print("3. Run detection: python -m tuberculosis.tb_detect")
    print("4. For interactive analysis: open tuberculosis/tb_test.ipynb")


if __name__ == "__main__":
    main()

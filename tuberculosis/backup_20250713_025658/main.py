def main():
    """
    Main entry point for the tuberculosis detection application.
    
    This function loads the trained model and sets up the environment for TB detection.
    To use this module, run it directly or import and call main().
    """
    print("Tuberculosis Detection and Segmentation Tool")
    print("===========================================")
    print("This module provides tools for detecting tuberculosis lesions in chest X-rays.")
    print("\nAvailable modules:")
    print("- tb_train.py: Train a new TB detection model")
    print("- tb_detect.py: Detect TB in a single image")
    print("- tb_test.ipynb: Interactive notebook for testing and visualization")
    print("\nUsage:")
    print("1. Place your chest X-ray data in the tb_data directory with the following structure:")
    print("   tb_data/")
    print("   ├── TrainImages/  - Contains training images (.png)")
    print("   ├── TrainMasks/   - Contains training masks (.png)")
    print("   ├── TestImages/   - Contains test images (.png)")
    print("   └── TestMasks/    - Contains test masks (.png)")
    print("\n2. Train the model using: python -m tuberculosis.tb_train")
    print("\n3. Detect TB in an image: python -m tuberculosis.tb_detect")
    print("\n4. For interactive analysis, open: tuberculosis/tb_test.ipynb")


if __name__ == "__main__":
    main()

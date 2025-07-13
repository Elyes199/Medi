# Tuberculosis Detection and Segmentation

This module provides tools for detecting and segmenting tuberculosis lesions in chest X-ray images.

## Overview

Unlike the main Medi project which focuses on lung segmentation, this module specifically targets the detection and segmentation of tuberculosis lesions within the lungs. Tuberculosis (TB) lesions typically appear as opacity or consolidation in chest X-rays and have distinct characteristics from normal lung tissue.

## Features

- TB lesion segmentation in chest X-rays
- Enhanced data preprocessing pipeline for TB detection
- Deeper network architecture optimized for small lesion detection
- Advanced evaluation metrics (precision, recall, F1, IoU, AUC)
- TB severity assessment based on affected area percentage
- Interactive visualization tools for TB detection results

## Directory Structure

```plaintext
tuberculosis/
├── main.py                - Main entry point with usage information
├── tb_train.py            - Training script for TB segmentation model
├── tb_preprocess.py       - Data preprocessing for TB images
├── tb_utilities.py        - Utility functions for training and evaluation
├── tb_detect.py           - Script for detecting TB in images using segmentation
├── tb_test.ipynb          - Jupyter notebook for testing segmentation
├── tb_dataset.py          - Dataset loader for TBX11K dataset
├── tb_detection.py        - Object detection model implementation
├── tb_train_detector.py   - Training script for object detection model
├── tb_detect_bbox.py      - Script for detecting TB using bounding boxes
├── tb_detection_test.ipynb- Jupyter notebook for testing object detection
├── tb_setup.py            - Script for setting up TBX11K dataset
├── TBX11K_SETUP.md        - Instructions for TBX11K dataset setup
└── models/                - Directory for saved model weights and metrics
    ├── segmentation/      - Segmentation model weights and metrics
    │   └── best_tb_model.pth - Best trained segmentation model
    └── detection/         - Detection model weights and metrics
        └── tb_detector_final.pth - Best trained detection model
```

## Data Organization

This module supports two data organization methods:

### 1. Simple Directory Structure (for segmentation masks)

For basic TB lesion segmentation:

```
tb_data/
├── TrainImages/  - Contains training chest X-ray images (.png)
├── TrainMasks/   - Contains training TB lesion masks (.png)
├── TestImages/   - Contains test chest X-ray images (.png)
└── TestMasks/    - Contains test TB lesion masks (.png)
```

### 2. TBX11K Dataset Structure (for object detection)

For the TBX11K dataset with COCO format annotations:

```
tb_data/
├── images/
│   ├── health/    - Healthy chest X-ray images
│   ├── tb/        - TB chest X-ray images
│   ├── sick/      - Non-TB disease chest X-ray images
│   ├── extra/     - Additional chest X-ray images
│   └── test/      - Test set images
├── annotations/
│   ├── train.json - Training annotations in COCO format
│   └── val.json   - Validation annotations in COCO format
└── lists/
    ├── all_train.txt - List of all training images
    ├── all_val.txt   - List of all validation images
    └── all_test.txt  - List of all test images
```

For more details on setting up the TBX11K dataset, see [TBX11K_SETUP.md](TBX11K_SETUP.md)

## Getting Started

### For Segmentation Model

1. **Data Preparation**:
   
   - Place your chest X-ray images and masks in the appropriate directories
   - Ensure TB lesion masks highlight only the TB-affected areas

2. **Training**:

   ```bash
   python -m tuberculosis.tb_train
   ```

3. **Testing and Visualization**:
   
   - Use the Jupyter notebook for interactive analysis:

   ```bash
   jupyter notebook tuberculosis/tb_test.ipynb
   ```
   
   - Or use the segmentation detection script:

   ```bash
   python -m tuberculosis.tb_detect --image test_image.png --model models/best_tb_model.pth
   ```

### For Object Detection Model

1. **Set up the TBX11K dataset**:

   ```bash
   python -m tuberculosis.tb_setup --dataset_path /path/to/TBX11K
   ```

2. **Train the detection model**:

   ```bash
   python -m tuberculosis.tb_train_detector \
       --ann_file tb_data/annotations/train.json \
       --img_prefix tb_data/images/ \
       --img_list tb_data/lists/all_train.txt
   ```

3. **Run inference with the detection model**:

   ```bash
   python -m tuberculosis.tb_detect_bbox \
       --image tb_data/images/tb/tb0005.png \
       --model tuberculosis/models/detection/tb_detector_final.pth \
       --output results/detected.png
   ```

4. **Interactive visualization**:

   ```bash
   jupyter notebook tuberculosis/tb_detection_test.ipynb
   ```

See [TBX11K_SETUP.md](TBX11K_SETUP.md) for more detailed instructions on working with the TBX11K dataset.

## Model Architectures

### Segmentation Model

The TB segmentation model uses a deeper UNet architecture with:

- 5 levels of feature extraction (vs 4 in the lung segmentation)
- Increased number of residual units (3 vs 2)
- Dropout regularization to prevent overfitting
- Advanced augmentation pipeline for improved generalization

### Object Detection Model

The TB object detection model uses Faster R-CNN with:

- ResNet-50-FPN backbone pretrained on ImageNet
- Region proposal network for potential TB lesion regions
- Multi-class classification head for TB types
- Bounding box regression for accurate localization
- Feature Pyramid Network for handling lesions of various sizes

## Evaluation Metrics

In addition to Dice coefficient, the models evaluate:

- Precision: Accuracy of positive predictions
- Recall: Ability to detect all TB lesions
- F1 Score: Harmonic mean of precision and recall
- IoU: Intersection over Union
- AUC-ROC: Area under the ROC curve

For the object detection model, we also compute:

- Mean Average Precision (mAP)
- Average Precision (AP) per class
- Precision-Recall curves

## TB Severity Classification

Based on the percentage of affected area:

- None/Minimal: < 0.5%
- Mild: 0.5% - 2%
- Moderate: 2% - 5%
- Severe: > 5%

## Comparison of Approaches

| Feature | Segmentation Model | Object Detection Model |
|---------|-------------------|----------------------|
| Task | Pixel-level mask prediction | Bounding box detection |
| Granularity | Fine-grained lesion details | Coarse lesion localization |
| Training Data | Requires pixel masks | Only needs bounding boxes |
| Inference Speed | Slower | Faster |
| Multiple Lesions | Handles naturally | Explicitly identifies each |
| TB Type Classification | No | Yes (multi-class) |

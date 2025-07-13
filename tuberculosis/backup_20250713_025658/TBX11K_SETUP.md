# TBX11K Dataset Setup

This document provides instructions on how to set up the TBX11K dataset for tuberculosis detection.

## Quick Start

```bash
# 1. Install dependencies
pip install pycocotools torch torchvision monai

# 2. Download TBX11K dataset
# Follow the instructions at https://github.com/LongTailMedicalInc/TBX11K

# 3. Configure the dataset
python -m tuberculosis.tb_setup --dataset_path /path/to/TBX11K

# 4. Train the TB detection model
python -m tuberculosis.tb_train_detector \
    --ann_file tb_data/annotations/train.json \
    --img_prefix tb_data/images/ \
    --img_list tb_data/lists/all_train.txt

# 5. Run inference on a test image
python -m tuberculosis.tb_detect_bbox \
    --image tb_data/images/tb/tb0005.png \
    --model tuberculosis/models/detection/tb_detector_final.pth \
    --output results/detected.png
```

## Dataset Overview

The TBX11K dataset is a large-scale chest X-ray dataset specifically designed for tuberculosis detection. It contains:

- 11,200 chest X-ray images
- Annotations for TB-related findings in COCO format
- Multiple TB classes: ActiveTuberculosis, ObsoletePulmonaryTuberculosis, and PulmonaryTuberculosis

## Dataset Structure

The TBX11K dataset has the following structure:

```
TBX11K/
├── images/
│   ├── extra/            - Additional chest X-ray images
│   ├── health/           - Normal/healthy chest X-rays
│   ├── sick/             - Non-TB abnormal chest X-rays
│   ├── tb/               - TB-positive chest X-rays
│   └── test/             - Test set images
├── lists/
│   ├── all_test.txt      - List of all test images
│   ├── all_train.txt     - List of all training images
│   ├── all_trainval.txt  - List of all training and validation images
│   ├── all_val.txt       - List of all validation images
│   ├── tbx11k_train.txt  - List of TB training images
│   ├── tbx11k_val.txt    - List of TB validation images
│   └── tbx11k_trainval.txt - List of TB training and validation images
└── annotations/
    ├── train.json        - Training annotations in COCO format
    └── val.json          - Validation annotations in COCO format
```

## Dataset Download

1. Download the TBX11K dataset from its [official source](https://github.com/LongTailMedicalInc/TBX11K).

2. Extract the downloaded files and organize them into the directory structure shown above.

## Dataset Download and Preparation

1. Download the TBX11K dataset from its [official source](https://github.com/LongTailMedicalInc/TBX11K).

2. Install required dependencies:

```bash
pip install pycocotools torch torchvision
```

3. Set up the dataset directories:

```python
import os
import shutil
from pathlib import Path

# Create directories
tbx11k_dir = Path("tb_data")
os.makedirs(tbx11k_dir / "images", exist_ok=True)
os.makedirs(tbx11k_dir / "annotations", exist_ok=True)
os.makedirs(tbx11k_dir / "lists", exist_ok=True)

# Set path to downloaded dataset (update this)
original_dataset = Path("/path/to/downloaded/TBX11K")

# Copy or link image directories
for subdir in ["tb", "health", "sick", "extra", "test"]:
    src_dir = original_dataset / "images" / subdir
    dst_dir = tbx11k_dir / "images" / subdir
    
    if not dst_dir.exists():
        if os.name == 'nt':  # Windows
            # Copy directories on Windows (no symlinks)
            shutil.copytree(src_dir, dst_dir)
        else:
            # Create symlinks on Unix systems
            os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
            os.symlink(src_dir, dst_dir)

# Copy annotation files
for ann_file in ["train.json", "val.json"]:
    src_file = original_dataset / "annotations" / ann_file
    dst_file = tbx11k_dir / "annotations" / ann_file
    shutil.copy(src_file, dst_file)

# Copy list files
for list_file in ["all_train.txt", "all_val.txt", "all_test.txt", 
                 "tbx11k_train.txt", "tbx11k_val.txt"]:
    src_file = original_dataset / "lists" / list_file
    dst_file = tbx11k_dir / "lists" / list_file
    shutil.copy(src_file, dst_file)

print("Dataset setup complete!")
```

## Annotation Format

The TBX11K annotations are in COCO format with the following structure:

```json
{
  "info": [{
    "contributor": "Yun Liu, Yu-Huan Wu, Yunfeng Ban, Huifang Wang, Ming-Ming Cheng",
    "date_created": "2020/06/22",
    "description": "TBX11K Dataset",
    "url": "http://mmcheng.net/tb",
    "version": "1.0",
    "year": 2020
  }],
  "licenses": [{
    "id": 1,
    "name": "Attribution-NonCommercial-ShareAlike License",
    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
  }],
  "categories": [
    {"id": 1, "name": "ActiveTuberculosis", "supercategory": "Tuberculosis"},
    {"id": 2, "name": "ObsoletePulmonaryTuberculosis", "supercategory": "Tuberculosis"},
    {"id": 3, "name": "PulmonaryTuberculosis", "supercategory": "Tuberculosis"}
  ],
  "images": [
    {
      "id": 1,
      "file_name": "tb/tb0005.png",
      "width": 512,
      "height": 512,
      "date_captured": "2025-07-05 15:53:29.339921",
      "license": 1,
      "coco_url": "",
      "flickr_url": ""
    },
    ...
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 3,
      "bbox": [214.17, 120.75, 99.13, 40.23],  // [x, y, width, height]
      "area": 3987.0,
      "segmentation": [...],  // Polygon points for segmentation mask
      "iscrowd": 0
    },
    ...
  ]
}
```

### Important Notes about Annotations

1. **File Names**: The `file_name` field in the `images` section includes the subdirectory path (e.g., `tb/tb0005.png`). This is why the dataset loader needs to handle paths with subdirectories.

2. **Image Sets**: The dataset uses text files in the `lists/` directory to define train/validation/test splits. Each file contains paths relative to the `images/` directory.

3. **Categories**: There are three TB-related categories with different IDs:
   - ID 1: ActiveTuberculosis
   - ID 2: ObsoletePulmonaryTuberculosis
   - ID 3: PulmonaryTuberculosis

4. **Bounding Boxes**: The `bbox` field contains coordinates in the format `[x, y, width, height]`, where `(x, y)` is the top-left corner.

```

## Integration with the TB Detection System

The tuberculosis detection system already includes the necessary code to work with the TBX11K dataset:

- `tb_dataset.py`: Contains the `TBX11KDataset` class for loading and processing the dataset
- `tb_detection.py`: Provides utility functions for working with the detection model
- `tb_train_detector.py`: Script for training the detection model
- `tb_detect_bbox.py`: Script for detecting TB in images using the trained model
- `tb_detection_test.ipynb`: Notebook for interactive testing and visualization

## Training a Model

To train a TB detection model using the TBX11K dataset:

```bash
python -m tuberculosis.tb_train_detector \
    --ann_file tb_data/annotations/train.json \
    --img_prefix tb_data/images/ \
    --img_list tb_data/lists/all_train.txt \
    --model_dir tuberculosis/models/detection \
    --num_classes 3 \
    --batch_size 4 \
    --epochs 20
```

You can also specify a specific subset of the dataset by using the appropriate list file:

```bash
python -m tuberculosis.tb_train_detector \
    --ann_file tb_data/annotations/train.json \
    --img_prefix tb_data/images/ \
    --img_list tb_data/lists/tbx11k_train.txt \
    --model_dir tuberculosis/models/detection \
    --num_classes 3 \
    --batch_size 4 \
    --epochs 20
```

## Running Inference

To detect TB in a chest X-ray using the trained model:

```bash
python -m tuberculosis.tb_detect_bbox \
    --image tb_data/images/tb/tb0005.png \
    --model tuberculosis/models/detection/tb_detector_final.pth \
    --output results/detection_result.png \
    --threshold 0.5
```

For batch processing of multiple images:

```bash
python -m tuberculosis.tb_detect_bbox \
    --image_list tb_data/lists/all_test.txt \
    --img_prefix tb_data/images/ \
    --model tuberculosis/models/detection/tb_detector_final.pth \
    --output_dir results/detection_results \
    --threshold 0.5
```

## Visualization and Analysis

For a comprehensive analysis, you can use both the segmentation model (from the original tuberculosis code) and the object detection model together. This is demonstrated in the `tb_detection_test.ipynb` notebook.

### Using the Jupyter Notebook

1. Open the `tb_detection_test.ipynb` notebook:

```bash
jupyter notebook tuberculosis/tb_detection_test.ipynb
```

1. Update the paths in the notebook to match your dataset structure:

```python
# Dataset paths
img_prefix = "tb_data/images/"
ann_file = "tb_data/annotations/val.json"
image_list = "tb_data/lists/all_val.txt"

# Model paths
detection_model_path = "tuberculosis/models/detection/tb_detector_final.pth"
segmentation_model_path = "tuberculosis/models/segmentation/tb_segmenter_final.pth"
```

1. Run the cells to visualize the results:
   - Bounding box detection of TB lesions
   - Segmentation masks of TB affected areas
   - Combined visualization with both detection and segmentation

### Evaluating Model Performance

You can evaluate the model's performance on the validation or test set:

```python
from tuberculosis.tb_evaluate import evaluate_detection_model

metrics = evaluate_detection_model(
    model_path="tuberculosis/models/detection/tb_detector_final.pth",
    ann_file="tb_data/annotations/val.json",
    img_prefix="tb_data/images/",
    img_list="tb_data/lists/all_val.txt"
)

print(f"mAP: {metrics['mAP']}")
print(f"AP for each class: {metrics['AP']}")
```

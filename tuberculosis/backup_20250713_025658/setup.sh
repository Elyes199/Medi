#!/bin/bash

# Create directory for TB data if it doesn't exist
mkdir -p tb_data/TrainImages tb_data/TrainMasks tb_data/TestImages tb_data/TestMasks

# Create directory for models
mkdir -p tuberculosis/models

echo "Directory structure created for tuberculosis detection!"
echo ""
echo "Place your data in the following structure:"
echo "tb_data/"
echo "├── TrainImages/  - Training chest X-ray images"
echo "├── TrainMasks/   - Training TB lesion masks"
echo "├── TestImages/   - Test chest X-ray images"
echo "└── TestMasks/    - Test TB lesion masks"
echo ""
echo "To train the model:   python -m tuberculosis.tb_train"
echo "To run detection:     python -m tuberculosis.tb_detect"
echo "For interactive use:  open tuberculosis/tb_test.ipynb"

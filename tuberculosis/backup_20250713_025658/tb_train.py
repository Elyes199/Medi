import os
import torch
import torch.nn as nn
import torch.optim as optim
from monai.networks.nets import UNet
from tb_preprocess import prepare_tb
from tb_utilities import train, show_patient, calculate_pixels, calculate_weights

DATA_DIR = "tb_data"             # Root directory containing TrainImages/, TrainMasks/, etc.
MODEL_DIR = "tuberculosis/models" # Directory to save trained models and metrics
MAX_EPOCHS = 50                  # Training epochs
LEARNING_RATE = 1e-4             # Optimizer learning rate
TEST_INTERVAL = 1                # How often to evaluate on test set (in epochs)
USE_CLASS_WEIGHTS = True         # Use class weighting in loss function
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


os.makedirs(MODEL_DIR, exist_ok=True)

train_loader, test_loader = prepare_tb(
    in_dir=DATA_DIR,
    spatial_size=[256, 256],
    a_min=0,
    a_max=255,
    cache=True
)

# Use a deeper UNet model for better feature detection of TB lesions
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),  # Added an extra level for deeper features
    strides=(2, 2, 2, 2),
    num_res_units=3,  # Increased number of residual units
    dropout=0.2,      # Added dropout for regularization
).to(DEVICE)

if USE_CLASS_WEIGHTS:
    val = calculate_pixels(train_loader)
    class_weights = calculate_weights(val[0, 0], val[0, 1])
    # TB lesions are typically much smaller than lungs, so we might need higher weight for positive class
    loss_function = nn.BCEWithLogitsLoss(pos_weight=class_weights[1].to(DEVICE))
else:
    loss_function = nn.BCEWithLogitsLoss()

# Use AdamW optimizer for better convergence
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)


if __name__ == '__main__':
    train(
        model=model,
        data_in=(train_loader, test_loader),
        loss=loss_function,
        optim=optimizer,
        scheduler=scheduler,  # Added scheduler
        max_epochs=MAX_EPOCHS,
        model_dir=MODEL_DIR,
        test_interval=TEST_INTERVAL,
        device=DEVICE,
    )

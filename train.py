import os
import torch
import torch.nn as nn
import torch.optim as optim
from monai.networks.nets import UNet
from preprocess import prepare
from utilities import train, show_patient, calculate_pixels, calculate_weights

DATA_DIR = "data"                # Root directory containing TrainImages/, TrainMasks/, etc.
MODEL_DIR = "models"             # Directory to save trained models and metrics
MAX_EPOCHS = 30                  # Training epochs
LEARNING_RATE = 1e-3             # Optimizer learning rate
TEST_INTERVAL = 1                # How often to evaluate on test set (in epochs)
USE_CLASS_WEIGHTS = True         # Use class weighting in loss function
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


os.makedirs(MODEL_DIR, exist_ok=True)

train_loader, test_loader = prepare(
    in_dir=DATA_DIR,
    spatial_size=[256, 256],
    a_min=0,
    a_max=255,
    cache=True
)

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2,
).to(DEVICE)

if USE_CLASS_WEIGHTS:
    val = calculate_pixels(train_loader)
    class_weights = calculate_weights(val[0, 0], val[0, 1])
    loss_function = nn.BCEWithLogitsLoss(pos_weight=class_weights[1].to(DEVICE))
else:
    loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


if __name__ == '__main__':
    train(
        model=model,
        data_in=(train_loader, test_loader),
        loss=loss_function,
        optim=optimizer,
        max_epochs=MAX_EPOCHS,
        model_dir=MODEL_DIR,
        test_interval=TEST_INTERVAL,
        device=DEVICE,
    )
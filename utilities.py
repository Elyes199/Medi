from monai.utils import first
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from monai.losses import DiceLoss
from tqdm import tqdm

def dice_metric(predicted, target):
    """
    Compute the Dice coefficient as a metric.
    """
    dice_value = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    value = 1 - dice_value(predicted, target).item()
    return value

def calculate_weights(val1, val2):
    """
    Calculate class weights based on foreground and background pixel counts.
    """
    count = np.array([val1, val2])
    summ = count.sum()
    weights = count / summ
    weights = 1 / weights
    summ = weights.sum()
    weights = weights / summ
    return torch.tensor(weights, dtype=torch.float32)

def train(model, data_in, loss, optim, max_epochs, model_dir, test_interval=1, device=torch.device("cuda:0")):
    best_metric = -1
    best_metric_epoch = -1
    save_loss_train = []
    save_loss_test = []
    save_metric_train = []
    save_metric_test = []
    train_loader, test_loader = data_in

    for epoch in range(max_epochs):
        model.train()
        train_epoch_loss = 0
        epoch_metric_train = 0

        # tqdm progress bar
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{max_epochs}", leave=False, ncols=100)

        for step, batch_data in pbar:
            volume = batch_data["vol"]
            label = batch_data["seg"]
            label = (label != 0).float()
            volume, label = volume.to(device), label.to(device)

            optim.zero_grad()
            outputs = model(volume)
            train_loss = loss(outputs, label)
            train_loss.backward()
            optim.step()

            train_epoch_loss += train_loss.item()
            train_metric = dice_metric(outputs, label)
            epoch_metric_train += train_metric

            # Update progress bar description
            avg_loss = train_epoch_loss / (step + 1)
            avg_dice = epoch_metric_train / (step + 1)
            pbar.set_postfix({
                "Loss": f"{avg_loss:.4f}",
                "Dice": f"{avg_dice:.4f}",
                "Step": f"{step+1}/{len(train_loader)}"
            })

        # Save training metrics
        train_epoch_loss /= len(train_loader)
        epoch_metric_train /= len(train_loader)
        save_loss_train.append(train_epoch_loss)
        save_metric_train.append(epoch_metric_train)

        np.save(os.path.join(model_dir, 'loss_train.npy'), save_loss_train)
        np.save(os.path.join(model_dir, 'metric_train.npy'), save_metric_train)

        print(f"[Epoch {epoch+1}] Train Loss: {train_epoch_loss:.4f} | Dice: {epoch_metric_train:.4f}")

        if (epoch + 1) % test_interval == 0:
            model.eval()
            test_epoch_loss = 0
            epoch_metric_test = 0

            with torch.no_grad():
                for test_data in test_loader:
                    test_volume = test_data["vol"]
                    test_label = test_data["seg"]
                    test_label = (test_label != 0).float()
                    test_volume, test_label = test_volume.to(device), test_label.to(device)

                    test_outputs = model(test_volume)
                    test_loss = loss(test_outputs, test_label)
                    test_dice = dice_metric(test_outputs, test_label)

                    test_epoch_loss += test_loss.item()
                    epoch_metric_test += test_dice

            test_epoch_loss /= len(test_loader)
            epoch_metric_test /= len(test_loader)

            save_loss_test.append(test_epoch_loss)
            save_metric_test.append(epoch_metric_test)
            np.save(os.path.join(model_dir, 'loss_test.npy'), save_loss_test)
            np.save(os.path.join(model_dir, 'metric_test.npy'), save_metric_test)

            print(f"[Epoch {epoch+1}] Test  Loss: {test_epoch_loss:.4f} | Dice: {epoch_metric_test:.4f}")

            if epoch_metric_test > best_metric:
                best_metric = epoch_metric_test
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(model_dir, "best_metric_model.pth"))
                print(f"✅ New best model saved (Dice: {best_metric:.4f}) at epoch {best_metric_epoch}")

    print(f" Training completed. Best Dice: {best_metric:.4f} at epoch {best_metric_epoch}")

def show_patient(data, index=0, train=True, test=False):
    """
    Visualize a 2D image and its mask.
    """
    check_patient_train, check_patient_test = data

    if train:
        sample = list(check_patient_train)[index]
        img = sample["vol"][0]  # Shape: (1, H, W)
        mask = sample["seg"][0]

        plt.figure("Train Sample", (10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Image")
        plt.imshow(img, cmap="gray")
        plt.subplot(1, 2, 2)
        plt.title("Mask")
        plt.imshow(mask)
        plt.show()

    if test:
        sample = list(check_patient_test)[index]
        img = sample["vol"][0]
        mask = sample["seg"][0]

        plt.figure("Test Sample", (10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Image")
        plt.imshow(img, cmap="gray")
        plt.subplot(1, 2, 2)
        plt.title("Mask")
        plt.imshow(mask)
        plt.show()

def calculate_pixels(data):
    """
    Count background and foreground pixels in 2D masks.
    """
    val = np.zeros((1, 2))
    for batch in tqdm(data):
        batch_label = batch["seg"]
        batch_label = batch_label != 0
        labels, counts = np.unique(batch_label.numpy(), return_counts=True)
        count_dict = dict(zip(labels, counts))
        background = count_dict.get(0, 0)
        foreground = count_dict.get(1, 0)
        val += np.array([[background, foreground]])

    print('Final pixel counts [background, foreground]:', val)
    return val

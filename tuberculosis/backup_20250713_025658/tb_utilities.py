from monai.utils import first
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import cv2
from monai.losses import DiceLoss, DiceCELoss, FocalLoss
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import cv2


def dice_metric(predicted, target):
    """
    Compute the Dice coefficient as a metric.
    """
    dice_value = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    value = 1 - dice_value(predicted, target).item()
    return value


def calculate_weights(val1, val2):
    """
    Calculate class weights based on foreground (TB lesions) and background pixel counts.
    TB lesions are likely to be much smaller than the entire lung, so the weighting is important.
    """
    count = np.array([val1, val2])
    summ = count.sum()
    weights = count / summ
    weights = 1 / weights
    summ = weights.sum()
    weights = weights / summ
    return torch.tensor(weights, dtype=torch.float32)


def calculate_additional_metrics(y_true, y_pred):
    """
    Calculate additional metrics for TB segmentation evaluation:
    - Precision
    - Recall
    - F1 Score
    - IoU (Intersection over Union)
    - AUC-ROC
    """
    y_true = y_true.detach().cpu().numpy().flatten()
    y_pred = y_pred.detach().cpu().numpy().flatten()
    
    # Apply threshold
    y_pred_binary = (y_pred > 0.5).astype(np.int32)
    
    # Calculate metrics
    try:
        precision = precision_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary)
        f1 = f1_score(y_true, y_pred_binary)
        auc = roc_auc_score(y_true, y_pred)
        
        # Calculate IoU
        intersection = np.logical_and(y_true, y_pred_binary).sum()
        union = np.logical_or(y_true, y_pred_binary).sum()
        iou = intersection / union if union > 0 else 0
        
        return {
            'precision': precision, 
            'recall': recall, 
            'f1': f1, 
            'iou': iou,
            'auc': auc
        }
    except:
        return {
            'precision': 0, 
            'recall': 0, 
            'f1': 0, 
            'iou': 0,
            'auc': 0
        }


def train(model, data_in, loss, optim, max_epochs, model_dir, scheduler=None, test_interval=1, device=torch.device("cuda:0")):
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
        train_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0, 'auc': 0}

        # tqdm progress bar
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{max_epochs}", leave=False, ncols=100)

        for step, batch_data in pbar:
            volume = batch_data["vol"]
            label = batch_data["seg"]
            label = (label != 0).float()  # Convert to binary
            volume, label = volume.to(device), label.to(device)

            optim.zero_grad()
            outputs = model(volume)
            train_loss = loss(outputs, label)
            train_loss.backward()
            optim.step()

            train_epoch_loss += train_loss.item()
            train_metric = dice_metric(outputs, label)
            epoch_metric_train += train_metric
            
            # Calculate and accumulate additional metrics
            with torch.no_grad():
                batch_metrics = calculate_additional_metrics(label, torch.sigmoid(outputs))
                for key in train_metrics:
                    train_metrics[key] += batch_metrics[key]

            # Update progress bar description
            avg_loss = train_epoch_loss / (step + 1)
            avg_dice = epoch_metric_train / (step + 1)
            pbar.set_postfix({
                "Loss": f"{avg_loss:.4f}",
                "Dice": f"{avg_dice:.4f}",
                "Step": f"{step+1}/{len(train_loader)}"
            })

        # Calculate average metrics for this epoch
        train_epoch_loss /= len(train_loader)
        epoch_metric_train /= len(train_loader)
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)
        
        save_loss_train.append(train_epoch_loss)
        save_metric_train.append(epoch_metric_train)

        np.save(os.path.join(model_dir, 'loss_train.npy'), save_loss_train)
        np.save(os.path.join(model_dir, 'metric_train.npy'), save_metric_train)

        print(f"[Epoch {epoch+1}] Train Loss: {train_epoch_loss:.4f} | Dice: {epoch_metric_train:.4f} | " +
              f"Precision: {train_metrics['precision']:.4f} | Recall: {train_metrics['recall']:.4f} | " +
              f"F1: {train_metrics['f1']:.4f} | IoU: {train_metrics['iou']:.4f}")

        if (epoch + 1) % test_interval == 0:
            model.eval()
            test_epoch_loss = 0
            epoch_metric_test = 0
            test_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0, 'auc': 0}

            with torch.no_grad():
                for test_data in test_loader:
                    test_volume = test_data["vol"]
                    test_label = test_data["seg"]
                    test_label = (test_label != 0).float()
                    test_volume, test_label = test_volume.to(device), test_label.to(device)

                    test_outputs = model(test_volume)
                    test_loss = loss(test_outputs, test_label)
                    test_dice = dice_metric(test_outputs, test_label)
                    
                    # Calculate additional metrics
                    batch_metrics = calculate_additional_metrics(test_label, torch.sigmoid(test_outputs))
                    for key in test_metrics:
                        test_metrics[key] += batch_metrics[key]

                    test_epoch_loss += test_loss.item()
                    epoch_metric_test += test_dice

            test_epoch_loss /= len(test_loader)
            epoch_metric_test /= len(test_loader)
            for key in test_metrics:
                test_metrics[key] /= len(test_loader)

            save_loss_test.append(test_epoch_loss)
            save_metric_test.append(epoch_metric_test)
            np.save(os.path.join(model_dir, 'loss_test.npy'), save_loss_test)
            np.save(os.path.join(model_dir, 'metric_test.npy'), save_metric_test)

            print(f"[Epoch {epoch+1}] Test  Loss: {test_epoch_loss:.4f} | Dice: {epoch_metric_test:.4f} | " +
                  f"Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f} | " +
                  f"F1: {test_metrics['f1']:.4f} | IoU: {test_metrics['iou']:.4f}")
                  
            # Update learning rate based on validation performance if scheduler is provided
            if scheduler is not None:
                scheduler.step(epoch_metric_test)

            if epoch_metric_test > best_metric:
                best_metric = epoch_metric_test
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(model_dir, "best_tb_model.pth"))
                print(f"✅ New best model saved (Dice: {best_metric:.4f}) at epoch {best_metric_epoch}")

    print(f" Training completed. Best Dice: {best_metric:.4f} at epoch {best_metric_epoch}")


def show_patient(data, index=0, train=True, test=False):
    """
    Visualize a 2D image and its TB segmentation mask.
    """
    check_patient_train, check_patient_test = data

    if train:
        sample = list(check_patient_train)[index]
        img = sample["vol"][0]  # Shape: (1, H, W)
        mask = sample["seg"][0]

        plt.figure("TB Train Sample", (10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Lung Image")
        plt.imshow(img, cmap="gray")
        plt.subplot(1, 2, 2)
        plt.title("TB Lesion Mask")
        plt.imshow(mask, cmap="hot")  # Use a different colormap to highlight TB lesions
        plt.colorbar(label='TB Severity')
        plt.show()

    if test:
        sample = list(check_patient_test)[index]
        img = sample["vol"][0]
        mask = sample["seg"][0]

        plt.figure("TB Test Sample", (10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Lung Image")
        plt.imshow(img, cmap="gray")
        plt.subplot(1, 2, 2)
        plt.title("TB Lesion Mask")
        plt.imshow(mask, cmap="hot")  # Use a different colormap to highlight TB lesions
        plt.colorbar(label='TB Severity')
        plt.show()


def calculate_pixels(data):
    """
    Count background and foreground (TB lesion) pixels in 2D masks.
    """
    val = np.zeros((1, 2))
    for batch in tqdm(data, desc="Calculating class balance"):
        batch_label = batch["seg"]
        batch_label = batch_label != 0
        labels, counts = np.unique(batch_label.numpy(), return_counts=True)
        count_dict = dict(zip(labels, counts))
        background = count_dict.get(0, 0)
        foreground = count_dict.get(1, 0)
        val += np.array([[background, foreground]])

    # Calculate class imbalance ratio
    imbalance_ratio = val[0, 0] / val[0, 1] if val[0, 1] > 0 else float('inf')
    print('TB lesion pixels: {:,} ({:.2f}%)'.format(val[0, 1], 100 * val[0, 1] / (val[0, 0] + val[0, 1])))
    print('Background pixels: {:,} ({:.2f}%)'.format(val[0, 0], 100 * val[0, 0] / (val[0, 0] + val[0, 1])))
    print('Class imbalance ratio (background/TB): {:.2f}'.format(imbalance_ratio))
    
    return val


def calculate_class_weights_tbx11k():
    """
    Calculate class weights for the TBX11K dataset to handle class imbalance.
    These weights can be used in the loss function during training.
    
    Returns:
        torch.Tensor: Tensor of class weights for the 6 classes in TBX11K
    """
    # Class distribution in the training set
    class_counts = {
        "Healthy": 3000,
        "Sick & Non-TB": 3000, 
        "Active TB": 473,
        "Latent TB": 104,
        "Active & Latent TB": 23,
        "Uncertain TB": 0  # Only in test set
    }
    
    # Convert to list in the proper order
    counts = [
        class_counts["Healthy"],
        class_counts["Sick & Non-TB"],
        class_counts["Active TB"],
        class_counts["Latent TB"],
        class_counts["Active & Latent TB"],
        class_counts["Uncertain TB"]
    ]
    
    # Add a small constant to avoid division by zero
    counts = np.array(counts) + 1.0
    
    # Compute inverse frequency weights
    total = np.sum(counts)
    weights = total / (len(counts) * counts)
    
    # Normalize weights
    weights = weights / np.sum(weights) * len(counts)
    
    return torch.tensor(weights, dtype=torch.float32)


def get_tbx11k_class_names():
    """
    Return the class names for the TBX11K dataset.
    
    Returns:
        list: List of class names for the TBX11K dataset
    """
    return [
        "Healthy", 
        "Sick & Non-TB", 
        "Active TB", 
        "Latent TB", 
        "Active & Latent TB",
        "Uncertain TB"
    ]


def get_tb_categories():
    """
    Return information about the TB categories in the TBX11K dataset.
    
    Returns:
        dict: Dictionary with category information
    """
    return {
        "Active TB": {
            "id": 2,
            "description": "CXR images with only active TB"
        },
        "Latent TB": {
            "id": 3,
            "description": "CXR images with only latent TB"
        },
        "Active & Latent TB": {
            "id": 4,
            "description": "CXR images with both active and latent TB"
        },
        "Uncertain TB": {
            "id": 5,
            "description": "TB CXR images where the type of TB infection cannot be recognized"
        }
    }


def analyze_tbx11k_dataset_split():
    """
    Print information about the TBX11K dataset split.
    """
    data = {
        "Classes": ["Healthy", "Sick & Non-TB", "Active TB", "Latent TB", "Active & Latent TB", "Uncertain TB"],
        "Train": [3000, 3000, 473, 104, 23, 0],
        "Val": [800, 800, 157, 36, 7, 0],
        "Test": [1200, 1200, 294, 72, 24, 10],
        "Total": [5000, 5000, 924, 212, 54, 10]
    }
    
    print("\nTBX11K Dataset Split:")
    print("-" * 80)
    print(f"{'Classes':<20} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-" * 80)
    
    # Non-TB section
    print("Non-TB:")
    for i in range(2):  # Healthy and Sick & Non-TB
        print(f"  {data['Classes'][i]:<18} {data['Train'][i]:<10} {data['Val'][i]:<10} {data['Test'][i]:<10} {data['Total'][i]:<10}")
    
    # TB section
    print("TB:")
    for i in range(2, 6):  # All TB classes
        print(f"  {data['Classes'][i]:<18} {data['Train'][i]:<10} {data['Val'][i]:<10} {data['Test'][i]:<10} {data['Total'][i]:<10}")
    
    # Totals
    print("-" * 80)
    train_total = sum(data["Train"])
    val_total = sum(data["Val"])
    test_total = sum(data["Test"])
    grand_total = sum(data["Total"])
    print(f"{'Total':<20} {train_total:<10} {val_total:<10} {test_total:<10} {grand_total:<10}")
    
    # Class imbalance analysis
    print("\nClass Imbalance Analysis:")
    max_class = max(data["Total"])
    for i, cls in enumerate(data["Classes"]):
        ratio = max_class / data["Total"][i] if data["Total"][i] > 0 else float('inf')
        print(f"{cls:<20} 1:{ratio:.1f} ({data['Total'][i]/grand_total*100:.1f}% of dataset)")


def visualize_tb_detection(image, boxes, scores, labels, threshold=0.5, output_path=None):
    """
    Visualize TB detection results with bounding boxes and class labels.
    
    Args:
        image (numpy.ndarray): The input image (H, W, 3) or (H, W) for grayscale
        boxes (numpy.ndarray): Detected bounding boxes in format [x1, y1, x2, y2]
        scores (numpy.ndarray): Confidence scores for each box
        labels (numpy.ndarray): Class labels for each box
        threshold (float): Confidence threshold for displaying detections
        output_path (str, optional): Path to save the visualization. If None, display only.
        
    Returns:
        numpy.ndarray: The image with detection visualizations
    """
    # Make a copy of the image to avoid modifying the original
    if len(image.shape) == 2:  # If grayscale
        vis_image = np.stack([image] * 3, axis=2)  # Convert to RGB
    else:
        vis_image = image.copy()
    
    # Ensure vis_image is in 0-255 range and uint8 type
    if vis_image.max() <= 1.0:
        vis_image = (vis_image * 255).astype(np.uint8)
    
    # Get class names
    class_names = get_tbx11k_class_names()
    
    # Colors for different TB types (in BGR for OpenCV)
    colors = {
        0: (0, 255, 0),      # Healthy - Green
        1: (0, 255, 255),    # Sick & Non-TB - Yellow
        2: (0, 0, 255),      # Active TB - Red
        3: (255, 0, 0),      # Latent TB - Blue
        4: (255, 0, 255),    # Active & Latent TB - Purple
        5: (128, 128, 128)   # Uncertain TB - Gray
    }
    
    # Font settings
    font = 0  # Default font
    font_scale = 0.5
    thickness = 2
    
    # Draw all valid detections
    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue
            
        # Convert box to integers
        x1, y1, x2, y2 = [int(coord) for coord in box]
        
        # Get class name and color
        class_name = class_names[int(label)]
        color = colors.get(int(label), (255, 255, 255))  # Default to white
        
        # Draw the box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text
        label_text = f"{class_name}: {score:.2f}"
        
        # Calculate text size and position
        (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        cv2.rectangle(vis_image, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
        
        # Draw text
        cv2.putText(vis_image, label_text, (x1, y1 - 5),
                    font, font_scale, (255, 255, 255), thickness)
    
    # Display or save the image
    if output_path:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        print(f"Visualization saved to {output_path}")
    else:
        # Convert for matplotlib display
        plt.figure(figsize=(10, 8))
        plt.imshow(vis_image)
        plt.axis('off')
        plt.title("TB Detection Results")
        plt.tight_layout()
        plt.show()
    
    return vis_image

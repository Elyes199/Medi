import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tb_dataset import TBX11KDataset

def create_tb_detection_model(num_classes):
    """
    Create a Faster R-CNN model for TB detection.
    
    Args:
        num_classes: Number of TB-related classes to detect
        
    Returns:
        A PyTorch model for object detection
    """
    # Load a pre-trained Faster R-CNN model with ResNet-50 backbone
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    
    # Replace the classifier with a new one for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes + 1)  # +1 for background
    
    return model


def train_tb_detector(model, data_loader, optimizer, device, epochs, save_path, scheduler=None):
    """
    Train the TB detection model.
    
    Args:
        model: The model to train
        data_loader: DataLoader for training data
        optimizer: Optimizer for training
        device: Device to train on (CPU/GPU)
        epochs: Number of training epochs
        save_path: Path to save the model
        scheduler: Learning rate scheduler (optional)
    """
    model.to(device)
    
    # Initialize lists to store metrics
    losses_per_epoch = []
    classifier_losses = []
    box_reg_losses = []
    objectness_losses = []
    rpn_losses = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_loss_classifier = 0.0
        running_loss_box_reg = 0.0
        running_loss_objectness = 0.0
        running_loss_rpn = 0.0
        
        pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch+1}/{epochs}")
        
        for i, batch in pbar:
            # Move data to device
            images = [img.to(device) for img in batch['image']]
            targets = [{k: v.to(device) for k, v in t.items()} for t in batch['target']]
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss_dict = model(images, targets)
            
            # Calculate total loss
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass and optimize
            losses.backward()
            optimizer.step()
            
            # Update running losses
            running_loss += losses.item()
            running_loss_classifier += loss_dict.get('loss_classifier', 0).item()
            running_loss_box_reg += loss_dict.get('loss_box_reg', 0).item()
            running_loss_objectness += loss_dict.get('loss_objectness', 0).item()
            running_loss_rpn += loss_dict.get('loss_rpn_box_reg', 0).item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{running_loss/(i+1):.4f}", 
                'cls_loss': f"{running_loss_classifier/(i+1):.4f}", 
                'box_loss': f"{running_loss_box_reg/(i+1):.4f}"
            })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(data_loader)
        epoch_loss_classifier = running_loss_classifier / len(data_loader)
        epoch_loss_box_reg = running_loss_box_reg / len(data_loader)
        epoch_loss_objectness = running_loss_objectness / len(data_loader)
        epoch_loss_rpn = running_loss_rpn / len(data_loader)
        
        # Store metrics
        losses_per_epoch.append(epoch_loss)
        classifier_losses.append(epoch_loss_classifier)
        box_reg_losses.append(epoch_loss_box_reg)
        objectness_losses.append(epoch_loss_objectness)
        rpn_losses.append(epoch_loss_rpn)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, "
              f"Classifier: {epoch_loss_classifier:.4f}, Box: {epoch_loss_box_reg:.4f}")
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            scheduler.step()
        
        # Save the model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, os.path.join(save_path, f"tb_detector_epoch_{epoch+1}.pth"))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_path, "tb_detector_final.pth"))
    
    # Save loss curves
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(losses_per_epoch)
    plt.title('Total Loss')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(classifier_losses, label='Classifier')
    plt.plot(box_reg_losses, label='Box Regression')
    plt.plot(objectness_losses, label='Objectness')
    plt.plot(rpn_losses, label='RPN')
    plt.title('Component Losses')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_loss.png'))
    plt.close()
    
    # Save loss values
    np.save(os.path.join(save_path, 'total_loss.npy'), np.array(losses_per_epoch))
    np.save(os.path.join(save_path, 'classifier_loss.npy'), np.array(classifier_losses))
    np.save(os.path.join(save_path, 'box_reg_loss.npy'), np.array(box_reg_losses))
    
    print(f"Training complete. Model saved to {save_path}")
    return model


def evaluate_tb_detector(model, data_loader, device, score_threshold=0.5):
    """
    Evaluate the TB detection model on a validation set.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for validation data
        device: Device to evaluate on (CPU/GPU)
        score_threshold: Threshold for detection confidence
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        all_detections = []
        all_targets = []
        
        for batch in tqdm(data_loader, desc="Evaluating"):
            images = [img.to(device) for img in batch['image']]
            targets = [{k: v.to(device) for k, v in t.items()} for t in batch['target']]
            
            # Get predictions
            predictions = model(images)
            
            # Filter predictions by confidence threshold
            for pred, target in zip(predictions, targets):
                mask = pred['scores'] >= score_threshold
                filtered_pred = {
                    'boxes': pred['boxes'][mask],
                    'labels': pred['labels'][mask],
                    'scores': pred['scores'][mask]
                }
                all_detections.append(filtered_pred)
                all_targets.append(target)
    
    # TODO: Implement mAP calculation for evaluation
    
    return {"status": "Evaluation complete"}


def visualize_detection(model, dataset, idx, device, score_threshold=0.5, save_path=None):
    """
    Visualize TB detection on a sample image.
    
    Args:
        model: The detection model
        dataset: Dataset containing images
        idx: Index of the image to visualize
        device: Device (CPU/GPU)
        score_threshold: Confidence threshold for detections
        save_path: Path to save the visualization
    """
    model.to(device)
    model.eval()
    
    # Get sample
    sample = dataset[idx]
    image = sample['image'].unsqueeze(0).to(device)
    target = sample['target']
    
    # Get predictions
    with torch.no_grad():
        predictions = model(image)
    
    # Convert tensor to numpy for visualization
    image_np = sample['image'].permute(1, 2, 0).numpy()
    
    # Normalize image for display if needed
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    
    # Get category names from dataset if possible
    category_names = getattr(dataset, 'get_category_names', lambda: {})()
    
    plt.figure(figsize=(12, 6))
    
    # Plot original image with ground truth
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.imshow(image_np)
    
    # Draw ground truth boxes
    for box, label in zip(target['boxes'].numpy(), target['labels'].numpy()):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                            fill=False, edgecolor='green', linewidth=2)
        plt.gca().add_patch(rect)
        label_name = category_names.get(label, str(label))
        plt.text(x1, y1, label_name, 
                bbox=dict(facecolor='green', alpha=0.5),
                fontsize=8, color='white')
    
    # Plot predictions
    plt.subplot(1, 2, 2)
    plt.title("Predictions")
    plt.imshow(image_np)
    
    # Filter predictions by confidence
    pred = predictions[0]
    mask = pred['scores'] >= score_threshold
    boxes = pred['boxes'][mask].cpu().numpy()
    labels = pred['labels'][mask].cpu().numpy()
    scores = pred['scores'][mask].cpu().numpy()
    
    # Draw prediction boxes
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                            fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
        label_name = category_names.get(label, str(label))
        plt.text(x1, y1, f"{label_name}: {score:.2f}", 
                bbox=dict(facecolor='red', alpha=0.5),
                fontsize=8, color='white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
    return predictions[0]

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tb_dataset import TBX11KDataset
import torchvision.transforms as T
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

def load_tb_detection_model(model_path, num_classes=3, device=None):
    """
    Load a trained TB detection model.
    
    Args:
        model_path: Path to the saved model weights
        num_classes: Number of TB classes
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Import here to avoid circular imports
    from tb_detection import create_tb_detection_model
    
    model = create_tb_detection_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model


def detect_tb(model, image_path, transforms=None, device=None, score_threshold=0.5):
    """
    Detect tuberculosis in a single image.
    
    Args:
        model: The detection model
        image_path: Path to the image
        transforms: Image transforms to apply
        device: Device to run inference on
        score_threshold: Threshold for detection confidence
        
    Returns:
        Detected boxes, labels, scores and processed image
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    
    if transforms is None:
        transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    img_tensor = transforms(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        predictions = model(img_tensor)
    
    # Extract predictions
    pred = predictions[0]
    
    # Filter by confidence
    mask = pred['scores'] >= score_threshold
    boxes = pred['boxes'][mask].cpu()
    labels = pred['labels'][mask].cpu()
    scores = pred['scores'][mask].cpu()
    
    # Convert PIL image to numpy for visualization
    image_np = np.array(image)
    
    return {
        'boxes': boxes,
        'labels': labels,
        'scores': scores,
        'image': image_np
    }


def visualize_tb_detection(detection_result, category_names=None, save_path=None):
    """
    Visualize TB detection results.
    
    Args:
        detection_result: Results from detect_tb function
        category_names: Dictionary mapping class IDs to names
        save_path: Path to save visualization
        
    Returns:
        None
    """
    image = detection_result['image']
    boxes = detection_result['boxes']
    labels = detection_result['labels']
    scores = detection_result['scores']
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot original image
    plt.imshow(image)
    
    # If no categories provided, use generic names
    if category_names is None:
        category_names = {
            1: "TB",
            2: "Suspicious",
            3: "Other"
        }
    
    # Draw boxes and labels
    for box, label, score in zip(boxes.numpy(), labels.numpy(), scores.numpy()):
        x1, y1, x2, y2 = box.astype(int)
        class_name = category_names.get(label.item(), f"Class {label}")
        
        # Draw rectangle
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, 
                             edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
        
        # Draw label
        plt.text(x1, y1-10, f"{class_name}: {score:.2f}", 
                 bbox=dict(facecolor='red', alpha=0.5),
                 fontsize=12, color='white')
    
    plt.title("Tuberculosis Detection Results")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.tight_layout()
    plt.show()
    
    # Also return a summary of detections
    print(f"Detected {len(boxes)} TB-related findings:")
    for i, (label, score) in enumerate(zip(labels, scores)):
        class_name = category_names.get(label.item(), f"Class {label}")
        print(f"  {i+1}. {class_name} (confidence: {score:.2f})")


def create_heatmap_overlay(detection_result, save_path=None):
    """
    Create a heatmap overlay based on TB detection boxes.
    
    Args:
        detection_result: Results from detect_tb function
        save_path: Path to save heatmap overlay
        
    Returns:
        Heatmap image
    """
    image = detection_result['image']
    boxes = detection_result['boxes'].numpy()
    scores = detection_result['scores'].numpy()
    
    # Create an empty heatmap
    heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    
    # Add gaussian spots for each detection, weighted by confidence
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.astype(int)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1
        
        # Create gaussian kernel with size proportional to the box
        sigma_x = width / 6  # Standard deviation
        sigma_y = height / 6
        x = np.arange(0, image.shape[1], 1, float)
        y = np.arange(0, image.shape[0], 1, float)
        x_grid, y_grid = np.meshgrid(x, y)
        
        # Create gaussian
        gaussian = score * np.exp(
            -((x_grid - center_x)**2 / (2 * sigma_x**2) + 
              (y_grid - center_y)**2 / (2 * sigma_y**2))
        )
        
        # Add to heatmap
        heatmap += gaussian
    
    # Normalize heatmap
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # Convert heatmap to colormap
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Convert image to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Overlay heatmap on original image
    overlay = cv2.addWeighted(image_bgr, 0.7, heatmap_colored, 0.3, 0)
    
    # Convert back to RGB for matplotlib
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        print(f"Heatmap overlay saved to {save_path}")
    
    return overlay_rgb


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect TB in chest X-ray images using object detection')
    parser.add_argument('--image', type=str, required=True, 
                        help='Path to the image to detect TB in')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained TB detection model')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the visualization')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection confidence threshold')
    parser.add_argument('--heatmap', action='store_true',
                        help='Generate heatmap overlay')
    
    args = parser.parse_args()
    
    # Load model
    model = load_tb_detection_model(args.model)
    
    # Detect TB
    detections = detect_tb(model, args.image, score_threshold=args.threshold)
    
    # Visualize results
    visualize_tb_detection(detections, save_path=args.output)
    
    # If heatmap requested, create and show it
    if args.heatmap:
        heatmap_path = args.output.replace('.png', '_heatmap.png') if args.output else None
        heatmap = create_heatmap_overlay(detections, save_path=heatmap_path)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(heatmap)
        plt.title("TB Detection Heatmap")
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    main()

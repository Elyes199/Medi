import os
import torch
import torchvision.transforms as T
from tuberculosis.tb_dataset import TBX11KDataset
from tuberculosis.tb_detection import create_tb_detection_model
from tuberculosis.tb_detect_bbox import detect_tb, visualize_tb_detection, create_heatmap_overlay
from tuberculosis.tb_detect import process_image as segment_tb
import matplotlib.pyplot as plt
import argparse
import numpy as np
import cv2

def integrate_detection_segmentation(image_path, detection_model_path, segmentation_model_path,
                                    num_classes=3, threshold=0.5, save_path=None):
    """
    Perform integrated TB analysis using both object detection and segmentation models.
    
    Args:
        image_path: Path to the chest X-ray image
        detection_model_path: Path to the object detection model weights
        segmentation_model_path: Path to the segmentation model weights
        num_classes: Number of TB classes for object detection
        threshold: Detection confidence threshold
        save_path: Path to save the visualization
        
    Returns:
        Combined analysis results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if models and image exist
    if not os.path.exists(detection_model_path):
        raise FileNotFoundError(f"Detection model not found at {detection_model_path}")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Load detection model
    print(f"Loading detection model from {detection_model_path}...")
    detection_model = create_tb_detection_model(num_classes)
    detection_model.load_state_dict(torch.load(detection_model_path, map_location=device))
    detection_model.to(device)
    detection_model.eval()
    
    # Run object detection
    print(f"Running object detection on {image_path}...")
    detections = detect_tb(detection_model, image_path, score_threshold=threshold)
    
    # Run segmentation if model path is provided
    segmentation_results = None
    if segmentation_model_path and os.path.exists(segmentation_model_path):
        try:
            print(f"Running TB segmentation on {image_path}...")
            segmentation_results = segment_tb(image_path)
        except Exception as e:
            print(f"Segmentation failed: {e}")
            segmentation_results = None
    
    # Generate combined visualization
    if segmentation_results:
        # Create a figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(detections['image'])
        axes[0].set_title("Original Chest X-ray")
        axes[0].axis('off')
        
        # Detection results
        axes[1].imshow(detections['image'])
        axes[1].set_title("TB Detection (Object Detection)")
        axes[1].axis('off')
        
        # Draw detection boxes
        for box, label, score in zip(detections['boxes'].numpy(), 
                                    detections['labels'].numpy(), 
                                    detections['scores'].numpy()):
            x1, y1, x2, y2 = box.astype(int)
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, 
                                 edgecolor='red', linewidth=2)
            axes[1].add_patch(rect)
            axes[1].text(x1, y1-5, f"TB: {score:.2f}", 
                        bbox=dict(facecolor='red', alpha=0.5),
                        fontsize=8, color='white')
        
        # Segmentation results - use the probability map from segmentation
        im = axes[2].imshow(segmentation_results['probability_map'], cmap='hot')
        axes[2].set_title(f"TB Segmentation (Lesion Area: {segmentation_results['lesion_percentage']:.2f}%)")
        axes[2].axis('off')
        fig.colorbar(im, ax=axes[2], shrink=0.6)
    else:
        # Only show detection if segmentation is not available
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        axes[0].imshow(detections['image'])
        axes[0].set_title("Original Chest X-ray")
        axes[0].axis('off')
        
        # Detection results
        axes[1].imshow(detections['image'])
        axes[1].set_title("TB Detection (Object Detection)")
        axes[1].axis('off')
        
        # Draw detection boxes
        for box, label, score in zip(detections['boxes'].numpy(), 
                                    detections['labels'].numpy(), 
                                    detections['scores'].numpy()):
            x1, y1, x2, y2 = box.astype(int)
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, 
                                 edgecolor='red', linewidth=2)
            axes[1].add_patch(rect)
            axes[1].text(x1, y1-5, f"TB: {score:.2f}", 
                        bbox=dict(facecolor='red', alpha=0.5),
                        fontsize=8, color='white')
    
    plt.tight_layout()
    
    # Save visualization if requested
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    # Show visualization
    plt.show()
    
    # Create analysis report
    report = {
        'detection': {
            'boxes': detections['boxes'].numpy().tolist() if len(detections['boxes']) else [],
            'scores': detections['scores'].numpy().tolist() if len(detections['scores']) else [],
            'labels': detections['labels'].numpy().tolist() if len(detections['labels']) else [],
            'detected': len(detections['boxes']) > 0
        }
    }
    
    if segmentation_results:
        report['segmentation'] = {
            'detected': segmentation_results['detected'],
            'lesion_percentage': segmentation_results['lesion_percentage'],
            'severity': segmentation_results['severity']
        }
        
        # Check if detection and segmentation agree
        detection_positive = len(detections['boxes']) > 0
        segmentation_positive = segmentation_results['detected']
        
        if detection_positive and segmentation_positive:
            report['conclusion'] = "TB DETECTED (confirmed by both detection and segmentation)"
            report['confidence'] = "High"
        elif detection_positive and not segmentation_positive:
            report['conclusion'] = "SUSPICIOUS (detected by object detection only)"
            report['confidence'] = "Medium"
        elif not detection_positive and segmentation_positive:
            report['conclusion'] = "SUSPICIOUS (detected by segmentation only)"
            report['confidence'] = "Medium"
        else:
            report['conclusion'] = "NO TB DETECTED"
            report['confidence'] = "High"
    else:
        report['conclusion'] = "TB DETECTED" if len(detections['boxes']) > 0 else "NO TB DETECTED"
        report['confidence'] = "Medium"  # Only one model was used
    
    # Print report
    print("\n===== TB ANALYSIS REPORT =====")
    print(f"Image: {os.path.basename(image_path)}")
    
    if 'segmentation' in report:
        print(f"\nSegmentation Analysis:")
        print(f"- TB Detected: {'Yes' if report['segmentation']['detected'] else 'No'}")
        print(f"- Lesion Area: {report['segmentation']['lesion_percentage']:.2f}%")
        print(f"- Severity: {report['segmentation']['severity']}")
    
    print(f"\nObject Detection Analysis:")
    print(f"- TB Regions Detected: {len(report['detection']['boxes'])}")
    if report['detection']['boxes']:
        for i, (score, label) in enumerate(zip(report['detection']['scores'], report['detection']['labels'])):
            print(f"  Region {i+1}: Confidence {score:.2f}, Class {label}")
    
    print(f"\nConclusion: {report['conclusion']} (Confidence: {report['confidence']})")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Integrated TB analysis using object detection and segmentation")
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the chest X-ray image')
    parser.add_argument('--detection-model', type=str, required=True,
                        help='Path to the object detection model weights')
    parser.add_argument('--segmentation-model', type=str, default=None,
                        help='Path to the segmentation model weights (optional)')
    parser.add_argument('--num-classes', type=int, default=3,
                        help='Number of TB classes for object detection')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection confidence threshold')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the visualization')
    
    args = parser.parse_args()
    
    try:
        report = integrate_detection_segmentation(
            args.image,
            args.detection_model,
            args.segmentation_model,
            args.num_classes,
            args.threshold,
            args.output
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()

import os
import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import argparse

from tb_dataset import TBX11KDataset
from tb_detection import create_tb_detection_model, train_tb_detector, evaluate_tb_detector, visualize_detection

# Default paths - these will be overridden by command line arguments
DEFAULT_ANN_FILE = "tb_data/annotations/tbx11k_train.json"
DEFAULT_IMG_PREFIX = "tb_data/images/"
DEFAULT_MODEL_DIR = "tuberculosis/models/detection"
DEFAULT_NUM_CLASSES = 3  # Typically TB has classes like "TB", "suspicious", etc.

def setup_tb_detection(ann_file, img_prefix, model_dir, num_classes, batch_size=4):
    """
    Set up the TBX11K dataset and model for tuberculosis detection.
    
    Args:
        ann_file: Path to annotation file in COCO format
        img_prefix: Path to directory containing images
        model_dir: Directory to save trained models
        num_classes: Number of TB classes to detect
        batch_size: Batch size for training
        
    Returns:
        model, train_loader, val_loader, optimizer
    """
    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    # Set up transforms
    transforms = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = TBX11KDataset(ann_file, img_prefix, transforms=transforms)
    
    # Split dataset: 80% train, 20% validation
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    def collate_fn(batch):
        return {
            'image': [item['image'] for item in batch],
            'target': [item['target'] for item in batch]
        }
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=collate_fn, num_workers=2)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=collate_fn, num_workers=2)
    
    # Create model
    model = create_tb_detection_model(num_classes)
    
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Set up scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    return model, train_loader, val_loader, optimizer, scheduler


def main():
    parser = argparse.ArgumentParser(description='Train a TB detection model using TBX11K dataset')
    parser.add_argument('--ann_file', type=str, default=DEFAULT_ANN_FILE,
                        help='Path to the COCO annotation file')
    parser.add_argument('--img_prefix', type=str, default=DEFAULT_IMG_PREFIX,
                        help='Path to the directory containing images')
    parser.add_argument('--model_dir', type=str, default=DEFAULT_MODEL_DIR,
                        help='Directory to save trained models')
    parser.add_argument('--num_classes', type=int, default=DEFAULT_NUM_CLASSES,
                        help='Number of TB classes to detect')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only run evaluation on a trained model')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize detections on sample images')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to a saved model for evaluation or visualization')
    
    args = parser.parse_args()
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set up dataset and model
    model, train_loader, val_loader, optimizer, scheduler = setup_tb_detection(
        args.ann_file, args.img_prefix, args.model_dir, args.num_classes, args.batch_size)
    
    # If a model path is provided, load the model
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    if args.eval_only:
        # Evaluate the model
        print("Evaluating model...")
        results = evaluate_tb_detector(model, val_loader, device)
        print(results)
        
    elif args.visualize:
        # Visualize detections on a few samples
        print("Visualizing detections...")
        dataset = train_loader.dataset
        
        # Visualize 5 random samples
        for i in range(5):
            idx = i % len(dataset)
            save_path = os.path.join(args.model_dir, f"detection_sample_{idx}.png")
            visualize_detection(model, dataset, idx, device, save_path=save_path)
    
    else:
        # Train the model
        print("Training model...")
        train_tb_detector(model, train_loader, optimizer, device, args.epochs, 
                         args.model_dir, scheduler=scheduler)
        
        # Evaluate the model
        print("Evaluating model...")
        results = evaluate_tb_detector(model, val_loader, device)
        print(results)


if __name__ == "__main__":
    main()

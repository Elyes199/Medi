import torch
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstD, ScaleIntensityRanged, Resize, ToTensord
from monai.networks.nets import UNet
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tb_preprocess import RGBToGray

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the TB detection model
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=3,
    dropout=0.2,
).to(device)

# Load the trained model weights
model.load_state_dict(torch.load("tuberculosis/models/best_tb_model.pth", map_location=device))
model.eval()

# Setup transforms for inference
infer_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstD(keys=["image"]),
    RGBToGray(keys=["image"]),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
    Resize(spatial_size=(256, 256), mode="bilinear"),
    ToTensord(keys=["image"])
])

def process_image(img_path):
    """Process a chest X-ray image and detect TB lesions."""
    test_data = {"image": img_path}
    input_data = infer_transforms(test_data)
    input_tensor = input_data["image"].unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)
    
    # Get the prediction mask
    pred_mask = output.squeeze().cpu().numpy()
    
    # Load the original image for visualization
    original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize to match the mask size
    if original_img.shape != (256, 256):
        original_img = cv2.resize(original_img, (256, 256))
    
    # Apply threshold to get binary mask
    binary_mask = (pred_mask > 0.5).astype(np.uint8)
    
    # Calculate TB stats
    lesion_pixels = np.sum(binary_mask)
    total_pixels = binary_mask.size
    lesion_percentage = (lesion_pixels / total_pixels) * 100
    
    # Create colored overlay for visualization
    heatmap = cv2.applyColorMap((pred_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Convert grayscale original image to RGB
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    
    # Create overlay
    overlay = cv2.addWeighted(original_rgb, 0.7, heatmap, 0.3, 0)
    
    # Add contours of the detected TB regions
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Chest X-ray")
    plt.imshow(original_img, cmap="gray")
    
    plt.subplot(1, 3, 2)
    plt.title("TB Lesion Probability Map")
    plt.imshow(pred_mask, cmap="jet")
    plt.colorbar(label='Probability')
    
    plt.subplot(1, 3, 3)
    plt.title(f"TB Detection Overlay\nLesion Area: {lesion_percentage:.2f}%")
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    
    plt.tight_layout()
    plt.show()
    
    # Return diagnosis info
    severity = "None"
    if lesion_percentage < 0.5:
        severity = "None/Minimal"
    elif lesion_percentage < 2:
        severity = "Mild"
    elif lesion_percentage < 5:
        severity = "Moderate"
    else:
        severity = "Severe"
        
    return {
        "detected": lesion_pixels > 0,
        "lesion_percentage": lesion_percentage,
        "severity": severity,
        "binary_mask": binary_mask,
        "probability_map": pred_mask
    }

# Example usage
if __name__ == "__main__":
    # Replace with your test image path
    img_path = "tb_data/TestImages/sample.png"
    
    result = process_image(img_path)
    
    if result["detected"]:
        print(f"TB DETECTED - Severity: {result['severity']}")
        print(f"Affected lung area: {result['lesion_percentage']:.2f}%")
    else:
        print("No TB detected")

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO

class TBX11KDataset(Dataset):
    def __init__(self, ann_file, img_prefix, transforms=None, image_set=None):
        """
        TBX11K Dataset for tuberculosis detection using the COCO format.
        
        Args:
            ann_file: Path to the annotation file in COCO format
            img_prefix: Path to the directory containing images
            transforms: Transforms to be applied to the images
            image_set: Optional path to a text file listing image IDs to use (e.g., tbx11k_train.txt)
        """
        self.coco = COCO(ann_file)
        self.img_prefix = img_prefix
        self.transforms = transforms
        
        # Get category mapping (id to name)
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in cats}
        self.cat_name_to_id = {cat['name']: cat['id'] for cat in cats}
        
        # Get image IDs
        if image_set and os.path.exists(image_set):
            # Load image IDs from file
            with open(image_set, 'r') as f:
                image_names = [line.strip() for line in f.readlines()]
                
            # Convert file names to image IDs
            self.img_ids = []
            all_imgs = self.coco.imgs
            for img_id, img_info in all_imgs.items():
                file_name = img_info['file_name'].split('/')[-1]  # e.g., tb0005.png
                if file_name in image_names:
                    self.img_ids.append(img_id)
        else:
            # Use all images in the dataset
            self.img_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        # Handle nested directory structure (e.g., "tb/tb0005.png")
        file_path = img_info['file_name']
        img_path = os.path.join(self.img_prefix, file_path)
        
        # Load the image
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            # If file not found in the expected subdirectory, try to find it directly in img_prefix
            img_path = os.path.join(self.img_prefix, os.path.basename(file_path))
            image = Image.open(img_path).convert('RGB')

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # For detection: bounding boxes and labels
        boxes = []
        labels = []
        for ann in anns:
            # COCO bbox format: [x, y, width, height]
            # Convert to [x1, y1, x2, y2] format for PyTorch
            x, y, width, height = ann['bbox']
            boxes.append([x, y, x + width, y + height])
            labels.append(ann['category_id'])

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4)),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            'image_id': torch.tensor([img_id])
        }

        if self.transforms:
            image = self.transforms(image)

        return {'image': image, 'target': target}
    
    def get_category_names(self):
        """Return a dictionary of category IDs to names"""
        return self.cat_id_to_name

import os
from glob import glob
from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    LoadImaged,
    Resized,
    ToTensord,
    MapTransform,
    ScaleIntensityRanged,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandShiftIntensityd,
    RandHistogramShiftd,
    RandGaussianSmoothd,
    Spacingd,
    CropForegroundd,
)
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism


class RGBToGray(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            x = d[key]
            # x shape: (C, H, W)
            if x.shape[0] == 3:  # RGB image
                d[key] = x.mean(dim=0, keepdim=True)  # convert to grayscale
        return d


def prepare_tb(in_dir, pixdim=(1.5, 1.5, 1.0), a_min=0, a_max=255, spatial_size=[256, 256], cache=True):
    """
    Preprocessing pipeline for tuberculosis segmentation.
    Added more data augmentations to help with lesion detection and model generalization.
    """

    set_determinism(seed=42)  # Different seed for reproducibility

    path_train_volumes = sorted(glob(os.path.join(in_dir, "TrainImages", "*.png")))
    path_train_segmentation = sorted(glob(os.path.join(in_dir, "TrainMasks", "*.png")))

    path_test_volumes = sorted(glob(os.path.join(in_dir, "TestImages", "*.png")))
    path_test_segmentation = sorted(glob(os.path.join(in_dir, "TestMasks", "*.png")))

    train_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in
                   zip(path_train_volumes, path_train_segmentation)]
    test_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in
                  zip(path_test_volumes, path_test_segmentation)]

    # Training transforms include data augmentation for better generalization
    train_transforms = Compose([
        LoadImaged(keys=["vol", "seg"]),
        EnsureChannelFirstD(keys=["vol", "seg"]),
        RGBToGray(keys=["vol"]),
        ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=["vol", "seg"], spatial_size=spatial_size),
        # Data augmentations for better model generalization
        RandFlipd(keys=["vol", "seg"], prob=0.5),
        RandRotate90d(keys=["vol", "seg"], prob=0.5, max_k=3),
        RandGaussianNoised(keys=["vol"], prob=0.3, mean=0.0, std=0.1),
        RandAdjustContrastd(keys=["vol"], prob=0.3, gamma=(0.7, 1.3)),
        RandShiftIntensityd(keys=["vol"], prob=0.3, offsets=0.1),
        RandHistogramShiftd(keys=["vol"], prob=0.2, num_control_points=10),
        RandGaussianSmoothd(keys=["vol"], prob=0.2, sigma_x=(0.5, 1.0)),
        ToTensord(keys=["vol", "seg"]),
    ])

    # Test transforms (no augmentation)
    test_transforms = Compose([
        LoadImaged(keys=["vol", "seg"]),
        EnsureChannelFirstD(keys=["vol", "seg"]),
        RGBToGray(keys=["vol"]),
        ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=["vol", "seg"], spatial_size=spatial_size),
        ToTensord(keys=["vol", "seg"]),
    ])

    if cache:
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)

        test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0)
        test_loader = DataLoader(test_ds, batch_size=4, num_workers=4)

        return train_loader, test_loader

    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

        test_ds = Dataset(data=test_files, transform=test_transforms)
        test_loader = DataLoader(test_ds, batch_size=4)

        return train_loader, test_loader

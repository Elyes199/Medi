import os
from glob import glob
from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    LoadImaged,
    Resized,
    ToTensord,
    MapTransform,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
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

def prepare(in_dir, pixdim=(1.5, 1.5, 1.0), a_min=-200, a_max=200, spatial_size=[128, 128, 64], cache=True):
    """
    This function is for preprocessing, it contains only the basic transforms, but you can add more operations that you
    find in the Monai documentation.
    https://monai.io/docs.html
    """

    set_determinism(seed=0)

    path_train_volumes = sorted(glob(os.path.join(in_dir, "TrainImages", "*.png")))
    path_train_segmentation = sorted(glob(os.path.join(in_dir, "TrainMasks", "*.png")))

    path_test_volumes = sorted(glob(os.path.join(in_dir, "TestImages", "*.png")))
    path_test_segmentation = sorted(glob(os.path.join(in_dir, "TestMasks", "*.png")))


    train_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in
                   zip(path_train_volumes, path_train_segmentation)]
    test_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in
                  zip(path_test_volumes, path_test_segmentation)]

    common_transforms = Compose([
    LoadImaged(keys=["vol", "seg"]),
    EnsureChannelFirstD(keys=["vol", "seg"]),
    RGBToGray(keys=["vol"]),
    ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
    Resized(keys=["vol", "seg"], spatial_size=spatial_size),
    ToTensord(keys=["vol", "seg"]),
])



    if cache:
        train_ds = CacheDataset(data=train_files, transform=common_transforms, cache_rate=1.0)
        train_loader = DataLoader(train_ds, batch_size=1)

        test_ds = CacheDataset(data=test_files, transform=common_transforms, cache_rate=1.0)
        test_loader = DataLoader(test_ds, batch_size=1)

        return train_loader, test_loader

    else:
        train_ds = Dataset(data=train_files, transform=common_transforms)
        train_loader = DataLoader(train_ds, batch_size=1)

        test_ds = Dataset(data=test_files, transform=common_transforms)
        test_loader = DataLoader(test_ds, batch_size=1)

        return train_loader, test_loader
from pathlib import Path

import SimpleITK as sitk
import torch
from monai.data import decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscreted,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    NormalizeIntensityd,
    SaveImaged,
    Spacingd,
)


def test_itk_pipeline(tmp_path: Path):
    image = sitk.Image(256, 256, 145, sitk.sitkFloat32)
    image[:, :, :] = 0
    image[10:-20, 10:-10, 17:-8] = 1
    assert image.GetDimension() == 3

    file_path = str(tmp_path / "test_image.nii.gz")
    sitk.WriteImage(image, file_path)

    pre_transforms = Compose(
        [
            LoadImaged(keys="image", reader="ITKReader"),
            EnsureChannelFirstd(keys="image"),
            # Orientationd(keys=keys, axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True),
            CropForegroundd(keys="image", source_key="image"),
            Spacingd(keys="image", pixdim=[1.0] * 3),
            EnsureTyped(keys="image"),
        ]
    )

    post_transforms = Compose(
        [
            Invertd(
                keys="pred",
                transform=pre_transforms,
                orig_keys="image",
                nearest_interp=False,
            ),
            AsDiscreted(keys="pred", threshold=0.5),
            SaveImaged(keys="pred", output_dir=str(tmp_path), resample=False),
        ]
    )

    # pre-process
    test_data = pre_transforms({"image": file_path})

    print(test_data["image"].shape)

    # skip inference, directly post-process
    # def rename_keys(key: str):
    #    if key.startswith("image"):
    #        return key.replace("image", "pred")
    #    return key
    # pred = {rename_keys(k): v for k, v in input.items()}

    device = torch.device("cpu")
    net = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    # net.load_state_dict(torch.load("best_metric_model_segmentation3d_dict.pth"))

    inferer = SlidingWindowInferer(
        roi_size=(96, 96, 96), sw_batch_size=4, device=device
    )

    net.eval()
    with torch.no_grad():
        val_pred = inferer(test_data["image"].to(device), net)

        test_data["pred"] = val_pred
        for i in decollate_batch(test_data):
            post_transforms(i)

    # post_transforms(pred)

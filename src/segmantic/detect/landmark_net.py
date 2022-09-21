from typing import Callable, List, Sequence

import pytorch_lightning as pl
import torch
from monai.data import decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference
from monai.metrics import MAEMetric
from monai.networks.nets.attentionunet import AttentionUnet
from monai.transforms import (
    Activationsd,
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureType,
    EnsureTyped,
    GaussianSmoothd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandZoomd,
    ScaleIntensityd,
    SelectItemsd,
    Spacingd,
    SpatialPadd,
    ToNumpyd,
)
from torch.nn import L1Loss

from ..seg.dataset import PairedDataSet
from .transforms import ExtractVertPosition, VertHeatMap


class LandmarkNet(pl.LightningModule):
    dataset: PairedDataSet

    def __init__(
        self,
        landmark_names: List[str],
        num_channels: int = 1,
        spatial_dims: int = 3,
        spatial_size: Sequence[int] = None,
        spacing: Sequence[float] = None,
    ):
        super().__init__()

        self.save_hyperparameters()

        num_landmarks = len(landmark_names)
        self._model = AttentionUnet(
            spatial_dims=spatial_dims,
            in_channels=num_channels,
            out_channels=num_landmarks,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            kernel_size=3,
            up_kernel_size=3,
            dropout=0.0,
        )
        self.landmark_names = landmark_names
        self.spatial_size = spatial_size if spatial_size else [96] * 3
        self.spacing = spacing if spacing else [1.0] * 3
        self.loss_function = L1Loss()

        # TODO: post_pred, post_label
        self.post_pred = Compose(
            [
                EnsureType(),
                AsDiscrete(argmax=True, to_onehot=num_landmarks),
            ]
        )
        self.post_label = Compose([EnsureType(), AsDiscrete(to_onehot=num_landmarks)])
        self.error_metric = MAEMetric(reduction="mean", get_not_nans=False)

    @property
    def num_classes(self):
        return self._model.out_channels

    @property
    def spatial_dims(self):
        return self._model.dimensions

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return [
            LoadImaged(keys=("image", "label"), reader="ITKReader"),
            EnsureChannelFirstd(keys=("image", "label")),
            Orientationd(keys=("image", "label"), axcodes="RAS"),
            Spacingd(
                keys=("image", "label"),
                pixdim=self.spacing,
                mode=("bilinear", "nearest"),
            ),
            CropForegroundd(keys=("image", "label"), source_key="label"),
            VertHeatMap(keys="label", label_names=self.landmark_names),
            GaussianSmoothd(keys="image", sigma=0.75),
            NormalizeIntensityd(keys="image", divisor=2048.0),  # type: ignore [arg-type]
            ScaleIntensityd(keys="image", minv=-1.0, maxv=1.0),
            RandScaleIntensityd(keys="image", factors=(0.75, 1.25), prob=0.80),
            RandShiftIntensityd(keys="image", offsets=(-0.25, 0.25), prob=0.80),
            RandRotated(
                keys=("image", "label"),
                range_x=(-0.26, 0.26),
                range_y=(-0.26, 0.26),
                range_z=(-0.26, 0.26),
                prob=0.80,
            ),
            RandZoomd(keys=("image", "label"), prob=0.70, min_zoom=0.85, max_zoom=1.15),
            SpatialPadd(keys="image", spatial_size=self.spatial_size),
            SelectItemsd(keys=("image", "label")),
        ]

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Activationsd(keys="pred", other=torch.nn.functional.leaky_relu),
            # Activationsd(keys="pred", sigmoid=True),
            ToNumpyd(keys="pred"),
            # Restored(keys="pred", ref_image="image"),
            ScaleIntensityd(keys="pred", minv=0.0, maxv=100.0),
            ExtractVertPosition(keys="pred"),
        ]

    def forward(self, x):
        return self._model(x)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            collate_fn=list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_ds, batch_size=1, num_workers=0
        )
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), 1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        tensorboard_logs = {"train_loss": loss.item()}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = tuple(160 for _ in range(self.spatial_dims))
        sw_batch_size = 4
        outputs = sliding_window_inference(
            images, roi_size, sw_batch_size, self.forward
        )
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        return {"val_loss": loss, "val_number": len(outputs)}

    def validation_epoch_end(self, outputs):
        val_loss, num_items = 0, 0
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
        }
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
            f"\ncurrent epoch: {self.current_epoch} "
            f"mean val dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.log("val_dice", mean_val_dice)
        self.log("val_loss", mean_val_loss)
        return {"log": tensorboard_logs}

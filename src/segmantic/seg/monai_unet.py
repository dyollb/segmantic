from monai.transforms.spatial.dictionary import Spacingd
from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    AddChanneld,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByLabelClassesd,
    RandFlipd,
    NormalizeIntensityd,
    EnsureTyped,
    EnsureType,
    SaveImaged,
    Invertd,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate, decollate_batch, NiftiSaver
from monai.networks.utils import one_hot
import numpy as np
import torch
import torch.utils.data
import pytorch_lightning
import pytorch_lightning.loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os
import json
from typing import List, Optional, Dict, Sequence
from pathlib import Path

from .evaluation import confusion_matrix
from .utils import make_device
from .dataset import DataSet
from .visualization import make_random_cmap, plot_confusion_matrix


class Net(pytorch_lightning.LightningModule):
    def __init__(
        self,
        num_classes: int,
        num_channels: int = 1,
        spatial_dims: int = 3,
        spatial_size: Sequence[int] = None,
        dataset: Optional[DataSet] = None,
    ):
        super().__init__()
        self._model = UNet(
            spatial_dims=spatial_dims,
            in_channels=num_channels,
            out_channels=num_classes,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )
        self.dataset = dataset
        self.spatial_size = spatial_size
        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.post_pred = Compose(
            [
                EnsureType(),
                AsDiscrete(argmax=True, to_onehot=True, n_classes=num_classes),
            ]
        )
        self.post_label = Compose(
            [EnsureType(), AsDiscrete(to_onehot=True, n_classes=num_classes)]
        )
        self.dice_metric = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False
        )
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.save_hyperparameters(
            "num_classes", "num_channels", "spatial_dims", "spatial_size"
        )

    @property
    def num_classes(self):
        return self._model.out_channels

    @property
    def spatial_dims(self):
        return self._model.dimensions

    def set_dataset(self, dataset: DataSet):
        self.dataset = dataset

    def create_transforms(
        self,
        keys: List[str],
        train: bool = False,
        spacing: Sequence[float] = None,
    ):
        # loading and normalization
        xforms = [
            LoadImaged(keys=keys, reader="itkreader"),
            AddChanneld(keys="label"),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys=keys, axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(keys=keys, source_key="image"),
        ]

        # resample
        if spacing:
            xforms.append(Spacingd(keys=keys, pixdim=spacing))

        # add augmentation
        if train:
            xforms.extend(
                [
                    RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
                    RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
                ]
            )
            if self.spatial_dims > 2:
                xforms.append(
                    RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2)
                )

            if self.spatial_size is None:
                spatial_size = tuple(96 for _ in range(self.spatial_dims))
            xforms.append(
                RandCropByLabelClassesd(
                    keys=["image", "label"],
                    label_key="label",
                    image_key="image",
                    spatial_size=spatial_size,
                    num_classes=self.num_classes,
                    num_samples=4,
                    image_threshold=-np.inf,
                )
            )
        return Compose(xforms + [EnsureTyped(keys=keys)])

    def forward(self, x):
        return self._model(x)

    def prepare_data(self) -> None:
        if not self.dataset:
            raise RuntimeError("The dataset is not set")

        # set deterministic training for reproducibility
        set_determinism(seed=0)

        # define the data transforms
        train_transforms = self.create_transforms(
            keys=["image", "label"],
            train=True,
        )
        val_transforms = self.create_transforms(
            keys=["image", "label"],
            train=False,
        )

        # we use cached datasets - these are 10x faster than regular datasets
        self.train_ds = CacheDataset(
            data=self.dataset.training_files(),
            transform=train_transforms,
            cache_rate=1.0,
            num_workers=0,
        )
        self.val_ds = CacheDataset(
            data=self.dataset.validation_files(),
            transform=val_transforms,
            cache_rate=1.0,
            num_workers=0,
        )

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
            f"current mean dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.log("val_dice", mean_val_dice)
        self.log("val_loss", mean_val_loss)
        return {"log": tensorboard_logs}


def train(
    image_dir: Path,
    labels_dir: Path,
    log_dir: Path,
    num_classes: int,
    num_channels: int,
    model_file_name: Path,
    output_dir: Path,
    spatial_dims: int = 3,
    spatial_size: Sequence[int] = None,
    max_epochs: int = 600,
    save_nifti: bool = True,
    gpu_ids: list = [],
):
    """Run the training"""
    # initialise the LightningModule
    net = Net(
        num_classes=num_classes,
        num_channels=num_channels,
        spatial_dims=spatial_dims,
        spatial_size=spatial_size,
        dataset=DataSet(image_dir=image_dir, labels_dir=labels_dir),
    )

    # set up loggers and checkpoints
    tb_logger = pytorch_lightning.loggers.TensorBoardLogger(save_dir=f"{log_dir}")
    checkpoint_callback = ModelCheckpoint(
        filename=os.path.join(output_dir, "{epoch}-{val_loss:.2f}-{val_dice:.4f}"),
        monitor="val_dice",
        mode="max",
        dirpath=output_dir if output_dir else log_dir,
        save_top_k=3,
    )

    # initialise Lightning's trainer.
    # other options:
    #  - precision=16 (todo: evaluate speed-up)
    #  - max_time={"days": 1, "hours": 5}
    trainer = pytorch_lightning.Trainer(
        gpus=gpu_ids,
        max_epochs=max_epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=1,
    )

    # train
    trainer.fit(net)

    print(
        f"train completed, best_metric: {net.best_val_dice:.4f} "
        f"at epoch {net.best_val_epoch}"
    )

    settings = {"num_classes": num_classes, "num_channels": num_channels}
    with model_file_name.with_suffix(".json").open("w") as json_file:
        json.dump(settings, json_file)

    trainer.save_checkpoint(model_file_name)

    """## View training in tensorboard"""

    # Commented out IPython magic to ensure Python compatibility.
    # %load_ext tensorboard
    # %tensorboard --logdir=log_dir

    """## Check best model output with the input image and label"""
    if save_nifti:
        os.makedirs(output_dir, exist_ok=True)
        saver = NiftiSaver(output_dir=output_dir, separate_folder=False, resample=False)

    net.eval()
    device = make_device(gpu_ids)
    net.to(device)
    with torch.no_grad():
        for i, val_data in enumerate(net.val_dataloader()):
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(
                val_data["image"].to(device), roi_size, sw_batch_size, net
            )

            cmap = make_random_cmap(num_classes + 1)

            plt.figure("check", (18, 6))
            for row, slice in enumerate([80, 180]):
                plt.subplot(2, 3, 1 + row * 3)
                plt.title(f"image {i}")
                plt.imshow(val_data["image"][0, 0, :, :, slice], cmap="gray")
                plt.subplot(2, 3, 2 + row * 3)
                plt.title(f"label {i}")
                plt.imshow(val_data["label"][0, 0, :, :, slice], cmap=cmap)
                plt.subplot(2, 3, 3 + row * 3)
                plt.title(f"output {i}")
                plt.imshow(
                    torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice],
                    cmap=cmap,
                )
            plot_file_path = output_dir / f"drcmr{num_classes:02d}_case{i}.png"
            plt.savefig(f"{plot_file_path}")

            if saver:
                pred_labels = val_outputs.argmax(dim=1, keepdim=True)
                saver.save_batch(pred_labels, val_data["image_meta_dict"])


def predict(
    model_file: Path,
    output_dir: Path,
    test_images: List[Path],
    test_labels: Optional[List[Path]] = None,
    tissue_dict: Dict[str, int] = None,
    save_nifti: bool = True,
    gpu_ids: list = [],
):
    # load trained model
    if model_file.exists():
        with model_file.with_suffix(".json").open() as json_file:
            settings = json.load(json_file)
        net = Net.load_from_checkpoint(f"{model_file}", **settings)
    else:
        net = Net.load_from_checkpoint(f"{model_file}")
    num_classes = net.num_classes

    net.eval()
    device = make_device(gpu_ids)
    net.to(device)

    # define data / data loader
    if test_labels:
        test_files = [
            {"image": i, "label": l} for i, l in zip(test_images, test_labels)
        ]
    else:
        test_files = [{"image": i} for i in test_images]

    pre_transforms = net.create_transforms(
        keys=["image", "label"] if test_labels else ["image"],
        train=False,
        spacing=(0.85, 0.85, 0.85),
    )

    test_ds = CacheDataset(
        data=test_files,
        transform=pre_transforms,
        cache_rate=1.0,
        num_workers=0,
    )

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, num_workers=0)

    # for saving output
    save_transforms = []
    if save_nifti:
        os.makedirs(output_dir, exist_ok=True)
        save_transforms.append(
            SaveImaged(
                keys="pred",
                meta_keys="pred_meta_dict",
                output_dir=output_dir,
                output_postfix="seg",
                resample=False,
                separate_folder=False,
            )
        )

    # invert transforms (e.g. cropping)
    post_transforms = Compose(
        [
            EnsureTyped(keys="pred"),
            # Activationsd(keys="pred", sigmoid=True),
            Invertd(
                keys="pred",  # invert the `pred` data field, also support multiple fields
                transform=pre_transforms,
                orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
                # then invert `pred` based on this information. we can use same info
                # for multiple fields, also support different orig_keys for different fields
                meta_keys="pred_meta_dict",  # key field to save inverted meta data, every item maps to `keys`
                orig_meta_keys="image_meta_dict",  # get the meta data from `img_meta_dict` field when inverting,
                # for example, may need the `affine` to invert `Spacingd` transform,
                # multiple fields can use the same meta data to invert
                meta_key_postfix="meta_dict",  # if `meta_keys=None`, use "{keys}_{meta_key_postfix}" as the meta key,
                # if `orig_meta_keys=None`, use "{orig_keys}_{meta_key_postfix}",
                # otherwise, no need this arg during inverting
                nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
                # to ensure a smooth output, then execute `AsDiscreted` transform
                to_tensor=True,  # convert to PyTorch Tensor after inverting
            ),
            AsDiscreted(keys="pred", argmax=True),
        ]
        + save_transforms
    )

    # evaluate accuracy
    dice_metric = DiceMetric(
        include_background=False, reduction="mean", get_not_nans=False
    )
    conf_matrix = ConfusionMatrixMetric(
        metric_name=["sensitivity", "specificity", "precision", "accuracy"]
    )

    to_one_hot = lambda x: one_hot(x, num_classes=num_classes, dim=0)

    tissue_names = [""] * num_classes
    if tissue_dict:
        for name in tissue_dict.keys():
            idx = tissue_dict[name]
            tissue_names[idx] = name

    with torch.no_grad():
        for test_data in test_loader:
            val_image = test_data["image"].to(device)

            val_pred = sliding_window_inference(
                inputs=val_image, roi_size=(96, 96, 96), sw_batch_size=4, predictor=net
            )

            test_data["pred"] = val_pred
            for i in decollate_batch(test_data):
                post_transforms(i)

            if test_labels:
                val_pred = val_pred.argmax(dim=1, keepdim=True)
                val_labels = test_data["label"].to(device).long()

                d = dice_metric(y_pred=to_one_hot(val_pred), y=to_one_hot(val_labels))
                print("Class Dice = ", d)
                print("Mean Dice = ", dice_metric.aggregate().item())
                dice_metric.reset()

                conf_matrix(y_pred=to_one_hot(val_pred), y=to_one_hot(val_labels))
                print("Conf. Matrix Metrics = ", conf_matrix.aggregate())

                filename_or_obj = test_data["image_meta_dict"]["filename_or_obj"]
                if filename_or_obj and isinstance(filename_or_obj, list):
                    filename_or_obj = filename_or_obj[0]
                if filename_or_obj:

                    base = Path(filename_or_obj).with_suffix("").name
                    c = confusion_matrix(
                        num_classes=num_classes,
                        y_pred=val_pred.view(-1).cpu().numpy(),
                        y=val_labels.view(-1).cpu().numpy(),
                    )
                    plot_confusion_matrix(
                        c,
                        tissue_names,
                        file_name=output_dir / (base + "_confusion.png"),
                    )

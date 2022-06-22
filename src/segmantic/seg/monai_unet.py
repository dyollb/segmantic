import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
from monai.bundle import ConfigParser
from monai.config import print_config
from monai.data import CacheDataset, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import ConfusionMatrixMetric, DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.networks.utils import one_hot
from monai.transforms import (
    AddChanneld,
    AsDiscrete,
    AsDiscreted,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureType,
    EnsureTyped,
    Invertd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByLabelClassesd,
    SaveImaged,
)
from monai.transforms.spatial.dictionary import Spacingd
from monai.transforms.transform import Transform
from monai.utils import set_determinism
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from ..prepro.labels import load_tissue_list
from .dataset import PairedDataSet
from .evaluation import confusion_matrix
from .utils import make_device
from .visualization import plot_confusion_matrix


class Net(pl.LightningModule):
    dataset: Optional[PairedDataSet]
    cache_rate: float = 1.0
    config_preprocessing: dict = {}
    config_augmentation: dict = {}
    intensity_augmentation: bool = False
    spatial_augmentation: bool = False
    best_val_dice = 0
    best_val_epoch = 0

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 1,
        spatial_dims: int = 3,
        spatial_size: Sequence[int] = None,
    ):
        super().__init__()

        self.save_hyperparameters()

        self._model = UNet(
            spatial_dims=spatial_dims,
            in_channels=num_channels,
            out_channels=num_classes,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )
        self.spatial_size = spatial_size if spatial_size else [96] * 3
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

    @property
    def num_classes(self):
        return self._model.out_channels

    @property
    def spatial_dims(self):
        return self._model.dimensions

    def default_preprocessing(
        self,
        keys: List[str],
        spacing: Sequence[float] = None,
    ) -> Transform:

        xforms = [
            LoadImaged(keys=keys, reader="itkreader"),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys=keys, axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(keys=keys, source_key="image"),
            EnsureTyped(keys=keys, dtype=np.float32, device=self.device),
        ]

        if "label" in keys:
            xforms.insert(1, AddChanneld(keys="label"))

        if spacing:
            xforms.append(Spacingd(keys=keys, pixdim=spacing))

        return Compose(xforms)

    def default_augmentation(self, keys: List[str]):
        return RandCropByLabelClassesd(
            keys=keys,
            label_key="label",
            image_key="image",
            spatial_size=self.spatial_size,
            num_classes=self.num_classes,
            num_samples=4,
        )

    def forward(self, x):
        return self._model(x)

    def prepare_data(self) -> None:
        if not self.dataset:
            raise RuntimeError("The dataset is not set")

        # set deterministic training for reproducibility
        set_determinism(seed=0)

        # define the data transforms
        preprocessing = None
        if self.config_preprocessing:
            parser = ConfigParser(
                {
                    "image_key": "image",
                    "label_key": "label",
                    "preprocessing": self.config_preprocessing,
                }
            )
            parser.parse(True)
            preprocessing = parser.get_parsed_content("preprocessing")
        if not preprocessing:
            print("Using default preprocessing")
            preprocessing = self.default_preprocessing(keys=["image", "label"])

        augmentation = None
        if self.config_augmentation:
            parser = ConfigParser(
                {
                    "image_key": "image",
                    "label_key": "label",
                    "augmentation": self.config_augmentation,
                }
            )
            parser.parse(True)
            augmentation = parser.get_parsed_content("augmentation")
        if not augmentation:
            print("Using default augmentation")
            augmentation = self.default_augmentation(keys=["image", "label"])

        # we use cached datasets - these are 10x faster than regular datasets
        self.train_ds = CacheDataset(
            data=self.dataset.training_files(),
            transform=Compose((preprocessing, augmentation)).flatten(),
            cache_rate=self.cache_rate,
            num_workers=0,
        )
        self.val_ds = CacheDataset(
            data=self.dataset.validation_files(),
            transform=Compose(preprocessing).flatten(),
            cache_rate=self.cache_rate,
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
            f"mean val dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.log("val_dice", mean_val_dice)
        self.log("val_loss", mean_val_loss)
        return {"log": tensorboard_logs}


def train(
    *,
    dataset: Union[Path, List[Path]] = [],
    image_dir: Path = None,
    labels_dir: Path = None,
    tissue_list: Path,
    output_dir: Path,
    checkpoint_file: Path = None,
    num_channels: int = 1,
    spatial_dims: int = 3,
    spatial_size: Sequence[int] = None,
    preprocessing: dict = None,
    augmentation: dict = None,
    augment_intensity: bool = False,
    augment_spatial: bool = False,
    max_epochs: int = 600,
    mixed_precision: bool = True,
    cache_rate: float = 1.0,
    gpu_ids: List[int] = [0],
) -> pl.LightningModule:

    print_config()

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    log_dir = output_dir / "logs"

    tissue_dict = load_tissue_list(tissue_list)
    num_classes = max(tissue_dict.values()) + 1
    if len(tissue_dict) != num_classes:
        raise ValueError("Expecting contiguous labels in range [0,N-1]")

    # initialise the LightningModule
    if checkpoint_file and Path(checkpoint_file).exists():
        net = Net.load_from_checkpoint(f"{checkpoint_file}")
    else:
        net = Net(
            spatial_dims=spatial_dims,
            num_channels=num_channels,
            num_classes=num_classes,
            spatial_size=spatial_size,
        )
    if image_dir and labels_dir:
        net.dataset = PairedDataSet(image_dir=image_dir, labels_dir=labels_dir)
    elif dataset:
        net.dataset = PairedDataSet.load_from_json(dataset)
    else:
        raise ValueError(
            "Either provide a dataset file, or an image_dir, labels_dir pair."
        )
    net.config_preprocessing = preprocessing
    net.config_augmentation = augmentation
    net.intensity_augmentation = augment_intensity
    net.spatial_augmentation = augment_spatial
    net.cache_rate = cache_rate

    # store dataset used for training
    (output_dir / "Dataset.json").write_text(net.dataset.dump_dataset())

    # set up loggers and checkpoints
    tb_logger = TensorBoardLogger(save_dir=f"{log_dir}")
    checkpoint_callback = ModelCheckpoint(
        filename=os.path.join(output_dir, "{epoch}-{val_loss:.2f}-{val_dice:.4f}"),
        monitor="val_dice",
        mode="max",
        dirpath=output_dir if output_dir else log_dir,
        save_top_k=3,
    )

    # initialise Lightning's trainer.
    trainer = pl.Trainer(
        gpus=gpu_ids,
        precision=16 if mixed_precision else 32,
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

    return net


def predict(
    model_file: Path,
    output_dir: Path,
    test_images: List[Path],
    test_labels: Optional[List[Path]] = None,
    tissue_dict: Dict[str, int] = None,
    save_nifti: bool = True,
    gpu_ids: list = [],
) -> None:
    # load trained model
    model_settings_json = model_file.with_suffix(".json")
    if model_settings_json.exists():
        print(f"WARNING: Loading legacy model settings from {model_settings_json}")
        with model_settings_json.open() as json_file:
            settings = json.load(json_file)
        net = Net.load_from_checkpoint(f"{model_file}", **settings)
    else:
        net = Net.load_from_checkpoint(f"{model_file}")
    num_classes = net.num_classes

    net.freeze()
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
            Invertd(
                keys="pred",
                transform=pre_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
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

    def to_one_hot(x):
        return one_hot(x, num_classes=num_classes, dim=0)

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
            assert isinstance(val_pred, torch.Tensor)

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

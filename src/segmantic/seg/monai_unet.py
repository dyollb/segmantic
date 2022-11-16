import json
import os
import subprocess as sp
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pytorch_lightning as pl
import torch
from adabelief_pytorch import AdaBelief
from monai.bundle import ConfigParser
from monai.config import print_config
from monai.data import (
    CacheDataset,
    DataLoader,
    Dataset,
    decollate_batch,
    list_data_collate,
)
from monai.inferers import SlidingWindowInferer, sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import ConfusionMatrixMetric, CumulativeAverage, DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.networks.utils import one_hot
from monai.transforms import (
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
    RandAdjustContrastd,
    RandBiasFieldd,
    RandCropByLabelClassesd,
    RandFlipd,
    RandGibbsNoised,
    RandHistogramShiftd,
    RandKSpaceSpikeNoised,
    RandRotated,
    RandZoomd,
    SaveImaged,
)
from monai.transforms.spatial.dictionary import Spacingd
from monai.transforms.transform import MapTransform, Transform
from monai.utils import set_determinism
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from ..image.labels import load_tissue_list
from ..utils import config
from .dataset import PairedDataSet
from .evaluation import confusion_matrix
from .utils import make_device
from .visualization import plot_confusion_matrix


class Net(pl.LightningModule):
    dataset: PairedDataSet
    cache_rate: float = 1.0
    config_preprocessing: dict = {}
    config_augmentation: dict = {}
    augment_intensity: bool = False
    augment_spatial: bool = False
    best_val_dice = 0
    best_val_epoch = 0
    num_samples: int = 4
    optimizer: dict = {
        "optimizer": "Adam",
        "lr": 1e-4,
        "momentum": 0.9,
        "epsilon": 1e-8,
        "amsgrad": False,
        "weight_decouple": False,
    }
    lr_scheduling: dict = {
        "scheduler": "Constant",
        "factor": 0.5,
        "patience": 10,
        "T_0": 50,
        "T_multi": 1,
    }

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 1,
        spatial_dims: int = 3,
        spatial_size: Sequence[int] = None,
        channels: Tuple[int, ...] = (16, 32, 64, 128, 256),
        strides: Tuple[int, ...] = (2, 2, 2, 2),
        dropout: float = 0.0,
    ):
        super().__init__()

        self.save_hyperparameters()

        self._model = UNet(
            spatial_dims=spatial_dims,
            in_channels=num_channels,
            out_channels=num_classes,
            channels=channels,
            strides=strides,
            dropout=dropout,
            num_res_units=2,
            norm=Norm.BATCH,
        )
        # ToDo: make true/false dependant on if optimizer is default or not
        self.automatic_optimization = False
        self.spatial_size = spatial_size if spatial_size else [96] * 3
        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.post_pred = Compose(
            [
                EnsureType(),
                AsDiscrete(argmax=True, to_onehot=num_classes),
            ]
        )
        self.post_label = Compose([EnsureType(), AsDiscrete(to_onehot=num_classes)])
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
        spacing: Sequence[float] = [],
    ) -> Transform:

        xforms = [
            LoadImaged(keys=keys, reader="ITKReader"),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True),
            CropForegroundd(keys=keys, source_key="label"),
            EnsureTyped(keys=keys, dtype=np.float32, device=self.device),  # type: ignore
        ]

        if spacing:
            xforms.append(Spacingd(keys=keys, pixdim=spacing))

        return Compose(xforms)

    def default_augmentation(self, keys: List[str]):
        xforms: List[MapTransform] = []

        if self.augment_spatial:
            mode = ["nearest" if k == "label" else "bilinear" for k in keys]
            xforms.append(RandRotated(keys=keys, prob=0.2, range_z=0.4, mode=mode))
            if self.spatial_dims > 2:
                xforms.append(RandRotated(keys=keys, prob=0.2, range_x=0.4, mode=mode))
                xforms.append(RandRotated(keys=keys, prob=0.2, range_y=0.4, mode=mode))

            mode = ["nearest" if k == "label" else "area" for k in keys]
            xforms.append(
                RandZoomd(keys=keys, prob=0.2, min_zoom=0.8, max_zoom=1.3, mode=mode)
            )
        # Set probability of background label being chosen as corp center to 0 and rest to 1.
        xforms += [
            RandCropByLabelClassesd(
                keys=keys,
                label_key="label",
                spatial_size=self.spatial_size,
                num_classes=self.num_classes,
                num_samples=self.num_samples,
                ratios=[0 if x == 0 else 1 for x in range(self.num_classes)],
            )
        ]

        if self.augment_intensity:
            xforms += [
                RandAdjustContrastd(keys="image", prob=0.2, gamma=(0.5, 4.5)),
                RandHistogramShiftd(keys="image", prob=0.2, num_control_points=10),
                RandBiasFieldd(keys="image", prob=0.2),
                RandGibbsNoised(keys="image", prob=0.2, alpha=(0.0, 1.0)),
                RandKSpaceSpikeNoised(keys="image", prob=0.2),
            ]

        xforms += [
            RandFlipd(keys=keys, prob=0.2, spatial_axis=a)
            for a in range(self.spatial_dims)
        ]

        return Compose(xforms)

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
            print(
                f"Using default augmentation (intensity={self.augment_intensity}, spatial={self.augment_spatial})"
            )
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
        train_loader = DataLoader(
            self.train_ds,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            collate_fn=list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_ds, batch_size=1, num_workers=0)
        return val_loader

    def configure_optimizers(self):
        if self.optimizer["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(
                self._model.parameters(),
                lr=self.optimizer["lr"],
                momentum=self.optimizer["momentum"],
            )
        elif self.optimizer["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(
                self._model.parameters(),
                lr=self.optimizer["lr"],
                amsgrad=self.optimizer["amsgrad"],
            )
        elif self.optimizer["optimizer"] == "AdaBelief":
            optimizer = AdaBelief(
                self._model.parameters(),
                lr=self.optimizer["lr"],
                eps=self.optimizer["epsilon"],  # try 1e-8 and 1e-16
                betas=(0.9, 0.999),
                weight_decouple=self.optimizer["weight_decouple"],  # Try True/False
                fixed_decay=False,
                rectify=False,
            )

        if self.lr_scheduling["scheduler"] == "Constant":
            lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer=optimizer, factor=1, total_iters=0
            )

        elif self.lr_scheduling["scheduler"] == "ReduceOnPlateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                factor=self.lr_scheduling["factor"],
                patience=self.lr_scheduling["patience"],
                verbose=True,
            )

        elif self.lr_scheduling["scheduler"] == "Cosine":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=self.lr_scheduling["T_0"],
                T_mult=self.lr_scheduling["T_multi"],
                eta_min=0,
            )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)
        optimizer = self.optimizers()
        optimizer.zero_grad()
        loss = self.loss_function(output, labels)
        self.manual_backward(loss)
        optimizer.step()
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

        scheduler = self.lr_schedulers()
        if self.lr_scheduling["scheduler"] == "ReduceOnPlateau":
            scheduler.step(mean_val_loss)
        else:
            scheduler.step()

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
            f"\ncurrent mean loss: {mean_val_loss:.4f}"
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
    output_dir: Path,
    checkpoint_file: Path = None,
    num_classes: int = 0,
    num_channels: int = 1,
    spatial_dims: int = 3,
    spatial_size: Sequence[int] = [],
    preprocessing: dict = {},
    augmentation: dict = {},
    augment_intensity: bool = False,
    augment_spatial: bool = False,
    channels: Tuple[int, ...] = (16, 32, 64, 128, 256),
    strides: Tuple[int, ...] = (2, 2, 2, 2),
    dropout: float = 0.0,
    num_samples: int = 4,
    optimizer=None,
    lr_scheduling=None,
    max_epochs: int = 600,
    early_stop_patience: int = 50,
    mixed_precision: bool = True,
    cache_rate: float = 1.0,
    gpu_ids: List[int] = [0],
    tissue_list: Path = None,
) -> pl.LightningModule:

    if optimizer is None:
        optimizer = {
            "optimizer": "Adam",
            "lr": 1e-4,
            "momentum": 0.9,
            "epsilon": 1e-8,
            "amsgrad": False,
            "weight_decouple": False,
        }
    if lr_scheduling is None:
        lr_scheduling = {
            "scheduler": "Constant",
            "factor": 0.5,
            "patience": 10,
            "T_0": 50,
            "T_multi": 1,
        }

    # initialise the LightningModule
    if checkpoint_file and Path(checkpoint_file).exists():
        net = cast(Net, Net.load_from_checkpoint(f"{checkpoint_file}"))
    else:
        if num_classes > 0 and tissue_list:
            raise ValueError(
                "'num_classes' and 'tissue_list' are redundant. Prefer 'num_classes'."
            )
        if tissue_list:
            tissue_dict = load_tissue_list(tissue_list)
            num_classes = max(tissue_dict.values()) + 1
            if len(tissue_dict) != num_classes:
                raise ValueError("Expecting contiguous labels in range [0,N-1]")
        if num_classes <= 1:
            raise ValueError("'num_classes' is expected to be > 1")

        net = Net(
            spatial_dims=spatial_dims,
            num_channels=num_channels,
            num_classes=num_classes,
            spatial_size=spatial_size,
            channels=channels,
            strides=strides,
            dropout=dropout,
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
    net.augment_intensity = augment_intensity
    net.augment_spatial = augment_spatial
    net.num_samples = num_samples
    net.optimizer = optimizer
    net.lr_scheduling = lr_scheduling
    net.cache_rate = cache_rate

    # store dataset used for training
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    log_dir = output_dir / "logs"
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

    # defining early stopping. When val loss improves less than 0 over 30 epochs, the training will be stopped.
    early_stop_callback = EarlyStopping(
        monitor="val_dice",
        min_delta=0.00,
        patience=early_stop_patience,
        mode="max",
        check_finite=True,
    )

    lr_monitor = LearningRateMonitor(log_momentum=True, logging_interval="epoch")

    print_config()

    # initialise Lightning's trainer.
    # ToDo: Add self.batch_size to init and actually make the auto_scale_batch_size work.
    trainer = pl.Trainer(
        gpus=gpu_ids,
        # auto_scale_batch_size=True,
        precision=16 if mixed_precision else 32,
        max_epochs=max_epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
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
    test_images: List[Path],
    test_labels: Optional[List[Path]] = None,
    output_dir: Path = None,
    tissue_dict: Dict[str, int] = None,
    channels: Tuple[int, ...] = (16, 32, 64, 128, 256),
    strides: Tuple[int, ...] = (2, 2, 2, 2),
    dropout: float = 0.0,
    spacing: Sequence[float] = [],
    gpu_ids: List[int] = [],
) -> None:
    # load trained model
    model_settings_json = model_file.with_suffix(".json")
    if model_settings_json.exists():
        print(f"WARNING: Loading legacy model settings from {model_settings_json}")
        with model_settings_json.open() as json_file:
            settings = json.load(json_file)
        net = cast(Net, Net.load_from_checkpoint(f"{model_file}", **settings))
    else:
        net = cast(
            Net,
            Net.load_from_checkpoint(
                f"{model_file}", channels=channels, strides=strides, dropout=dropout
            ),
        )
    num_classes = net.num_classes

    net.freeze()
    net.eval()
    device = make_device(gpu_ids)
    net.to(device)

    # pre-processing transforms
    if test_labels:
        assert len(test_images) == len(test_labels)
        test_files = [
            {"image": i, "label": l} for i, l in zip(test_images, test_labels)
        ]
    else:
        test_files = [{"image": i} for i in test_images]

    pre_transforms = net.default_preprocessing(
        keys=["image", "label"] if test_labels else ["image"],
        spacing=spacing,
    )

    # save predicted labels
    save_transforms: List[MapTransform] = []
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        save_transforms = [
            SaveImaged(
                keys="pred",
                output_dir=output_dir,
                output_postfix="seg",
                resample=False,
                separate_folder=False,
                print_log=False,
                writer="ITKWriter",
            )
        ]

    # invert transforms (e.g. cropping)
    post_transforms = Compose(
        [
            EnsureTyped(keys="pred"),
            Invertd(
                keys="pred",
                transform=pre_transforms,  # type: ignore [arg-type]
                orig_keys="image",
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(keys="pred", argmax=True),
        ]
        + save_transforms
    )

    # data loader
    test_loader = DataLoader(
        Dataset(
            data=test_files,
            transform=pre_transforms,
        ),
        batch_size=1,
        num_workers=0,
    )

    inferer = SlidingWindowInferer(
        roi_size=net.spatial_size, sw_batch_size=4, device=device
    )

    # evaluate metrics
    dice_metric = DiceMetric(
        include_background=False, reduction="mean", get_not_nans=False
    )
    confusion_metrics = ["sensitivity", "specificity", "precision", "accuracy"]
    conf_matrix = ConfusionMatrixMetric(metric_name=confusion_metrics)
    mean_class_dice = CumulativeAverage()

    def to_one_hot(x):
        return one_hot(x, num_classes=num_classes, dim=0)

    tissue_names = [f"{id}" for id in range(num_classes)]
    if tissue_dict:
        for name in tissue_dict.keys():
            idx = tissue_dict[name]
            tissue_names[idx] = name

    def print_table(header, vals, indent="\t"):
        print(indent + "\t".join(header).expandtabs(30))
        print(indent + "\t".join(f"{x}" for x in vals).expandtabs(30))

    all_mean_dice = []
    with torch.no_grad():
        for test_data in test_loader:

            val_pred = inferer(test_data["image"].to(device), net)
            assert isinstance(val_pred, torch.Tensor)

            test_data["pred"] = val_pred
            for i in decollate_batch(test_data):
                post_transforms(i)

            if test_labels:
                val_pred = val_pred.argmax(dim=1, keepdim=True)
                val_labels = test_data["label"].to(device).long()

                dice = dice_metric(
                    y_pred=to_one_hot(val_pred), y=to_one_hot(val_labels)
                )
                mean_class_dice.append(dice)
                conf_matrix(y_pred=to_one_hot(val_pred), y=to_one_hot(val_labels))

                dice_np = dice.cpu().numpy()
                print("Mean Dice: ", np.mean(dice_np))
                print("Class Dice:")
                print_table(tissue_names, np.squeeze(dice_np))

                all_mean_dice.append(dice_metric.aggregate().item())

                filename_or_obj = test_data["image_meta_dict"]["filename_or_obj"]
                if filename_or_obj and isinstance(filename_or_obj, list):
                    filename_or_obj = filename_or_obj[0]

                if output_dir and filename_or_obj:
                    base = Path(filename_or_obj).stem.replace(".nii", "")
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
        if output_dir is None:
            print("No output path specified, dice scores won't be saved.")
        else:
            np.savetxt(
                output_dir.joinpath(
                    f"mean_dice_{model_file.stem}_generalized_score.txt"
                ),
                all_mean_dice,
                delimiter=",",
            )

        if test_labels:
            print("*" * 80)
            print("Total Mean Dice: ", dice_metric.aggregate().item())
            print("Total Class Dice:")
            print_table(
                tissue_names, np.squeeze(mean_class_dice.aggregate().cpu().numpy())
            )
            print("Total Conf. Matrix Metrics:")
            print_table(
                confusion_metrics,
                (np.squeeze(x.cpu().numpy()) for x in conf_matrix.aggregate()),
            )


def cross_validate(
    image_dir: Path,
    labels_dir: Path,
    tissue_list: Path,
    output_dir: Path,
    config_files_dir: Path,
    test_image_dir: Path = None,
    test_labels_dir: Path = None,
    num_splits: int = 7,
    gpu_ids: List[int] = [0],
):
    print_config()
    print("Cross-validating")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    tissue_dict = load_tissue_list(tissue_list)
    print(tissue_dict)

    data_dicts = PairedDataSet.create_data_dict(
        image_dir=image_dir, labels_dir=labels_dir
    )

    test_data_dicts = []

    if test_image_dir and test_labels_dir:
        test_data_dicts = PairedDataSet.create_data_dict(
            image_dir=test_image_dir, labels_dir=test_labels_dir
        )

    all_datafold_paths: List[Path] = PairedDataSet.kfold_crossval(
        num_splits=num_splits,
        data_dicts=data_dicts,
        output_dir=output_dir,
        test_data_dicts=test_data_dicts,
    )

    for config_file in Path(config_files_dir).iterdir():

        assert config_file.suffix in [".json", ".yaml"]
        is_json = config_file and config_file.suffix.lower() == ".json"
        dumps = partial(config.dumps, is_json)
        loads = partial(config.loads, is_json)

        output_dir_scenario = output_dir / config_file.name
        output_dir_scenario.mkdir(exist_ok=True)

        for count, dataset_path in enumerate(all_datafold_paths):

            current_output = output_dir_scenario / str(count)
            print(current_output)

            current_output.mkdir(exist_ok=True)

            data: dict = loads(config_file.read_text())

            data["dataset"] = str(dataset_path)
            data["output_dir"] = str(current_output)
            current_layers = data["channels"]
            current_strides = data["strides"]

            config_file.write_text(dumps(data))

            print("start training")

            result = sp.run(
                [
                    sys.executable,
                    str(Path.cwd().joinpath("run_monai_unet.py")),
                    "train-config",
                    "-c",
                    config_file,
                ]
            )
            print(result)
            print("training finished")
            if test_image_dir is not None and test_labels_dir is not None:
                assert test_image_dir.is_dir() and test_labels_dir.is_dir()

                test_images = sorted(list(test_image_dir.glob(".nii.gz")))
                test_labels = sorted(list(test_labels_dir.glob(".nii.gz")))

                assert len(test_images) == len(test_labels)

                for file in current_output.iterdir():
                    if file.match("*.ckpt"):
                        print("start prediction")
                        predict(
                            model_file=file,
                            output_dir=current_output,
                            test_images=test_images,
                            test_labels=test_labels,
                            tissue_dict=tissue_dict,
                            channels=current_layers,
                            strides=current_strides,
                            dropout=0.0,
                            spacing=[1, 1, 1],
                            gpu_ids=gpu_ids,
                        )

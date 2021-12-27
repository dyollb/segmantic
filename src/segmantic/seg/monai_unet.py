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
    RandAdjustContrastd,
    RandHistogramShiftd,
    RandGibbsNoised,
    RandKSpaceSpikeNoised,
    RandCropByLabelClassesd,
    RandFlipd,
    RandRotated,
    RandZoomd,
    NormalizeIntensityd,
    EnsureTyped,
    EnsureType,
    SaveImaged,
    Invertd,
    SqueezeDim,
    Lambdad,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate, decollate_batch, NiftiSaver
from monai.networks.utils import one_hot
from monai.config import print_config
from adabelief_pytorch import AdaBelief
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import pytorch_lightning
import pytorch_lightning.loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import matplotlib.pyplot as plt
import os
import json
from typing import List, Optional, Dict, Sequence
from pathlib import Path
import shutil

from sklearn.model_selection import KFold, StratifiedKFold

from ..prepro.labels import load_tissue_list
from .evaluation import confusion_matrix
from .utils import make_device
from .dataset import DataSet
from .visualization import make_tissue_cmap, plot_confusion_matrix
from .genetic_algorithm import (
    initialize_population,
    binary_tournament_selection,
    crossover,
    environmental_selection,
    mutation,
)
from .decode_genotypes import decode_optimizer_gene, decode_lr_scheduler_gene


class Net(pytorch_lightning.LightningModule):
    def __init__(
            self,
            num_classes: int,
            num_channels: int = 1,
            spatial_dims: int = 3,
            spatial_size: Sequence[int] = None,
            dataset: Optional[DataSet] = None,
            layers: tuple = (16, 32, 64, 128, 256),
            strides: tuple = (2, 2, 2, 2),
            dropout: float = 0.0,
    ):
        super().__init__()
        self._model = UNet(
            spatial_dims=spatial_dims,
            in_channels=num_channels,
            out_channels=num_classes,
            channels=layers,
            strides=strides,
            dropout=dropout,
            num_res_units=2,
            norm=Norm.BATCH,
        )
        self.automatic_optimization = False
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

    dataset: Optional[DataSet]
    cache_rate: float = 1.0
    intensity_augmentation: bool = False
    spatial_augmentation: bool = False
    num_samples: int = 4
    optimizer: dict = {'optimizer': 'Adam',
                       'lr': 1e-4,
                       'momentum': 0.9,
                       'epsilon': 1e-8,
                       'amsgrad': False,
                       'weight_decouple': False
                       }
    lr_scheduling: dict = {'scheduler': 'Constant',
                           'factor': 0.5,
                           'patience': 10,
                           'T_0': 50,
                           'T_multi': 1}

    @property
    def num_classes(self):
        return self._model.out_channels

    @property
    def spatial_dims(self):
        return self._model.dimensions

    def squeeze_dim(self, img):
        shape = img.shape
        print(shape)
        if len(shape) == 5:
            trans = SqueezeDim(dim=4)
            return trans(img)
        else:
            return img

    def create_transforms(
            self,
            keys: List[str],
            train: bool = False,
            spacing: Sequence[float] = None,
    ):
        # loading and normalization
        xforms = [
            LoadImaged(keys=keys, reader="itkreader"),
            EnsureChannelFirstd(keys="image"),
            # SqueezeDimd(keys="image", dim=4),
            # Lambdad(("image",), self.squeeze_dim),
            Orientationd(keys=keys, axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(keys=keys, source_key="image"),
        ]

        if "label" in keys:
            xforms.insert(1, AddChanneld(keys="label"))

        # resample
        if spacing:
            xforms.append(Spacingd(keys=keys, pixdim=spacing))

        # add augmentation
        if train:
            xforms.extend(
                [
                    RandFlipd(keys=keys, prob=0.2, spatial_axis=a)
                    for a in range(self.spatial_dims)
                ]
            )

            if self.intensity_augmentation:
                xforms.extend(
                    [
                        RandAdjustContrastd(keys="image", prob=0.2, gamma=(0.5, 4.5)),
                        RandHistogramShiftd(
                            keys="image", prob=0.2, num_control_points=10
                        ),
                        RandGibbsNoised(keys="image", prob=0.2, alpha=(0.0, 1.0)),
                        RandKSpaceSpikeNoised(keys="image", global_prob=0.1, prob=0.2),
                    ]
                )

            if self.spatial_augmentation:
                mode = ["nearest" if k == "label" else "bilinear" for k in keys]
                xforms.append(RandRotated(keys=keys, prob=0.2, range_z=0.4, mode=mode))
                if self.spatial_dims > 2:
                    xforms.append(
                        RandRotated(keys=keys, prob=0.2, range_x=0.4, mode=mode)
                    )
                    xforms.append(
                        RandRotated(keys=keys, prob=0.2, range_y=0.4, mode=mode)
                    )

                mode = ["nearest" if k == "label" else "area" for k in keys]
                xforms.append(
                    RandZoomd(
                        keys=keys, prob=0.2, min_zoom=0.8, max_zoom=1.3, mode=mode
                    )
                )

            if self.spatial_size is None:
                spatial_size = tuple(96 for _ in range(self.spatial_dims))
            xforms.append(
                RandCropByLabelClassesd(
                    keys=keys,
                    label_key="label",
                    image_key="image",
                    spatial_size=spatial_size,
                    num_classes=self.num_classes,
                    num_samples=self.num_samples,
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
            cache_rate=self.cache_rate,
            num_workers=0,
        )
        self.val_ds = CacheDataset(
            data=self.dataset.validation_files(),
            transform=val_transforms,
            cache_rate=self.cache_rate,
            num_workers=0,
        )

    def train_dataloader(self):
        print(len(self.train_ds))
        print(type(self.train_ds))
        train_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=6,
            shuffle=True,
            num_workers=0,
            collate_fn=list_data_collate,
            # collate_fn=pad_list_data_collate
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_ds, batch_size=1, num_workers=0
        )
        return val_loader

    def configure_optimizers(self):
        if self.optimizer['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(self._model.parameters(),
                                        lr=self.optimizer['lr'],
                                        momentum=self.optimizer['momentum'])
        elif self.optimizer['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(self._model.parameters(),
                                         lr=self.optimizer['lr'],
                                         amsgrad=self.optimizer['amsgrad'])
        elif self.optimizer['optimizer'] == 'AdaBelief':
            optimizer = AdaBelief(self._model.parameters(),
                                  lr=self.optimizer['lr'],
                                  eps=self.optimizer['epsilon'],  # try 1e-8 and 1e-16
                                  betas=(0.9, 0.999),
                                  weight_decouple=self.optimizer['weight_decouple'],  # Try True/False
                                  fixed_decay=False,
                                  rectify=False)

        if self.lr_scheduling['scheduler'] == 'Constant':
            lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer=optimizer,
                                                               factor=1,
                                                               total_iters=0)

        elif self.lr_scheduling['scheduler'] == 'ReduceOnPlateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                      mode='min',
                                                                      factor=self.lr_scheduling['factor'],
                                                                      patience=self.lr_scheduling['patience'],
                                                                      verbose=True)

        elif self.lr_scheduling['scheduler'] == 'Cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                                                T_0=self.lr_scheduling['T_0'],
                                                                                T_mult=self.lr_scheduling['T_multi'],
                                                                                eta_min=0)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)

        optimizer = self.optimizers()
        # scheduler = self.lr_schedulers()

        optimizer.zero_grad()
        loss = self.loss_function(output, labels)
        self.manual_backward(loss)
        optimizer.step()

        # if self.trainer.is_last_batch:
        #    print('updating learning rate')
        #    scheduler.step()

        # if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 10 == 0:
        #    print('updating learning rate')
        #    scheduler.step()

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
        if self.lr_scheduling['scheduler'] == 'ReduceOnPlateau':
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
            f"current mean dice: {mean_val_dice:.4f}"
            f"\ncurrent mean loss: {mean_val_loss:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.log("val_dice", mean_val_dice)
        self.log("val_loss", mean_val_loss)
        return {"log": tensorboard_logs}


def train(
        image_dir: Path,
        labels_dir: Path,
        tissue_list: Path,
        output_dir: Path,
        checkpoint_file: Path = None,
        num_channels: int = 1,
        spatial_dims: int = 3,
        spatial_size: Sequence[int] = None,
        layers: tuple = (16, 32, 64, 128, 256),
        strides: tuple = (2, 2, 2, 2),
        dropout: float = 0.0,
        max_epochs: int = 600,
        augment_intensity: bool = False,
        augment_spatial: bool = False,
        num_samples: int = 4,
        optimizer=None,
        lr_scheduling=None,
        mixed_precision: bool = True,
        cache_rate: float = 1.0,
        save_nifti: bool = True,
        gpu_ids: List[int] = [0],
):
    if optimizer is None:
        optimizer = {'optimizer': 'Adam',
                     'lr': 1e-4,
                     'momentum': 0.9,
                     'epsilon': 1e-8,
                     'amsgrad': False,
                     'weight_decouple': False
                     }
    if lr_scheduling is None:
        lr_scheduling = {'scheduler': 'Constant',
                         'factor': 0.5,
                         'patience': 10,
                         'T_0': 50,
                         'T_multi': 1}
    print_config()

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    log_dir = output_dir / "logs"

    tissue_dict = load_tissue_list(tissue_list)
    num_classes = max(tissue_dict.values()) + 1
    if not len(tissue_dict) == num_classes:
        raise ValueError("Expecting contiguous labels in range [0,N-1]")

    device = make_device(gpu_ids)

    """Run the training"""
    # initialise the LightningModule
    if checkpoint_file and Path(checkpoint_file).exists():
        net = Net.load_from_checkpoint(f"{checkpoint_file}")
    else:
        net = Net(
            num_classes=num_classes,
            num_channels=num_channels,
            spatial_dims=spatial_dims,
            spatial_size=spatial_size,
            layers=layers,
            strides=strides,
            dropout=dropout
        )
    net.dataset = DataSet(image_dir=image_dir, labels_dir=labels_dir)
    net.intensity_augmentation = augment_intensity
    net.spatial_augmentation = augment_spatial
    net.num_samples = num_samples
    net.optimizer = optimizer
    net.lr_scheduling = lr_scheduling
    net.cache_rate = cache_rate

    # set up loggers and checkpoints
    tb_logger = pytorch_lightning.loggers.TensorBoardLogger(save_dir=f"{log_dir}")
    checkpoint_callback = ModelCheckpoint(
        filename=os.path.join(output_dir, "{epoch}-{val_loss:.2f}-{val_dice:.4f}"),
        monitor="val_dice",
        mode="max",
        dirpath=output_dir if output_dir else log_dir,
        save_top_k=3,
    )

    # defining early stopping. When val loss improves less than 0 over 30 epochs, the training will be stopped.
    early_stop_callback = EarlyStopping(monitor="val_dice",
                                        min_delta=0.00,
                                        patience=50,
                                        mode='max',
                                        check_finite=True,
                                        )

    lr_monitor = LearningRateMonitor(log_momentum=True,
                                     logging_interval='epoch')

    # initialise Lightning's trainer.
    # other options:
    #  - max_time={"days": 1, "hours": 5}
    trainer = pytorch_lightning.Trainer(
        gpus=gpu_ids,
        auto_scale_batch_size=True,
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

    """## View training in tensorboard"""

    # Commented out IPython magic to ensure Python compatibility.
    # %load_ext tensorboard
    # %tensorboard --logdir=log_dir

    """## Check best model output with the input image and label"""
    if save_nifti:
        os.makedirs(output_dir, exist_ok=True)
        saver = NiftiSaver(output_dir=output_dir, separate_folder=False, resample=False)

    net.eval()
    net.to(device)
    with torch.no_grad():
        cmap = make_tissue_cmap(tissue_list)

        for i, val_data in enumerate(net.val_dataloader()):
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(
                val_data["image"].to(device), roi_size, sw_batch_size, net
            )

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

            # if saver:
            #    pred_labels = val_outputs.argmax(dim=1, keepdim=True)
            #    saver.save_batch(pred_labels, val_data["image_meta_dict"])


def predict(
        model_file: Path,
        output_dir: Path,
        test_images: List[Path],
        test_labels: Optional[List[Path]] = None,
        tissue_dict: Dict[str, int] = None,
        layers: tuple = (16, 32, 64, 128, 256),
        strides: tuple = (2, 2, 2, 2),
        dropout: float = 0.0,
        save_nifti: bool = True,
        gpu_ids: list = [],
):
    # load trained model
    model_settings_json = model_file.with_suffix(".json")
    if model_settings_json.exists():
        print(f"Loading legacy model settings from {model_settings_json}")
        with model_settings_json.open() as json_file:
            settings = json.load(json_file)
        net = Net.load_from_checkpoint(f"{model_file}", **settings)
    else:
        net = Net.load_from_checkpoint(f"{model_file}", layers=layers, strides=strides, dropout=dropout)
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
    all_mean_dice = []
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
                all_mean_dice.append(dice_metric.aggregate().item())
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
        np.savetxt(output_dir.joinpath('mean_dice_' + str(model_file.stem) + '_generalize.txt'), all_mean_dice,
                   delimiter=',')


def cross_validate(
        image_dir: Path,
        labels_dir: Path,
        tissue_list: Path,
        output_dir: Path,
        checkpoint_file: Path = None,
        num_channels: int = 1,
        spatial_dims: int = 3,
        spatial_size: Sequence[int] = None,
        max_epochs: int = 100,
        augment_intensity: bool = False,
        augment_spatial: bool = False,
        num_samples: int = 4,
        mixed_precision: bool = True,
        cache_rate: float = 1.0,
        n_splits: int = 7,
        save_nifti: bool = True,
        gpu_ids: List[int] = [0],
        evaluate: bool = False,
):
    print_config()
    print('Cross-validating')
    print(augment_intensity)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    tissue_dict = load_tissue_list(tissue_list)

    if not evaluate:
        kf = KFold(n_splits=n_splits)

        image_files = sorted(Path(image_dir).glob("*.nii.gz"))
        label_files = sorted(Path(labels_dir).glob("*.nii.gz"))

        image_idx = np.arange(len(image_files))

        cv_split = Path(Path(image_dir).parent) / 'CV_Split'
        cv_split_train = cv_split / 'train'
        cv_split_predict = cv_split / 'predict'
        cv_split_train_img = cv_split_train / 'images'
        cv_split_train_lbl = cv_split_train / 'labels'
        cv_split_predict_img = cv_split_predict / 'images'
        cv_split_predict_lbl = cv_split_predict / 'labels'

        all_paths = [cv_split, cv_split_train, cv_split_predict,
                     cv_split_train_img, cv_split_train_lbl, cv_split_predict_img, cv_split_predict_lbl]

        for path in all_paths:
            if not path.exists():
                path.mkdir()

        # test_layers = [(16, 32, 64, 128), (16, 32, 64, 128, 256), (16, 32, 64, 128, 256, 516)]
        # test_strides = [(2, 2, 2), (2, 2, 2, 2), (2, 2, 2, 2, 2)]
        # test_layers = [(16, 32, 64, 128, 256), (16, 32, 64, 128, 256, 516)]
        # test_strides = [(2, 2, 2, 2), (2, 2, 2, 2, 2)]
        test_layers = [(16, 32, 64, 128, 256)]
        test_strides = [(2, 2, 2, 2)]

        for scenario in range(len(test_layers)):
            output_dir_scenario = output_dir.joinpath(str(test_layers[scenario]))
            if not output_dir_scenario.exists():
                output_dir_scenario.mkdir()
            for train_idx, test_idx in kf.split(image_idx, image_idx):

                current_output = output_dir_scenario.joinpath(str(test_idx))
                print(current_output)
                if current_output.exists():
                    continue
                else:
                    current_output.mkdir()

                # Delete all files from the temporary file dirs
                for folder in all_paths:
                    if folder.match('images') or folder.match('labels'):
                        for file in folder.iterdir():
                            file.unlink()

                # set new splits
                image_train = [image_files[index] for index in train_idx]
                image_test = [image_files[index] for index in test_idx]
                label_train = [label_files[index] for index in train_idx]
                label_test = [label_files[index] for index in test_idx]

                for file in image_train:
                    shutil.copyfile(file, cv_split_train_img.joinpath(file.name))
                for file in image_test:
                    shutil.copyfile(file, cv_split_predict_img.joinpath(file.name))
                for file in label_train:
                    shutil.copyfile(file, cv_split_train_lbl.joinpath(file.name))
                for file in label_test:
                    shutil.copyfile(file, cv_split_predict_lbl.joinpath(file.name))
                print('start training')
                train(image_dir=cv_split_train_img,
                      labels_dir=cv_split_train_lbl,
                      tissue_list=tissue_list,
                      output_dir=current_output,
                      num_channels=num_channels,
                      spatial_dims=spatial_dims,
                      spatial_size=spatial_size,
                      layers=test_layers[scenario],
                      strides=test_strides[scenario],
                      max_epochs=600,
                      augment_intensity=augment_intensity,
                      augment_spatial=augment_spatial,
                      num_samples=num_samples,
                      optimizer='Adam',
                      lr_scheduling='Constant',
                      mixed_precision=mixed_precision,
                      cache_rate=cache_rate,
                      save_nifti=save_nifti,
                      gpu_ids=gpu_ids)

                print('training finished')

                for file in current_output.iterdir():
                    if file.match('*.ckpt'):
                        print('start prediction')
                        predict(model_file=file,
                                output_dir=current_output,
                                test_images=image_test,
                                test_labels=label_test,
                                tissue_dict=tissue_dict,
                                layers=test_layers[scenario],
                                strides=test_strides[scenario],
                                save_nifti=save_nifti,
                                gpu_ids=gpu_ids)
    else:
        print('evaluating')
        model_scores = []
        model_general_scores = []
        column_names = []
        for model in Path(output_dir).iterdir():
            if model.is_dir():
                print(model.name)
                column_names.append(model.name)
                temp_fold_mean = []
                temp_fold_general_mean = []
                for fold in model.iterdir():
                    print(fold.name)
                    all_scores_fold = []
                    all_scores_general_fold = []
                    for file in fold.iterdir():
                        if file.suffix == '.txt' and 'generalize' not in str(file):
                            temp_data = np.genfromtxt(file, delimiter=',')
                            temp_mean_model = np.mean(temp_data)
                            all_scores_fold.append(temp_mean_model)
                        elif file.suffix == '.txt' and 'generalize' in str(file):
                            temp_data = np.genfromtxt(file, delimiter=',')
                            temp_mean_model = np.mean(temp_data)
                            all_scores_general_fold.append(temp_mean_model)
                    temp_fold_mean.append(np.mean(all_scores_fold))
                    temp_fold_general_mean.append(np.mean(all_scores_general_fold))
                model_scores.append(temp_fold_mean)
                model_general_scores.append(temp_fold_general_mean)
            else:
                continue

        model_scores_np = np.asarray(model_scores)
        model_general_scores_np = np.asarray(model_general_scores)
        print(column_names)
        print(model_scores_np)
        print(model_general_scores_np)

        # column_names = ['4_layers', '5_layers', '6_layers']
        model_scores_df = pd.DataFrame(data=model_scores_np.transpose(),
                                       columns=column_names)
        model_scores_df.plot(kind='box')

        print(model_scores_df)
        plt.show()

        model_scores_df = pd.DataFrame(data=model_general_scores_np.transpose(),
                                       columns=column_names)
        model_scores_df.plot(kind='box')

        print(model_scores_df)
        plt.show()
        assert False
        generalize_test_img_path = Path(
            "M:/DATA/ITIS/MasterThesis/Datasets/T1_T2/T1_T2_spacing_(1, 1, 1)/Hummel_Nibabel")
        generalize_test_label_path = Path("M:/DATA/ITIS/MasterThesis/Labels/SimNIBS/cat_12/no_ventricles/Hummel")

        generalize_img = []
        generalize_label = []

        for file in generalize_test_img_path.iterdir():
            generalize_img.append(file)
        for file in generalize_test_label_path.iterdir():
            generalize_label.append(file)

        print(generalize_img)
        print(generalize_label)

        test_layers = [(16, 32, 64, 128, 256)]
        test_strides = [(2, 2, 2, 2)]

        for folder in Path(output_dir).iterdir():
            if folder.is_dir():
                for fold in folder.iterdir():
                    for file in fold.iterdir():
                        if file.match('*.ckpt'):
                            print('start prediction')
                            predict(model_file=file,
                                    output_dir=fold,
                                    test_images=generalize_img,
                                    test_labels=generalize_label,
                                    tissue_dict=tissue_dict,
                                    layers=test_layers[0],
                                    strides=test_strides[0],
                                    save_nifti=save_nifti,
                                    gpu_ids=gpu_ids)
            else:
                continue


def evolution(
        image_dir: Path,
        labels_dir: Path,
        test_img_dir: Path,
        test_lbl_dir: Path,
        tissue_list: Path,
        output_dir: Path,
        checkpoint_file: Path = None,
        num_channels: int = 1,
        spatial_dims: int = 3,
        spatial_size: Sequence[int] = None,
        max_epochs: int = 150,
        population_size: int = 10,
        number_of_generations: int = 50,
        mixed_precision: bool = True,
        cache_rate: float = 1.0,
        save_nifti: bool = False,
        gpu_ids: List[int] = [0],
):
    print_config()
    print('Starting population evolution')
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    test_img_files = sorted(Path(test_img_dir).glob("*.nii.gz"))
    test_lbl_files = sorted(Path(test_lbl_dir).glob("*.nii.gz"))

    tissue_dict = load_tissue_list(tissue_list)

    # Initialize population
    parent_population = initialize_population(population_size)
    print(f'This is the initial population: {parent_population}')

    generation_0_dir = output_dir / '0'
    generation_0_dir.mkdir(exist_ok=True)

    # Evaluate fitness of initial population
    parent_population_fitness = fitness(parent_population,
                                        image_dir=image_dir,
                                        labels_dir=labels_dir,
                                        test_img_files=test_img_files,
                                        test_lbl_files=test_lbl_files,
                                        tissue_list=tissue_list,
                                        tissue_dict=tissue_dict,
                                        output_dir=generation_0_dir,
                                        checkpoint_file=checkpoint_file,
                                        num_channels=num_channels,
                                        spatial_dims=spatial_dims,
                                        spatial_size=spatial_size,
                                        max_epochs=max_epochs,
                                        mixed_precision=mixed_precision,
                                        cache_rate=cache_rate,
                                        save_nifti=save_nifti,
                                        gpu_ids=gpu_ids)

    for generation in range(number_of_generations):
        offspring_population = []
        current_generation_dir = generation_0_dir.with_name(str(generation+1))
        current_generation_dir.mkdir(exist_ok=True)

        while len(offspring_population) < population_size:
            o_1, o_2 = crossover(parent_population, fitness=parent_population_fitness,
                                 p_c=0.9,
                                 mu=0.2)
            o_1 = mutation(o_1, 0.7, 0.05)
            o_2 = mutation(o_2, 0.7, 0.05)

            offspring_population.append(o_1)
            offspring_population.append(o_2)

        # evaluate fitness of offspring generation
        offspring_population_fitness = fitness(parent_population,
                                               image_dir=image_dir,
                                               labels_dir=labels_dir,
                                               test_img_files=test_img_files,
                                               test_lbl_files=test_lbl_files,
                                               tissue_list=tissue_list,
                                               tissue_dict=tissue_dict,
                                               output_dir=current_generation_dir,
                                               checkpoint_file=checkpoint_file,
                                               num_channels=num_channels,
                                               spatial_dims=spatial_dims,
                                               spatial_size=spatial_size,
                                               max_epochs=max_epochs,
                                               mixed_precision=mixed_precision,
                                               cache_rate=cache_rate,
                                               save_nifti=save_nifti,
                                               gpu_ids=gpu_ids)

        parent_population, parent_population_fitness = environmental_selection(parent_population,
                                                                               offspring_population,
                                                                               fitness_parents=parent_population_fitness,
                                                                               fitness_offspring=offspring_population_fitness)
        # Intermediate saves of the current parrent population
        parent_population_np = np.asarray(parent_population)
        for count, genotype in enumerate(parent_population_np):
            parent_population_df = pd.DataFrame(genotype)
            temp_name = f'parent_gen_genotype_{count}.csv'
            temp_path = current_generation_dir / temp_name
            parent_population_df.to_csv(temp_path)

    # Saving the final parent population
    print(parent_population)
    print(parent_population_fitness)
    parent_population_np = np.asarray(parent_population)
    for count, genotype in enumerate(parent_population_np):
        parent_population_df = pd.DataFrame(genotype)
        temp_name = f'final_gen_genotype_{count}.csv'
        temp_path = output_dir / temp_name
        parent_population_df.to_csv(temp_path)


def fitness(
        population,
        image_dir: Path,
        labels_dir: Path,
        test_img_files: List[Path],
        test_lbl_files: List[Path],
        tissue_list: Path,
        tissue_dict: dict,
        output_dir: Path,
        checkpoint_file: Path = None,
        num_channels: int = 1,
        spatial_dims: int = 3,
        spatial_size: Sequence[int] = None,
        max_epochs: int = 150,
        mixed_precision: bool = True,
        cache_rate: float = 1.0,
        save_nifti: bool = False,
        gpu_ids: List[int] = [0],

):
    population_fitness = []
    for count, genotype in enumerate(population):
        print(type(genotype))
        # Create output folder for genotype
        folder_name = f'{count}.{genotype}'
        current_output = output_dir / folder_name
        current_output.mkdir(exist_ok=True)

        # genotype = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1]
        # decode genotype
        print(genotype)
        augment_intensity = True if genotype[0] else False
        augment_spatial = True if genotype[1] else False

        dropout_gene = genotype[2:4]
        if dropout_gene == [0, 0]:
            dropout = 0.0
        elif dropout_gene == [0, 1]:
            dropout = 0.2
        elif dropout_gene == [1, 0]:
            dropout = 0.3
        else:
            dropout = 0.5

        print(dropout_gene)
        print(dropout)

        layers_gene = genotype[4:6]
        if layers_gene == [0, 0]:
            layers = (16, 32, 64, 128)
            strides = (2, 2, 2)
        elif layers_gene == [0, 1]:
            layers = (16, 32, 64, 128, 256)
            strides = (2, 2, 2, 2)
        elif layers_gene == [1, 0]:
            layers = (64, 128, 256, 512, 1024)
            strides = (2, 2, 2, 2)
        else:
            layers = (16, 32, 64, 128, 256, 516)
            strides = (2, 2, 2, 2, 2)

        print(layers_gene)
        print(layers)
        print(strides)

        num_samples_gene = genotype[6:8]
        if num_samples_gene == [0, 0]:
            num_samples = 3
        elif num_samples_gene == [0, 1]:
            num_samples = 4
        elif num_samples_gene == [1, 0]:
            num_samples = 5
        else:
            num_samples = 6

        print(num_samples_gene)
        print(num_samples)

        optimizer_gene = genotype[8:12]
        optimizer_gene_np = np.asarray(optimizer_gene)
        bin_to_dec = optimizer_gene_np.dot(1 << np.arange(optimizer_gene_np.shape[-1]))
        optimizer = decode_optimizer_gene(bin_to_dec)

        print(optimizer_gene)
        print(bin_to_dec)
        print(optimizer)

        lr_scheduler_gene = genotype[12:15]
        lr_scheduler_gene_np = np.asarray(lr_scheduler_gene)
        bin_to_dec = lr_scheduler_gene_np.dot(1 << np.arange(lr_scheduler_gene_np.shape[-1]))
        lr_scheduler = decode_lr_scheduler_gene(bin_to_dec)

        print(lr_scheduler_gene)
        print(bin_to_dec)
        print(lr_scheduler)

        # train network with decoded settings
        train(image_dir=image_dir,
              labels_dir=labels_dir,
              tissue_list=tissue_list,
              output_dir=current_output,
              num_channels=num_channels,
              spatial_dims=spatial_dims,
              spatial_size=spatial_size,
              layers=layers,
              strides=strides,
              max_epochs=max_epochs,
              augment_intensity=augment_intensity,
              augment_spatial=augment_spatial,
              num_samples=num_samples,
              optimizer=optimizer,
              lr_scheduling=lr_scheduler,
              dropout=dropout,
              mixed_precision=mixed_precision,
              cache_rate=cache_rate,
              save_nifti=save_nifti,
              gpu_ids=gpu_ids)

        print('training finished')
        # predict on test set
        temp_best_dice = 0
        for file in current_output.iterdir():
            if file.match('*.ckpt'):
                if float(file.stem[-6:]) > temp_best_dice:
                    temp_best_dice = float(file.stem[-6:])
                    current_best_model = file
                    print(temp_best_dice)
                    print(current_best_model)
        print('start prediction')
        predict(model_file=current_best_model,
                output_dir=current_output,
                test_images=test_img_files,
                test_labels=test_lbl_files,
                tissue_dict=tissue_dict,
                layers=layers,
                strides=strides,
                dropout=dropout,
                save_nifti=save_nifti,
                gpu_ids=gpu_ids)

        # save dice score to population fitness
        for file in current_output.iterdir():
            if file.match('*.txt'):
                temp_data = np.genfromtxt(file, delimiter=',')
                temp_mean_model = np.mean(temp_data)
        population_fitness.append(temp_mean_model)
        print(population_fitness)

    return population_fitness

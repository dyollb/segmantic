from monai.transforms.spatial.dictionary import Spacingd
from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByLabelClassesd,
    RandFlipd,
    RandAffined,
    NormalizeIntensityd,
    EnsureTyped,
    EnsureType,
    Activations,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate, decollate_batch, NiftiSaver
from monai.config import print_config
from monai.networks.utils import one_hot
import torch
import pytorch_lightning
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import matplotlib.pyplot as plt
from matplotlib import colors
import colorsys
import numpy as np
import random
import os
import glob
import json
import argparse

from segmantic.prepro.labels import load_tissue_list


def make_random_cmap(num_classes):
    '''Make a random color map for <num_classes> different classes'''
    def random_color(l, max_label):
        if l == 0:
            return (0, 0, 0)
        hue = l / (2.0 * max_label) + (l % 2) * 0.5
        hue = min(hue, 1.0)
        return colorsys.hls_to_rgb(hue, 0.5, 1.0)

    col = np.zeros(
        (
            num_classes,
            3,
        )
    )

    random.seed(0)
    for i in random.sample(range(num_classes), num_classes):
        r, g, b = random_color(i, num_classes)
        col[i, 0] = r
        col[i, 1] = g
        col[i, 2] = b
    return colors.ListedColormap(col)


def compute_confusion_naive(num_classes, y_pred, y):
    """
    Compute confusion matrix similar to sklearn.metrics.confusion_matrix

    num_classes:    number of labels including '0', i.e. max(y)+1
    y_pred:         predicted labels
    y:              true labels
    """
    confusion_matrix = torch.zeros(num_classes, num_classes)
    for t, p in zip(y.view(-1), y_pred.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
    return confusion_matrix


def compute_confusion(y_pred, y):
    """
    Returns confusion matrix computed with sklearn.metrics.confusion_matrix

    num_classes:    number of labels including '0', i.e. max(y)+1
    y_pred:         predicted labels
    y:              true labels
    """
    from sklearn.metrics import confusion_matrix

    return confusion_matrix(y.view(-1).cpu().numpy(), y_pred.view(-1).cpu().numpy())


def plot_confusion_matrix(
    cm,
    target_names,
    title="Confusion matrix",
    cmap=None,
    normalize=True,
    file_name=None,
):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap("Blues")

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(
                j,
                i,
                "{:0.4f}".format(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )
        else:
            plt.text(
                j,
                i,
                "{:,}".format(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel(
        "Predicted label\naccuracy={:0.4f}; misclass={:0.4f}".format(accuracy, misclass)
    )
    if file_name:
        plt.savefig(file_name)
    else:
        plt.show()


def create_transforms(keys, train=False, num_classes=0, spacing=None):
    # loading and normalization
    xforms = [
        LoadImaged(keys=keys),
        AddChanneld(keys=keys),
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
                RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            ]
        )
        if num_classes > 0:
            xforms.append(
                RandCropByLabelClassesd(
                    keys=["image", "label"],
                    label_key="label",
                    image_key="image",
                    spatial_size=(96, 96, 96),
                    num_classes=num_classes,
                    num_samples=4,
                )
            )
    return Compose(xforms + [EnsureTyped(keys=keys)])


class Net(pytorch_lightning.LightningModule):
    def __init__(self, n_classes, image_dir="", labels_dir="", model_file_name=""):
        super().__init__()
        self._model = UNet(
            dimensions=3,
            in_channels=1,
            out_channels=n_classes,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )
        self.num_classes = n_classes
        self.image_dir = image_dir
        self.labels_dir = labels_dir
        self.model_file_name = model_file_name
        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.post_pred = Compose(
            [EnsureType(), AsDiscrete(argmax=True, to_onehot=True, n_classes=n_classes)]
        )
        self.post_label = Compose(
            [EnsureType(), AsDiscrete(to_onehot=True, n_classes=n_classes)]
        )
        self.dice_metric = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False
        )
        self.best_val_dice = 0
        self.best_val_epoch = 0
        # self.save_hyperparameters("n_classes")

    def forward(self, x):
        return self._model(x)

    def prepare_data(self):
        # set up the correct data path
        train_images = sorted(glob.glob(os.path.join(self.image_dir, "*.nii.gz")))
        train_labels = sorted(glob.glob(os.path.join(self.labels_dir, "*.nii.gz")))
        data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(train_images, train_labels)
        ]

        # use first 4 files for validation, rest for training
        # todo: make this configurable/not hard-coded
        train_files, val_files = data_dicts[4:], data_dicts[:4]

        # set deterministic training for reproducibility
        set_determinism(seed=0)

        # define the data transforms
        train_transforms = create_transforms(
            keys=["image", "label"], num_classes=self.num_classes, train=True
        )
        val_transforms = create_transforms(keys=["image", "label"], train=False)

        # we use cached datasets - these are 10x faster than regular datasets
        self.train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=1.0,
            num_workers=0,
        )
        self.val_ds = CacheDataset(
            data=val_files,
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
        roi_size = (160, 160, 160)
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
    image_dir: str,
    labels_dir: str,
    log_dir: str,
    num_classes: int,
    model_file_name: str,
    max_epochs: int = 600,
    output_dir=None,
):
    """Run the training"""

    # initialise the LightningModule
    net = Net(n_classes=num_classes, image_dir=image_dir, labels_dir=labels_dir)

    # set up loggers and checkpoints
    tb_logger = pytorch_lightning.loggers.TensorBoardLogger(save_dir=log_dir)
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
        gpus=[0],
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

    settings = {"num_classes": num_classes}
    with open(model_file_name.rsplit(".", 1)[0] + ".json", "w") as json_file:
        json.dump(settings, json_file)

    trainer.save_checkpoint(model_file_name)

    """## View training in tensorboard"""

    # Commented out IPython magic to ensure Python compatibility.
    # %load_ext tensorboard
    # %tensorboard --logdir=log_dir

    """## Check best model output with the input image and label"""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        saver = NiftiSaver(output_dir=output_dir, separate_folder=False, resample=False)

    net.eval()
    device = torch.device("cuda:0")
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
            if output_dir:
                plt.savefig(
                    os.path.join(output_dir, "drcmr%02d_case%d.png" % (num_classes, i))
                )
            else:
                plt.show()

            if output_dir:
                pred_labels = val_outputs.argmax(dim=1, keepdim=True)
                saver.save_batch(pred_labels, val_data["image_meta_dict"])


def predict(
    model_file: str,
    test_images: list,
    test_labels: list = None,
    tissue_dict: dict = None,
    output_dir: str = None,
    save_nifti: bool = False,
):

    with open(model_file.rsplit(".", 1)[0] + ".json") as json_file:
        settings = json.load(json_file)

    net = Net.load_from_checkpoint(model_file, n_classes=settings["num_classes"])

    net.eval()
    device = torch.device("cuda:0")
    net.to(device)

    if test_labels:
        test_files = [
            {"image": i, "label": l} for i, l in zip(test_images, test_labels)
        ]
    else:
        test_files = [{"image": i} for i in test_images]

    test_ds = CacheDataset(
        data=test_files,
        transform=create_transforms(
            keys=["image", "label"] if test_labels else ["image"],
            train=False,
            spacing=(0.85, 0.85, 0.85),
        ),
        cache_rate=1.0,
        num_workers=0,
    )

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, num_workers=0)

    # for saving output
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        saver = NiftiSaver(output_dir=output_dir, separate_folder=False, resample=False)

    dice_metric = DiceMetric(
        include_background=False, reduction="mean", get_not_nans=False
    )
    conf_matrix = ConfusionMatrixMetric(
        metric_name=["sensitivity", "specificity", "precision", "accuracy"]
    )

    to_one_hot = lambda x: one_hot(x, num_classes=num_classes, dim=0)

    tissue_names = [""] * num_classes
    if tissue_dict:
        for idx in tissue_dict.keys():
            tissue_names[idx] = tissue_dict[idx]

    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            val_image = test_data["image"].to(device)

            roi_size = (96, 96, 96)
            sw_batch_size = 4
            val_pred = sliding_window_inference(val_image, roi_size, sw_batch_size, net)
            val_pred = val_pred.argmax(dim=1, keepdim=True)

            if test_labels:
                val_labels = test_data["label"].to(device)

                d = dice_metric(y_pred=to_one_hot(val_pred), y=to_one_hot(val_labels))
                print("Class Dice = ", d)
                print("Mean Dice = ", dice_metric.aggregate().item())
                dice_metric.reset()

                conf_matrix(y_pred=to_one_hot(val_pred), y=to_one_hot(val_labels))
                print("Conf. Matrix Metrics = ", conf_matrix.aggregate())

                c = compute_confusion(y_pred=val_pred, y=val_labels)
                # print("Conf. Matrix = ", c)
                plot_confusion_matrix(c, tissue_names)

            if save_nifti and output_dir:
                saver.save_batch(val_pred, test_data["image_meta_dict"])


def get_nifti_files(dir):
    if not dir:
        return []
    return sorted([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".nii.gz")])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and predict.')
    parser.add_argument('-i', '--image_dir', dest='image_dir', type=str, required=True, help='image directory')
    parser.add_argument('-l', '--labels_dir', dest='labels_dir', type=str, help='label image directory')
    parser.add_argument('-o', '--results_dir', dest='results_dir', default='.', type=str, help='results directory')
    parser.add_argument('--tissue_list', type=str, required=True, help='file containing label descriptors')
    parser.add_argument('--predict', action='store_true', help='run prediction')
    args = parser.parse_args()

    print_config()

    tissue_dict = load_tissue_list(args.tissue_list)
    num_classes = max(tissue_dict.keys()) + 1
    assert len(tissue_dict) == num_classes, "Expecting contiguous labels in range [0,N-1]"

    os.makedirs(args.results_dir, exist_ok=True)
    log_dir = os.path.join(args.results_dir, "logs")
    model_file = os.path.join(args.results_dir, "drcmr_%d.ckpt" % num_classes)


    if args.predict:
        predict(
            model_file=model_file,
            test_labels=get_nifti_files(args.image_dir),
            test_images=get_nifti_files(args.labels_dir),
            tissue_dict=tissue_dict,
            output_dir=args.results_dir,
            save_nifti=True,
        )
    else:
        train(
            image_dir=args.image_dir,
            labels_dir=args.labels_dir,
            log_dir=log_dir,
            num_classes=num_classes,
            model_file_name=model_file,
            max_epochs=600,
            output_dir=args.results_dir,
        )

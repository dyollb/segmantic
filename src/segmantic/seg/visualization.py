import colorsys
import itertools
import random
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from ..image.labels import RGBTuple, load_tissue_colors


def make_tissue_cmap(tissue_list_file: Path) -> colors.ListedColormap:
    """Make a color map for from an iSEG tissue list file"""
    tissue_color_map = load_tissue_colors(tissue_list_file)
    num_classes = max(tissue_color_map.keys()) + 1
    col = np.zeros((num_classes, 3))
    for idx, c in tissue_color_map.items():
        r, g, b = c
        col[idx, 0] = r
        col[idx, 1] = g
        col[idx, 2] = b
    return colors.ListedColormap(col)


def make_random_cmap(num_classes: int) -> colors.ListedColormap:
    """Make a random color map for <num_classes> different classes"""

    def random_color(id: int, max_label: int) -> RGBTuple:
        if id == 0:
            return (0, 0, 0)
        hue = id / (2.0 * max_label) + (id % 2) * 0.5
        hue = min(hue, 1.0)
        return colorsys.hls_to_rgb(hue, 0.5, 1.0)

    col = np.zeros((num_classes, 3))

    random.seed(0)
    for i in random.sample(range(num_classes), num_classes):
        r, g, b = random_color(i, num_classes)
        col[i, 0] = r
        col[i, 1] = g
        col[i, 2] = b
    return colors.ListedColormap(col)


def plot_confusion_matrix(
    cm: np.ndarray,
    target_names: List[str],
    title: str = "Confusion matrix",
    cmap: colors.Colormap = None,
    normalize: bool = True,
    file_name: Optional[Path] = None,
) -> None:
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

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap("Blues")

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure(figsize=(16, 16))
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
                f"{cm[i, j]:0.4f}",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )
        else:
            plt.text(
                j,
                i,
                f"{cm[i, j]:,}",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel(f"Predicted label\naccuracy={accuracy:0.4f}; misclass={misclass:0.4f}")
    if file_name:
        plt.savefig(file_name)
    else:
        plt.show()
    plt.close(fig)

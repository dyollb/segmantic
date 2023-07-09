import warnings
from typing import List, Optional, Union

import torch
from monai.networks import one_hot
from monai.utils import LossReduction
from torch.nn.modules.loss import _Loss


class BoundaryLoss(_Loss):
    def __init__(
        self,
        include_background: bool = True,
        argmax: bool = True,
        threshold: Optional[float] = None,
        to_onehot_y: bool = False,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get overwhelmed
                by the signal from the background so excluding it in such cases helps convergence.
            to_onehot_y: whether to convert `y` into the one-hot format.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

        Raises:
            ValueError: When more than 1 of [``argmax=True``, ``threshold is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if argmax and (threshold is not None):
            raise ValueError(
                "Incompatible values: more than 1 of [argmax=True, threshold is not None]."
            )
        self.include_background = include_background
        self.argmax = argmax
        self.threshold = threshold
        self.to_onehot_y = to_onehot_y

    def forward(
        self, pred: torch.Tensor, seg_gt: torch.Tensor, dist_gt: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: the shape should be BNH[WD], where N is the number of classes.
            seg_gt: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.
            dist_gt: the shape should be BNH[WD], where N is the number of classes.

        Raises:
            AssertionError: When pred and seg_gt (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        num_classes = dist_gt.shape[1]

        if num_classes == 1:
            warnings.warn("single channel prediction, `argmax=True` ignored.")
        else:
            if self.argmax:
                pred = torch.argmax(pred, dim=1)
                pred = one_hot(pred, num_classes=num_classes, dim=1)
            if self.threshold is not None:
                pred = pred >= self.threshold

        if self.to_onehot_y:
            if num_classes == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                seg_gt = one_hot(seg_gt, num_classes=num_classes, dim=1)

        if dist_gt.shape != pred.shape:
            raise AssertionError(
                f"boundary distance has different shape ({dist_gt.shape}) from input ({pred.shape})"
            )
        if seg_gt.shape != pred.shape:
            raise AssertionError(
                f"ground truth has different shape ({seg_gt.shape}) from input ({pred.shape})"
            )

        if not self.include_background:
            if num_classes == 1:
                warnings.warn(
                    "single channel prediction, `include_background=False` ignored."
                )
            else:
                # if skipping background, removing first channel
                dist_gt = dist_gt[:, 1:]
                pred = pred[:, 1:]
                seg_gt = seg_gt[:, 1:]

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: List[int] = torch.arange(2, len(pred.shape)).tolist()

        pred_gt_xor = torch.logical_xor(seg_gt, pred)

        f = torch.sum(dist_gt * pred_gt_xor, dim=reduce_axis)

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(pred.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(
                f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].'
            )

        return f

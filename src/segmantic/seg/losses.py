from __future__ import annotations

import torch
from monai.networks import one_hot
from monai.utils import LossReduction
from torch.nn.modules.loss import _Loss


class AsymmetricUnifiedFocalLoss(_Loss):
    """
    AsymmetricUnifiedFocalLoss is a variant of Focal Loss.

    Implementation of the Asymmetric Unified Focal Loss described in:

    - "Unified Focal Loss: Generalising Dice and Cross Entropy-based Losses to Handle Class Imbalanced Medical Image Segmentation",
    Michael Yeung, Computerized Medical Imaging and Graphics
    - https://github.com/mlyg/unified-focal-loss/issues/8
    """

    def __init__(
        self,
        to_onehot_y: bool = False,
        num_classes: int = 2,
        gamma: float = 0.75,
        delta: float = 0.7,
        reduction: LossReduction | str = LossReduction.MEAN,
    ):
        """
        Args:
            to_onehot_y : whether to convert `y` into the one-hot format. Defaults to False.
            num_classes : number of classes. Defaults to 2.
            delta : weight of the background. Defaults to 0.7.
            gamma : value of the exponent gamma in the definition of the Focal loss. Defaults to 0.75.

        Example:
            >>> import torch
            >>> from monai.losses import AsymmetricUnifiedFocalLoss
            >>> pred = torch.ones((1,1,32,32), dtype=torch.float32)
            >>> grnd = torch.ones((1,1,32,32), dtype=torch.int64)
            >>> fl = AsymmetricUnifiedFocalLoss(to_onehot_y=True)
            >>> fl(pred, grnd)
        """
        super().__init__(reduction=LossReduction(reduction).value)
        self.to_onehot_y = to_onehot_y
        self.num_classes = num_classes
        self.gamma = gamma
        self.delta = delta
        self.epsilon = torch.tensor(1e-7)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred : the shape should be BNH[WD], where N is the number of classes.
                It only supports binary segmentation.
                The input should be the original logits since it will be transformed by
                    a sigmoid in the forward function.
            y_true : the shape should be BNH[WD], where N is the number of classes.
                It only supports binary segmentation.

        Raises:
            ValueError: When input and target are different shape
            ValueError: When len(y_pred.shape) != 4 and len(y_pred.shape) != 5
            ValueError: When num_classes
            ValueError: When the number of classes entered does not match the expected number
        """
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"ground truth has different shape ({y_true.shape}) from input ({y_pred.shape})"
            )

        if len(y_pred.shape) != 4 and len(y_pred.shape) != 5:
            raise ValueError(f"input shape must be 4 or 5, but got {y_pred.shape}")

        if y_pred.shape[1] == 1:
            y_pred = one_hot(y_pred, num_classes=self.num_classes)
            y_true = one_hot(y_true, num_classes=self.num_classes)

        if torch.max(y_true) != self.num_classes - 1:
            raise ValueError(
                f"Please make sure the number of classes is {self.num_classes-1}"
            )

        # use pytorch CrossEntropyLoss ?
        # https://github.com/Project-MONAI/MONAI/blob/2d463a7d19166cff6a83a313f339228bc812912d/monai/losses/dice.py#L741
        epsilon = torch.tensor(self.epsilon, device=y_pred.device)
        y_pred = torch.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * torch.log(y_pred)

        # calculate losses separately for each class, only enhancing foreground class
        back_ce = (
            torch.pow(1 - y_pred[:, 0, ...], self.gamma) * cross_entropy[:, 0, ...]
        )
        back_ce = (1 - self.delta) * back_ce

        losses = [back_ce]
        for i in range(1, self.num_classes):
            i_ce = cross_entropy[:, i, ...]
            i_ce = self.delta * i_ce
            losses.append(i_ce)

        loss = torch.stack(losses)
        if self.reduction == LossReduction.SUM.value:
            return torch.sum(loss)  # sum over the batch and channel dims
        if self.reduction == LossReduction.NONE.value:
            return loss  # returns [N, num_classes] losses
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(loss)
        raise ValueError(
            f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].'
        )

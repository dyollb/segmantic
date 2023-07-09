import torch
from monai.networks import one_hot

from segmantic.seg.boundary_loss import BoundaryLoss
from segmantic.transforms.distance import DistanceTransform


def test_BoundaryLoss_exact_match():
    mask = torch.zeros(8, 9, dtype=torch.bool)
    mask[2:5, 3:6] = 1

    distance_transform = DistanceTransform()
    df = distance_transform(mask)

    # add batch dimension
    mask = mask.unsqueeze(0).unsqueeze(0)
    df = df.unsqueeze(0)

    # pseudo prediction
    pred = one_hot(mask, num_classes=2, dim=1)

    print(pred.shape)
    print(mask.shape)
    print(df.shape)

    # loss for gt should be zero
    loss_function = BoundaryLoss(to_onehot_y=True, argmax=False)
    loss = loss_function(pred=pred, seg_gt=mask, dist_gt=df)
    assert loss == 0.0

    loss_function = BoundaryLoss(to_onehot_y=True, argmax=False, reduction="sum")
    loss = loss_function(pred=pred, seg_gt=mask, dist_gt=df)
    assert loss == 0.0


if __name__ == "__main__":
    test_BoundaryLoss_exact_match()

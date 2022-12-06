from typing import Dict, Hashable, Optional, Sequence, Union

import numpy as np
import torch
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import MapTransform, Transform
from monai.utils.enums import TransformBackends
from monai.utils.type_conversion import convert_to_dst_type, convert_to_numpy
from scipy.ndimage import binary_erosion, distance_transform_cdt, distance_transform_edt


def get_mask_edges(seg_gt, label_idx: int = 1) -> np.ndarray:
    """Get edges of mask region
    copied from monai.metrics.utils.get_mask_edges
    """

    if isinstance(seg_gt, torch.Tensor):
        seg_gt = seg_gt.detach().cpu().numpy()

    # If not binary images, convert them
    if seg_gt.dtype != bool:
        seg_gt = seg_gt == label_idx

    # Do binary erosion and use XOR to get edges
    edges_gt: np.ndarray = binary_erosion(seg_gt) ^ seg_gt

    return edges_gt


def get_boundary_distance(
    labels: np.ndarray,
    distance_metric: str = "euclidean",
    num_classes: int = 2,
    spacing: Optional[Union[float, Sequence[float]]] = None,
) -> np.ndarray:
    """
    This function is used to compute the surface distances to `labels`.
    Args:
        labels: the edge of the ground truth.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.
            - ``"euclidean"``, uses Exact Euclidean distance transform.
            - ``"chessboard"``, uses `chessboard` metric in chamfer type of transform.
            - ``"taxicab"``, uses `taxicab` metric in chamfer type of transform.
    Note:
        If labels is all 0, may result in nan/inf distance.
    """

    masks: np.ndarray
    if labels.dtype == bool:
        masks = np.expand_dims(labels, axis=0)
    else:
        masks = np.empty((num_classes - 1,) + labels.shape)
        for label_idx in range(1, num_classes):
            masks[label_idx - 1, ...] = labels == label_idx

    result = np.empty_like(masks, dtype=float)
    for i, binary_mask in enumerate(masks):
        if not np.any(binary_mask):
            dis = np.inf * np.ones(binary_mask.shape, dtype=float)
        else:
            edges = get_mask_edges(binary_mask)
            if distance_metric == "euclidean":
                dis = distance_transform_edt(~edges, sampling=spacing)
            elif distance_metric in {"chessboard", "taxicab"}:
                dis = distance_transform_cdt(~edges, metric=distance_metric)
            else:
                raise ValueError(
                    f"distance_metric {distance_metric} is not implemented."
                )
        result[i, ...] = dis
    return result


class DistanceTransform(Transform):
    def __init__(
        self,
        dtype: DtypeLike = np.float32,
        num_classes: int = 2,
        spacing: Optional[Union[float, Sequence[float]]] = None,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.num_classes = num_classes
        self.spacing = spacing

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        img_ = convert_to_numpy(img)
        dist = get_boundary_distance(
            img_, num_classes=self.num_classes, spacing=self.spacing
        )
        ret = convert_to_dst_type(dist, dst=img, dtype=self.dtype or img.dtype)[0]
        return ret


class DistanceTransformd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        output_keys: Union[str, Sequence[str]] = "dist",
        dtype: DtypeLike = np.float32,
        num_classes: int = 2,
        spacing: Optional[Union[float, Sequence[float]]] = None,
    ):
        super().__init__(keys)

        self.output_keys = (
            (output_keys,) if isinstance(output_keys, str) else tuple(output_keys)
        )
        if len(self.keys) != len(self.output_keys):
            raise RuntimeError("Length of `output_keys` must match `keys`")

        self.dt = DistanceTransform(dtype, num_classes, spacing)

    def __call__(
        self, data: Dict[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, output_key in zip(self.keys, self.output_keys):
            d[output_key] = self.dt(d[key])
        return d


def test_DistanceTransform_2d():
    mask = torch.zeros(8, 9, dtype=torch.bool)
    mask[2:5, 3:6] = 1

    distance_transform = DistanceTransform()
    df = distance_transform(mask)
    assert isinstance(df, torch.Tensor)
    assert df.shape == (
        1,
        8,
        9,
    )


def test_DistanceTransform_3d():
    mask = torch.zeros(8, 9, 7, dtype=torch.bool)
    mask[2:5, 3:6, 3:5] = 1

    distance_transform = DistanceTransform()
    df = distance_transform(mask)
    assert isinstance(df, torch.Tensor)
    assert df.shape == (
        1,
        8,
        9,
        7,
    )


def test_DistanceTransformd():
    mask = torch.zeros(8, 9, dtype=torch.bool)
    mask[2:5, 3:6] = 1

    distance_transform = DistanceTransformd(keys="label", output_keys="dist")
    df = distance_transform({"label": mask})
    assert isinstance(df, dict)
    assert "dist" in df
    assert isinstance(df["dist"], torch.Tensor)


def test_DistanceTransform_MultiClass():
    mask = np.zeros((6, 7), dtype=int)
    mask[2:5, 3:6] = 1
    mask[4, 5] = 2

    spacing = (1.2, 1.7)

    distance_transform = DistanceTransform(num_classes=3, spacing=spacing)
    df = distance_transform(mask)
    assert df.shape == (
        2,
        6,
        7,
    )
    ref = distance_transform_edt(~(mask == 2), sampling=spacing)
    np.testing.assert_almost_equal(df[1, ...], ref, decimal=6)


if __name__ == "__main__":
    test_DistanceTransform_MultiClass()

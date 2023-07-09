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
        num_classes: number of classes including background (num_classes == max(labels) + 1)
        spacing: the image spacing
    Note:
        If labels is all 0, may result in inf distance. (TODO: `inf` in boundary loss?)
    """

    masks: np.ndarray
    if labels.dtype in (bool,):
        masks = np.empty((2,) + labels.shape)
        masks[0, ...] = ~labels
        masks[1, ...] = labels
    else:
        masks = np.empty((num_classes,) + labels.shape)
        for label_idx in range(num_classes):
            masks[label_idx, ...] = labels == label_idx

    result = np.empty_like(masks, dtype=float)
    for i, binary_mask in enumerate(masks):
        if not np.any(binary_mask):
            result[i, ...] = np.inf * np.ones(binary_mask.shape, dtype=float)
        else:
            edges = get_mask_edges(binary_mask)
            if distance_metric == "euclidean":
                result[i, ...] = distance_transform_edt(~edges, sampling=spacing)
            elif distance_metric in {"chessboard", "taxicab"}:
                result[i, ...] = distance_transform_cdt(~edges, metric=distance_metric)
            else:
                raise ValueError(
                    f"distance_metric {distance_metric} is not implemented."
                )
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

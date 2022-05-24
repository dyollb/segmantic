import numpy as np
import torch
from monai.config import DtypeLike
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms import Transform
from monai.utils import convert_data_type
from monai.utils.enums import TransformBackends
from scipy.interpolate import interp1d


class Interpolate1d:
    def __init__(self, x: NdarrayOrTensor, y: NdarrayOrTensor) -> None:
        if len(x.shape) != 1 or len(y.shape) != 1:
            raise RuntimeError("x and y must be 1D arrays")
        self.x = x if isinstance(x, torch.Tensor) else torch.tensor(x)
        self.y = y if isinstance(y, torch.Tensor) else torch.tensor(y)
        self.slopes = (self.y[1:] - self.y[:-1]) / (self.x[1:] - self.x[:-1])

    def __call__(self, xnew: torch.Tensor) -> torch.Tensor:
        ind = torch.bucketize(xnew.view(-1), self.x, right=True)
        ind -= 1
        ind = torch.clamp(ind, 0, self.x.shape[0] - 1 - 1)

        def sel(a):
            return a.contiguous().view(-1)[ind]

        return (sel(self.y) + sel(self.slopes) * (xnew.view(-1) - sel(self.x))).view(
            xnew.shape
        )


class NyulNormalize(Transform):

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
    quantiles: NdarrayOrTensor
    standard_scale: NdarrayOrTensor

    def __init__(
        self,
        quantiles: NdarrayOrTensor,
        standard_scale: NdarrayOrTensor,
        nonzero: bool = False,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
    ) -> None:

        if isinstance(quantiles, torch.Tensor):
            self.quantiles, ids = torch.sort(quantiles, stable=True)
            self.standard_scale = standard_scale[ids]
        else:
            indices = np.argsort(quantiles, kind="stable")
            self.quantiles = quantiles[indices]
            self.standard_scale = standard_scale[indices]  # type: ignore [index]
        self.nonzero = nonzero
        self.channel_wise = channel_wise
        self.dtype = dtype

    def _normalize(self, img: NdarrayOrTensor) -> NdarrayOrTensor:

        img, *_ = convert_data_type(img, dtype=torch.float32)

        if self.nonzero:
            mask = img != 0
        else:
            if isinstance(img, np.ndarray):
                mask = np.ones_like(img, dtype=bool)
            else:
                mask = torch.ones_like(img, dtype=torch.bool)
        if not mask.any():
            return img

        quantiles, standard_scale = self.quantiles, self.standard_scale
        if isinstance(img, torch.Tensor):
            if not isinstance(quantiles, torch.Tensor):
                quantiles = torch.tensor(self.quantiles)
            if not isinstance(standard_scale, torch.Tensor):
                standard_scale = torch.tensor(self.standard_scale)

            landmarks = torch.quantile(img[mask], quantiles.to(img.device))
            f = Interpolate1d(landmarks, standard_scale.to(img.device))
            img[mask] = f(img[mask])
        else:
            if isinstance(quantiles, torch.Tensor):
                quantiles = quantiles.numpy()
            if isinstance(standard_scale, torch.Tensor):
                standard_scale = standard_scale.numpy()

            landmarks = np.quantile(img[mask], quantiles)
            f = interp1d(
                landmarks,
                standard_scale,
                assume_sorted=True,
                fill_value="extrapolate",
            )
            img[mask] = f(img[mask])
        return img

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        dtype = self.dtype or img.dtype
        if self.channel_wise:
            for i, d in enumerate(img):
                img[i] = self._normalize(d)  # type: ignore
        else:
            img = self._normalize(img)

        out, *_ = convert_data_type(img, dtype=dtype)
        return out

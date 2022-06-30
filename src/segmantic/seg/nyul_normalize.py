import numpy as np
import torch
from monai.config import NdarrayOrTensor
from monai.transforms import Transform
from monai.utils import convert_to_dst_type
from monai.utils.enums import TransformBackends
from scipy.interpolate import interp1d


class NyulNormalize(Transform):

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
    quantiles: NdarrayOrTensor
    standard_scale: NdarrayOrTensor

    def __init__(
        self,
        quantiles: np.ndarray,
        standard_scale: np.ndarray,
        nonzero: bool = False,
        channel_wise: bool = False,
    ) -> None:

        indices = np.argsort(quantiles, kind="stable")
        self.quantiles = quantiles[indices]
        self.standard_scale = standard_scale[indices]
        self.nonzero = nonzero
        self.channel_wise = channel_wise

    def interp1d(
        self, x: NdarrayOrTensor, xp: NdarrayOrTensor, fp: NdarrayOrTensor
    ) -> NdarrayOrTensor:
        ns = torch if isinstance(x, torch.Tensor) else np
        if isinstance(x, np.ndarray):
            # TODO: check if this is actually faster
            return interp1d(xp, fp, assume_sorted=True, fill_value="extrapolate")(x)  # type: ignore

        m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
        b = fp[:-1] - (m * xp[:-1])

        indices = ns.searchsorted(xp.reshape(-1), x.reshape(-1)) - 1
        indices = ns.clip(indices, 0, len(m) - 1)

        f = (m[indices] * x.reshape(-1) + b[indices]).reshape(x.shape)
        return f

    def _normalize(self, img: NdarrayOrTensor) -> NdarrayOrTensor:

        # TODO: allow user-specified mask `select_fn`
        if self.nonzero:
            mask = img != 0
        else:
            if isinstance(img, np.ndarray):
                mask = np.ones_like(img, dtype=bool)
            else:
                mask = torch.ones_like(img, dtype=torch.bool)

        # mask is empty, skip the rest
        if not mask.any():
            return img

        quantiles, *_ = convert_to_dst_type(self.quantiles, dst=img)
        standard_scale, *_ = convert_to_dst_type(self.standard_scale, dst=img)

        landmarks: NdarrayOrTensor
        if isinstance(img, torch.Tensor):
            assert isinstance(quantiles, torch.Tensor)
            landmarks = torch.quantile(img[mask], quantiles)
        else:
            landmarks = np.quantile(img[mask], quantiles)

        img[mask] = self.interp1d(img[mask], landmarks, standard_scale)  # type: ignore
        return img

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        if self.channel_wise:
            for i, d in enumerate(img):
                img[i] = self._normalize(d)  # type: ignore
        else:
            img = self._normalize(img)
        return img

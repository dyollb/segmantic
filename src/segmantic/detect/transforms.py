import json
import logging
import traceback
from pathlib import Path
from typing import Dict, Hashable, List, Mapping

import numpy as np
import torch
from monai.config import KeysCollection, PathLike
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.folder_layout import FolderLayout
from monai.transforms import (
    GaussianSmooth,
    ScaleIntensity,
    generate_spatial_bounding_box,
)
from monai.transforms.transform import MapTransform
from monai.utils import ImageMetaKey as Key
from monai.utils import convert_to_numpy, convert_to_tensor
from monai.utils.enums import PostFix

__all__ = ["LoadVert", "SaveVert", "EmbedVert", "ExtractVertPosition", "VertHeatMap"]

DEFAULT_POST_FIX = PostFix.meta()


class LoadVert(MapTransform):
    """
    Dictionary-based transform to load landmark positions from
    supported file types (json).
    """

    def __init__(self, keys: KeysCollection, meta_key_postfix: str = DEFAULT_POST_FIX):
        super().__init__(keys, False)

        self.meta_key_postfix = meta_key_postfix

    def __call__(self, data: Mapping[Hashable, PathLike]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            filename = d[key]
            data_k = json.loads(Path(filename).read_text())

            meta_key = f"{key}_{self.meta_key_postfix}"
            try:
                id_map = {n: int(n) for n in data_k}
            except ValueError:
                id_map = {n: id for id, n in enumerate(sorted(data_k), start=1)}

            d[key] = {id_map[n]: np.asarray(data_k[n]) for n in data_k}
            d[meta_key] = {Key.FILENAME_OR_OBJ: filename, "id_map": id_map}
        return d


class SaveVert(MapTransform):
    """
    Dictionary-based transform to load landmark positions from
    supported file types (json).
    """

    def __init__(
        self,
        keys: KeysCollection,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        output_dir: PathLike = "./",
        output_postfix: str = "trans",
        output_ext: str = ".json",
        data_root_dir: PathLike = "",
        separate_folder: bool = True,
        print_log: bool = True,
    ):
        super().__init__(keys, False)

        self.meta_key_postfix = meta_key_postfix
        self.folder_layout = FolderLayout(
            output_dir=output_dir,
            postfix=output_postfix,
            extension=output_ext,
            parent=separate_folder,
            makedirs=True,
            data_root_dir=data_root_dir,
        )
        self.verbose = print_log
        self._data_index = 0

    def __call__(self, data):

        err = []
        d = dict(data)
        for key in self.key_iterator(d):
            meta_key = f"{key}_{self.meta_key_postfix}"
            meta_data = d[meta_key] if meta_key in d else {}
            subject = (
                meta_data[Key.FILENAME_OR_OBJ] if meta_data else str(self._data_index)
            )
            self._data_index += 1
            filename = self.folder_layout.filename(subject=f"{subject}")

            verts = d[key]
            id_map = meta_data.get("id_map", {str(id): id for id in verts})
            name_map = {v: k for k, v in id_map.items()}
            out = {name_map[k]: list(v) for k, v in verts.items()}
            if self.verbose:
                logging.getLogger("SaveVert").info(f"wrote {filename}")
            try:
                Path(filename).write_text(json.dumps(out))
            except Exception as e:
                err.append(traceback.format_exc())
                logging.getLogger(self.__class__.__name__).debug(e, exc_info=True)
                logging.getLogger(self.__class__.__name__).info(
                    f"{self.__class__.__name__}: unable to write {filename}.\n"
                )

        if err:
            msg = "\n".join([f"{e}" for e in err])
            raise RuntimeError(
                f"{self.__class__.__name__} cannot write vertices:\n{msg}"
            )


class EmbedVert(MapTransform):
    """
    Dictionary-based transform to write landmark position
    in an image.
    """

    def __init__(
        self,
        keys: KeysCollection,
        ref_key: str,
        meta_key_postfix: str = DEFAULT_POST_FIX,
    ):
        super().__init__(keys, False)

        self.ref_key = ref_key
        self.meta_key_postfix = meta_key_postfix

    def __call__(self, data):

        d = dict(data)
        ref_meta_key = f"{self.ref_key}_{self.meta_key_postfix}"
        if "affine" in d[ref_meta_key]:
            affine = convert_to_numpy(d[ref_meta_key]["affine"])
        else:
            affine = np.eye(4, 4)
        rot_inv = np.linalg.inv(affine[:3, :3])
        t = affine[:3, 3]

        ref_image = d[self.ref_key]
        for k in self.keys:
            vertices = d[k]
            out = np.zeros_like(ref_image)
            for label in vertices:
                p = vertices[label]
                continuous_idx = np.dot(rot_inv, p - t)
                idx = np.round(continuous_idx).astype(int)
                out[idx[0], idx[1], idx[2]] = label
            d[k] = out

            meta_key = f"{k}_{self.meta_key_postfix}"
            d[meta_key].update({"affine": affine, "original_channel_dim": "no_channel"})

        return d


class ExtractVertPosition(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        threshold: float = 0.5,
        meta_key_postfix: str = DEFAULT_POST_FIX,
    ):
        """
        Postprocess Vertebra localization
        :param keys: The ``keys`` parameter will be used to get and set the actual data item to transform
        """
        super().__init__(keys, allow_missing_keys)

        self.threshold = threshold
        self.meta_key_postfix = meta_key_postfix

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> Dict[Hashable, np.ndarray]:

        d: Dict = dict(data)
        for key in self.key_iterator(d):
            # locate vertices
            vertices = {}
            for id in range(1, d[key].shape[0]):
                if d[key][id, ...].max() < self.threshold:
                    continue
                X, Y, Z = np.where(d[key][id, ...] == d[key][id, ...].max())
                p = np.asarray([X[0], Y[0], Z[0]], dtype=float)
                vertices[id] = p

            # transform to physical coordinates
            meta_key = f"{key}_{self.meta_key_postfix}"
            meta = d.get(meta_key, {})
            if "affine" in meta:
                affine = meta["affine"]
                rot = affine[:3, :3]
                t = affine[:3, 3]
                for id in vertices:
                    p = vertices[id]
                    vertices[id] = np.dot(rot, p) + t
            d[key] = vertices

        return d


class BoundingBoxd(MapTransform):
    def __init__(
        self, keys: KeysCollection, result: str = "result", bbox: str = "bbox"
    ):
        super().__init__(keys)
        self.result = result
        self.bbox = bbox

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            bbox = generate_spatial_bounding_box(d[key])
            if d.get(self.result) is None:
                d[self.result] = dict()
            d[self.result][self.bbox] = np.array(bbox).astype(int).tolist()
        return d


class VertHeatMap(MapTransform):
    def __init__(
        self, keys: KeysCollection, gamma: float = 1000.0, label_names: List[str] = []
    ):
        super().__init__(keys)
        self.label_names = label_names
        self.gamma = gamma

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> Dict[Hashable, NdarrayOrTensor]:

        d: Dict[Hashable, NdarrayOrTensor] = dict(data)
        for k in self.keys:
            i = convert_to_tensor(d[k], dtype=torch.long)
            # one hot if necessary
            is_onehot = i.shape[0] > 1
            if is_onehot:
                out = torch.zeros_like(i)
            else:
                # +1 for background
                out = torch.nn.functional.one_hot(i, len(self.label_names) + 1)
                out = torch.movedim(out[0], -1, 0)
                out.fill_(0.0)
                out = out.float()

            # loop over all segmentation classes
            for seg_class in torch.unique(i):
                # skip background
                if seg_class == 0:
                    continue
                # get CoM for given segmentation class
                centre = [
                    np.average(indices.cpu()).astype(int)
                    for indices in torch.where(i[0] == seg_class)
                ]
                label_num = seg_class.item()
                centre.insert(0, label_num)
                out[tuple(centre)] = 1.0
                sigma = 1.6 + (label_num - 1.0) * 0.1
                # Gaussian smooth
                out[label_num] = GaussianSmooth(sigma)(out[label_num].cuda()).cpu()
                # Normalize to [0,1]
                out[label_num] = ScaleIntensity()(out[label_num])  # type: ignore
                out[label_num] = out[label_num] * self.gamma

            d[k] = out

        return d

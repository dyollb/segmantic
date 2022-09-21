import sys
from typing import Any, Dict, Hashable, Mapping, Tuple

import h5py
import numpy as np
from monai.config import KeysCollection, PathLike
from monai.data.folder_layout import FolderLayout
from monai.transforms.transform import MapTransform
from monai.utils import ImageMetaKey as Key
from monai.utils import convert_to_numpy
from monai.utils.enums import PostFix

DEFAULT_POST_FIX = PostFix.meta()

LabelInfo = Tuple[str, float, float, float]


def voxel_sizes(affine: np.ndarray) -> np.ndarray:
    """
    Get voxel sizes in mm from affine
    Copied from nibable.affines.voxel_sizes
    """
    top_left = affine[:-1, :-1]
    return np.sqrt(np.sum(top_left**2, axis=0))  # type: ignore [no-any-return]


def export_to_iseg(
    iseg_file_path,
    label_field: np.ndarray,
    image: np.ndarray,
    affine: np.ndarray,
    labels: Dict[int, LabelInfo],
) -> None:
    with h5py.File(iseg_file_path, "w") as f:
        f.create_dataset(
            "Tissue",
            dtype=np.uint16,
            data=label_field.flatten(),
            compression="gzip",
            compression_opts=1,
        )
        f.create_dataset(
            "Source",
            dtype=float,
            data=image.flatten(),
            compression="gzip",
            compression_opts=1,
        )
        f.create_dataset(
            "Target",
            dtype=float,
            data=np.zeros(image.flatten().shape),
            compression="gzip",
            compression_opts=1,
        )

        rot = affine[:-1, :-1]
        dims = image.shape
        origin = affine[:-1, -1]
        spacing = voxel_sizes(affine)

        f.create_dataset("rotation", dtype=float, data=rot.flatten())
        f.create_dataset("dimensions", dtype=float, data=dims)
        f.create_dataset("offset", dtype=float, data=origin)
        f.create_dataset("pixelsize", dtype=float, data=spacing)

        tissues = f.create_group("Tissues")
        for idx, info in labels.items():
            try:
                name, r, g, b = info
                rgbo = np.array([r, g, b, 0.5])
                T = tissues.create_group(name)
                T.create_dataset("index", dtype=np.int32, data=np.array([idx]))
                T.create_dataset("rgbo", dtype=float, data=rgbo)
            except Exception:
                print(f"Problems writing: {name}", file=sys.stderr)
        tissues.create_dataset("bkg_rgbo", dtype=float, data=np.array([0, 0, 0, 0.5]))
        tissues.create_dataset("version", dtype=np.int32, data=np.array([0]))


class iSegSaver(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        label_dict: Dict[int, LabelInfo],
        image_key: str = "image",
        label_key: str = "label",
        allow_missing_keys: bool = False,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        output_dir: PathLike = "./",
        output_postfix: str = "trans",
        output_ext: str = ".h5",
        data_root_dir: PathLike = "",
        separate_folder: bool = True,
        print_log: bool = True,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

        self.label_dict = label_dict
        self.image_key = image_key
        self.label_key = label_key
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

    def __call__(self, data: Mapping[Hashable, Any]) -> None:
        d = dict(data)
        if not self.allow_missing_keys and any(k not in d for k in self.keys):
            raise RuntimeError(f"{self.__class__.__name__}: missing keys in data")

        if not any(k in d for k in (self.image_key, self.label_key)):
            raise RuntimeError(
                f"{self.__class__.__name__}: neither {self.image_key} nor {self.label_key} found in data"
            )

        image_key = self.image_key if self.image_key in d else self.label_key
        label_key = self.label_key if self.label_key in d else self.image_key

        image = d[image_key]
        image = np.squeeze(convert_to_numpy(image))
        label = d[label_key]
        label = np.squeeze(convert_to_numpy(label))

        if image.shape != label.shape:
            raise RuntimeError(
                f"{self.__class__.__name__}: image and label have different number of elements or shape"
            )

        meta_key = f"{image_key}_{self.meta_key_postfix}"
        if "affine" in d[meta_key]:
            affine = convert_to_numpy(d[meta_key]["affine"])
        else:
            affine = np.eye(4, 4)

        meta_data = d[meta_key] if meta_key in d else {}
        subject = meta_data[Key.FILENAME_OR_OBJ] if meta_data else str(self._data_index)
        self._data_index += 1
        filename = self.folder_layout.filename(subject=f"{subject}")
        export_to_iseg(
            filename,
            label_field=label,
            image=image,
            affine=affine,
            labels=self.label_dict,
        )

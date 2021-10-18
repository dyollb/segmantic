from numpy.lib.shape_base import tile
import torch
import torchvision.transforms as transforms
import numpy as np
import itk
from typing import Any, List, Tuple, Sequence, Union
from pathlib import Path

from ..prepro.core import crop, make_image, pixeltype, Image3, Image2

from .pix2pix_cyclegan.models.networks import define_G

Pix2PixGenerator = Any
CycleGanGenerator = Any


def load_pix2pix_generator(
    model_file_path: Path, device: torch.device, eval: bool = False
) -> Pix2PixGenerator:
    """Load a trained pix2pix model

    Args:
        model_file_path: Trained pix2pix model file (.pth)
        device: For selecting GPU index or CPU
        eval: Run in eval mode. Defaults to False.

    Returns:
        Returns the generator
    """
    gen = define_G(
        input_nc=1,
        output_nc=1,
        ngf=64,
        netG="unet_256",
        norm="batch",
        use_dropout=True,
        init_type="normal",
        init_gain=0.02,
        gpu_ids=[device.index] if device.index else [],
    )

    if isinstance(gen, torch.nn.DataParallel):
        gen = gen.module

    print(f"loading the model from {model_file_path}")

    state_dict = torch.load(model_file_path, map_location=device)
    if hasattr(state_dict, "_metadata"):
        del state_dict._metadata
    gen.load_state_dict(state_dict)

    # dropout and batchnorm has different behavioir during training and test.
    if eval:
        gen.eval()
    return gen


def load_cyclegan_generator(
    model_file_path: Path, device: torch.device
) -> CycleGanGenerator:
    """Load a trained cyclegan model

    Args:
        model_file_path: Trained cyclegan model file (.pth)
        device: For selecting GPU index or CPU

    Returns:
        Returns the generator
    """
    gen = define_G(
        input_nc=1,
        output_nc=1,
        ngf=64,
        netG="resnet_9blocks",
        norm="instance",
        use_dropout=False,
        init_type="normal",
        init_gain=0.02,
        gpu_ids=[device.index] if device.index else [],
    )

    if isinstance(gen, torch.nn.DataParallel):
        gen = gen.module

    print(f"loading the model from {model_file_path}")

    state_dict = torch.load(model_file_path, map_location=device)
    if hasattr(state_dict, "_metadata"):
        del state_dict._metadata
    gen.load_state_dict(state_dict)
    return gen


def make_tiles(
    size: Sequence[int],
    tile_size: Tuple[int, int],
    overlap: int = 0,
    add_center_tile: bool = False,
) -> List[Tuple[int, int]]:
    """Break rectangular region into overlapping tiles of fixed size

    Args:
        size: shape of 2D image
        tile_size: shape of 2D tiles
        overlap: ensure overlap
        add_center_tile: append centered tile

    Returns:
        list of start indices of tiles
    """
    if size[0] < tile_size[0] and size[1] < tile_size[1]:
        return [(0, 0)]

    # ensure we don't go over end
    fix_tile = lambda start: (
        max(min(start[0], size[0] - tile_size[0]), 0),
        max(min(start[1], size[1] - tile_size[1]), 0),
    )

    tile_indices = []
    start = [0, 0]
    while start[0] < size[0]:
        start[1] = 0
        while start[1] < size[1]:
            tile_indices.append(fix_tile(start))
            start[1] += tile_size[1]
            if start[1] + 1 < size[1]:
                start[1] -= overlap
        start[0] += tile_size[0]
        if start[0] + 1 < size[0]:
            start[0] -= overlap

    if add_center_tile:
        delta = [max(s, t) - t for s, t in zip(size, tile_size)]
        if any(delta):
            crop_low = [(d + 1) // 2 for d in delta]
            tile_indices.append((crop_low[0], crop_low[1]))

    return tile_indices


def tile_image(
    image: Image2, tile_indices: List[Tuple[int, int]], tile_size: Tuple[int, int]
) -> List[Image2]:
    """Break image into tiles

    Args:
        image: input image
        tile_indices: start index (corner) of tiles
        tile_size: fixed size of tiles

    Returns:
        image tiles
    """
    return [
        crop(image, target_offset=start, target_size=tile_size)
        for start in tile_indices
    ]


def merge_tiles(
    tiles: List[Image2], tile_indices: List[Tuple[int, int]], tile_size: Tuple[int, int]
) -> Image2:
    """Combine tiles into a single image

    Args:
        tiles: list of image patches
        tile_indices: start index (corner) of tiles
        tile_size: fixed size of tiles

    Returns:
        merged image
    """
    size = [0, 0]
    for t in tile_indices:
        size[0] = max(size[0], t[0] + tile_size[0])
        size[1] = max(size[1], t[1] + tile_size[1])

    example = tiles[0]
    image = make_image(
        shape=size, spacing=itk.spacing(example), pixel_type=pixeltype(example)
    )
    image_view = itk.array_view_from_image(image)
    for t, start in zip(tiles, tile_indices):
        image_view[
            start[1] : start[1] + tile_size[1], start[0] : start[0] + tile_size[0]
        ] = itk.array_view_from_image(t)
    return image


def translate(
    img: Union[np.ndarray, List[np.ndarray]],
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    """translate 2D image(s) using the given model

    Args:
        img: 2D input image / list of images
        model: generator
        device: torch.device

    Returns:
        translated image, batches are concatenated along first dimension
    """
    from PIL import Image

    to_pil = lambda x: Image.fromarray(x).convert("RGB")

    tr = transforms.Compose(
        [
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    with torch.no_grad():
        if isinstance(img, list):
            batch_size = len(img)
            fake = (
                model(
                    torch.cat([tr(to_pil(i)) for i in img]).view(
                        batch_size, 1, 256, 256
                    )
                )
                .detach()
                .cpu()
                .numpy()
            ).reshape([batch_size, 256, 256])
        else:
            fake = (
                model(tr(to_pil(img)).to(device).view(1, 1, 256, 256))
                .detach()
                .cpu()
                .numpy()
            ).reshape([256, 256])
    return fake


def translate_3d(
    image: Image3, model: torch.nn.Module, axis: int, device: torch.device
) -> Image3:
    """Split 3D image along specified axis and do style transfer on each slice

    Args:
        image: 3D input image
        model: generator
        axis: axis along which the 3D image is split
        device: torch.device

    Returns:
        translated image
    """
    output = itk.image_duplicator(image)

    # split into slice views of image
    axis = 2 - axis  # invert axis
    arr = itk.array_view_from_image(output)
    slices = []
    for k in range(arr.shape[axis]):
        if axis == 0:
            slices.append(itk.image_view_from_array(arr[k, :, :]))
        elif axis == 1:
            slices.append(itk.image_view_from_array(arr[:, k, :]))
        else:
            slices.append(itk.image_view_from_array(arr[:, :, k]))

    # tiling only needs to be computed once
    tile_size = (256, 256)
    tile_indices = make_tiles(
        size=itk.size(slices[0]), tile_size=tile_size, overlap=0, add_center_tile=True
    )

    # translate slices
    for slice_views in slices:
        tiles = tile_image(
            image=slice_views, tile_indices=tile_indices, tile_size=(256, 256)
        )
        batch = translate([itk.array_view_from_image(t) for t in tiles], model, device)
        batch_size = len(tiles)
        # overwrite slice view
        slice_views[:] = merge_tiles(
            tiles=[
                itk.image_view_from_array(batch[i, :, :]) for i in range(batch_size)
            ],
            tile_indices=tile_indices,
            tile_size=tile_size,
        )

    return output

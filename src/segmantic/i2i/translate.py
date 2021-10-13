import torch
import torchvision.transforms as transforms
import numpy as np
import itk
from typing import Any, List, Tuple, Sequence, overload
from pathlib import Path

from ..prepro.core import crop, make_image, Image3, Image2

from .pix2pix_cyclegan.models.networks import define_G


def load_pix2pix_generator(
    model_file_path: Path, device: torch.device, eval: bool = False
) -> Any:
    """Load a trained pix2pix model

    Args:
        model_file_path (Path): Trained pix2pix model file (.pth)
        device (torch.device): For selecting GPU index or CPU
        eval (bool, optional): Run in eval mode. Defaults to False.

    Returns:
        Tuple[Any, torch.device]: Returns the generator and torch device
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


def load_cyclegan_generator(model_file_path: Path, device: torch.device):
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

    state_dict = torch.load(model_file_path, map_location=device)
    if hasattr(state_dict, "_metadata"):
        del state_dict._metadata
    gen.load_state_dict(state_dict)
    return gen


def make_tiles(
    size: Sequence[int], tile_size: Tuple[int, int], overlap: int = 0
) -> List[Tuple[int, int]]:
    """Break rectangular region into overlapping tiles of fixed size

    Args:
        size: shape of 2D image
        tile_size: shape of 2D tiles
        overlap: ensure overlap

    Returns:
        List[Tuple[int, int]]: list of start indices of tiles
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
        List[Image2]: image tiles
    """
    tiles = []
    for start in tile_indices:
        tiles.append(crop(image, target_offset=start, target_size=tile_size))
    return tiles


def merge_tiles(
    tiles: List[Image2], tile_indices: List[Tuple[int, int]], tile_size: Tuple[int, int]
) -> Image2:
    """Combine tiles into a single image

    Args:
        tiles: list of image patches
        tile_indices: start index (corner) of tiles
        tile_size: fixed size of tiles

    Returns:
        [Image2]: merged image
    """
    size = [0, 0]
    for t in tile_indices:
        size[0] = max(size[0], t[0] + tile_size[0])
        size[1] = max(size[1], t[1] + tile_size[1])

    example = tiles[0]
    image = make_image(
        shape=size, spacing=itk.spacing(example), pixel_type=itk.template(example)[1][0]
    )
    image_view = itk.array_view_from_image(image)
    for t, start in zip(tiles, tile_indices):
        image_view[
            start[1] : start[1] + tile_size[1], start[0] : start[0] + tile_size[0]
        ] = itk.array_view_from_image(t)
    return image


def translate(img: np.ndarray, model: torch.nn.Module, device: torch.device):
    from PIL import Image

    img_p = Image.fromarray(img).convert("RGB")
    tr = transforms.Compose(
        [
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    with torch.no_grad():
        fake = (
            model(tr(img_p).to(device).view(1, 1, 256, 256)).detach().cpu().numpy()
        ).reshape(img.shape)
    return fake


def translate_3d(
    image: Image3, model: torch.nn.Module, axis: int, device: torch.device
):
    """Split 3D image along specified axis and do style transfer on each slice"""
    arr = itk.array_from_image(image)
    axis = 2 - axis
    for k in range(arr.shape[axis]):
        if axis == 0:
            arr[k, :, :] = translate(arr[k, :, :], model, device)
        elif axis == 1:
            arr[:, k, :] = translate(arr[:, k, :], model, device)
        else:
            arr[:, :, k] = translate(arr[:, :, k], model, device)
    return itk.image_view_from_array(arr)

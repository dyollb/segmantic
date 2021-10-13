import torch
import torchvision.transforms as transforms
import numpy as np
import itk
from typing import Any, List, Tuple, Sequence
from pathlib import Path

from ..prepro.core import crop, scale_to_range, Image3, Image2

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
    size: Sequence[int], tile_size: Tuple[int, int], overlap: int = 2
) -> List[Tuple[int, int]]:
    """Break image into tiles of fixed size

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

    tiles = []
    start = [0, 0]
    while start[0] < size[0]:
        start[1] = 0
        while start[1] < size[1]:
            tiles.append(fix_tile(start))
            start[1] += tile_size[1] - overlap
        start[0] += tile_size[0] - overlap
    return tiles


def tile_image(
    image: Image2, tiles: List[Tuple[int, int]], tile_size: Tuple[int, int]
) -> List[Image2]:
    image_view = itk.array_view_from_image(image)
    patches = []
    for start in tiles:
        patch = image_view[
            start[1] : start[1] + tile_size[1], start[0] : start[0] + tile_size[0]
        ]
        patch = itk.image_from_array(patch)
        patch["spacing"] = image["spacing"]
        # patch['origin'] = image['origin'] + image['spacing']
        patches.append(crop(image, target_size=tile_size))
    return patches


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

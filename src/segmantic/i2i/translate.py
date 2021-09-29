import torch
import torchvision.transforms as transforms
import numpy as np
import itk
from typing import Any, List, Tuple
from pathlib import Path

from ..prepro.core import crop, scale_to_range, Image3

from .pix2pix_cyclegan.models.networks import define_G


def load_pix2pix_generator(
    model_file_path: Path, gpu_ids: List[int] = [], eval: bool = False
) -> Tuple[Any, torch.device]:
    """Load a trained pix2pix model

    Args:
        model_file_path (Path): Trained pix2pix model file (.pth)
        gpu_ids (List[int], optional): For selecting the GPU. Defaults to [].
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
        gpu_ids=gpu_ids,
    )

    # get device name: CPU or GPU
    device = (
        torch.device("cuda:{}".format(gpu_ids[0])) if gpu_ids else torch.device("cpu")
    )

    if isinstance(gen, torch.nn.DataParallel):
        gen = gen.module
    print("loading the model from %s" % model_file_path)

    # if you are using PyTorch newer than 0.4, you can remove str() on self.device
    state_dict = torch.load(model_file_path, map_location=str(device))
    if hasattr(state_dict, "_metadata"):
        del state_dict._metadata
    gen.load_state_dict(state_dict)

    # dropout and batchnorm has different behavioir during training and test.
    if eval:
        gen.eval()
    return gen, device


def load_cyclegan_generator(model_file_path: Path, gpu_ids: List[int] = []):
    gen = define_G(
        input_nc=1,
        output_nc=1,
        ngf=64,
        netG="resnet_9blocks",
        norm="instance",
        use_dropout=False,
        init_type="normal",
        init_gain=0.02,
        gpu_ids=gpu_ids,
    )

    # get device name: CPU or GPU
    device = (
        torch.device("cuda:{}".format(gpu_ids[0])) if gpu_ids else torch.device("cpu")
    )

    # if you are using PyTorch newer than 0.4, you can remove str() on self.device
    state_dict = torch.load(model_file_path, map_location=str(device))
    if hasattr(state_dict, "_metadata"):
        del state_dict._metadata
    gen.load_state_dict(state_dict)
    return gen


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

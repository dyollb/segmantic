import torch
import numpy as np
from pix2pix_cyclegan.models.networks import define_G
import context
from prepro.core import image_2d
from typing import Union

import itk
image_3d = Union[itk.Image[itk.F,3], itk.Image[itk.D,3]]


def load_pix2pix_generator(model_file_path: str, gpu_ids: list = []):
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

    # if you are using PyTorch newer than 0.4, you can remove str() on self.device
    state_dict = torch.load(model_file_path, map_location=str(device))
    if hasattr(state_dict, "_metadata"):
        del state_dict._metadata
    gen.load_state_dict(state_dict)
    return gen


def load_cyclegan_generator(model_file_path: str, gpu_ids: list = []):
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


def translate(img: torch.Tensor, model: torch.nn.Module):
    return model(img)


def translate(img: np.ndarray, model: torch.nn.Module):
    return model(torch.from_numpy(img)).detach().cpu().numpy()


def translate_3d(image: image_3d, model: torch.nn.Module, axis: int):
    translate_slice = lambda x, m: translate(x.reshape((1, 1, x.shape[0], x.shape[1],)), m)
    img = itk.array_from_image(image)
    for k in range(img.shape[axis]):
        if axis == 0:
            img[k, :, :] = translate_slice(img[k, :, :], model)
        elif axis == 1:
            img[:, k, :] = translate_slice(img[:, k, :], model)
        else:
            img[:, :, k] = translate_slice(img[:, :, k], model)
    return image


if __name__ == "__main__":
    import itk

    netg = load_pix2pix_generator(r"E:\Develop\Scripts\ML-SEG\cyclegan\checkpoints\t1w2ctm\latest_net_G.pth")

    img_t1 = itk.imread(r"F:\Data\DRCMR-Thielscher\all_data\images\X10679.nii.gz")

    img_ct = translate_3d(img_t1, model=netg, axis=2)

    #itk.imwrite(img_ct, r"F:\temp\X10679_fake_ct.nii.gz")
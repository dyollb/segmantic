import torch
import numpy as np
import itk
from pix2pix_cyclegan.models.networks import define_G
from ..prepro.core import crop, scale_to_range, Image3


def load_pix2pix_generator(
    model_file_path: str, gpu_ids: list = [], eval: bool = False
):
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
    print('loading the model from %s' % model_file_path)

    # if you are using PyTorch newer than 0.4, you can remove str() on self.device
    state_dict = torch.load(model_file_path, map_location=str(device))
    if hasattr(state_dict, "_metadata"):
        del state_dict._metadata
    gen.load_state_dict(state_dict)

    # dropout and batchnorm has different behavioir during training and test.
    if eval:
        gen.eval()
    return gen, device


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


def translate(img: np.ndarray, model: torch.nn.Module, device: torch.device):
    with torch.no_grad():
        fake = model(torch.from_numpy(img.reshape((1, 1) + img.shape)).to(device)).detach().cpu().numpy()
    return fake.reshape(img.shape)


def preprocess_mri(x):
    x_view = itk.array_view_from_image(x)
    x_view *= 255.0 / 280.0
    np.clip(x_view, a_min=0, a_max=255, out=x_view)
    return x


def translate_3d(image: Image3, model: torch.nn.Module, axis: int, device: torch.device):
    imaget = itk.image_duplicator(image)
    arr = itk.array_view_from_image(imaget, keep_axes=True)
    for k in range(arr.shape[axis]):
        if axis == 0:
            arr[k, :, :] = translate(arr[k, :, :], model, device)
        elif axis == 1:
            arr[:, k, :] = translate(arr[:, k, :], model, device)
        else:
            arr[:, :, k] = translate(arr[:, :, k], model, device)
    return imaget


if __name__ == "__main__":
    import itk

    netg, device = load_pix2pix_generator(
        model_file_path=r"E:\Develop\Scripts\ML-SEG\cyclegan\checkpoints\t1w2ctm\latest_net_G.pth",
        #gpu_ids=[0]
    )

    img_t1 = itk.imread(r"F:\Data\DRCMR-Thielscher\all_data\images\X10679.nii.gz")

    img_t1 = crop(preprocess_mri(img_t1), target_size=(256, 256, 10))

    img_ct = translate_3d(img_t1, model=netg, axis=2, device=device)

    itk.imwrite(img_t1, r"F:\temp\X10679_real_t1.nii.gz")
    itk.imwrite(img_ct, r"F:\temp\X10679_fake_ct.nii.gz")

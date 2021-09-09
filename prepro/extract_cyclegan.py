import os
import numpy as np
import itk
from random import randint
import math
from typing import List, Optional, Union


image_3d = itk.itkImagePython.itkImageBase3
image_2d = itk.itkImagePython.itkImageBase2


def as_image(x: Union[image_3d, image_2d, np.ndarray]):
    if isinstance(x, np.ndarray):
        return itk.image_view_from_array(x)
    return x


def identity(x):
    return x


def bbox(img):
    """Get foreground (non-zero) bounding box from 2D image"""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def extract_slices(img: image_3d, axis: int= 2):
    ''' Get 2D image slices from 3D image '''
    slices = []
    for k in range(img.shape[axis]):
        if axis == 0:
            slices.append(img[k, :, :])
        elif axis == 1:
            slices.append(img[:, k, :])
        else:
            slices.append(img[:, :, k])
    return slices


def export_slices(
    image_files: List[str],
    output_dir: str,
    axis: int = 0,
    flip_lr: bool = False,
    process_img=identity,
):
    """Load list of 3D images and extract & export slices"""
    # create folders for output
    os.makedirs(output_dir, exist_ok=True)

    # loop over 3d images
    for file_path in image_files:
        img = itk.imread(file_path)

        slices = extract_slices(img=process_img(img), axis=axis)
        f = os.path.basename(file_path)

        # for each slice
        for k, slice in enumerate(slices):

            # skip if 'empty'
            if np.percentile(slice, 98) < 10:
                continue

            # pad to have minimum width 256, or 300, ...
            pad_size = 300
            if slice.shape[0] < pad_size or slice.shape[1] < pad_size:
                sz = slice.shape
                pad_x = (pad_size + 1 - min(sz[0], pad_size)) // 2
                pad_y = (pad_size + 1 - min(sz[1], pad_size)) // 2
                slice = np.pad(
                    slice,
                    ((pad_x, pad_x), (pad_y, pad_y)),
                    mode="constant",
                    constant_values=0,
                )

            # random crop
            if slice.shape[0] > 256 or slice.shape[1] > 256:

                rmin, rmax, cmin, cmax = bbox(slice > 4)
                s1 = randint(0, min(rmin, max(0, slice.shape[0] - 256)))
                s2 = randint(0, min(cmin, max(0, slice.shape[1] - 256)))
                slice = slice[s1 : s1 + 256, s2 : s2 + 256]

            # flip?
            if flip_lr:
                slice = np.rot90(slice, k=1)

            itk.imwrite(
                itk.image_from_array(slice).astype(itk.SS),
                os.path.join(output_dir, f.replace(".nii.gz", "_%03d.tif" % k)),
                compression=True,
            )


def scale_to_uchar(img: Union[image_2d, image_3d]):
    """scale numpy itk.Image to fit in range [0,255]"""
    x_view = itk.array_view_from_image(img)
    x_min, x_max = np.min(x_view), np.max(x_view)
    x_view -= x_min
    np.multiply(x_view, 255.0 / (x_max - x_min), out=x_view, casting="unsafe")
    np.clip(x_view, a_min=0, a_max=255, out=x_view)
    return img


def resample(img: Union[image_2d, image_3d], target_spacing=Optional[tuple]):
    """resample N-D itk.Image to a fixed spacing (default:0.85)"""
    dim = img.GetImageDimension()
    interpolator = itk.LinearInterpolateImageFunction.New(img)
    transform = itk.IdentityTransform[itk.D, dim].New()

    if not target_spacing:
        target_spacing = [0.85] * dim

    size = itk.size(img)
    spacing = itk.spacing(img)
    for d in range(dim):
        size[d] = math.ceil(size[d] * spacing[d] / 0.85)
        spacing[d] = target_spacing[d]

    # resample to target resolution
    resampled = itk.resample_image_filter(
        img,
        transform=transform,
        interpolator=interpolator,
        size=size,
        output_spacing=spacing,
        output_origin=itk.origin(img),
        output_direction=img.GetDirection(),
    )
    return resampled


def pad_slice(img: image_2d, target_size: tuple = (256, 256)):
    size = itk.size(img)
    delta = (t - min(s, t) for s, t in zip(size, target_size))

    if any(delta):
        pad_lo = ((d + 1) // 2 for d in delta)
        pad_hi = (delta[i] - p for i, p in enumerate(pad_lo))
        img = itk.constant_pad_image_filter(
            img,
            pad_lower_bound=pad_lo,
            pad_upper_bound=pad_hi,
            constant=0,
        )
    return img


def preprocess_ct(img: Union[image_2d, image_3d]):
    # median filter for salt and pepper noise
    img = itk.median_image_filter(img, radius=1)
    # range clamped to [-1100, 3100] and scaled to [0, 255]
    img_view = itk.array_view_from_image(img)
    img_view += 1100
    img_view *= 255.0 / (1100.0 + 3100.0)
    np.clip(img_view, a_min=0, a_max=300, out=img_view)
    return img


def get_files(dir: str, cond=lambda x: True, ext: str = ".nii.gz"):
    return [
        os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(ext) and cond(f)
    ]


def convert_to_rgb(files: List[str]):
    from PIL import Image

    for f in files:
        im = Image.open(f)
        imgc = im.convert("RGB")
        imgc.save(f)
        im.close()
        imgc.close()


def randomize_files(dir: str, ext: str = ".tif"):
    import shutil
    from random import sample

    random_sample = lambda x: sample(x, len(x))

    files = random_sample([f for f in os.listdir(dir) if f.endswith(ext)])
    for i, f in enumerate(files):
        shutil.move(os.path.join(dir, f), os.path.join(dir, "im_%05d.tif" % i))


if __name__ == "__main__":

    t1_images = get_files(r"F:\Data\DRCMR-Thielscher\all_data\images")
    ixi_t1_images = get_files(
        r"C:\Users\lloyd\Downloads\IXI-T1", cond=lambda x: "Guys" in x
    )

    # export_slices(
    #    image_files=t1_images[:-6],
    #    output_dir=r"F:\temp\cyclegan\t1_drcmr2ixi\trainA",
    #    process_img=scale_to_uchar,
    # )
    # export_slices(
    #    image_files=t1_images[-6:],
    #    output_dir=r"F:\temp\cyclegan\t1_drcmr2ixi\testA",
    #    process_img=scale_to_uchar,
    # )

    # export_slices(
    #    image_files=ixi_t1_images[:15],
    #    output_dir=r"F:\temp\cyclegan\t1_drcmr2ixi\trainB",
    #    process_img=lambda x: resample(scale_to_uchar(x)),
    #    axis=1,
    #    flip_lr=True,
    # )
    # convert_to_rgb(get_files(r"F:\temp\cyclegan\t1_drcmr2ixi\trainB", ext=".tif"))

    export_slices(
        image_files=ixi_t1_images[15:21],
        output_dir=r"F:\temp\cyclegan\t1_drcmr2ixi\testB",
        process_img=lambda x: resample(scale_to_uchar(x)),
        axis=1,
        flip_lr=True,
    )
    convert_to_rgb(get_files(r"F:\temp\cyclegan\t1_drcmr2ixi\testB", ext=".tif"))
    randomize_files(r"F:\temp\cyclegan\t1_drcmr2ixi\testB")

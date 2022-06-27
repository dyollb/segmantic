import itk
import numpy as np

from .core import ImageNd, NdarrayImage, as_array

# def bias_correct(img):
#    pass


def scale_clamp_ct(img: ImageNd) -> ImageNd:
    """Prepare CT images: median -> clamp to [-1100,3100] -> scale to [0,255]"""
    # median filter for salt and pepper noise
    img = itk.median_image_filter(img, radius=1)
    # range clamped to [-1100, 3100]
    img_view = itk.array_view_from_image(img)
    np.clip(img_view, a_min=-1100, a_max=3100, out=img_view)
    # and scaled to [0, 255]
    img_view += 1100
    np.multiply(img_view, 255.0 / (1100.0 + 3100.0), out=img_view, casting="unsafe")
    return img


def unscale_ct(img: NdarrayImage) -> NdarrayImage:
    """Invert 'scale_clamp_ct' operation, except for clamping"""
    img_view = as_array(img)
    np.multiply(img_view, (1100.0 + 3100.0) / 255.0, out=img_view, casting="unsafe")
    img_view -= 1100
    return img

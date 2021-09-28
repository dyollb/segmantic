import numpy as np
import itk
import argparse

from segmantic.prepro.core import crop
from segmantic.i2i.translate import load_pix2pix_generator, translate_3d


def preprocess_mri(x):
    x_view = itk.array_view_from_image(x)
    x_mean, x_std = np.mean(x_view), np.std(x_view)
    x_view -= x_mean
    x_view *= 0.5 / x_std
    x_view += 0.5
    # np.clip(x_view, a_min=0, a_max=1, out=x_view)
    return x


def main():
    parser = argparse.ArgumentParser(description="Translate image.")
    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        type=str,
        help="generator pth file",
        default=r"E:\Develop\Scripts\ML-SEG\cyclegan\checkpoints\t1w2ctm\latest_net_G.pth",
    )
    parser.add_argument(
        "-i", "--input", dest="input", type=str, required=True, help="input image (A)"
    )
    parser.add_argument(
        "-o", "--output", dest="output", type=str, help="output image (B)"
    )
    parser.add_argument(
        "--axis", type=int, help="translation applied on slice YZ=0, XZ=1, XY=2"
    )
    parser.add_argument(
        "--debug_axis", action="store_true", help="extract center slice and exit"
    )
    parser.add_argument(
        "--resample", action="store_true", help="resample input to target spacing"
    )
    parser.add_argument(
        "--pad", action="store_true", help="pad to next multiple of 256 / 4"
    )
    parser.add_argument(
        "--gpu_ids",
        nargs="+",
        type=int,
        help="space seperated list of GPU ids",
        default=[],
    )
    args = parser.parse_args()

    if not args.input:
        args.input = r"F:\Data\DRCMR-Thielscher\all_data\images\X10679.nii.gz"

    if args.debug_axis:
        crop_size = [1024, 1024, 1024]
        crop_size[args.axis] = 2
        itk.imwrite(crop(itk.imread(args.input), target_size=crop_size), args.output)
        return

    # resample/pad
    preprocess = lambda img: crop(img, target_size=(256, 256, 10))
    postprocess = lambda img: img

    # load model
    netg, device = load_pix2pix_generator(
        model_file_path=args.model, gpu_ids=args.gpu_ids, eval=False
    )

    # print(netg)
    # assert False

    # load input image
    img_t1 = itk.imread(args.input)

    # translate slice-by-slice
    img_ct = translate_3d(preprocess(img_t1), model=netg, axis=2, device=device)

    # write translated image
    itk.imwrite(postprocess(img_ct), args.output)


if __name__ == "__main__":
    main()

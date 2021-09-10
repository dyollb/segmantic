import numpy as np
import itk

from segmania.prepro.core import crop
from segmania.i2i.translate import load_pix2pix_generator, translate_3d


def preprocess_mri(x):
    x_view = itk.array_view_from_image(x)
    x_view *= 255.0 / 280.0
    np.clip(x_view, a_min=0, a_max=255, out=x_view)
    return x


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

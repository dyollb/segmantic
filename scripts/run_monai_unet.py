from monai.config import print_config
import os
import argparse

from segmantic.prepro.labels import load_tissue_list
from segmantic.seg.monai_unet import train, predict



def get_nifti_files(dir):
    if not dir:
        return []
    return sorted([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".nii.gz")])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and predict.')
    parser.add_argument('-i', '--image_dir', dest='image_dir', type=str, required=True, help='image directory')
    parser.add_argument('-l', '--labels_dir', dest='labels_dir', type=str, help='label image directory')
    parser.add_argument('-o', '--results_dir', dest='results_dir', default='.', type=str, help='results directory')
    parser.add_argument('--tissue_list', type=str, required=True, help='file containing label descriptors')
    parser.add_argument('--predict', action='store_true', help='run prediction')
    parser.add_argument('--gpu_ids', nargs="+", type=int, help='space seperated list of GPU ids, -1 is for CPU', default=[0])
    args = parser.parse_args()

    print_config()

    tissue_dict = load_tissue_list(args.tissue_list)
    num_classes = max(tissue_dict.values()) + 1
    assert len(tissue_dict) == num_classes, "Expecting contiguous labels in range [0,N-1]"

    os.makedirs(args.results_dir, exist_ok=True)
    log_dir = os.path.join(args.results_dir, "logs")
    model_file = os.path.join(args.results_dir, "drcmr_%d.ckpt" % num_classes)


    if args.predict:
        predict(
            model_file=model_file,
            test_images=get_nifti_files(args.image_dir),
            test_labels=get_nifti_files(args.labels_dir),
            tissue_dict=tissue_dict,
            output_dir=args.results_dir,
            save_nifti=True,
            gpu_ids=args.gpu_ids,
        )
    else:
        train(
            image_dir=args.image_dir,
            labels_dir=args.labels_dir,
            log_dir=log_dir,
            num_classes=num_classes,
            model_file_name=model_file,
            max_epochs=600,
            output_dir=args.results_dir,
            gpu_ids=args.gpu_ids,
        )

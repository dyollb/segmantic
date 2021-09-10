"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import math
import numpy as np
import itk

class TestOptions3D(TestOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = TestOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--input', type=str, help='input nifty file.')
        parser.add_argument('--output', type=str, help='output nifty file.')
        parser.add_argument('--axis', type=int, default=2, help='nifty file.')
        return parser

def resample(x):
    dim = x.GetImageDimension()
    interpolator = itk.LinearInterpolateImageFunction.New(x)
    transform = itk.IdentityTransform[itk.D, dim].New()

    size=itk.size(x)
    spacing=itk.spacing(x)
    for d in range(dim):
        size[d] = math.ceil(size[d] * spacing[d] / 0.85)
        spacing[d] = 0.85

    # resample to target resolution
    resampled = itk.resample_image_filter(
        x,
        transform=transform,
        interpolator=interpolator,
        size=size,
        output_spacing=spacing,
        output_origin=itk.origin(x),
        output_direction=x.GetDirection(),
    )
    return resampled

def normalize(x):
    x_view = itk.array_view_from_image(x)
    x_min, x_max = np.min(x_view), np.max(x_view)
    x_view -= x_min
    np.multiply(x_view, 255.0 / (x_max - x_min), out=x_view, casting='unsafe')
    np.clip(x_view, a_min=0, a_max=255, out=x_view)
    return x

def save_slices(nifty_file_path:str, folder:str, axis:int = 2):
    im = itk.imread(nifty_file_path)
    im = resample(normalize(im))

    os.makedirs(folder, exist_ok=True)

    for i in range(im.shape[axis]):
        slice = np.take(im, indices=i, axis=axis)
        itk.imwrite(itk.image_from_array(slice).astype(itk.SS), os.path.join(folder, "%04d.tif" % i))

def assemble_slices(folder:str, output_file_path:str, axis:int=2):
    os.makedirs(os.path.split(output_file_path)[0], exist_ok=True)
    pass


if __name__ == '__main__':
    opt = TestOptions3D().parse()  # get test options
    # hard-code some parameters for test
    opt.dataroot = r"F:\temp\_cyclegan_ixi2drcmr"
    opt.results_dir = r"F:\temp\_cyclegan_ixi2drcmr\results"
    opt.netG = "resnet_9blocks"
    opt.no_dropout = True
    opt.norm  = "instance"
    opt.dataset_mode = "unaligned"
    #opt.name = "modality_transform"
    opt.input_nc = 1
    opt.output_nc = 1
    opt.model = "cycle_gan"
    opt.preprocess = "none"
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    # make test data for cyclegan
    save_slices(opt.input, folder=os.path.join(opt.dataroot, "testA"), axis=opt.axis)

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    #os.makedirs(web_dir, exist_ok=True)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML

    assemble_slices(folder=opt.results_dir, axis=opt.axis, output_file_path=opt.output)

import torch.nn as nn
import functools

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.simplelayers import SkipConnection

from .factories import Act, Conv, Norm, Pad


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(spatial_dims: int, norm_type: str = "instance"):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        return functools.partial(
            Norm[Norm.BATCH, spatial_dims], affine=True, track_running_stats=True
        )
    elif norm_type == "instance":
        return functools.partial(
            Norm[Norm.INSTANCE, spatial_dims], affine=False, track_running_stats=False
        )
    elif norm_type == "none":

        def norm_layer(x):
            return Identity()

        return norm_layer
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, spatial_dims, input_nc=1, ndf=64, n_layers=3, norm_layer=None):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        conv_type = Conv[Conv.CONV, spatial_dims]
        instance_norm_type = Norm[Norm.INSTANCE, spatial_dims]
        if norm_layer is None:
            norm_layer = Norm[Norm.BATCH, spatial_dims]
        elif (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == instance_norm_type
        else:
            use_bias = norm_layer == instance_norm_type

        kw = 4
        padw = 1
        sequence = [
            conv_type(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                conv_type(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            conv_type(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            conv_type(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(
        self,
        spatial_dims,
        input_nc=1,
        output_nc=1,
        ngf=64,
        norm_layer=None,
        use_dropout=False,
        n_blocks=6,
        padding_type="reflect",
    ):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        conv_type = Conv[Conv.CONV, spatial_dims]
        conv_trans_type = Conv[Conv.CONVTRANS, spatial_dims]
        instance_norm_type = Norm[Norm.INSTANCE, spatial_dims]
        pad_type = Pad[Pad.REFLECTIONPAD, spatial_dims]
        if norm_layer is None:
            norm_layer = Norm[Norm.BATCH, spatial_dims]
        elif type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == instance_norm_type
        else:
            use_bias = norm_layer == instance_norm_type

        model = [
            pad_type(3),
            conv_type(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [
                conv_type(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [
                ResnetBlock(
                    spatial_dims,
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            ]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [
                conv_trans_type(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]
        model += [pad_type(3)]
        model += [conv_type(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(
        self, spatial_dims, dim, padding_type, norm_layer, use_dropout, use_bias
    ):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            spatial_dims, dim, padding_type, norm_layer, use_dropout, use_bias
        )

    def build_conv_block(
        self, spatial_dims, dim, padding_type, norm_layer, use_dropout, use_bias
    ):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_type = Conv[Conv.CONV, spatial_dims]
        if padding_type == "reflect":
            pad_type = Pad[Pad.REFLECTIONPAD, spatial_dims]
        elif padding_type == "replicate":
            pad_type = Pad[Pad.REPLICATIONPAD, spatial_dims]

        conv_block = []
        p = 0
        if padding_type == "reflect" or padding_type == "replicate":
            conv_block += [pad_type(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            conv_type(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect" or padding_type == "replicate":
            conv_block += [pad_type(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [
            conv_type(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

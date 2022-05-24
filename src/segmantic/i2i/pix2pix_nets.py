from typing import Callable, Iterable

import torch
from monai.networks.layers.factories import Conv, Norm
from torch import nn


class UpSampleConv(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels,
        out_channels,
        kernel=4,
        strides=2,
        padding=1,
        activation=True,
        batchnorm=True,
        dropout=False,
    ):
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout

        deconv_type: Callable = Conv[Conv.CONVTRANS, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        self.deconv = deconv_type(in_channels, out_channels, kernel, strides, padding)

        if batchnorm:
            self.bn = norm_type(out_channels)

        if activation:
            self.act = nn.ReLU(True)

        if dropout:
            self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.deconv(x)
        if self.batchnorm:
            x = self.bn(x)

        if self.dropout:
            x = self.drop(x)
        return x


class DownSampleConv(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels,
        out_channels,
        kernel=4,
        strides=2,
        padding=1,
        activation=True,
        batchnorm=True,
    ):
        """Paper details:

        - C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        """
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        self.conv = conv_type(in_channels, out_channels, kernel, strides, padding)

        if batchnorm:
            self.bn = norm_type(out_channels)

        if activation:
            self.act = nn.LeakyReLU(0.2)  # TODO: inplace=True

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.bn(x)
        if self.activation:
            x = self.act(x)
        return x


class Generator(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int):
        """Paper details:

        - Encoder: C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        - Decoder: CD512-CD1024-CD1024-C1024-C1024-C512 -C256-C128
        """
        super().__init__()

        conv_type: Callable = Conv[Conv.CONVTRANS, spatial_dims]

        # encoder/donwsample convs
        self.encoders: Iterable[nn.Module] = [
            DownSampleConv(
                spatial_dims, in_channels, 64, batchnorm=False
            ),  # bs x 64 x 128 x 128
            DownSampleConv(spatial_dims, 64, 128),  # bs x 128 x 64 x 64
            DownSampleConv(spatial_dims, 128, 256),  # bs x 256 x 32 x 32
            DownSampleConv(spatial_dims, 256, 512),  # bs x 512 x 16 x 16
            DownSampleConv(spatial_dims, 512, 512),  # bs x 512 x 8 x 8
            DownSampleConv(spatial_dims, 512, 512),  # bs x 512 x 4 x 4
            DownSampleConv(spatial_dims, 512, 512),  # bs x 512 x 2 x 2
            DownSampleConv(spatial_dims, 512, 512, batchnorm=False),  # bs x 512 x 1 x 1
        ]

        # decoder/upsample convs
        self.decoders: Iterable[nn.Module] = [
            UpSampleConv(spatial_dims, 512, 512, dropout=True),  # bs x 512 x 2 x 2
            UpSampleConv(spatial_dims, 1024, 512, dropout=True),  # bs x 512 x 4 x 4
            UpSampleConv(spatial_dims, 1024, 512, dropout=True),  # bs x 512 x 8 x 8
            UpSampleConv(spatial_dims, 1024, 512),  # bs x 512 x 16 x 16
            UpSampleConv(spatial_dims, 1024, 256),  # bs x 256 x 32 x 32
            UpSampleConv(spatial_dims, 512, 128),  # bs x 128 x 64 x 64
            UpSampleConv(spatial_dims, 256, 64),  # bs x 64 x 128 x 128
        ]
        self.decoder_channels = [512, 512, 512, 512, 256, 128, 64]
        self.final_conv = conv_type(
            64, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.tanh = nn.Tanh()

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, x):
        skips_cons = []
        for encoder in self.encoders:
            x = encoder(x)

            skips_cons.append(x)

        skips_cons = list(reversed(skips_cons[:-1]))
        decoders = self.decoders[:-1]

        for decoder, skip in zip(decoders, skips_cons):
            x = decoder(x)
            # print(x.shape, skip.shape)
            x = torch.cat((x, skip), axis=1)

        x = self.decoders[-1](x)
        # print(x.shape)
        x = self.final_conv(x)
        return self.tanh(x)


class PatchGAN(nn.Module):
    def __init__(self, spatial_dims: int, input_channels: int):
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]

        self.d1 = DownSampleConv(spatial_dims, input_channels, 64, batchnorm=False)
        self.d2 = DownSampleConv(spatial_dims, 64, 128)
        self.d3 = DownSampleConv(spatial_dims, 128, 256)
        self.d4 = DownSampleConv(spatial_dims, 256, 512)
        self.final = conv_type(512, 1, kernel_size=1)

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x0 = self.d1(x)
        x1 = self.d2(x0)
        x2 = self.d3(x1)
        x3 = self.d4(x2)
        xn = self.final(x3)
        return xn

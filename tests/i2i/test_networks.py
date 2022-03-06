from typing import Tuple
import numpy as np
import random
import torch
from torchinfo import summary

try:  # not available for Python 3.7
    import torcheck

    HAVE_TORCHECK = True
except:
    HAVE_TORCHECK = False

from segmantic.i2i.networks import (
    Pad,
    ResnetGenerator,
    NLayerDiscriminator,
    get_norm_layer,
)
from segmantic.i2i.translate import define_G


def test_Pad_factory():
    input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)

    # pre-existing monai pading type
    assert Pad[Pad.CONSTANTPAD, 2]((1, 3), 1.5)

    # test reflection padding
    pad_type = Pad[Pad.REFLECTIONPAD, 2]
    m = pad_type(2)
    output = m(input)
    assert output.shape[-2] == 3 + 4
    assert output.shape[-1] == 3 + 4

    m = pad_type((1, 1, 2, 1))
    output = m(input)
    assert output.shape[-2] == 3 + 3
    assert output.shape[-1] == 3 + 2

    input = torch.arange(12, dtype=torch.float).reshape(1, 1, 2, 3, 2)
    pad_type = Pad[Pad.REFLECTIONPAD, 3]
    m = pad_type((1, 1, 2, 1, 1, 1))
    assert len(m.padding) == 6
    output = m(input)
    assert output.shape[-3] == input.shape[-3] + 2
    assert output.shape[-2] == input.shape[-2] + 3
    assert output.shape[-1] == input.shape[-1] + 2

    output_arr = output.numpy()
    np.testing.assert_array_equal(output_arr[..., 0], output_arr[..., 2])
    np.testing.assert_array_equal(output_arr[..., -1], output_arr[..., -3])
    np.testing.assert_array_equal(output_arr[..., 0:1, :], output_arr[..., 4:3:-1, :])
    np.testing.assert_array_equal(output_arr[..., -1, :], output_arr[..., -3, :])
    np.testing.assert_array_equal(output_arr[..., 0, :, :], output_arr[..., 2, :, :])
    np.testing.assert_array_equal(output_arr[..., -1, :, :], output_arr[..., -3, :, :])


def test_ResnetGenerator():
    spatial_dims = 2
    norm_layer = get_norm_layer(spatial_dims, "instance")
    net2d = ResnetGenerator(
        spatial_dims,
        input_nc=1,
        output_nc=1,
        ngf=64,
        norm_layer=norm_layer,
        use_dropout=False,
        n_blocks=6,
    )
    # summary(net2d, input_size=(1, 1, 224, 224))

    gen = define_G(
        input_nc=1,
        output_nc=1,
        ngf=64,
        netG="resnet_6blocks",
        norm="instance",
        use_dropout=False,
        init_type="none",
        init_gain=0.02,
        gpu_ids=[],
    )
    # summary(gen, input_size=(1, 1, 224, 224))

    num_params = lambda m: sum([param.nelement() for param in m.parameters()])

    assert num_params(net2d) == num_params(gen)

    spatial_dims = 3
    norm_layer = get_norm_layer(spatial_dims, "instance")
    net3d = ResnetGenerator(
        spatial_dims,
        input_nc=1,
        output_nc=1,
        ngf=64,
        norm_layer=norm_layer,
        use_dropout=False,
        n_blocks=6,
    )
    summary(net3d, input_size=(1, 1, 16, 96, 96))


def test_PatchGAN():
    # for PatchGAN (netD == 'basic') n_layers=3
    spatial_dims = 2
    norm_layer = get_norm_layer(spatial_dims, "instance")
    net2d = NLayerDiscriminator(
        spatial_dims=spatial_dims, input_nc=1, ndf=64, n_layers=3, norm_layer=norm_layer
    )
    summary(net2d, input_size=(1, 1, 224, 224))

    spatial_dims = 3
    norm_layer = get_norm_layer(spatial_dims, "instance")
    net3d = NLayerDiscriminator(
        spatial_dims=spatial_dims, input_nc=1, ndf=64, n_layers=3, norm_layer=norm_layer
    )
    summary(net3d, input_size=(1, 1, 24, 96, 96))

    input = torch.empty(1, 1, 24, 96, 96)
    output = net3d(input)
    print(output.shape)


def test_PatchGAN_optimize():
    spatial_dims = 3
    norm_layer = get_norm_layer(spatial_dims, "instance")
    net = NLayerDiscriminator(
        spatial_dims=spatial_dims, input_nc=1, ndf=64, n_layers=3, norm_layer=norm_layer
    )

    def striped_image(shape: Tuple[int, ...], vertical: bool) -> torch.Tensor:
        t = torch.zeros(size=shape)
        if vertical:
            t[:, :, 2 : shape[2] // 2] = 1
        else:
            t[:, :, :, 2 : shape[3] // 2] = 1
        return t

    classes = [-1.0 if random.uniform(0, 1) > 0.5 else 1.0]
    data = {striped_image((1, 1, 24, 24, 24), c > 0): c for c in classes}

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    if HAVE_TORCHECK:
        torcheck.register(optimizer)
        torcheck.add_module(
            net, module_name="PatchGAN3d", changing=True, check_inf=True, check_nan=True
        )

    for _ in range(10):
        for batch, c in data.items():
            y = net(batch)
            if c > 0.0:
                loss = y.mean()
            else:
                loss = -y.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    test_PatchGAN_optimize()

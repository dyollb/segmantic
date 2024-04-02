import numpy as np
import torch
from monai.bundle import ConfigParser

from segmantic.seg.transforms import MapLabels, SelectChannel, SelectChanneld


def test_MapLabels():
    labels = torch.tensor([2, 1, 2, 0]).reshape(1, 4, 1, 1)

    mapper = MapLabels({1: 3, 2: 1, 0: 0})
    labels_mapped = mapper(labels)
    assert (labels_mapped == torch.tensor([1, 3, 1, 0]).reshape(1, 4, 1, 1)).all()


def test_Bundle_MapLabels():
    parser = ConfigParser(
        {
            "imports": ["$import segmantic"],
            "mapping": "${1: 3, 2: 1, 0: 0}",
            "postpro": "$segmantic.seg.transforms.MapLabels(@mapping)",
        }
    )
    parser.parse(True)
    mapper = parser.get_parsed_content("postpro")
    print(mapper)
    assert isinstance(mapper, MapLabels)

    labels = torch.tensor([2, 1, 2, 0]).reshape(1, 4, 1, 1)
    labels_mapped = mapper(labels)
    assert (labels_mapped == torch.tensor([1, 3, 1, 0]).reshape(1, 4, 1, 1)).all()


def test_SelectChannel():
    for asarray in (np.asarray, torch.tensor):
        img = asarray([2, 1, 2, 0, 5, 6, 7, 3]).reshape(2, 4, 1, 1)

        select = SelectChannel(channel_dim=0, channel=0)
        img_channel0 = select(img)
        assert (img_channel0 == asarray([2, 1, 2, 0]).reshape(1, 4, 1, 1)).all()


def test_SelectChannel_single_channel_input():
    for asarray in (np.asarray, torch.tensor):
        img = asarray([2, 1, 2, 0]).reshape(1, 4, 1, 1)

        select = SelectChannel(channel_dim=0, channel=0)
        img_channel0 = select(img)
        assert (img_channel0 == img).all()


def test_SelectChanneld():
    img = torch.tensor([2, 1, 2, 0, 5, 6, 7, 3]).reshape(2, 4, 1, 1)

    select = SelectChanneld(
        keys="img", channel_dim=0, channel=0, new_key_postfix="_sel"
    )
    img_channel0 = select({"img": img})
    assert isinstance(img_channel0, dict)
    assert "img_sel" in img_channel0
    assert (
        img_channel0["img_sel"] == torch.tensor([2, 1, 2, 0]).reshape(1, 4, 1, 1)
    ).all()


if __name__ == "__main__":
    test_SelectChannel_single_channel_input()

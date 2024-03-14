import torch
from monai.bundle import ConfigParser

from segmantic.seg.transforms import MapLabels


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


if __name__ == "__main__":
    test_Bundle_MapLabels()

import itk

import pytest
from segmantic.i2i.translate import make_tiles, tile_image
from tests.fixture import make_image


def test_tile_image():
    image = make_image(shape=(30, 27), spacing=(1.5, 2.0), value=1.0)
    start_ids = make_tiles(size=itk.size(image), tile_size=(15, 14), overlap=0)
    print(start_ids)
    assert len(start_ids) == 4
    for start in start_ids:
        assert start[0] in (0, 15)
        assert start[1] in (0, 13)

    patches = tile_image(image, tiles=start_ids)
    for p in patches:
        d = itk.size(d)
        assert d[0] == 15 and d[1] == 15

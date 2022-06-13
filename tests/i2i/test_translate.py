import itk
import numpy as np

from segmantic.i2i.translate import make_tiles, tile_image, merge_tiles
from segmantic.prepro.core import make_image


def test_tile_image():
    tile_size = (15, 14)
    image = make_image(shape=(29, 27), spacing=(1.5, 2.0), value=1.0)

    tile_indices = make_tiles(size=itk.size(image), tile_size=tile_size, overlap=1)
    print(tile_indices)
    assert len(tile_indices) == 4
    for start in tile_indices:
        assert start[0] in (0, 14)
        assert start[1] in (0, 13)

    tiles = tile_image(image, tile_indices=tile_indices, tile_size=tile_size)
    for p in tiles:
        d = itk.size(p)
        assert d[0] == tile_size[0] and d[1] == tile_size[1]

    for i, p in enumerate(tiles):
        p[:] = i

    out = merge_tiles(tiles, tile_indices, tile_size)
    re_tiled = tile_image(out, tile_indices=tile_indices, tile_size=tile_size)
    assert np.unique(itk.array_view_from_image(re_tiled[-1])) == np.array([3])
    assert (
        np.unique(itk.array_view_from_image(re_tiled[-2])) == np.array([2, 3])
    ).all()
    # note: pixel value 2 is overwritten by 3
    assert (
        np.unique(itk.array_view_from_image(re_tiled[-3])) == np.array([1, 3])
    ).all()
    assert (
        np.unique(itk.array_view_from_image(re_tiled[-4])) == np.array([0, 1, 2, 3])
    ).all()

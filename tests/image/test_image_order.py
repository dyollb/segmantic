import numpy as np

from segmantic.image.utils import array_view_reverse_ordering


def test_array_view_reverse_ordering():

    rng = np.random.default_rng()
    im3d = rng.random((12, 13, 14))
    im3d_f = array_view_reverse_ordering(im3d)

    assert im3d_f.flags.owndata is False
    assert im3d.shape == im3d_f.shape[::-1]

    for k in range(12):
        for j in range(13):
            for i in range(14):
                assert im3d[k, j, i] == im3d_f[i, j, k]


def profile_image_ordering():
    def in_order_multiply(arr, scalar):
        for plane in list(range(arr.shape[0])):
            arr[plane, :, :] *= scalar

    def out_of_order_multiply(arr, scalar):
        for plane in list(range(arr.shape[2])):
            arr[:, :, plane] *= scalar

    rng = np.random.default_rng()
    im3d = rng.random((500, 500, 500))
    im3d_f = array_view_reverse_ordering(im3d)

    import time

    s0 = time.time()
    _ = out_of_order_multiply(im3d_f, 5)
    s1 = time.time()
    print("%.2f seconds" % (s1 - s0))

    t0 = time.time()
    _ = in_order_multiply(im3d, 5)
    t1 = time.time()
    print("%.2f seconds" % (t1 - t0))

    print("Speedup: %.1fx" % ((s1 - s0) / (t1 - t0)))


if __name__ == "__main__":
    profile_image_ordering()

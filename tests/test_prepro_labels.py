# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package

import unittest

import os
from tempfile import TemporaryDirectory

from segmantic.prepro import labels


class TestPreproLabels(unittest.TestCase):
    def test_tissue_list_io(self):
        with TemporaryDirectory() as d:
            file_path = os.path.join(d, "tissue.txt")
            tissue_dict = {"BG": 0, "Bone": 1, "Fat": 2, "Skin": 3}
            labels.save_tissue_list(tissue_dict, file_path)

            tissue_dict2 = labels.load_tissue_list(file_path)

            self.assertEqual(len(tissue_dict), len(tissue_dict2))
            for n in tissue_dict:
                if n == "BG":
                    continue
                self.assertIn(n, tissue_dict2)
                self.assertEqual(tissue_dict[n], tissue_dict2[n])

    def test_build_tissue_map(self):
        tissue_dict = {"BG": 0, "Bone": 1, "Fat": 2, "Skin": 3}

        def map(name):
            if name == "BG":
                return "Background"
            if name == "Bone":
                return "Bone"
            return "Other"

        omap, i2o = labels.build_tissue_mapping(tissue_dict, map)
        self.assertEqual(len(omap), 3)
        for n1 in tissue_dict:
            n2 = map(n1)
            i1 = tissue_dict[n1]
            i2 = omap[n2]
            self.assertEqual(i2o[i1], i2)


if __name__ == "__main__":
    unittest.main()

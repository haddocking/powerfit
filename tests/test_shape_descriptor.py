from unittest import TestCase, main

import numpy as np

from powerfit_em.shape_descriptor import ShapeDescriptor


class TestShapeDescriptor(TestCase):

    def test_relative_shape_anisotropy(self):
        # coordinates on a line
        coor = np.zeros((3, 10), dtype=np.float64)
        coor[0] = np.arange(10)
        sd = ShapeDescriptor(coor)
        self.assertAlmostEqual(sd.relative_shape_anisotropy, 1)

        # points on a unit sphere
        coor = np.zeros((3, 6), dtype=np.float64)
        coor[0] = [1, -1, 0, 0, 0, 0]
        coor[1] = [0, 0, 1, -1, 0, 0]
        coor[2] = [0, 0, 0, 0, 1, -1]
        sd = ShapeDescriptor(coor)
        self.assertAlmostEqual(sd.relative_shape_anisotropy, 0)


if __name__ == "__main__":
    main()

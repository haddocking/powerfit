import unittest

import numpy as np

from powerfit_em.rotations import random_rotmat, euler, euler_to_rotmat


class TestRotations(unittest.TestCase):

    def test_random_rotmat(self):

        rotmat = random_rotmat()
        # the determinant of a rotmat is 1
        self.assertAlmostEqual(np.linalg.det(rotmat), 1)

    def test_euler(self):
        # 90 z-rotation
        angle = np.radians(90)
        out = euler(angle, "z")
        answer = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        self.assertTrue(np.allclose(answer, out))

        # 90 y-rotation
        out = euler(angle, "y")
        answer = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        self.assertTrue(np.allclose(answer, out))

        # 90 x-rotation
        out = euler(angle, "x")
        answer = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        self.assertTrue(np.allclose(answer, out))

    def test_euler_to_rotmat(self):
        # 90 z-rotation
        angles = [np.radians(x) for x in (0, 0, 90)]
        answer = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        out = euler_to_rotmat(angles)
        self.assertTrue(np.allclose(answer, out))


if __name__ == "__main__":
    unittest.main()

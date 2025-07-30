import unittest
from unittest import skipIf

import numpy as np

from powerfit_em import _powerfit
from powerfit_em.rotations import euler
from powerfit_em.powerfitter import OPENCL


@skipIf(not OPENCL, "CL not available")
@skipIf(True, "powerfit._powerfit module does not have rotate_image3d function")
class TestRotateImage3d(unittest.TestCase):
    """Test the rotate_image3d function."""

    def setUp(self):
        self.shape = (5, 6, 7)
        self.radius = 3
        self.center = np.asarray([0, 0, 0], dtype=np.float64)
        self.rotmat = np.asarray([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64).reshape(
            3, 3
        )
        self.image = np.zeros(self.shape, dtype=np.float64)
        self.image[0, 0, : self.radius] = 1
        self.out = np.zeros_like(self.image)
        self.answer = np.zeros_like(self.image)

    def test_identity(self):
        # simple identity rotation
        _powerfit.rotate_image3d(
            self.image, self.rotmat, self.center, self.radius, self.out
        )
        self.assertTrue(np.allclose(self.out, self.image))

    def test_radius(self):
        # radius too short
        radius = 1
        _powerfit.rotate_image3d(
            self.image, self.rotmat, self.center, self.radius, self.out
        )
        self.answer = self.image
        self.assertTrue(np.allclose(self.answer, self.out))

    def test_center(self):
        self.center = np.asarray((1, 0, 0), dtype=np.float64)
        _powerfit.rotate_image3d(
            self.image, self.rotmat, self.center, self.radius, self.out
        )
        self.answer[0, 0, : self.radius - 1] = 1
        self.answer[0, 0, -1] = 1
        self.assertTrue(np.allclose(self.answer, self.out))

    def test_z_rotation_m90(self):
        # -90 degree z-rotation
        self.rotmat = euler(np.radians(-90), "z")
        _powerfit.rotate_image3d(
            self.image, self.rotmat, self.center, self.radius, self.out
        )
        self.answer[0, -self.radius + 1 :, 0] = 1
        self.answer[0, 0, 0] = 1
        self.assertTrue(np.allclose(self.answer, self.out))

    def test_z_rotation_90(self):
        # +90 degree z-rotation
        self.rotmat = euler(np.radians(90), "z")
        _powerfit.rotate_image3d(
            self.image, self.rotmat, self.center, self.radius, self.out
        )
        self.answer[0, : self.radius, 0] = 1
        self.assertTrue(np.allclose(self.answer, self.out))

    def test_near(self):
        self.rotmat = euler(np.radians(85), "z")
        _powerfit.rotate_image3d(
            self.image, self.rotmat, self.center, self.radius, self.out, nearest=True
        )
        self.answer[0, : self.radius, 0] = 1
        self.assertTrue(np.allclose(self.answer, self.out))


if __name__ == "__main__":
    unittest.main()

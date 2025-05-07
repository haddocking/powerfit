from unittest import TestCase, main, skipIf

import numpy as np
import numpy.testing as npt

from powerfit_em.helpers import get_queue
from powerfit_em._extensions import rotate_grid3d

try:
    from powerfit_em.powerfitter import CLKernels
    import pyopencl as cl
    import pyopencl.array as cl_array

    HAS_CL = True
except ImportError:
    HAS_CL = False


@skipIf(not HAS_CL, "CL not available")
class TestCLKernels(TestCase):

    @classmethod
    def setUpClass(self):

        self.queue = get_queue()
        self.shape = (4, 5, 6)
        self.size = 4 * 5 * 6
        self.values = {
            "shape_x": self.shape[2],
            "shape_y": self.shape[1],
            "shape_z": self.shape[0],
            "llength": 2,
        }
        self.k = CLKernels(self.queue.context, self.values)
        self.grid = np.zeros(self.shape, dtype=np.float64)
        self.grid[0, 0, 0] = 1
        self.grid[0, 0, 1] = 1
        self.grid[0, 1, 1] = 1
        self.grid[0, 0, 2] = 1
        self.grid[0, 0, -1] = 1
        self.grid[-1, 0, 0] = 1
        self.cl_image = cl.image_from_array(
            self.queue.context, self.grid.astype(np.float32)
        )
        self.sampler_linear = cl.Sampler(
            self.queue.context, True, cl.addressing_mode.REPEAT, cl.filter_mode.LINEAR
        )
        self.sampler_nearest = cl.Sampler(
            self.queue.context, False, cl.addressing_mode.CLAMP, cl.filter_mode.NEAREST
        )
        self.out = np.zeros(self.shape, dtype=np.float64)
        self.cl_out = cl_array.zeros(self.queue, self.shape, dtype=np.float32)

    def test_rotate_image3d_linear(self):
        k = self.k._program.rotate_image3d

        rotmat = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        cl_rotmat = np.asarray([1, 0, 0, 0, 1, 0, 0, 0, 1] + [0] * 7, dtype=np.float32)
        args = (self.cl_image, self.sampler_linear, cl_rotmat, self.cl_out.data)
        k(self.queue, (5, 5, 5), None, *args)
        rotate_grid3d(self.grid, rotmat, 2, self.out, False)

        test = np.allclose(self.cl_out.get(), self.out)
        self.assertTrue(test)

        # 90 degree rotation around z-axis
        rotmat = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        cl_rotmat = np.asarray([0, -1, 0, 1, 0, 0, 0, 0, 1] + [0] * 7, dtype=np.float32)
        args = (self.cl_image, self.sampler_linear, cl_rotmat, self.cl_out.data)
        k(self.queue, (5, 5, 5), None, *args)
        rotate_grid3d(self.grid, rotmat, 2, self.out, False)

        test = np.allclose(self.cl_out.get(), self.out)
        self.assertTrue(test)

        # Non-integer rotation
        rotmat = np.asarray(
            [
                [0.30901699, -0.5, 0.80901699],
                [-0.80901699, 0.30901699, 0.5],
                [-0.5, -0.80901699, -0.30901699],
            ],
            dtype=np.float64,
        )
        cl_rotmat = np.asarray(rotmat.ravel().tolist() + [0] * 7, dtype=np.float32)
        args = (self.cl_image, self.sampler_linear, cl_rotmat, self.cl_out.data)
        k(self.queue, (1, 1, 1), None, *args)
        rotate_grid3d(self.grid, rotmat, 2, self.out, False)
        test = np.allclose(self.cl_out.get(), self.out)
        # There will be difference between CPU and GPU, but the results should
        # be less than 5% different. This is an ad hoc number.
        diff = np.nan_to_num(np.abs((self.cl_out.get() - self.out) / self.out)).max()
        self.assertTrue(diff < 0.05)

    def test_rotate_grid3d_nearest(self):
        """Test rotate_grid3d kernel using nearest interpolation."""
        k = self.k._program.rotate_grid3d

        # Identity rotation
        rotmat = np.asarray([1, 0, 0, 0, 1, 0, 0, 0, 1] + [0] * 7, dtype=np.float32)
        grid = np.zeros((4, 5, 6), dtype=np.float32)
        grid[0, 0, 0] = 1
        grid[0, 0, 1] = 1
        grid[0, 1, 1] = 1
        grid[0, 0, 2] = 1
        grid[0, 0, -1] = 1
        grid[-1, 0, 0] = 1
        self.cl_grid = cl_array.to_device(self.queue, grid)
        self.cl_out = cl_array.zeros_like(self.cl_grid)

        args = (self.cl_grid.data, rotmat, self.cl_out.data, np.int32(True))
        gws = tuple([2 * self.values["llength"] + 1] * 3)
        k(self.queue, gws, None, *args)

        self.assertTrue(np.allclose(self.cl_grid.get(), self.cl_out.get()))

        # 90' rotation around z-axis
        self.cl_out.fill(0)
        rotmat = np.asarray([0, -1, 0, 1, 0, 0, 0, 0, 1] + [0] * 7, dtype=np.float32)
        args = (self.cl_grid.data, rotmat, self.cl_out.data, np.int32(True))
        gws = tuple([2 * self.values["llength"] + 1] * 3)
        k(self.queue, gws, None, *args)

        answer = np.zeros(self.shape, dtype=np.float32)
        answer[0, 0, 0] = 1
        answer[0, 1, 0] = 1
        answer[0, 1, -1] = 1
        answer[0, 2, 0] = 1
        answer[0, -1, 0] = 1
        answer[-1, 0, 0] = 1
        self.assertTrue(np.allclose(answer, self.cl_out.get()))

    @skipIf(
        True, "rotate_grid3d not executed by powerfit cli with --gpu, ignoring test"
    )
    def test_rotate_grid3d_linear(self):
        """Test rotate_grid3d kernel using nearest interpolation."""
        k = self.k._program.rotate_grid3d

        # Identity rotation
        rotmat = np.asarray([1, 0, 0, 0, 1, 0, 0, 0, 1] + [0] * 7, dtype=np.float32)
        grid = np.zeros((4, 5, 6), dtype=np.float32)
        grid[0, 0, 0] = 1
        grid[0, 0, 1] = 1
        grid[0, 1, 1] = 1
        grid[0, 0, 2] = 1
        grid[0, 0, -1] = 1
        grid[-1, 0, 0] = 1
        self.cl_grid = cl_array.to_device(self.queue, grid)
        self.cl_out = cl_array.zeros_like(self.cl_grid)

        args = (self.cl_grid.data, rotmat, self.cl_out.data, np.int32(False))
        gws = tuple([2 * self.values["llength"] + 1] * 3)
        k(self.queue, gws, None, *args)

        with open("cl_grid.txt", "w") as f:
            f.write(str(self.cl_grid.get()))
        with open("cl_out.txt", "w") as f:
            f.write(str(self.cl_out.get()))

        npt.assert_allclose(
            self.cl_grid.get(),
            self.cl_out.get(),
        )

        # 90' rotation around z-axis
        self.cl_out.fill(0)
        rotmat = np.asarray([0, -1, 0, 1, 0, 0, 0, 0, 1] + [0] * 7, dtype=np.float32)
        args = (self.cl_grid.data, rotmat, self.cl_out.data, np.int32(False))
        gws = tuple([2 * self.values["llength"] + 1] * 3)
        k(self.queue, gws, None, *args)

        answer = np.zeros(self.shape, dtype=np.float32)
        answer[0, 0, 0] = 1
        answer[0, 1, 0] = 1
        answer[0, 1, -1] = 1
        answer[0, 2, 0] = 1
        answer[0, -1, 0] = 1
        answer[-1, 0, 0] = 1
        npt.assert_allclose(answer, self.cl_out.get())

        # Non-integer rotation
        rotmat = np.asarray(
            [
                [0.30901699, -0.5, 0.80901699],
                [-0.80901699, 0.30901699, 0.5],
                [-0.5, -0.80901699, -0.30901699],
            ],
            dtype=np.float64,
        )
        cl_rotmat = np.asarray(rotmat.ravel().tolist() + [0] * 7, dtype=np.float32)
        args = (self.cl_grid.data, cl_rotmat, self.cl_out.data, np.int32(False))
        k(self.queue, gws, None, *args)
        rotate_grid3d(self.grid, rotmat, 2, self.out, False)
        npt.assert_allclose(self.cl_out.get(), self.out)

        # self.assertTrue(test)


if __name__ == "__main__":
    main()

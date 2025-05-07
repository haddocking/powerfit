import unittest

import numpy as np
from scipy.ndimage import laplace

try:
    import pyfftw

    PYFFTW = True
except ImportError:
    PYFFTW = False
try:
    import pyopencl as cl
    import pyopencl.array as cl_array

    PYOPENCL = True
except ImportError:
    PYOPENCL = False

from powerfit_em.powerfitter import BaseCorrelator, CPUCorrelator
from powerfit_em.rotations import euler

if PYOPENCL:
    from powerfit_em.powerfitter import CLKernels


class TestCPUCorrelator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.shape = (5, 6, 7)
        cls.target = np.random.rand(*cls.shape)
        cls.corr = CPUCorrelator(cls.target)

    @unittest.skip("To implement")
    def test___init__(self):
        pass

    def test_allocate_array(self):
        arr = self.corr._allocate_array(self.shape, np.float64, PYFFTW)
        self.assertTrue(np.allclose(arr, 0))

    def test_rotate_grids(self):
        # identity matrix
        rotmat = np.identity(3, dtype=np.float64)
        mask = np.ones(self.shape, dtype=np.float64)
        template = np.zeros(self.shape, dtype=np.float64)
        radius = 3
        template[0, 0, 0:radius] = 1
        self.corr._template = template
        self.corr._mask = mask
        self.corr._center = np.asarray([0, 0, 0], dtype=np.float64)
        self.corr._rmax = radius
        # get clean arrays
        self.corr._allocate_arrays(self.shape)
        self.corr._rotate_grids(rotmat)
        self.assertTrue(np.allclose(self.corr._rot_template, template))

        # 90 degree Z-rotation.
        rotmat = euler(np.radians(90), "z")
        self.corr._allocate_arrays(self.shape)
        self.corr._rotate_grids(rotmat)
        answer = np.zeros(self.shape, dtype=np.float64)
        answer[0, :radius, 0] = 1
        self.assertTrue(np.allclose(self.corr._rot_template, answer))

    def test_get_lcc(self):
        """Test for lcc unity with perfect fit."""
        # overwrite the lcc_mask
        self.corr._lcc_mask.fill(1)
        self.corr.template = self.target.copy()
        self.corr.mask = np.ones(self.shape, dtype=np.float64)
        # fill the rotated arrays
        self.corr._rot_template[:] = self.corr._template
        self.corr._rot_mask[:] = self.corr._mask

        self.corr._get_lcc()
        self.assertAlmostEqual(self.corr._lcc_scan.max(), 1)
        self.assertEqual(self.corr._lcc_scan.argmax(), 0)

    @unittest.skip("To implement")
    def test_translational_scan(self):
        pass


class TestBaseCorrelator(unittest.TestCase):

    def setUp(self):
        self.shape = (8, 9, 10)
        self.target = np.random.rand(*self.shape)
        self.corr = BaseCorrelator(self.target)

    def test___init__(self):
        self.assertEqual(self.corr._target.max(), 1)

    def test_rotations_setter(self):
        self.corr.rotations = [0] * 9
        self.assertIsInstance(self.corr._rotations, np.ndarray)
        with self.assertRaises(ValueError):
            self.corr.rotations = [0] * 3
        self.corr.rotations = [0] * 27
        self.assertEqual(self.corr._rotations.shape, (3, 3, 3))

    def test_template_setter(self):
        template = np.random.rand(*(3, 3, 3))
        with self.assertRaises(ValueError):
            self.corr.template = template

        # test resetting of mask
        self.corr._mask = 1
        self.corr.template = np.random.rand(*self.shape)
        self.assertTrue(self.corr._mask is None)

    @unittest.skip("Broken")
    def test_laplace_filter(self):
        template = np.random.rand(*self.shape)
        self.corr._template = self.corr._laplace_filter(template)
        self.assertTrue(
            np.allclose(self.corr._template, laplace(template, mode="constant"))
        )

    # FIXME: Where is this `_get_center` defined?
    @unittest.skip("Broken")
    def test_get_center(self):
        center = self.corr._get_center(self.shape)
        self.assertTrue(np.allclose(center, (5, 4.5, 4)))

    def test_normalize_template(self):
        self.corr.template = np.random.rand(*self.shape)
        self.corr._mask = np.ones(self.shape)
        self.corr._normalize_template(self.corr._mask != 0)
        self.assertAlmostEqual(self.corr._template.mean(), 0)
        self.assertAlmostEqual(self.corr._template.std(), 1)

    # TODO
    @unittest.skip("Not yet implemented")
    def test_get_rmax(self):
        self.corr.template = np.random.rand(*self.shape)
        mask = np.zeros(self.shape, dtype=np.float64)
        mask[3:6] = 1
        self.corr._mask = mask
        self.corr._get_center(self.shape)
        self.corr._get_rmax()

    def test_mask_setter(self):
        # Template is not set, so ValueError should be raised no matter what
        with self.assertRaises(ValueError):
            self.corr.mask = None

        template = np.random.rand(*self.shape)
        self.corr._template = template
        # Shape of mask is not correct
        with self.assertRaises(ValueError):
            self.corr.mask = np.random.rand(10)

        # If mask only is zero, raise ValueError
        with self.assertRaises(ValueError):
            self.corr.mask = np.zeros(self.shape)

        self.corr._laplace = True
        self.corr._template = template.copy()
        mask = np.ones(self.shape)
        self.corr.mask = mask
        self.assertTrue(np.allclose(self.corr._mask, mask))

    def test_scan(self):
        # No requirement has been set
        with self.assertRaises(ValueError):
            self.corr.scan()


@unittest.skipIf(not PYOPENCL, "GPU resources are not available.")
class TestCLKernels(unittest.TestCase):
    """Tests for the OpenCL kernels"""

    # @classmethod
    # def setUpClass(cls):
    #    p = cl.get_platforms()[0]
    #    devs = p.get_devices()
    #    cls.ctx = cl.Context(devices=devs)
    #    cls.queue = cl.CommandQueue(cls.ctx, device=devs[0])
    #    cls.k = CLKernels(cls.ctx)
    #    cls.s_linear = cl.Sampler(cls.ctx, False, cl.addressing_mode.CLAMP,
    #            cl.filter_mode.LINEAR)
    #    cls.s_nearest = cl.Sampler(cls.ctx, False, cl.addressing_mode.CLAMP,
    #            cl.filter_mode.NEAREST)

    def setUp(self):
        p = cl.get_platforms()[0]
        devs = p.get_devices()
        self.ctx = cl.Context(devices=devs)
        self.queue = cl.CommandQueue(self.ctx, device=devs[0])
        values = {
            "shape_x": 10,
            "shape_y": 0,
            "shape_z": 0,
            "llength": 5,
        }
        self.k = CLKernels(self.ctx, values=values)
        self.s_linear = cl.Sampler(
            self.ctx, False, cl.addressing_mode.CLAMP, cl.filter_mode.LINEAR
        )
        self.s_nearest = cl.Sampler(
            self.ctx, False, cl.addressing_mode.CLAMP, cl.filter_mode.NEAREST
        )

    def test_multiply(self):
        np_in1 = np.arange(10, dtype=np.float32)
        np_in2 = np.arange(10, dtype=np.float32)
        np_out = np_in1 * np_in2

        cl_in1 = cl_array.to_device(self.queue, np_in1)
        cl_out = cl_array.to_device(self.queue, np.zeros(10, dtype=np.float32))
        cl_in2 = cl_array.to_device(self.queue, np_in2)

        self.k.multiply(cl_in1, cl_in2, cl_out)
        self.assertTrue(np.allclose(np_out, cl_out.get()))

    def test_conj_multiply(self):
        np_in1 = np.zeros(10, dtype=np.complex64)
        np_in2 = np.zeros(10, dtype=np.complex64)
        np_in1.real = np.random.rand(10)
        np_in1.imag = np.random.rand(10)
        np_in2.real = np.random.rand(10)
        np_in2.imag = np.random.rand(10)
        np_out = np_in1.conj() * np_in2

        cl_in1 = cl_array.to_device(self.queue, np_in1)
        cl_in2 = cl_array.to_device(self.queue, np_in2)
        cl_out = cl_array.to_device(self.queue, np.zeros(10, dtype=np.complex64))
        self.k.conj_multiply(cl_in1, cl_in2, cl_out)
        self.assertTrue(np.allclose(np_out, cl_out.get()))

    # TODO
    # def test_calc_lcc(self):
    #    pass
    # def test_take_best(self):
    #    pass

    @unittest.skip("kernel does not have rotate_template")
    def test_rotate_template_mask(self):
        shape = (5, 5, 5)
        values = {
            "shape_x": 5,
            "shape_y": 5,
            "shape_z": 5,
            "llength": 2,
        }
        self.k = CLKernels(self.ctx, values=values)
        template = np.zeros(shape, dtype=np.float32)
        template[2, 2, 1:4] = 1
        template[2, 1:4, 2] = 1
        rotmat = np.asarray([1, 0, 0, 0, 1, 0, 0, 0, 1] + [0] * 7, dtype=np.float32)

        self.queue.finish()
        cl_template = cl.image_from_array(self.queue.context, template)
        cl_out = cl_array.to_device(self.queue, np.zeros(shape, dtype=np.float32))
        center = np.asarray([2, 2, 2, 0], dtype=np.float32)
        shape = np.asarray([5, 5, 5, 125], dtype=np.int32)

        self.k.rotate_template(
            self.queue,
            (125,),
            None,
            self.s_linear,
            cl_template,
            rotmat,
            cl_out.data,
            center,
            shape,
        )

        answer = np.zeros((5, 5, 5), dtype=np.float32)
        answer[0, 0, :2] = 1
        answer[0, 0, -1] = 1
        answer[0, :2, 0] = 1
        answer[0, -1, 0] = 1

        self.assertTrue(np.allclose(cl_out.get(), answer))

    @unittest.skip("kernel does not have rotate_grids_and_multiply")
    def test_rotate_grids_and_multiply(self):
        shape = (5, 5, 5)
        values = {
            "shape_x": 5,
            "shape_y": 5,
            "shape_z": 5,
            "llength": 2,
        }
        self.k = CLKernels(self.ctx, values=values)

        template = np.zeros(shape, dtype=np.float32)
        template[2, 2, 1:4] = 1
        template[2, 1:4, 2] = 1
        mask = template * 2
        np_out_template = np.zeros(shape, dtype=np.float32)
        np_out_template[0, 0, :2] = 1
        np_out_template[0, 0, -1] = 1
        np_out_template[0, :2, 0] = 1
        np_out_template[0, -1, 0] = 1
        np_out_mask = np_out_template * 2
        np_out_mask2 = np_out_mask**2

        cl_template = cl.image_from_array(self.ctx, template)
        cl_mask = cl.image_from_array(self.ctx, mask)
        cl_rotmat = np.asarray([1, 0, 0, 0, 1, 0, 0, 0, 1] + [0] * 7, dtype=np.float32)
        cl_center = np.asarray([2, 2, 2, 0], dtype=np.float32)
        cl_shape = np.asarray([5, 5, 5, 125], dtype=np.int32)
        cl_radius = np.int32(2)

        cl_out_template = cl_array.to_device(
            self.queue, np.zeros(shape, dtype=np.float32)
        )
        cl_out_mask = cl_array.to_device(self.queue, np.zeros(shape, dtype=np.float32))
        cl_out_mask2 = cl_array.to_device(self.queue, np.zeros(shape, dtype=np.float32))

        gws = tuple([int(2 * cl_radius + 1)] * 3)
        args = (
            cl_template,
            cl_mask,
            cl_rotmat,
            self.s_linear,
            self.s_nearest,
            cl_center,
            cl_shape,
            cl_radius,
            cl_out_template.data,
            cl_out_mask.data,
            cl_out_mask2.data,
        )
        self.k.rotate_grids_and_multiply(self.queue, gws, None, *args)
        self.queue.finish()

        self.assertTrue(np.allclose(np_out_template, cl_out_template.get()))
        self.assertTrue(np.allclose(np_out_mask, cl_out_mask.get()))
        self.assertTrue(np.allclose(np_out_mask2, cl_out_mask2.get()))


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from powerfit_em.helpers import opencl_available

if opencl_available():
    import pyopencl as cl
    import pyopencl.array as cl_array

    from powerfit_em.correlators.clkernels import CLKernels


@unittest.skipIf(not opencl_available(), "GPU resources are not available.")
class TestCLKernels(unittest.TestCase):
    """Tests for the OpenCL kernels"""

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
        self.s_linear = cl.Sampler(self.ctx, False, cl.addressing_mode.CLAMP, cl.filter_mode.LINEAR)
        self.s_nearest = cl.Sampler(self.ctx, False, cl.addressing_mode.CLAMP, cl.filter_mode.NEAREST)

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


if __name__ == "__main__":
    unittest.main()

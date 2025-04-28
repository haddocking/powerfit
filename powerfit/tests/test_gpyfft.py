
import unittest

import numpy as np
import numpy.testing as npt
from numpy import fft
try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    from gpyfft import GpyFFT
    OPENCL = True
except:
    OPENCL = False

@unittest.skipIf(not OPENCL, "GPU resources not found.")
class TestGPyFFT(unittest.TestCase):

    """Test FFT-plans with batches."""

    @classmethod
    def setUpClass(cls):
        cls.p = cl.get_platforms()[0]
        cls.dev = cls.p.get_devices()
        cls.ctx = cl.Context(devices=cls.dev)
        cls.queue = cl.CommandQueue(cls.ctx, device=cls.dev[0])
        cls.G = GpyFFT()
        cls.CLFFT_COMPLEX_INTERLEAVED = 1
        cls.CLFFT_COMPLEX_PLANAR = 2
        cls.CLFFT_HERMITIAN_INTERLEAVED = 3
        cls.CLFFT_HERMITIAN_PLANAR = 3
        cls.CLFFT_REAL = 5

    
    def test_simple_plan_1d(self):
        """Test 1D plan with multiple batches"""
        if self.dev[0].type == cl.device_type.CPU:
            self.skipTest("Skip when OpenCL runtime is CPU, core dumps on CPU")
        shape = (10,)
        plan = self.G.create_plan(self.ctx, shape)
        plan.bake(self.queue)
        plan.inplace = False
        plan.layouts = (self.CLFFT_COMPLEX_INTERLEAVED, self.CLFFT_COMPLEX_INTERLEAVED)
        plan.batch_size = 2

        np_in = np.zeros(20, dtype=np.complex64)
        np_out = np.zeros(20, dtype=np.complex64)
        np_in.real = np.random.rand(20)

        cl_in = cl_array.to_device(self.queue, np_in)
        cl_out = cl_array.to_device(self.queue, np_out)

        plan.enqueue_transform(self.queue, cl_in.data, cl_out.data)
        self.queue.finish()
        out = fft.fftn(np_in[:10])
        self.assertTrue(np.allclose(out[:10], cl_out[:10].get()))

    def test_simple_plan_2d(self):
        shape = (8, 8)
        plan = self.G.create_plan(self.ctx, shape)
        plan.inplace = False
        plan.layouts = (self.CLFFT_COMPLEX_INTERLEAVED, self.CLFFT_COMPLEX_INTERLEAVED)
        plan.batch_size = 2
        plan.distances = (64, 64)
        plan.bake(self.queue)

        np_in = np.zeros(128, dtype=np.complex64)
        np_out = np.zeros(128, dtype=np.complex64)
        np_in[:64].real = np.random.rand(64)
        np_in[:64].imag = np.random.rand(64)
        np_in[64:].real = np.random.rand(64)
        np_in[64:].imag = np.random.rand(64)

        cl_in = cl_array.to_device(self.queue, np_in)
        cl_out = cl_array.to_device(self.queue, np_out)

        plan.enqueue_transform(self.queue, cl_in.data, cl_out.data)
        out = fft.fftn(np_in[64:].reshape(8, 8)).ravel()

        self.assertTrue(np.allclose(out, cl_out[64:].get()))

    def test_rfft_1d(self):
        shape = (8,)
        plan = self.G.create_plan(self.ctx, shape)
        plan.inplace = False
        plan.layouts = (self.CLFFT_REAL, self.CLFFT_HERMITIAN_INTERLEAVED)
        plan.batch_size = 2
        plan.distances = (8, 5)
        plan.bake(self.queue)

        np_in = np.random.rand(16).astype(np.float32)
        np_out = np.zeros(10, dtype=np.complex64)

        answer1 = fft.rfft(np_in[:8])
        answer2 = fft.rfft(np_in[8:])

        cl_in = cl_array.to_device(self.queue, np_in)
        cl_out = cl_array.to_device(self.queue, np_out)
        plan.enqueue_transform(self.queue, cl_in.data, cl_out.data)

        self.assertTrue(np.allclose(answer1, cl_out[:5].get()))
        self.assertTrue(np.allclose(answer2, cl_out[5:].get()))

    def test_rfft_2d(self):
        shape = (8, 8)
        plan = self.G.create_plan(self.ctx, shape)
        plan.inplace = False
        plan.layouts = (self.CLFFT_REAL, self.CLFFT_HERMITIAN_INTERLEAVED)
        plan.batch_size = 2
        plan.distances = (64, 5 * 8)
        plan.strides_in = (8, 1)
        plan.strides_out = (8, 1)
        plan.bake(self.queue)

        np_in = np.random.rand(64 * 2).astype(np.float32)
        np_out = np.zeros(40 * 2, dtype=np.complex64)

        answer1 = fft.fftn(np_in[:64].reshape(8, 8))[:5]
        answer2 = fft.fftn(np_in[64:].reshape(8, 8))[:5]

        cl_in = cl_array.to_device(self.queue, np_in)
        cl_out = cl_array.to_device(self.queue, np_out)
        plan.enqueue_transform(self.queue, cl_in.data, cl_out.data)
        npt.assert_allclose(
            answer1,
            cl_out[:40].get().reshape(5, 8),
            rtol=1e-4,
        )
        npt.assert_allclose(
            answer2,
            cl_out[40:].get().reshape(5, 8),
            rtol=1e-4,
        )



if __name__ == '__main__':
    unittest.main()

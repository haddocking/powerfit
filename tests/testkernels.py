from __future__ import print_function, division
import unittest

from powerfit.kernels import Kernels
from powerfit.helpers import get_queue
import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np


class TestKernels(unittest.TestCase):

    def setUp(self):
        self.queue = get_queue()
        self.kernels = Kernels(self.queue.context)

    def test_rmultiply(self):

        shape = [100, 100, 100]
        np_x = np.random.rand(*shape).astype(np.float32)
        cl_x = cl_array.to_device(self.queue, np_x)

        np_y = np.random.rand(*shape).astype(np.float32)
        cl_y = cl_array.to_device(self.queue, np_y)

        np_out = np.zeros(shape, dtype=np.float32)
        cl_out = cl_array.to_device(self.queue, np_out)

        self.kernels.rmultiply(self.queue, cl_x, cl_y, cl_out)

        np.multiply(np_x, np_y, np_out)
        self.assertTrue(np.allclose(np_out, cl_out.get()))

    def test_cconjmultiply(self):
        shape = [100, 100, 100]
        dtype = np.complex64

        np_x = np.random.rand(*shape).astype(dtype)
        cl_x = cl_array.to_device(self.queue, np_x)

        np_y = np.random.rand(*shape).astype(dtype)
        cl_y = cl_array.to_device(self.queue, np_y)

        np_out = np.zeros(shape, dtype=dtype)
        cl_out = cl_array.to_device(self.queue, np_out)

        np.multiply(np_x.conj(), np_y, np_out)
        self.kernels.cconjmultiply(self.queue, cl_x, cl_y, cl_out)

        self.assertTrue(np.allclose(np_out, cl_out.get()))
        
if __name__=='__main__':
    unittest.main()

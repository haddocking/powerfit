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

        self.np_x = np.rand(100, 100).astype(np.float32)
        self.cl_x = cl_array.to_device(queue, self.np_x)

        self.np_y = np.rand(100, 100).astype(np.float32)
        self.cl_y = cl_array.to_device(queue, self.np_y)

        self.np_out = np.zeros((100, 100), dtype=np.float32)
        self.cl_out = cl_array.to_device(queue, self.np_y)

    def test_rmultiply(self):

        self.kernels.rmultiply(self.queue, self.cl_x, self.cl_y, self.cl_out)

        self.assertTrue(np.allclose(self.np_x*self.np_y, self.cl_out.get()))
        



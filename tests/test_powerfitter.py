import unittest
from unittest.mock import patch

import numpy as np

from powerfit_em.helpers import opencl_available
from powerfit_em.powerfitter import PowerFitter
from powerfit_em.volume import Volume

if opencl_available():
    import pyopencl as cl
    import pyopencl.array as cl_array

    from powerfit_em.correlators.clkernels import CLKernels


def make_powerfitter(queue=None, cuda_stream=None):
    target = Volume(np.ones((4, 4, 4), dtype=np.float32))
    template = Volume(np.ones((4, 4, 4), dtype=np.float32))
    mask = Volume(np.ones((4, 4, 4), dtype=np.float32))
    rotations = np.asarray([np.eye(3, dtype=np.float32)])
    return PowerFitter(target, rotations, template, mask, queue=queue, cuda_stream=cuda_stream)


@unittest.skipIf(not opencl_available(), "GPU resources are not available.")
class TestCLKernels(unittest.TestCase):
    """Tests for the OpenCL kernels"""

    def setUp(self):
        try:
            p = cl.get_platforms()[0]
        except cl.LogicError as exc:
            raise unittest.SkipTest(f"OpenCL platform not available: {exc}") from exc
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


class TestPowerFitterBackendDispatch(unittest.TestCase):
    def test_scan_uses_opencl_correlator_when_queue_is_set(self):
        queue = object()
        with patch("powerfit_em.correlators.gpu.OpenCLCorrelator", create=True) as correlator_cls:
            pf = make_powerfitter(queue=queue)
            pf.scan(progress=None)

        correlator_cls.assert_called_once()
        correlator_cls.return_value.scan.assert_called_once_with(None)

    def test_scan_uses_cuda_correlator_when_stream_is_set(self):
        stream = object()
        with patch("powerfit_em.correlators.cuda.CUDACorrelator") as correlator_cls:
            pf = make_powerfitter(cuda_stream=stream)
            pf.scan(progress=None)

        correlator_cls.assert_called_once()
        correlator_cls.return_value.scan.assert_called_once_with(None)


if __name__ == "__main__":
    unittest.main()

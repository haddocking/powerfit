import unittest

import numpy as np
import pytest

from powerfit_em.gpu import cuda_available, opencl_available
from powerfit_em.powerfitter import PowerFitter
from powerfit_em.volume import Volume

CUDA_AVAILABLE = cuda_available()
OPENCL_AVAILABLE = opencl_available()

if opencl_available():
    import pyopencl as cl
    import pyopencl.array as cl_array

    from powerfit_em.correlators.clkernels import CLKernels


def _make_tiny_inputs():
    """Return target/template/mask Volumes and single identity rotation for smoke tests."""
    arr = np.zeros((8, 8, 8), dtype=np.float32)
    arr[2:6, 2:6, 2:6] = 1.0
    target = Volume(arr)
    template = Volume(arr.copy())
    mask = Volume(np.ones((8, 8, 8), dtype=np.float32))
    rotations = np.asarray([np.eye(3, dtype=np.float32)])
    return target, template, mask, rotations


@pytest.mark.skipif(not OPENCL_AVAILABLE, reason="OpenCL (pyopencl) not installed")
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


@pytest.mark.gpu_integration
class TestPowerFitterIntegration:
    """Integration smoke tests comparing GPU backends against the CPU correlator."""

    @pytest.mark.requires_opencl
    @pytest.mark.skipif(not OPENCL_AVAILABLE, reason="OpenCL (pyopencl) not installed")
    def test_opencl_scan_matches_cpu(self):
        from powerfit_em.gpu import get_opencl_queue

        try:
            queue = get_opencl_queue("0:0")
        except Exception as exc:
            pytest.skip(str(exc))

        target, template, mask, rotations = _make_tiny_inputs()

        cpu_pf = PowerFitter(target, rotations, template, mask, queue=None)
        cpu_pf.scan(progress=None)

        ocl_pf = PowerFitter(target, rotations, template, mask, queue=queue)
        ocl_pf.scan(progress=None)

        assert np.allclose(cpu_pf.lcc, ocl_pf.lcc, atol=1e-4, rtol=1e-4)
        assert np.array_equal(cpu_pf.rot, ocl_pf.rot)

    @pytest.mark.requires_cuda
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA (cupy) not installed")
    def test_cuda_scan_matches_cpu(self):
        from powerfit_em.gpu import get_cuda_stream

        try:
            stream = get_cuda_stream(0)
        except (RuntimeError, ValueError) as exc:
            pytest.skip(str(exc))

        target, template, mask, rotations = _make_tiny_inputs()

        cpu_pf = PowerFitter(target, rotations, template, mask, queue=None)
        cpu_pf.scan(progress=None)

        cuda_pf = PowerFitter(target, rotations, template, mask, queue=None, cuda_stream=stream)
        cuda_pf.scan(progress=None)

        assert np.allclose(cpu_pf.lcc, cuda_pf.lcc, atol=1e-4, rtol=1e-4)
        assert np.array_equal(cpu_pf.rot, cuda_pf.rot)


if __name__ == "__main__":
    unittest.main()

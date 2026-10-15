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


@pytest.mark.parametrize("cpu_backend", ["py", "rust"])
@pytest.mark.parametrize("nproc", [1, 2])
@pytest.mark.parametrize("scan_first", [False, True])
def test_set_template_cpu(cpu_backend, nproc, scan_first):
    """Replacing a template must give the same result as a fresh CPU fitter."""
    rng = np.random.default_rng(150)
    target = Volume(rng.random((8, 8, 8), dtype=np.float32))
    template = Volume(rng.random(target.shape, dtype=np.float32))
    replacement = Volume(rng.random(target.shape, dtype=np.float32))
    mask_array = np.ones(target.shape, dtype=np.float32)
    mask = Volume(mask_array)
    replacement_mask = Volume(mask_array.copy())
    replacement_mask.array[0, :4, :4] = 0
    rotations = np.asarray([np.eye(3), [[0, -1, 0], [1, 0, 0], [0, 0, 1]]], dtype=np.float32)
    kwargs = {"queue": None, "nproc": nproc, "cpu_backend": cpu_backend}

    fitter = PowerFitter(target, rotations, template, mask, **kwargs)
    original_lcc = None
    if scan_first:
        fitter.scan()
        original_lcc = fitter.lcc

    fitter.set_template(replacement, replacement_mask)
    fitter.scan()

    fresh = PowerFitter(target, rotations, replacement, replacement_mask, **kwargs)
    fresh.scan()
    assert np.any(fresh.lcc > 0)
    if original_lcc is not None:
        assert not np.allclose(original_lcc, fresh.lcc)
    np.testing.assert_allclose(fitter.lcc, fresh.lcc, atol=1e-6)
    np.testing.assert_array_equal(fitter.rot, fresh.rot)


@pytest.mark.skipif(not OPENCL_AVAILABLE, reason="OpenCL (pyopencl) not installed")
class TestCLKernels:
    """Tests for the OpenCL kernels"""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Build an OpenCL context, queue, and CLKernels instance for each test."""
        try:
            p = cl.get_platforms()[0]
        except cl.LogicError as exc:
            pytest.skip(f"OpenCL platform not available: {exc}")
        devs = p.get_devices()
        self.ctx = cl.Context(devices=devs)
        self.queue = cl.CommandQueue(self.ctx, device=devs[0])
        values = {
            "shape_x": 10,
            "shape_y": 0,
            "shape_z": 0,
            "llength": 5,
        }
        try:
            self.k = CLKernels(self.ctx, values=values)
        except cl.RuntimeError as e:
            pytest.skip(f"pocl failed to build OpenCL kernel: {e}")
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
        assert np.allclose(np_out, cl_out.get())

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
        assert np.allclose(np_out, cl_out.get())


@pytest.mark.gpu_integration
class TestPowerFitterIntegration:
    """Integration smoke tests comparing GPU backends against the CPU correlator."""

    @pytest.mark.requires_opencl
    @pytest.mark.skipif(not OPENCL_AVAILABLE, reason="OpenCL (pyopencl) not installed")
    def test_opencl_scan_matches_cpu(self) -> None:
        """Compare OpenCL scan results against the CPU correlator on tiny inputs."""
        import pyopencl as cl

        from powerfit_em.gpu import get_opencl_queue

        target, template, mask, rotations = _make_tiny_inputs()

        cpu_pf = PowerFitter(target, rotations, template, mask, queue=None)
        cpu_pf.scan(progress=None)

        try:
            queue = get_opencl_queue("0:0")
            ocl_pf = PowerFitter(target, rotations, template, mask, queue=queue)
            ocl_pf.scan(progress=None)
        except (cl.RuntimeError, cl.LogicError, RuntimeError) as e:
            pytest.skip(str(e))

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
